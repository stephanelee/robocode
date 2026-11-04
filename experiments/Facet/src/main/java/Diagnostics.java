import java.io.*;
import java.nio.file.*;
import java.util.*;
/** Observation-only, bounded files, no model import and no network. */
final class Diagnostics implements AutoCloseable {
  String gameId = "fixture-" + java.util.UUID.randomUUID();
  final Set<Integer> completed = new HashSet<>();
  final Map<String, double[]> observations = new TreeMap<>();
  final Map<String, Long> counters = new TreeMap<>();
  final Map<String, long[]> timing = new TreeMap<>();
  final Map<String, Integer> streak = new TreeMap<>();
  final Set<String> disabled = new HashSet<>();
  final BufferedWriter trace;
  interface Sink extends AutoCloseable {
    void append(String row) throws IOException;
    default void close() throws IOException {}
  }
  /** Roll back partial writes before retry; force content before declaring completion. */
  static final class FileSink implements Sink {
    final java.nio.channels.FileChannel channel;
    boolean poisoned;
    interface IOAction {
      void run() throws IOException;
    }
    IOAction beforeCommit = () -> {};
    FileSink(Path path) throws IOException {
      Files.createDirectories(path.toAbsolutePath().getParent());
      channel = java.nio.channels.FileChannel.open(
          path, StandardOpenOption.CREATE, StandardOpenOption.READ, StandardOpenOption.WRITE);
    }
    public void append(String row) throws IOException {
      if (poisoned)
        throw new IOException("sink quarantined after failed rollback");
      long start = channel.size();
      try {
        channel.position(start);
        var bytes = java.nio.ByteBuffer.wrap(row.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        while (bytes.hasRemaining()) channel.write(bytes);
        beforeCommit.run();
        channel.force(true);
      } catch (IOException ex) {
        try {
          channel.truncate(start);
          channel.force(true);
        } catch (IOException rollback) {
          poisoned = true;
          ex.addSuppressed(rollback);
          throw new IOException("rollback failed; sink requires repair", ex);
        }
        throw ex;
      }
    }
    public void close() throws IOException {
      channel.close();
    }
  }
  final Sink metrics, critical;
  final LinkedHashMap<Integer, String> pending = new LinkedHashMap<>();
  long criticalLines;
  boolean criticalOverflowReported;
  int terminalTurn = -1;
  long terminalSequence = -1;
  Runnable beforeSnapshot = () -> {};
  String terminalObservation = "not-requested", terminalHash = "not-requested";
  static final long MAX_CRITICAL = 10000;
  long traceLines, metricLines;
  static final long MAX_TRACE = 100000, MAX_METRICS = 10000;
  Diagnostics() {
    trace = open("FACET_TRACE_PATH");
    metrics = sink("FACET_METRICS_PATH");
    String base = System.getenv("FACET_METRICS_PATH");
    if (base == null || base.isBlank())
      base = System.getenv("FACET_TRACE_PATH");
    critical = base == null || base.isBlank() ? null : fileSink(Path.of(base + ".critical.jsonl"));
    if (base != null && !base.isBlank()) {
      try (var lines = Files.lines(Path.of(base + ".critical.jsonl"))) {
        criticalLines = lines.limit(MAX_CRITICAL).count();
      } catch (IOException ex) {
        throw new IllegalStateException(ex);
      }
    }
    traceLines = existing("FACET_TRACE_PATH", MAX_TRACE);
    metricLines = existing("FACET_METRICS_PATH", MAX_METRICS);
  }
  Diagnostics(Sink metrics, Sink critical) {
    this.trace = null;
    this.metrics = metrics;
    this.critical = critical;
  }
  static Sink fileSink(Path path) {
    try {
      return new FileSink(path);
    } catch (IOException ex) {
      throw new IllegalStateException(ex);
    }
  }
  static Sink sink(String env) {
    String value = System.getenv(env);
    if (value == null || value.isBlank())
      return null;
    Path path = Path.of(value).toAbsolutePath();
    if (!path.getFileName().toString().startsWith("FACET_"))
      throw new IllegalArgumentException(env + " basename must start FACET_");
    return fileSink(path);
  }
  void critical(String kind, int round, int turn, String value) {
    if (value.length() > 8192) {
      value = value.substring(0, 8192) + "[truncated]";
      count("critical.valueTruncated");
    }
    String row = "{\"schema\":3,\"gameId\":\"" + escape(gameId) + "\",\"kind\":\"" + escape(kind)
        + "\",\"round\":" + round + ",\"turn\":" + turn + ",\"value\":\"" + escape(value)
        + "\"}\n";
    if (critical != null && criticalLines < MAX_CRITICAL) {
      try {
        critical.append(row);
        criticalLines++;
        return;
      } catch (IOException ex) {
        count("critical.ioFailure");
      }
    }
    // Separate bounded critical channel failed/exhausted: preserve an explicit stderr record.
    count("critical.unpersisted");
    if (!criticalOverflowReported) {
      System.err.print(row);
      criticalOverflowReported = true;
    }
  }
  static long existing(String env, long cap) {
    String value = System.getenv(env);
    if (value == null || value.isBlank())
      return 0;
    try (var lines = Files.lines(Path.of(value))) {
      return lines.limit(cap + 1).count();
    } catch (IOException ex) {
      throw new IllegalStateException(ex);
    }
  }
  static BufferedWriter open(String env) {
    String value = System.getenv(env);
    if (value == null || value.isBlank())
      return null;
    try {
      Path p = Path.of(value).toAbsolutePath();
      if (!p.getFileName().toString().startsWith("FACET_"))
        throw new IllegalArgumentException(env + " basename must start FACET_");
      if (p.getParent() != null)
        Files.createDirectories(p.getParent());
      return Files.newBufferedWriter(p, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    } catch (IOException e) {
      throw new IllegalStateException("private diagnostics path unavailable", e);
    }
  }
  void observe(String name, double value) {
    Physics.finite(value);
    double[] s = observations.computeIfAbsent(
        name, k -> new double[] {0, 0, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY});
    s[0]++;
    s[1] += value;
    s[2] = Math.min(s[2], value);
    s[3] = Math.max(s[3], value);
  }
  void count(String name) {
    counters.merge(name, 1L, Long::sum);
  }
  void success(String system) {
    streak.put(system, 0);
  }
  void failure(String system, RuntimeException e) {
    count(system + ".failure");
    if (streak.merge(system, 1, Integer::sum) >= 3)
      disabled.add(system);
    event("failure", 0, 0, system + ":" + e.getClass().getSimpleName());
    critical("failure", 0, 0, system + ":" + e);
  }
  void time(String system, long elapsed) {
    long[] bins = timing.computeIfAbsent(system, k -> new long[32]);
    int bin = elapsed <= 0 ? 0 : Math.min(31, 63 - Long.numberOfLeadingZeros(elapsed));
    bins[bin]++;
  }
  void event(String kind, int round, int turn, String value) {
    if (trace == null || traceLines >= MAX_TRACE) {
      if (trace != null)
        count("trace.truncated");
      return;
    }
    try {
      trace.write("{\"schema\":3,\"gameId\":\"" + escape(gameId) + "\",\"kind\":\"" + escape(kind)
          + "\",\"round\":" + round + ",\"turn\":" + turn + ",\"value\":\"" + escape(value)
          + "\"}\n");
      traceLines++;
    } catch (IOException e) {
      count("trace.ioFailure");
    }
  }
  static String counterJson(Map<String, Long> values) {
    StringJoiner j = new StringJoiner(",", "{", "}");
    values.forEach((k, v) -> j.add("\"" + escape(k) + "\":" + v));
    return j.toString();
  }
  static String arraysJson(Map<String, double[]> values) {
    StringJoiner j = new StringJoiner(",", "{", "}");
    values.forEach((k, v) -> j.add("\"" + escape(k) + "\":" + Arrays.toString(v)));
    return j.toString();
  }
  String timingJson() {
    StringJoiner j = new StringJoiner(",", "{", "}");
    timing.forEach((k, v) -> j.add("\"" + escape(k) + "\":" + Arrays.toString(v)));
    return j.toString();
  }
  void snapshot(
      int round, FireControl fire, Memory.Budget budget, Waves.Service waves, Planner planner) {
    if (completed.contains(round) || pending.containsKey(round)) {
      retryPending();
      return;
    }
    if (pending.size() + metricLines >= MAX_METRICS)
      throw new IllegalStateException("metric capacity exhausted; completion not recorded");
    String json;
    try {
      beforeSnapshot.run();
      json = captureSnapshot(round, fire, budget, waves, planner);
    } catch (RuntimeException ex) {
      count("metrics.captureFailure");
      critical("completion.snapshotFailed", round, 0, ex.toString());
      // Minimal evidence row explicitly marks capture failure, retaining identity and failure.
      json = "{\"schema\":3,\"gameId\":\"" + escape(gameId) + "\",\"round\":" + round
          + ",\"terminalTurn\":" + terminalTurn + ",\"ownStateVersion\":" + terminalSequence
          + ",\"snapshotStatus\":\"failed\",\"error\":\"" + escape(ex.toString())
          + "\",\"terminalObservation\":\"" + terminalObservation + "\",\"terminalHash\":\""
          + terminalHash + "\"}\n";
    }
    pending.put(round, json);
    retryPending();
  }
  String captureSnapshot(
      int round, FireControl fire, Memory.Budget budget, Waves.Service waves, Planner planner) {
    long gcCount = 0, gcMillis = 0;
    for (var gc : java.lang.management.ManagementFactory.getGarbageCollectorMXBeans()) {
      gcCount += Math.max(0, gc.getCollectionCount());
      gcMillis += Math.max(0, gc.getCollectionTime());
    }
    return "{\"schema\":3,\"gameId\":\"" + escape(gameId)
        + ("\",\"scope\":\"game-cumulative; neighborBytes gauge; GC "
            + "process-cumulative\",\"round\":")
        + round + ",\"terminalTurn\":" + terminalTurn + ",\"ownStateVersion\":" + terminalSequence
        + ",\"snapshotStatus\":\"success\",\"terminalObservation\":\"" + terminalObservation
        + "\",\"terminalHash\":\"" + terminalHash + "\",\"shotsRequested\":" + fire.requested
        + ",\"shotsAccepted\":" + fire.accepted + ",\"shotsConfirmed\":" + fire.confirmed
        + ",\"shotsCancelled\":" + fire.cancelled
        + ",\"unmatchedConfirmations\":" + fire.unmatched + ",\"neighborBytes\":" + budget.bytes
        + ",\"neighborEvictions\":" + budget.evictions + ",\"waveEvictions\":" + waves.evictions
        + ",\"surfTruncations\":" + planner.truncations + ",\"geometryFallbacks\":"
        + planner.missingGeometry + ",\"gcCount\":" + gcCount + ",\"gcMillis\":" + gcMillis
        + ",\"counters\":" + counterJson(counters) + ",\"observationsCountSumMinMax\":"
        + arraysJson(observations) + ",\"timingLog2Nanos\":" + timingJson() + "}\n";
  }
  void retryPending() {
    var entries = pending.entrySet().iterator();
    while (entries.hasNext()) {
      var entry = entries.next();
      try {
        if (metrics != null)
          metrics.append(entry.getValue());
      } catch (IOException | RuntimeException ex) {
        count("metrics.ioFailure");
        critical("completion.pending", entry.getKey(), 0, ex.toString());
        return;
      }
      metricLines++;
      completed.add(entry.getKey());
      entries.remove();
      if (metrics == null) {
        count("metrics.disabled");
        critical("completion.acknowledged", entry.getKey(), 0,
            "metric output disabled; in-memory acknowledgement only");
      } else {
        critical("completion.persisted", entry.getKey(), 0, "metric row flushed and forced");
      }
    }
    if (trace != null) {
      try {
        trace.flush();
      } catch (IOException ex) {
        count("trace.ioFailure");
      }
    }
  }
  String observationString() {
    StringBuilder b = new StringBuilder();
    observations.forEach((k, v) -> b.append(k).append(Arrays.toString(v)));
    return b.toString();
  }
  String timingString() {
    StringBuilder b = new StringBuilder();
    timing.forEach((k, v) -> b.append(k).append(Arrays.toString(v)));
    return b.toString();
  }
  static String escape(String s) {
    return s.replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t");
  }
  void roundReset() {
    streak.clear();
    disabled.clear();
  }
  public void close() {
    retryPending();
    if (!pending.isEmpty()) {
      critical("completion.unclosed", 0, 0, "pending=" + pending.keySet());
      return;
    }
    try {
      if (trace != null)
        trace.close();
      if (metrics != null)
        metrics.close();
      if (critical != null)
        critical.close();
    } catch (IOException e) {
      count("close.ioFailure");
    }
  }
}
