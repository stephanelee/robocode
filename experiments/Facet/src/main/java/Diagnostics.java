import java.io.*;
import java.nio.file.*;
import java.util.*;
/** Observation-only, bounded files, no model import and no network. */
final class Diagnostics implements AutoCloseable {
  final Map<String, double[]> observations = new TreeMap<>();
  final Map<String, Long> counters = new TreeMap<>();
  final Map<String, long[]> timing = new TreeMap<>();
  final Map<String, Integer> streak = new TreeMap<>();
  final Set<String> disabled = new HashSet<>();
  final BufferedWriter trace, metrics;
  long traceLines, metricLines;
  static final long MAX_TRACE = 100000, MAX_METRICS = 10000;
  Diagnostics() {
    trace = open("FACET_TRACE_PATH");
    metrics = open("FACET_METRICS_PATH");
    traceLines = existing("FACET_TRACE_PATH", MAX_TRACE);
    metricLines = existing("FACET_METRICS_PATH", MAX_METRICS);
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
      trace.write("{\"kind\":\"" + escape(kind) + "\",\"round\":" + round + ",\"turn\":" + turn
          + ",\"value\":\"" + escape(value) + "\"}\n");
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
    if (metrics == null || metricLines >= MAX_METRICS)
      return;
    long gcCount = 0, gcMillis = 0;
    for (var gc : java.lang.management.ManagementFactory.getGarbageCollectorMXBeans()) {
      gcCount += Math.max(0, gc.getCollectionCount());
      gcMillis += Math.max(0, gc.getCollectionTime());
    }
    String json = "{\"schema\":1,\"round\":" + round + ",\"shotsRequested\":" + fire.requested
        + ",\"shotsAccepted\":" + fire.accepted + ",\"shotsConfirmed\":" + fire.confirmed
        + ",\"shotsCancelled\":" + fire.cancelled
        + ",\"unmatchedConfirmations\":" + fire.unmatched + ",\"neighborBytes\":" + budget.bytes
        + ",\"neighborEvictions\":" + budget.evictions + ",\"waveEvictions\":" + waves.evictions
        + ",\"surfTruncations\":" + planner.truncations + ",\"geometryFallbacks\":"
        + planner.missingGeometry + ",\"gcCount\":" + gcCount + ",\"gcMillis\":" + gcMillis
        + ",\"counters\":" + counterJson(counters) + ",\"observationsCountSumMinMax\":"
        + arraysJson(observations) + ",\"timingLog2Nanos\":" + timingJson() + "}\n";
    try {
      metrics.write(json);
      metricLines++;
      metrics.flush();
      if (trace != null)
        trace.flush();
    } catch (IOException e) {
      count("metrics.ioFailure");
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
    try {
      if (trace != null)
        trace.close();
      if (metrics != null)
        metrics.close();
    } catch (IOException e) {
      count("close.ioFailure");
    }
  }
}
