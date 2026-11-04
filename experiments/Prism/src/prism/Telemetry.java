package prism;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Map;
import java.util.TreeMap;

/**
 * Observation-only, bounded telemetry. Counters never influence decisions. Latency samples are
 * kept in a fixed ring per subsystem. Output is written only at round/game end, never in the
 * decision path, and only when PRISM_TELEMETRY_PATH is set.
 */
public final class Telemetry {
    private static final int LAT = 4096;
    private final Map<String, Long> counters = new TreeMap<>();
    private final Map<String, long[]> latency = new TreeMap<>();
    private final Map<String, Integer> latencyN = new TreeMap<>();
    private final Path path;

    public Telemetry(Path path) { this.path = path; }

    public void inc(String k) { add(k, 1); }
    public void add(String k, long v) { counters.merge(k, v, Long::sum); }
    public long get(String k) { return counters.getOrDefault(k, 0L); }
    public Map<String, Long> counters() { return counters; }

    public void time(String k, long nanos) {
        long[] r = latency.computeIfAbsent(k, x -> new long[LAT]);
        int n = latencyN.getOrDefault(k, 0);
        r[n % LAT] = nanos;
        latencyN.put(k, n + 1);
    }

    public double quantileMicros(String k, double q) {
        long[] r = latency.get(k);
        if (r == null) return Double.NaN;
        int n = Math.min(LAT, latencyN.get(k));
        long[] s = Arrays.copyOf(r, n);
        Arrays.sort(s);
        return s[Math.min(n - 1, (int) Math.floor(q * (n - 1)))] / 1000.0;
    }

    public String json(String header) {
        StringBuilder b = new StringBuilder("{").append(header);
        b.append("\"counters\":{");
        boolean first = true;
        for (var e : counters.entrySet()) { if (!first) b.append(','); first = false; b.append('"').append(e.getKey()).append("\":").append(e.getValue()); }
        b.append("},\"latencyMicros\":{");
        first = true;
        for (String k : latency.keySet()) {
            if (!first) b.append(','); first = false;
            b.append('"').append(k).append("\":{\"p50\":").append(quantileMicros(k, 0.5)).append(",\"p99\":").append(quantileMicros(k, 0.99))
                    .append(",\"max\":").append(quantileMicros(k, 1.0)).append(",\"n\":").append(latencyN.get(k)).append('}');
        }
        return b.append("}}").toString();
    }

    public void writeLine(String line) {
        if (path == null) return;
        try {
            Files.writeString(path, line + "\n", StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            inc("telemetry.ioFailure");
        }
    }
}
