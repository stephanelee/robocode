package prism.learn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Bounded FIFO store with exact nearest-neighbour search by linear scan.
 * Distance is D = sum((w_i*(x_i-q_i))^2); scale w_i is squared in the distance. Ordering is by
 * increasing distance, then by insertion sequence. Eviction is FIFO by insertion sequence,
 * independent of neighbour ordering. Labels are opaque double arrays whose type is declared by
 * the store's {@link LabelType}; stores never mix label types.
 */
public final class NeighborStore {
    public enum LabelType { PRECISE_GF, THEORETICAL_GF, DISPLACEMENT }

    public record Sample(double[] x, double[] label, double quality, long sequence, int round, int turn, long sourceWave) {}

    public record Neighbor(Sample sample, double distance, int recencyRank) {
        /** Finite inverse-distance weight with the 1e-6 floor. */
        public double inverseDistance() { return 1.0 / Math.sqrt(Math.max(distance, 1e-6)); }
        public double recencyWeight() { return Math.pow(2, -recencyRank / 32.0); }
    }

    private final String schema;
    private final LabelType labelType;
    private final double[] weights;
    private final int capacity;
    private final Sample[] ring;
    private int head, size;
    private long version;
    private long nextSeq;

    public NeighborStore(String schema, LabelType labelType, double[] weights, int capacity) {
        if (capacity <= 0) throw new IllegalArgumentException("capacity");
        for (double w : weights) if (!Double.isFinite(w)) throw new IllegalArgumentException("weight");
        this.schema = schema;
        this.labelType = labelType;
        this.weights = weights.clone();
        this.capacity = capacity;
        this.ring = new Sample[capacity];
    }

    public String schema() { return schema; }
    public LabelType labelType() { return labelType; }
    public int size() { return size; }
    public int capacity() { return capacity; }
    public long version() { return version; }
    public int dims() { return weights.length; }

    /** Adds one observation; returns false (and stores nothing) for non-finite input. */
    public boolean add(double[] x, double[] label, double quality, int round, int turn, long sourceWave) {
        if (x.length != weights.length) throw new IllegalArgumentException("dimension " + x.length + " != " + weights.length);
        for (double v : x) if (!Double.isFinite(v)) return false;
        for (double v : label) if (!Double.isFinite(v)) return false;
        if (!(quality > 0) || !Double.isFinite(quality)) return false;
        Sample s = new Sample(x.clone(), label.clone(), quality, nextSeq++, round, turn, sourceWave);
        int idx = (head + size) % capacity;
        if (size == capacity) { ring[head] = s; head = (head + 1) % capacity; }
        else { ring[idx] = s; size++; }
        version++;
        return true;
    }

    public double distance(double[] a, double[] q) {
        double d = 0;
        for (int i = 0; i < weights.length; i++) { double t = weights[i] * (a[i] - q[i]); d += t * t; }
        return d;
    }

    /** Exact k nearest neighbours. Empty store or non-finite query yields an empty list. */
    public List<Neighbor> query(double[] q, int k) {
        if (q.length != weights.length) throw new IllegalArgumentException("dimension");
        for (double v : q) if (!Double.isFinite(v)) return List.of();
        if (size == 0 || k <= 0) return List.of();
        int n = Math.min(k, size);
        // bounded max-heap via sorted insertion (k is small)
        Sample[] best = new Sample[n];
        double[] bd = new double[n];
        int filled = 0;
        for (int j = 0; j < size; j++) {
            Sample s = ring[(head + j) % capacity];
            double d = distance(s.x(), q);
            if (filled < n) {
                int p = filled++;
                while (p > 0 && worse(bd[p - 1], best[p - 1], d, s)) { bd[p] = bd[p - 1]; best[p] = best[p - 1]; p--; }
                bd[p] = d; best[p] = s;
            } else if (worse(bd[n - 1], best[n - 1], d, s)) {
                int p = n - 1;
                while (p > 0 && worse(bd[p - 1], best[p - 1], d, s)) { bd[p] = bd[p - 1]; best[p] = best[p - 1]; p--; }
                bd[p] = d; best[p] = s;
            }
        }
        // recency rank among the selected neighbours: newest = 0
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Long.compare(best[b].sequence(), best[a].sequence()));
        int[] rank = new int[n];
        for (int r = 0; r < n; r++) rank[order[r]] = r;
        List<Neighbor> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) out.add(new Neighbor(best[i], bd[i], rank[i]));
        return out;
    }

    /** True when (d0,s0) sorts after (d,s). */
    private static boolean worse(double d0, Sample s0, double d, Sample s) {
        return d0 > d || (d0 == d && s0.sequence() > s.sequence());
    }

    /** Snapshot in insertion order (oldest first); for tests and diagnostics. */
    public List<Sample> samples() {
        List<Sample> out = new ArrayList<>(size);
        for (int j = 0; j < size; j++) out.add(ring[(head + j) % capacity]);
        return out;
    }

    /** Approximate payload bytes for the global memory cap. */
    public long payloadBytes() {
        if (size == 0) return 0;
        Sample s = ring[head];
        return (long) size * (48L + 8L * (s.x().length + s.label().length) + 32L);
    }

    /** Evicts the oldest n samples (used only by the global memory guard). */
    public void evictOldest(int n) {
        int m = Math.min(n, size);
        for (int i = 0; i < m; i++) { ring[head] = null; head = (head + 1) % capacity; size--; }
        if (m > 0) version++;
    }
}
