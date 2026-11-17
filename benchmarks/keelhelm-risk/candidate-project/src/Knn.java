import java.util.ArrayList;

/** Linear-scan nearest neighbours over (features, guess factor) entries. Small enough to stay simple. */
final class Knn {
    record Entry(double[] f, double gf, double weight, long seq) {}
    record Hit(Entry e, double dist) {}

    private final ArrayList<Entry> entries = new ArrayList<>();
    private final double[] fw;
    private final int cap;
    private long seq = 0;

    Knn(double[] featureWeights, int cap) { this.fw = featureWeights; this.cap = cap; }

    int size() { return entries.size(); }
    long seq() { return seq; }

    void add(double[] f, double gf, double weight) {
        entries.add(new Entry(f, gf, weight, seq++));
        if (entries.size() > cap) entries.remove(0);
    }

    double dist(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) { double d = (a[i] - b[i]) * fw[i]; s += d * d; }
        return s;
    }

    /** k nearest neighbours (unsorted). */
    ArrayList<Hit> nearest(double[] q, int k) {
        int n = entries.size();
        ArrayList<Hit> out = new ArrayList<>();
        if (n == 0 || k <= 0) return out;
        k = Math.min(k, n);
        double[] best = new double[k];
        int[] idx = new int[k];
        int filled = 0, worst = 0;
        for (int i = 0; i < n; i++) {
            double d = dist(q, entries.get(i).f());
            if (filled < k) {
                best[filled] = d; idx[filled] = i; filled++;
                if (filled == k) for (int j = 1; j < k; j++) if (best[j] > best[worst]) worst = j;
                continue;
            }
            if (d < best[worst]) {
                best[worst] = d; idx[worst] = i;
                worst = 0;
                for (int j = 1; j < k; j++) if (best[j] > best[worst]) worst = j;
            }
        }
        for (int j = 0; j < filled; j++) out.add(new Hit(entries.get(idx[j]), best[j]));
        return out;
    }

    /** Last m entries, newest last. */
    java.util.List<Entry> recent(int m) {
        int n = entries.size();
        return entries.subList(Math.max(0, n - m), n);
    }
}
