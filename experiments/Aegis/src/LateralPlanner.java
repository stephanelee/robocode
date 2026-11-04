import java.util.Arrays;

/**
 * Exact-physics dynamic programme over forward/back motion along a fixed body axis.
 * Positions/speeds stay on a half-unit lattice; transitions that would leave it are not used.
 * Costs may depend on (turn, position, speed) and on the speed change entering a turn.
 */
final class LateralPlanner {
    static final int SCALE = 2, NV = 16 * SCALE + 1, ZERO_V = NV / 2;
    final int minS, maxS, horizon;
    private final double[][] cost;     // [t 1..H][s * NV + v]
    private final double[] accelWeight; // [t 1..H] weight on (v_t - v_{t-1})^2
    private static final int[][] TABLE_V = new int[NV][], TABLE_U = new int[NV][];
    static {
        for (int v = 0; v < NV; v++) {
            java.util.List<int[]> list = new java.util.ArrayList<>(); double last = Double.NaN;
            for (int u = -8; u <= 8; u++) {
                double nvel = Physics.newSpeed(speedOf(v), u);
                if (nvel == last || !onLattice(nvel)) { last = nvel; continue; }
                last = nvel; list.add(new int[]{(int) Math.rint(nvel * SCALE) + ZERO_V, u});
            }
            TABLE_V[v] = list.stream().mapToInt(a -> a[0]).toArray(); TABLE_U[v] = list.stream().mapToInt(a -> a[1]).toArray();
        }
    }

    private final double[] bufferA, bufferB;
    private final byte[][] choice;     // [t][s * NV + v]

    LateralPlanner(double minS, double maxS, int horizon) {
        this.minS = (int) Math.ceil(minS * SCALE); this.maxS = (int) Math.floor(maxS * SCALE); this.horizon = horizon;
        int ns = this.maxS - this.minS + 1;
        cost = new double[horizon + 1][ns * NV];
        accelWeight = new double[horizon + 1];
        bufferA = new double[ns * NV]; bufferB = new double[ns * NV]; choice = new byte[horizon][ns * NV];
    }

    /** Clears all costs so the same buffers can be reused next turn (identical results to a fresh planner). */
    void reset() {
        for (double[] layer : cost) Arrays.fill(layer, 0);
        Arrays.fill(accelWeight, 0);
    }

    static double speedOf(int v) { return (v - ZERO_V) / (double) SCALE; }
    static boolean onLattice(double value) { return Math.abs(value * SCALE - Math.rint(value * SCALE)) < 1e-9; }
    int states() { return maxS - minS + 1; }
    double position(int index) { return (index + minS) / (double) SCALE; }
    void addCost(int t, int s, int v, double c) { cost[t][s * NV + v] += c; }
    void addPositionCost(int t, int s, double c) { double[] layer = cost[t]; int base = s * NV; for (int v = 0; v < NV; v++) layer[base + v] += c; }
    void addAccelWeight(int t, double w) { accelWeight[t] += w; }

    record Result(double targetSpeed, double[] positions, double[] speeds, double cost) {}

    Result plan(double s0, double v0) {
        int ns = states();
        double[] next = bufferA, cur = bufferB;
        for (int s = 0; s < ns; s++) for (int v = 0; v < NV; v++) next[s * NV + v] = 0.02 * Math.abs(v - ZERO_V);
        for (int t = horizon - 1; t >= 0; t--) {
            double[] c1 = cost[t + 1]; double aw = accelWeight[t + 1]; byte[] ch = choice[t];
            for (int s = 0; s < ns; s++) for (int v = 0; v < NV; v++) {
                double best = Double.POSITIVE_INFINITY; int arg = -1;
                int[] tv = TABLE_V[v];
                for (int q = 0; q < tv.length; q++) {
                    int ni = tv[q], si = s + (ni - ZERO_V);
                    if (si < 0 || si >= ns) continue;
                    double dv = (ni - v) / (double) SCALE;
                    int idx = si * NV + ni;
                    double c = c1[idx] + next[idx] + aw * dv * dv;
                    if (c < best) { best = c; arg = q; }
                }
                cur[s * NV + v] = best; ch[s * NV + v] = (byte) arg;
            }
            double[] swap = next; next = cur; cur = swap;
        }
        int s = Math.max(0, Math.min(ns - 1, (int) Math.rint(s0 * SCALE) - minS));
        int v = Math.max(0, Math.min(NV - 1, (int) Math.rint(v0 * SCALE) + ZERO_V));
        double[] positions = new double[horizon + 1], speeds = new double[horizon + 1];
        positions[0] = position(s); speeds[0] = speedOf(v);
        double total = next[s * NV + v]; double first = 0;
        for (int t = 0; t < horizon; t++) {
            int q = choice[t][s * NV + v];
            if (q < 0) { positions[t + 1] = position(s); speeds[t + 1] = 0; v = ZERO_V; continue; } // boxed in: brake
            if (t == 0) first = TABLE_U[v][q];
            int ni = TABLE_V[v][q]; s = s + (ni - ZERO_V); v = ni;
            positions[t + 1] = position(s); speeds[t + 1] = speedOf(v);
        }
        return new Result(first, positions, speeds, total);
    }
}
