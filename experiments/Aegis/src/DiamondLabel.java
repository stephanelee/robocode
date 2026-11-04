/**
 * Offline-derived model of how the pinned opponent's 1v1 gun labels a wave from our scanned states:
 * wave rings [(t-fireTime)v, (t-fireTime+1)v], an axis-aligned 36x36 footprint, the break-state test and the
 * centre of the angular extent of corner/side intersections. It only uses our own positions and the inferred
 * wave; nothing is read from the opponent at runtime.
 */
final class DiamondLabel {
    static final double HALF = 18, MAX_RADIUS = 18 / Math.cos(Math.PI / 4);
    private DiamondLabel() {}

    /** Returns the label centre bearing (degrees, east-zero) minus the reference bearing, or NaN if never broken. */
    static double offset(double ox, double oy, int waveFireTime, double speed, int firstTurn, double[] xs, double[] ys, double refX, double refY) {
        double ref = Math.atan2(refY - oy, refX - ox);
        double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY; boolean any = false;
        for (int i = 0; i < xs.length; i++) {
            int t = firstTurn + i; double x = xs[i], y = ys[i];
            double start = (t - waveFireTime) * speed, end = (t + 1 - waveFireTime) * speed;
            double dist = Math.hypot(x - ox, y - oy);
            if (dist > end + MAX_RADIUS || distanceToBox(ox, oy, x, y) > end) continue; // midair
            if (dist <= end) { // not BREAKING_FRONT: check GONE
                boolean beyond = false;
                for (int c = 0; c < 4; c++) { double cx = x + ((c & 2) == 0 ? -HALF : HALF), cy = y + ((c & 1) == 0 ? -HALF : HALF);
                    if (sq(cx - ox) + sq(cy - oy) > start * start) beyond = true; }
                if (!beyond) continue; // gone
            }
            double[] angles = new double[12]; int n = 0;
            for (int c = 0; c < 4; c++) {
                double cx = x + ((c & 2) == 0 ? -HALF : HALF), cy = y + ((c & 1) == 0 ? -HALF : HALF), d2 = sq(cx - ox) + sq(cy - oy);
                if (d2 < end * end && !(d2 < start * start)) angles[n++] = Math.atan2(cy - oy, cx - ox);
            }
            for (double r : new double[]{start, end}) {
                for (int side = 0; side < 4; side++) {
                    boolean horizontal = side % 2 == 0; double fixed = side == 0 ? y - HALF : side == 1 ? x + HALF : side == 2 ? y + HALF : x - HALF;
                    double lo = horizontal ? x - HALF : y - HALF, hi = horizontal ? x + HALF : y + HALF;
                    double centreFixed = horizontal ? oy : ox, centreFree = horizontal ? ox : oy;
                    double disc = r * r - sq(fixed - centreFixed);
                    if (disc < 0) continue;
                    double root = Math.sqrt(disc);
                    for (double free : root > 0 ? new double[]{centreFree + root, centreFree - root} : new double[]{centreFree}) {
                        if (free > lo && free < hi) { double px = horizontal ? free : fixed, py = horizontal ? fixed : free; angles[n++] = Math.atan2(py - oy, px - ox); }
                    }
                }
            }
            for (int k = 0; k < n; k++) { double a = ref + Math.IEEEremainder(angles[k] - ref, 2 * Math.PI); any = true; min = Math.min(min, a); max = Math.max(max, a); }
        }
        if (!any) return Double.NaN;
        return Math.toDegrees((min + max) / 2 - ref);
    }

    private static double sq(double v) { return v * v; }

    private static double distanceToBox(double px, double py, double x, double y) {
        if (px > x - HALF && px < x + HALF && py > y - HALF && py < y + HALF) return 0;
        double dx = Math.max(Math.max(x - HALF - px, 0), px - (x + HALF)), dy = Math.max(Math.max(y - HALF - py, 0), py - (y + HALF));
        return Math.hypot(dx, dy);
    }
}
