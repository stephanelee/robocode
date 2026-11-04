package prism.core;

import java.util.ArrayList;
import java.util.List;

/**
 * Engine-native wave geometry. In Tank Royale 1.4.0 a bullet is tested once per turn as the
 * segment between its radii (k+1)b and (k+2)b from the origin (k turns after the fire turn)
 * against the circular bot footprint at its end-of-movement position. The set of bullet bearings
 * that could hit is therefore exactly the angular extent, seen from the origin, of the region
 * disk(centre, R) intersected with the annulus r0 <= r <= r1. Intervals are expressed as
 * offsets relative to a reference bearing so wraparound is handled once, in {@link #rel}.
 */
public final class Geometry {
    private Geometry() {}

    /** Closed interval of relative offsets in radians; lo <= hi. */
    public record Interval(double lo, double hi) {
        public Interval {
            if (!(Double.isFinite(lo) && Double.isFinite(hi)) || hi < lo) throw new IllegalArgumentException("bad interval " + lo + "," + hi);
        }
        public double width() { return hi - lo; }
        public double mid() { return (lo + hi) / 2; }
        public boolean contains(double a) { return a >= lo && a <= hi; }
    }

    public static double rel(double bearing, double reference) { return Angles.wrap(bearing - reference); }

    /**
     * Relative angular interval of the disk/annulus intersection seen from the source, or null when
     * the annulus does not touch the disk. When the source is inside the disk the full circle is
     * returned.
     */
    public static Interval footprint(double sx, double sy, double cx, double cy, double radius,
                                     double r0, double r1, double reference) {
        double dx = cx - sx, dy = cy - sy;
        double d = Math.hypot(dx, dy);
        if (r1 < r0) return null;
        if (r1 < d - radius || r0 > d + radius) return null;
        if (d <= radius) return new Interval(-Math.PI, Math.PI);
        double c = rel(Math.atan2(dy, dx), reference);
        double phi = halfExtent(d, radius, r0, r1);
        if (Double.isNaN(phi)) return null;
        return new Interval(c - phi, c + phi);
    }

    /** Maximum angular offset from the centre bearing of disk(d,R) ∩ annulus[r0,r1]; NaN if empty. */
    static double halfExtent(double d, double R, double r0, double r1) {
        double best = Double.NaN;
        double rt = Math.sqrt(Math.max(0, d * d - R * R));
        if (rt >= r0 && rt <= r1) return Math.asin(Math.min(1, R / d));
        for (double r : new double[] {r0, r1}) {
            if (r <= 0) continue;
            if (r >= d - R && r <= d + R) {
                double cos = (r * r + d * d - R * R) / (2 * r * d);
                double phi = Math.acos(Math.max(-1, Math.min(1, cos)));
                best = Double.isNaN(best) ? phi : Math.max(best, phi);
            }
        }
        if (Double.isNaN(best) && r0 <= d - R && r1 >= d + R) best = Math.asin(Math.min(1, R / d));
        if (Double.isNaN(best) && r0 <= 0 && r1 >= d - R) best = Math.asin(Math.min(1, R / d));
        return best;
    }

    /** Full angular half-width of a footprint at distance d. */
    public static double halfWidth(double d) { return d <= Rules.BOT_RADIUS ? Math.PI : Math.asin(Rules.BOT_RADIUS / d); }

    /**
     * Bearings (relative to reference) from origin O for which a ray point at radius in [r0,r1]
     * lies on segment AB. Used for bullet shadows: our bullet's per-turn segment against an
     * incoming wave's per-turn shell. Returns zero, one or two intervals.
     */
    public static List<Interval> segmentShell(double ox, double oy, double ax, double ay, double bx, double by,
                                              double r0, double r1, double reference) {
        List<Interval> out = new ArrayList<>();
        double ux = bx - ax, uy = by - ay;
        double px = ax - ox, py = ay - oy;
        double a = ux * ux + uy * uy;
        if (a < 1e-12) {
            double r = Math.hypot(px, py);
            if (r >= r0 && r <= r1) {
                double t = rel(Math.atan2(py, px), reference);
                out.add(new Interval(t, t));
            }
            return out;
        }
        // |P + λU|^2 = r^2 -> aλ^2 + 2bλ + c - r^2 = 0
        double b = px * ux + py * uy, c = px * px + py * py;
        // λ where |.| <= r1 : between roots of r1; λ where |.| >= r0 : outside roots of r0
        double[] in = roots(a, b, c - r1 * r1);
        if (in == null) return out;
        double lo = Math.max(0, in[0]), hi = Math.min(1, in[1]);
        if (lo > hi) return out;
        double[] ex = roots(a, b, c - r0 * r0);
        List<double[]> segs = new ArrayList<>();
        if (ex == null) segs.add(new double[] {lo, hi});
        else {
            if (lo < ex[0]) segs.add(new double[] {lo, Math.min(hi, ex[0])});
            if (hi > ex[1]) segs.add(new double[] {Math.max(lo, ex[1]), hi});
        }
        for (double[] s : segs) {
            if (s[0] > s[1]) continue;
            double t0 = rel(Math.atan2(py + s[0] * uy, px + s[0] * ux), reference);
            double t1 = rel(Math.atan2(py + s[1] * uy, px + s[1] * ux), reference);
            out.add(new Interval(Math.min(t0, t1), Math.max(t0, t1)));
        }
        return out;
    }

    private static double[] roots(double a, double b, double c) {
        double disc = b * b - a * c;
        if (disc < 0) return null;
        double s = Math.sqrt(disc);
        return new double[] {(-b - s) / a, (-b + s) / a};
    }

    /** Sorted, merged union of intervals; disconnected parts are retained. */
    public static List<Interval> union(List<Interval> xs) {
        List<Interval> s = new ArrayList<>(xs);
        s.sort((p, q) -> Double.compare(p.lo(), q.lo()));
        List<Interval> out = new ArrayList<>();
        for (Interval i : s) {
            if (!out.isEmpty() && i.lo() <= out.get(out.size() - 1).hi()) {
                Interval last = out.remove(out.size() - 1);
                out.add(new Interval(last.lo(), Math.max(last.hi(), i.hi())));
            } else out.add(i);
        }
        return out;
    }

    /** Total measure of the intersection of interval x with a merged interval list. */
    public static double overlap(Interval x, List<Interval> merged) {
        double m = 0;
        for (Interval i : merged) {
            double lo = Math.max(x.lo(), i.lo()), hi = Math.min(x.hi(), i.hi());
            if (hi > lo) m += hi - lo;
        }
        return m;
    }
}
