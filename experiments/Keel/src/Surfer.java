import java.util.ArrayList;
import java.util.List;

/**
 * Wave-surfing movement. Waves come only from legal enemy energy drops between consecutive scans.
 * Danger is a hit probability: the angular share of a per-wave bullet-angle density covered by our
 * 18px circle while the wave's engine hit segment crosses it, integrated over the planned path.
 */
final class Surfer {
    static final int BINS = 47;
    static final double FLAT = 2.0;    // context flattener weight relative to the hit-profile mass of 3
    static final double DESIRED = 520;

    static final class Wave {
        final int t;                // detection turn = engine fire turn
        final double ox, oy, power, speed, direct, mea;
        final int latDir;
        final double[] features;
        final double[] density;     // normalized over BINS
        final double[] shade = new double[BINS]; // fraction of each bin blocked by our bullets
        boolean visited;
        Wave(int t, double ox, double oy, double power, double direct, int latDir, double[] features, double[] density) {
            this.t = t; this.ox = ox; this.oy = oy; this.power = power; this.speed = Phys.bulletSpeed(power);
            this.direct = direct; this.latDir = latDir; this.mea = Phys.mea(power);
            this.features = features; this.density = density;
        }
        /** Engine hit segment at the end of turn T is [near, far] from the origin. */
        double near(int T) { return (T - t + 1) * speed; }
        double far(int T) { return (T - t + 2) * speed; }
        double gf(double angle) { return Phys.norm(angle - direct) / mea * latDir; }
    }

    final List<Wave> waves = new ArrayList<>();

    /** Our confirmed bullet: fired in engine turn F from (ox, oy) along dir. */
    record Own(int id, int F, double ox, double oy, double dir, double speed) {
        double r0x(int T) { return ox + Phys.cos(dir) * (T - F + 1) * speed; }
        double r0y(int T) { return oy + Phys.sin(dir) * (T - F + 1) * speed; }
    }
    final List<Own> bullets = new ArrayList<>();
    /** Bullet shadows: implemented and contract-tested, disabled in v1 (no measurable development effect). */
    static boolean shadows = false;

    /** Recomputes which wave angles our in-flight bullets will intercept (bullet-bullet collisions). */
    void reshade(double w, double h) {
        for (Wave wv : waves) {
            java.util.Arrays.fill(wv.shade, 0);
            if (!shadows) continue;
            for (Own b : bullets) shade(wv, b, w, h);
        }
    }

    private static void shade(Wave wv, Own b, double w, double h) {
        int T = Math.max(wv.t, b.F());
        double lo = Double.POSITIVE_INFINITY, hi = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < 120; i++, T++) {
            double ax = b.r0x(T), ay = b.r0y(T);
            double bx = ax + Phys.cos(b.dir()) * b.speed(), by = ay + Phys.sin(b.dir()) * b.speed();
            if (ax < 0 || ay < 0 || ax > w || ay > h) break;
            double ra = Phys.distance(wv.ox, wv.oy, ax, ay), rb = Phys.distance(wv.ox, wv.oy, bx, by);
            double near = wv.near(T), far = wv.far(T);
            if (near > Math.max(ra, rb) + 1) break;
            if (far < Math.min(ra, rb) - b.speed()) continue;
            for (int k = 0; k <= 10; k++) {
                double u = k / 10.0, px = ax + (bx - ax) * u, py = ay + (by - ay) * u;
                double r = Phys.distance(wv.ox, wv.oy, px, py);
                if (r < near || r > far) continue;
                double g = wv.gf(Phys.bearing(wv.ox, wv.oy, px, py));
                lo = Math.min(lo, g); hi = Math.max(hi, g);
            }
        }
        if (lo > hi) return;
        double width = 2.0 / (BINS - 1);
        for (int k = 0; k < BINS; k++) {
            double c = gfOf(k), a = Math.max(lo, c - width / 2), z = Math.min(hi, c + width / 2);
            if (z > a) wv.shade[k] = Math.max(wv.shade[k], (z - a) / width);
        }
    }
    private static final double[] W = {3, 1, 1.5, 1, 1.5, 1.5, 0.7, 1};
    final Knn hits = new Knn(W, 600);
    final Knn visits = new Knn(W, 3000);
    int enemyShots = 0, enemyHits = 0;

    void newRound() { waves.clear(); bullets.clear(); }

    static int bin(double gf) { return (int) Math.round(Phys.clip((gf + 1) / 2 * (BINS - 1), 0, BINS - 1)); }
    static double gfOf(int b) { return 2.0 * b / (BINS - 1) - 1; }

    private static void kernel(double[] a, double gf, double w, double sd) {
        for (int b = 0; b < BINS; b++) { double z = (gfOf(b) - gf) / sd; a[b] += w * Math.exp(-0.5 * z * z); }
    }

    /** Bullet-angle density for a new wave from what the enemy could have used: our past features. */
    double[] density(double[] f) {
        double[] a = new double[BINS];
        int n = hits.size();
        double priorW = 3.0 / (3 + n);
        kernel(a, 0, 1.0 * priorW, 0.10);      // head-on emphasis
        kernel(a, 0.85, 0.35 * priorW, 0.12);  // forward lead
        kernel(a, -0.5, 0.10 * priorW, 0.20);
        for (int b = 0; b < BINS; b++) a[b] += 0.02;
        if (n > 0) {
            int k = Math.min(n, Math.max(3, n / 4));
            long last = hits.seq();
            double total = 0;
            double[] h = new double[BINS];
            for (var hit : hits.nearest(f, k)) {
                double age = last - hit.e().seq();
                double w = hit.e().weight() / (1 + hit.dist()) * (0.4 + Math.exp(-age / 25.0));
                kernel(h, hit.e().gf(), w, 0.07);
                total += w;
            }
            // Recent hits regardless of context (adaptive guns re-target fast).
            for (var e : hits.recent(6)) { kernel(h, e.gf(), 0.25 * e.weight(), 0.07); total += 0.25 * e.weight(); }
            if (total > 0) for (int b = 0; b < BINS; b++) a[b] += (1 - priorW * 0.5) * h[b] / total * 3;
        }
        if (visits.size() > 20) {
            double[] v = new double[BINS];
            double total = 0;
            for (var hit : visits.nearest(f, Math.min(visits.size(), 40))) {
                double w = 1 / (1 + hit.dist());
                kernel(v, hit.e().gf(), w, 0.07); total += w;
            }
            double fw = FLAT;
            if (total > 0) for (int b = 0; b < BINS; b++) a[b] += fw * v[b] / total;
        }
        double s = 0;
        for (double x : a) s += x;
        for (int b = 0; b < BINS; b++) a[b] /= s;
        return a;
    }

    /** Maximum angular half-width of an 18px circle at distance D visible inside radii [r0, r1]. */
    static double halfWidth(double D, double r0, double r1) {
        double R = Phys.BOT_RADIUS;
        double lo = Math.max(r0, D - R), hi = Math.min(r1, D + R);
        if (lo > hi) return -1;
        if (D <= R) return 180;
        double tangent = Math.sqrt(D * D - R * R);
        double r = Phys.clip(tangent, lo, hi);
        double c = (r * r + D * D - R * R) / (2 * r * D);
        return Math.toDegrees(Math.acos(Phys.clip(c, -1, 1)));
    }

    /** Hit probability of wave w against the given path starting at turn T0 (poses at end of T0, T0+1, ...). */
    static double danger(Wave w, List<Phys.Pose> path, int T0) {
        double lo = Double.POSITIVE_INFINITY, hi = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < path.size(); i++) {
            int T = T0 + i;
            Phys.Pose p = path.get(i);
            double D = Phys.distance(w.ox, w.oy, p.x(), p.y());
            double near = w.near(T);
            if (near > D + Phys.BOT_RADIUS) break;
            double hw = halfWidth(D, near, w.far(T));
            if (hw < 0) continue;
            double c = w.gf(Phys.bearing(w.ox, w.oy, p.x(), p.y()));
            double g = hw / w.mea;
            lo = Math.min(lo, c - g); hi = Math.max(hi, c + g);
        }
        if (lo > hi) return 0;
        return covered(w.density, w.shade, lo, hi);
    }

    static double covered(double[] density, double[] shade, double lo, double hi) {
        double width = 2.0 / (BINS - 1), s = 0;
        for (int b = 0; b < BINS; b++) {
            double c = gfOf(b), a = Math.max(lo, c - width / 2), z = Math.min(hi, c + width / 2);
            if (z > a) s += density[b] * (1 - shade[b]) * (z - a) / width;
        }
        // Angles beyond the GF range are clipped into the end bins' mass already.
        return s;
    }

    /** Wall-smoothed orbit command (turn rate, target speed) around (ex, ey). dir +1 = counterclockwise. */
    static double[] orbit(Phys.Pose p, double ex, double ey, int dir, double w, double h) {
        double d = Phys.distance(ex, ey, p.x(), p.y());
        double away = Phys.bearing(ex, ey, p.x(), p.y());
        double tilt = Phys.clip((DESIRED - d) / DESIRED * 70, -25, 40);
        double heading = away + dir * (90 - tilt);
        for (int i = 0; i < 40; i++) {
            double x = p.x() + Phys.cos(heading) * 130, y = p.y() + Phys.sin(heading) * 130;
            if (x > 26 && y > 26 && x < w - 26 && y < h - 26) break;
            heading += dir * 4.5;
        }
        double delta = Phys.norm(heading - p.h());
        if (Math.abs(delta) > 90) return new double[]{Phys.norm(delta + 180), -8};
        return new double[]{delta, 8};
    }

    record Plan(int dir, int stopAt, double danger, double turn, double speed) {}

    /**
     * Chooses among: orbit in either direction for k turns then stop instantly (k = 0..until passage),
     * or orbit throughout; the second wave is scored from each end state with three basic options.
     */
    Plan plan(Phys.Pose me, int T, double ex, double ey, double w, double h, double myEnergy) {
        Wave w1 = null, w2 = null;
        double t1 = Double.MAX_VALUE, t2 = Double.MAX_VALUE;
        for (Wave wv : waves) {
            double D = Phys.distance(wv.ox, wv.oy, me.x(), me.y());
            if (wv.near(T) > D + Phys.BOT_RADIUS) continue;
            double eta = (D - wv.near(T)) / wv.speed;
            if (eta < t1) { w2 = w1; t2 = t1; w1 = wv; t1 = eta; }
            else if (eta < t2) { w2 = wv; t2 = eta; }
        }
        if (w1 == null) return null;
        Plan best = null;
        for (int dir = -1; dir <= 1; dir += 2) {
            ArrayList<Phys.Pose> full = new ArrayList<>();
            Phys.Pose p = me;
            int passIndex = -1;
            for (int i = 0; i < 110; i++) {
                double[] c = orbit(p, ex, ey, dir, w, h);
                p = Phys.step(p, c[0], c[1], w, h);
                full.add(p);
                if (w1.near(T + 1 + i) > Phys.distance(w1.ox, w1.oy, p.x(), p.y()) + Phys.BOT_RADIUS) { passIndex = i; break; }
            }
            if (passIndex < 0) passIndex = full.size() - 1;
            for (int k = 0; k <= passIndex + 1; k += (k < 8 ? 1 : 2)) {
                int stop = Math.min(k, passIndex + 1);
                ArrayList<Phys.Pose> path = new ArrayList<>();
                Phys.Pose q = me;
                for (int i = 0; i < 110; i++) {
                    if (i < stop) q = full.get(i);
                    else q = new Phys.Pose(q.x(), q.y(), q.h(), 0);
                    path.add(q);
                    if (w1.near(T + 1 + i) > Phys.distance(w1.ox, w1.oy, q.x(), q.y()) + Phys.BOT_RADIUS) break;
                }
                double d1 = danger(w1, path, T + 1) * Phys.damage(w1.power);
                double d2 = 0;
                Phys.Pose end = path.get(path.size() - 1);
                int Tend = T + path.size();
                if (w2 != null) d2 = 0.85 * secondWave(w2, end, Tend + 1, ex, ey, w, h) * Phys.damage(w2.power);
                double dist = Phys.distance(ex, ey, end.x(), end.y());
                double pos = Math.max(0, 300 - dist) / 300 * 1.5;
                double total = d1 + d2 + pos;
                boolean full_ = stop > passIndex;
                double turn, speed;
                if (stop == 0) {
                    double[] c = orbit(me, ex, ey, dir, w, h);
                    turn = c[0]; speed = 0;
                } else {
                    double[] c = orbit(me, ex, ey, dir, w, h);
                    turn = c[0]; speed = c[1];
                }
                if (best == null || total < best.danger()) best = new Plan(dir, full_ ? -1 : stop, total, turn, speed);
            }
        }
        return best;
    }

    private double secondWave(Wave wv, Phys.Pose start, int T0, double ex, double ey, double w, double h) {
        double best = Double.MAX_VALUE;
        for (int opt = -1; opt <= 1; opt++) {
            ArrayList<Phys.Pose> path = new ArrayList<>();
            Phys.Pose p = start;
            for (int i = 0; i < 110; i++) {
                if (opt == 0) p = new Phys.Pose(p.x(), p.y(), p.h(), 0);
                else { double[] c = orbit(p, ex, ey, opt, w, h); p = Phys.step(p, c[0], c[1], w, h); }
                path.add(p);
                if (wv.near(T0 + i) > Phys.distance(wv.ox, wv.oy, p.x(), p.y()) + Phys.BOT_RADIUS) break;
            }
            best = Math.min(best, danger(wv, path, T0));
        }
        return best;
    }

    /** Removes passed waves; records our crossing guess factor for the flattener. */
    void update(int T, Phys.Pose me) {
        waves.removeIf(wv -> {
            double D = Phys.distance(wv.ox, wv.oy, me.x(), me.y());
            if (!wv.visited && wv.near(T) >= D) {
                wv.visited = true;
                visits.add(wv.features, wv.gf(Phys.bearing(wv.ox, wv.oy, me.x(), me.y())), 1);
            }
            return wv.near(T) > D + 50;
        });
    }

    /** Legal hit label: the exact bullet heading of a bullet that hit us. */
    void hitBy(int T, double bx, double by, double heading, double power) {
        Wave match = null;
        double err = Double.MAX_VALUE;
        for (Wave wv : waves) {
            if (Math.abs(wv.power - power) > 0.001) continue;
            double r = Phys.distance(wv.ox, wv.oy, bx, by);
            double e = Math.min(Math.abs(r - wv.near(T)), Math.abs(r - wv.far(T)));
            if (r >= wv.near(T) - 2 * wv.speed && r <= wv.far(T) + 2 * wv.speed && e < err) { err = e; match = wv; }
        }
        enemyHits++;
        if (match == null) return;
        hits.add(match.features, Phys.clip(match.gf(heading), -1.1, 1.1), 1);
        waves.remove(match);
    }

    /** Our bullet destroyed an enemy bullet: label it too and drop the now-empty wave. */
    void bulletCollision(int T, double bx, double by, double heading, double power) {
        Wave match = null;
        for (Wave wv : waves) {
            if (Math.abs(wv.power - power) > 0.001) continue;
            double r = Phys.distance(wv.ox, wv.oy, bx, by);
            if (r >= wv.near(T) - 2 * wv.speed && r <= wv.far(T) + 2 * wv.speed) { match = wv; break; }
        }
        if (match == null) return;
        hits.add(match.features, Phys.clip(match.gf(heading), -1.1, 1.1), 0.5);
        waves.remove(match);
    }
}
