package prism.core;

/**
 * Engine-native orbit simulator shared by precise escape reachability and wall availability.
 * The target drives at full speed perpendicular to the source bearing in a signed orbit
 * direction, using back-as-front control and the pinned predictor. Wall availability counts steps
 * until the unsmoothed orbit touches a wall. Reachability uses light wall smoothing and records the
 * largest signed angular offset reached before the wave passes the target.
 */
public final class Escape {
    private Escape() {}

    public record Reach(double positive, double negative, int steps) {}

    /** Steps (capped) until a pure perpendicular orbit in direction s hits a wall. */
    public static int wallSteps(double sx, double sy, Physics.Motion m, int s, int cap, Rules rules) {
        Physics.Motion cur = m;
        for (int i = 1; i <= cap; i++) {
            double h = Angles.bearing(sx, sy, cur.x(), cur.y()) + s * Math.PI / 2;
            Drive.Control c = Drive.toward(cur, h, Rules.MAX_SPEED);
            Physics.Step st = Physics.step(cur, c.targetSpeed(), c.turnRate(), rules);
            if (st.hitWall()) return i - 1;
            cur = st.next();
        }
        return cap;
    }

    /**
     * Reachable escape magnitudes (radians, floored at 1e-6) for a wave from (sx,sy) with speed b
     * fired on turn fireTurn, target state m at turn stateTurn (stateTurn <= fireTurn allowed).
     */
    public static Reach reach(double sx, double sy, Physics.Motion m, int stateTurn, int fireTurn, double b,
                              int orbitSign, Rules rules) {
        int s = orbitSign == 0 ? 1 : orbitSign;
        double beta = Angles.bearing(sx, sy, m.x(), m.y());
        double pos = one(sx, sy, m, stateTurn, fireTurn, b, s, beta, rules);
        double neg = one(sx, sy, m, stateTurn, fireTurn, b, -s, beta, rules);
        return new Reach(Math.max(1e-6, pos), Math.max(1e-6, neg), 0);
    }

    private static double one(double sx, double sy, Physics.Motion m, int stateTurn, int fireTurn, double b,
                              int dir, double beta, Rules rules) {
        Physics.Motion cur = m;
        double best = 0;
        // dir is relative to the reference orbit sign; the caller maps it back: we measure the
        // signed offset in the direction of travel dir and return its magnitude.
        for (int t = stateTurn + 1; t <= stateTurn + 200; t++) {
            double brg = Angles.bearing(sx, sy, cur.x(), cur.y());
            double h = brg + dir * Math.PI / 2;
            // light smoothing: rotate toward the source until 30 units ahead is inside the field
            for (int k = 0; k < 90; k++) {
                double ex = cur.x() + Math.cos(h) * 30, ey = cur.y() + Math.sin(h) * 30;
                if (rules.safe(ex, ey, 0)) break;
                h += dir * Math.toRadians(2);
            }
            Drive.Control c = Drive.toward(cur, h, Rules.MAX_SPEED);
            cur = Physics.step(cur, c.targetSpeed(), c.turnRate(), rules).next();
            double off = dir * Angles.wrap(Angles.bearing(sx, sy, cur.x(), cur.y()) - beta);
            if (off > best) best = off;
            double d = Math.hypot(cur.x() - sx, cur.y() - sy);
            if (t >= fireTurn && (t - fireTurn + 2) * b >= d) break;
        }
        return best;
    }
}
