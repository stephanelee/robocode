package prism.learn;

import prism.core.Angles;
import prism.core.Escape;
import prism.core.Physics;
import prism.core.Rules;

/**
 * Frozen-at-origin context (specification section 7.1). Each wave stores one {@link Context};
 * labels later attach to the stored vectors, never to recomputed outcome-time features.
 */
public final class Features {
    private Features() {}

    /** Raw normalized coordinates. {@code quality} is halved once if any feature was missing. */
    public record Context(double power, double flight, double lateral, double advancing, double accel,
                          double headingSin, double headingCos, double fwdWall, double revWall,
                          double velocityAge, double reversalAge, double disp8, double disp20, double disp40,
                          double fireProximity, double population, double quality, boolean meleeOrigin) {

        /** Long gun: power, flight, lateral, advancing, sin, cos, accel, walls, velocity age, fire proximity. */
        public double[] longGun() { return new double[] {power, flight, lateral, advancing, headingSin, headingCos, accel, fwdWall, revWall, velocityAge, fireProximity}; }
        /** Recent gun: same movement context without power or fire proximity. */
        public double[] recentGun() { return new double[] {flight, lateral, advancing, headingSin, headingCos, accel, fwdWall, revWall, velocityAge}; }
        /** Melee displacement gun: flight, movement, walls, recent displacement. */
        public double[] meleeGun() { return new double[] {flight, lateral, advancing, headingSin, headingCos, accel, fwdWall, revWall, disp8, disp20, disp40}; }
        public double[] simpleHit() { return new double[] {flight, lateral, accel}; }
        public double[] contextHit() { return new double[] {flight, lateral, accel, headingSin, headingCos, fwdWall, revWall, velocityAge, disp8}; }
        public double[] visit() { return new double[] {flight, lateral, accel, headingSin, headingCos, fwdWall, revWall, velocityAge, disp8, reversalAge, disp20, disp40}; }
    }

    // weights: heading 1; flight, lateral, acceleration, walls 2; all others 1 (fixed in v1)
    public static final double[] LONG_W = {1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1};
    public static final double[] RECENT_W = {2, 2, 1, 1, 1, 2, 2, 2, 1};
    public static final double[] MELEE_W = {2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1};
    public static final double[] SIMPLE_W = {2, 2, 2};
    public static final double[] CONTEXT_W = {2, 2, 2, 1, 1, 2, 2, 1, 1};
    public static final double[] VISIT_W = {2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1};

    /** Inputs describing the mover at the origin turn. Missing displacement uses NaN. */
    public record Mover(Physics.Motion motion, double accel, int velocityAge, int reversalAge,
                        double disp8, double disp20, double disp40) {}

    public static Context compute(double sx, double sy, Mover mv, double power, double fireProximity,
                                  double population, boolean meleeOrigin, int orbitSign, Rules rules) {
        Physics.Motion m = mv.motion();
        double b = Rules.bulletSpeed(power);
        double range = Math.hypot(m.x() - sx, m.y() - sy);
        double F = range / b, H = Math.max(1, F);
        double lat = Angles.lateralSpeed(sx, sy, m.x(), m.y(), m.heading(), m.speed()) / 8;
        double adv = Angles.advancingSpeed(sx, sy, m.x(), m.y(), m.heading(), m.speed()) / 8;
        double a = Math.max(-2, Math.min(1, mv.accel()));
        double eff = Angles.effectiveHeading(m.heading(), m.speed());
        double rel = eff - Angles.bearing(sx, sy, m.x(), m.y());
        int s = orbitSign == 0 ? 1 : orbitSign;
        int cap = (int) Math.ceil(1.5 * H);
        double fw = Math.min(1.5, Escape.wallSteps(sx, sy, m, s, cap, rules) / H);
        double rw = Math.min(1.5, Escape.wallSteps(sx, sy, m, -s, cap, rules) / H);
        boolean missing = false;
        double d8 = mv.disp8(), d20 = mv.disp20(), d40 = mv.disp40();
        if (Double.isNaN(d8)) { d8 = 0; missing = true; }
        if (Double.isNaN(d20)) { d20 = 0; missing = true; }
        if (Double.isNaN(d40)) { d40 = 0; missing = true; }
        double q = missing ? 0.5 : 1.0;
        return new Context(power / 3, Math.min(F, 100) / 100, lat, adv, (a + 2) / 3,
                Math.sin(rel), Math.cos(rel), fw, rw,
                Math.min(mv.velocityAge() / H, 1), Math.min(mv.reversalAge() / H, 1),
                d8 / 64, d20 / 160, d40 / 320, Math.max(0, Math.min(1, fireProximity / 8)), population, q, meleeOrigin);
    }
}
