/**
 * Tank Royale 1.4.0 physics in degrees, counterclockwise from east. Pure functions; no bot API.
 * Speed law follows the pinned server's calcNewBotSpeed (a zero target stops a moving bot at once);
 * movement uses the updated speed on the old heading, then the turn is limited by the new speed.
 */
final class Phys {
    private Phys() {}
    static final double BOT_RADIUS = 18;

    record Pose(double x, double y, double h, double v) {}

    static double clip(double x, double lo, double hi) { return Math.max(lo, Math.min(x, hi)); }
    static double norm(double a) { return Math.IEEEremainder(a, 360); }
    static double angle(double dx, double dy) { return Math.toDegrees(Math.atan2(dy, dx)); }
    static double bearing(double fx, double fy, double tx, double ty) { return angle(tx - fx, ty - fy); }
    static double cos(double h) { return Math.cos(Math.toRadians(h)); }
    static double sin(double h) { return Math.sin(Math.toRadians(h)); }
    static double bulletSpeed(double power) { return 20 - 3 * power; }
    static double mea(double power) { return Math.toDegrees(Math.asin(8 / bulletSpeed(power))); }
    static double maxTurn(double v) { return 10 - 0.75 * Math.abs(v); }
    static double damage(double p) { return 4 * p + (p > 1 ? 2 * (p - 1) : 0); }
    static double gunHeat(double p) { return 1 + p / 5; }

    static double speed(double v, double target) {
        if (v < 0) return -speed(-v, -target);
        if (v > 0) {
            if (target > 0) {
                double diff = target - v;
                return diff >= 0 ? Math.min(v + Math.min(diff, 1), 8) : Math.max(v + Math.max(diff, -2), -8);
            }
            if (target == 0) return 0;
            return v / 2 - 1; // (1 - v / -DECELERATION) * -ACCELERATION, DECELERATION = -2
        }
        double diff = target;
        double acc = Math.min(Math.abs(diff), 1);
        return diff >= 0 ? Math.min(acc, 8) : Math.max(-acc, -8);
    }

    private static double[] constrain(double ox, double oy, double x, double y, double w, double h) {
        double nx = x, ny = y;
        if (x < 18 || x > w - 18) {
            nx = clip(x, 18, w - 18);
            if (x != ox) ny = oy + (y - oy) * (nx - ox) / (x - ox);
        }
        if (y < 18 || y > h - 18) {
            ny = clip(y, 18, h - 18);
            if (y != oy) nx = ox + (x - ox) * (ny - oy) / (y - oy);
        }
        return new double[]{nx, ny};
    }

    /** One engine turn for a body command (turn rate request, target speed). */
    static Pose step(Pose p, double turn, double target, double w, double h) {
        double v = speed(p.v(), target);
        double x = p.x() + cos(p.h()) * v, y = p.y() + sin(p.h()) * v;
        double cap = maxTurn(v);
        double heading = p.h() + clip(turn, -cap, cap);
        double[] first = constrain(p.x(), p.y(), x, y, w, h);
        if (first[0] != x || first[1] != y) v = 0;
        double[] second = constrain(p.x(), p.y(), first[0], first[1], w, h);
        return new Pose(second[0], second[1], heading, v);
    }

    static double distance(double ax, double ay, double bx, double by) { return Math.hypot(ax - bx, ay - by); }
}
