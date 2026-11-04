/** Pinned Tank Royale 1.4 rules used by the planner and shield solver (east-zero degrees). */
final class Physics {
    static final double BOT_RADIUS = 18, MAX_SPEED = 8;
    private Physics() {}

    /** Exact copy of server rules calcNewBotSpeed. */
    static double newSpeed(double current, double target) {
        if (current < 0) return -newSpeed(-current, -target);
        if (current > 0) {
            if (Math.signum(current) == Math.signum(target)) {
                double diff = target - current;
                if (diff >= 0) return Math.min(current + Math.min(diff, 1.0), MAX_SPEED);
                return Math.max(current + Math.max(diff, -2.0), -MAX_SPEED);
            }
            if (target == 0) return 0;
            double decelerationTime = current / 2.0;
            return (1 - decelerationTime) * -1.0;
        }
        double diff = target - current, acceleration = Math.min(Math.abs(diff), 1.0);
        return diff >= 0 ? Math.min(current + acceleration, MAX_SPEED) : Math.max(current - acceleration, -MAX_SPEED);
    }

    static double bulletSpeed(double power) { return 20 - 3 * power; }
    static double gunHeat(double power) { return 1 + power / 5; }
    static double damage(double power) { return 4 * power + (power > 1 ? 2 * (power - 1) : 0); }
    static double normalize(double degrees) { double a = degrees % 360; if (a > 180) a -= 360; if (a < -180) a += 360; return a; }
    static double bearing(double x1, double y1, double x2, double y2) { return Math.toDegrees(Math.atan2(y2 - y1, x2 - x1)); }

    /** Server model/math isLineIntersectingLine (Franklin Antonio), without the collinear special case. */
    static boolean segmentsIntersect(double a1x, double a1y, double a2x, double a2y, double b1x, double b1y, double b2x, double b2y) {
        double ax = a2x - a1x, ay = a2y - a1y, bx = b1x - b2x, by = b1y - b2y, cx = a1x - b1x, cy = a1y - b1y;
        double alpha = by * cx - bx * cy, common = ay * bx - ax * by;
        if (common > 0 && (alpha < 0 || alpha > common)) return false;
        if (common < 0 && (alpha > 0 || alpha < common)) return false;
        double beta = ax * cy - ay * cx;
        if (common > 0 && (beta < 0 || beta > common)) return false;
        if (common < 0 && (beta > 0 || beta < common)) return false;
        return common != 0;
    }

    /** Server isLineIntersectingCircle. */
    static boolean segmentHitsCircle(double x1, double y1, double x2, double y2, double cx, double cy, double r) {
        if (Math.hypot(x1 - cx, y1 - cy) <= r || Math.hypot(x2 - cx, y2 - cy) <= r) return true;
        double dx = x2 - x1, dy = y2 - y1, len2 = dx * dx + dy * dy;
        if (len2 == 0) return false;
        double dot = ((cx - x1) * dx + (cy - y1) * dy) / len2;
        if (dot < 0 || dot > 1) return false;
        return Math.hypot(x1 + dot * dx - cx, y1 + dot * dy - cy) <= r;
    }
}
