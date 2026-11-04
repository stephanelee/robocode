/**
 * Pure geometry for dodging the two simple aim models a duel opponent can fire along an inferred wave:
 * head-on at our position when it aimed, and iterative wall-clamped linear prediction from our velocity then.
 * Inputs are only our own recorded state and the inferred wave origin/speed; no opponent-private state is used.
 */
final class AimCorridor {
    static final double BOT_RADIUS = 18.0;
    static final int MAX_LINEAR_TURNS = 150;

    private AimCorridor() {}

    /** Our own state at a turn, as recorded by this bot. */
    record OwnState(int turn, double x, double y, double direction, double speed) {}

    static double headOn(double originX, double originY, OwnState own) {
        return CombatMath.bearing(own.x() - originX, own.y() - originY);
    }

    /** Constant-velocity prediction stepped one turn at a time and clamped to the arena by bot radius. */
    static double linear(double originX, double originY, OwnState own, double bulletSpeed,
                         double arenaWidth, double arenaHeight) {
        double heading = Math.toRadians(own.direction());
        double dx = Math.cos(heading) * own.speed(), dy = Math.sin(heading) * own.speed();
        double x = own.x(), y = own.y();
        for (int k = 1; k <= MAX_LINEAR_TURNS; k++) {
            x = clamp(x + dx, BOT_RADIUS, arenaWidth - BOT_RADIUS);
            y = clamp(y + dy, BOT_RADIUS, arenaHeight - BOT_RADIUS);
            if (bulletSpeed * k >= Math.hypot(x - originX, y - originY)) break;
        }
        return CombatMath.bearing(x - originX, y - originY);
    }

    /**
     * Danger in [0, 1] for standing at (px, py) when a bullet aimed at {@code aim} arrives. The angular corridor
     * combines the shooter's alignment tolerance at the aiming distance with our hit radius at the point.
     */
    static double danger(double originX, double originY, double aimDistance, double aim, double px, double py) {
        double distance = Math.hypot(px - originX, py - originY);
        if (!(distance > 0) || !Double.isFinite(aim) || !Double.isFinite(aimDistance)) return 0;
        double tolerance = Math.toDegrees(Math.atan(BOT_RADIUS / Math.max(aimDistance, BOT_RADIUS)));
        double body = Math.toDegrees(Math.atan(BOT_RADIUS / Math.max(distance, BOT_RADIUS)));
        double corridor = tolerance + body;
        double offset = Math.abs(CombatMath.relativeAngle(CombatMath.bearing(px - originX, py - originY) - aim));
        return offset < corridor ? 1 - offset / corridor : 0;
    }

    private static double clamp(double value, double low, double high) {
        return Math.max(low, Math.min(high, value));
    }
}
