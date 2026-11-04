/** Independent 1v1 radar lock using only the most recent legal scan. */
final class DuelRadar {
    private DuelRadar() {}

    static double turn(double ownX, double ownY, double radarDirection,
                       double targetX, double targetY, int scanAge, int searchSign) {
        int sign = searchSign < 0 ? -1 : 1;
        if (scanAge < 0 || scanAge > 2) return sign * 45.0;
        double error = CombatMath.relativeAngle(CombatMath.bearing(targetX - ownX, targetY - ownY) - radarDirection);
        // Cross the observed bearing rather than settling on an exactly zero turn.
        double overshoot = Math.copySign(8.0, error == 0 ? sign : error);
        return Math.max(-45, Math.min(45, error * 2 + overshoot));
    }
}
