/** An inferred enemy bullet wave; unmatched waves never train danger statistics. */
final class EnemyWave {
    final double originX, originY, bulletSpeed, bearingToUs;
    final int fireTurn, shooterId;

    EnemyWave(double x, double y, double speed, double bearing, int turn, int shooter) {
        originX = x; originY = y; bulletSpeed = speed;
        bearingToUs = bearing; fireTurn = turn; shooterId = shooter;
    }

    double remaining(double x, double y, int turn) {
        return Math.hypot(x - originX, y - originY) - (turn - fireTurn) * bulletSpeed;
    }

    double matchError(int shooter, double power, double x, double y, int turn) {
        if (shooter != shooterId || Math.abs((20 - 3 * power) - bulletSpeed) > 0.01)
            return Double.POSITIVE_INFINITY;
        double error = Math.abs(remaining(x, y, turn));
        return error <= bulletSpeed + 18 ? error : Double.POSITIVE_INFINITY;
    }

    int hitBin(double x, double y, int bins) {
        double angle = CombatMath.bearing(x - originX, y - originY);
        double maxEscape = Math.toDegrees(Math.asin(Math.min(1, 8 / bulletSpeed)));
        return CombatMath.gfBin(CombatMath.relativeAngle(angle - bearingToUs) / maxEscape, bins);
    }
}
