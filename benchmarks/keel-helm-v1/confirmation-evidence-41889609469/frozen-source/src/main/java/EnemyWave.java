/** An inferred enemy bullet wave; unmatched waves never train danger statistics. */
final class EnemyWave {
    final double originX, originY, bulletSpeed, bearingToUs;
    final int fireTurn, shooterId;
    /** Head-on and linear aim bearings reconstructed from our own state at the inferred aim turn, if known. */
    double headOnAim = Double.NaN, linearAim = Double.NaN, aimDistance = Double.NaN;

    EnemyWave(double x, double y, double speed, double bearing, int turn, int shooter) {
        originX = x; originY = y; bulletSpeed = speed;
        bearingToUs = bearing; fireTurn = turn; shooterId = shooter;
    }

    void setAims(double headOn, double linear, double distance) {
        headOnAim = headOn; linearAim = linear; aimDistance = distance;
    }

    /** Larger of the two aim-corridor dangers at a point; zero when our aiming-time state is unknown. */
    double aimDanger(double x, double y) {
        if (Double.isNaN(aimDistance)) return 0;
        return Math.max(AimCorridor.danger(originX, originY, aimDistance, headOnAim, x, y),
                AimCorridor.danger(originX, originY, aimDistance, linearAim, x, y));
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

    void learnHit(double[] danger, double x, double y, boolean training) {
        if (training && danger != null) danger[hitBin(x, y, danger.length)]++;
    }
}
