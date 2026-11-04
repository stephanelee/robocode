/** Pure snapshot of the existing aimGF geometry; preserves accepted arithmetic and quirks. */
final class TargetingGeometry {
    private TargetingGeometry() {}

    record Snapshot(double shooterX, double shooterY, int arenaWidth, int arenaHeight,
                    int currentTurn, double gunDirection, double targetX, double targetY,
                    double targetDirection, double targetSpeed, double targetTurnRate,
                    int lastScanTurn, double initialDistance, double bulletSpeed,
                    double guessFactor, double lateralVelocity) {}

    record Aim(double x, double y, double baseAngle, double aimAngle,
               double lateralDirection, double gunTurn, double gunTurnRate) {}

    static Aim calculate(Snapshot s) {
        double maxAngle = Math.toDegrees(Math.asin(Math.min(1.0, 8.0 / s.bulletSpeed)));
        double latDir = Math.signum(s.lateralVelocity);
        if (latDir == 0) latDir = 1;

        double ticks = s.initialDistance / s.bulletSpeed;
        double[] prediction = null;
        for (int i = 0; i < 5; i++) {
            prediction = CombatMath.project(s.targetX, s.targetY, s.targetDirection, s.targetSpeed,
                    s.targetTurnRate, ticks + s.currentTurn - s.lastScanTurn);
            ticks = Math.hypot(prediction[0] - s.shooterX, prediction[1] - s.shooterY) / s.bulletSpeed;
        }
        double predX = prediction[0], predY = prediction[1];
        predX = clamp(predX, 18, s.arenaWidth - 18);
        predY = clamp(predY, 18, s.arenaHeight - 18);
        double baseAngle = Math.toDegrees(Math.atan2(predY - s.shooterY, predX - s.shooterX));
        // Exact IBaseBot.normalizeAbsoluteAngle semantics, including signed zero.
        baseAngle = (baseAngle %= 360) >= 0 ? baseAngle : baseAngle + 360;
        double aimAngle = baseAngle + latDir * s.guessFactor * maxAngle;
        double gunTurn = aimAngle - s.gunDirection;
        while (gunTurn > 180.0) gunTurn -= 360.0;
        while (gunTurn < -180.0) gunTurn += 360.0;
        return new Aim(predX, predY, baseAngle, aimAngle, latDir, gunTurn, clamp(gunTurn, -20, 20));
    }

    private static double clamp(double value, double low, double high) {
        return Math.max(low, Math.min(high, value));
    }
}
