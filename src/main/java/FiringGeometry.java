/** Current-heading firing gate; the server fires before this turn's gun rotation. */
final class FiringGeometry {
    private FiringGeometry() {}

    static double tolerance(double predictedRange) {
        if (!Double.isFinite(predictedRange) || predictedRange < 0) return 0;
        return Math.min(10.0, Math.toDegrees(Math.atan2(18.0, predictedRange)));
    }

    static boolean aligned(double gunTurn, double predictedRange) {
        return Double.isFinite(gunTurn) && Math.abs(gunTurn) < tolerance(predictedRange);
    }
}
