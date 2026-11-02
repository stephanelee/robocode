/** Full-circle heading lookahead using Tank Royale's east-zero coordinates. */
final class WallSmoothing {
    private static final double EPSILON = 1e-9;

    private WallSmoothing() {}

    static double heading(double x, double y, double width, double height,
                          double margin, double stick, double desired, int rotation) {
        if (!Double.isFinite(x) || !Double.isFinite(y) || !Double.isFinite(width)
                || !Double.isFinite(height) || !Double.isFinite(margin)
                || !Double.isFinite(stick) || !Double.isFinite(desired)
                || width <= 0 || height <= 0 || margin < 0 || stick <= 0
                || x < 0 || x > width || y < 0 || y > height
                || (rotation != 1 && rotation != -1)) {
            throw new IllegalArgumentException("Invalid wall-smoothing geometry or rotation");
        }
        // Include the desired heading, then every remaining five-degree sample.
        for (int step = 0; step < 72; step++) {
            double candidate = normalize(desired + rotation * step * 5.0);
            double[] endpoint = endpoint(x, y, candidate, stick);
            if (inside(endpoint[0], endpoint[1], width, height, margin)) return candidate;
        }

        // Near a corner the inner margin may be unreachable with this stick length.
        // Aim inward and check the segment ending at or before the arena centre.
        double dx = width / 2 - x, dy = height / 2 - y;
        double distance = Math.hypot(dx, dy);
        double inward = distance > EPSILON ? normalize(CombatMath.bearing(dx, dy)) : normalize(desired);
        double[] endpoint = endpoint(x, y, inward, Math.min(stick, distance));
        if (!inside(endpoint[0], endpoint[1], width, height, 0)) {
            throw new IllegalStateException("Inward wall-smoothing fallback left the arena");
        }
        return inward;
    }

    private static double[] endpoint(double x, double y, double heading, double distance) {
        double angle = Math.toRadians(heading);
        return new double[]{x + Math.cos(angle) * distance, y + Math.sin(angle) * distance};
    }

    private static boolean inside(double x, double y, double width, double height, double margin) {
        return x >= margin - EPSILON && x <= width - margin + EPSILON
                && y >= margin - EPSILON && y <= height - margin + EPSILON;
    }

    private static double normalize(double angle) { return ((angle % 360) + 360) % 360; }
}
