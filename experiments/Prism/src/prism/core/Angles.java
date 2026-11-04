package prism.core;

/**
 * Native angle convention used everywhere inside Prism: radians, east = 0, counterclockwise
 * positive, Cartesian positions. Tank Royale reports degrees with the same orientation, so the
 * only conversion at the API boundary is degrees to radians.
 */
public final class Angles {
    public static final double TWO_PI = 2 * Math.PI;

    private Angles() {}

    /** Wraps a difference into [-pi, pi). */
    public static double wrap(double a) {
        if (!Double.isFinite(a)) throw new IllegalArgumentException("non-finite angle");
        double r = (a + Math.PI) % TWO_PI;
        if (r < 0) r += TWO_PI;
        return r - Math.PI;
    }

    /** Wraps a heading into [0, 2*pi). */
    public static double heading(double a) {
        if (!Double.isFinite(a)) throw new IllegalArgumentException("non-finite angle");
        double r = a % TWO_PI;
        if (r < 0) r += TWO_PI;
        return r >= TWO_PI ? 0 : r;
    }

    public static double bearing(double fromX, double fromY, double toX, double toY) {
        return heading(Math.atan2(toY - fromY, toX - fromX));
    }

    /** Effective travel heading: the body heading when moving forward, reversed when backing. */
    public static double effectiveHeading(double bodyHeading, double signedSpeed) {
        return signedSpeed < 0 ? heading(bodyHeading + Math.PI) : heading(bodyHeading);
    }

    /**
     * Signed mathematical orbit direction of a target around a source: +1 when the target moves
     * counterclockwise around the source, -1 clockwise, 0 when lateral motion is negligible.
     */
    public static int orbitSign(double srcX, double srcY, double x, double y, double heading, double speed) {
        double lateral = lateralSpeed(srcX, srcY, x, y, heading, speed);
        if (Math.abs(lateral) < 1e-9) return 0;
        return lateral > 0 ? 1 : -1;
    }

    /** Signed tangential velocity (counterclockwise around the source is positive). */
    public static double lateralSpeed(double srcX, double srcY, double x, double y, double heading, double speed) {
        double b = Math.atan2(y - srcY, x - srcX);
        return speed * Math.sin(heading - b);
    }

    /** Signed velocity toward the source (approaching is positive). */
    public static double advancingSpeed(double srcX, double srcY, double x, double y, double heading, double speed) {
        double b = Math.atan2(y - srcY, x - srcX);
        return -speed * Math.cos(heading - b);
    }

    /** Classic Robocode angle for analyst comparison only; never used for decisions. */
    public static double classicFromNative(double nativeAngle) {
        return heading(Math.PI / 2 - nativeAngle);
    }
}
