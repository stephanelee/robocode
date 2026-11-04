package prism.core;

/**
 * Pinned rule profile. Constants and formulas follow the official Tank Royale 1.4.0 server
 * (server/rules/rules.kt and math.kt at tag v1.4.0). Arena size and cooling rate come from the
 * delivered game setup.
 */
public record Rules(double width, double height, double coolingRate, int initialEnemies) {
    public static final String ENGINE_PROFILE = "tank-royale-1.4.0";
    public static final double BOT_RADIUS = 18.0;
    public static final double MAX_SPEED = 8.0;
    public static final double ACCELERATION = 1.0;
    public static final double DECELERATION = -2.0;
    public static final double MAX_TURN_DEG = 10.0;
    public static final double MAX_GUN_TURN_DEG = 20.0;
    public static final double MAX_RADAR_TURN_DEG = 45.0;
    public static final double MIN_POWER = 0.1;
    public static final double MAX_POWER = 3.0;
    public static final double INITIAL_GUN_HEAT = 3.0;
    public static final double RAM_DAMAGE = 0.6;
    public static final double RADAR_RANGE = 1200.0;
    /** Independent safety margin inside the engine's legal centre bounds. */
    public static final double SAFETY_MARGIN = 1.0;

    public Rules {
        if (!(width > 0 && height > 0 && coolingRate > 0)) throw new IllegalArgumentException("bad rules");
    }

    public static Rules standard() { return new Rules(800, 600, 0.1, 1); }

    public static double bulletSpeed(double power) { return 20 - 3 * clampPower(power); }

    public static double bulletDamage(double power) {
        double p = clampPower(power);
        double d = 4 * p;
        if (power > 1) d += 2 * (power - 1);
        return d;
    }

    public static double hitBonus(double power) { return 3 * clampPower(power); }

    public static double gunHeat(double power) { return 1 + clampPower(power) / 5; }

    public static double clampPower(double p) { return Math.max(MIN_POWER, Math.min(MAX_POWER, p)); }

    public static double maxTurnRad(double speed) {
        double s = Math.max(-MAX_SPEED, Math.min(MAX_SPEED, speed));
        return Math.toRadians(MAX_TURN_DEG - 0.75 * Math.abs(s));
    }

    public static double wallDamage(double speed) {
        double s = Math.max(-MAX_SPEED, Math.min(MAX_SPEED, speed));
        return Math.max(0, Math.abs(s) / 2 - 1);
    }

    /** Theoretical maximum escape angle for a bullet speed. */
    public static double maxEscape(double bulletSpeed) {
        return Math.asin(Math.min(1, MAX_SPEED / bulletSpeed));
    }

    /** Power that exactly kills a target with energy e (inverse of the damage rule). */
    public static double killPower(double e) {
        if (e <= 4) return e / 4;
        return (e + 2) / 6;
    }

    /**
     * Cools heat exactly as the engine does: each turn without firing subtracts the rate and
     * coerces at zero. Returns the number of whole turns until heat reaches exactly zero.
     */
    public int turnsToCool(double heat) {
        int n = 0;
        double h = heat;
        while (h > 0 && n < 10_000) { h = Math.max(0, h - coolingRate); n++; }
        return n;
    }

    public double minCenterX() { return BOT_RADIUS; }
    public double maxCenterX() { return width - BOT_RADIUS; }
    public double minCenterY() { return BOT_RADIUS; }
    public double maxCenterY() { return height - BOT_RADIUS; }

    /** True when the point is inside the legal centre bounds shrunk by the safety margin. */
    public boolean safe(double x, double y) {
        double m = BOT_RADIUS + SAFETY_MARGIN;
        return x >= m && x <= width - m && y >= m && y <= height - m;
    }

    public boolean safe(double x, double y, double margin) {
        double m = BOT_RADIUS + margin;
        return x >= m && x <= width - m && y >= m && y <= height - m;
    }

    public double clampX(double x, double margin) { return Math.max(BOT_RADIUS + margin, Math.min(width - BOT_RADIUS - margin, x)); }
    public double clampY(double y, double margin) { return Math.max(BOT_RADIUS + margin, Math.min(height - BOT_RADIUS - margin, y)); }
}
