package prism;

/**
 * Prism's independent initial profile (specification sections 6–12). These are proposed,
 * unmeasured choices — not recovered reference tuning. Changing any value requires a separately
 * registered experiment and a new {@link #VERSION}.
 */
public final class Profile {
    private Profile() {}

    public static final String VERSION = "prism-initial-profile-1";

    // Section 6: fire inference and power prediction
    public static final int MAX_SHOT_HYPOTHESES = 3;
    public static final int ENEMY_POWER_CAPACITY = 512;
    public static final double COLD_ENEMY_POWER = 2.0;
    public static final double FORECAST_CONFIDENCE = 0.25;
    public static final double ENERGY_TOLERANCE = 1e-6;

    // Section 7: learning
    public static final int LONG_GUN_CAPACITY = 8192;
    public static final int[] RECENT_GUN_CAPACITIES = {128, 512, 2048};
    public static final int MOVE_HIT_CAPACITY = 1024;
    public static final int MOVE_VISIT_CAPACITY = 2048;
    public static final int RECENT_HIT_CAPACITY = 64;
    public static final int MAX_INTERP_GAP = 8;

    // Section 8: gun
    public static final int GUN_OFFSETS = 61;
    public static final int LEARNED_GUN_MIN_CROSSINGS = 16;
    public static final double SELECTOR_ALPHA = 0.05;
    public static final double SELECTOR_MARGIN = 0.02;
    public static final int SELECTOR_MIN_RESOLVED = 8;
    public static final double DIFFICULTY_CAP = 10;
    public static final double DUEL_POWER = 2.0;
    public static final double CLOSE_POWER = 2.8;
    public static final double CLOSE_RANGE = 180;
    public static final double ENERGY_RESERVE = 0.2;
    public static final int PLAN_EXPIRY_TURNS = 2;
    public static final double MAX_FIRE_TOLERANCE = Math.toRadians(0.5);
    public static final double MIN_FIRE_TOLERANCE = Math.toRadians(0.05);
    public static final int EXPENSIVE_AIM_COOL_TURNS = 3;
    public static final double MELEE_POWER = 2.5;

    // Section 9: duel movement
    public static final double DESIRED_RANGE = 600;
    public static final double SURF_GAIN = 0.5;
    public static final double ORBIT_GAIN = 1.2;
    public static final double WALL_STICK = 140;
    public static final int WALL_SMOOTH_STEPS = 180;
    public static final int MAX_TRAJECTORY_TICKS = 160;
    public static final int MAX_PREDICTOR_STEPS = 2400;
    public static final double CLOSE_PENALTY = 0.05;
    public static final int FLATTENER_MIN_WAVES = 50;
    public static final double FLATTENER_MIN_LOWER_BOUND = 0.08;
    public static final double SHADOW_CAP_INFERRED = 0.75;

    // Section 10: melee
    public static final int MELEE_DESTINATIONS = 48;
    public static final double MELEE_MIN_RADIUS = 100;
    public static final double MELEE_MAX_RADIUS = 180;
    public static final double MELEE_RETAIN = 0.85;
    public static final int MELEE_SAFE_STEPS = 8;
    public static final int MELEE_GUN_SAMPLES = 80;

    // Section 11: radar
    public static final double RADAR_OVERSHOOT = Math.toRadians(20);
    public static final double RADAR_MIN_OVERSHOOT = Math.toRadians(2);
    public static final int RADAR_FULL_SWEEP_AFTER = 4;
    public static final int RADAR_MISMATCH_TURNS = 24;

    // Section 12: bounds
    public static final int MAX_WAVES_PER_ENEMY = 256;
    public static final int HISTORY_TICKS = 512;
    public static final long GLOBAL_STORE_BYTES = 128L * 1024 * 1024;
    public static final int FAILURES_TO_DISABLE = 3;

    /** Stable textual digest input describing every value above. */
    public static String describe() {
        return VERSION + "|" + MAX_SHOT_HYPOTHESES + "," + ENEMY_POWER_CAPACITY + "," + COLD_ENEMY_POWER + "," + FORECAST_CONFIDENCE
                + "|" + LONG_GUN_CAPACITY + "," + java.util.Arrays.toString(RECENT_GUN_CAPACITIES) + "," + MOVE_HIT_CAPACITY + "," + MOVE_VISIT_CAPACITY
                + "|" + GUN_OFFSETS + "," + LEARNED_GUN_MIN_CROSSINGS + "," + SELECTOR_ALPHA + "," + SELECTOR_MARGIN + "," + SELECTOR_MIN_RESOLVED
                + "|" + DUEL_POWER + "," + CLOSE_POWER + "," + CLOSE_RANGE + "," + ENERGY_RESERVE + "," + PLAN_EXPIRY_TURNS
                + "|" + DESIRED_RANGE + "," + SURF_GAIN + "," + ORBIT_GAIN + "," + WALL_STICK + "," + MAX_TRAJECTORY_TICKS + "," + MAX_PREDICTOR_STEPS
                + "|" + MELEE_DESTINATIONS + "," + MELEE_MIN_RADIUS + "," + MELEE_MAX_RADIUS + "," + MELEE_RETAIN
                + "|" + MAX_WAVES_PER_ENEMY + "," + HISTORY_TICKS + "," + GLOBAL_STORE_BYTES;
    }
}
