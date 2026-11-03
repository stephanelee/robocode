import java.nio.file.Path;
import java.util.Map;

/** Runtime settings shared by interactive launches and automated benchmarks. */
record BotConfig(Mode mode, long seed, Path modelPath, int fixedAction, Path metricsPath, DuelPolicy duelPolicy) {
    enum Mode { TRAIN, EVAL, FIXED }
    enum DuelPolicy { CONTROLLER, LEARNED }

    BotConfig(Mode mode, long seed, Path modelPath, int fixedAction, Path metricsPath) {
        this(mode, seed, modelPath, fixedAction, metricsPath, DuelPolicy.CONTROLLER);
    }

    boolean training() { return mode == Mode.TRAIN; }
    boolean forceDuel() { return training() && duelPolicy == DuelPolicy.CONTROLLER; }

    static BotConfig fromEnvironment(Map<String, String> env) {
        Mode mode;
        try { mode = Mode.valueOf(env.getOrDefault("AISURVIVOR_MODE", "train").toUpperCase(java.util.Locale.ROOT)); }
        catch (IllegalArgumentException ex) { throw new IllegalArgumentException("AISURVIVOR_MODE must be train, eval, or fixed", ex); }
        long seed = Long.parseLong(env.getOrDefault("AISURVIVOR_SEED", "0"));
        int action = Integer.parseInt(env.getOrDefault("AISURVIVOR_ACTION", "0"));
        if (action < 0 || action >= 6) throw new IllegalArgumentException("AISURVIVOR_ACTION must be 0..5");
        String modelValue = env.getOrDefault("AISURVIVOR_MODEL_PATH", "AISurvivor-qtable.dat");
        if (modelValue.isBlank()) throw new IllegalArgumentException("AISURVIVOR_MODEL_PATH must not be blank");
        Path model = Path.of(modelValue);
        String metrics = env.get("AISURVIVOR_METRICS_PATH");
        if (metrics != null && metrics.isBlank()) throw new IllegalArgumentException("AISURVIVOR_METRICS_PATH must not be blank");
        Path metricsPath = metrics == null ? null : Path.of(metrics);
        if (metricsPath != null && model.toAbsolutePath().normalize().equals(metricsPath.toAbsolutePath().normalize()))
            throw new IllegalArgumentException("Model and metrics paths must differ");
        DuelPolicy duel;
        try { duel = DuelPolicy.valueOf(env.getOrDefault("AISURVIVOR_DUEL_POLICY", "controller").toUpperCase(java.util.Locale.ROOT)); }
        catch (IllegalArgumentException ex) { throw new IllegalArgumentException("AISURVIVOR_DUEL_POLICY must be controller or learned", ex); }
        return new BotConfig(mode, seed, model, action, metricsPath, duel);
    }
}
