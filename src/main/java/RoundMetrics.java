import java.io.IOException;
import java.nio.file.*;
import java.util.Locale;

/** One JSON object per completed round; enabled only with an explicit output path. */
final class RoundMetrics {
    int shotsFired, bulletHits, wallCollisions, botCollisions, skippedTurns;
    final int[] actionCounts = new int[6];
    int forcedDuelTurns, duelEligibleTurns;
    double damageDealt, damageTaken, reward;
    /** Per-turn decision latency, tick receipt to go(), in microseconds. */
    long maxTurnMicros;
    int slowTurns;
    static final long SLOW_TURN_MICROS = 20_000;

    void recordTurnLatency(long micros) {
        maxTurnMicros = Math.max(maxTurnMicros, micros);
        if (micros >= SLOW_TURN_MICROS) slowTurns++;
    }

    void reset() {
        shotsFired = bulletHits = wallCollisions = botCollisions = skippedTurns = 0;
        damageDealt = damageTaken = reward = 0;
        java.util.Arrays.fill(actionCounts, 0);
        forcedDuelTurns = duelEligibleTurns = 0;
        maxTurnMicros = 0;
        slowTurns = 0;
    }

    void write(BotConfig config, int round, int turns, int totalRounds,
               boolean checkpointLoaded, boolean won, boolean died) {
        if (config.metricsPath() == null) return;
        try {
            Path path = config.metricsPath().toAbsolutePath();
            Files.createDirectories(path.getParent());
            Files.writeString(path, json(config, round, turns, totalRounds, checkpointLoaded, won, died) + "\n",
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException ex) { System.out.println("[AISurvivor] Metrics failed: " + ex.getMessage()); }
    }

    String json(BotConfig config, int round, int turns, int totalRounds,
                boolean checkpointLoaded, boolean won, boolean died) {
        return String.format(Locale.ROOT,
                "{\"schemaVersion\":1,\"round\":%d,\"turns\":%d,\"mode\":\"%s\",\"seed\":%d,\"action\":%s,"
                + "\"totalRounds\":%d,\"checkpointLoaded\":%s,\"shotsFired\":%d,\"bulletHits\":%d,"
                + "\"hitRate\":%.6f,\"damageDealt\":%.6f,\"damageTaken\":%.6f,\"wallCollisions\":%d,"
                + "\"botCollisions\":%d,\"skippedTurns\":%d,\"won\":%s,\"died\":%s,\"reward\":%.6f,"
                + "\"duelPolicy\":\"%s\",\"actionCounts\":%s,\"duelEligibleTurns\":%d,\"forcedDuelTurns\":%d,"
                + "\"maxTurnMicros\":%d,\"slowTurns\":%d}",
                round, turns, config.mode().name().toLowerCase(Locale.ROOT), config.seed(),
                config.mode() == BotConfig.Mode.FIXED ? Integer.toString(config.fixedAction()) : "null",
                totalRounds, checkpointLoaded, shotsFired, bulletHits,
                shotsFired == 0 ? 0 : (double) bulletHits / shotsFired, damageDealt, damageTaken,
                wallCollisions, botCollisions, skippedTurns, won, died, reward,
                config.duelPolicy().name().toLowerCase(Locale.ROOT), java.util.Arrays.toString(actionCounts),
                duelEligibleTurns, forcedDuelTurns, maxTurnMicros, slowTurns);
    }
}
