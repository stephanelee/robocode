import java.io.IOException;
import java.nio.file.*;
import java.util.Locale;

/** One JSON object per completed round; enabled only with an explicit output path. */
final class RoundMetrics {
    int shotsFired, bulletHits, wallCollisions, botCollisions, skippedTurns;
    double damageDealt, damageTaken, reward;

    void reset() {
        shotsFired = bulletHits = wallCollisions = botCollisions = skippedTurns = 0;
        damageDealt = damageTaken = reward = 0;
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
                + "\"botCollisions\":%d,\"skippedTurns\":%d,\"won\":%s,\"died\":%s,\"reward\":%.6f}",
                round, turns, config.mode().name().toLowerCase(Locale.ROOT), config.seed(),
                config.mode() == BotConfig.Mode.FIXED ? Integer.toString(config.fixedAction()) : "null",
                totalRounds, checkpointLoaded, shotsFired, bulletHits,
                shotsFired == 0 ? 0 : (double) bulletHits / shotsFired, damageDealt, damageTaken,
                wallCollisions, botCollisions, skippedTurns, won, died, reward);
    }
}
