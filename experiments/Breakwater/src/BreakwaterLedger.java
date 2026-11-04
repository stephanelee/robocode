import com.google.gson.Gson;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.*;

/** Snapshot each round once. Cumulative totals are accumulated from the same sealed numeric row. */
final class BreakwaterLedger {
    static final class Counters {
        int turns, shotsFired, bulletHits, attackHits, shieldHits, shieldShots, attackShots, unclassifiedShots;
        int shieldIntercepts, inferredWaves, shieldSolutions, matchedRays, wallCollisions, botCollisions;
        int skippedTurns, rangeObservedTurns;
        double damageDealt, damageTaken, attackDamage, shieldDamage, energySpent, rangeSum;
    }
    private static final Gson JSON = new Gson();
    private final Path metrics, endpoint;
    private final long seed;
    private final JsonObject totals = new JsonObject();
    private int started, completed;
    boolean won, died;
    Counters row;

    BreakwaterLedger(long seed, Path metrics, Path endpoint) {
        this.seed = seed;
        this.metrics = metrics.toAbsolutePath().normalize();
        this.endpoint = endpoint.toAbsolutePath().normalize();
        if (this.metrics.equals(this.endpoint) || Files.exists(this.metrics) || Files.exists(this.endpoint))
            throw new IllegalArgumentException("Breakwater needs distinct absent metrics and endpoint files");
    }

    boolean begin(int round) {
        if (round == started) return false;
        if (round != completed + 1 || started != completed) throw new IllegalStateException("Nonsequential round start");
        started = round;
        row = new Counters();
        won = died = false;
        return true;
    }

    boolean open(int round) { return started == round && started > completed; }

    void seal(int round, int turns, double[] corrections, int[] attackScores, int[] attackHistory) {
        if (round <= completed) return;
        if (!open(round)) throw new IllegalStateException("Unopened round end");
        completed = round;
        row.turns = turns;
        JsonObject numeric = JSON.toJsonTree(row).getAsJsonObject();
        JsonObject result = identity();
        for (var entry : numeric.entrySet()) {
            String key = entry.getKey();
            var value = entry.getValue().getAsNumber();
            result.addProperty(key, value);
            Number prior = totals.has(key) ? totals.get(key).getAsNumber() : 0;
            if (value instanceof Double) totals.addProperty(key, prior.doubleValue() + value.doubleValue());
            else totals.addProperty(key, prior.longValue() + value.longValue());
        }
        result.addProperty("round", completed);
        result.addProperty("won", won);
        result.addProperty("died", died);
        JsonObject model = identity();
        model.addProperty("roundsCompleted", completed);
        model.add("totals", totals);
        model.add("corrections", JSON.toJsonTree(corrections));
        model.add("attackScores", JSON.toJsonTree(attackScores));
        model.add("attackHistory", JSON.toJsonTree(attackHistory));
        try {
            Files.createDirectories(metrics.getParent());
            Files.createDirectories(endpoint.getParent());
            Files.writeString(metrics, result + "\n", StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            Path tmp = Files.createTempFile(endpoint.getParent(), "breakwater-", ".tmp");
            Files.writeString(tmp, model + "\n");
            Files.move(tmp, endpoint, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException ex) { throw new UncheckedIOException(ex); }
    }

    private JsonObject identity() {
        JsonObject result = new JsonObject();
        result.addProperty("schemaVersion", 1);
        result.addProperty("name", "Breakwater");
        result.addProperty("version", "0.1.0");
        result.addProperty("seed", seed);
        result.addProperty("checkpointLoaded", false);
        return result;
    }
}
