import com.google.gson.Gson;
import com.google.gson.JsonObject;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.*;

/** Snapshot each round once. Cumulative totals are accumulated from the same sealed numeric row. */
final class BreakwaterLedger {
    static final class Counters {
        int turns, shotsFired, bulletHits, attackHits, attackShots, unclassifiedShots;
        int inferredWaves, matchedRays, wallCollisions, botCollisions;
        int decisions, waveDecisions, cwDecisions, ccwDecisions, movingTargetTurns, observedMovingTurns;
        int linearDecisions, stoppedDecisions, circularDecisions;
        int envelopeDecisions, envelopeOpportunities, envelopeHolds, envelopeFireRequests, envelopeShots, envelopeHits;
        int linearShots, stoppedShots, circularShots, linearHits, stoppedHits, circularHits;
        int skippedTurns, rangeObservedTurns;
        double damageDealt, damageTaken, attackDamage, energySpent, rangeSum;
    }
    private static final Gson JSON = new Gson();
    private final Path metrics, endpoint;
    private final long seed;
    private final String gunMode;
    private final BufferedWriter shadow, scans;
    private final JsonObject totals = new JsonObject();
    private int started, completed, officialFirstPlaces;
    boolean won, died;
    Counters row;

    BreakwaterLedger(long seed, Path metrics, Path endpoint, String mode, Path shadowPath) {
        if (!"shadow".equals(mode) && !"envelope".equals(mode)) throw new IllegalArgumentException("Invalid gun mode");
        if (mode.equals("shadow") != (shadowPath != null)) throw new IllegalArgumentException("Missing or unexpected shadow output");
        this.gunMode = mode;
        this.seed = seed;
        this.metrics = metrics.toAbsolutePath().normalize();
        this.endpoint = endpoint.toAbsolutePath().normalize();
        if (this.metrics.equals(this.endpoint) || Files.exists(this.metrics) || Files.exists(this.endpoint))
            throw new IllegalArgumentException("Breakwater needs distinct absent metrics and endpoint files");
        if (shadowPath == null) {
            shadow = scans = null;
        } else {
            var path = shadowPath.toAbsolutePath().normalize();
            var scanPath = path.resolveSibling("shadow-scans.jsonl");
            if (path.equals(this.metrics) || path.equals(this.endpoint) || scanPath.equals(path) ||
                    scanPath.equals(this.metrics) || scanPath.equals(this.endpoint))
                throw new IllegalArgumentException("Shadow outputs overlap evidence");
            try {
                Files.createDirectories(path.getParent());
                shadow = Files.newBufferedWriter(path, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
                scans = Files.newBufferedWriter(scanPath, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
            } catch (IOException ex) { throw new UncheckedIOException(ex); }
        }
    }

    private record Scan(String kind, int round, int turn, double x, double y, double heading, double speed, double energy) {}

    void recordScan(BreakwaterPolicy.Target t) {
        if (!open(started) || scans == null) throw new IllegalStateException("Unexpected shadow scan");
        try {
            scans.write(JSON.toJson(new Scan("scan", started, t.turn(), t.x(), t.y(), t.heading(), t.speed(), t.energy())));
            scans.newLine();
        } catch (IOException ex) { throw new UncheckedIOException(ex); }
    }

    void recordShadow(BreakwaterEnvelope.Row entry) {
        if (!open(entry.round()) || shadow == null) throw new IllegalStateException("Unexpected shadow record");
        try {
            shadow.write(JSON.toJson(entry));
            shadow.newLine();
        } catch (IOException ex) { throw new UncheckedIOException(ex); }
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

    void seal(int round, int turns, int firstPlaces, double[] corrections, int[] attackScores, int[] attackHistory) {
        if (round <= completed) return;
        if (!open(round)) throw new IllegalStateException("Unopened round end");
        int roundWins = firstPlaces - officialFirstPlaces;
        if (roundWins < 0 || roundWins > 1) throw new IllegalStateException("Invalid official first-place increment");
        officialFirstPlaces = firstPlaces;
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
        result.addProperty("officialFirstPlaces", officialFirstPlaces);
        result.addProperty("officialRoundWins", roundWins);
        result.addProperty("won", won);
        result.addProperty("died", died);
        JsonObject model = identity();
        model.addProperty("roundsCompleted", completed);
        model.addProperty("officialFirstPlaces", officialFirstPlaces);
        model.add("totals", totals);
        model.add("corrections", JSON.toJsonTree(corrections));
        model.add("attackScores", JSON.toJsonTree(attackScores));
        model.add("attackHistory", JSON.toJsonTree(attackHistory));
        try {
            if (shadow != null) {
                shadow.flush();
                scans.flush();
            }
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
        result.addProperty("schemaVersion", 3);
        result.addProperty("name", "Breakwater");
        result.addProperty("version", "0.3.0");
        result.addProperty("gunMode", gunMode);
        result.addProperty("seed", seed);
        result.addProperty("checkpointLoaded", false);
        return result;
    }
}
