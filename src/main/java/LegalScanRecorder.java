import com.google.gson.*;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;

/** Optional legal observations only; buffered writes never choose or change bot commands. */
final class LegalScanRecorder {
    record Own(double x, double y, double direction, double speed, double gunDirection,
               Double radarDirection, Double energy) {}
    record Scan(int observedTurn, double x, double y, double direction, double speed, Double energy) {}
    record Target(int targetId, Scan scan) {}
    record Intent(int submittedTurn, double targetSpeed, double turnRate, double gunTurnRate,
                  double radarTurnRate, Double firepower, int action) {}
    record Shot(int turn, int bulletId, double power, double direction, double x, double y) {}

    private static final Gson JSON = new GsonBuilder().serializeNulls().create();
    private static final LegalScanRecorder OFF = new LegalScanRecorder();
    private final Path path;
    private final String gameId;
    private final int maxRows;
    private final StringBuilder buffer = new StringBuilder();
    private final List<Shot> pendingShots = new ArrayList<>();
    private final Map<Integer, Shot> seenShots = new HashMap<>();
    private final JsonArray rounds = new JsonArray();
    private Intent previousIntent;
    private int round, rows, lastTurn;
    private boolean failed, finished, gameFinished, failureFlushed;
    private String error;

    private LegalScanRecorder() { path = null; gameId = null; maxRows = 0; }
    LegalScanRecorder(Path path, String gameId, int maxRows) throws IOException {
        if (gameId == null || !gameId.matches("[A-Za-z0-9][A-Za-z0-9._-]{0,127}") || maxRows < 1)
            throw new IllegalArgumentException("Invalid dataset game ID or row bound");
        this.path = path.toAbsolutePath(); this.gameId = gameId; this.maxRows = maxRows;
        Files.createDirectories(this.path.getParent());
        if (Files.exists(Path.of(this.path + ".status.json")))
            throw new IOException("Dataset status already exists");
        Files.createFile(this.path); // Refuse stale/mixed captures; never overwrite an existing dataset.
        status();
    }
    static LegalScanRecorder fromEnvironment(Map<String, String> env) {
        String output = env.get("AISURVIVOR_DATASET_PATH");
        if (output == null) return OFF;
        if (output.isBlank()) throw new IllegalArgumentException("Dataset path must not be blank");
        try { return new LegalScanRecorder(Path.of(output), env.get("AISURVIVOR_GAME_ID"), 20_000); }
        catch (IOException error) { throw new IllegalArgumentException("Cannot create fresh dataset capture", error); }
    }
    boolean enabled() { return path != null && !failed; }
    boolean configured() { return path != null; }

    synchronized void startRound(int number) {
        if (!enabled()) return;
        if (number <= round) { fail("Invalid or repeated capture round"); return; }
        if (round != 0 && !finished) { fail("Round started before preceding capture was completed"); return; }
        round = number; rows = lastTurn = 0; finished = false; previousIntent = null;
        buffer.setLength(0); pendingShots.clear(); seenShots.clear();
    }
    synchronized void submitted(Intent intent) {
        if (enabled()) previousIntent = intent;
    }
    synchronized void confirmed(Shot shot) {
        if (!enabled()) return;
        Shot old = seenShots.putIfAbsent(shot.bulletId, shot);
        if (old != null && !old.equals(shot)) { fail("Conflicting confirmed own bullet identity"); return; }
        if (old == null) pendingShots.add(shot);
    }
    synchronized void frame(int number, int turn, Own own, List<Target> targets) {
        if (!enabled() || targets.isEmpty()) return;
        try {
            if (number != round || finished || turn <= lastTurn || turn < 1)
                throw new IllegalArgumentException("Invalid or repeated capture turn");
            if (previousIntent != null && previousIntent.submittedTurn >= turn)
                throw new IllegalArgumentException("Current/future intent cannot enter a pre-decision snapshot");
            if (pendingShots.stream().anyMatch(shot -> shot.turn > turn))
                throw new IllegalArgumentException("Future own-shot event");
            if (rows + targets.size() > maxRows) throw new IllegalArgumentException("Round capture row bound exceeded");
            JsonElement ownJson = JSON.toJsonTree(own), intentJson = JSON.toJsonTree(previousIntent);
            JsonElement shotsJson = JSON.toJsonTree(pendingShots);
            Set<Integer> identities = new HashSet<>();
            StringBuilder tick = new StringBuilder();
            for (Target target : targets) {
                if (target.targetId < 1 || !identities.add(target.targetId)
                        || (target.scan != null && target.scan.observedTurn > turn))
                    throw new IllegalArgumentException("Invalid target or future scan");
                JsonObject row = new JsonObject();
                row.addProperty("schemaVersion", 1); row.addProperty("gameId", gameId);
                row.addProperty("round", number); row.addProperty("turn", turn); row.addProperty("targetId", target.targetId);
                row.add("own", ownJson); row.add("scan", JSON.toJsonTree(target.scan));
                row.add("previousIntent", intentJson); row.add("confirmedShots", shotsJson);
                tick.append(JSON.toJson(row)).append('\n');
            }
            if (buffer.length() + tick.length() > 8 * 1024 * 1024)
                throw new IllegalArgumentException("Round capture byte bound exceeded");
            buffer.append(tick); rows += targets.size(); lastTurn = turn; pendingShots.clear();
        } catch (RuntimeException error) { fail(error.getMessage()); }
    }
    synchronized void finishRound(int number) {
        if (!configured()) return;
        if (failed) { flushFailure(); return; }
        if (finished) return;
        if (number != round) { fail("Mismatched capture round end"); flushFailure(); return; }
        try {
            append();
            JsonObject result = new JsonObject(); result.addProperty("round", round);
            result.addProperty("records", rows); result.addProperty("complete", true); rounds.add(result);
            finished = true; status();
        } catch (IOException error) { fail("Round flush failed: " + error.getMessage()); flushFailure(); }
    }
    synchronized void finishGame() {
        if (!configured()) return;
        if (failed) { flushFailure(); return; }
        if (round != 0 && !finished) { fail("Game ended before capture round completion"); flushFailure(); return; }
        gameFinished = true;
        try { status(); } catch (IOException error) { fail("Game status flush failed: " + error.getMessage()); flushFailure(); }
    }
    private void append() throws IOException {
        if (buffer.length() != 0) {
            Files.writeString(path, buffer, StandardOpenOption.APPEND);
            buffer.setLength(0);
        }
    }
    private void status() throws IOException {
        JsonObject result = new JsonObject(); result.addProperty("schemaVersion", 1);
        result.addProperty("collectorVersion", "legal-scan-v1"); result.addProperty("gameId", gameId);
        result.addProperty("status", failed ? "failed" : gameFinished ? "complete" : "recording");
        result.add("rounds", rounds.deepCopy()); result.addProperty("error", error);
        Files.writeString(Path.of(path + ".status.json"), JSON.toJson(result) + "\n");
    }
    private void fail(String message) {
        failed = true; error = message;
    }
    private void flushFailure() {
        if (failureFlushed) return;
        failureFlushed = true;
        try { append(); status(); }
        catch (IOException secondary) { System.err.println("[AISurvivor] Dataset evidence flush failed: " + secondary.getMessage()); }
        System.err.println("[AISurvivor] Dataset capture invalid: " + error);
    }
}
