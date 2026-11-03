import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import dev.robocode.tankroyale.botapi.BulletState;
import dev.robocode.tankroyale.botapi.events.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPInputStream;

/**
 * Replays a {@link GameTrace} through a fresh {@link SurvivorBrain} and requires every recorded intent call, in order
 * and with bit-identical arguments, to be reproduced. A refactor or rewrite that claims behavioral parity must replay
 * the committed golden traces cleanly. Usage: {@code java -cp AISurvivor.jar TraceReplay trace.jsonl.gz...}.
 */
final class TraceReplay {
    private TraceReplay() {}

    /**
     * A replay outcome: command batches compared, rounds completed, and the first mismatch or completeness failure
     * (null only when every batch matched bit for bit and the trace holds at least one complete game).
     */
    record Result(int batches, int rounds, String mismatch) {
        boolean identical() { return mismatch == null; }
    }

    /**
     * Bit-exact comparison of two intent-call lists: same calls in the same order; numeric arguments equal under
     * {@link Double#doubleToLongBits} (so -0.0 differs from 0.0); strings and booleans equal.
     */
    static boolean sameCalls(JsonArray expected, JsonArray actual) {
        if (expected.size() != actual.size()) return false;
        for (int i = 0; i < expected.size(); i++) {
            JsonArray e = expected.get(i).getAsJsonArray(), a = actual.get(i).getAsJsonArray();
            if (e.size() != a.size()) return false;
            for (int j = 0; j < e.size(); j++) {
                var ep = e.get(j).getAsJsonPrimitive();
                var ap = a.get(j).getAsJsonPrimitive();
                if (ep.isNumber() != ap.isNumber() || ep.isBoolean() != ap.isBoolean() || ep.isString() != ap.isString())
                    return false;
                if (ep.isNumber() ? Double.doubleToLongBits(ep.getAsDouble()) != Double.doubleToLongBits(ap.getAsDouble())
                        : !ep.equals(ap)) return false;
            }
        }
        return true;
    }

    static Result replay(Path trace, Path modelPath) throws IOException {
        try (BufferedReader reader = open(trace)) {
            JsonObject header = JsonParser.parseString(reader.readLine()).getAsJsonObject();
            if (!"header".equals(header.get("k").getAsString()) || header.get("version").getAsInt() != GameTrace.VERSION)
                throw new IOException("Unsupported trace header: " + header);
            BotConfig config = new BotConfig(BotConfig.Mode.valueOf(header.get("mode").getAsString()),
                    header.get("seed").getAsLong(), modelPath, header.get("fixedAction").getAsInt(), null,
                    BotConfig.DuelPolicy.valueOf(header.get("duelPolicy").getAsString()));
            ReplayPort port = new ReplayPort();
            String movement=header.has("movementPolicy")?header.get("movementPolicy").getAsString():"accepted-duelsurf";
            if(!movement.equals("accepted-duelsurf")&&!movement.equals("breakwater-v1"))
                throw new IOException("Unsupported movement policy: "+movement);
            SurvivorBrain brain = new SurvivorBrain(port, config, LegalScanRecorder.fromEnvironment(Map.of()),movement.equals("breakwater-v1"));
            int batches = 0, line = 1, started = 0, ended = 0;
            boolean gameEnded = false;
            String text;
            while ((text = reader.readLine()) != null) {
                line++;
                JsonObject o = JsonParser.parseString(text).getAsJsonObject();
                switch (o.get("k").getAsString()) {
                    case "tick" -> {
                        port.load(o);
                        brain.onTick(new TickEvent(port.turn, port.round, null, List.of(), List.of()));
                    }
                    case "ev" -> {
                        String type = o.get("type").getAsString();
                        if (type.equals("RoundStartedEvent")) started++;
                        if (type.equals("RoundEndedEvent")) ended++;
                        if (type.equals("GameEndedEvent")) gameEnded = true;
                        dispatch(brain, o);
                    }
                    case "begin", "cmd" -> {
                        if (o.get("k").getAsString().equals("begin")) brain.beginRun();
                        else brain.turn();
                        batches++;
                        JsonArray expected = o.getAsJsonArray("calls");
                        if (!sameCalls(expected, port.calls))
                            return new Result(batches, ended, "line " + line + " turn " + port.turn + " round "
                                    + port.round + ": expected " + expected + " but got " + port.calls);
                        port.calls = new JsonArray();
                    }
                    default -> throw new IOException("Unknown record at line " + line + ": " + text);
                }
            }
            String incomplete = completeness(batches, started, ended, gameEnded);
            return new Result(batches, ended, incomplete);
        }
    }

    /** A successful replay must cover at least one whole game: every started round ended, then the game ended. */
    static String completeness(int batches, int started, int ended, boolean gameEnded) {
        if (batches == 0) return "incomplete trace: no command batches";
        if (started == 0 || started != ended) return "incomplete trace: " + started + " rounds started, " + ended + " ended";
        if (!gameEnded) return "incomplete trace: no GameEndedEvent";
        return null;
    }

    private static void dispatch(SurvivorBrain brain, JsonObject o) {
        int turn = o.has("turn") ? o.get("turn").getAsInt() : 0;
        switch (o.get("type").getAsString()) {
            case "RoundStartedEvent" -> brain.onRoundStarted(new RoundStartedEvent(o.get("round").getAsInt()));
            case "RoundEndedEvent" -> brain.onRoundEnded(new RoundEndedEvent(o.get("round").getAsInt(), turn, null));
            case "GameEndedEvent" -> brain.onGameEnded(new GameEndedEvent(o.get("rounds").getAsInt(), null));
            case "ScannedBotEvent" -> brain.onScannedBot(new ScannedBotEvent(turn, o.get("scannedBy").getAsInt(),
                    o.get("id").getAsInt(), d(o, "energy"), d(o, "x"), d(o, "y"), d(o, "dir"), d(o, "speed")));
            case "HitByBulletEvent" -> brain.onHitByBullet(new HitByBulletEvent(turn, bullet(o), d(o, "damage"), d(o, "energy")));
            case "BulletHitBotEvent" -> brain.onBulletHit(new BulletHitBotEvent(turn, o.get("victim").getAsInt(), bullet(o),
                    d(o, "damage"), d(o, "energy")));
            case "BulletFiredEvent" -> brain.onBulletFired(new BulletFiredEvent(turn, bullet(o)));
            case "BotDeathEvent" -> brain.onBotDeath(new BotDeathEvent(turn, o.get("victim").getAsInt()));
            case "DeathEvent" -> brain.onDeath(new DeathEvent(turn));
            case "HitWallEvent" -> brain.onHitWall(new HitWallEvent(turn));
            case "HitBotEvent" -> brain.onHitBot(new HitBotEvent(turn, o.get("victim").getAsInt(), d(o, "energy"),
                    d(o, "x"), d(o, "y"), o.get("rammed").getAsBoolean()));
            case "WonRoundEvent" -> brain.onWonRound(new WonRoundEvent(turn));
            case "SkippedTurnEvent" -> brain.onSkippedTurn(new SkippedTurnEvent(turn));
            default -> throw new IllegalArgumentException("Unknown event " + o);
        }
    }

    private static BulletState bullet(JsonObject o) {
        JsonObject b = o.getAsJsonObject("bullet");
        return new BulletState(b.get("id").getAsInt(), b.get("owner").getAsInt(), d(b, "power"), d(b, "x"), d(b, "y"),
                d(b, "dir"), null);
    }

    private static double d(JsonObject o, String key) { return o.get(key).getAsDouble(); }

    private static BufferedReader open(Path path) throws IOException {
        InputStream in = Files.newInputStream(path);
        if (path.toString().endsWith(".gz")) in = new GZIPInputStream(in);
        return new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
    }

    /** Port backed by the recorded turn state; records intent calls exactly as {@link AISurvivor} traces them. */
    static final class ReplayPort implements BotPort {
        int turn, round, enemies, width, height;
        double x, y, direction, speed, energy, gunHeat, gunDirection, radarDirection;
        double turnRate, targetSpeed, gunTurnRate, radarTurnRate;
        JsonArray calls = new JsonArray();

        void load(JsonObject o) {
            turn = o.get("turn").getAsInt(); round = o.get("round").getAsInt(); enemies = o.get("enemies").getAsInt();
            width = o.get("w").getAsInt(); height = o.get("h").getAsInt();
            x = d(o, "x"); y = d(o, "y"); direction = d(o, "dir"); speed = d(o, "speed"); energy = d(o, "energy");
            gunHeat = d(o, "gunHeat"); gunDirection = d(o, "gunDir"); radarDirection = d(o, "radarDir");
            turnRate = d(o, "turnRate"); targetSpeed = d(o, "targetSpeed");
            gunTurnRate = d(o, "gunTurnRate"); radarTurnRate = d(o, "radarTurnRate");
        }

        @Override public int turnNumber() { return turn; }
        @Override public int roundNumber() { return round; }
        @Override public int enemyCount() { return enemies; }
        @Override public int arenaWidth() { return width; }
        @Override public int arenaHeight() { return height; }
        @Override public double x() { return x; }
        @Override public double y() { return y; }
        @Override public double direction() { return direction; }
        @Override public double speed() { return speed; }
        @Override public double energy() { return energy; }
        @Override public double gunHeat() { return gunHeat; }
        @Override public double gunDirection() { return gunDirection; }
        @Override public double radarDirection() { return radarDirection; }
        @Override public double turnRate() { return turnRate; }
        @Override public double targetSpeed() { return targetSpeed; }
        @Override public double gunTurnRate() { return gunTurnRate; }
        @Override public double radarTurnRate() { return radarTurnRate; }

        @Override public void commandTurnRate(double rate) { call("turnRate", rate); }
        @Override public void commandTargetSpeed(double value) { call("targetSpeed", value); }
        @Override public void commandGunTurnRate(double rate) { call("gunTurnRate", rate); }
        @Override public void commandRadarTurnRate(double rate) { call("radarTurnRate", rate); }
        /** BaseBotInternals.setFire: NaN is rejected; firing needs energy at least the power and a cold gun. */
        @Override public boolean commandFire(double power) {
            if (Double.isNaN(power)) throw new IllegalArgumentException("'firepower' cannot be NaN");
            boolean accepted = !(energy < power || gunHeat > 0);
            JsonArray c = new JsonArray();
            c.add("fire"); c.add(power); c.add(accepted);
            calls.add(c);
            return accepted;
        }
        @Override public void adjustGunForBodyTurn(boolean adjust) { flag("adjustGunForBodyTurn", adjust); }
        @Override public void adjustRadarForGunTurn(boolean adjust) { flag("adjustRadarForGunTurn", adjust); }
        @Override public void adjustRadarForBodyTurn(boolean adjust) { flag("adjustRadarForBodyTurn", adjust); }

        private void call(String name, double value) {
            JsonArray c = new JsonArray();
            c.add(name); c.add(value);
            calls.add(c);
        }

        private void flag(String name, boolean value) {
            JsonArray c = new JsonArray();
            c.add(name); c.add(value);
            calls.add(c);
        }
    }

    public static void main(String[] args) throws IOException {
        if (args.length == 0) { System.err.println("usage: TraceReplay trace.jsonl[.gz]..."); System.exit(2); }
        boolean ok = true;
        for (String arg : args) {
            Path model = Files.createTempFile("aisurvivor-replay", ".dat");
            Files.delete(model);
            Result result = replay(Path.of(arg), model);
            Files.deleteIfExists(model);
            System.out.println((result.identical() ? "IDENTICAL " : "MISMATCH ") + arg + " batches=" + result.batches()
                    + " rounds=" + result.rounds()
                    + (result.identical() ? "" : " " + result.mismatch()));
            ok &= result.identical();
        }
        System.exit(ok ? 0 : 1);
    }
}
