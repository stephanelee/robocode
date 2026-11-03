import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import dev.robocode.tankroyale.botapi.BotInfo;
import dev.robocode.tankroyale.botapi.BulletState;
import dev.robocode.tankroyale.botapi.events.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.*;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.GZIPInputStream;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.Mockito.*;

/**
 * Decision parity between master's pre-extraction bot and the extracted {@link SurvivorBrain}: drives an unmodified
 * copy of master e4e4c94's {@code AISurvivor} ({@link MasterAISurvivor}) through the golden traces recorded by the
 * extracted brain, with every final Bot API getter stubbed from the recorded state, and requires every intent call to
 * match bit for bit.
 */
class MasterParityTest {
    /** Golden traces that only exercise behaviour shared by master and the extracted brain. */
    static final String[] PRE_CHANGE_GOLDENS = {
            "benchmarks/replay-goldens/retcon-1v1-3r.jsonl.gz",
            "benchmarks/replay-goldens/walls-1v1-3r.jsonl.gz",
            "benchmarks/replay-goldens/superseded/melee8-3r-before-coherent-melee.jsonl.gz"};

    static final class State {
        int turn, round, enemies, width, height;
        double x, y, direction, speed, energy, gunHeat, gunDirection, radarDirection;
        boolean running;
        JsonArray calls = new JsonArray();

        void load(JsonObject o) {
            turn = o.get("turn").getAsInt(); round = o.get("round").getAsInt(); enemies = o.get("enemies").getAsInt();
            width = o.get("w").getAsInt(); height = o.get("h").getAsInt();
            x = d(o, "x"); y = d(o, "y"); direction = d(o, "dir"); speed = d(o, "speed"); energy = d(o, "energy");
            gunHeat = d(o, "gunHeat"); gunDirection = d(o, "gunDir"); radarDirection = d(o, "radarDir");
        }

        void call(String name, double value) { JsonArray c = new JsonArray(); c.add(name); c.add(value); calls.add(c); }
        void flag(String name, boolean value) { JsonArray c = new JsonArray(); c.add(name); c.add(value); calls.add(c); }
    }

    static double d(JsonObject o, String key) { return o.get(key).getAsDouble(); }

    /** Command batches compared by the most recent {@link #replay} call. */
    static int batches;

    static MasterAISurvivor master(State s, JsonObject header, Path model) throws Exception {
        // stubOnly: do not record millions of invocations (a recording spy runs out of memory on the melee trace).
        MasterAISurvivor real = new MasterAISurvivor(BotInfo.fromFile("AISurvivor/AISurvivor.json"));
        MasterAISurvivor bot = mock(MasterAISurvivor.class, withSettings().spiedInstance(real)
                .defaultAnswer(CALLS_REAL_METHODS).stubOnly());
        BotConfig config = new BotConfig(BotConfig.Mode.valueOf(header.get("mode").getAsString()),
                header.get("seed").getAsLong(), model, header.get("fixedAction").getAsInt(), null,
                BotConfig.DuelPolicy.valueOf(header.get("duelPolicy").getAsString()));
        Field field = MasterAISurvivor.class.getDeclaredField("config");
        field.setAccessible(true);
        field.set(bot, config);
        Method load = MasterAISurvivor.class.getDeclaredMethod("loadAgent");
        load.setAccessible(true);
        load.invoke(bot);
        doAnswer(i -> s.turn).when(bot).getTurnNumber();
        doAnswer(i -> s.round).when(bot).getRoundNumber();
        doAnswer(i -> s.enemies).when(bot).getEnemyCount();
        doAnswer(i -> s.width).when(bot).getArenaWidth();
        doAnswer(i -> s.height).when(bot).getArenaHeight();
        doAnswer(i -> s.x).when(bot).getX();
        doAnswer(i -> s.y).when(bot).getY();
        doAnswer(i -> s.direction).when(bot).getDirection();
        doAnswer(i -> s.speed).when(bot).getSpeed();
        doAnswer(i -> s.energy).when(bot).getEnergy();
        doAnswer(i -> s.gunHeat).when(bot).getGunHeat();
        doAnswer(i -> s.gunDirection).when(bot).getGunDirection();
        doAnswer(i -> s.radarDirection).when(bot).getRadarDirection();
        doAnswer(i -> { s.call("turnRate", i.getArgument(0)); return null; }).when(bot).setTurnRate(anyDouble());
        doAnswer(i -> { s.call("targetSpeed", i.getArgument(0)); return null; }).when(bot).setTargetSpeed(anyDouble());
        doAnswer(i -> { s.call("gunTurnRate", i.getArgument(0)); return null; }).when(bot).setGunTurnRate(anyDouble());
        doAnswer(i -> { s.call("radarTurnRate", i.getArgument(0)); return null; }).when(bot).setRadarTurnRate(anyDouble());
        doAnswer(i -> {
            double power = i.getArgument(0);
            boolean accepted = !(s.energy < power || s.gunHeat > 0);
            JsonArray c = new JsonArray(); c.add("fire"); c.add(power); c.add(accepted); s.calls.add(c);
            return accepted;
        }).when(bot).setFire(anyDouble());
        doAnswer(i -> { s.flag("adjustGunForBodyTurn", i.getArgument(0)); return null; }).when(bot).setAdjustGunForBodyTurn(anyBoolean());
        doAnswer(i -> { s.flag("adjustRadarForGunTurn", i.getArgument(0)); return null; }).when(bot).setAdjustRadarForGunTurn(anyBoolean());
        doAnswer(i -> { s.flag("adjustRadarForBodyTurn", i.getArgument(0)); return null; }).when(bot).setAdjustRadarForBodyTurn(anyBoolean());
        doAnswer(i -> s.running).when(bot).isRunning();
        doAnswer(i -> { s.running = false; return null; }).when(bot).go();
        return bot;
    }

    /** Returns null when every batch matches, otherwise the first mismatch. */
    static String replay(Path trace, Path model) throws Exception {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new GZIPInputStream(Files.newInputStream(trace)), StandardCharsets.UTF_8))) {
            JsonObject header = JsonParser.parseString(reader.readLine()).getAsJsonObject();
            State s = new State();
            MasterAISurvivor bot = master(s, header, model);
            String text;
            int line = 1;
            batches = 0;
            while ((text = reader.readLine()) != null) {
                line++;
                JsonObject o = JsonParser.parseString(text).getAsJsonObject();
                switch (o.get("k").getAsString()) {
                    case "tick" -> s.load(o);
                    case "ev" -> dispatch(bot, o);
                    case "begin" -> {
                        s.running = false;
                        bot.run(); // master's run(): two adjustments, then the loop exits immediately
                        batches++;
                        String mismatch = compare(o, s, line, 0);
                        if (mismatch != null) return mismatch;
                    }
                    case "cmd" -> {
                        s.running = true;
                        bot.run(); // one loop iteration; master's run() repeats its two start adjustments first
                        batches++;
                        String mismatch = compare(o, s, line, 2);
                        if (mismatch != null) return mismatch;
                    }
                    default -> throw new IOException("Unknown record " + text);
                }
            }
            return null;
        }
    }

    static String compare(JsonObject o, State s, int line, int skipPrefix) {
        JsonArray actual = new JsonArray();
        for (int i = skipPrefix; i < s.calls.size(); i++) actual.add(s.calls.get(i));
        s.calls = new JsonArray();
        JsonArray expected = o.getAsJsonArray("calls");
        return expected.equals(actual) ? null
                : "line " + line + " turn " + s.turn + " round " + s.round + ": expected " + expected + " but got " + actual;
    }

    static void dispatch(MasterAISurvivor bot, JsonObject o) {
        int turn = o.has("turn") ? o.get("turn").getAsInt() : 0;
        switch (o.get("type").getAsString()) {
            case "RoundStartedEvent" -> bot.onRoundStarted(new RoundStartedEvent(o.get("round").getAsInt()));
            case "RoundEndedEvent" -> bot.onRoundEnded(new RoundEndedEvent(o.get("round").getAsInt(), turn, null));
            case "GameEndedEvent" -> bot.onGameEnded(new GameEndedEvent(o.get("rounds").getAsInt(), null));
            case "ScannedBotEvent" -> bot.onScannedBot(new ScannedBotEvent(turn, o.get("scannedBy").getAsInt(),
                    o.get("id").getAsInt(), d(o, "energy"), d(o, "x"), d(o, "y"), d(o, "dir"), d(o, "speed")));
            case "HitByBulletEvent" -> bot.onHitByBullet(new HitByBulletEvent(turn, bullet(o), d(o, "damage"), d(o, "energy")));
            case "BulletHitBotEvent" -> bot.onBulletHit(new BulletHitBotEvent(turn, o.get("victim").getAsInt(), bullet(o),
                    d(o, "damage"), d(o, "energy")));
            case "BulletFiredEvent" -> bot.onBulletFired(new BulletFiredEvent(turn, bullet(o)));
            case "BotDeathEvent" -> bot.onBotDeath(new BotDeathEvent(turn, o.get("victim").getAsInt()));
            case "DeathEvent" -> bot.onDeath(new DeathEvent(turn));
            case "HitWallEvent" -> bot.onHitWall(new HitWallEvent(turn));
            case "HitBotEvent" -> bot.onHitBot(new HitBotEvent(turn, o.get("victim").getAsInt(), d(o, "energy"),
                    d(o, "x"), d(o, "y"), o.get("rammed").getAsBoolean()));
            case "WonRoundEvent" -> bot.onWonRound(new WonRoundEvent(turn));
            case "SkippedTurnEvent" -> bot.onSkippedTurn(new SkippedTurnEvent(turn));
            default -> throw new IllegalArgumentException("Unknown event " + o);
        }
    }

    static BulletState bullet(JsonObject o) {
        JsonObject b = o.getAsJsonObject("bullet");
        return new BulletState(b.get("id").getAsInt(), b.get("owner").getAsInt(), d(b, "power"), d(b, "x"), d(b, "y"),
                d(b, "dir"), null);
    }

    @Test void masterMakesIdenticalDecisionsOnEveryGoldenTrace(@TempDir Path temp) throws Exception {
        for (String golden : PRE_CHANGE_GOLDENS) {
            String mismatch = replay(Path.of(golden), temp.resolve(Path.of(golden).getFileName() + ".dat"));
            assertNull(mismatch, golden + ": " + mismatch);
            assertTrue(batches > 900, golden + " compared only " + batches + " batches");
            System.out.println("[MasterParityTest] identical " + golden + " batches=" + batches);
        }
    }

    @Test void masterDetectsTheIntendedCoherentMeleeChange(@TempDir Path temp) throws Exception {
        // A trace recorded by the coherent-melee candidate must NOT replay on master: the harness can fail.
        Path candidate = Path.of("benchmarks/replay-goldens/melee8-3r.jsonl.gz");
        Path superseded = Path.of("benchmarks/replay-goldens/superseded/melee8-3r-before-coherent-melee.jsonl.gz");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(superseded), "only on branches with the melee change");
        String mismatch = replay(candidate, temp.resolve("c.dat"));
        assertNotNull(mismatch);
        assertTrue(mismatch.contains(" round 1:"), "the melee change must show in the first round: " + mismatch);
        assertNull(replay(superseded, temp.resolve("s.dat")), "the pre-change melee trace still matches master");
    }
}
