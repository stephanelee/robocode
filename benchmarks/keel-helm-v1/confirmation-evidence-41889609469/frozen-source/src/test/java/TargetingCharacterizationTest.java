import com.google.gson.*;
import org.junit.jupiter.api.Test;
import java.lang.reflect.*;
import java.nio.file.*;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.Mockito.*;

/** Actual accepted aimGF command/state recordings, frozen before extraction. */
class TargetingCharacterizationTest {
    static final Path GOLDEN = Path.of("benchmarks/targeting-refactor/accepted-goldens.json");
    static final String ACCEPTED = "58d519c5e89399381760b1e7ed3fab166f60a85d";
    record Scenario(String name, double sx, double sy, double tx, double ty,
                    double heading, double speed, double turnRate, double lateral, int age,
                    double gun, double energy, double heat, int scans, int bin,
                    boolean stats, boolean accepted, boolean pending, boolean target) {}

    static List<Scenario> scenarios() {
        var rows = new ArrayList<Scenario>();
        for (double range : new double[]{50, 100, 199.999, 200, 499.999, 500}) {
            for (double gun : new double[]{9, 10, 350, 351})
                rows.add(new Scenario("range-" + range + "-gun-" + gun, 200, 300, 200 + range, 300,
                        0, 0, 0, 0, 0, gun, 100, 0, 0, 23, true, true, false, true));
        }
        rows.add(new Scenario("known-wall-projection", 400, 300, 770, 300, 45, 8, 0, 8, 0,
                0, 100, 0, 40, 23, true, true, false, true));
        rows.add(new Scenario("positive-180", 400, 300, 100, 300, 180, 0, 0, 0, 0,
                0, 100, 0, 10, 23, true, true, false, true));
        rows.add(new Scenario("negative-180", 400, 300, 700, 300, 0, 0, 0, 0, 0,
                180, 100, 0, 10, 23, true, true, false, true));
        for (String name : List.of("no-target", "null-stats", "rejected-fire", "hot-gun", "low-energy"))
            rows.add(new Scenario(name, 400, 300, 700, 300, 0, 0, 0, 0, 0, 0,
                    name.equals("low-energy") ? 11.99 : 100, name.equals("hot-gun") ? 0.1 : 0,
                    40, 23, !name.equals("null-stats"), !name.equals("rejected-fire"), true, !name.equals("no-target")));
        Random random = new Random(13371337);
        for (int i = 0; i < 256; i++)
            rows.add(new Scenario("wide-" + i, 18 + random.nextDouble() * 764, 18 + random.nextDouble() * 564,
                    18 + random.nextDouble() * 764, 18 + random.nextDouble() * 564,
                    random.nextDouble() * 360, random.nextDouble() * 16 - 8,
                    random.nextDouble() * 20 - 10, random.nextDouble() * 16 - 8, random.nextInt(41),
                    random.nextDouble() * 360, random.nextDouble() * 100, random.nextBoolean() ? 0 : 0.1,
                    random.nextInt(61), random.nextInt(47), random.nextBoolean(), random.nextBoolean(), random.nextBoolean(), true));
        return rows;
    }

    static void set(Object object, String name, Object value) throws Exception {
        Field field = object instanceof SurvivorBrain ? SurvivorBrain.class.getDeclaredField(name)
                : object.getClass().getDeclaredField(name);
        field.setAccessible(true);
        field.set(object, value);
    }
    static Object get(Object object, String name) throws Exception {
        Field field = (object instanceof SurvivorBrain ? SurvivorBrain.class : object.getClass()).getDeclaredField(name);
        field.setAccessible(true);
        return field.get(object);
    }
    static Object wave() throws Exception {
        Constructor<?> constructor = Class.forName("SurvivorBrain$TargetingWave").getDeclaredConstructor();
        constructor.setAccessible(true);
        return constructor.newInstance();
    }
    static String bits(double number) { return Long.toHexString(Double.doubleToRawLongBits(number)); }

    static JsonObject record(Scenario s) throws Exception {
        return record(s, null);
    }
    static JsonObject record(Scenario s, LegalScanRecorder recorder) throws Exception {
        SurvivorBrain bot = mock(SurvivorBrain.class, CALLS_REAL_METHODS); BotPort port = mock(BotPort.class); TargetingCharacterizationTest.set(bot, "port", port);
        doReturn(s.sx).when(port).x(); doReturn(s.sy).when(port).y();
        doReturn(s.energy).when(port).energy(); doReturn(s.heat).when(port).gunHeat();
        doReturn(s.gun).when(port).gunDirection(); doReturn(1000).when(port).turnNumber();
        doReturn(800).when(port).arenaWidth(); doReturn(600).when(port).arenaHeight();
        doReturn(1).when(port).enemyCount();
        JsonArray guns = new JsonArray(), fires = new JsonArray();
        doAnswer(call -> { guns.add(bits(call.getArgument(0))); return null; }).when(port).commandGunTurnRate(anyDouble());
        doAnswer(call -> { fires.add(bits(call.getArgument(0))); return s.accepted; }).when(port).commandFire(anyDouble());
        var config = new BotConfig(BotConfig.Mode.EVAL, 539, Path.of("unused-characterization-model"), 0, null);
        set(bot, "config", config);
        DQNAgent agent = new DQNAgent(14, 6, config), control = new DQNAgent(14, 6, config);
        for (double[][] category : agent.classGF) for (double[] bins : category) bins[s.bin] = 4;
        set(bot, "agent", agent);
        Class<?> enemyClass = Class.forName("SurvivorBrain$Enemy");
        Constructor<?> constructor = enemyClass.getDeclaredConstructors()[0];
        constructor.setAccessible(true);
        Object enemy = constructor.newInstance(7, s.tx, s.ty, s.heading, s.speed, 100.0, s.lateral, 100.0, s.turnRate, 1000 - s.age);
        set(enemy, "scanCount", s.scans);
        Map<Integer, Object> enemies = new HashMap<>();
        if (s.target) enemies.put(7, enemy);
        set(bot, "enemies", enemies);
        double[][] stats = new double[9][47];
        for (double[] bins : stats) bins[s.bin] = 4;
        set(bot, "gfStats", s.stats ? new HashMap<>(Map.of(7, stats)) : new HashMap<>());
        Object previous = s.pending ? wave() : null;
        set(bot, "pendingTargetingWave", previous);
        int statsHash = Arrays.deepHashCode(stats), classHash = Arrays.deepHashCode(agent.classGF);
        if (recorder != null) {
            set(bot, "scanRecorder", recorder);
            doReturn(1).when(port).roundNumber();
            doReturn(0.0).when(port).direction(); doReturn(0.0).when(port).speed();
            doReturn(0.0).when(port).radarDirection();
            Method capture = SurvivorBrain.class.getDeclaredMethod("recordLegalFrame");
            capture.setAccessible(true); capture.invoke(bot);
        }
        Method aim = SurvivorBrain.class.getDeclaredMethod("aimGF"); aim.setAccessible(true); aim.invoke(bot);
        assertEquals(statsHash, Arrays.deepHashCode(stats)); assertEquals(classHash, Arrays.deepHashCode(agent.classGF));
        assertEquals(control.rng.nextLong(), agent.rng.nextLong(), "aiming must not consume bot RNG");
        Object pending = get(bot, "pendingTargetingWave");
        JsonObject result = new JsonObject(); result.addProperty("name", s.name); result.add("gun", guns); result.add("fire", fires);
        result.addProperty("waveChanged", pending != previous); result.addProperty("wavePresent", pending != null);
        if (pending != null) {
            JsonObject values = new JsonObject();
            for (String name : List.of("originX", "originY", "baseAngle", "bulletSpeed", "lateralDir", "aimX", "aimY"))
                values.addProperty(name, bits((double) get(pending, name)));
            for (String name : List.of("fireTurn", "targetId", "segment")) values.addProperty(name, (int) get(pending, name));
            values.addProperty("binsAlias", get(pending, "bins") == stats[(int) get(pending, "segment")]);
            result.add("wave", values);
        }
        return result;
    }

    @Test void actualAimCommandsAndPendingWaveStateMatchAcceptedGoldens() throws Exception {
        JsonArray actual = new JsonArray();
        for (Scenario scenario : scenarios()) actual.add(record(scenario));
        if ("1".equals(System.getenv("CAPTURE_TARGETING_GOLDENS"))) {
            assertFalse(Files.exists(GOLDEN), "Never overwrite an existing accepted recording");
            Files.createDirectories(GOLDEN.getParent());
            JsonObject captured = new JsonObject(); captured.addProperty("acceptedSource", ACCEPTED);
            captured.addProperty("botApi", "1.4.0"); captured.addProperty("wideFixtureSeed", 13371337);
            captured.add("outcomes", actual);
            Files.writeString(GOLDEN, new GsonBuilder().setPrettyPrinting().create().toJson(captured) + "\n");
        }
        JsonObject golden = JsonParser.parseString(Files.readString(GOLDEN)).getAsJsonObject();
        assertEquals(ACCEPTED, golden.get("acceptedSource").getAsString());
        assertEquals(golden.getAsJsonArray("outcomes"), actual);
    }
}
