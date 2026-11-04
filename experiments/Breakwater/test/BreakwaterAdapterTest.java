import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.*;
import static org.junit.jupiter.api.Assertions.*;

final class BreakwaterAdapterTest {
    @Test void consecutiveScansCorrectDamageAndBonusesThenRejectHotDuplicates() {
        var w = new BreakwaterWaves(450);
        var enemy = new BreakwaterPolicy.Target(29, 180, 200, 90, 0, 100);
        assertFalse(w.scan(2, enemy, false, 800, 600, .1));
        w.own(29, 600, 200, 90, 8);
        w.damage(2, 29, 4);
        w.bonus(2, 29, .1);
        assertTrue(w.scan(2, new BreakwaterPolicy.Target(30, 180, 200, 90, 0, 94.35), false, 800, 600, .1));
        assertEquals(1.95, w.active.get(0).power(), 1e-6);
        assertEquals(8, w.active.get(0).lateral(), 1e-9);
        w.own(30, 600, 200, 90, 8);
        assertFalse(w.scan(2, new BreakwaterPolicy.Target(31, 180, 200, 90, 0, 93.35), false, 800, 600, .1));
        assertNull(w.match(35, 2, 1.95, 180 + 6 * 14.15, 220, 0));
        assertNull(w.match(35, 5, 1.95, 180 + 6 * 14.15, 200, 0));
        assertNotNull(w.match(35, 2, 1.95, 180 + 6 * 14.15, 200, 0));
        assertTrue(w.active.isEmpty());
    }
    @Test void ambiguousShotsCannotProduceAnIncomingRay() {
        var w = new BreakwaterWaves(450);
        w.scan(2, new BreakwaterPolicy.Target(30, 100, 100, 90, 0, 100), false, 800, 600, .1);
        w.own(31, 300, 200, 90, 8);
        assertFalse(w.scan(2, new BreakwaterPolicy.Target(32, 100, 100, 90, 0, 98), false, 800, 600, .1));
        w.own(32, 300, 200, 90, 8);
        assertFalse(w.scan(2, new BreakwaterPolicy.Target(33, 100, 100, 90, 0, 96), true, 800, 600, .1));
        w.own(33, 300, 200, 90, 8);
        assertFalse(w.scan(2, new BreakwaterPolicy.Target(34, 20, 100, 90, 0, 94), false, 800, 600, .1));
    }
    @Test void inactivityDecayIsAmbiguousAndDoesNotConsumeGunHeat() {
        var w = new BreakwaterWaves(450);
        w.scan(2, new BreakwaterPolicy.Target(450, 100, 100, 0, 0, 95), false, 800, 600, .1);
        w.own(450, 600, 200, 90, 8);
        assertFalse(w.scan(2, new BreakwaterPolicy.Target(451, 100, 100, 0, 0, 94.9), false, 800, 600, .1));
        w.own(451, 600, 200, 90, 8);
        assertTrue(w.scan(2, new BreakwaterPolicy.Target(452, 100, 100, 0, 0, 93), false, 800, 600, .1));
    }
    @Test void releasedJarReplayRejectsLateCallbacks(@TempDir Path dir) throws Exception {
        var jar = Path.of("build/bot/Breakwater/Breakwater.jar").toAbsolutePath();
        Files.copy(Path.of("bot/Breakwater.json"), dir.resolve("Breakwater.json"));
        var fixture = Path.of("fixtures/ReleasedLifecycle.java").toAbsolutePath();
        var pb = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "-cp", jar.toString(), fixture.toString()).directory(dir.toFile()).redirectErrorStream(true);
        pb.environment().put("BREAKWATER_MODEL_PATH", dir.resolve("endpoint").toString());
        pb.environment().put("BREAKWATER_METRICS_PATH", dir.resolve("rows").toString());
        var process = pb.start();
        String output = new String(process.getInputStream().readAllBytes());
        assertEquals(0, process.waitFor(), output);
        var model = JsonParser.parseString(Files.readString(dir.resolve("endpoint"))).getAsJsonObject();
        assertEquals("0.2.0", model.get("version").getAsString());
        assertEquals(2, model.get("schemaVersion").getAsInt());
        assertEquals(2, model.get("roundsCompleted").getAsInt());
        assertEquals(1, model.getAsJsonObject("totals").get("matchedRays").getAsInt());
        assertEquals(2 / Math.toDegrees(Math.asin(8 / 14.15)), model.getAsJsonArray("corrections").get(0).getAsDouble(), 1e-9);
        assertEquals(9.7, model.getAsJsonObject("totals").get("damageTaken").getAsDouble(), 1e-9);
        var rows = Files.readAllLines(dir.resolve("rows"));
        assertEquals(2, rows.size());
        var first = JsonParser.parseString(rows.get(0)).getAsJsonObject();
        var second = JsonParser.parseString(rows.get(1)).getAsJsonObject();
        assertEquals(1, first.get("officialRoundWins").getAsInt());
        assertFalse(first.get("won").getAsBoolean(), "The fixture omits WonRound intentionally");
        assertEquals(0, second.get("officialRoundWins").getAsInt());
        assertEquals(1, second.get("officialFirstPlaces").getAsInt());
        assertEquals(1, model.get("officialFirstPlaces").getAsInt());
        assertEquals(0, second.get("damageTaken").getAsInt());
    }
    @Test void sealedRoundIsIdempotentAndLatePacketsAreIneligible(@TempDir Path dir) throws Exception {
        Path metrics = dir.resolve("metrics"), model = dir.resolve("model");
        var log = new BreakwaterLedger(13, metrics, model);
        assertTrue(log.begin(1));
        log.row.attackHits = 3; log.row.attackDamage = 4.9;
        log.seal(1, 44, 1, new double[]{-.5}, new int[]{2, 1, 1, 4}, new int[]{1,2,4,1});
        assertFalse(log.open(1));
        log.seal(1, 99, 1, new double[0], new int[0], new int[0]);
        assertFalse(log.begin(1));
        assertTrue(log.begin(2));
        assertFalse(log.open(1));
        log.row.attackDamage = .3;
        log.seal(2, 57, 1, new double[0], new int[4], new int[0]);
        var endpoint = JsonParser.parseString(Files.readString(model)).getAsJsonObject();
        var totals = endpoint.getAsJsonObject("totals");
        assertEquals(101, totals.get("turns").getAsInt());
        assertEquals(3, totals.get("attackHits").getAsInt());
        assertEquals(5.2, totals.get("attackDamage").getAsDouble(), 1e-9);
        assertEquals(2, Files.readAllLines(metrics).size());
        assertThrows(IllegalArgumentException.class, () -> new BreakwaterLedger(13, metrics, model));
    }
}
