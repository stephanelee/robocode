package prism;

import org.junit.jupiter.api.Test;
import prism.core.Angles;
import prism.core.Command;
import prism.core.Obs;
import prism.core.Physics;
import prism.core.Rules;
import prism.wave.Wave;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Replays a real recorded Tank Royale 1.4.0 trace (fixtures/engine-trace-1, produced by the
 * scripted probe bots in an isolated embedded-server battle) through Prism's predictor and turn
 * transaction. These are actual engine events, not synthetic fixtures.
 */
class EngineTraceTest {
    static final Path DIR = Path.of("fixtures", "engine-trace-1");
    static final Rules RULES = new Rules(800, 600, 0.1, 1);

    static List<Map<String, Object>> load(String name) throws IOException {
        List<Map<String, Object>> out = new ArrayList<>();
        for (String l : Files.readAllLines(DIR.resolve(name))) if (!l.isBlank()) out.add(Json.obj(l));
        return out;
    }

    static boolean has(Map<String, Object> r, String type) {
        for (Map<String, Object> e : Json.list(r, "events")) if (type.equals(e.get("type"))) return true;
        return false;
    }

    @Test
    void motionTransitionsMatchRecordedEngine() throws IOException {
        List<Map<String, Object>> m = load("mover.jsonl");
        int checked = 0, accel = 0, stop = 0, reverseHigh = 0, reverseLow = 0, maxTurn = 0, wall = 0;
        java.util.Set<Long> turnSpeeds = new java.util.HashSet<>();
        for (int i = 0; i + 1 < m.size(); i++) {
            Map<String, Object> a = m.get(i), b = m.get(i + 1);
            if (Json.num(b, "turn") != Json.num(a, "turn") + 1) continue;
            if (has(b, "hitBot") || has(b, "death")) continue;
            Map<String, Object> c = Json.sub(a, "cmd");
            double v = Json.num(a, "speed"), target = Json.num(c, "targetSpeed"), tr = Json.num(c, "turnRate");
            Physics.Motion s = new Physics.Motion(Json.num(a, "x"), Json.num(a, "y"), Math.toRadians(Json.num(a, "dir")), v);
            boolean contactBefore = has(a, "hitWall");
            Physics.Step st = Physics.step(s, target, Math.toRadians(tr), RULES, contactBefore);
            assertEquals(Json.num(b, "x"), st.next().x(), 1e-6, "x at turn " + Json.num(b, "turn"));
            assertEquals(Json.num(b, "y"), st.next().y(), 1e-6, "y at turn " + Json.num(b, "turn"));
            assertEquals(Json.num(b, "speed"), st.next().speed(), 1e-6, "speed at turn " + Json.num(b, "turn"));
            double dd = Math.toDegrees(Angles.wrap(st.next().heading() - Math.toRadians(Json.num(b, "dir"))));
            assertEquals(0, dd, 1e-6, "heading at turn " + Json.num(b, "turn"));
            assertEquals(has(b, "hitWall"), st.hitWall() && !contactBefore, "wall event at turn " + Json.num(b, "turn"));
            // energy: wall damage is the only self-inflicted change besides hits and our own shot
            double de = Json.num(a, "energy") - Json.num(b, "energy");
            double expected = st.wallDamage();
            for (Map<String, Object> e : Json.list(b, "events")) {
                if ("hitByBullet".equals(e.get("type"))) expected += Json.num(e, "damage");
                if ("fired".equals(e.get("type"))) expected += Json.num(Json.sub(e, "bullet"), "power");
            }
            assertEquals(expected, de, 1e-9, "energy at turn " + Json.num(b, "turn"));
            checked++;
            double nv = Json.num(b, "speed");
            if (Math.abs(nv) > Math.abs(v) && Math.signum(nv) == Math.signum(target)) accel++;
            if (target == 0 && Math.abs(v) > 2 && nv == 0 && !st.hitWall()) stop++;
            if (target != 0 && v != 0 && Math.signum(v) != Math.signum(target)) { if (Math.abs(v) > 2) reverseHigh++; else reverseLow++; }
            if (Math.abs(tr) >= 10 && Math.abs(nv) > 0) { maxTurn++; turnSpeeds.add(Math.round(Math.abs(nv))); }
            if (st.hitWall()) wall++;
        }
        assertTrue(checked > 300, "transitions checked: " + checked);
        assertTrue(accel > 0 && stop > 0 && reverseHigh > 0 && reverseLow > 0 && wall > 0,
                "coverage accel=" + accel + " stop=" + stop + " revHigh=" + reverseHigh + " revLow=" + reverseLow + " wall=" + wall);
        assertTrue(turnSpeeds.size() >= 2, "max turns at several speeds: " + turnSpeeds);
    }

    @Test
    void fireUsesHeadingObservedOnRequestTickAndConfirmsNextTurn() throws IOException {
        List<Map<String, Object>> s = load("shooter.jsonl");
        int requests = 0, confirmations = 0;
        for (int i = 0; i + 1 < s.size(); i++) {
            Map<String, Object> a = s.get(i), b = s.get(i + 1);
            Map<String, Object> c = Json.sub(a, "cmd");
            boolean requested = Json.num(c, "fire") > 0 && Boolean.TRUE.equals(c.get("accepted"));
            Map<String, Object> fired = null;
            for (Map<String, Object> e : Json.list(b, "events")) if ("fired".equals(e.get("type"))) fired = e;
            if (requested) requests++;
            if (fired != null) {
                confirmations++;
                assertTrue(requested, "confirmation without request at turn " + Json.num(b, "turn"));
                assertEquals(Json.num(a, "turn") + 1, Json.num(fired, "turn"));
                Map<String, Object> bl = Json.sub(fired, "bullet");
                // shot direction is the gun heading observed on the request tick, not that plus the requested 15 degree turn
                assertEquals(0, Angles.wrap(Math.toRadians(Json.num(bl, "dir") - Json.num(a, "gunDir"))), 1e-9);
                assertNotEquals(Json.num(b, "gunDir"), Json.num(bl, "dir"), 1e-3);
                assertEquals(Json.num(a, "x"), Json.num(bl, "x"), 1e-9);
                assertEquals(Json.num(a, "y"), Json.num(bl, "y"), 1e-9);
                assertEquals(Json.num(c, "fire"), Json.num(bl, "power"), 1e-12);
            } else assertFalse(requested && Json.num(b, "turn") == Json.num(a, "turn") + 1 && !has(b, "death") && !has(b, "won"),
                    "accepted request without confirmation at turn " + Json.num(a, "turn"));
            // the engine only accepts a request when the observed heat is exactly zero
            if (requested) assertEquals(0.0, Json.num(a, "gunHeat"));
        }
        assertTrue(requests >= 5, "requests " + requests);
        assertTrue(confirmations >= 5);
    }

    @Test
    void scanEnergyVisibilityAndInferenceThroughTheTransaction() throws IOException {
        // Shooter's view: our hits show in the scan one turn later; the mover fired exactly one
        // power-1 shot and hit walls repeatedly. Prism must infer that one shot and no others.
        List<Map<String, Object>> s = load("shooter.jsonl");
        List<Map<String, Object>> mv = load("mover.jsonl");
        int moverFireTurn = -1;
        for (Map<String, Object> r : mv) for (Map<String, Object> e : Json.list(r, "events")) if ("fired".equals(e.get("type"))) moverFireTurn = (int) Json.num(e, "turn");
        assertTrue(moverFireTurn > 0);

        // visibility contract on isolated hits
        int verifiedHits = 0;
        Map<Integer, Double> scanEnergy = new java.util.HashMap<>();
        for (Map<String, Object> r : s) for (Map<String, Object> e : Json.list(r, "events"))
            if ("scan".equals(e.get("type"))) scanEnergy.put((int) Json.num(e, "turn"), Json.num(e, "energy"));
        for (Map<String, Object> r : s) for (Map<String, Object> e : Json.list(r, "events")) {
            if (!"bulletHit".equals(e.get("type"))) continue;
            int k = (int) Json.num(e, "turn");
            Double before = scanEnergy.get(k - 1), at = scanEnergy.get(k), after = scanEnergy.get(k + 1);
            if (before == null || at == null || after == null) continue;
            if (Math.abs(before - at) > 1e-9) continue; // another effect on k
            if (Math.abs((at - after) - Json.num(e, "damage")) < 1e-9) verifiedHits++;
        }
        assertTrue(verifiedHits >= 3, "isolated hits verified: " + verifiedHits);

        Telemetry tel = new Telemetry(null);
        Brain brain = new Brain(RULES, 1, tel);
        List<Wave> incoming = new ArrayList<>();
        for (Map<String, Object> r : s) {
            Obs.Turn t = toTurn(r);
            Command c = brain.turn(t);
            assertNotNull(c);
            for (Wave w : brain.waves()) if (w.role == Wave.Role.INCOMING && !incoming.contains(w)) incoming.add(w);
        }
        assertEquals(1, incoming.size(), "inferred shots: " + incoming.size());
        Wave w = incoming.get(0);
        assertTrue(w.fireTurnMin <= moverFireTurn && moverFireTurn <= w.fireTurnMax, "fire turn " + w.fireTurnMin + ".." + w.fireTurnMax + " vs " + moverFireTurn);
        assertEquals(1.0, w.power, 1e-6);
        assertEquals(0, tel.get("failure.scan") + tel.get("failure.waves") + tel.get("failure.move") + tel.get("failure.gun") + tel.get("failure.radar"));
    }

    static Obs.Turn toTurn(Map<String, Object> r) {
        int turn = (int) Json.num(r, "turn");
        List<Obs.Event> ev = new ArrayList<>();
        for (Map<String, Object> e : Json.list(r, "events")) {
            int et = (int) Json.num(e, "turn");
            switch ((String) e.get("type")) {
                case "scan" -> ev.add(new Obs.Scan(et, (int) Json.num(e, "id"), Json.num(e, "energy"), Json.num(e, "x"), Json.num(e, "y"), Math.toRadians(Json.num(e, "dir")), Json.num(e, "speed")));
                case "fired" -> { Map<String, Object> b = Json.sub(e, "bullet"); ev.add(new Obs.BulletFired(et, (int) Json.num(b, "id"), Json.num(b, "power"), Json.num(b, "x"), Json.num(b, "y"), Math.toRadians(Json.num(b, "dir")))); }
                case "bulletHit" -> { Map<String, Object> b = Json.sub(e, "bullet"); ev.add(new Obs.BulletHit(et, (int) Json.num(b, "id"), (int) Json.num(e, "victim"), Json.num(e, "damage"), Json.num(e, "energy"), Json.num(b, "power"))); }
                case "hitByBullet" -> { Map<String, Object> b = Json.sub(e, "bullet"); ev.add(new Obs.HitByBullet(et, (int) Json.num(b, "id"), (int) Json.num(b, "owner"), Json.num(b, "power"), Json.num(b, "x"), Json.num(b, "y"), Math.toRadians(Json.num(b, "dir")), Json.num(e, "damage"), Json.num(e, "energy"))); }
                case "bulletWall" -> ev.add(new Obs.BulletHitWall(et, (int) Json.num(Json.sub(e, "bullet"), "id")));
                case "hitWall" -> ev.add(new Obs.HitWall(et));
                case "death" -> ev.add(new Obs.Death(et));
                case "won" -> ev.add(new Obs.Won(et));
                default -> {}
            }
        }
        Obs.Own own = new Obs.Own(turn, Json.num(r, "x"), Json.num(r, "y"), Math.toRadians(Json.num(r, "dir")), Math.toRadians(Json.num(r, "gunDir")),
                Math.toRadians(Json.num(r, "radarDir")), Json.num(r, "speed"), Json.num(r, "energy"), Json.num(r, "gunHeat"), (int) Json.num(r, "enemies"));
        return new Obs.Turn((int) Json.num(r, "round"), own, ev);
    }
}
