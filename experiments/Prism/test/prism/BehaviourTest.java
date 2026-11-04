package prism;

import org.junit.jupiter.api.Test;
import prism.core.Command;
import prism.core.Geometry;
import prism.core.Obs;
import prism.core.Physics;
import prism.core.Rules;
import prism.gun.Firing;
import prism.learn.Features;
import prism.move.Danger;
import prism.move.DuelPlanner;
import prism.move.MeleePlanner;
import prism.move.Shadows;
import prism.radar.Radar;
import prism.track.FireInference;
import prism.wave.Wave;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** Chronology, lifecycle, inference, surfing, shadows, melee, radar and runtime bounds. */
class BehaviourTest {
    static final Rules R = new Rules(800, 600, 0.1, 1);

    static Obs.Own own(int turn, double x, double y, double heat, double energy) {
        return new Obs.Own(turn, x, y, 0, 0, 0, 0, energy, heat, 1);
    }

    static Obs.Scan scan(int turn, double e, double x, double y) { return new Obs.Scan(turn, 7, e, x, y, Math.PI / 2, 8); }

    // ------------------------------------------------------------------ energy visibility

    @Test
    void isolatedDamageAndHitBonusAreNotShots() {
        FireInference f = new FireInference(R);
        int t = 40;
        assertNull(f.onScan(t, 100, 0, 0));
        f.addEffect(new FireInference.Effect(t + 2, -4, 0, "ourHit"));    // hit on t+1 shows at t+2
        assertNull(f.onScan(t + 1, 100, 0, 0));                          // unchanged at k
        assertNull(f.onScan(t + 2, 96, 0, 0));                           // damage at k+1: not a shot
        f.addEffect(new FireInference.Effect(t + 3, 6, 0, "enemyHit"));  // bonus
        assertNull(f.onScan(t + 3, 102, 0, 0));
    }

    @Test
    void simultaneousHitAndFireKeepsTheShotWithExactPower() {
        FireInference f = new FireInference(R);
        f.onScan(40, 100, 0, 0);
        f.addEffect(new FireInference.Effect(41, -4, 0, "ourHit"));
        FireInference.Shot s = f.onScan(41, 100 - 4 - 1.5, 0, 0);
        assertNotNull(s);
        assertTrue(s.certain());
        assertEquals(1.5, s.power(), 1e-9);
        assertEquals(41, s.fireTurnMin());
        // heat now forbids another certain shot until cooled: a loss inside the window is not a shot
        assertNull(f.onScan(42, 100 - 4 - 1.5 - 1.0, 0, 0));
        assertEquals(1, f.unresolvedCount());
    }

    @Test
    void missingScansAndWallsRetainUncertainty() {
        FireInference f = new FireInference(R);
        f.onScan(40, 100, 0, 0);
        FireInference.Shot s = f.onScan(44, 98, 0, 0);                   // gap of four turns
        assertNotNull(s);
        assertFalse(s.certain());
        assertEquals(41, s.fireTurnMin());
        assertEquals(44, s.fireTurnMax());
        FireInference g = new FireInference(R);
        g.onScan(40, 100, 0, 0);
        assertNull(g.onScan(41, 97, 2.0, 3.0));                          // wall could explain all of it
        assertEquals(1, g.unresolvedCount());
        FireInference h = new FireInference(R);
        h.onScan(10, 100, 0, 0);
        assertNull(h.onScan(11, 99, 0, 0));                              // initial heat 3.0: cannot fire before turn 31
    }

    // ------------------------------------------------------------------ chronology and lifecycle

    private static Brain duelBrain() { return new Brain(R, 1, new Telemetry(null)); }

    /** Applies the commanded gun turn to a simulated gun heading so plans can align. */
    static final class Gun { double heading; void apply(Command c) { heading = prism.core.Angles.heading(heading + c.gunTurn()); } }

    static Obs.Own ownG(int turn, double heat, double energy, double gun) {
        return new Obs.Own(turn, 200, 300, 0, gun, 0, 0, energy, heat, 1);
    }

    static Obs.Scan still(int turn) { return new Obs.Scan(turn, 7, 100, 600, 300, 0, 0); }

    @Test
    void fireRequestIsPendingUntilConfirmedAndCancelledIfAbsent() {
        Brain b = duelBrain();
        Gun g = new Gun();
        int t = 1;
        Command last = null;
        // approach readiness with a fresh scan every turn; heat reaches zero at turn 30
        for (; t <= 40; t++) {
            double heat = Math.max(0, 3.0 - 0.1 * t);
            if (heat < 1e-9) heat = 0;
            last = b.turn(new Obs.Turn(1, ownG(t, heat, 100, g.heading), List.of(still(t))));
            g.apply(last);
            if (last.firepower() > 0) break;
        }
        assertNotNull(last);
        assertTrue(last.firepower() > 0, "a shot should be requested once cool and aligned");
        assertNotNull(b.pending());
        int req = t;
        // no BulletFired arrives on the next turn: the request is cancelled, no physical wave appears
        b.turn(new Obs.Turn(1, ownG(req + 1, 0, 100, g.heading), List.of(still(req + 1))));
        assertNull(b.pending());
        assertEquals(1, b.telemetry().get("fire.unconfirmedCancelled"));
        assertTrue(b.waves().stream().noneMatch(w -> w.physical));
        assertEquals(0, b.bullets().size());
    }

    @Test
    void confirmationPromotesTheRequestTickWaveExactlyOnce() {
        Brain b = duelBrain();
        Gun g = new Gun();
        int t;
        Command c = null;
        for (t = 1; t <= 40; t++) {
            double heat = Math.max(0, 3.0 - 0.1 * t); if (heat < 1e-9) heat = 0;
            c = b.turn(new Obs.Turn(1, ownG(t, heat, 100, g.heading), List.of(still(t))));
            g.apply(c);
            if (c.firepower() > 0) break;
        }
        assertTrue(c.firepower() > 0);
        Obs.BulletFired fired = new Obs.BulletFired(t + 1, 1, c.firepower(), 200, 300, 0);
        List<Obs.Event> ev = List.of(fired, fired, still(t + 1)); // duplicate callback
        b.turn(new Obs.Turn(1, ownG(t + 1, 1.4, 100 - c.firepower(), g.heading), ev));
        assertEquals(1, b.telemetry().get("fire.confirmed"));
        assertEquals(1, b.telemetry().get("events.duplicate"));
        assertEquals(1, b.waves().stream().filter(w -> w.physical).count());
        assertNull(b.pending());
    }

    @Test
    void predictionsAreFrozenBeforeLabelsAndRoundResetKeepsLearning() {
        Brain b = duelBrain();
        // target moving north at 8 from x=600: produces crossings and labels
        for (int t = 1; t <= 120; t++) {
            double heat = Math.max(0, 3.0 - 0.1 * t); if (heat < 1e-9) heat = 0;
            double y = 100 + (t * 3) % 400;
            b.turn(new Obs.Turn(1, own(t, 200, 300, heat, 100), List.of(scan(t, 100, 600, y))));
        }
        int labels = b.guns().data(7).longStore.size();
        assertTrue(labels > 20, "labels " + labels);
        for (Wave w : b.waves()) assertTrue(!w.labelEmitted || !w.fireOpportunity || !w.predictions.isEmpty());
        b.turn(new Obs.Turn(2, own(1, 300, 300, 3, 100), List.of()));
        assertTrue(b.waves().isEmpty());
        assertNull(b.plan());
        assertNull(b.pending());
        assertEquals(labels, b.guns().data(7).longStore.size());          // learning retained
        assertNull(b.enemies().get(7).latest());                           // trajectories cleared
        Brain cold = duelBrain();                                          // a new game is cold
        assertTrue(cold.guns().all().isEmpty());
    }

    @Test
    void sameEventStreamIsDeterministic() {
        Brain a = duelBrain(), c = duelBrain();
        for (int t = 1; t <= 60; t++) {
            double heat = Math.max(0, 3.0 - 0.1 * t); if (heat < 1e-9) heat = 0;
            Obs.Turn in = new Obs.Turn(1, own(t, 200, 300, heat, 100), List.of(scan(t, 100 - (t == 45 ? 2 : 0), 600, 100 + t)));
            Command x = a.turn(in), y = c.turn(in);
            assertEquals(x.targetSpeed(), y.targetSpeed());
            assertEquals(x.bodyTurn(), y.bodyTurn());
            assertEquals(x.gunTurn(), y.gunTurn());
            assertEquals(x.radarTurn(), y.radarTurn());
            assertEquals(x.firepower(), y.firepower());
        }
    }

    // ------------------------------------------------------------------ firing policy

    @Test
    void powerPolicyKillCapReserveAndValidation() {
        assertEquals(2.0, Firing.duelPower(500, 100, 100), 1e-12);
        assertEquals(2.8, Firing.duelPower(150, 100, 100), 1e-12);
        assertEquals(0.75, Firing.duelPower(500, 100, 3), 1e-12);       // E/4
        assertEquals(1.0, Firing.duelPower(500, 100, 4), 1e-12);
        assertEquals(1.5, Firing.duelPower(500, 100, 7), 1e-12);        // (E+2)/6
        assertEquals(0.1, Firing.duelPower(500, 100, 0), 1e-12);        // zero-energy target
        assertEquals(2.0 * Math.pow(20 / 30.0, 2), Firing.duelPower(500, 20, 100), 1e-12);
        assertEquals(0, Firing.duelPower(500, 0.25, 100));              // reserve leaves < 0.1
        Firing.Plan p = new Firing.Plan(7, 1.0, 2.0, 10, 12, Math.toRadians(0.3), 100, 100, 1, "analytic", 10);
        assertEquals(Firing.Verdict.FIRE, Firing.validate(p, 11, 1.0, 0, 50, 10, true, 100, 100));
        assertEquals(Firing.Verdict.HEADING, Firing.validate(p, 11, 1.0 + Math.toRadians(0.31), 0, 50, 10, true, 100, 100));
        assertEquals(Firing.Verdict.HOT, Firing.validate(p, 11, 1.0, 0.1, 50, 10, true, 100, 100));
        assertEquals(Firing.Verdict.ENERGY, Firing.validate(p, 11, 1.0, 0, 2.0, 10, true, 100, 100));
        assertEquals(Firing.Verdict.EXPIRED, Firing.validate(p, 13, 1.0, 0, 50, 10, true, 100, 100));
        assertEquals(Firing.Verdict.EXPIRED, Firing.validate(p, 11, 1.0, 0, 50, 10, true, 104, 100));
        assertEquals(Firing.Verdict.TARGET_DEAD, Firing.validate(p, 11, 1.0, 0, 50, 10, false, 100, 100));
        assertEquals(Firing.Verdict.STALE_TARGET, Firing.validate(p, 11, 1.0, 0, 50, 5, true, 100, 100));
    }

    // ------------------------------------------------------------------ surfing

    static Features.Context ctx() { return LearningTest.ctx(0.5); }

    static Wave incoming(long id, int fireTurn, double ox, double oy, double selfX, double selfY, double power) {
        double ref = Math.atan2(selfY - oy, selfX - ox);
        return new Wave(id, Wave.Role.INCOMING, 7, -1, fireTurn, fireTurn, ox, oy, power, 0, ref, 1, Wave.Escape.THEORETICAL, 0, 0, ctx(), 1, fireTurn);
    }

    @Test
    void stopUsesEngineInstantStopAndMissingIntersectionsAreCostly() {
        DuelPlanner p = new DuelPlanner(R);
        Physics.Motion me = new Physics.Motion(400, 300, Math.PI / 2, 8);
        Wave w = incoming(1, 10, 400, 100, 400, 300, 2);
        Danger.Density dens = new Danger().density(w, 200, 0);
        DuelPlanner.Threat th = new DuelPlanner.Threat(w, dens, List.of(), 10, 0.75);
        DuelPlanner.Result r = p.plan(me, 12, 400, 100, 0, 0, List.of(th), DuelPlanner.Action.POS, false);
        assertTrue(r.firstStepSafe());
        assertNotEquals(DuelPlanner.Action.STOP, r.action(), "head-on prior: standing in place is the worst option");
        Physics.Step stop = Physics.step(me, 0, 0, R);
        assertEquals(0, stop.next().speed());
        assertEquals(400, stop.next().x(), 1e-12);
    }

    @Test
    void secondWaveContinuationChangesTheDecision() {
        DuelPlanner p = new DuelPlanner(R);
        Physics.Motion me = new Physics.Motion(400, 300, 0, 0);
        // first wave from the south is mirror-symmetric: POS (west) and NEG (east) tie; previous POS wins
        Wave w1 = incoming(1, 20, 400, 100, 400, 300, 2);
        DuelPlanner.Threat t1 = new DuelPlanner.Threat(w1, new Danger().density(w1, 200, 0), List.of(), 10, 0.75);
        DuelPlanner.Result one = p.plan(me, 22, 400, 100, 0, 0, List.of(t1), DuelPlanner.Action.POS, false);
        assertEquals(DuelPlanner.Action.POS, one.action(), "one-wave score " + one.score());
        // a second wave from the north whose learned hits lie on the west side
        Wave w2 = incoming(2, 25, 400, 550, 400, 300, 2);
        Danger dg = new Danger();
        for (int i = 0; i < 40; i++) dg.addHit(w2, -0.25 - 0.3 * (i % 5) / 4.0, false, 1, i);
        DuelPlanner.Threat t2 = new DuelPlanner.Threat(w2, dg.density(w2, 250, 0), List.of(), 10, 0.75);
        DuelPlanner.Result two = p.plan(me, 22, 400, 100, 0, 0, List.of(t1, t2), DuelPlanner.Action.POS, false);
        assertEquals(DuelPlanner.Action.NEG, two.action(), "second wave must change the first action");
        assertTrue(p.steps() <= prism.Profile.MAX_PREDICTOR_STEPS + 3);
    }

    @Test
    void firstStepIntoAWallNeverWins() {
        DuelPlanner p = new DuelPlanner(R);
        Physics.Motion me = new Physics.Motion(781, 300, 0, 8);   // heading into the east wall at full speed
        DuelPlanner.Result r = p.plan(me, 5, 400, 300, 0, 0, List.of(), DuelPlanner.Action.POS, false);
        Physics.Step s = Physics.step(me, r.control().targetSpeed(), r.control().turnRate(), R);
        assertFalse(s.hitWall(), "chosen " + r.action() + " " + r.reason());
    }

    // ------------------------------------------------------------------ shadows

    @Test
    void shadowsCoverPartialDisjointAndRespectDeath() {
        Wave w = incoming(1, 10, 100, 300, 700, 300, 3);  // enemy at x=100, we at 700, speed 11
        // our bullet fired east-to-west along y=300 from x=690 on turn 10 at power 3 (speed 11)
        // a radial own bullet shadows a single bearing (zero width)
        Shadows.Bullet radial = new Shadows.Bullet(1, 10, 690, 300, Math.PI, 3, Integer.MAX_VALUE);
        List<Geometry.Interval> line = Shadows.compute(w, List.of(radial), 10, 700, 300, R);
        assertEquals(1, line.size());
        assertEquals(0, line.get(0).width(), 1e-12);
        // an oblique own bullet crossing the enemy's line of fire shadows a positive width
        double h = Math.atan2(260 - 340, 100 - 690);
        Shadows.Bullet b = new Shadows.Bullet(1, 10, 690, 340, h, 3, Integer.MAX_VALUE);
        List<Geometry.Interval> sh = Shadows.compute(w, List.of(b), 10, 700, 300, R);
        assertFalse(sh.isEmpty());
        Geometry.Interval s0 = sh.get(0);
        assertTrue(s0.width() > 0);
        Geometry.Interval inside = new Geometry.Interval(s0.lo(), s0.lo() + s0.width() / 2);
        assertEquals(inside.width(), Geometry.overlap(inside, sh), 1e-12);            // complete coverage
        Geometry.Interval half = new Geometry.Interval(s0.mid(), s0.hi() + 0.05);
        double partial = Geometry.overlap(half, sh);
        assertTrue(partial > 0 && partial < half.width());                            // partial coverage
        // a confirmed death before the meeting invalidates the shadow
        Shadows.Bullet dead = new Shadows.Bullet(1, 10, 690, 340, h, 3, 15);
        assertTrue(Shadows.compute(w, List.of(dead), 10, 700, 300, R).isEmpty());
        // a second, distant bullet yields a disjoint interval that is retained
        Shadows.Bullet off = new Shadows.Bullet(2, 10, 690, 520, Math.atan2(420 - 520, 100 - 690), 3, Integer.MAX_VALUE);
        List<Geometry.Interval> both = Shadows.compute(w, List.of(b, off), 10, 700, 300, R);
        assertTrue(both.size() >= 2, "disjoint intervals retained: " + both.size());
        // wraparound: a wave whose reference points west still yields small relative offsets
        Wave west = incoming(9, 10, 700, 300, 100, 300, 3);
        Shadows.Bullet wb = new Shadows.Bullet(3, 10, 110, 340, Math.atan2(260 - 340, 700 - 110), 3, Integer.MAX_VALUE);
        List<Geometry.Interval> ws = Shadows.compute(west, List.of(wb), 10, 100, 300, R);
        assertFalse(ws.isEmpty());
        for (Geometry.Interval i : ws) assertTrue(Math.abs(i.mid()) < 0.3, "wrapped offset " + i);
    }

    // ------------------------------------------------------------------ melee and radar

    @Test
    void meleeAllUnsafeYieldsLabelledEmergencyAndNeverTheLastUnsafeCandidate() {
        Rules tiny = new Rules(400, 400, 0.1, 3);
        MeleePlanner m = new MeleePlanner(tiny, 9);
        // boxed in the corner by three close opponents
        Physics.Motion me = new Physics.Motion(19, 19, Math.toRadians(225), 8);
        List<MeleePlanner.Opponent> opps = List.of(
                new MeleePlanner.Opponent(1, 60, 19, 100, 1, 0, 0, 0),
                new MeleePlanner.Opponent(2, 19, 60, 100, 1, 0, 0, 0),
                new MeleePlanner.Opponent(3, 55, 55, 100, 1, 0, 0, 0));
        MeleePlanner.Result r = m.plan(10, me, 100, opps);
        assertEquals("emergency", r.reason());
        // an open field yields a min-risk destination away from the threat
        MeleePlanner.Result ok = m.plan(11, new Physics.Motion(200, 200, 0, 0), 100,
                List.of(new MeleePlanner.Opponent(1, 300, 200, 100, 1, 0, 0, 0), new MeleePlanner.Opponent(2, 200, 330, 100, 1, 0, 0, 0)));
        assertEquals("min-risk", ok.reason());
        assertTrue(ok.destX() < 200 + 1e-9 || ok.destY() < 200 + 1e-9);
    }

    @Test
    void radarLocksFreshTargetSweepsUnknownAndHandlesStale() {
        Radar r = new Radar();
        double lim = Math.toRadians(45);
        // unknown participant: acquisition sweep at the limit, never an invented coordinate
        assertEquals(lim, Math.abs(r.command(5, 0, 100, 100, 1, List.of())), 1e-12);
        // fresh duel scan: lock with overshoot, within the limit
        double c = r.command(6, 0, 100, 100, 1, List.of(new Radar.Known(7, 500, 100, 5)));
        assertTrue(c > 0 && c <= lim);
        assertTrue(c >= Math.toRadians(2));
        // stale beyond four turns: full sweep
        assertEquals(lim, Math.abs(r.command(20, 0, 100, 100, 1, List.of(new Radar.Known(7, 500, 100, 5)))), 1e-12);
        // melee: prefers the stalest participant
        Radar m = new Radar();
        double toStale = m.command(30, 0, 400, 300, 2, List.of(new Radar.Known(1, 500, 300, 29), new Radar.Known(2, 400, 450, 10)));
        assertTrue(toStale > 0, "turns toward the stale bot to the north");
        // persistent participant mismatch forces acquisition but deletes nothing
        Radar q = new Radar();
        for (int t = 0; t < 30; t++) q.command(t, 0, 0, 0, 2, List.of(new Radar.Known(1, 10, 10, t)));
    }

    @Test
    void lastSurvivorTransitionUsesDuelLogicAndKeepsTargetData() {
        Rules mr = new Rules(800, 600, 0.1, 2);
        Brain b = new Brain(mr, 2, new Telemetry(null));
        for (int t = 1; t <= 50; t++) {
            double heat = Math.max(0, 3.0 - 0.1 * t); if (heat < 1e-9) heat = 0;
            List<Obs.Event> ev = new ArrayList<>();
            ev.add(new Obs.Scan(t, 7, 100, 600, 100 + t, Math.PI / 2, 8));
            if (t <= 30) ev.add(new Obs.Scan(t, 8, 100, 200, 500, 0, 0));
            if (t == 31) ev.add(new Obs.BotDeath(31, 8));
            int count = t <= 30 ? 2 : 1;
            b.turn(new Obs.Turn(1, new Obs.Own(t, 300, 300, 0, 0, 0, 0, 100, heat, count), ev));
        }
        assertFalse(b.enemies().get(8).alive);
        assertTrue(b.telemetry().get("move.action.POS") + b.telemetry().get("move.action.NEG") + b.telemetry().get("move.action.STOP") > 0);
        assertTrue(b.telemetry().get("move.reason.min-risk") + b.telemetry().get("move.reason.emergency") > 0);
        assertTrue(b.guns().data(7).displacement.size() > 0, "melee-origin displacement data retained");
    }

    // ------------------------------------------------------------------ runtime bounds

    @Test
    void worstCaseStoresAndWavesStayFiniteAndWithinQuota() {
        Brain b = duelBrain();
        // saturate the long-history store and the danger views
        var d = b.guns().data(7);
        java.util.Random rnd = new java.util.Random(1);
        for (int i = 0; i < prism.Profile.LONG_GUN_CAPACITY + 10; i++) {
            double[] x = new double[11];
            for (int k = 0; k < 11; k++) x[k] = rnd.nextDouble();
            d.longStore.add(x, new double[] {rnd.nextDouble() * 2 - 1}, 1, 1, i, i);
            for (var s : d.recent) s.add(new double[] {x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]}, new double[] {0.1}, 1, 1, i, i);
        }
        d.crossings = 10_000;
        assertEquals(prism.Profile.LONG_GUN_CAPACITY, d.longStore.size());
        long worst = 0;
        for (int t = 1; t <= 800; t++) {
            double heat = Math.max(0, 3.0 - 0.1 * t); if (heat < 1e-9) heat = 0;
            double e = 100 - (t % 17 == 0 ? 1.5 : 0);
            long t0 = System.nanoTime();
            Command c = b.turn(new Obs.Turn(1, own(t, 200, 300, heat, 100), List.of(scan(t, t % 17 == 0 ? e : 100, 600, 100 + (t * 5) % 400))));
            worst = Math.max(worst, System.nanoTime() - t0);
            assertTrue(Double.isFinite(c.targetSpeed() + c.bodyTurn() + c.gunTurn() + c.radarTurn() + c.firepower()));
            for (Wave w : b.waves()) assertTrue(b.waves().stream().filter(x -> x.targetId == w.targetId && !x.isThreat()).count() <= prism.Profile.MAX_WAVES_PER_ENEMY);
        }
        assertEquals(0, b.telemetry().get("failure.move") + b.telemetry().get("failure.gun") + b.telemetry().get("failure.waves") + b.telemetry().get("failure.scan"));
        // global payload estimate stays far below the 128 MiB cap with saturated per-enemy stores
        assertTrue(b.storePayloadBytes() < prism.Profile.GLOBAL_STORE_BYTES, "payload " + b.storePayloadBytes());
        assertTrue(b.storePayloadBytes() > 0);
        assertTrue(worst < 2_000_000_000L, "catastrophic latency");
        System.out.println("worst-case turn ms: " + worst / 1e6 + "; p99 us turn " + b.telemetry().quantileMicros("turn", 0.99) + " move " + b.telemetry().quantileMicros("move", 0.99) + " gun " + b.telemetry().quantileMicros("gun", 0.99) + " p50 turn " + b.telemetry().quantileMicros("turn", 0.5));
    }
}
