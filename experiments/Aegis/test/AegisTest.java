import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class AegisTest {
    // ------------------------------------------------------------ pinned engine rules
    @Test void speedLawMatchesPinnedServerRules() {
        assertEquals(6, Physics.newSpeed(8, 5), 0);       // decelerate by at most 2
        assertEquals(8, Physics.newSpeed(7.5, 8), 0);     // accelerate by at most 1, capped
        assertEquals(0, Physics.newSpeed(8, 0), 0);       // target 0 stops at once (not classic 6)
        assertEquals(3, Physics.newSpeed(8, -8), 0);      // reversal formula (1 - v/2) * -1 gives +3 (classic 6)
        assertEquals(-0.5, Physics.newSpeed(1, -8), 1e-12);
        assertEquals(-1, Physics.newSpeed(0, -3), 0);
        assertEquals(-3, Physics.newSpeed(-8, 8), 0);     // symmetric for negative speeds
    }

    @Test void bulletRulesMatchPinnedServer() {
        assertEquals(19.7, Physics.bulletSpeed(0.1), 1e-12);
        assertEquals(14.15, Physics.bulletSpeed(1.95), 1e-12);
        assertEquals(1.02, Physics.gunHeat(0.1), 1e-12);
        assertEquals(4 * 1.95 + 2 * 0.95, Physics.damage(1.95), 1e-12);
    }

    @Test void segmentAndCircleTests() {
        assertTrue(Physics.segmentsIntersect(0, 0, 10, 10, 0, 10, 10, 0));
        assertFalse(Physics.segmentsIntersect(0, 0, 10, 0, 0, 1, 10, 1));      // parallel
        assertFalse(Physics.segmentsIntersect(0, 0, 4, 4, 6, 0, 10, -4));
        assertTrue(Physics.segmentHitsCircle(-30, 17, 30, 17, 0, 0, 18));
        assertFalse(Physics.segmentHitsCircle(-30, 19, 30, 19, 0, 0, 18));
        assertTrue(Physics.segmentHitsCircle(5, 5, 6, 6, 0, 0, 18));           // endpoint inside
    }

    // ------------------------------------------------------------ inferred shot timing
    @Test void shotTimingFollowsCreationThenTickSegments() {
        Shot shot = new Shot(100, 1.95, 0, 0, 600, 0);
        assertEquals(14.15, shot.travelled(100), 1e-12);                       // end of creation turn = tick 1
        int contact = shot.contactTurn();
        assertTrue(shot.travelled(contact) + shot.speed >= 600 - 18);
        assertTrue(shot.travelled(contact - 1) + shot.speed < 600 - 18);
        assertTrue(shot.breakFirst() <= contact && shot.breakLast() >= contact);
        assertTrue(shot.travelled(shot.breakFirst() + 1) >= 600 - 22 && shot.travelled(shot.breakFirst()) < 600 - 22);
        assertTrue(shot.travelled(shot.breakLast()) <= 600 + 22 && shot.travelled(shot.breakLast() + 1) > 600 + 22);
    }

    @Test void inferredPowerAppliesOneScanLateCorrections() {
        assertEquals(1.95, Aegis.inferredPower(100, 98.05, 0), 1e-9);
        assertTrue(Double.isNaN(Aegis.inferredPower(100, 100, 0)));
        // our 9.75 damage landed last turn (correction -9.75) and the shooter fired 1.95 this turn
        assertEquals(1.95, Aegis.inferredPower(100, 100 - 9.75 - 1.95, -9.75), 1e-9);
        // its hit bonus 5.85 arrived and it fired 1.95
        assertEquals(1.95, Aegis.inferredPower(50, 50 + 5.85 - 1.95, 5.85), 1e-9);
        assertTrue(Double.isNaN(Aegis.inferredPower(100, 96.5, 0)));           // 3.5 is not a legal power
    }

    @Test void classicSelfPredictionModel() {
        assertEquals(6, Aegis.classicSpeed(8, 0), 0);    // engine stopped at once, classic predictor decelerates by 2
        assertEquals(6, Aegis.classicSpeed(8, 3), 0);    // engine reversal quirk, classic decelerates by 2
        assertEquals(-4, Aegis.classicSpeed(-6, -2), 0);
        assertEquals(7, Aegis.classicSpeed(8, 7), 0);    // ordinary change: same in both
        assertEquals(1, Aegis.classicSpeed(0, 1), 0);
    }

    // ------------------------------------------------------------ shield solver against an independent per-turn check
    @Test void committedDirectionCollidesUnderEngineSegments() {
        Shot shot = new Shot(0, 1.95, 600, 0, 0, 0);
        int contact = shot.contactTurn();
        ShieldSolver.Positions own = new ShieldSolver.Positions() {  // pass laterally through the aim point at contact
            public double x(int turn) { return 0; }
            public double y(int turn) { return 8.0 * (contact - turn); }
        };
        ShieldSolver.Plan best = null;
        for (int f = contact - 6; f <= contact; f++) { var p = ShieldSolver.best(shot, f, own); if (best == null || p.score() > best.score()) best = p; }
        assertNotNull(best); assertTrue(best.covered() >= 5, "covered " + best.covered());
        assertTrue((best.mask() & (1 << 4)) != 0, "centre line covered");
        // independent replay of the centre line: bullet segments [tick j-N+1, j-N+2], shield [j-f+1, j-f+2]
        double px = own.x(best.fireTurn() - 1), py = own.y(best.fireTurn() - 1), w = Math.toRadians(best.direction());
        boolean collided = false; int hitTurn = Integer.MAX_VALUE;
        for (int j = 0; j < 80 && !collided; j++) {
            double e1 = 600 - 14.15 * (j + 1), e2 = 600 - 14.15 * (j + 2);
            if (Physics.segmentHitsCircle(e1, 0, e2, 0, own.x(j), own.y(j), 18)) { hitTurn = j; break; }
            if (j < best.fireTurn()) continue;
            double s1 = 19.7 * (j - best.fireTurn() + 1), s2 = s1 + 19.7;
            collided = Physics.segmentsIntersect(e1, 0, e2, 0, px + Math.cos(w) * s1, py + Math.sin(w) * s1, px + Math.cos(w) * s2, py + Math.sin(w) * s2);
            if (collided) assertTrue(j < hitTurn);
        }
        assertTrue(collided);
    }

    @Test void degenerateOnLineShieldIsNotClaimed() {
        Shot shot = new Shot(0, 1.95, 600, 0, 0, 0);
        ShieldSolver.Positions still = new ShieldSolver.Positions() { public double x(int t) { return 0; } public double y(int t) { return 0; } };
        int contact = shot.contactTurn();
        for (int f = contact - 6; f <= contact; f++) assertEquals(0, ShieldSolver.best(shot, f, still).mask() & (1 << 4), "centre line through our own origin is never claimed");
    }

    // ------------------------------------------------------------ label model
    /** Research fixtures: labels computed offline by the pinned opponent's own wave code in a dev-only instrumented copy. */
    @Test void labelModelReproducesOfflineOpponentLabels() {
        double[][] cases = {
            {160.99761024176328, 446.06753069838715, 30, 50, 326.1068422226021, 179.06548007053823, -0.9465629665605279},
            {163.3105060563667, 451.20239964221486, 45, 66, 302.0638375170746, 152.9468291725692, 2.240266310529687},
            {96.61338948649033, 389.75188651633414, 60, 80, 332.5408857353489, 186.05497819816372, -0.6698357870800538},
            {98.34935049059868, 416.3227844868236, 75, 96, 308.1592471607293, 159.56845897768812, 1.9983879937623625}};
        double[][] xs = {{316.96372775711984, 319.67279870985533, 322.3818696625908, 325.0909406153263},
            {313.91602293529235, 311.8842197207407, 309.85241650618906, 307.8206132916374},
            {325.0909406153262, 328.4772793062456, 331.86361799716497, 335.24995668808435},
            {318.6568971025794, 317.3023616262117, 315.94782614984393, 314.5932906734762, 313.23875519710845}};
        double[][] ys = {{169.13303536285989, 172.07598194291273, 175.01892852296558, 177.96187510301843},
            {165.8222204603004, 163.61501052526077, 161.40780059022114, 159.20059065518151},
            {177.96187510301846, 181.64055832808452, 185.3192415531506, 188.99792477821666},
            {170.9723769753929, 169.50090368536647, 168.02943039534003, 166.5579571053136, 165.08648381528715}};
        for (int i = 0; i < cases.length; i++) {
            double[] c = cases[i];
            double got = DiamondLabel.offset(c[0], c[1], (int) c[2], 14.15, (int) c[3], xs[i], ys[i], c[4], c[5]);
            assertEquals(c[6], got, 1e-6, "case " + i);
        }
    }

    @Test void stationaryPassHasZeroLabel() {
        double[] xs = new double[8], ys = new double[8]; java.util.Arrays.fill(xs, 300); java.util.Arrays.fill(ys, 200);
        // an axis-aligned square's angular centre is slightly off its bearing: well under one pixel here
        assertEquals(0, DiamondLabel.offset(0, 0, 0, 14.15, 20, xs, ys, 300, 200), 0.1);
    }

    // ------------------------------------------------------------ planner obeys exact physics and bounds
    @Test void plannerTrajectoryIsEnginePhysicsAndBounded() {
        LateralPlanner planner = new LateralPlanner(-20, 30, 20);
        for (int i = 0; i < planner.states(); i++) { double s = planner.position(i); planner.addPositionCost(10, i, (s - 25) * (s - 25)); }
        var result = planner.plan(0, 0);
        for (int t = 0; t < 20; t++) {
            double v = result.speeds()[t], nv = result.speeds()[t + 1]; boolean legal = false;
            for (int u = -8; u <= 8; u++) if (Physics.newSpeed(v, u) == nv) legal = true;
            assertTrue(legal, "speed transition " + v + "->" + nv);
            assertEquals(result.positions()[t] + nv, result.positions()[t + 1], 1e-9);
            assertTrue(result.positions()[t + 1] >= -20 && result.positions()[t + 1] <= 30);
        }
        assertEquals(25, result.positions()[10], 0.6);
        assertEquals(Physics.newSpeed(0, result.targetSpeed()), result.speeds()[1], 0);
        // reuse after reset gives exactly the same plan as a fresh planner
        planner.reset();
        for (int i = 0; i < planner.states(); i++) { double s = planner.position(i); planner.addPositionCost(10, i, (s - 25) * (s - 25)); }
        assertArrayEquals(result.positions(), planner.plan(0, 0).positions(), 0);
    }
}
