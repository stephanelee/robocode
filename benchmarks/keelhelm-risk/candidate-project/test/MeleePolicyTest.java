import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

/** Legal-scan and policy contracts; no competitive claims from fixtures. */
class MeleePolicyTest {
    @Test void queuedDeadScanCannotResurrectOpponentOrMaskSurvivorAtDuelBoundary() {
        var m = new MeleePolicy(); m.newRound(3);
        m.observe(7, 10, 100, 200, 0, 8, 100);
        m.observe(8, 10, 700, 500, 90, 4, 80);
        m.died(7);
        for (int t : new int[]{9, 10, 11}) m.observe(7, t, 100, 200, 0, 8, 100);
        assertEquals(1, m.opponents().size());
        assertTrue(m.enterDuel(1));
        assertTrue(m.isDead(7), "death tombstone must survive the mode transition");
        assertFalse(m.isDead(8), "survivor's same-turn scan remains eligible");
        m.observe(7, 12, 100, 200, 0, 8, 100);
        assertTrue(m.opponents().isEmpty()); assertFalse(m.active());
        m.newRound(3); assertFalse(m.isDead(7), "IDs can be reused next round");
        m.newRound(1); m.died(7);
        assertFalse(m.isDead(7), "from-start duel keeps its original scan event path");
    }
    @Test void simultaneousScansRemainSeparateAndOutOfOrderDataCannotOverwrite() {
        var m = new MeleePolicy(); m.newRound(3);
        m.observe(7, 10, 100, 200, 0, 8, 100);
        m.observe(8, 10, 700, 500, 90, 4, 80);
        m.observe(7, 9, 900, 900, 0, 0, 0);
        assertEquals(2, m.opponents().size());
        var seven = m.opponents().stream().filter(o -> o.id == 7).findFirst().orElseThrow();
        assertEquals(100, seven.x); assertEquals(10, seven.turn);
    }

    @Test void deadAndStaleTargetsAreUnavailable() {
        var m = new MeleePolicy(); m.newRound(3);
        var self = new Phys.Pose(400, 300, 0, 0);
        m.observe(7, 10, 450, 300, 0, 0, 10);
        m.observe(8, 10, 700, 500, 0, 0, 100);
        assertEquals(7, m.target(10, self).id);
        m.died(7); assertEquals(8, m.target(10, self).id);
        assertNull(m.target(43, self));
    }

    @Test void meleeToDuelBoundaryClearsAllTransientIdentitiesExactlyOnce() {
        var m = new MeleePolicy(); m.newRound(3);
        m.observe(7, 1, 100, 200, 0, 0, 100);
        assertFalse(m.enterDuel(2)); assertTrue(m.active());
        assertTrue(m.enterDuel(1)); assertFalse(m.active());
        assertTrue(m.opponents().isEmpty()); assertFalse(m.enterDuel(1));
        m.newRound(1); assertFalse(m.enterDuel(1));
        m.observe(9, 2, 100, 200, 0, 0, 100);
        assertTrue(m.enterDuel(1), "first-tick count may not yet be available at round start");
    }

    @Test void circularInterceptStaysInsideTheArenaIncludingStaleWallwardTargets() {
        var m = new MeleePolicy(); m.newRound(3);
        m.observe(7, 1, 780, 580, 45, 8, 100);
        m.observe(7, 2, 782, 582, 49, 8, 100);
        var o = m.opponents().iterator().next();
        var p = MeleePolicy.intercept(o, 25, new Phys.Pose(100, 100, 0, 0), 2, 800, 600);
        assertTrue(p.x() >= 18 && p.x() <= 782);
        assertTrue(p.y() >= 18 && p.y() <= 582);
        assertTrue(Double.isFinite(p.h()));
    }

    @Test void movementEscapesNearbyClusterAndDoesNotCommandThroughWall() {
        var m = new MeleePolicy(); m.newRound(3);
        m.observe(1, 0, 540, 470, 0, 0, 100);
        m.observe(2, 0, 550, 500, 0, 0, 100);
        m.observe(3, 0, 540, 530, 0, 0, 100);
        var p = new Phys.Pose(500, 500, 0, 0);
        for (int t = 0; t < 20; t++) {
            var c = m.move(t, p, 100, 1000, 1000);
            p = Phys.step(p, c.turn(), c.speed(), 1000, 1000);
        }
        assertTrue(p.x() < 500, "escape the clustered opponents to the east");
        m.newRound(3); p = new Phys.Pose(19, 500, 180, 0);
        var c = m.move(0, p, 100, 1000, 1000);
        for (int t = 0; t < 24; t++) {
            p = Phys.step(p, c.turn(), c.speed(), 1000, 1000);
            assertTrue(p.x() >= 18 && p.x() <= 982);
        }
    }

    @Test void firepowerIsBoundedAndConservesLowEnergy() {
        assertTrue(MeleePolicy.power(200, 100, 100) > MeleePolicy.power(700, 100, 100));
        assertTrue(MeleePolicy.power(200, 3, 100) < 1);
        assertEquals(0.1, MeleePolicy.power(200, 100, 0));
        assertTrue(MeleePolicy.power(200, 100, 1) <= 0.3);
    }
    @org.junit.jupiter.api.Test void saturatedRequestsDoNotCountAsEffectiveMovementExposure() {
        var self = new Phys.Pose(500, 500, 0, 8);
        org.junit.jupiter.api.Assertions.assertFalse(MeleePolicy.effectiveDifference(self,
                new MeleePolicy.Command(35,8),new MeleePolicy.Command(50,8),1000,1000));
        org.junit.jupiter.api.Assertions.assertTrue(MeleePolicy.effectiveDifference(self,
                new MeleePolicy.Command(0,8),new MeleePolicy.Command(50,8),1000,1000));
    }
}
