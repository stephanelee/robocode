import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class MeleeFireTest {
    private static final int TURN = 20;
    private static final Phys.Pose SELF = new Phys.Pose(400, 300, 0, 0);
    private static final double DIRECTION = Phys.bearing(400, 300, 600, 300);
    private final MeleePolicy policy = new MeleePolicy();

    @Test void alignedIncumbentTakesPriorityOverCloserAlternates() {
        var incumbent = aim(add(1, 650, 300, TURN), TURN);
        add(2, 600, 300, TURN);
        assertSame(incumbent, choose(incumbent, 100, 0));
    }

    @Test void shortestEligibleRangeWinsWithStableTieById() {
        var incumbent = aim(add(1, 400, 550, TURN), TURN);
        add(8, 560, 300, TURN - 3);
        var closer = add(4, 560, 300, TURN);
        add(3, 580, 300, TURN);
        assertSame(closer, choose(incumbent, 100, 0).target());
    }

    @Test void alternateDoesNotChangeIncumbentTracking() {
        var incumbent = add(1, 400, 450, TURN);
        var alternate = add(2, 600, 300, TURN);
        assertSame(incumbent, policy.target(TURN, SELF));
        var originalAim = aim(incumbent, TURN);
        assertSame(alternate, choose(originalAim, 100, 0).target());
        assertSame(incumbent, policy.target(TURN, SELF));
        assertEquals(originalAim, aim(incumbent, TURN));
    }

    @Test void hotGunAndInadequateReserveRejectEveryRequest() {
        var incumbent = aim(add(1, 400, 550, TURN), TURN);
        add(2, 550, 300, TURN);
        assertNull(choose(incumbent, 100, 0.0001));
        assertNull(choose(incumbent, 0.52, 0));
    }

    @Test void excludesDeadStaleFutureDistantAndMisalignedAlternates() {
        var incumbent = aim(add(1, 400, 550, TURN), TURN);
        add(2, 500, 300, TURN);
        policy.died(2);
        add(2, 500, 300, TURN);
        add(3, 550, 300, TURN - 4);
        add(4, 600, 300, TURN + 1);
        add(5, 900, 300, TURN);
        add(6, 600, 320, TURN);
        assertNull(choose(incumbent, 100, 0));
    }

    @Test void confirmedShotRequiresExactTurnAndPowerAndEquivalentDirection() {
        var request = new MeleeFire.Request(TURN + 1, 7, 1.6, 0);
        assertTrue(request.matches(TURN + 1, 1.6, 360));
        assertFalse(request.matches(TURN, 1.6, 0));
        assertFalse(request.matches(TURN + 2, 1.6, 0));
        assertFalse(request.matches(TURN + 1, 1, 0));
        assertFalse(request.matches(TURN + 1, 1.6, 1));
    }

    private MeleePolicy.Opponent add(int id, double x, double y, int turn) {
        policy.observe(id, turn, x, y, 0, 0, 100);
        return policy.opponents().stream().filter(o -> o.id == id).findFirst().orElse(null);
    }

    private MeleeFire.Aim aim(MeleePolicy.Opponent opponent, int turn) {
        double range = Phys.distance(SELF.x(), SELF.y(), opponent.x, opponent.y);
        return MeleeFire.aim(opponent, turn, SELF, MeleePolicy.power(range, 100, opponent.energy), range,
                DIRECTION, 1000, 1000);
    }

    private MeleeFire.Aim choose(MeleeFire.Aim incumbent, double energy, double heat) {
        return MeleeFire.choose(policy, incumbent, TURN, SELF, energy, heat, DIRECTION, 1000, 1000);
    }
}
