import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class AimCorridorTest {
    private static final AimCorridor.OwnState STILL = new AimCorridor.OwnState(10, 450, 300, 90, 0);

    @Test void headOnUsesEastZeroBearingToOurAimingPosition() {
        assertEquals(0, AimCorridor.headOn(0, 300, STILL), 1e-9);
        assertEquals(90, AimCorridor.headOn(450, 0, STILL), 1e-9);
    }

    @Test void stationaryLinearAimEqualsHeadOn() {
        assertEquals(0, AimCorridor.linear(0, 300, STILL, 16.4, 800, 600), 1e-9);
    }

    @Test void linearAimMatchesIterativeInterceptForPerpendicularMotion() {
        // Moving north at 8 from 450 units east: first k with 16.4k >= hypot(450, 8k) is 32 (lateral 256).
        var own = new AimCorridor.OwnState(10, 450, 100, 90, 8);
        double expected = Math.toDegrees(Math.atan2(256, 450));
        assertEquals(expected, AimCorridor.linear(0, 100, own, 16.4, 800, 1000), 1e-9);
    }

    @Test void linearPredictionStopsAtTheWallClampLikeTheOpponentModel() {
        var own = new AimCorridor.OwnState(10, 450, 500, 90, 8);
        // Northward path clamps at y = 600 - 18 = 582.
        assertEquals(Math.toDegrees(Math.atan2(82, 450)), AimCorridor.linear(0, 500, own, 11, 800, 600), 1e-9);
    }

    @Test void reversingSpeedPredictsTheOppositeSide() {
        var forward = new AimCorridor.OwnState(10, 450, 300, 90, 8);
        var backward = new AimCorridor.OwnState(10, 450, 300, 90, -8);
        double a = AimCorridor.linear(0, 300, forward, 16.4, 800, 600);
        double b = AimCorridor.linear(0, 300, backward, 16.4, 800, 600);
        assertTrue(a > 0 && b < 0);
        assertEquals(a, -b, 1e-9);
    }

    @Test void dangerIsOneOnTheRayAndFallsToZeroAtTheCorridorEdge() {
        assertEquals(1, AimCorridor.danger(0, 0, 450, 0, 450, 0), 1e-12);
        double corridor = 2 * Math.toDegrees(Math.atan(18.0 / 450));
        double inside = Math.toRadians(corridor * 0.5);
        assertEquals(0.5, AimCorridor.danger(0, 0, 450, 0, 450 * Math.cos(inside), 450 * Math.sin(inside)), 1e-9);
        double edge = Math.toRadians(corridor);
        assertEquals(0, AimCorridor.danger(0, 0, 450, 0, 450 * Math.cos(edge), 450 * Math.sin(edge)), 1e-12);
        // Sixty units of lateral displacement at range 450 is outside both corridors.
        assertEquals(0, AimCorridor.danger(0, 0, 450, 0, 450, 60), 1e-12);
    }

    @Test void dangerWrapsAnglesAndRejectsInvalidInputs() {
        assertEquals(1, AimCorridor.danger(0, 0, 450, 180, -450, 0), 1e-9);
        assertEquals(1, AimCorridor.danger(0, 0, 450, -180, -450, 0), 1e-9);
        assertEquals(0, AimCorridor.danger(0, 0, 450, 0, 0, 0));
        assertEquals(0, AimCorridor.danger(0, 0, 450, Double.NaN, 450, 0));
        assertEquals(0, AimCorridor.danger(0, 0, Double.NaN, 0, 450, 0));
    }

    @Test void closeRangeCorridorIsWiderThanLongRange() {
        double nearOffset = Math.toRadians(8);
        assertTrue(AimCorridor.danger(0, 0, 100, 0, 100 * Math.cos(nearOffset), 100 * Math.sin(nearOffset)) > 0);
        assertEquals(0, AimCorridor.danger(0, 0, 500, 0, 500 * Math.cos(nearOffset), 500 * Math.sin(nearOffset)));
    }

    @Test void waveWithoutAimingStateAddsNoDanger() {
        EnemyWave wave = new EnemyWave(0, 0, 16.4, 0, 10, 7);
        assertEquals(0, wave.aimDanger(450, 0));
        wave.setAims(0, 30, 450);
        assertEquals(1, wave.aimDanger(450, 0), 1e-12);
        double linear = Math.toRadians(30);
        assertEquals(1, wave.aimDanger(450 * Math.cos(linear), 450 * Math.sin(linear)), 1e-9);
        double between = Math.toRadians(15);
        assertEquals(0, wave.aimDanger(450 * Math.cos(between), 450 * Math.sin(between)));
    }
}
