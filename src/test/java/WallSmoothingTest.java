import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class WallSmoothingTest {
    @Test void findsSafeDirectionBeyondTheOldHalfTurnSearch() {
        double heading = WallSmoothing.heading(120, 120, 800, 600, 80, 80, 125, 1);
        assertEquals(330, heading, 1e-9);
        assertInnerEndpoint(120, 120, heading);
    }

    @Test void negativeRotationFindsTheFirstSafeHeadingInItsOwnDirection() {
        double heading = WallSmoothing.heading(120, 120, 800, 600, 80, 80, 125, -1);
        assertEquals(120, heading, 1e-9);
        assertInnerEndpoint(120, 120, heading);
    }

    @Test void keepsAnAlreadySafeDesiredHeading() {
        assertEquals(37, WallSmoothing.heading(400, 300, 800, 600, 80, 80, 37, 1));
        assertEquals(270, WallSmoothing.heading(400, 300, 800, 600, 80, 80, -90, -1));
    }

    @Test void unreachableCornerEnvelopeFallsBackTowardCentre() {
        double heading = WallSmoothing.heading(18, 18, 800, 600, 80, 80, 225, 1);
        double expected = Math.toDegrees(Math.atan2(300 - 18, 400 - 18));
        assertEquals(expected, heading, 1e-9);
        double[] endpoint = CombatMath.project(18, 18, heading, 80, 0, 1);
        assertTrue(endpoint[0] > 18 && endpoint[1] > 18);
        assertTrue(endpoint[0] < 800 && endpoint[1] < 600);
        assertTrue(Math.hypot(endpoint[0] - 400, endpoint[1] - 300) < Math.hypot(382, 282));
    }

    @Test void headingsStayFiniteAndNormalizedAtEachArenaCorner() {
        for (double x : new double[]{18, 782}) {
            for (double y : new double[]{18, 582}) {
                for (int rotation : new int[]{-1, 1}) {
                    double heading = WallSmoothing.heading(x, y, 800, 600, 80, 80, 1085, rotation);
                    assertTrue(Double.isFinite(heading) && heading >= 0 && heading < 360);
                    double[] endpoint = CombatMath.project(x, y, heading, 80, 0, 1);
                    assertTrue(endpoint[0] >= 0 && endpoint[0] <= 800);
                    assertTrue(endpoint[1] >= 0 && endpoint[1] <= 600);
                }
            }
        }
    }

    @Test void rejectsInvalidGeometryInsteadOfReturningAnUncheckedHeading() {
        assertThrows(IllegalArgumentException.class,
                () -> WallSmoothing.heading(18, 18, 800, 600, 80, 80, Double.NaN, 1));
        assertThrows(IllegalArgumentException.class,
                () -> WallSmoothing.heading(18, 18, 800, 600, 80, 80, 125, 0));
    }

    private static void assertInnerEndpoint(double x, double y, double heading) {
        double[] endpoint = CombatMath.project(x, y, heading, 80, 0, 1);
        assertTrue(endpoint[0] >= 80 - 1e-9 && endpoint[0] <= 720 + 1e-9);
        assertTrue(endpoint[1] >= 80 - 1e-9 && endpoint[1] <= 520 + 1e-9);
    }
}
