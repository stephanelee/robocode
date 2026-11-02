import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class WallSmoothTest {
    private static boolean safe(double x, double y, double h) {
        double tx = x + Math.cos(Math.toRadians(h)) * 80, ty = y + Math.sin(Math.toRadians(h)) * 80;
        return tx > 80 && tx < 720 && ty > 80 && ty < 520;
    }

    @Test void safeHeadingIsReturnedUnchanged() {
        assertEquals(30, CombatMath.nearestSafeHeading(400, 300, 30, 800, 600, 80), 1e-9);
    }

    @Test void topWallPicksNearestInwardHeadingNotTheOppositeHalfTurn() {
        // Recorded pinning case: (275,582) wanting 330; old one-way half search returned 150 (into the wall).
        double h = CombatMath.nearestSafeHeading(275, 582, 330, 800, 600, 80);
        assertEquals(305, h, 1e-9);
        assertTrue(safe(275, 582, h));
    }

    @Test void eastWallSearchesTheShorterDirection() {
        // Recorded case: (782,402) wanting 243; old search returned 62/63 (into the wall).
        double h = CombatMath.nearestSafeHeading(782, 402, 243, 800, 600, 80);
        assertEquals(218, h, 1e-9);
        assertTrue(safe(782, 402, h));
    }

    @Test void searchIsSymmetricAndNormalized() {
        double up = CombatMath.nearestSafeHeading(400, 582, 10, 800, 600, 80);
        double down = CombatMath.nearestSafeHeading(400, 582, 170, 800, 600, 80);
        assertTrue(safe(400, 582, up) && safe(400, 582, down));
        assertEquals(305, up, 1e-9);
        assertEquals(235, down, 1e-9);
        double wrapped = CombatMath.nearestSafeHeading(400, 300, -30, 800, 600, 80);
        assertEquals(330, wrapped, 1e-9);
    }

    @Test void cornerWithoutSafeHeadingPointsToCentre() {
        double h = CombatMath.nearestSafeHeading(18, 18, 200, 800, 600, 80);
        assertEquals(Math.toDegrees(Math.atan2(300 - 18, 400 - 18)), h, 1e-9);
    }
}
