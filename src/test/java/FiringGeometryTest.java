import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class FiringGeometryTest {
    @Test void distantMisalignmentIsBlockedWhileNearToleranceStays() {
        assertFalse(FiringGeometry.aligned(9, 500));
        assertTrue(FiringGeometry.aligned(2, 500));
        assertTrue(FiringGeometry.aligned(9, 100));
        assertEquals(10, FiringGeometry.tolerance(100));
        assertEquals(10, FiringGeometry.tolerance(0));
    }

    @Test void strictBoundaryAndSignsAgree() {
        double limit = FiringGeometry.tolerance(500);
        assertFalse(FiringGeometry.aligned(limit, 500));
        assertFalse(FiringGeometry.aligned(-limit, 500));
        assertTrue(FiringGeometry.aligned(Math.nextDown(limit), 500));
        assertTrue(FiringGeometry.aligned(-Math.nextDown(limit), 500));
        assertFalse(FiringGeometry.aligned(10, 50));
    }

    @Test void invalidInputsFailClosed() {
        for (double range : new double[]{-1, Double.NaN, Double.POSITIVE_INFINITY})
            assertFalse(FiringGeometry.aligned(0, range));
        assertFalse(FiringGeometry.aligned(Double.NaN, 500));
        assertFalse(FiringGeometry.aligned(Double.POSITIVE_INFINITY, 500));
    }

    @Test void actualHandlerKeepsGunCommandsButBlocksDistantRequest() throws Exception {
        var near = scenario("near", 100, 9);
        var far = scenario("far", 500, 9);
        var aligned = scenario("far-aligned", 500, 2);
        var nearResult = TargetingCharacterizationTest.record(near);
        var farResult = TargetingCharacterizationTest.record(far);
        var alignedResult = TargetingCharacterizationTest.record(aligned);
        assertEquals(nearResult.get("gun"), farResult.get("gun"));
        assertEquals(1, nearResult.getAsJsonArray("fire").size());
        assertEquals(0, farResult.getAsJsonArray("fire").size());
        assertFalse(farResult.get("waveChanged").getAsBoolean());
        assertTrue(farResult.get("wavePresent").getAsBoolean(), "keep existing pending wave");
        assertEquals(1, alignedResult.getAsJsonArray("fire").size());
        assertEquals(TargetingCharacterizationTest.bits(700.0 / 550.0),
                alignedResult.getAsJsonArray("fire").get(0).getAsString());
    }

    private static TargetingCharacterizationTest.Scenario scenario(String name, double range, double gun) {
        return new TargetingCharacterizationTest.Scenario(name, 100, 300, 100 + range, 300,
                0, 0, 0, 0, 0, gun, 100, 0, 40, 23, true, true, true, true);
    }
}
