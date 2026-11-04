import static org.junit.jupiter.api.Assertions.*;
import java.util.ArrayList;
import org.junit.jupiter.api.Test;

class SurfPlanTest {
    @Test void speculativeShotOnlyFillsAnOpenSlotWhenGunIsReady() {
        var waves = new ArrayList<HitAngles.Wave>();
        var enemy = new Flight.Pose(150, 300, 90, 8);
        var own = new Flight.Pose(650, 300, 90, 8);
        var planner = new SurfPlan(new HitAngles(), enemy, 800, 600);
        int ready = SurfPlan.readyAfter(31, 1.95, .1);
        assertEquals(46, ready);
        assertEquals(45, SurfPlan.readyAfter(31, 1, .1));
        assertFalse(planner.choose(own, 44, 1, waves, ready, 1.95).warned());
        waves.add(new HitAngles.Wave(31, 150, 300, 0, 23, 1.95));
        var prediction = planner.choose(own, 45, 1, waves, ready, 1.95);
        assertTrue(prediction.warned());
        assertEquals(2, prediction.fronts());
        assertEquals(1, waves.size());
        waves.add(new HitAngles.Wave(43, 150, 300, 0, 23, .5));
        assertFalse(planner.choose(own, 45, 1, waves, ready, 1.95).warned());
    }
}
