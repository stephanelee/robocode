import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class HitAnglesTest {
    private final HitAngles angles = new HitAngles();
    private final HitAngles.Wave wave = new HitAngles.Wave(10, 100, 100, 350, 375, 2);
    private HitAngles.Label label(List<HitAngles.Wave> waves, int turn, double heading) {
        double distance = 14 * 10;
        return angles.observe(waves, turn, 100 + distance * Flight.cos(heading),
                100 + distance * Flight.sin(heading), heading, 2);
    }
    @Test void requireUniqueLaunchOnCorrectTick() {
        assertEquals(HitAngles.Label.UNMATCHED, label(List.of(wave), 20, 10));
        assertEquals(HitAngles.Label.AMBIGUOUS, label(List.of(wave, wave), 19, 10));
        assertEquals(HitAngles.Label.ACCEPTED, label(List.of(wave), 19, 10));
    }
    @Test void learnedDirectionMirrorsWhenLateralMotionReverses() {
        var reverse = new HitAngles.Wave(10, 100, 100, 350, 325, 2);
        double heading = 350 - wave.escape() * .625;
        double dx = 300 * Flight.cos(heading), dy = 300 * Flight.sin(heading);
        double before = angles.risk(reverse, dx, dy);
        assertEquals(HitAngles.Label.ACCEPTED, label(List.of(wave), 19, 350 + wave.escape() * .625));
        assertTrue(angles.risk(reverse, dx, dy) > before);
    }
    @Test void ambiguousHitDoesNotChangeRisk() {
        double before = angles.risk(wave, 140, 20);
        label(List.of(wave, wave), 19, 10);
        assertEquals(before, angles.risk(wave, 140, 20));
    }
}
