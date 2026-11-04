import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CrossingGunTest {
    final CrossingGun gun = new CrossingGun();
    final Flight.Pose own = new Flight.Pose(100, 300, 0, 0);
    final Flight.Pose target = new Flight.Pose(242, 300, 90, 8);
    final CrossingGun.Launch launch = CrossingGun.launch(10, own, target, 2, 800, 600);
    Flight.Pose at(double degrees) {
        return new Flight.Pose(100 + 142 * Flight.cos(degrees), 300 + 142 * Flight.sin(degrees), 90, 8);
    }
    void fire() { gun.requested(launch); gun.fired(10, 100, 300, 2); }

    @Test void learnsOnlyFromConfirmedShotsAndBracketingScans() {
        gun.requested(launch);
        assertEquals(0, gun.observe(19, at(20), 18, at(19)));
        gun.fired(10, 100, 300, 2);
        assertEquals(0, gun.observe(18, at(19), 17, at(18)));
        assertEquals(1, gun.observe(19, at(20), 18, at(19)));
        assertEquals(0, gun.observe(20, at(20), 19, at(20)));
        fire();
        assertEquals(0, gun.observe(19, at(20), 17, at(18)));
        fire();
        gun.newRound();
        assertEquals(0, gun.observe(19, at(20), 18, at(19)));
    }
    @Test void mirrorsResolvedBearingsAndRetainsOnlyCompletedExamplesAcrossRounds() {
        for (int i = 0; i < 8; i++) {
            assertTrue(Double.isNaN(gun.aim(launch)));
            fire(); gun.observe(19, at(20), 18, at(19));
        }
        gun.newRound();
        assertEquals(20, gun.aim(launch), launch.escape() * .05);
        var mirror = CrossingGun.launch(10, own, new Flight.Pose(242,300,270,8),2,800,600);
        assertEquals(-20, gun.aim(mirror), launch.escape() * .05);
        var wallward = CrossingGun.launch(10, new Flight.Pose(100, 40, 0, 0),
                new Flight.Pose(242, 40, 270, 8), 2, 800, 600);
        assertTrue(Double.isNaN(gun.aim(wallward)));
    }
}
