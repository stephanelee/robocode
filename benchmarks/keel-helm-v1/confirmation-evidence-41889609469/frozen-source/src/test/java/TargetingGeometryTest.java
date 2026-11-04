import dev.robocode.tankroyale.botapi.IBaseBot;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class TargetingGeometryTest {
    /** Frozen accepted math, using real pinned API default distance/bearing operations. */
    static TargetingGeometry.Aim accepted(TargetingGeometry.Snapshot s) {
        IBaseBot api = mock(IBaseBot.class, CALLS_REAL_METHODS);
        doReturn(s.shooterX()).when(api).getX(); doReturn(s.shooterY()).when(api).getY();
        double maxAngle = Math.toDegrees(Math.asin(Math.min(1.0, 8.0 / s.bulletSpeed())));
        double latDir = Math.signum(s.lateralVelocity());
        if (latDir == 0) latDir = 1;
        double ticks = s.initialDistance() / s.bulletSpeed();
        double[] prediction = null;
        for (int i = 0; i < 5; i++) {
            prediction = CombatMath.project(s.targetX(), s.targetY(), s.targetDirection(), s.targetSpeed(), s.targetTurnRate(),
                    ticks + s.currentTurn() - s.lastScanTurn());
            ticks = api.distanceTo(prediction[0], prediction[1]) / s.bulletSpeed();
        }
        double predX = Math.max(18, Math.min(s.arenaWidth() - 18, prediction[0]));
        double predY = Math.max(18, Math.min(s.arenaHeight() - 18, prediction[1]));
        double baseAngle = api.directionTo(predX, predY);
        double aimAngle = baseAngle + latDir * s.guessFactor() * maxAngle;
        double gunTurn = aimAngle - s.gunDirection();
        while (gunTurn > 180.0) gunTurn -= 360.0;
        while (gunTurn < -180.0) gunTurn += 360.0;
        return new TargetingGeometry.Aim(predX, predY, baseAngle, aimAngle, latDir, gunTurn,
                Math.max(-20, Math.min(20, gunTurn)));
    }

    @Test void everyReturnedFieldMatchesAcceptedArithmeticBitForBit() {
        for (var fixture : TargetingCharacterizationTest.scenarios()) {
            double distance = Math.hypot(fixture.tx() - fixture.sx(), fixture.ty() - fixture.sy());
            double power = fixture.energy() < 12 ? 0 : Math.max(0.5, Math.min(3, 700.0 / (distance + 50)));
            if (fixture.energy() >= 12 && fixture.energy() < 25) power = Math.min(power, 0.8);
            else if (fixture.energy() >= 25 && fixture.energy() < 50) power = Math.min(power, 1.5);
            var snapshot = new TargetingGeometry.Snapshot(fixture.sx(), fixture.sy(), 800, 600, 1000,
                    fixture.gun(), fixture.tx(), fixture.ty(), fixture.heading(), fixture.speed(), fixture.turnRate(),
                    1000 - fixture.age(), distance, 20.0 - 3.0 * (power > 0 ? power : 1),
                    fixture.bin() / 46.0 * 2.0 - 1.0, fixture.lateral());
            var expected = accepted(snapshot);
            var actual = TargetingGeometry.calculate(snapshot);
            assertEquals(expected, actual, fixture.name());
            for (double[] pair : new double[][]{{expected.x(), actual.x()}, {expected.y(), actual.y()},
                    {expected.baseAngle(), actual.baseAngle()}, {expected.aimAngle(), actual.aimAngle()},
                    {expected.lateralDirection(), actual.lateralDirection()}, {expected.gunTurn(), actual.gunTurn()},
                    {expected.gunTurnRate(), actual.gunTurnRate()}})
                assertEquals(Double.doubleToRawLongBits(pair[0]), Double.doubleToRawLongBits(pair[1]), fixture.name());
        }
    }
}
