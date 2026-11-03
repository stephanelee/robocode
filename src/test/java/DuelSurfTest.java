import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class DuelSurfTest {
    @Test void speedFollowsEngineAccelerationAndDeceleration() {
        assertEquals(1, DuelSurf.nextSpeed(0, 8));
        assertEquals(8, DuelSurf.nextSpeed(7.5, 8));
        assertEquals(6, DuelSurf.nextSpeed(8, 0));
        assertEquals(0, DuelSurf.nextSpeed(1, 0));
        assertEquals(6, DuelSurf.nextSpeed(8, -8));
        assertEquals(-0.5, DuelSurf.nextSpeed(1, -8), 1e-12);
        assertEquals(-6, DuelSurf.nextSpeed(-8, -6));
        assertEquals(8, DuelSurf.nextSpeed(8, 8));
    }

    @Test void travelIsPerpendicularAtPreferredRangeAndTiltsToHoldRange() {
        // Enemy due south (bearing 270) at 450: +1 travels bearing + 90 = east, -1 travels west.
        assertEquals(0, CombatMath.relativeAngle(DuelSurf.travelDirection(500, 500, 500, 50, 1)), 1e-9);
        assertEquals(180, Math.abs(CombatMath.relativeAngle(DuelSurf.travelDirection(500, 500, 500, 50, -1))), 1e-9);
        // Range 800: tilt 30 toward the enemy (south of east). Range 200: tilt -25 away (north of east).
        assertEquals(-30, CombatMath.relativeAngle(DuelSurf.travelDirection(500, 900, 500, 100, 1)), 1e-9);
        assertEquals(25, CombatMath.relativeAngle(DuelSurf.travelDirection(500, 500, 500, 300, 1)), 1e-9);
        // Tilt is capped at 30 degrees.
        assertEquals(-30, CombatMath.relativeAngle(DuelSurf.travelDirection(500, 3000, 500, 0, 1)), 1e-9);
    }

    @Test void withoutThreatsKeepsMovingInTheCurrentDirection() {
        DuelSurf.Command command = DuelSurf.choose(500, 500, 0, 8, 100, 500, 50, List.of(), 1, 1000, 1000);
        assertEquals(1, command.travelSign());
        assertEquals(8, command.targetSpeed());
        assertEquals(0, CombatMath.relativeAngle(command.bodyHeading()), 1e-9);
    }

    @Test void reversesInsteadOfDrivingIntoAWall() {
        // Travelling east 60 units from the east wall: continuing would leave the envelope within the wall horizon.
        DuelSurf.Command command = DuelSurf.choose(940, 500, 0, 8, 100, 940, 50, List.of(), 1, 1000, 1000);
        assertEquals(-1, command.travelSign());
        // Body stays east-facing and drives backward instead of turning around.
        assertEquals(-8, command.targetSpeed());
        assertEquals(0, CombatMath.relativeAngle(command.bodyHeading()), 1e-9);
    }

    @Test void leavesTheLinearPredictionOfAConstantVelocityPath() {
        // Shooter due north at 450 aimed two turns ago at our then-current state (moving east at 8).
        var own = new AimCorridor.OwnState(98, 484, 500, 0, 8);
        double headOn = AimCorridor.headOn(500, 50, own);
        double linear = AimCorridor.linear(500, 50, own, 16.4, 1000, 1000);
        var threat = new DuelSurf.Threat(500, 50, 16.4, 98, headOn, linear);
        double keep = DuelSurf.score(500, 500, 0, 8, DuelSurf.travelDirection(500, 500, 500, 50, 1), 1, 100,
                List.of(threat), 1000, 1000);
        assertTrue(keep < 18, "continuing east must run into the linear aim point, got " + keep);
        DuelSurf.Command command = DuelSurf.choose(500, 500, 0, 8, 100, 500, 50, List.of(threat), 1, 1000, 1000);
        assertNotEquals(1, command.travelSign());
        double chosen = DuelSurf.score(500, 500, 0, 8, command.travelSign() == 0 ? 0
                : DuelSurf.travelDirection(500, 500, 500, 50, command.travelSign()), command.travelSign(), 100,
                List.of(threat), 1000, 1000);
        assertTrue(chosen > 18, "chosen option must clear both aim points by more than a bot radius, got " + chosen);
    }

    @Test void passedThreatsDoNotConstrainMovement() {
        var own = new AimCorridor.OwnState(10, 484, 500, 0, 8);
        var old = new DuelSurf.Threat(500, 50, 16.4, 10, AimCorridor.headOn(500, 50, own),
                AimCorridor.linear(500, 50, own, 16.4, 1000, 1000));
        // At turn 100 the wave radius (~1500) is far past us: every option keeps full clearance.
        assertEquals(DuelSurf.CLEARANCE_CAP, DuelSurf.score(500, 500, 0, 8, 0, 1, 100, List.of(old), 1000, 1000));
        assertEquals(1, DuelSurf.choose(500, 500, 0, 8, 100, 500, 50, List.of(old), 1, 1000, 1000).travelSign());
    }

    @Test void cornerRotatesTravelTowardTheCentreInsteadOfHittingEitherWall() {
        // Near the north-east corner with the enemy to the south-west: both perpendicular directions are blocked.
        for (int sign : new int[]{1, -1}) {
            double raw = DuelSurf.travelDirection(960, 960, 700, 700, sign);
            double safe = DuelSurf.safeTravelDirection(960, 960, 315, 0, 100, 700, 700, sign, 1000, 1000);
            assertTrue(DuelSurf.score(960, 960, 315, 0, raw, sign, 100, List.of(), 1000, 1000) < -500);
            assertTrue(DuelSurf.score(960, 960, 315, 0, safe, sign, 100, List.of(), 1000, 1000) > -500,
                    "sign " + sign + " still blocked at " + safe);
            assertTrue(Math.abs(CombatMath.relativeAngle(safe - raw)) <= 90);
        }
        DuelSurf.Command command = DuelSurf.choose(960, 960, 315, 0, 100, 700, 700, List.of(), 1, 1000, 1000);
        assertNotEquals(0, command.travelSign());
    }

    @Test void openArenaTravelIsNotRotated() {
        assertEquals(DuelSurf.travelDirection(500, 500, 500, 50, 1),
                DuelSurf.safeTravelDirection(500, 500, 0, 8, 100, 500, 50, 1, 1000, 1000), 1e-12);
    }

    @Test void botTouchingAWallMovesAlongOrAwayInsteadOfBackIntoIt() {
        // Stopped against the west wall facing it, stationary enemy to the east (Target-like, no threats).
        DuelSurf.Command command = DuelSurf.choose(18, 500, 180, 0, 100, 600, 500, List.of(), 1, 1000, 1000);
        assertNotEquals(0, command.travelSign());
        double travel = command.targetSpeed() > 0 ? command.bodyHeading() : command.bodyHeading() + 180;
        assertTrue(Math.cos(Math.toRadians(travel)) >= -1e-9, "must not travel west into the wall: " + travel);
        // Driving straight into the wall from the contact point is still rejected.
        assertTrue(DuelSurf.score(18, 500, 180, 0, 180, 1, 100, List.of(), 1000, 1000) < -500);
    }
}
