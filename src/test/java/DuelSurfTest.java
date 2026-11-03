import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class DuelSurfTest {
    private static final double E = 1e-12;

    @Test void speedMatchesPinnedEngineAccelerationAndDeceleration() {
        // Values from Tank Royale 1.4.0 rules/math.kt calcNewBotSpeed.
        assertEquals(1, DuelSurf.nextSpeed(0, 8), E);
        assertEquals(-1, DuelSurf.nextSpeed(0, -8), E);
        assertEquals(8, DuelSurf.nextSpeed(7.5, 8), E);
        assertEquals(8, DuelSurf.nextSpeed(8, 8), E);
        assertEquals(6, DuelSurf.nextSpeed(8, 3), E);
        assertEquals(-6, DuelSurf.nextSpeed(-8, -6), E);
    }

    @Test void stoppingIsImmediateInBothDirections() {
        assertEquals(0, DuelSurf.nextSpeed(8, 0), E);
        assertEquals(0, DuelSurf.nextSpeed(-8, 0), E);
        assertEquals(0, DuelSurf.nextSpeed(1, 0), E);
    }

    @Test void reversalFollowsTheEngineZeroCrossingRule() {
        assertEquals(3, DuelSurf.nextSpeed(8, -8), E);
        assertEquals(-3, DuelSurf.nextSpeed(-8, 8), E);
        assertEquals(0.5, DuelSurf.nextSpeed(3, -8), E);
        assertEquals(0, DuelSurf.nextSpeed(2, -8), E);
        assertEquals(-0.5, DuelSurf.nextSpeed(1, -8), E);
        assertEquals(0.5, DuelSurf.nextSpeed(-1, 8), E);
    }

    @Test void turnLimitFollowsTheEngine() {
        assertEquals(10, DuelSurf.maxTurn(0), E);
        assertEquals(4, DuelSurf.maxTurn(8), E);
        assertEquals(4, DuelSurf.maxTurn(-8), E);
    }

    @Test void travelIsPerpendicularAtPreferredRangeAndTiltsToHoldRange() {
        // Enemy due south (bearing 270) at 450: +1 travels bearing + 90 = east, -1 travels west.
        assertEquals(0, CombatMath.relativeAngle(DuelSurf.travelDirection(500, 500, 500, 50, 1)), 1e-9);
        assertEquals(180, Math.abs(CombatMath.relativeAngle(DuelSurf.travelDirection(500, 500, 500, 50, -1))), 1e-9);
        assertEquals(-30, CombatMath.relativeAngle(DuelSurf.travelDirection(500, 900, 500, 100, 1)), 1e-9);
        assertEquals(25, CombatMath.relativeAngle(DuelSurf.travelDirection(500, 500, 500, 300, 1)), 1e-9);
        assertEquals(-30, CombatMath.relativeAngle(DuelSurf.travelDirection(500, 3000, 500, 0, 1)), 1e-9);
    }

    @Test void commandDrivesBackwardInsteadOfTurningAround() {
        DuelSurf.Command back = DuelSurf.command(0, 180, -1);
        assertEquals(0, back.bodyHeading(), 1e-9);
        assertEquals(-8, back.targetSpeed(), E);
        DuelSurf.Command stop = DuelSurf.command(45, 123, 0);
        assertEquals(45, stop.bodyHeading(), 1e-9);
        assertEquals(0, stop.targetSpeed(), E);
    }

    @Test void withoutThreatsKeepsMovingInTheCurrentDirection() {
        DuelSurf.Command command = DuelSurf.choose(500, 500, 0, 8, 100, 500, 50, List.of(), 1, 1000, 1000);
        assertEquals(1, command.travelSign());
        assertEquals(8, command.targetSpeed(), E);
        assertEquals(0, CombatMath.relativeAngle(command.bodyHeading()), 1e-9);
    }

    @Test void reversesWellBeforeTheWall() {
        // 120 units from the east wall travelling east: continuing would enter the envelope within the horizon.
        DuelSurf.Command command = DuelSurf.choose(880, 500, 0, 8, 100, 880, 50, List.of(), 1, 1000, 1000);
        assertEquals(-1, command.travelSign());
        assertEquals(-8, command.targetSpeed(), E);
        assertEquals(0, CombatMath.relativeAngle(command.bodyHeading()), 1e-9);
    }

    @Test void reviewNearWallCounterexampleDoesNotHitTheWall() {
        // PR #38 review: at x=780 heading east at speed 8 in an 800x600 arena, a reversal still moves +3 first
        // (x=783 > legal 782). The controller must choose a command whose first engine step stays legal.
        DuelSurf.Command command = DuelSurf.choose(780, 300, 0, 8, 100, 500, 300, List.of(), 1, 800, 600);
        double next = DuelSurf.nextSpeed(8, command.targetSpeed());
        assertTrue(780 + next <= 782, "first step must stay inside the legal wall, chose " + command);
        assertTrue(DuelSurf.score(780, 300, 0, 8, command, 100, List.of(), 800, 600) > -500);
        // The reversal it previously chose is now recognised as a wall hit on the first step.
        assertEquals(-999, DuelSurf.score(780, 300, 0, 8, new DuelSurf.Command(87, -8, -1), 100, List.of(), 800, 600), E);
    }

    @Test void simulationMovesAlongTheCurrentHeadingBeforeTurning() {
        // Facing north at x=30, asked to drive west: the first step still goes north, so it is not a wall hit.
        assertTrue(DuelSurf.score(30, 500, 90, 8, new DuelSurf.Command(180, 8, 1), 100, List.of(), 1000, 1000) < -500,
                "turning west near the west wall must eventually be rejected");
        assertTrue(DuelSurf.score(30, 500, 90, 8, new DuelSurf.Command(90, 8, 1), 100, List.of(), 1000, 1000) > -500);
    }

    @Test void leavesTheLinearPredictionOfAConstantVelocityPath() {
        var own = new AimCorridor.OwnState(98, 484, 500, 0, 8);
        double headOn = AimCorridor.headOn(500, 50, own);
        double linear = AimCorridor.linear(500, 50, own, 16.4, 1000, 1000);
        var threat = new DuelSurf.Threat(500, 50, 16.4, 98, headOn, linear);
        double keep = DuelSurf.score(500, 500, 0, 8, DuelSurf.command(0, 0, 1), 100, List.of(threat), 1000, 1000);
        assertTrue(keep < 18, "continuing east must run into the linear aim point, got " + keep);
        DuelSurf.Command command = DuelSurf.choose(500, 500, 0, 8, 100, 500, 50, List.of(threat), 1, 1000, 1000);
        assertNotEquals(1, command.travelSign());
        double chosen = DuelSurf.score(500, 500, 0, 8, command, 100, List.of(threat), 1000, 1000);
        assertTrue(chosen > 18, "chosen option must clear both aim points by more than a bot radius, got " + chosen);
    }

    @Test void passedThreatsDoNotConstrainMovement() {
        var own = new AimCorridor.OwnState(10, 484, 500, 0, 8);
        var old = new DuelSurf.Threat(500, 50, 16.4, 10, AimCorridor.headOn(500, 50, own),
                AimCorridor.linear(500, 50, own, 16.4, 1000, 1000));
        assertEquals(DuelSurf.CLEARANCE_CAP,
                DuelSurf.score(500, 500, 0, 8, DuelSurf.command(0, 0, 1), 100, List.of(old), 1000, 1000), E);
        assertEquals(1, DuelSurf.choose(500, 500, 0, 8, 100, 500, 50, List.of(old), 1, 1000, 1000).travelSign());
    }

    @Test void cornerRotatesTravelTowardTheCentreInsteadOfHittingEitherWall() {
        for (int sign : new int[]{1, -1}) {
            double raw = DuelSurf.travelDirection(960, 960, 700, 700, sign);
            double safe = DuelSurf.safeTravelDirection(960, 960, 315, 0, 100, 700, 700, sign, 1000, 1000);
            assertTrue(DuelSurf.score(960, 960, 315, 0, DuelSurf.command(315, raw, sign), 100, List.of(), 1000, 1000) < -500);
            assertTrue(DuelSurf.score(960, 960, 315, 0, DuelSurf.command(315, safe, sign), 100, List.of(), 1000, 1000) > -500,
                    "sign " + sign + " still blocked at " + safe);
            assertTrue(Math.abs(CombatMath.relativeAngle(safe - raw)) <= 90);
        }
        assertNotEquals(0, DuelSurf.choose(960, 960, 315, 0, 100, 700, 700, List.of(), 1, 1000, 1000).travelSign());
    }

    @Test void openArenaTravelIsNotRotated() {
        assertEquals(DuelSurf.travelDirection(500, 500, 500, 50, 1),
                DuelSurf.safeTravelDirection(500, 500, 0, 8, 100, 500, 50, 1, 1000, 1000), 1e-12);
    }

    @Test void botTouchingAWallMovesAlongOrAwayInsteadOfBackIntoIt() {
        DuelSurf.Command command = DuelSurf.choose(18, 500, 180, 0, 100, 600, 500, List.of(), 1, 1000, 1000);
        assertNotEquals(0, command.travelSign());
        double travel = command.targetSpeed() > 0 ? command.bodyHeading() : command.bodyHeading() + 180;
        assertTrue(Math.cos(Math.toRadians(travel)) >= -1e-9, "must not travel west into the wall: " + travel);
        assertTrue(DuelSurf.score(18, 500, 180, 0, new DuelSurf.Command(180, 8, 1), 100, List.of(), 1000, 1000) < -500);
    }
}
