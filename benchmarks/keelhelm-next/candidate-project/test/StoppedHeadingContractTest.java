import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

/** Source-contract fixtures only; no server, bot, opponent or competitive result. */
class StoppedHeadingContractTest {
    private static final double EPS = 1e-10;

    // Engine contract for an already legal stationary center: speed0, no translation,
    // then the requested body turn clipped to +/-10 degrees. Kept independent of Phys.step.
    private static Phys.Pose engineStopped(Phys.Pose p, double turn) {
        double limited = Math.max(-10, Math.min(10, turn));
        return new Phys.Pose(p.x(), p.y(), p.h() + limited, 0);
    }

    @Test void multiTurnStoppedSuffixUsesEmittedTurnForBothSidesAndSpeedSigns() {
        for (int dir : new int[]{-1, 1}) {
            for (double speed : new double[]{-8, 0, 8}) {
                for (int length : new int[]{1, 3, 9}) {
                    Phys.Pose actual = new Phys.Pose(400, 300, 0, speed);
                    Phys.Pose expected = actual;
                    for (int tick = 0; tick < length; tick++) {
                        double turn = Surfer.orbit(expected, 400, 0, dir, 800, 600)[0];
                        expected = engineStopped(expected, turn);
                        actual = Surfer.stoppedStep(actual, 400, 0, dir, 800, 600);
                        assertPose(expected, actual);
                        assertEquals(400, actual.x(), EPS);
                        assertEquals(300, actual.y(), EPS);
                        assertEquals(0, actual.v(), 0);
                    }
                    assertNotEquals(0, actual.h(), "legacy frozen heading would fail");
                }
            }
        }
    }

    @Test void rotatingStopEndpointChangesSubsequentMovingEscapeAsEngineRequires() {
        for (int dir : new int[]{-1, 1}) {
            for (double speed : new double[]{-8, 8}) {
                Phys.Pose stopped = new Phys.Pose(400, 300, 0, speed);
                for (int i = 0; i < 3; i++)
                    stopped = Surfer.stoppedStep(stopped, 400, 0, dir, 800, 600);
                double[] command = Surfer.orbit(stopped, 400, 0, dir, 800, 600);
                Phys.Pose moved = Phys.step(stopped, command[0], command[1], 800, 600);
                // From rest, the first speed is +/-1; motion precedes body turn.
                double newSpeed = Math.copySign(1, command[1]);
                assertEquals(stopped.x() + Math.cos(Math.toRadians(stopped.h())) * newSpeed, moved.x(), EPS);
                assertEquals(stopped.y() + Math.sin(Math.toRadians(stopped.h())) * newSpeed, moved.y(), EPS);
                assertEquals(newSpeed, moved.v(), EPS);
                assertEquals(stopped.h() + Math.max(-9.25, Math.min(9.25, command[0])), moved.h(), EPS);
                Phys.Pose legacy = new Phys.Pose(stopped.x(), stopped.y(), 0, 0);
                double[] oldCommand = Surfer.orbit(legacy, 400, 0, dir, 800, 600);
                Phys.Pose oldEscape = Phys.step(legacy, oldCommand[0], oldCommand[1], 800, 600);
                assertTrue(Math.abs(moved.y() - oldEscape.y()) > 0.1,
                        "stopped endpoint heading must feed the moving continuation");
            }
        }
    }

    @Test void legalWallAndCornerStopsRetainCenterWhileTurning() {
        double[][] centers = {{18, 18}, {18, 582}, {782, 18}, {782, 582}, {18, 300}, {782, 300}};
        for (double[] xy : centers) {
            for (int dir : new int[]{-1, 1}) {
                for (double speed : new double[]{-8, 8}) {
                    Phys.Pose actual = new Phys.Pose(xy[0], xy[1], 37, speed);
                    Phys.Pose expected = actual;
                    for (int i = 0; i < 7; i++) {
                        expected = engineStopped(expected, Surfer.orbit(expected, 400, 300, dir, 800, 600)[0]);
                        actual = Surfer.stoppedStep(actual, 400, 300, dir, 800, 600);
                        assertPose(expected, actual);
                    }
                }
            }
        }
    }

    @Test void ordinaryMovingCommandStillFollowsOriginalNativeContract() {
        // This path never enters the changed stopped suffix.
        Phys.Pose p = Phys.step(new Phys.Pose(400, 300, 0, 8), 10, 8, 800, 600);
        assertPose(new Phys.Pose(408, 300, 4, 8), p);
        Phys.Pose reverse = Phys.step(new Phys.Pose(400, 300, 0, 8), -10, -8, 800, 600);
        assertPose(new Phys.Pose(403, 300, -7.75, 3), reverse);
    }

    @Test void candidateCountersArePerRoundAndExerciseActualFirstLegLoop() {
        Surfer s = new Surfer();
        double[] density = new double[Surfer.BINS];
        java.util.Arrays.fill(density, 1.0 / Surfer.BINS);
        s.waves.add(new Surfer.Wave(0, 200, 300, 2, 0, 1, new double[Track.DIMS], density));
        s.waves.add(new Surfer.Wave(8, 200, 300, 2, 0, 1, new double[Track.DIMS], density));
        assertNotNull(s.plan(new Phys.Pose(400, 300, 0, 8), 10, 200, 300, 800, 600, 100));
        assertTrue(s.stoppedRolloutSteps > 0);
        assertTrue(s.headingChangedSteps > 0);
        assertTrue(s.secondWaveEvaluations > 0);
        assertTrue(s.headingChangedSteps <= s.stoppedRolloutSteps);
        s.executedStopNow = 5;
        s.newRound();
        assertEquals(0, s.stoppedRolloutSteps);
        assertEquals(0, s.headingChangedSteps);
        assertEquals(0, s.secondWaveEvaluations);
        assertEquals(0, s.executedStopNow);
    }

    private static void assertPose(Phys.Pose expected, Phys.Pose actual) {
        assertEquals(expected.x(), actual.x(), EPS);
        assertEquals(expected.y(), actual.y(), EPS);
        assertEquals(expected.h(), actual.h(), EPS);
        assertEquals(expected.v(), actual.v(), EPS);
    }
}
