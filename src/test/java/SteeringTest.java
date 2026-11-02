import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SteeringTest {
    @Test void directlyBehindUsesReverseWithoutTurningTheBody() {
        Steering.Command west = Steering.toward(0, 180, 8, 8);
        assertEquals(0, west.turnRate());
        assertEquals(-8, west.targetSpeed());
        Steering.Command south = Steering.toward(90, 270, 4, 0);
        assertEquals(0, south.turnRate());
        assertEquals(-4, south.targetSpeed());
    }

    @Test void rearDiagonalUsesTheShortBodyTurnAndReversedSpeed() {
        assertEquals(new Steering.Command(-4, -8), Steering.toward(0, 170, 8, 8));
        assertEquals(new Steering.Command(4, -8), Steering.toward(0, 190, 8, 8));
        assertEquals(new Steering.Command(-10, -8), Steering.toward(0, 170, 8, 0));
    }

    @Test void wrappedForwardHeadingsAndPerpendicularTiesRemainForward() {
        assertEquals(new Steering.Command(4, 8), Steering.toward(359, 1, 8, 8));
        assertEquals(new Steering.Command(-4, 8), Steering.toward(1, 359, 8, 8));
        assertEquals(8, Steering.toward(0, 90, 8, 8).targetSpeed());
        assertEquals(8, Steering.toward(0, 270, 8, 8).targetSpeed());
        assertEquals(-8, Steering.toward(0, 90.001, 8, 8).targetSpeed());
    }

    @Test void commandHonorsTurnLimitsForForwardAndReverseCurrentSpeed() {
        for (int current = -8; current <= 8; current++) {
            for (int body = 0; body < 360; body += 15) {
                for (int desired = 0; desired < 360; desired += 15) {
                    Steering.Command command = Steering.toward(body, desired, 8, current);
                    assertTrue(Math.abs(command.turnRate()) <= 10 - 0.75 * Math.abs(current));
                    assertEquals(8, Math.abs(command.targetSpeed()));
                    double selectedBody = desired + (command.targetSpeed() < 0 ? 180 : 0);
                    assertTrue(Math.abs(CombatMath.relativeAngle(selectedBody - body)) <= 90);
                }
            }
        }
    }
}
