import java.util.List;
import java.util.SplittableRandom;

/** Short-range interception pressure, not AIS's shot-reactive long-range orbit. */
final class PressureMove {
    static final double RANGE = 140;
    private PressureMove() {}

    static int phaseDuration(SplittableRandom random) { return random.nextInt(20, 46); }

    static DuelSurf.Command choose(double x, double y, double heading, double speed, int now,
                                  DuelGun.Scan enemy, double enemyTurnRate, int flank,
                                  double width, double height) {
        double desired;
        if (enemy == null || now - enemy.turn() > 2) {
            desired = CombatMath.bearing(width / 2 - x, height / 2 - y);
        } else {
            double distance = Math.hypot(enemy.x() - x, enemy.y() - y);
            int lead = distance > 260 ? Math.min(18, (int) (distance / 15)) : 2;
            double[] point = DuelGun.forecast(enemy, Math.max(0, now - enemy.turn()) + lead, 3,
                    enemyTurnRate, width, height);
            double approach = Math.max(15, Math.min(135, 90 + (RANGE - distance) * 0.45));
            desired = CombatMath.bearing(point[0] - x, point[1] - y) + (flank < 0 ? -1 : 1) * approach;
        }
        // Preserve pressure/flanking first; rotate only when its real turning path is unsafe.
        for (double offset : new double[]{0, 15, -15, 30, -30, 60, -60, 90, -90}) {
            var command = DuelSurf.command(heading, desired + offset, flank < 0 ? -1 : 1);
            if (DuelSurf.score(x, y, heading, speed, command, now, List.of(), width, height) > -500)
                return command;
        }
        // In1.4 a stop is immediate; a reversal at speed8 still first travels+3.
        double center = CombatMath.bearing(width / 2 - x, height / 2 - y);
        var orientation = DuelSurf.command(heading, center, 1);
        return new DuelSurf.Command(orientation.bodyHeading(), 0, 0);
    }
}
