import java.util.List;
import java.util.Random;

/**
 * Pre-game JVM warm-up for the per-turn decision code. Runs before the bot connects, so class loading and JIT
 * compilation of the hot paths happen outside the server's turn timeout.
 * <p>
 * Uses only pure static functions and throwaway instances with their own fixed seed. It never touches the bot's
 * agent, RNG, statistics, files or network, so it cannot change any gameplay decision.
 */
final class Warmup {
    private Warmup() {}

    /** Exercises the hot paths {@code iterations} times; returns a checksum so the work cannot be eliminated. */
    static double run(int iterations) {
        double checksum = 0;
        Random rng = new Random(20261103);
        MLP network = new MLP(14, 32, 6, new Random(1));
        double[] features = new double[14];
        for (int i = 0; i < iterations; i++) {
            double x = 100 + rng.nextDouble() * 600, y = 100 + rng.nextDouble() * 400;
            double ex = 100 + rng.nextDouble() * 600, ey = 100 + rng.nextDouble() * 400;
            double heading = rng.nextDouble() * 360, speed = rng.nextDouble() * 16 - 8;
            int turn = 100 + i;

            var own = new AimCorridor.OwnState(turn - 3, x, y, heading, speed);
            double headOn = AimCorridor.headOn(ex, ey, own);
            double linear = AimCorridor.linear(ex, ey, own, 16.4, 800, 600);
            var threats = List.of(new DuelSurf.Threat(ex, ey, 16.4, turn - 3, headOn, linear));
            DuelSurf.Command command = DuelSurf.choose(x, y, heading, speed, turn, ex, ey, threats, 1, 800, 600);
            checksum += command.bodyHeading() + command.targetSpeed();

            DuelGun gun = new DuelGun();
            gun.onScan(new DuelGun.Scan(2, turn - 1, ex, ey, heading, speed, 80));
            gun.onScan(new DuelGun.Scan(2, turn, ex + 1, ey + 1, heading + 2, speed, 80));
            DuelGun.Plan plan = gun.decide(new DuelGun.Input(turn, x, y, heading, 90, 0, 800, 600,
                    new DuelGun.Scan(2, turn, ex + 1, ey + 1, heading + 2, speed, 80),
                    new DuelGun.LegacyAim(headOn, Math.hypot(ex - x, ey - y), 1.5)));
            gun.onFired(turn, x, y, plan.power());
            checksum += plan.direction() + DuelGun.firepower(90, 80, Math.hypot(ex - x, ey - y));
            checksum += DuelRadar.turn(x, y, heading, ex, ey, i % 4, 1);

            TargetingGeometry.Aim aim = TargetingGeometry.calculate(new TargetingGeometry.Snapshot(
                    x, y, 800, 600, turn, heading, ex, ey, heading, speed, 2, turn - 1,
                    Math.hypot(ex - x, ey - y), 14, 0.2, speed));
            checksum += aim.aimAngle() + CombatMath.turnRate(heading, heading + 3, 2);

            EnemyFireTracker tracker = new EnemyFireTracker();
            tracker.recordBulletDamage(2, turn - 1, 4);
            checksum += tracker.inferPower(2, turn - 1, turn, 80, 78.5);

            for (int f = 0; f < features.length; f++) features[f] = rng.nextDouble();
            checksum += network.forward(features)[i % 6];
            network.trainSingle(features, i % 6, 1, 0.002);
        }
        return checksum;
    }
}
