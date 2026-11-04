import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;

/** Selects today's command by searching through up to two predicted bullet arrivals. */
final class SurfPlan {
    record Front(HitAngles.Wave wave, boolean speculative) {
        double distanceLeft(Flight.Pose pose, int turn) {
            return Math.hypot(pose.x() - wave.x(), pose.y() - wave.y())
                    - Flight.farEdge(wave.power(), turn - wave.t() + 1);
        }
    }
    record Choice(int mode, boolean warned, int fronts) {}
    private final HitAngles danger;
    private final Flight.Pose enemy;
    private final int width, height;

    SurfPlan(HitAngles danger, Flight.Pose enemy, int width, int height) {
        this.danger = danger;
        this.enemy = enemy;
        this.width = width;
        this.height = height;
    }

    static int readyAfter(int launch, double power, double cooling) {
        return launch + 1 + coolingTurns(1 + power / 5, cooling);
    }

    static int coolingTurns(double heat, double cooling) {
        if (!(cooling > 0)) throw new IllegalArgumentException("Positive gun cooling rate required");
        int turns = 0;
        while (heat > 0) {
            heat = Math.max(0, heat - cooling);
            turns++;
        }
        return turns;
    }

    Flight.Command command(Flight.Pose pose, int mode) {
        if (mode == 0) return new Flight.Command(0, 0);
        return Flight.orbit(pose, enemy.x(), enemy.y(), mode, width, height);
    }

    Choice choose(Flight.Pose own, int turn, int side, Collection<HitAngles.Wave> inferred,
                  int earliest, double lastPower) {
        List<Front> fronts = new ArrayList<>();
        for (var wave : inferred) {
            var front = new Front(wave, false);
            if (front.distanceLeft(own, turn) > 0) fronts.add(front);
        }
        boolean warn = fronts.size() < 2 && earliest <= turn + 1;
        if (warn) {
            double bearing = Flight.angle(own.x() - enemy.x(), own.y() - enemy.y());
            double lateral = own.v() * Flight.sin(own.h() - bearing);
            double lead = bearing + Math.toDegrees(Math.asin(lateral / (20 - 3 * lastPower)));
            fronts.add(new Front(new HitAngles.Wave(turn + 1, enemy.x(), enemy.y(), bearing, lead, lastPower), true));
        }
        if (fronts.isEmpty()) return new Choice(side, false, 0);
        fronts.sort(Comparator.comparingDouble(f -> f.distanceLeft(own, turn) / f.wave.speed()));
        Front[] nearest = fronts.stream().limit(2).toArray(Front[]::new);
        int best = side;
        double bestRisk = Double.POSITIVE_INFINITY;
        for (int mode : new int[]{side, -side, 0}) {
            double risk = (mode == side ? 0 : 2) + simulate(own, turn, mode, nearest);
            if (risk < bestRisk) {
                bestRisk = risk;
                best = mode;
            }
        }
        return new Choice(best, warn, nearest.length);
    }

    private double simulate(Flight.Pose pose, int start, int mode, Front[] fronts) {
        double risk = 0;
        // Even the slowest bullet traverses this arena's diagonal within this budget.
        int limit = (int) Math.ceil(Math.hypot(width, height) / 11) + 3;
        for (int age = 1; age <= limit; age++) {
            pose = Flight.step(pose, command(pose, mode), width, height);
            double edge = Math.min(Math.min(pose.x(), width - pose.x()), Math.min(pose.y(), height - pose.y()));
            risk += .10 * Math.max(0, 40 - edge);
            int crossed = 0;
            Front remaining = null;
            for (Front front : fronts) {
                if (front.distanceLeft(pose, start + age) > 0) {
                    remaining = front;
                    continue;
                }
                var w = front.wave;
                double damage = 4 * w.power() + 2 * Math.max(0, w.power() - 1);
                risk += danger.risk(w, pose.x() - w.x(), pose.y() - w.y())
                        * damage / 8 * (front.speculative ? .5 : 1);
                crossed++;
            }
            if (crossed == 0) continue;
            if (remaining == null) return risk;
            double nextRisk = Double.POSITIVE_INFINITY;
            for (int next = -1; next <= 1; next++) {
                nextRisk = Math.min(nextRisk, simulate(pose, start + age, next, new Front[]{remaining}));
            }
            return risk + nextRisk;
        }
        throw new IllegalStateException("Unreachable wave");
    }
}
