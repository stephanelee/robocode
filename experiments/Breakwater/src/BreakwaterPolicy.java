import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;

/** Breakwater's game-local controller. Coordinates and headings follow Tank Royale. */
final class BreakwaterPolicy {
    record Own(int turn, double x, double y, double heading, double speed, double gun, double heat,
               double energy, double cooling, double width, double height) {}
    record Target(int turn, double x, double y, double heading, double speed, double energy) {}
    record Wave(int creation, double x, double y, double power, double reference, double lateral) {
        Wave(int creation, double x, double y, double power, double reference) {
            this(creation, x, y, power, reference, 0);
        }
        double speed() { return 20 - 3 * power; }
        double escape() { return Math.toDegrees(Math.asin(8 / speed())); }
    }
    record Decision(double turnRate, double targetSpeed, double aim, double power, int gun, int waves, int orbitSign) {}
    record Pose(double x, double y, double heading, double speed) {}
    record Motion(double turn, double speed) {}
    private record Path(int side, int switchAfter) {
        int sign(int elapsed) { return elapsed < switchAfter ? side : -side; }
    }
    private record Rays(double linear, double stopped, double circular) {
        double get(int gun) { return switch (gun) { case 0 -> linear; case 1 -> stopped; default -> circular; }; }
    }
    private record Pending(int creation, double x, double y, double power, Rays rays) {}

    static final double BODY = 18;
    private static final int SEARCH = 90, MEMORY = 32;
    private final SplittableRandom random;
    private final ArrayDeque<Double> residuals = new ArrayDeque<>();
    private final ArrayDeque<Integer> results = new ArrayDeque<>();
    private final List<Flight> flights = new ArrayList<>();
    private int side, since, primary, secondary;
    private Path path;
    private Target latest;
    private double turnRate;
    private Pending pending;

    BreakwaterPolicy(long seed) {
        random = new SplittableRandom(seed);
        startRound();
    }

    void startRound() {
        flights.clear();
        pending = null;
        latest = null;
        turnRate = 0;
        path = null;
        side = random.nextBoolean() ? 1 : -1;
    }

    void matched(Wave wave, double heading) {
        if (Math.abs(wave.lateral()) < 1) return;
        double gf = CombatMath.relativeAngle(heading - wave.reference()) * Math.signum(wave.lateral()) / wave.escape();
        if (!Double.isFinite(gf)) return;
        if (residuals.size() == MEMORY) residuals.removeFirst();
        residuals.addLast(clamp(gf, -1, 1));
    }

    double[] corrections() { return residuals.stream().mapToDouble(Double::doubleValue).toArray(); }

    Decision decide(Own own, Target target, List<Wave> waves) {
        var pose = new Pose(own.x(), own.y(), own.heading(), own.speed());
        double ex = target == null ? own.width() / 2 : target.x();
        double ey = target == null ? own.height() / 2 : target.y();
        var threats = waves.stream().filter(w -> w.creation() <= own.turn() && eta(own, w) >= -2)
                .sorted(Comparator.comparingDouble((Wave w) -> eta(own, w)).thenComparingInt(Wave::creation)).limit(2).toList();
        int first = threats.isEmpty() ? -1 : threats.get(0).creation();
        int second = threats.size() < 2 ? -1 : threats.get(1).creation();
        if (path == null || primary != first || secondary != second || own.turn() - since >= 16) {
            path = choose(own, pose, ex, ey, threats);
            since = own.turn();
            primary = first;
            secondary = second;
        }
        side = path.sign(own.turn() - since);
        Motion motion = travel(pose, ex, ey, side, own.width(), own.height());

        pending = null;
        if (target == null || own.turn() < target.turn() || own.turn() - target.turn() > 1)
            return new Decision(motion.turn(), motion.speed(), own.gun(), 0, -1, threats.size(), side);
        double range = Math.hypot(ex - own.x(), ey - own.y());
        double power = shotPower(own, target, range);
        int launch = own.turn() + 1;
        Rays rays = new Rays(forecast(own, target, power, 0), bearing(own.x(), own.y(), ex, ey),
                forecast(own, target, power, turnRate));
        int[] counts = attackScores();
        int gun = 2; // Circular forecast includes straight motion as the zero-turn case.
        for (int g = 0; g < 2; g++) if (counts[g] > counts[gun]) gun = g;
        double aim = rays.get(gun);
        double half = Math.toDegrees(Math.asin(Math.min(1, BODY / Math.max(BODY, range))));
        boolean fire = own.energy() > power && own.heat() <= 1e-9
                && Math.abs(CombatMath.relativeAngle(aim - own.gun())) <= .75 * half;
        // In envelope mode the actual ray can be aligned even when the legacy ray is not.
        // The shared comparator still admits these rays only after a matching confirmed launch.
        if (own.energy() > power && own.heat() <= 1e-9)
            pending = new Pending(launch, own.x(), own.y(), power, rays);
        return new Decision(motion.turn(), motion.speed(), aim, fire ? power : 0, gun, threats.size(), side);
    }

    private static double eta(Own own, Wave w) {
        return (Math.hypot(own.x() - w.x(), own.y() - w.y()) - BODY) / w.speed() - (own.turn() - w.creation() + 2);
    }

    private Path choose(Own own, Pose pose, double ex, double ey, List<Wave> waves) {
        Path[] options = {new Path(side, SEARCH + 1), new Path(-side, SEARCH + 1), new Path(side, 5), new Path(side, 10)};
        int n = waves.isEmpty() ? 2 : 4;
        double[] score = new double[n];
        double best = Double.POSITIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            score[i] = risk(own, pose, ex, ey, waves, options[i]);
            best = Math.min(best, score[i]);
        }
        // Seeded reservoir choice among near-equal complete trajectories; hold the chosen policy between replans.
        int seen = 0, pick = 0;
        for (int i = 0; i < n; i++)
            if (score[i] <= best + .08 && random.nextInt(++seen) == 0) pick = i;
        return options[pick];
    }

    private double risk(Own own, Pose pose, double enemyX, double enemyY, List<Wave> waves, Path option) {
        double[] peakDanger = new double[waves.size()];
        double wallPenalty = 0;
        for (int step = 1; step <= SEARCH; step++) {
            Motion motion = travel(pose, enemyX, enemyY, option.sign(step - 1), own.width(), own.height());
            pose = step(pose, motion);
            double wallGap = DuelSurf.wallDistance(pose.x(), pose.y(), own.width(), own.height()) - BODY;
            if (wallGap < 0) return 1000 - step;
            wallPenalty = Math.max(wallPenalty, Math.max(0, 72 - wallGap) / 72);
            boolean allPassed = true;
            for (int i = 0; i < waves.size(); i++) {
                Wave wave = waves.get(i);
                double distance = Math.hypot(pose.x() - wave.x(), pose.y() - wave.y());
                double innerRadius = (own.turn() + step - wave.creation() + 1) * wave.speed();
                if (innerRadius > distance + BODY) continue;
                allPassed = false;
                if (innerRadius + wave.speed() >= distance - BODY) {
                    peakDanger[i] = Math.max(peakDanger[i], density(wave, pose, distance));
                }
            }
            if (allPassed && step >= 12) break;
        }
        double total = .18 * wallPenalty;
        for (int i = 0; i < peakDanger.length; i++) total += peakDanger[i] / (1 + i);
        return total;
    }

    /** Density of actual incoming ray angles, mirroring samples about launch-time lateral direction. */
    private double density(Wave w, Pose p, double distance) {
        double sign = w.lateral() < 0 ? -1 : 1;
        double gf = CombatMath.relativeAngle(bearing(w.x(), w.y(), p.x(), p.y()) - w.reference()) * sign / w.escape();
        double width = .10 + Math.toDegrees(Math.asin(Math.min(1, BODY / Math.max(BODY, distance)))) / w.escape();
        double linear = Math.toDegrees(Math.asin(Math.min(1, Math.abs(w.lateral()) / w.speed()))) / w.escape();
        double sum = kernel(gf, width) + 2 * kernel(gf - linear, width);
        int count = 3;
        if (Math.abs(w.lateral()) >= 1) {
            for (double hit : residuals) sum += kernel(gf - hit, width);
            count += residuals.size();
        }
        return sum / count;
    }

    private static double kernel(double offset, double width) {
        double z = offset / width;
        return 1 / (1 + z * z);
    }

    private static Motion travel(Pose pose, double enemyX, double enemyY, int side, double width, double height) {
        double range = Math.hypot(enemyX - pose.x(), enemyY - pose.y());
        double angle = bearing(pose.x(), pose.y(), enemyX, enemyY) + side * (90 - clamp((range - 430) / 7, -32, 32));
        // Smooth a 100px travel stick toward center. Back-as-front preserves the same travel vector.
        double center = bearing(pose.x(), pose.y(), width / 2, height / 2);
        double towardCenter = Math.signum(CombatMath.relativeAngle(center - angle));
        for (int i = 0; i < 18; i++) {
            double radians = Math.toRadians(angle);
            double probeX = pose.x() + 100 * Math.cos(radians);
            double probeY = pose.y() + 100 * Math.sin(radians);
            if (DuelSurf.wallDistance(probeX, probeY, width, height) >= 30) break;
            angle += 10 * towardCenter;
        }
        double difference = CombatMath.relativeAngle(angle - pose.heading());
        boolean reverse = Math.abs(difference) > 90;
        if (reverse) difference = CombatMath.relativeAngle(difference + 180);
        double maximumTurn = DuelSurf.maxTurn(pose.speed());
        double turn = clamp(difference, -maximumTurn, maximumTurn);
        Motion motion = new Motion(turn, reverse ? -8 : 8);
        Pose next = step(pose, motion);
        if (DuelSurf.wallDistance(next.x(), next.y(), width, height) < BODY) return new Motion(turn, 0);
        return motion;
    }

    static Pose step(Pose pose, Motion motion) {
        double speed = DuelSurf.nextSpeed(pose.speed(), motion.speed());
        double angle = Math.toRadians(pose.heading());
        double limit = DuelSurf.maxTurn(speed);
        return new Pose(pose.x() + speed * Math.cos(angle), pose.y() + speed * Math.sin(angle),
                pose.heading() + clamp(motion.turn(), -limit, limit), speed);
    }

    private static double forecast(Own own, Target target, double power, double turnRate) {
        double bulletSpeed = 20 - 3 * power;
        double x = target.x(), y = target.y();
        double heading = target.heading(), speed = target.speed();
        int launch = own.turn() + 1;
        for (int tick = target.turn() + 1; tick <= launch + SEARCH; tick++) {
            double nextX = x + speed * Math.cos(Math.toRadians(heading));
            double nextY = y + speed * Math.sin(Math.toRadians(heading));
            x = clamp(nextX, BODY, own.width() - BODY);
            y = clamp(nextY, BODY, own.height() - BODY);
            if (nextX != x || nextY != y) speed = 0;
            double limit = DuelSurf.maxTurn(speed);
            heading += clamp(turnRate, -limit, limit);
            double outerRadius = bulletSpeed * (tick - launch + 2);
            if (tick >= launch && outerRadius >= Math.hypot(x - own.x(), y - own.y())) break;
        }
        return bearing(own.x(), own.y(), x, y);
    }

    private static final class Flight {
        final int launch;
        final double x, y, speed;
        final Rays rays;
        int observed, hits;

        Flight(int launch, double x, double y, double speed, Rays rays) {
            this.launch = launch;
            this.x = x;
            this.y = y;
            this.speed = speed;
            this.rays = rays;
            observed = launch - 1;
        }
    }

    void attackLaunched(int creation, double x, double y, double power, Target original) {
        Pending request = pending;
        pending = null;
        if (original == null || request == null || creation != request.creation()
                || Math.abs(power - request.power()) > 1e-6 || Math.hypot(x - request.x(), y - request.y()) > 1e-6) return;
        if (flights.size() == 16) flights.remove(0);
        flights.add(new Flight(creation, x, y, 20 - 3 * power, request.rays()));
        if (latest != null && latest.turn() == creation) score(latest);
    }

    void observe(Target target) {
        if (target == null || latest != null && latest.turn() >= target.turn()) return;
        turnRate = latest != null && latest.turn() + 1 == target.turn()
                ? CombatMath.relativeAngle(target.heading() - latest.heading()) : 0;
        latest = target;
        score(target);
    }

    private void score(Target scan) {
        flights.removeIf(flight -> {
            if (scan.turn() == flight.observed) return false;
            if (scan.turn() != flight.observed + 1 || scan.turn() - flight.launch >= SEARCH) return true;
            flight.observed = scan.turn();
            double innerRadius = flight.speed * (scan.turn() - flight.launch + 1);
            for (int gun = 0; gun < 3; gun++) {
                double distance = DuelGun.segmentDistance(scan.x(), scan.y(), flight.x, flight.y,
                        flight.rays.get(gun), innerRadius, innerRadius + flight.speed);
                if (distance <= BODY) flight.hits |= 1 << gun;
            }
            double range = Math.hypot(scan.x() - flight.x, scan.y() - flight.y);
            if (flight.hits != 7 && innerRadius <= range + BODY) return false;
            if (results.size() == MEMORY) results.removeFirst();
            results.addLast(flight.hits);
            return true;
        });
    }

    int[] attackHistory() { return results.stream().mapToInt(Integer::intValue).toArray(); }

    int[] attackScores() {
        int[] counts = {0, 0, 0, results.size()};
        for (int hits : results) {
            for (int gun = 0; gun < 3; gun++) counts[gun] += (hits >> gun) & 1;
        }
        return counts;
    }

    static double shotPower(Own own, Target target, double range) {
        double power = Math.min(range <= 320 ? 2 : 1.35, own.energy() / 16);
        return clamp(Math.min(power, (target.energy() + .001) / 4), .1, 2);
    }

    private static double bearing(double x, double y, double targetX, double targetY) {
        return CombatMath.bearing(targetX - x, targetY - y);
    }

    private static double clamp(double value, double low, double high) { return Math.max(low, Math.min(high, value)); }
}
