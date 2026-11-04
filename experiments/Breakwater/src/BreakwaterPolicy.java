import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/** Breakwater's game-local controller. Coordinates and headings follow Tank Royale. */
final class BreakwaterPolicy {
    record Own(int turn, double x, double y, double heading, double speed, double gun, double heat,
               double energy, double cooling, double width, double height) {}
    record Target(int turn, double x, double y, double heading, double speed, double energy) {}
    record Wave(int creation, double x, double y, double power, double reference) {
        double speed() { return 20 - 3 * power; }
    }
    enum Mode { BAIT, SHIELD, EVADE, ATTACK }
    record Decision(double turnRate, double targetSpeed, double aim, double power,
                    Mode mode, int waveTurn, boolean solution) {}
    private record Pose(double x, double y, double heading, double speed) {}
    private record Motion(double turn, double speed) {}
    private record Shot(double aim, boolean now) {}
    private record Rays(double linear, double stopped, boolean agreement) {}

    static final double BODY = 18, SHIELD_SPEED = 19.7;
    private static final int STEP_TURNS = 4, SEARCH = 90, RECENT_ATTACKS = 32;
    private final SplittableRandom random;
    private final ArrayDeque<Double> residuals = new ArrayDeque<>();
    private final ArrayDeque<Integer> results = new ArrayDeque<>();
    private final List<Flight> flights = new ArrayList<>();
    private final Set<Integer> shielded = new HashSet<>();
    private int side;
    private Target latest;
    private double width, height;

    BreakwaterPolicy(long seed) {
        random = new SplittableRandom(seed);
        startRound();
    }

    void startRound() {
        flights.clear();
        shielded.clear();
        latest = null;
        side = random.nextBoolean() ? 1 : -1;
    }

    void matched(Wave wave, double heading) {
        double offset = CombatMath.relativeAngle(heading - wave.reference());
        if (!Double.isFinite(offset)) return;
        if (residuals.size() == 8) residuals.removeFirst();
        residuals.addLast(clamp(offset, -12, 12));
    }

    double[] corrections() { return residuals.stream().mapToDouble(Double::doubleValue).toArray(); }

    private double correction() {
        double sum = 0;
        for (double value : residuals) sum += value;
        return residuals.isEmpty() ? 0 : sum / residuals.size();
    }

    void shieldLaunched(int creation) { shielded.add(creation); }

    Decision decide(Own own, Target target, List<Wave> waves) {
        width = own.width();
        height = own.height();
        Pose pose = new Pose(own.x(), own.y(), own.heading(), own.speed());
        boolean fresh = target != null && own.turn() >= target.turn() && own.turn() - target.turn() <= 1;
        double direct = target == null ? own.gun() : bearing(pose.x(), pose.y(), target.x(), target.y());
        double range = target == null ? 0 : Math.hypot(target.x() - own.x(), target.y() - own.y());
        Wave first = null;
        double nearest = Double.POSITIVE_INFINITY;
        shielded.removeIf(id -> waves.stream().noneMatch(w -> w.creation() == id));
        boolean emergency = false;
        for (Wave w : waves) {
            double remaining = (Math.hypot(own.x() - w.x(), own.y() - w.y()) - BODY) / w.speed()
                    - (own.turn() - w.creation() + 2);
            if (remaining < -2 || w.creation() > own.turn()) continue;
            if (shielded.contains(w.creation())) {
                // A fired shield is not a confirmed collision. Escape if that wave still approaches.
                if (remaining <= 5) emergency = true;
            } else if (remaining < nearest) {
                nearest = remaining;
                first = w;
            }
        }
        if (first != null && fresh && range >= 350 && !emergency && own.energy() > .1) {
            Shot shot = shield(own, pose, direct, first);
            if (shot != null) {
                Motion motion = sidestep(pose, direct, own.turn() - first.creation());
                return decision(motion, shot.aim(), shot.now() ? .1 : 0, Mode.SHIELD, first.creation(), true);
            }
        }

        boolean evade = !fresh || range < 350 || emergency || first != null;
        Motion motion = evade ? escape(pose, direct) : steer(pose, direct + 90 * side, 0);
        if (fresh && first == null && !emergency) {
            double power = target.energy() <= 4 ? clamp((target.energy() + .001) / 4, .1, 1) : 1;
            Rays rays = rays(own.turn() + 1, own.x(), own.y(), power, target);
            if (rays.agreement() || target.energy() <= 4) {
                int[] scores = attackScores();
                double aim = scores[1] > scores[0] ? rays.stopped() : rays.linear();
                double half = Math.toDegrees(Math.asin(Math.min(1, BODY / Math.max(BODY, range))));
                boolean aligned = Math.abs(CombatMath.relativeAngle(aim - own.gun())) <= half / 2;
                boolean canFire = own.energy() > power && own.heat() <= 1e-9 && aligned;
                return decision(motion, aim, canFire ? power : 0, Mode.ATTACK, -1, false);
            }
        }
        return decision(motion, direct, 0, evade ? Mode.EVADE : Mode.BAIT, first == null ? -1 : first.creation(), false);
    }

    private static Decision decision(Motion m, double aim, double power, Mode mode, int wave, boolean solution) {
        return new Decision(m.turn(), m.speed(), aim, power, mode, wave, solution);
    }

    private Motion sidestep(Pose p, double direct, int age) {
        return steer(p, direct + 90 * side, age < STEP_TURNS ? 4 : 0);
    }

    private Shot shield(Own own, Pose initial, double direct, Wave wave) {
        double heading = wave.reference() + correction();
        double ux = Math.cos(Math.toRadians(heading)), uy = Math.sin(Math.toRadians(heading));
        Pose pose = initial;
        int cooled = (int) Math.ceil(Math.max(0, own.heat() - 1e-9) / own.cooling());
        for (int delay = 0; delay < SEARCH; delay++) {
            int command = own.turn() + delay, launch = command + 1;
            Pose next = step(pose, sidestep(pose, direct, command - wave.creation()));
            if (!inside(next)) return null;
            // Fire before movement, using the command-turn gun heading and position.
            double offset = Math.abs((pose.x() - wave.x()) * uy - (pose.y() - wave.y()) * ux);
            if (delay >= cooled && offset >= 2) {
                for (int tick = launch; tick < launch + SEARCH; tick++) {
                    double lo = wave.speed() * (tick - wave.creation() + 1);
                    // Stop before the threatening wave can touch even this stationary firing position.
                    if (lo + wave.speed() >= Math.hypot(pose.x() - wave.x(), pose.y() - wave.y()) - BODY) break;
                    double ax = wave.x() + lo * ux, ay = wave.y() + lo * uy;
                    double bx = ax + wave.speed() * ux, by = ay + wave.speed() * uy;
                    double inner = SHIELD_SPEED * (tick - launch + 1);
                    if (delay == 0 && intersectsRay(pose, own.gun(), inner, ax, ay, bx, by))
                        return new Shot(own.gun(), true);
                    if (delay == 0) continue;
                    double aim = crossingAim(pose, inner, ax, ay, bx, by);
                    if (Double.isFinite(aim) && Math.abs(CombatMath.relativeAngle(aim - own.gun())) <= 20 * delay)
                        return new Shot(aim, false);
                }
            }
            if (wave.speed() * (launch - wave.creation() + 1) >= Math.hypot(pose.x() - wave.x(), pose.y() - wave.y()) - BODY)
                break;
            pose = next;
        }
        return null;
    }

    /** Find the middle of the radial overlap on a monotone part of the enemy segment. */
    private static double crossingAim(Pose p, double inner, double ax, double ay, double bx, double by) {
        double dx = bx - ax, dy = by - ay;
        double foot = ((p.x() - ax) * dx + (p.y() - ay) * dy) / (dx * dx + dy * dy);
        double split = clamp(foot, 0, 1);
        double min = Math.hypot(ax + split * dx - p.x(), ay + split * dy - p.y());
        double first = Math.hypot(ax - p.x(), ay - p.y()), last = Math.hypot(bx - p.x(), by - p.y());
        double outer = Math.min(inner + SHIELD_SPEED, Math.max(first, last));
        double lower = Math.max(inner, min);
        if (lower >= outer) return Double.NaN;
        double r = (lower + outer) * .5;
        double fx = ax + foot * dx - p.x(), fy = ay + foot * dy - p.y();
        double shift = Math.sqrt(Math.max(0, (r * r - fx * fx - fy * fy) / (dx * dx + dy * dy)));
        double u = first >= last ? foot - shift : foot + shift;
        double aim = bearing(p.x(), p.y(), ax + u * dx, ay + u * dy);
        return intersectsRay(p, aim, inner, ax, ay, bx, by) ? aim : Double.NaN;
    }

    private static boolean intersectsRay(Pose p, double aim, double inner, double ax, double ay, double bx, double by) {
        double ux = Math.cos(Math.toRadians(aim)), uy = Math.sin(Math.toRadians(aim));
        return collide(p.x() + inner * ux, p.y() + inner * uy, p.x() + (inner + SHIELD_SPEED) * ux,
                p.y() + (inner + SHIELD_SPEED) * uy, ax, ay, bx, by);
    }

    /** Same-tick BulletLine endpoint prefilter and segment tests from pinned CollisionDetector / model/math. */
    static boolean collide(double x1, double y1, double x2, double y2,
                           double x3, double y3, double x4, double y4) {
        if (Math.hypot(x2 - x4, y2 - y4) > 2 * SHIELD_SPEED) return false;
        double ax = x2 - x1, ay = y2 - y1, bx = x3 - x4, by = y3 - y4;
        double cx = x1 - x3, cy = y1 - y3;
        double den = ay * bx - ax * by, a = by * cx - bx * cy, b = ax * cy - ay * cx;
        if (den > 0) return a >= 0 && a <= den && b >= 0 && b <= den;
        if (den < 0) return a <= 0 && a >= den && b <= 0 && b >= den;
        return x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) == 0
                && Math.max(Math.min(x1, x2), Math.min(x3, x4)) <= Math.min(Math.max(x1, x2), Math.max(x3, x4))
                && Math.max(Math.min(y1, y2), Math.min(y3, y4)) <= Math.min(Math.max(y1, y2), Math.max(y3, y4));
    }

    private Motion escape(Pose pose, double direct) {
        for (double angle : new double[]{direct + 90 * side, direct - 90 * side,
                bearing(pose.x(), pose.y(), width / 2, height / 2)}) {
            Pose projected = pose;
            boolean safe = true;
            for (int i = 0; i < 8; i++) {
                projected = step(projected, steer(projected, angle, 8));
                if (!inside(projected)) { safe = false; break; }
            }
            if (safe) return steer(pose, angle, 8);
        }
        return steer(pose, bearing(pose.x(), pose.y(), width / 2, height / 2), 0);
    }

    private boolean inside(Pose p) {
        return p.x() > BODY && p.x() < width - BODY && p.y() > BODY && p.y() < height - BODY;
    }

    private static Motion steer(Pose p, double angle, double magnitude) {
        double diff = CombatMath.relativeAngle(angle - p.heading());
        boolean reverse = Math.abs(diff) > 90;
        if (reverse) diff = CombatMath.relativeAngle(diff + 180);
        return new Motion(clamp(diff, -DuelSurf.maxTurn(p.speed()), DuelSurf.maxTurn(p.speed())), reverse ? -magnitude : magnitude);
    }

    private static Pose step(Pose p, Motion m) {
        double v = DuelSurf.nextSpeed(p.speed(), m.speed()), a = Math.toRadians(p.heading());
        double limit = DuelSurf.maxTurn(v);
        return new Pose(p.x() + v * Math.cos(a), p.y() + v * Math.sin(a),
                p.heading() + clamp(m.turn(), -limit, limit), v);
    }

    private Rays rays(int launch, double x, double y, double power, Target t) {
        double speed = 20 - 3 * power, px = t.x(), py = t.y();
        double dx = t.speed() * Math.cos(Math.toRadians(t.heading())), dy = t.speed() * Math.sin(Math.toRadians(t.heading()));
        for (int tick = t.turn() + 1; tick <= launch + SEARCH; tick++) {
            double nx = clamp(px + dx, BODY, width - BODY), ny = clamp(py + dy, BODY, height - BODY);
            if (nx != px + dx || ny != py + dy) { dx = 0; dy = 0; }
            px = nx; py = ny;
            if (tick >= launch && speed * (tick - launch + 2) >= Math.hypot(px - x, py - y)) break;
        }
        double linear = bearing(x, y, px, py), stopped = bearing(x, y, t.x(), t.y());
        // Require both candidate rays to pass inside both predicted footprints. A winning model may use either.
        double gap = Math.abs(CombatMath.relativeAngle(linear - stopped));
        double far = Math.max(Math.hypot(px - x, py - y), Math.hypot(t.x() - x, t.y() - y));
        boolean agreement = gap <= Math.toDegrees(Math.asin(Math.min(1, BODY / Math.max(BODY, far))));
        return new Rays(linear, stopped, agreement);
    }

    private static final class Flight {
        final int launch;
        final double x, y, speed;
        final Rays rays;
        int observed, hits;
        Flight(int launch, double x, double y, double speed, Rays rays) {
            this.launch = launch; this.x = x; this.y = y; this.speed = speed; this.rays = rays;
            observed = launch - 1;
        }
    }

    void attackLaunched(int creation, double x, double y, double power, Target original) {
        if (original == null || width <= 2 * BODY || height <= 2 * BODY) return;
        Flight f = new Flight(creation, x, y, 20 - 3 * power, rays(creation, x, y, power, original));
        if (flights.size() == 16) flights.remove(0);
        flights.add(f);
        if (latest != null && latest.turn() == creation) score(latest);
    }

    void observe(Target t) {
        if (t == null || latest != null && latest.turn() >= t.turn()) return;
        latest = t;
        score(t);
    }

    private void score(Target t) {
        flights.removeIf(f -> {
            if (t.turn() == f.observed) return false;
            if (t.turn() != f.observed + 1 || t.turn() - f.launch >= SEARCH) return true;
            f.observed = t.turn();
            double inner = f.speed * (t.turn() - f.launch + 1);
            if (DuelGun.segmentDistance(t.x(), t.y(), f.x, f.y, f.rays.linear(), inner, inner + f.speed) <= BODY) f.hits |= 1;
            if (DuelGun.segmentDistance(t.x(), t.y(), f.x, f.y, f.rays.stopped(), inner, inner + f.speed) <= BODY) f.hits |= 2;
            if (f.hits != 3 && inner <= Math.hypot(t.x() - f.x, t.y() - f.y) + BODY) return false;
            if (results.size() == RECENT_ATTACKS) results.removeFirst();
            results.addLast(f.hits);
            return true;
        });
    }

    int[] attackHistory() { return results.stream().mapToInt(Integer::intValue).toArray(); }

    int[] attackScores() {
        int[] counts = {0, 0, results.size()};
        for (int result : results) { counts[0] += result & 1; counts[1] += (result >> 1) & 1; }
        return counts;
    }

    private static double bearing(double x, double y, double tx, double ty) { return CombatMath.bearing(tx - x, ty - y); }
    private static double clamp(double value, double low, double high) { return Math.max(low, Math.min(high, value)); }
}
