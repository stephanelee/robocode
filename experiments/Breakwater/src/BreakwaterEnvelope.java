import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;

/** Three legal target-motion hypotheses and an observation-only, common-opportunity ray scorer. */
final class BreakwaterEnvelope {
    record Plan(double aim, double power, boolean supported, boolean opportunity) {}
    record Row(int round, int commandTurn, int creationTurn, int lastScanTurn, double x, double y,
               double power, double currentGun, int baselineGun, double baselineAim, double candidateAim,
               int support, boolean baselineEligible, boolean candidateEligible, String status, int endTurn, int hits,
               double targetX, double targetY, double targetHeading, double targetSpeed, double width, double height,
               int actualSupport) {}
    record Arc(int path, double center, double half) {}
    record Choice(double angle, int support, int actualSupport) {}
    private static final double BODY = 18;
    private static final int HORIZON = 90;
    private final Consumer<Row> writer;
    private final List<Cohort> cohorts = new ArrayList<>();
    private int round;
    private int lastScanTurn;

    BreakwaterEnvelope(Consumer<Row> writer) { this.writer = writer; }

    void startRound(int number) {
        if (!cohorts.isEmpty()) throw new IllegalStateException("Unsealed shadow opportunities");
        round = number;
        lastScanTurn = -1;
    }

    Plan decide(BreakwaterPolicy.Own own, BreakwaterPolicy.Target target, BreakwaterPolicy.Decision baseline) {
        if (target == null || baseline.gun() < 0 || own.turn() < target.turn() || own.turn() - target.turn() > 1)
            return new Plan(own.gun(), 0, false, false);
        double range = Math.hypot(target.x() - own.x(), target.y() - own.y());
        double power = BreakwaterPolicy.shotPower(own, target, range);
        double direct = CombatMath.bearing(target.x() - own.x(), target.y() - own.y());
        List<Arc> arcs = footprints(own, target, power);
        Choice choice = choose(arcs, direct, own.gun());
        boolean opportunity = own.heat() <= 1e-9 && own.energy() > power;
        double tolerance = .75 * Math.toDegrees(Math.asin(Math.min(1, BODY / Math.max(BODY, range))));
        boolean supported = choice.support() >= 2;
        boolean aligned = Math.abs(CombatMath.relativeAngle(choice.angle() - own.gun())) <= tolerance;
        boolean eligible = opportunity && supported && choice.actualSupport() >= 2 && aligned;
        if (opportunity && writer != null)
            cohorts.add(new Cohort(own, target, baseline, choice, eligible, power));
        return new Plan(choice.angle(), eligible ? power : 0, supported, opportunity);
    }

    private static List<Arc> footprints(BreakwaterPolicy.Own own, BreakwaterPolicy.Target target, double power) {
        List<Arc> arcs = new ArrayList<>();
        double bulletSpeed = 20 - 3 * power;
        double radial = CombatMath.bearing(target.x() - own.x(), target.y() - own.y());
        double lateral = target.speed() * Math.sin(Math.toRadians(target.heading() - radial));
        int sign = lateral < 0 ? -1 : 1;
        int launch = own.turn() + 1;
        for (int path = 0; path < 3; path++) {
            var pose = new BreakwaterPolicy.Pose(target.x(), target.y(), target.heading(), target.speed());
            int orbit = switch (path) { case 0 -> sign; case 1 -> -sign; default -> 0; };
            for (int tick = target.turn() + 1; tick < launch + HORIZON; tick++) {
                pose = advance(pose, own, orbit);
                if (tick < launch) continue;
                double distance = Math.hypot(pose.x() - own.x(), pose.y() - own.y());
                double inner = bulletSpeed * (tick - launch + 1);
                if (inner > distance + BODY) break;
                double half = halfFootprint(distance, inner, inner + bulletSpeed);
                if (Double.isFinite(half)) {
                    double angle = CombatMath.bearing(pose.x() - own.x(), pose.y() - own.y());
                    arcs.add(new Arc(path, angle, half));
                }
            }
        }
        return arcs;
    }

    /** Gun hypotheses orbit the fixed firing origin and use the port's 650px surf distance and 160px wall stick. */
    static BreakwaterPolicy.Pose advance(BreakwaterPolicy.Pose pose, BreakwaterPolicy.Own own, int orbit) {
        if (orbit == 0) return BreakwaterPolicy.step(pose, new BreakwaterPolicy.Motion(0, 0));
        double dx = pose.x() - own.x(), dy = pose.y() - own.y();
        double radial = CombatMath.bearing(dx, dy);
        double tilt = clamp(.6 * (Math.hypot(dx, dy) - 650) / 650, -.45 * Math.PI, .45 * Math.PI);
        double travel = radial + orbit * (90 + Math.toDegrees(tilt));
        for (int i = 0; i < 72; i++) {
            double angle = Math.toRadians(travel);
            double px = pose.x() + 160 * Math.cos(angle), py = pose.y() + 160 * Math.sin(angle);
            if (px >= BODY && py >= BODY && px <= own.width() - BODY && py <= own.height() - BODY) break;
            travel += orbit * 5;
        }
        double turn = CombatMath.relativeAngle(travel - pose.heading());
        boolean back = Math.abs(turn) > 90;
        if (back) turn = CombatMath.relativeAngle(turn + 180);
        double limit = DuelSurf.maxTurn(pose.speed());
        var next = BreakwaterPolicy.step(pose, new BreakwaterPolicy.Motion(clamp(turn, -limit, limit), back ? -8 : 8));
        return wall(pose, next, own.width(), own.height());
    }

    /** With the previous pose as origin, both server wall passes intersect the same movement line. */
    static BreakwaterPolicy.Pose wall(BreakwaterPolicy.Pose before, BreakwaterPolicy.Pose after, double width, double height) {
        double x = after.x(), y = after.y();
        if (x >= BODY && x <= width - BODY && y >= BODY && y <= height - BODY) return after;
        for (int pass = 0; pass < 2; pass++) {
            double oldX = x, oldY = y;
            if (oldX < BODY || oldX > width - BODY) {
                x = clamp(oldX, BODY, width - BODY);
                if (oldX != before.x()) y = before.y() + (oldY - before.y()) * (x - before.x()) / (oldX - before.x());
            }
            if (oldY < BODY || oldY > height - BODY) {
                y = clamp(oldY, BODY, height - BODY);
                if (oldY != before.y()) x = before.x() + (oldX - before.x()) * (y - before.y()) / (oldY - before.y());
            }
        }
        return new BreakwaterPolicy.Pose(x, y, after.heading(), 0);
    }

    /** Angular half-width subtended by the part of a radius-18 circle inside this tick's bullet annulus. */
    static double halfFootprint(double distance, double inner, double outer) {
        if (outer < distance - BODY || inner > distance + BODY) return Double.NaN;
        if (distance == 0) return 180;
        double tangent = Math.sqrt(Math.max(0, distance * distance - BODY * BODY));
        double radius = clamp(tangent, inner, outer);
        double cosine = (distance * distance + radius * radius - BODY * BODY) / (2 * distance * radius);
        return Math.toDegrees(Math.acos(clamp(cosine, -1, 1)));
    }

    /** Maximize distinct-path coverage. Prefer wide supported sectors, then proximity to the direct ray. */
    static Choice choose(List<Arc> arcs, double direct, double gun) {
        List<Double> edges = new ArrayList<>();
        for (Arc arc : arcs) {
            edges.add(normalize(arc.center() - arc.half()));
            edges.add(normalize(arc.center() + arc.half()));
        }
        if (edges.isEmpty()) return new Choice(direct, 0, 0);
        Collections.sort(edges);
        double bestAngle = direct, bestWidth = -1;
        int best = -1;
        for (int i = 0; i < edges.size(); i++) {
            double start = edges.get(i);
            double end = i + 1 < edges.size() ? edges.get(i + 1) : edges.get(0) + 360;
            // Endpoints include exact tangencies where two closed footprints touch at only one angle.
            for (int interior = 0; interior <= 1; interior++) {
                double width = interior == 1 ? end - start : 0;
                double angle = normalize(start + width / 2);
                int count = support(arcs, angle);
                double offset = Math.abs(CombatMath.relativeAngle(angle - direct));
                double bestOffset = Math.abs(CombatMath.relativeAngle(bestAngle - direct));
                if (count > best || count == best && (width > bestWidth
                        || width == bestWidth && (offset < bestOffset || offset == bestOffset && angle < bestAngle))) {
                    best = count;
                    bestWidth = width;
                    bestAngle = angle;
                }
            }
        }
        return new Choice(bestAngle, best, support(arcs, gun));
    }

    private static int support(List<Arc> arcs, double angle) {
        int mask = 0;
        for (Arc arc : arcs) {
            if (Math.abs(CombatMath.relativeAngle(angle - arc.center())) <= arc.half() + 1e-9)
                mask |= 1 << arc.path();
        }
        return Integer.bitCount(mask);
    }

    private final class Cohort {
        final BreakwaterPolicy.Own own;
        final BreakwaterPolicy.Target target;
        final BreakwaterPolicy.Decision baseline;
        final Choice choice;
        final boolean eligible;
        final double power;
        final int creation;
        int expected, hits;

        Cohort(BreakwaterPolicy.Own own, BreakwaterPolicy.Target target, BreakwaterPolicy.Decision baseline,
               Choice choice, boolean eligible, double power) {
            this.own = own;
            this.target = target;
            this.baseline = baseline;
            this.choice = choice;
            this.eligible = eligible;
            this.power = power;
            creation = own.turn() + 1;
            expected = creation;
        }

        void finish(String status, int end, int outcome) {
            writer.accept(new Row(round, own.turn(), creation, target.turn(), own.x(), own.y(), power,
                    own.gun(), baseline.gun(), baseline.aim(), choice.angle(), choice.support(),
                    baseline.power() > 0, eligible, status, end, outcome, target.x(), target.y(), target.heading(),
                    target.speed(), own.width(), own.height(), choice.actualSupport()));
        }

        boolean observe(BreakwaterPolicy.Target scan) {
            if (scan.turn() < expected) return false;
            if (scan.turn() != expected) {
                finish("gap", scan.turn(), -1);
                return true;
            }
            if (scan.turn() - creation >= HORIZON) {
                finish("horizon", scan.turn(), -1);
                return true;
            }
            expected++;
            double speed = 20 - 3 * power;
            double lo = speed * (scan.turn() - creation + 1);
            double[] rays = {baseline.aim(), choice.angle(), own.gun()};
            for (int i = 0; i < rays.length; i++) {
                if (DuelGun.segmentDistance(scan.x(), scan.y(), own.x(), own.y(), rays[i], lo, lo + speed) <= BODY)
                    hits |= 1 << i;
            }
            if (lo <= Math.hypot(scan.x() - own.x(), scan.y() - own.y()) + BODY) return false;
            finish("resolved", scan.turn(), hits);
            return true;
        }
    }

    void observe(BreakwaterPolicy.Target scan) {
        if (scan.turn() <= lastScanTurn) return;
        lastScanTurn = scan.turn();
        cohorts.removeIf(cohort -> cohort.observe(scan));
    }

    void endRound(int turn) {
        for (Cohort cohort : cohorts) cohort.finish("terminal", turn, -1);
        cohorts.clear();
    }

    private static double normalize(double angle) { return (angle % 360 + 360) % 360; }
    private static double clamp(double value, double low, double high) { return Math.max(low, Math.min(high, value)); }
}
