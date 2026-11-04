package prism.move;

import prism.Profile;
import prism.core.Angles;
import prism.core.Drive;
import prism.core.Geometry;
import prism.core.Physics;
import prism.core.Rules;
import prism.wave.Wave;

import java.util.ArrayList;
import java.util.List;

/**
 * Two-wave, three-action surfing planner (specification sections 9.1 and 9.3). Every candidate is
 * simulated with the engine-native predictor. Unknown enemy commands are modelled only by a bounded
 * constant-velocity collision hypothesis.
 */
public final class DuelPlanner {
    public enum Action { NEG, STOP, POS }

    /** A threat with its frozen density and merged shadow intervals for this turn. */
    public record Threat(Wave wave, Danger.Density density, List<Geometry.Interval> shadows, double damage, double shadowCap) {}

    public record Result(Action action, Drive.Control control, Physics.Motion next, double score, String reason,
                         boolean truncated, int steps, boolean firstStepSafe) {}

    private final Rules rules;
    private int steps;
    private boolean truncated;

    public DuelPlanner(Rules rules) { this.rules = rules; }

    public double desiredRange() { return Math.min(Profile.DESIRED_RANGE, 0.65 * Math.min(rules.width(), rules.height())); }

    /** Orbit control around (cx,cy) in mathematical sign s with distance control and wall smoothing. */
    public Drive.Control orbit(Physics.Motion m, double cx, double cy, int s, boolean surfing) {
        double desired = desiredRange();
        double range = Math.hypot(m.x() - cx, m.y() - cy);
        double a = Math.max(-0.8, Math.min(0.8, (range - desired) / desired));
        double gain = surfing ? Profile.SURF_GAIN : Profile.ORBIT_GAIN;
        double out = Angles.bearing(cx, cy, m.x(), m.y());
        double h = out + s * (Math.PI / 2 + gain * a);
        h = smooth(m.x(), m.y(), h, s);
        return Drive.toward(m, h, Rules.MAX_SPEED);
    }

    /** Rotates toward the orbit inside by one degree until a point WALL_STICK ahead is safe. */
    public double smooth(double x, double y, double h, int s) {
        double step = Math.toRadians(1);
        for (int i = 0; i <= Profile.WALL_SMOOTH_STEPS; i++) {
            double hh = h + s * step * i;
            if (rules.safe(x + Math.cos(hh) * Profile.WALL_STICK, y + Math.sin(hh) * Profile.WALL_STICK)) return Angles.heading(hh);
        }
        return Angles.heading(h);
    }

    Drive.Control control(Action a, Physics.Motion m, double cx, double cy, boolean surfing) {
        return switch (a) {
            case STOP -> Drive.stop();
            case POS -> orbit(m, cx, cy, 1, surfing);
            case NEG -> orbit(m, cx, cy, -1, surfing);
        };
    }

    /** Selects the next action. {@code previous} is the last executed orbit action (for ties). */
    public Result plan(Physics.Motion now, int turn, double ex, double ey, double evx, double evy,
                       List<Threat> threats, Action previous, boolean wallContactNow) {
        steps = 0;
        truncated = false;
        boolean surfing = !threats.isEmpty();
        List<Threat> two = threats.size() > 2 ? threats.subList(0, 2) : threats;
        Result best = null;
        List<Result> all = new ArrayList<>();
        for (Action a : Action.values()) {
            Drive.Control c0 = control(a, now, ex, ey, surfing);
            Physics.Step first = Physics.step(now, c0.targetSpeed(), c0.turnRate(), rules, wallContactNow);
            steps++;
            boolean safe = !first.hitWall();
            double bound = best == null ? Double.POSITIVE_INFINITY : best.score();
            double score = evaluate(a, now, turn, ex, ey, evx, evy, two, surfing, bound, wallContactNow);
            if (Double.isNaN(score)) continue; // pruned by bound: cannot win
            Result r = new Result(a, c0, first.next(), score, surfing ? "surf" : "orbit", truncated, steps, safe);
            all.add(r);
            if (best == null || better(r, best, previous)) best = r;
        }
        // first-step safety: a candidate must pass the next actual translation to win
        Result bestSafe = null;
        for (Result r : all) if (r.firstStepSafe() && (bestSafe == null || better(r, bestSafe, previous))) bestSafe = r;
        if (bestSafe != null) best = bestSafe;
        else best = emergency(now, ex, ey, wallContactNow);
        return new Result(best.action(), best.control(), best.next(), best.score(), best.reason(), truncated, steps, best.firstStepSafe());
    }

    private boolean better(Result r, Result b, Action previous) {
        if (r.score() < b.score() - 1e-6) return true;
        if (r.score() > b.score() + 1e-6) return false;
        if (r.action() == previous && b.action() != previous && previous != Action.STOP) return true;
        if (b.action() == previous && previous != Action.STOP) return false;
        if (r.action() != Action.STOP && b.action() == Action.STOP) return true;
        if (b.action() != Action.STOP && r.action() == Action.STOP) return false;
        return r.action() == Action.NEG && b.action() == Action.POS;
    }

    /** All first steps hit a wall: pick the control minimizing immediate collision damage. */
    private Result emergency(Physics.Motion now, double ex, double ey, boolean wallContact) {
        Drive.Control[] opts = {Drive.stop(), new Drive.Control(-Math.signum(now.speed()) * 8, 0),
                new Drive.Control(now.speed(), Math.toRadians(10)), new Drive.Control(now.speed(), Math.toRadians(-10))};
        Result best = null;
        for (Drive.Control c : opts) {
            Physics.Step s = Physics.step(now, c.targetSpeed(), c.turnRate(), rules, wallContact);
            Result r = new Result(Action.STOP, c, s.next(), s.wallDamage(), "emergency", truncated, steps, !s.hitWall());
            if (best == null || r.score() < best.score()) best = r;
        }
        return best;
    }

    private record Sim(List<Physics.Motion> path, int centerIndex, boolean complete, double extraCost) {}

    /**
     * Simulates action a until wave w is fully passed (or the cap), returning the path. Path index i
     * is the state at the end of turn startTurn+i+1.
     */
    private Sim simulate(Action a, Physics.Motion start, int startTurn, double ex, double ey, double evx, double evy,
                         Wave w, boolean surfing, boolean wallContact) {
        List<Physics.Motion> path = new ArrayList<>();
        Physics.Motion m = start;
        int center = -1;
        double extra = 0;
        boolean contact = wallContact;
        for (int i = 0; i < Profile.MAX_TRAJECTORY_TICKS; i++) {
            if (steps >= Profile.MAX_PREDICTOR_STEPS) { truncated = true; return new Sim(path, center, false, extra); }
            Drive.Control c = control(a, m, ex, ey, surfing);
            Physics.Step st = Physics.step(m, c.targetSpeed(), c.turnRate(), rules, contact);
            steps++;
            contact = st.hitWall();
            extra += st.wallDamage() + (st.hitWall() ? 0.5 : 0);
            m = st.next();
            path.add(m);
            int t = startTurn + i + 1;
            double pex = ex + evx * (t - startTurn), pey = ey + evy * (t - startTurn);
            if (Math.hypot(m.x() - pex, m.y() - pey) < 2 * Rules.BOT_RADIUS + 2) extra += Rules.RAM_DAMAGE;
            if (w == null) { if (i >= 8) return new Sim(path, i, true, extra); continue; }
            Wave.Phase p = w.phaseFor(t, m.x(), m.y());
            if (center < 0 && (p == Wave.Phase.CENTER_PASSED || p == Wave.Phase.PASSED)) center = i;
            if (p == Wave.Phase.PASSED) return new Sim(path, center, true, extra);
        }
        return new Sim(path, center, false, extra);
    }

    /** Returns NaN when pruned by the bound; otherwise the exact score of the best continuation. */
    private double evaluate(Action a, Physics.Motion now, int turn, double ex, double ey, double evx, double evy,
                            List<Threat> threats, boolean surfing, double bound, boolean wallContact) {
        Threat t1 = threats.isEmpty() ? null : threats.get(0);
        Sim s1 = simulate(a, now, turn, ex, ey, evx, evy, t1 == null ? null : t1.wave(), surfing, wallContact);
        double cost1 = s1.extraCost();
        if (t1 != null) cost1 += waveCost(t1, s1.path(), turn, now, s1.complete());
        if (threats.size() < 2) {
            Physics.Motion end = s1.path().isEmpty() ? now : s1.path().get(s1.path().size() - 1);
            return cost1 + rangePenalty(end, ex, ey);
        }
        if (cost1 >= bound) return Double.NaN; // nonnegative remaining cost: cannot beat the bound
        Threat t2 = threats.get(1);
        int ci = s1.centerIndex() < 0 ? s1.path().size() - 1 : s1.centerIndex();
        List<Physics.Motion> prefix = s1.path().subList(0, Math.max(0, ci + 1));
        Physics.Motion branchStart = prefix.isEmpty() ? now : prefix.get(prefix.size() - 1);
        int branchTurn = turn + prefix.size();
        double best2 = Double.POSITIVE_INFINITY;
        for (Action c : Action.values()) {
            Sim s2 = simulate(c, branchStart, branchTurn, ex + evx * prefix.size(), ey + evy * prefix.size(), evx, evy, t2.wave(), surfing, false);
            List<Physics.Motion> full = new ArrayList<>(prefix);
            full.addAll(s2.path());
            Physics.Motion end = full.isEmpty() ? now : full.get(full.size() - 1);
            double c2 = waveCost(t2, full, turn, now, s2.complete()) + s2.extraCost() + rangePenalty(end, ex, ey);
            best2 = Math.min(best2, c2);
            if (steps >= Profile.MAX_PREDICTOR_STEPS) { truncated = true; break; }
        }
        return cost1 + best2;
    }

    double rangePenalty(Physics.Motion m, double ex, double ey) {
        double r = Math.max(1, Math.hypot(m.x() - ex, m.y() - ey));
        double v = Math.max(0, desiredRange() / r - 1);
        return Profile.CLOSE_PENALTY * v * v;
    }

    /**
     * confidence * damage * integrated density * unshadowed fraction / max(1, timeToContact).
     * If the path never produced an interval although the wave was not shown to pass, a conservative
     * maximum-density fallback is used: missing geometry is never perfectly safe.
     */
    double waveCost(Threat th, List<Physics.Motion> path, int turn, Physics.Motion now, boolean complete) {
        Wave w = th.wave();
        List<Geometry.Interval> parts = new ArrayList<>();
        int contact = -1;
        for (int i = 0; i < path.size(); i++) {
            Geometry.Interval iv = w.footprint(turn + i + 1, path.get(i).x(), path.get(i).y());
            if (iv != null) { parts.add(iv); if (contact < 0) contact = i + 1; }
        }
        // the current position may already overlap
        Geometry.Interval nowIv = w.footprint(turn, now.x(), now.y());
        List<Geometry.Interval> merged = Geometry.union(parts);
        double prob = 0, unshadow = 1;
        if (merged.isEmpty()) {
            if (nowIv == null && wavePassedWithoutContact(w, turn, now)) return 0; // wave already beyond us
            double d = Math.hypot(now.x() - w.ox, now.y() - w.oy);
            double gw = 2 * Geometry.halfWidth(Math.max(d, 1)) / w.maxEscape;
            prob = th.density().max() * gw;
            contact = Math.max(1, contact);
        } else {
            double total = 0, shadowed = 0;
            for (Geometry.Interval iv : merged) {
                double g0 = w.orbitSign * iv.lo() / w.maxEscape, g1 = w.orbitSign * iv.hi() / w.maxEscape;
                prob += Danger.integrate(th.density(), Math.min(g0, g1), Math.max(g0, g1), w.maxEscape);
                total += iv.width();
                // shadows live in normalized [-pi, pi] pieces: compare in the same frame
                for (Geometry.Interval piece : Geometry.normalize(iv)) shadowed += Geometry.overlap(piece, th.shadows());
            }
            if (total > 0) unshadow = 1 - th.shadowCap() * Math.min(1, shadowed / total);
        }
        double ttc = Math.max(1, contact);
        return w.confidence * th.damage() * Math.max(0, prob) * unshadow / ttc;
    }

    private boolean wavePassedWithoutContact(Wave w, int turn, Physics.Motion now) {
        return w.innerRadius(turn) > Math.hypot(now.x() - w.ox, now.y() - w.oy) + Rules.BOT_RADIUS;
    }

    public int steps() { return steps; }
}
