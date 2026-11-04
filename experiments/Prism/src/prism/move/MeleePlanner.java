package prism.move;

import prism.Profile;
import prism.core.Angles;
import prism.core.Drive;
import prism.core.Physics;
import prism.core.Rules;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Minimum-risk melee movement (specification section 10). Candidate destinations are ranked by
 * the declared risk function and checked with at least eight actual predictor steps for walls
 * and bounded opponent-collision risk. An all-unsafe candidate set yields a labelled emergency
 * control rather than the last unsafe candidate.
 */
public final class MeleePlanner {
    public record Opponent(int id, double x, double y, double energy, double threat, int scanAge, double vx, double vy) {}

    public record Result(double destX, double destY, Drive.Control control, double risk, String reason, int candidates) {}

    private final Rules rules;
    private final SplittableRandom rng;
    private final ArrayDeque<double[]> recent = new ArrayDeque<>();
    private double destX = Double.NaN, destY = Double.NaN;
    private int lastSampleTurn = Integer.MIN_VALUE;

    public MeleePlanner(Rules rules, long seed) { this.rules = rules; this.rng = new SplittableRandom(seed); }

    public void resetRound() { recent.clear(); destX = destY = Double.NaN; lastSampleTurn = Integer.MIN_VALUE; }

    public void record(int turn, double x, double y) {
        if (turn - lastSampleTurn >= 8) {
            recent.addLast(new double[] {x, y});
            while (recent.size() > 32) recent.removeFirst();
            lastSampleTurn = turn;
        }
    }

    public double risk(double x, double y, double selfX, double selfY, double ownEnergy, List<Opponent> opps) {
        double travel = Angles.bearing(selfX, selfY, x, y);
        double sum = 0;
        for (Opponent o : opps) {
            double dx = x - o.x(), dy = y - o.y();
            double dist2 = dx * dx + dy * dy;
            double energyRatio = Math.max(0.25, Math.min(4, o.energy() / Math.max(ownEnergy, 0.1)));
            double align = 1 + Math.abs(Math.cos(Angles.bearing(selfX, selfY, o.x(), o.y()) - travel));
            double closer = 0;
            for (Opponent p : opps) {
                if (p == o) continue;
                double credit = p.scanAge() > 8 ? 0 : 1 - p.scanAge() / 9.0;
                if (credit <= 0) continue;
                double dp = Math.hypot(p.x() - o.x(), p.y() - o.y());
                if (dp < Math.sqrt(dist2)) closer += credit;
            }
            sum += o.threat() * energyRatio * align / ((dist2 + 36 * 36) * (1 + closer));
        }
        double revisit = 1;
        for (double[] r : recent) { double dx = x - r[0], dy = y - r[1]; revisit += 100 / (dx * dx + dy * dy + 100); }
        return sum * revisit;
    }

    public Result plan(int turn, Physics.Motion self, double ownEnergy, List<Opponent> opps) {
        double nearest = Double.POSITIVE_INFINITY;
        for (Opponent o : opps) nearest = Math.min(nearest, Math.hypot(o.x() - self.x(), o.y() - self.y()));
        List<double[]> cands = new ArrayList<>();
        for (int i = 0; i < Profile.MELEE_DESTINATIONS; i++) {
            double a = Angles.TWO_PI * i / Profile.MELEE_DESTINATIONS;
            double r = Profile.MELEE_MIN_RADIUS + rng.nextDouble() * (Profile.MELEE_MAX_RADIUS - Profile.MELEE_MIN_RADIUS);
            r = Math.min(r, Math.max(40, nearest * 0.8));
            cands.add(new double[] {rules.clampX(self.x() + Math.cos(a) * r, 20), rules.clampY(self.y() + Math.sin(a) * r, 20), 0});
        }
        if (!Double.isNaN(destX)) cands.add(new double[] {destX, destY, 1});
        double inx = rules.width() / 2 - self.x(), iny = rules.height() / 2 - self.y();
        double inl = Math.max(1, Math.hypot(inx, iny));
        cands.add(new double[] {rules.clampX(self.x() + inx / inl * 120, 20), rules.clampY(self.y() + iny / inl * 120, 20), 0});
        // deduplicate
        List<double[]> uniq = new ArrayList<>();
        for (double[] c : cands) {
            boolean dup = false;
            for (double[] u : uniq) if (Math.hypot(u[0] - c[0], u[1] - c[1]) < 1) { dup = true; u[2] = Math.max(u[2], c[2]); break; }
            if (!dup) uniq.add(c);
        }
        List<double[]> ranked = new ArrayList<>();
        for (double[] c : uniq) {
            double risk = risk(c[0], c[1], self.x(), self.y(), ownEnergy, opps);
            if (c[2] > 0) risk *= Profile.MELEE_RETAIN;
            ranked.add(new double[] {c[0], c[1], risk});
        }
        ranked.sort((p, q) -> Double.compare(p[2], q[2]));
        for (double[] c : ranked) {
            Drive.Control first = steer(self, c[0], c[1]);
            if (safe(self, c[0], c[1], opps)) {
                destX = c[0]; destY = c[1];
                return new Result(c[0], c[1], first, c[2], "min-risk", ranked.size());
            }
        }
        // all unsafe: emergency control minimizing imminent damage
        Drive.Control best = Drive.stop();
        double bestDmg = Double.POSITIVE_INFINITY;
        for (Drive.Control c : new Drive.Control[] {Drive.stop(), new Drive.Control(-Math.signum(self.speed()) * 8, 0),
                new Drive.Control(8, Math.toRadians(10)), new Drive.Control(-8, Math.toRadians(-10))}) {
            Physics.Step s = Physics.step(self, c.targetSpeed(), c.turnRate(), rules);
            double dmg = s.wallDamage() + (s.hitWall() ? 0.1 : 0) + collisionRisk(s.next(), 1, opps);
            if (dmg < bestDmg) { bestDmg = dmg; best = c; }
        }
        destX = Double.NaN;
        return new Result(self.x(), self.y(), best, Double.POSITIVE_INFINITY, "emergency", ranked.size());
    }

    Drive.Control steer(Physics.Motion m, double x, double y) {
        double d = Math.hypot(x - m.x(), y - m.y());
        if (d < 2) return Drive.stop();
        return Drive.toward(m, Angles.bearing(m.x(), m.y(), x, y), Math.min(Rules.MAX_SPEED, Math.max(2, d / 4)));
    }

    boolean safe(Physics.Motion start, double x, double y, List<Opponent> opps) {
        Physics.Motion m = start;
        for (int i = 1; i <= Profile.MELEE_SAFE_STEPS; i++) {
            Drive.Control c = steer(m, x, y);
            Physics.Step s = Physics.step(m, c.targetSpeed(), c.turnRate(), rules);
            if (s.hitWall()) return false;
            m = s.next();
            if (collisionRisk(m, i, opps) > 0) return false;
        }
        return true;
    }

    static double collisionRisk(Physics.Motion m, int ahead, List<Opponent> opps) {
        double r = 0;
        for (Opponent o : opps) {
            if (o.scanAge() > 16) continue;
            double px = o.x() + o.vx() * (ahead + o.scanAge()), py = o.y() + o.vy() * (ahead + o.scanAge());
            if (Math.hypot(m.x() - px, m.y() - py) < 2 * Rules.BOT_RADIUS + 4) r += Rules.RAM_DAMAGE;
        }
        return r;
    }
}
