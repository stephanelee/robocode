import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Map;
import java.util.TreeMap;
import java.util.HashSet;
import java.util.Set;

/** Bounded melee policy over legal per-opponent scans; no duel learning is shared here. */
final class MeleePolicy {
    static final int TARGET_MAX_AGE = 32;
    static final class Opponent {
        final int id;
        int turn;
        double x, y, heading, speed, energy, turnRate;
        Opponent(int id) { this.id = id; turn = -1; }
    }
    record Command(double turn, double speed) {}
    private final Map<Integer, Opponent> opponents = new TreeMap<>();
    private final Set<Integer> dead = new HashSet<>();
    private final ArrayDeque<Phys.Pose> recent = new ArrayDeque<>();
    private boolean active;
    private boolean enteredMelee;
    private int targetId = -1;
    private double destinationX = Double.NaN, destinationY;
    int smartDifferences, riskDifferences;

    void newRound(int enemies) {
        opponents.clear(); dead.clear(); recent.clear(); targetId = -1;
        destinationX = Double.NaN; active = enemies > 1; enteredMelee = active;
        smartDifferences = 0; riskDifferences = 0;
    }
    boolean active() { return active; }
    void activate() { active = true; enteredMelee = true; }
    /** True exactly once at the boundary; the caller must clear transient duel state. */
    boolean enterDuel(int enemies) {
        if (!active || enemies > 1) return false;
        active = false; opponents.clear(); recent.clear(); targetId = -1;
        destinationX = Double.NaN;
        return true;
    }
    void observe(int id, int turn, double x, double y, double heading, double speed, double energy) {
        if (isDead(id)) return;
        activate();
        Opponent o = opponents.computeIfAbsent(id, Opponent::new);
        if (turn <= o.turn) return;
        int elapsed = turn - o.turn;
        o.turnRate = o.turn < 0 ? 0 : Phys.clip(Phys.norm(heading - o.heading) / elapsed, -10, 10);
        o.turn = turn; o.x = x; o.y = y; o.heading = heading; o.speed = speed; o.energy = energy;
    }
    void died(int id) { dead.add(id); opponents.remove(id); if (targetId == id) targetId = -1; }
    boolean isDead(int id) { return enteredMelee && dead.contains(id); }
    Collection<Opponent> opponents() { return opponents.values(); }

    Opponent targetSmart(int turn, Phys.Pose self, double gunHeading, double energy,
                         double w, double h, MeleeGun gun, int features) {
        Opponent best = null, parent = null; double bestCost = Double.POSITIVE_INFINITY, parentCost = bestCost;
        for (Opponent o : opponents.values()) {
            if (turn - o.turn > TARGET_MAX_AGE) continue;
            int model = gun.choice(o.id, (features & 1) != 0);
            double d = Phys.distance(self.x(), self.y(), o.x, o.y);
            double baseCost = d * (.7 + Math.min(100,o.energy)/200) * (1+(turn-o.turn)/16.0);
            if (o.id == targetId) baseCost *= .8;
            if (baseCost < parentCost) { parent = o; parentCost = baseCost; }
            double p = gun.power(o, d, energy, model, (features & 2) != 0);
            double aim = MeleeGun.angles(o, turn, self, p, w, h)[model];
            double cost = (Math.abs(Phys.norm(aim - gunHeading)) / 20 + d / Phys.bulletSpeed(p) + 8)
                    / Math.max(.05, gun.accuracy(o.id, model)) * (.75 + Math.min(o.energy, 100) / 200);
            if (o.id == targetId) cost *= .8;
            if (cost < bestCost) { best = o; bestCost = cost; }
        }
        if (best != parent) smartDifferences++;
        targetId = best == null ? -1 : best.id;
        return best;
    }

    Opponent target(int turn, Phys.Pose self) {
        Opponent best = null;
        double bestCost = Double.POSITIVE_INFINITY;
        for (Opponent o : opponents.values()) {
            if (turn - o.turn > TARGET_MAX_AGE) continue;
            double cost = Phys.distance(self.x(), self.y(), o.x, o.y)
                    * (0.7 + Math.min(100, o.energy) / 200) * (1 + (turn - o.turn) / 16.0);
            if (o.id == targetId) cost *= 0.8;
            if (cost < bestCost) { best = o; bestCost = cost; }
        }
        targetId = best == null ? -1 : best.id;
        return best;
    }

    double revisitRadar(int turn, Phys.Pose self, double radarHeading, int sweepSign, int enemyCount, Opponent target) {
        double sweep = 45 * sweepSign;
        if ((turn & 7) < 6 || target == null || opponents.size() != enemyCount || turn - target.turn > 8) return sweep;
        for (Opponent o : opponents.values()) if (turn - o.turn > 10) return sweep;
        double error = Phys.norm(Phys.bearing(self.x(), self.y(), target.x, target.y) - radarHeading);
        if (Math.abs(error) > 45) return sweep;
        return Phys.clip(error + (error < 0 ? -15 : 15), -45, 45);
    }

    boolean obstructed(int turn, Phys.Pose self, Opponent target, double aim, double power, double w, double h) {
        double speed = Phys.bulletSpeed(power), distance = Phys.distance(self.x(), self.y(), target.x, target.y);
        int flight = Math.min(120, (int)Math.ceil(distance / speed));
        for (Opponent o : opponents.values()) {
            if (o.id == target.id || turn - o.turn > 8) continue;
            Phys.Pose p = new Phys.Pose(o.x, o.y, o.heading, o.speed);
            double rate = Phys.clip(o.turnRate, -Phys.maxTurn(o.speed), Phys.maxTurn(o.speed));
            for (int stale = 0; stale < Math.max(0, turn - o.turn); stale++) p = Phys.step(p, rate, o.speed, w, h);
            for (int t = 1; t <= flight && t * speed < distance - 18; t++) {
                p = Phys.step(p, rate, o.speed, w, h);
                if (Phys.distance(self.x() + t * speed * Phys.cos(aim), self.y() + t * speed * Phys.sin(aim), p.x(), p.y()) <= 18) return true;
            }
        }
        return false;
    }

    static double power(double distance, double ownEnergy, double enemyEnergy) {
        double p = distance < 250 ? 2.2 : distance < 500 ? 1.6 : 1;
        if (ownEnergy < 20) p = Math.min(p, 0.5 + ownEnergy / 20);
        p = Math.min(p, enemyEnergy <= 4 ? enemyEnergy / 4 + 0.05 : (enemyEnergy + 2) / 6 + 0.05);
        return Phys.clip(p, 0.1, 3);
    }

    /** Bounded circular intercept; stale extrapolation is capped at one radar sweep. */
    static Phys.Pose intercept(Opponent o, int turn, Phys.Pose self, double power, double w, double h) {
        Phys.Pose p = new Phys.Pose(o.x, o.y, o.heading, o.speed);
        double rate = Phys.clip(o.turnRate, -Phys.maxTurn(o.speed), Phys.maxTurn(o.speed));
        for (int i = 0; i < Math.min(TARGET_MAX_AGE, Math.max(0, turn - o.turn)); i++)
            p = Phys.step(p, rate, o.speed, w, h);
        double bulletSpeed = Phys.bulletSpeed(power);
        for (int flight = 1; flight <= 120; flight++) {
            p = Phys.step(p, rate, o.speed, w, h);
            if (flight * bulletSpeed >= Phys.distance(self.x(), self.y(), p.x(), p.y())) break;
        }
        return p;
    }

    Command move(int turn, Phys.Pose self, double energy, double w, double h) {
        return move(turn, self, energy, w, h, false);
    }
    Command move(int turn, Phys.Pose self, double energy, double w, double h, boolean predictive) {
        Opponent[] threats = opponents.values().stream().filter(o -> turn - o.turn <= 64).toArray(Opponent[]::new);
        Phys.Pose[][] forecasts = predictive ? forecasts(threats, turn, w, h) : null;
        if (turn % 8 == 0) {
            recent.addLast(self);
            if (recent.size() > 32) recent.removeFirst();
        }
        double bestRisk = Double.POSITIVE_INFINITY, bestX = self.x(), bestY = self.y();
        double parentBestRisk = Double.POSITIVE_INFINITY, parentX = self.x(), parentY = self.y();
        for (int i = 0; i < 65; i++) {
            if (i == 64 && Double.isNaN(destinationX)) continue;
            double angle = i * 360.0 / 64;
            double x = i == 64 ? destinationX : Phys.clip(self.x() + 160 * Phys.cos(angle), 36, w - 36);
            double y = i == 64 ? destinationY : Phys.clip(self.y() + 160 * Phys.sin(angle), 36, h - 36);
            if (Phys.distance(self.x(), self.y(), x, y) < 25) continue;
            Phys.Pose p = self;
            double risk = 0, parentRisk = 0;
            for (int step = 1; step <= 24; step++) {
                Command c = toward(p, x, y);
                p = Phys.step(p, c.turn(), c.speed(), w, h);
                if (step % 4 == 0) {
                    double baseRisk = risk(turn,p,self,energy,w,h);
                    parentRisk += baseRisk;
                    risk += predictive ? predictedRisk(p,self,energy,w,h,threats,forecasts[step/4-1]) : baseRisk;
                }
            }
            if (i == 64) risk *= 0.9; // avoid jitter while the existing route remains safe
            if (i == 64) parentRisk *= .9;
            if (parentRisk < parentBestRisk) { parentBestRisk = parentRisk; parentX = x; parentY = y; }
            if (risk < bestRisk) { bestRisk = risk; bestX = x; bestY = y; }
        }
        destinationX = bestX; destinationY = bestY;
        if (predictive) {
            Command parent = toward(self,parentX,parentY), actual = toward(self,bestX,bestY);
            if (parent.turn() != actual.turn() || parent.speed() != actual.speed()) riskDifferences++;
        }
        return toward(self, bestX, bestY);
    }

    private double risk(int turn, Phys.Pose p, Phys.Pose origin, double energy, double w, double h) {
        double wall = Math.min(Math.min(p.x() - 18, w - 18 - p.x()), Math.min(p.y() - 18, h - 18 - p.y()));
        double risk = 30 / Math.pow(Math.max(1, wall + 12), 2);
        double go = Phys.bearing(origin.x(), origin.y(), p.x(), p.y());
        for (Opponent o : opponents.values()) {
            if (turn - o.turn > 64) continue;
            double d = Phys.distance(p.x(), p.y(), o.x, o.y);
            double radial = Math.abs(Phys.cos(go - Phys.bearing(origin.x(), origin.y(), o.x, o.y)));
            risk += 10000 * Phys.clip((o.energy + 20) / (energy + 20), 0.5, 2)
                    * (1 + radial) / Math.max(400, d * d);
            if (d < 70) risk += 20;
        }
        for (Phys.Pose old : recent) {
            double d = Phys.distance(p.x(), p.y(), old.x(), old.y());
            risk += 20 / (d * d + 400);
        }
        return risk;
    }

    static Phys.Pose[][] forecasts(Opponent[] threats, int turn, double w, double h) {
        Phys.Pose[][] result = new Phys.Pose[6][threats.length];
        for (int j = 0; j < threats.length; j++) {
            Opponent o = threats[j];
            Phys.Pose p = new Phys.Pose(o.x, o.y, o.heading, o.speed);
            double rate = Phys.clip(o.turnRate, -Phys.maxTurn(o.speed), Phys.maxTurn(o.speed));
            for (int stale = 0; stale < Math.min(32, Math.max(0, turn - o.turn)); stale++) p = Phys.step(p, rate, o.speed, w, h);
            for (int step = 1; step <= 24; step++) {
                p = Phys.step(p, rate, o.speed, w, h);
                if (step % 4 == 0) result[step / 4 - 1][j] = p;
            }
        }
        return result;
    }

    private double predictedRisk(Phys.Pose p, Phys.Pose origin, double energy, double w, double h,
                                 Opponent[] threats, Phys.Pose[] future) {
        double wall = Math.min(Math.min(p.x() - 18, w - 18 - p.x()), Math.min(p.y() - 18, h - 18 - p.y()));
        double risk = 30 / Math.pow(Math.max(1, wall + 12), 2);
        double go = Phys.bearing(origin.x(), origin.y(), p.x(), p.y());
        double[] distances = new double[threats.length];
        for (int j = 0; j < threats.length; j++) {
            Phys.Pose q = future[j]; double d = Phys.distance(p.x(), p.y(), q.x(), q.y());
            distances[j] = d;
            double radial = Math.abs(Phys.cos(go - Phys.bearing(origin.x(), origin.y(), q.x(), q.y())));
            risk += 10000 * Phys.clip((threats[j].energy + 20) / (energy + 20), .5, 2)
                    * (1 + radial) / Math.max(400, d * d);
            if (d < 70) risk += 20;
        }
        if (crossfire(p,future,distances)) risk *= 1.25;
        if (Math.abs(p.v()) < .01) risk += .2;
        for (Phys.Pose old : recent) {
            double d = Phys.distance(p.x(), p.y(), old.x(), old.y()); risk += 20 / (d * d + 400);
        }
        return risk;
    }

    static boolean crossfire(Phys.Pose p, Phys.Pose[] future, double[] distances) {
        for (int j=0;j<future.length;j++) for (int k=0;k<j;k++) {
            if (distances[j]<500 && distances[k]<500 && Math.abs(Phys.norm(
                Phys.bearing(p.x(),p.y(),future[j].x(),future[j].y())
                - Phys.bearing(p.x(),p.y(),future[k].x(),future[k].y())))>135) return true;
        }
        return false;
    }

    static Command toward(Phys.Pose self, double x, double y) {
        double turn = Phys.norm(Phys.bearing(self.x(), self.y(), x, y) - self.h());
        double speed = 8;
        if (Math.abs(turn) > 90) { turn = Phys.norm(turn + 180); speed = -8; }
        return new Command(turn, speed);
    }
}
