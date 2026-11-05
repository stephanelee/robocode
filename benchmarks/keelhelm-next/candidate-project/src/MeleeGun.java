import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/** Per-opponent virtual guns trained only from confirmed shots and legal target scans. */
final class MeleeGun {
    static final int HEAD_ON = 0, LINEAR = 1, CIRCULAR = 2, MIN_SAMPLES = 16;
    static final class Scores {
        final double[] hit = {0.18, 0.18, 0.18};
        int samples;
    }
    static final class Wave {
        final int target, fireTurn;
        final double x, y, speed;
        final double[] angles;
        int previousTurn;
        double previousX, previousY;
        Wave(int id, int t, Phys.Pose self, double power, double[] angles, MeleePolicy.Opponent o) {
            target = id; fireTurn = t; x = self.x(); y = self.y(); speed = Phys.bulletSpeed(power);
            this.angles = angles.clone(); previousTurn = o.turn; previousX = o.x; previousY = o.y;
        }
        double radius(double t) { return (t - fireTurn + 1) * speed; }
    }
    private final Map<Integer, Scores> scores = new HashMap<>();
    private final ArrayList<Wave> waves = new ArrayList<>();
    private Wave pending;
    private double pendingPower;
    private int pendingModel;
    private boolean pendingPowerChanged;
    int confirmed, resolved;
    int changedPowerShots;
    final int[] choices = new int[3];

    void newRound() { clearTransient(); java.util.Arrays.fill(choices, 0); changedPowerShots = 0; }
    void clearTransient() { waves.clear(); pending = null; }
    void died(int id) { waves.removeIf(w -> w.target == id); if (pending != null && pending.target == id) pending = null; }
    int samples(int id) { Scores s = scores.get(id); return s == null ? 0 : s.samples; }
    double accuracy(int id, int model) {
        Scores s = scores.get(id);
        return s == null || s.samples < MIN_SAMPLES ? 0.18 : s.hit[model];
    }
    int choice(int id, boolean adaptive) {
        Scores s = scores.get(id);
        if (!adaptive || s == null || s.samples < MIN_SAMPLES) return CIRCULAR;
        int best = CIRCULAR;
        for (int model : new int[]{LINEAR, HEAD_ON}) if (s.hit[model] > s.hit[best]) best = model;
        return best;
    }
    static double[] angles(MeleePolicy.Opponent o, int turn, Phys.Pose self, double power, double w, double h) {
        var linear = new MeleePolicy.Opponent(o.id);
        linear.turn = o.turn; linear.x = o.x; linear.y = o.y; linear.heading = o.heading;
        linear.speed = o.speed; linear.energy = o.energy;
        Phys.Pose l = MeleePolicy.intercept(linear, turn, self, power, w, h);
        Phys.Pose c = MeleePolicy.intercept(o, turn, self, power, w, h);
        return new double[]{Phys.bearing(self.x(), self.y(), o.x, o.y),
                Phys.bearing(self.x(), self.y(), l.x(), l.y()), Phys.bearing(self.x(), self.y(), c.x(), c.y())};
    }
    void queue(int fireTurn, Phys.Pose self, double power, double[] angles, MeleePolicy.Opponent target, int model) {
        queue(fireTurn,self,power,angles,target,model,false);
    }
    void queue(int fireTurn, Phys.Pose self, double power, double[] angles, MeleePolicy.Opponent target, int model, boolean changedPower) {
        pending = new Wave(target.id, fireTurn, self, power, angles, target);
        pendingPower = power; pendingModel = model; pendingPowerChanged = changedPower;
    }
    void confirm(int fireTurn, double power) {
        Wave shot = pending; pending = null;
        if (shot == null || shot.fireTurn != fireTurn || Math.abs(power - pendingPower) > 1e-6) return;
        if (waves.stream().filter(w -> w.target == shot.target).count() >= 128) {
            for (Iterator<Wave> it = waves.iterator(); it.hasNext();) if (it.next().target == shot.target) { it.remove(); break; }
        }
        waves.add(shot); confirmed++; choices[pendingModel]++;
        if (pendingPowerChanged) changedPowerShots++;
    }
    void observe(int id, int turn, double x, double y) {
        for (Iterator<Wave> it = waves.iterator(); it.hasNext();) {
            Wave shot = it.next();
            if (shot.target != id || turn <= shot.previousTurn || turn < shot.fireTurn) continue;
            double distance = Phys.distance(shot.x, shot.y, x, y);
            if (shot.radius(turn) >= distance) {
                double lo = Math.max(shot.previousTurn, shot.fireTurn - 1), hi = turn;
                for (int i = 0; i < 12; i++) {
                    double mid = (lo + hi) / 2;
                    double f = (mid - shot.previousTurn) / (turn - shot.previousTurn);
                    double px = shot.previousX + f * (x - shot.previousX), py = shot.previousY + f * (y - shot.previousY);
                    if (shot.radius(mid) >= Phys.distance(shot.x, shot.y, px, py)) hi = mid; else lo = mid;
                }
                double f = (hi - shot.previousTurn) / (turn - shot.previousTurn);
                double px = shot.previousX + f * (x - shot.previousX), py = shot.previousY + f * (y - shot.previousY);
                double bearing = Phys.bearing(shot.x, shot.y, px, py);
                double halfWidth = Math.toDegrees(Math.atan(18 / Math.max(18, Phys.distance(shot.x, shot.y, px, py))));
                Scores s = scores.computeIfAbsent(id, unused -> new Scores());
                for (int model = 0; model < 3; model++) {
                    double hit = Math.abs(Phys.norm(shot.angles[model] - bearing)) <= halfWidth ? 1 : 0;
                    s.hit[model] = .97 * s.hit[model] + .03 * hit;
                }
                s.samples++; resolved++; it.remove();
            } else {
                shot.previousTurn = turn; shot.previousX = x; shot.previousY = y;
            }
        }
    }
    double power(MeleePolicy.Opponent o, double distance, double energy, int model, boolean adaptive) {
        double p = MeleePolicy.power(distance, energy, o.energy);
        if (!adaptive || samples(o.id) < MIN_SAMPLES) return p;
        double accuracy = accuracy(o.id, model);
        if (accuracy < .18) p *= .7;
        else if (accuracy > .30 && distance < 400 && energy >= 20) p *= 1.2;
        if (energy < 20) p = Math.min(p, .5 + energy / 20);
        p = Math.min(p, o.energy <= 4 ? o.energy / 4 + .05 : (o.energy + 2) / 6 + .05);
        return Phys.clip(p, .1, 3);
    }
}
