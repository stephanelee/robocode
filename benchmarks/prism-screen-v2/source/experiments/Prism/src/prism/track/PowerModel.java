package prism.track;

import prism.Profile;
import prism.core.Rules;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

/**
 * Enemy firepower predictor for forecast waves: averages the powers of the closest
 * min(16, max(1, floor(sqrt(N)))) reliable shot observations in (range, enemy energy, own
 * energy) space with equal weights. Cold default 2.0. Never changes an observed shot's power.
 */
public final class PowerModel {
    private record Obs(double range, double enemyEnergy, double ownEnergy, double power) {}

    private final ArrayDeque<Obs> obs = new ArrayDeque<>();

    public void add(double range, double enemyEnergy, double ownEnergy, double power) {
        if (!(Double.isFinite(range) && Double.isFinite(enemyEnergy) && Double.isFinite(ownEnergy) && Double.isFinite(power))) return;
        obs.addLast(new Obs(range / 1000, enemyEnergy / 100, ownEnergy / 100, power));
        while (obs.size() > Profile.ENEMY_POWER_CAPACITY) obs.removeFirst();
    }

    public int size() { return obs.size(); }

    public double predict(double range, double enemyEnergy, double ownEnergy) {
        double p;
        if (obs.isEmpty()) p = Profile.COLD_ENEMY_POWER;
        else {
            double qr = range / 1000, qe = enemyEnergy / 100, qo = ownEnergy / 100;
            List<double[]> d = new ArrayList<>();
            int seq = 0;
            for (Obs o : obs) {
                double dist = sq(o.range - qr) + sq(o.enemyEnergy - qe) + sq(o.ownEnergy - qo);
                d.add(new double[] {dist, seq++, o.power});
            }
            d.sort((a, b) -> a[0] != b[0] ? Double.compare(a[0], b[0]) : Double.compare(b[1], a[1]));
            int k = Math.min(16, Math.max(1, (int) Math.floor(Math.sqrt(obs.size()))));
            double s = 0;
            for (int i = 0; i < k; i++) s += d.get(i)[2];
            p = s / k;
        }
        double affordable = Math.max(Rules.MIN_POWER, enemyEnergy - 1e-6);
        return Math.max(Rules.MIN_POWER, Math.min(Rules.MAX_POWER, Math.min(p, affordable)));
    }

    private static double sq(double v) { return v * v; }
}
