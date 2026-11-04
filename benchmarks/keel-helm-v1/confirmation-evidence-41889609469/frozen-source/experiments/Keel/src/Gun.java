import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Guess-factor guns over legal scans. Every fresh scan launches a virtual wave from our position;
 * waves that carried a real (confirmed) bullet also train an anti-surfer gun. A virtual-gun score
 * on real waves chooses between them.
 */
final class Gun {
    static final class Wave {
        final int fireTurn;
        final double ox, oy, speed, direct, mea;
        final int latDir;
        final double[] features;
        final boolean real;
        final double[] predicted;
        Wave(int fireTurn, double ox, double oy, double power, double direct, int latDir, double[] f, boolean real, double[] predicted) {
            this.fireTurn = fireTurn; this.ox = ox; this.oy = oy; this.speed = Phys.bulletSpeed(power);
            this.direct = direct; this.mea = Phys.mea(power); this.latDir = latDir; this.features = f;
            this.real = real; this.predicted = predicted;
        }
        double radius(int T) { return (T - fireTurn + 1) * speed; }
    }

    private static final double[] MAIN_W = {4, 1.5, 2, 1, 2.5, 2, 1, 1.5};
    private static final double[] SURF_W = {3, 1, 2, 1, 2, 2, 1, 1};
    final Knn main = new Knn(MAIN_W, 40000);
    final Knn surf = new Knn(SURF_W, 400);
    final double[] score = {0.2, 0.1};
    private final List<Wave> waves = new ArrayList<>();
    int lastChoice = 0;

    void newRound() { waves.clear(); }

    double[] predictions(double[] f, double halfWidthGf) {
        return new double[]{pick(main, f, Math.min(50, Math.max(1, main.size() / 12)), halfWidthGf, 0),
                pick(surf, f, Math.min(12, Math.max(1, surf.size() / 4)), halfWidthGf, 1)};
    }

    private static double pick(Knn data, double[] f, int k, double hw, int mode) {
        if (data.size() == 0) return Double.NaN;
        var near = data.nearest(f, k);
        long last = data.seq();
        double bestScore = -1, best = 0;
        double bw = Math.max(0.02, hw * 0.8);
        for (var c : near) {
            double s = 0;
            for (var o : near) {
                double z = (c.e().gf() - o.e().gf()) / bw;
                double w = o.e().weight() / (1 + o.dist());
                if (mode == 1) w *= Math.exp(-(last - o.e().seq()) / 120.0) + 0.1;
                s += w * Math.exp(-0.5 * z * z);
            }
            if (s > bestScore) { bestScore = s; best = c.e().gf(); }
        }
        return best;
    }

    int choice() {
        lastChoice = surf.size() >= 10 && score[1] > score[0] ? 1 : 0;
        return lastChoice;
    }

    void launch(Wave w) { waves.add(w); }

    /** Resolves waves against a fresh scan of the target at the end of turn T. */
    void observe(int T, double x, double y) {
        for (Iterator<Wave> it = waves.iterator(); it.hasNext(); ) {
            Wave w = it.next();
            double D = Phys.distance(w.ox, w.oy, x, y);
            if (w.radius(T) < D - Phys.BOT_RADIUS * 0.5) continue;
            double gf = Phys.clip(Phys.norm(Phys.bearing(w.ox, w.oy, x, y) - w.direct) / w.mea * w.latDir, -1.2, 1.2);
            main.add(w.features, gf, w.real ? 1.0 : 0.5);
            if (w.real) {
                surf.add(w.features, gf, 1);
                double hw = Math.toDegrees(Math.atan(Phys.BOT_RADIUS / D)) / w.mea;
                for (int g = 0; g < score.length; g++) {
                    double p = w.predicted[g];
                    if (Double.isNaN(p)) continue;
                    double hit = Math.abs(p - gf) <= hw ? 1 : 0;
                    score[g] = 0.97 * score[g] + 0.03 * hit;
                }
            }
            it.remove();
        }
        if (waves.size() > 200) waves.remove(0);
    }
}
