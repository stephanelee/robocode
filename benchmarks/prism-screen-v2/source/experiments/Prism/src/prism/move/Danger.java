package prism.move;

import prism.Profile;
import prism.core.Geometry;
import prism.learn.Features;
import prism.learn.NeighborStore;
import prism.wave.Wave;

import java.util.ArrayList;
import java.util.List;

/**
 * Learned incoming danger for one enemy (specification section 9.2). All views use theoretical
 * guess-factor labels. Hit views learn from actual incoming aim evidence (bullet impacts on us and,
 * at half weight, bullet-bullet contacts); the flattener learns from reliable real-wave visits. The
 * two label kinds never share a store.
 */
public final class Danger {
    public final NeighborStore simple = new NeighborStore("hit-simple-v1", NeighborStore.LabelType.THEORETICAL_GF, Features.SIMPLE_W, Profile.MOVE_HIT_CAPACITY);
    public final NeighborStore context = new NeighborStore("hit-context-v1", NeighborStore.LabelType.THEORETICAL_GF, Features.CONTEXT_W, Profile.MOVE_HIT_CAPACITY);
    public final NeighborStore recent = new NeighborStore("hit-recent-v1", NeighborStore.LabelType.THEORETICAL_GF, Features.CONTEXT_W, Profile.RECENT_HIT_CAPACITY);
    public final NeighborStore visits = new NeighborStore("visit-v1", NeighborStore.LabelType.THEORETICAL_GF, Features.VISIT_W, Profile.MOVE_VISIT_CAPACITY);

    public int hitLabels;         // impacts plus contacts (aim evidence)
    public int rawHits;           // actual impacts only
    public int resolvedRealWaves; // real incoming waves resolved, excluding bullet-bullet contacts
    public int fallbacks;

    public void addHit(Wave w, double gf, boolean contact, int round, int turn) {
        double q = w.context.quality() * (contact ? 0.5 : 1.0) * Math.max(0.25, w.confidence);
        double[] lab = {gf};
        simple.add(w.context.simpleHit(), lab, q, round, turn, w.id);
        context.add(w.context.contextHit(), lab, q, round, turn, w.id);
        recent.add(w.context.contextHit(), lab, q, round, turn, w.id);
        hitLabels++;
    }

    public void addVisit(Wave w, double gf, int round, int turn) {
        visits.add(w.context.visit(), new double[] {gf}, w.context.quality() * w.labelQuality, round, turn, w.id);
    }

    /** 95% Wilson lower bound for the raw enemy hit rate. */
    public static double wilsonLower(int hits, int n) {
        if (n <= 0) return 0;
        double z = 1.959963984540054, p = hits / (double) n;
        double den = 1 + z * z / n;
        double c = p + z * z / (2 * n);
        double r = z * Math.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n));
        return Math.max(0, (c - r) / den);
    }

    public boolean flattenerEnabled() {
        return resolvedRealWaves >= Profile.FLATTENER_MIN_WAVES && wilsonLower(rawHits, resolvedRealWaves) > Profile.FLATTENER_MIN_LOWER_BOUND;
    }

    /** Frozen per-wave density in theoretical-GF space. */
    public static final class Density {
        final double sigma;            // in GF units
        final double priorWeight;
        final double linearGf;
        final List<double[]> views = new ArrayList<>(); // each: [weight, then pairs gf,w ...]
        Density(double sigma, double priorWeight, double linearGf) { this.sigma = sigma; this.priorWeight = priorWeight; this.linearGf = linearGf; }

        double gauss(double g, double mu) { double e = (g - mu) / sigma; return Math.exp(-0.5 * e * e) / (sigma * Math.sqrt(2 * Math.PI)); }

        double prior(double g) {
            return 0.5 * gauss(g, 0) + 0.3 * gauss(g, linearGf) + 0.2 * (Math.abs(g) <= 1 ? 0.5 : 0);
        }

        public double at(double g) {
            if (views.isEmpty()) return prior(g);
            double learned = 0, tw = 0;
            for (double[] v : views) {
                double s = 0, w = 0;
                for (int i = 1; i + 1 < v.length; i += 2) { s += v[i + 1] * gauss(g, v[i]); w += v[i + 1]; }
                if (w > 0) { learned += v[0] * s / w; tw += v[0]; }
            }
            if (tw <= 0) return prior(g);
            learned /= tw;
            return priorWeight * prior(g) + (1 - priorWeight) * learned;
        }

        /** Maximum density on a grid; used as a conservative fallback. */
        public double max() {
            double m = 0;
            for (int i = -60; i <= 60; i++) m = Math.max(m, at(i / 50.0));
            return m;
        }
    }

    public Density density(Wave w, double range, double linearGf) {
        double sigma = Math.max(1e-3, Geometry.halfWidth(Math.max(range, 1)) / w.maxEscape);
        double pw = Math.max(0.1, 1 - hitLabels / 32.0);
        Density d = new Density(sigma, pw, linearGf);
        if (hitLabels >= 1) d.views.add(view(simple, w.context.simpleHit(), Math.min(24, Math.max(1, simple.size() / 5)), 1));
        if (hitLabels >= 8) {
            d.views.add(view(context, w.context.contextHit(), Math.min(24, Math.max(1, context.size() / 5)), 2));
            d.views.add(view(recent, w.context.contextHit(), 5, 2));
        }
        if (flattenerEnabled()) d.views.add(view(visits, w.context.visit(), 32, 0.5));
        d.views.removeIf(v -> v.length < 3);
        return d;
    }

    private static double[] view(NeighborStore s, double[] q, int k, double weight) {
        List<NeighborStore.Neighbor> ns = s.query(q, k);
        double[] v = new double[1 + 2 * ns.size()];
        v[0] = weight;
        int i = 1;
        for (NeighborStore.Neighbor n : ns) {
            v[i++] = n.sample().label()[0];
            v[i++] = n.sample().quality() * n.inverseDistance() * n.recencyWeight();
        }
        return v;
    }

    /**
     * Integrated hit probability over a GF interval: nine-point trapezoid per segment, segments at
     * most 0.2 rad wide, at most 81 points; above the cap use max density times width.
     */
    public static double integrate(Density d, double gLo, double gHi, double maxEscape) {
        if (gHi < gLo) { double t = gLo; gLo = gHi; gHi = t; }
        double widthRad = (gHi - gLo) * maxEscape;
        int segs = Math.max(1, (int) Math.ceil(widthRad / 0.2 - 1e-12));
        if (segs * 8 + 1 > 81) return d.max() * (gHi - gLo);
        double total = 0;
        double sw = (gHi - gLo) / segs;
        for (int s = 0; s < segs; s++) {
            double a = gLo + s * sw;
            double acc = 0;
            for (int i = 0; i <= 8; i++) {
                double g = a + sw * i / 8.0;
                double wt = (i == 0 || i == 8) ? 0.5 : 1.0;
                acc += wt * d.at(g);
            }
            total += acc / 8.0 * sw;
        }
        return total;
    }
}
