package prism.gun;

import prism.Profile;
import prism.core.Angles;
import prism.core.Geometry;
import prism.core.Physics;
import prism.core.Rules;
import prism.learn.Features;
import prism.learn.NeighborStore;
import prism.wave.Wave;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Gun ensemble (specification section 8): analytic startup interception, long-history
 * contextual gun, recent multi-window gun, an anti-surfer gun trained only on confirmed bullets,
 * a melee displacement gun, and a virtual selector.
 * Predictions are frozen on waves before any outcome is consumed.
 */
public final class Guns {
    public static final String ANALYTIC = "analytic", LONG = "long", RECENT = "recent", SURFER = "surfer", DISPLACEMENT = "displacement", DIRECT = "direct";

    /** Exponential-mean rating of one gun over the shared resolved cohort. */
    public static final class Rating {
        public double mean, adjusted;
        public int n;
        void add(double raw, double adj) {
            if (n == 0) { mean = raw; adjusted = adj; }
            else { mean += Profile.SELECTOR_ALPHA * (raw - mean); adjusted += Profile.SELECTOR_ALPHA * (adj - adjusted); }
            n++;
        }
    }

    /** Per-target gun learning, retained across rounds within a game. */
    public static final class Data {
        public final NeighborStore longStore = new NeighborStore("long-v1", NeighborStore.LabelType.PRECISE_GF, Features.LONG_W, Profile.LONG_GUN_CAPACITY);
        public final NeighborStore[] recent = new NeighborStore[Profile.RECENT_GUN_CAPACITIES.length];
        /**
         * Anti-surfer views: precise-GF labels from engine-confirmed (physical) waves only. A wave
         * surfer reacts to the waves of real bullets, so these record the movement it shows when
         * actually shot at; virtual-wave crossings never enter them.
         */
        public final NeighborStore[] surfer = new NeighborStore[Profile.SURFER_GUN_CAPACITIES.length];
        public int physicalLabels;
        public final NeighborStore displacement = new NeighborStore("disp-v1", NeighborStore.LabelType.DISPLACEMENT, Features.MELEE_W, Profile.LONG_GUN_CAPACITY);
        public final Map<String, Rating> ratings = new LinkedHashMap<>();
        public int crossings;
        public String current = ANALYTIC;
        public int switches;
        Data() {
            for (int i = 0; i < recent.length; i++)
                recent[i] = new NeighborStore("recent-v1-" + Profile.RECENT_GUN_CAPACITIES[i], NeighborStore.LabelType.PRECISE_GF, Features.RECENT_W, Profile.RECENT_GUN_CAPACITIES[i]);
            for (int i = 0; i < surfer.length; i++)
                surfer[i] = new NeighborStore("surfer-v1-" + Profile.SURFER_GUN_CAPACITIES[i], NeighborStore.LabelType.PRECISE_GF, Features.RECENT_W, Profile.SURFER_GUN_CAPACITIES[i]);
        }
    }

    /** Everything an aim needs, frozen at the decision. */
    public record Aim(double ox, double oy, int fireTurn, Physics.Motion target, int scanTurn, double turnRatePerTurn,
                      double power, Features.Context context, int orbitSign, double escPos, double escNeg,
                      double gunHeading, boolean meleeOrigin) {
        public double reference() { return Angles.bearing(ox, oy, target.x(), target.y()); }
        public double range() { return Math.hypot(target.x() - ox, target.y() - oy); }
    }

    public record Choice(String gun, double angle, String fallback) {}

    private final Map<Integer, Data> data = new LinkedHashMap<>();
    private final Rules rules;

    public Guns(Rules rules) { this.rules = rules; }

    public Data data(int id) { return data.computeIfAbsent(id, k -> new Data()); }
    public Map<Integer, Data> all() { return data; }

    // ---------------------------------------------------------------- analytic

    /** Engine-native constant-velocity (and constant-turn) interception angles; empty if none. */
    public List<Double> analytic(Aim a) {
        List<Double> out = new ArrayList<>();
        Double lin = intercept(a, 0);
        if (lin != null) out.add(lin);
        if (Math.abs(a.turnRatePerTurn()) > 1e-4) {
            Double circ = intercept(a, a.turnRatePerTurn());
            if (circ != null) out.add(circ);
        }
        return out;
    }

    private Double intercept(Aim a, double turnRate) {
        double b = Rules.bulletSpeed(a.power());
        Physics.Motion m = a.target();
        int steps = a.fireTurn() - a.scanTurn();
        for (int i = 0; i < steps; i++) m = Physics.step(m, m.speed(), turnRate, rules).next();
        for (int k = 0; k < 200; k++) {
            double d = Math.hypot(m.x() - a.ox(), m.y() - a.oy());
            if ((k + 2) * b >= d) return Angles.bearing(a.ox(), a.oy(), m.x(), m.y());
            m = Physics.step(m, m.speed(), turnRate, rules).next();
        }
        return null;
    }

    // ---------------------------------------------------------------- learned guns

    private static double bandwidth(double range) { return Math.max(1e-4, 2 * Math.asin(Math.min(1, Rules.BOT_RADIUS / Math.max(range, 1e-6)))); }

    private record Kernel(double angle, double weight) {}

    private List<Kernel> gfKernels(NeighborStore store, double[] q, int k, Aim a) {
        List<Kernel> out = new ArrayList<>();
        double ref = a.reference();
        for (NeighborStore.Neighbor n : store.query(q, k)) {
            double g = n.sample().label()[0];
            double u = g >= 0 ? g * a.escPos() : g * a.escNeg();
            double w = n.sample().quality() * n.inverseDistance() * n.recencyWeight();
            if (w > 0 && Double.isFinite(w)) out.add(new Kernel(ref + a.orbitSign() * u, w));
        }
        return out;
    }

    private static double density(List<Kernel> ks, double angle, double bw) {
        double s = 0, w = 0;
        for (Kernel k : ks) { double e = Angles.wrap(angle - k.angle()) / bw; s += k.weight() * Math.exp(-0.5 * e * e); w += k.weight(); }
        return w > 0 ? s / w : 0;
    }

    /** Candidate angles: 61 offsets across the reachable interval, direct bearing and analytic angles. */
    public List<Double> candidates(Aim a) {
        List<Double> c = new ArrayList<>();
        double ref = a.reference();
        int n = Profile.GUN_OFFSETS;
        for (int i = 0; i < n; i++) {
            double g = -1 + 2.0 * i / (n - 1);
            double u = g >= 0 ? g * a.escPos() : g * a.escNeg();
            c.add(Angles.heading(ref + a.orbitSign() * u));
        }
        c.add(ref);
        c.addAll(analytic(a));
        return c;
    }

    /** Learned gun prediction, or null if the view has no valid evidence. */
    public Double learned(String gun, Aim a, Data d) {
        List<List<Kernel>> views = new ArrayList<>();
        if (gun.equals(LONG)) {
            int N = d.longStore.size();
            if (N == 0) return null;
            views.add(gfKernels(d.longStore, a.context().longGun(), Math.min(160, Math.max(1, N / 12)), a));
        } else if (gun.equals(RECENT)) {
            for (NeighborStore s : d.recent) if (s.size() > 0) views.add(gfKernels(s, a.context().recentGun(), 5, a));
        } else if (gun.equals(SURFER)) {
            if (d.physicalLabels < Profile.SURFER_GUN_MIN_LABELS) return null;
            for (NeighborStore s : d.surfer) if (s.size() > 0) views.add(gfKernels(s, a.context().recentGun(), Profile.SURFER_GUN_NEIGHBORS, a));
        } else return null;
        views.removeIf(List::isEmpty);
        if (views.isEmpty()) return null;
        double bw = bandwidth(a.range());
        return best(a, views, bw);
    }

    private Double best(Aim a, List<List<Kernel>> views, double bw) {
        double ref = a.reference();
        Double bestAngle = null;
        double bestD = -1, bestRot = 0, bestOff = 0;
        for (double c : candidates(a)) {
            double dsum = 0;
            for (List<Kernel> v : views) dsum += density(v, c, bw);
            dsum /= views.size();
            double rot = Math.abs(Angles.wrap(c - a.gunHeading()));
            double off = Angles.wrap(c - ref);
            boolean better;
            if (bestAngle == null || dsum > bestD + 1e-12) better = true;
            else if (dsum < bestD - 1e-12) better = false;
            else if (rot < bestRot - 1e-12) better = true;
            else if (rot > bestRot + 1e-12) better = false;
            else if (Math.abs(off) < Math.abs(bestOff) - 1e-12) better = true;
            else better = Math.abs(off) <= Math.abs(bestOff) + 1e-12 && off < bestOff;
            if (better) { bestAngle = c; bestD = dsum; bestRot = rot; bestOff = off; }
        }
        return bestD > 0 ? bestAngle : null;
    }

    /** Guns eligible for prediction freezing for this target. */
    public List<String> eligible(Data d) {
        List<String> g = new ArrayList<>();
        g.add(ANALYTIC);
        if (d.crossings >= Profile.LEARNED_GUN_MIN_CROSSINGS) { g.add(LONG); g.add(RECENT); }
        if (d.physicalLabels >= Profile.SURFER_GUN_MIN_LABELS) g.add(SURFER);
        return g;
    }

    public Double predict(String gun, Aim a, Data d) {
        if (gun.equals(ANALYTIC)) { List<Double> an = analytic(a); return an.isEmpty() ? null : an.get(0); }
        if (gun.equals(DIRECT)) return a.reference();
        return learned(gun, a, d);
    }

    /** The selected gun's angle with the required fallback chain recent → long → analytic → direct. */
    public Choice choose(Aim a, Data d) {
        String sel = d.current;
        Double v = predict(sel, a, d);
        if (v != null) return new Choice(sel, v, null);
        for (String g : new String[] {RECENT, LONG, ANALYTIC}) {
            if (g.equals(sel)) continue;
            if (!g.equals(ANALYTIC) && d.crossings < Profile.LEARNED_GUN_MIN_CROSSINGS) continue;
            v = predict(g, a, d);
            if (v != null) return new Choice(g, v, sel + "->" + g);
        }
        return new Choice(DIRECT, a.reference(), sel + "->direct");
    }

    // ---------------------------------------------------------------- learning and rating

    /** Records a resolved crossing label for training (precise GF). Exactly once per wave. */
    public boolean train(Wave w, int round, int turn) {
        if (w.labelEmitted || w.cancelled || Double.isNaN(w.labelBearing)) return false;
        Data d = data(w.targetId);
        double q = w.labelQuality * w.context.quality();
        boolean ok;
        if (w.context.meleeOrigin()) ok = true; // displacement labels are added separately
        else {
            double g = w.preciseGf(w.labelBearing);
            ok = d.longStore.add(w.context.longGun(), new double[] {g}, q, round, turn, w.id);
            for (NeighborStore s : d.recent) s.add(w.context.recentGun(), new double[] {g}, q, round, turn, w.id);
            if (w.physical && w.role == Wave.Role.OUTGOING) {
                boolean added = false;
                for (NeighborStore s : d.surfer) added |= s.add(w.context.recentGun(), new double[] {g}, q, round, turn, w.id);
                if (added) d.physicalLabels++;
            }
        }
        w.labelEmitted = true;
        if (ok) d.crossings++;
        return ok;
    }

    /** Scores every frozen prediction on a resolved, rated wave; updates the selector. */
    public void rate(Wave w) {
        if (!w.fireOpportunity || w.cancelled || Double.isNaN(w.labelBearing) || w.predictions.isEmpty()) return;
        Data d = data(w.targetId);
        double h = Math.max(w.labelHalfWidth, 1e-4);
        double difficulty = Math.min(Profile.DIFFICULTY_CAP, (w.escapePositive + w.escapeNegative) / (2 * h));
        for (Map.Entry<String, Double> e : w.predictions.entrySet()) {
            double err = Math.abs(Angles.wrap(e.getValue() - w.labelBearing));
            double raw = Math.exp(-0.5 * (err / h) * (err / h));
            d.ratings.computeIfAbsent(e.getKey(), k -> new Rating()).add(raw, raw * difficulty);
        }
        select(d);
    }

    static void select(Data d) {
        Rating cur = d.ratings.get(d.current);
        for (Map.Entry<String, Rating> e : d.ratings.entrySet()) {
            if (e.getKey().equals(d.current)) continue;
            Rating ch = e.getValue();
            if (cur == null || cur.n < Profile.SELECTOR_MIN_RESOLVED || ch.n < Profile.SELECTOR_MIN_RESOLVED) continue;
            if (ch.mean > cur.mean + Profile.SELECTOR_MARGIN) { d.current = e.getKey(); cur = ch; d.switches++; }
        }
    }

    // ---------------------------------------------------------------- displacement (melee)

    public record Target(int id, double ox, double oy, int fireTurn, Physics.Motion motion, int scanTurn,
                         Features.Context context, int orbitSign, double power) {}

    public record Projection(int targetId, double angle, double range, double weight) {}

    /** Displacement label: average velocity rotated into the movement frame, reflected by orbit sign. */
    public static double[] displacementLabel(Physics.Motion origin, int originTurn, double cx, double cy, int crossTurn, int orbitSign) {
        int dt = Math.max(1, crossTurn - originTurn);
        double vx = (cx - origin.x()) / dt, vy = (cy - origin.y()) / dt;
        double h = Angles.effectiveHeading(origin.heading(), origin.speed());
        double along = vx * Math.cos(h) + vy * Math.sin(h);
        double across = -vx * Math.sin(h) + vy * Math.cos(h);
        return new double[] {along, across * (orbitSign == 0 ? 1 : orbitSign)};
    }

    /** Projects one displacement label in a new context; null if invalid or out of the field. */
    public Projection project(Target t, double[] label, double weight) {
        Physics.Motion m = t.motion();
        double h = Angles.effectiveHeading(m.heading(), m.speed());
        int s = t.orbitSign() == 0 ? 1 : t.orbitSign();
        double vx = label[0] * Math.cos(h) - s * label[1] * Math.sin(h);
        double vy = label[0] * Math.sin(h) + s * label[1] * Math.cos(h);
        double b = Rules.bulletSpeed(t.power());
        int lead = t.fireTurn() - t.scanTurn();
        double tau = Math.hypot(m.x() - t.ox(), m.y() - t.oy()) / b, prev = Double.NaN, prev2 = Double.NaN;
        double qx = m.x(), qy = m.y();
        for (int i = 0; i < 12; i++) {
            double total = lead + tau;
            qx = m.x() + vx * total; qy = m.y() + vy * total;
            double next = Math.max(0, Math.hypot(qx - t.ox(), qy - t.oy()) / b - 1.5);
            if (!Double.isNaN(prev2) && Math.abs(next - prev2) < 1e-9 && Math.abs(next - prev) > 1e-9) { tau = Math.max(next, prev); break; }
            if (Math.abs(next - tau) < 1e-3) { tau = next; break; }
            prev2 = prev; prev = tau; tau = next;
        }
        double total = lead + tau;
        qx = m.x() + vx * total; qy = m.y() + vy * total;
        if (!(qx >= Rules.BOT_RADIUS && qx <= rules.width() - Rules.BOT_RADIUS && qy >= Rules.BOT_RADIUS && qy <= rules.height() - Rules.BOT_RADIUS)) return null;
        double r = Math.hypot(qx - t.ox(), qy - t.oy());
        return new Projection(t.id(), Angles.bearing(t.ox(), t.oy(), qx, qy), r, weight);
    }

    public record MeleeChoice(double angle, int primaryTarget, int projections, String fallback) {}

    /**
     * Combines displacement projections across all live targets: per-target normalized weights,
     * Gaussian angular density divided by projected range. Null when no valid projection exists.
     */
    public MeleeChoice aimMelee(List<Target> targets, double gunHeading) {
        if (targets.isEmpty()) return null;
        int per = Math.max(1, Profile.MELEE_GUN_SAMPLES / targets.size());
        List<Projection> all = new ArrayList<>();
        for (Target t : targets) {
            NeighborStore st = data(t.id()).displacement;
            List<Projection> ps = new ArrayList<>();
            for (NeighborStore.Neighbor n : st.query(t.context().meleeGun(), per)) {
                Projection p = project(t, n.sample().label(), n.sample().quality() * n.inverseDistance() * n.recencyWeight());
                if (p != null) ps.add(p);
            }
            double sw = 0;
            for (Projection p : ps) sw += p.weight();
            for (Projection p : ps) all.add(new Projection(p.targetId(), p.angle(), p.range(), p.weight() / sw));
        }
        if (all.isEmpty()) return null;
        double bestScore = -1, bestAngle = 0, bestRot = 0;
        int primary = -1;
        for (Projection c : all) {
            double score = 0;
            Map<Integer, Double> contrib = new LinkedHashMap<>();
            for (Projection p : all) {
                double bw = bandwidth(p.range());
                double e = Angles.wrap(c.angle() - p.angle()) / bw;
                double v = p.weight() * Math.exp(-0.5 * e * e) / Math.max(p.range(), 1);
                score += v;
                contrib.merge(p.targetId(), v, Double::sum);
            }
            double rot = Math.abs(Angles.wrap(c.angle() - gunHeading));
            if (score > bestScore + 1e-15 || (Math.abs(score - bestScore) <= 1e-15 && rot < bestRot)) {
                bestScore = score; bestAngle = c.angle(); bestRot = rot;
                primary = contrib.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
            }
        }
        return new MeleeChoice(bestAngle, primary, all.size(), null);
    }

    /** Footprint half-width used for fire tolerance. */
    public static double fireTolerance(double range) {
        double half = Geometry.halfWidth(range);
        return Math.max(Profile.MIN_FIRE_TOLERANCE, Math.min(Profile.MAX_FIRE_TOLERANCE, 0.5 * half));
    }
}
