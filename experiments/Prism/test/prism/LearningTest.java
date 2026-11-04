package prism;

import org.junit.jupiter.api.Test;
import prism.core.Rules;
import prism.gun.Guns;
import prism.learn.Features;
import prism.learn.NeighborStore;
import prism.move.Danger;
import prism.wave.Wave;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/** Neighbour mathematics, GF conventions, selector rules and danger views. */
class LearningTest {

    static Features.Context ctx(double v) {
        return new Features.Context(v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, 1, 1, false);
    }

    @Test
    void exactQueriesMatchBruteForceWithEvictionDuplicatesAndWeights() {
        Random r = new Random(3);
        double[] w = {1, 2, 0.5};
        NeighborStore s = new NeighborStore("t", NeighborStore.LabelType.PRECISE_GF, w, 50);
        assertTrue(s.query(new double[] {0, 0, 0}, 5).isEmpty());          // empty store
        for (int i = 0; i < 120; i++) {
            double[] x = {r.nextInt(4), r.nextInt(4), r.nextInt(4)};        // many exact duplicates
            assertTrue(s.add(x, new double[] {i}, 1, 1, i, i));
        }
        assertEquals(50, s.size());
        List<NeighborStore.Sample> kept = s.samples();
        assertEquals(70, kept.get(0).label()[0]);                          // FIFO eviction by sequence
        for (int q = 0; q < 50; q++) {
            double[] qq = {r.nextInt(4), r.nextInt(4), r.nextInt(4)};
            List<NeighborStore.Neighbor> got = s.query(qq, 7);
            List<NeighborStore.Sample> brute = new ArrayList<>(kept);
            brute.sort(Comparator.comparingDouble((NeighborStore.Sample x) -> {
                double d = 0; for (int i = 0; i < 3; i++) { double t = w[i] * (x.x()[i] - qq[i]); d += t * t; } return d;
            }).thenComparingLong(NeighborStore.Sample::sequence));
            for (int i = 0; i < 7; i++) {
                assertEquals(brute.get(i).sequence(), got.get(i).sample().sequence());
                assertTrue(Double.isFinite(got.get(i).inverseDistance()));
            }
        }
        // a zero-distance duplicate is finite and floored at 1e-6
        NeighborStore.Neighbor z = s.query(kept.get(0).x(), 1).get(0);
        assertEquals(0, z.distance());
        assertEquals(1000, z.inverseDistance(), 1e-9);
        // weight scale is squared in the distance
        assertEquals(4, s.distance(new double[] {0, 1, 0}, new double[] {0, 0, 0}), 1e-12);
        // non-finite input rejected without changing the version
        long v = s.version();
        assertFalse(s.add(new double[] {Double.NaN, 0, 0}, new double[] {0}, 1, 1, 1, 1));
        assertEquals(v, s.version());
        assertTrue(s.query(new double[] {Double.POSITIVE_INFINITY, 0, 0}, 3).isEmpty());
    }

    @Test
    void recencyRankIsAmongSelectedNeighbours() {
        NeighborStore s = new NeighborStore("t", NeighborStore.LabelType.PRECISE_GF, new double[] {1}, 10);
        for (int i = 0; i < 5; i++) s.add(new double[] {i}, new double[] {i}, 1, 1, i, i);
        List<NeighborStore.Neighbor> n = s.query(new double[] {0}, 3);
        assertEquals(2, n.get(0).recencyRank());   // oldest selected
        assertEquals(0, n.get(2).recencyRank());   // newest selected
        assertEquals(1, n.get(2).recencyWeight(), 1e-12);
        assertEquals(Math.pow(2, -2 / 32.0), n.get(0).recencyWeight(), 1e-12);
    }

    static Wave wave(double ref, int sign, double pos, double neg) {
        return new Wave(1, Wave.Role.VIRTUAL, -1, 2, 10, 10, 0, 0, 2, 0, ref, sign, Wave.Escape.PRECISE, pos, neg, ctx(0.5), 1, 9);
    }

    @Test
    void preciseGfSignAndAsymmetricScalesRoundTrip() {
        Wave w = wave(1.0, -1, 0.3, 0.6);
        // clockwise orbit: a bearing smaller than the reference is a positive offset
        assertEquals(1.0, w.preciseGf(1.0 - 0.3), 1e-12);
        assertEquals(-1.0, w.preciseGf(1.0 + 0.6), 1e-12);
        assertEquals(-0.5, w.preciseGf(1.0 + 0.3), 1e-12);
        for (double g = -1.4; g <= 1.4; g += 0.1) assertEquals(g, w.preciseGf(w.decodePrecise(g)), 1e-9); // outliers retained
        // theoretical scale is the symmetric asin(8/b) and differs from the precise one
        double m = Rules.maxEscape(Rules.bulletSpeed(2));
        assertEquals(1.0, w.theoreticalGf(1.0 - m), 1e-12);
        assertNotEquals(w.preciseGf(1.0 - m), w.theoreticalGf(1.0 - m), 1e-3);
        // zero lateral motion before any evidence uses +1
        Wave z = wave(0, 0, 0.3, 0.3);
        assertEquals(1, z.orbitSign);
    }

    @Test
    void storesDeclareLabelTypesThatNeverMix() {
        Guns.Data d = new Guns(Rules.standard()).data(5);
        assertEquals(NeighborStore.LabelType.PRECISE_GF, d.longStore.labelType());
        assertEquals(NeighborStore.LabelType.DISPLACEMENT, d.displacement.labelType());
        Danger dg = new Danger();
        assertEquals(NeighborStore.LabelType.THEORETICAL_GF, dg.simple.labelType());
        assertEquals(NeighborStore.LabelType.THEORETICAL_GF, dg.visits.labelType());
        // visits never become hits
        Wave w = new Wave(3, Wave.Role.INCOMING, 2, -1, 10, 10, 0, 0, 2, 0, 0, 1, Wave.Escape.THEORETICAL, 0, 0, ctx(0.5), 1, 9);
        dg.addVisit(w, 0.3, 1, 20);
        assertEquals(0, dg.hitLabels);
        assertEquals(0, dg.simple.size());
        assertEquals(1, dg.visits.size());
    }

    @Test
    void selectorNeedsEightResolvedAndAMargin() {
        Guns guns = new Guns(Rules.standard());
        Guns.Data d = guns.data(2);
        for (int i = 0; i < 7; i++) rate(guns, d, 0.1, 0.9);
        assertEquals(Guns.ANALYTIC, d.current);                 // too few resolved
        rate(guns, d, 0.1, 0.9);
        assertEquals(Guns.LONG, d.current);                     // both have eight, margin exceeded
        Guns g2 = new Guns(Rules.standard());
        Guns.Data e = g2.data(2);
        for (int i = 0; i < 20; i++) rate(g2, e, 0.5, 0.515);   // within 0.02: keep current
        assertEquals(Guns.ANALYTIC, e.current);
    }

    private static void rate(Guns guns, Guns.Data d, double analyticScore, double longScore) {
        // construct a resolved wave where predictions sit at chosen errors from the label
        Wave w = wave(0, 1, 0.5, 0.5);
        w.fireOpportunity = true;
        w.labelBearing = 0;
        w.labelHalfWidth = 0.02;
        w.predictions.put(Guns.ANALYTIC, errFor(analyticScore, 0.02));
        w.predictions.put(Guns.LONG, errFor(longScore, 0.02));
        guns.rate(w);
    }

    private static double errFor(double score, double h) { return h * Math.sqrt(-2 * Math.log(score)); }

    @Test
    void trainingIsOncePerWaveAndCancelledWavesNeverTrain() {
        Guns guns = new Guns(Rules.standard());
        Wave w = wave(0, 1, 0.5, 0.5);
        w.labelBearing = 0.25;
        assertTrue(guns.train(w, 1, 30));
        assertFalse(guns.train(w, 1, 31));
        assertEquals(1, guns.data(2).longStore.size());
        Wave c = wave(0, 1, 0.5, 0.5);
        c.labelBearing = 0.1;
        c.cancelled = true;
        assertFalse(guns.train(c, 1, 30));
        // a cancelled wave is excluded from every gun's rating denominator
        c.fireOpportunity = true;
        c.predictions.put(Guns.ANALYTIC, 0.1);
        guns.rate(c);
        assertTrue(guns.data(2).ratings.isEmpty());
    }

    @Test
    void surferGunTrainsOnlyOnConfirmedBulletWaves() {
        Guns guns = new Guns(Rules.standard());
        Wave virtual = wave(0, 1, 0.5, 0.5);
        virtual.labelBearing = 0.2;
        assertTrue(guns.train(virtual, 1, 30));
        Guns.Data d = guns.data(2);
        assertEquals(1, d.longStore.size());
        for (NeighborStore s : d.surfer) assertEquals(0, s.size(), "virtual crossings never enter the anti-surfer views");
        assertEquals(0, d.physicalLabels);
        Wave shot = wave(0, 1, 0.5, 0.5);
        shot.role = Wave.Role.OUTGOING;
        shot.physical = true;
        shot.labelBearing = -0.1;
        assertTrue(guns.train(shot, 1, 31));
        for (NeighborStore s : d.surfer) assertEquals(1, s.size());
        assertEquals(1, d.physicalLabels);
        assertEquals(NeighborStore.LabelType.PRECISE_GF, d.surfer[0].labelType());
        assertFalse(guns.train(shot, 1, 32));                     // exactly once
        assertEquals(1, d.physicalLabels);
    }

    @Test
    void surferGunBecomesEligibleAfterEightConfirmedLabelsAndCanBeSelected() {
        Guns guns = new Guns(Rules.standard());
        Guns.Data d = guns.data(2);
        for (int i = 0; i < 7; i++) {
            Wave shot = wave(0, 1, 0.5, 0.5);
            shot.role = Wave.Role.OUTGOING; shot.physical = true; shot.labelBearing = 0.3;
            guns.train(shot, 1, 40 + i);
        }
        assertFalse(guns.eligible(d).contains(Guns.SURFER));
        Wave eighth = wave(0, 1, 0.5, 0.5);
        eighth.role = Wave.Role.OUTGOING; eighth.physical = true; eighth.labelBearing = 0.3;
        guns.train(eighth, 1, 47);
        assertTrue(guns.eligible(d).contains(Guns.SURFER));
        // its prediction follows the confirmed-bullet labels
        Guns.Aim a = new Guns.Aim(0, 0, 11, new prism.core.Physics.Motion(400, 0, Math.PI / 2, 8), 10, 0, 2, ctx(0.5), 1, 0.5, 0.5, 0, false);
        Double p = guns.learned(Guns.SURFER, a, d);
        assertNotNull(p);
        assertEquals(0.3, prism.core.Angles.wrap(p - a.reference()), 0.02);
        // the unchanged selector rule picks it when it rates better on the shared cohort
        for (int i = 0; i < 8; i++) {
            Wave w = wave(0, 1, 0.5, 0.5);
            w.fireOpportunity = true; w.labelBearing = 0; w.labelHalfWidth = 0.02;
            w.predictions.put(Guns.ANALYTIC, errFor(0.2, 0.02));
            w.predictions.put(Guns.SURFER, errFor(0.8, 0.02));
            guns.rate(w);
        }
        assertEquals(Guns.SURFER, d.current);
    }

    @Test
    void dangerQuadratureWilsonAndActivation() {
        Danger d = new Danger();
        Wave w = new Wave(3, Wave.Role.INCOMING, 2, -1, 10, 10, 0, 0, 2, 0, 0, 1, Wave.Escape.THEORETICAL, 0, 0, ctx(0.5), 1, 9);
        Danger.Density prior = d.density(w, 400, 0.4);
        double total = Danger.integrate(prior, -1, 1, w.maxEscape);
        assertTrue(total > 0.85 && total < 1.05, "prior mass " + total);
        assertTrue(prior.at(0) > prior.at(-0.6));                       // head-on emphasis
        assertTrue(prior.at(0.4) > prior.at(-0.4));                     // linear-aim component
        // after a hit label the learned view gains weight near that GF
        for (int i = 0; i < 10; i++) d.addHit(w, -0.7, false, 1, i);
        Danger.Density learned = d.density(w, 400, 0.4);
        assertTrue(learned.at(-0.7) > prior.at(-0.7));
        assertEquals(0, Danger.wilsonLower(0, 0));
        assertTrue(Danger.wilsonLower(10, 50) > 0.08);
        d.resolvedRealWaves = 49; d.rawHits = 10;
        assertFalse(d.flattenerEnabled());
        d.resolvedRealWaves = 50;
        assertTrue(d.flattenerEnabled());
        d.rawHits = 3;
        assertFalse(d.flattenerEnabled());
    }
}
