import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Production-planner contracts with a frozen original-production contrast. No engine or games. */
class KeelWakeRefreshTest {
    private static final Phys.Pose ME = new Phys.Pose(650, 300, 0, 4);
    private static final double[] F = new double[Track.DIMS];

    @BeforeEach void noShadows() { Surfer.shadows = false; FrozenKeelSurfer.shadows = false; }

    private static Surfer.Wave pending(Surfer s, double x) {
        var w = s.newWave(0, x, 300, 1.5, 0, 1, F.clone());
        s.waves.add(w); return w;
    }
    private static FrozenKeelSurfer.Wave pending(FrozenKeelSurfer s, double x) {
        var f = F.clone();
        var w = new FrozenKeelSurfer.Wave(0, x, 300, 1.5, 0, 1, f, s.density(f));
        s.waves.add(w); return w;
    }
    private static void label(Surfer s, boolean collision, double gf) {
        s.waves.add(s.newWave(-10, 500, 300, 2, -gf * Phys.mea(2), 1, F.clone()));
        if (collision) s.bulletCollision(0, 654, 300, 0, 2);
        else s.hitBy(0, 654, 300, 0, 2);
    }
    private static void label(FrozenKeelSurfer s, boolean collision, double gf) {
        var f = F.clone();
        s.waves.add(new FrozenKeelSurfer.Wave(-10, 500, 300, 2, -gf * Phys.mea(2), 1, f, s.density(f)));
        if (collision) s.bulletCollision(0, 654, 300, 0, 2);
        else s.hitBy(0, 654, 300, 0, 2);
    }
    private static Surfer.Plan plan(Surfer s) { return s.plan(ME, 0, 500, 300, 800, 600, 100); }
    private static FrozenKeelSurfer.Plan plan(FrozenKeelSurfer s) { return s.plan(ME, 0, 500, 300, 800, 600, 100); }
    private static void normalized(double[] a) {
        assertEquals(Surfer.BINS, a.length);
        for (double x : a) assertTrue(Double.isFinite(x) && x > 0);
        assertEquals(1, Arrays.stream(a).sum(), 1e-12);
    }
    private static void validCounters(Surfer s) {
        assertTrue(s.refreshEligiblePlans <= s.refreshedWaves);
        assertTrue(s.refreshedWaves <= 2 * s.refreshEligiblePlans);
        assertTrue(s.densityChangedWaves <= s.refreshedWaves);
        for (long n : new long[]{s.matchedDangerLabels, s.refreshEligiblePlans, s.refreshedWaves, s.densityChangedWaves})
            assertTrue(n >= 0 && n <= Surfer.COUNTER_CAP);
    }

    @Test void noLabelMatchesFrozenOriginalProductionPlanExactly() {
        var s = new Surfer(); var old = new FrozenKeelSurfer();
        var wave = pending(s, 300); pending(old, 300);
        double[] born = wave.density;
        var p = plan(s); var q = plan(old);
        assertNotNull(p); assertNotNull(q);
        assertEquals(q.dir(), p.dir()); assertEquals(q.stopAt(), p.stopAt());
        assertEquals(q.danger(), p.danger(), 0); assertEquals(q.turn(), p.turn(), 0); assertEquals(q.speed(), p.speed(), 0);
        assertSame(born, wave.density); assertEquals(0, s.refreshedWaves); assertEquals(0, s.refreshEligiblePlans);
    }

    @Test void newMatchedHitRefreshesOnlyTwoSelectedWavesAndOriginalRemainsFrozen() {
        var s = new Surfer(); var old = new FrozenKeelSurfer();
        var lazy = pending(s, 100); var second = pending(s, 200); var first = pending(s, 300);
        pending(old, 100); pending(old, 200); var oldFirst = pending(old, 300);
        double[] lazyBorn = lazy.density, firstBorn = first.density.clone(), originalBorn = oldFirst.density.clone();
        label(s, false, 0.6); label(old, false, 0.6);
        assertEquals(1, s.matchedDangerLabels); assertEquals(old.hits.seq(), s.hits.seq());
        assertNotNull(plan(s)); assertNotNull(plan(old));
        assertArrayEquals(originalBorn, oldFirst.density, 0);
        assertFalse(Arrays.equals(firstBorn, first.density));
        assertSame(lazyBorn, lazy.density); assertEquals(0, lazy.densityEpoch);
        assertArrayEquals(s.density(first.features), first.density, 0);
        assertArrayEquals(s.density(second.features), second.density, 0);
        assertEquals(1, s.refreshEligiblePlans); assertEquals(2, s.refreshedWaves); assertEquals(2, s.densityChangedWaves);
        normalized(first.density); normalized(second.density); validCounters(s);
    }

    @Test void collisionKeepsHalfWeightAndRemovesOnlyMatchedWave() {
        var s = new Surfer(); var w = pending(s, 300);
        label(s, true, -0.6);
        assertEquals(1, s.hits.seq()); assertEquals(0.5, s.hits.recent(1).get(0).weight(), 0);
        assertEquals(1, s.waves.size()); assertSame(w, s.waves.get(0));
        plan(s); assertEquals(1, s.matchedDangerLabels); assertEquals(1, s.refreshedWaves); normalized(w.density);
    }

    @Test void hitKeepsUnitWeight() {
        var s = new Surfer(); pending(s, 300); label(s, false, 0.6); plan(s);
        assertEquals(1, s.hits.recent(1).get(0).weight(), 0); validCounters(s);
    }

    @Test void unmatchedCallbacksDoNotInvalidateCache() {
        var s = new Surfer(); var w = pending(s, 300); double[] born = w.density;
        s.hitBy(0, 40, 40, 0, 2); s.bulletCollision(0, 40, 40, 0, 3); plan(s);
        assertEquals(1, s.enemyHits); assertEquals(0, s.hits.seq()); assertSame(born, w.density);
        assertEquals(0, s.matchedDangerLabels); assertEquals(0, s.refreshedWaves); validCounters(s);
    }

    @Test void visitOnlyChangesDoNotRefreshEvenWhenEstimatorWouldDiffer() {
        var s = new Surfer(); var w = pending(s, 300); double[] born = w.density;
        for (int i = 0; i < 25; i++) s.visits.add(F.clone(), 0.8, 1);
        assertFalse(Arrays.equals(born, s.density(w.features))); plan(s);
        assertSame(born, w.density); assertEquals(0, s.refreshEligiblePlans); assertEquals(0, s.refreshedWaves);
    }

    @Test void newBirthUsesCurrentEpochAndDoesNotFakeExposure() {
        var s = new Surfer(); label(s, false, 0.6); var w = pending(s, 300); double[] born = w.density;
        assertEquals(s.hits.seq(), w.densityEpoch); plan(s);
        assertSame(born, w.density); assertEquals(0, s.refreshEligiblePlans); assertEquals(0, s.densityChangedWaves);
    }

    @Test void multipleLabelsCoalesceAndRepeatedPlansReuseOneCache() {
        var s = new Surfer(); var w = pending(s, 300);
        label(s, false, 0.6); label(s, true, -0.3); plan(s); double[] refreshed = w.density;
        assertEquals(2, s.matchedDangerLabels); assertEquals(2, w.densityEpoch);
        for (int i = 0; i < 3; i++) plan(s);
        assertSame(refreshed, w.density); assertEquals(1, s.refreshedWaves); assertEquals(1, s.refreshEligiblePlans); validCounters(s);
    }

    @Test void staleThirdWaveRefreshesWhenLaterSelectedAtSameModelEpoch() {
        var s = new Surfer(); var third = pending(s, 100); pending(s, 200); pending(s, 300);
        label(s, false, 0.6); plan(s); assertEquals(2, s.refreshedWaves); assertEquals(0, third.densityEpoch);
        s.waves.removeIf(w -> w != third); plan(s);
        assertEquals(s.hits.seq(), third.densityEpoch); assertEquals(3, s.refreshedWaves); assertEquals(2, s.refreshEligiblePlans); validCounters(s);
    }

    @Test void nextMatchedLabelInvalidatesSelectedCacheAgain() {
        var s = new Surfer(); var w = pending(s, 300); label(s, false, 0.6); plan(s); double[] first = w.density;
        label(s, true, -0.6); plan(s);
        assertNotSame(first, w.density); assertEquals(2, s.refreshedWaves); assertEquals(2, s.refreshEligiblePlans); validCounters(s);
    }

    @Test void capacityEvictionChangesSequenceEvenWhenSizeStaysConstant() {
        var s = new Surfer();
        for (int i = 0; i < 600; i++) s.hits.add(F.clone(), 0, 1);
        s.newRound(); var w = pending(s, 300); long atBirth = w.densityEpoch;
        label(s, false, 0.6); assertEquals(600, s.hits.size()); plan(s);
        assertEquals(atBirth + 1, w.densityEpoch); assertEquals(1, s.refreshedWaves); validCounters(s);
    }

    @Test void roundClearsWavesAndCountersButPreservesBothModels() {
        var s = new Surfer(); pending(s, 300); label(s, false, 0.6); plan(s);
        s.visits.add(F.clone(), 0.2, 1); long epoch = s.hits.seq();
        s.bullets.add(new Surfer.Own(1, 0, 600, 300, 0, 17)); s.newRound();
        assertTrue(s.waves.isEmpty()); assertTrue(s.bullets.isEmpty()); assertEquals(epoch, s.hits.seq()); assertEquals(1, s.visits.size());
        assertEquals(0, s.matchedDangerLabels); assertEquals(0, s.refreshEligiblePlans); assertEquals(0, s.refreshedWaves); assertEquals(0, s.densityChangedWaves);
        var w = pending(s, 300); plan(s); assertEquals(epoch, w.densityEpoch); assertEquals(0, s.refreshedWaves);
    }

    @Test void labelWithNoSurvivingWaveDoesNotCreateExposure() {
        var s = new Surfer(); label(s, false, 0.6);
        assertNull(plan(s)); assertEquals(1, s.matchedDangerLabels); assertEquals(0, s.refreshEligiblePlans); assertEquals(0, s.refreshedWaves);
    }

    @Test void refreshPreservesFireFeaturesPhysicsAndShades() {
        var s = new Surfer(); var w = pending(s, 300); double[] f = w.features.clone(), shade = w.shade.clone();
        double near = w.near(5), far = w.far(5), speed = w.speed, direct = w.direct, power = w.power, mea = w.mea;
        label(s, false, 0.6); plan(s);
        assertArrayEquals(f, w.features, 0); assertArrayEquals(shade, w.shade, 0);
        assertEquals(0, w.t); assertEquals(300, w.ox); assertEquals(300, w.oy); assertEquals(1, w.latDir);
        assertEquals(near, w.near(5), 0); assertEquals(far, w.far(5), 0); assertEquals(speed, w.speed, 0);
        assertEquals(direct, w.direct, 0); assertEquals(power, w.power, 0); assertEquals(mea, w.mea, 0);
    }

    @Test void passedWaveIsNotRefreshed() {
        var s = new Surfer(); var passed = s.newWave(-100, 100, 300, 1.5, 0, 1, F.clone()); s.waves.add(passed);
        var live = pending(s, 300); label(s, false, 0.6); plan(s);
        assertEquals(0, passed.densityEpoch); assertEquals(s.hits.seq(), live.densityEpoch); assertEquals(1, s.refreshedWaves);
    }

    @Test void equalEtaSelectionRetainsOriginalInsertionTieRule() {
        var s = new Surfer(); var a = pending(s, 300); var b = pending(s, 300); var c = pending(s, 300);
        label(s, false, 0.6); plan(s);
        assertEquals(s.hits.seq(), a.densityEpoch); assertEquals(s.hits.seq(), b.densityEpoch); assertEquals(0, c.densityEpoch);
        assertEquals(2, s.refreshedWaves);
    }

    @Test void labelRefreshUsesWholeUnchangedEstimatorIncludingNewerVisits() {
        var s = new Surfer(); var w = pending(s, 300);
        for (int i = 0; i < 25; i++) s.visits.add(F.clone(), -0.8, 1);
        label(s, false, 0.6); plan(s);
        assertArrayEquals(s.density(w.features), w.density, 0); normalized(w.density);
    }

    @Test void everyCounterOverflowEmitsExactlyOneFaultAndNeverWraps() {
        for (int counter = 0; counter < 4; counter++) {
            var s = new Surfer(); pending(s, 300); var reports = new AtomicInteger(); s.overflowReporter = reports::incrementAndGet;
            if (counter == 0) {
                s.matchedDangerLabels = Surfer.COUNTER_CAP;
                assertThrows(IllegalStateException.class, () -> label(s, false, 0.6));
                assertThrows(IllegalStateException.class, () -> label(s, false, 0.6));
                assertEquals(Surfer.COUNTER_CAP, s.matchedDangerLabels);
            } else {
                label(s, false, 0.6);
                if (counter == 1) s.refreshEligiblePlans = Surfer.COUNTER_CAP;
                if (counter == 2) s.refreshedWaves = Surfer.COUNTER_CAP;
                if (counter == 3) s.densityChangedWaves = Surfer.COUNTER_CAP;
                assertThrows(IllegalStateException.class, () -> plan(s));
                assertThrows(IllegalStateException.class, () -> plan(s));
            }
            assertEquals(1, reports.get());
        }
    }

    @Test void exactCounterLimitIsAllowedBeforeFirstAttemptedExceed() {
        var s = new Surfer(); var reports = new AtomicInteger(); s.overflowReporter = reports::incrementAndGet;
        s.matchedDangerLabels = Surfer.COUNTER_CAP - 1; label(s, false, 0.6);
        assertEquals(Surfer.COUNTER_CAP, s.matchedDangerLabels); assertEquals(0, reports.get());
        assertThrows(IllegalStateException.class, () -> label(s, false, 0.6)); assertEquals(1, reports.get());
    }
}
