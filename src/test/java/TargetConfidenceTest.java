import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class TargetConfidenceTest {
    @Test void emptyFrozenLocalStatsKeepThePersistedClassPeakWithoutMutation() {
        double[] local = new double[47];
        double[] savedClass = new double[47];
        savedClass[35] = 12;
        double[] localBefore = local.clone(), classBefore = savedClass.clone();
        for (int scan = 1; scan <= 100; scan++) {
            assertEquals(0, CombatMath.enemyWeight(local));
            assertEquals(35, CombatMath.bestBin(local, savedClass, CombatMath.enemyWeight(local)));
        }
        assertArrayEquals(localBefore, local);
        assertArrayEquals(classBefore, savedClass);
    }

    @Test void confidenceCountsActualSamplesOnlyInTheSelectedSegment() {
        double[] segment = new double[47];
        assertEquals(0, CombatMath.enemyWeight(null));
        segment[10] = 2; segment[30] = 1;
        assertEquals(0.1, CombatMath.enemyWeight(segment), 1e-12);
        segment[10] = 29;
        assertEquals(1, CombatMath.enemyWeight(segment));
        segment[10] = 100;
        assertEquals(1, CombatMath.enemyWeight(segment));
        assertEquals(0, CombatMath.enemyWeight(new double[47]));
    }

    @Test void realLocalEvidenceCanTakeOverWhileEmptyClassAndLocalRemainCentered() {
        double[] local = new double[47], savedClass = new double[47];
        assertEquals(23, CombatMath.bestBin(local, savedClass, CombatMath.enemyWeight(local)));
        savedClass[35] = 50;
        local[10] = 1;
        assertEquals(35, CombatMath.bestBin(local, savedClass, CombatMath.enemyWeight(local)));
        local[10] = 30;
        assertEquals(10, CombatMath.bestBin(local, savedClass, CombatMath.enemyWeight(local)));
    }
}
