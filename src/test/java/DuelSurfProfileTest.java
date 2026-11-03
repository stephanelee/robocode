import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class DuelSurfProfileTest {
    private static DuelSurf.Threat threat(double[] bins) {
        return new DuelSurf.Threat(500, 50, 16.4, 98, 90, 90, 90, bins);
    }

    @Test void insufficientOrInvalidHistoryKeepsTheAnalyticChoiceAndScoreExactly() {
        var original = new DuelSurf.Threat(500, 50, 16.4, 98, 90, 90);
        double[] sparse = new double[47]; sparse[7] = 2.99;
        double[] negative = new double[47]; negative[7] = 10; negative[8] = -1;
        double[] nonfinite = new double[47]; nonfinite[7] = 10; nonfinite[8] = Double.NaN;
        var command = DuelSurf.command(0, 0, 1);
        double score = DuelSurf.score(500, 500, 0, 8, command, 100, List.of(original), 1000, 1000);
        var chosen = DuelSurf.choose(500, 500, 0, 8, 100, 500, 50, List.of(original), 1, 1000, 1000);
        for (double[] bins : new double[][] {null, new double[47], sparse, negative, nonfinite}) {
            var candidate = threat(bins);
            assertTrue(Double.isNaN(DuelSurf.profileDanger(candidate, 90, 450)));
            assertEquals(score, DuelSurf.score(500, 500, 0, 8, command, 100, List.of(candidate), 1000, 1000));
            assertEquals(chosen, DuelSurf.choose(500, 500, 0, 8, 100, 500, 50, List.of(candidate), 1, 1000, 1000));
        }
    }

    @Test void integratesTheBotFootprintAndNormalizesAwayHistoryScale() {
        double[] bins = new double[47]; bins[23] = 3;
        var profile = threat(bins);
        assertEquals(1, DuelSurf.profileDanger(profile, 90, 450), 1e-12);
        assertEquals(0.5, DuelSurf.profileDanger(profile, 90, 10000), 1e-12);
        assertEquals(0, DuelSurf.profileDanger(profile, 110, 450), 1e-12);
        bins[23] = 300;
        assertEquals(DuelSurf.profileDanger(profile, 90, 450), DuelSurf.profileDanger(threat(bins), 90, 450));
        assertEquals(DuelSurf.profileDanger(profile, 90, 10000), DuelSurf.profileDanger(threat(bins), 90, 10000));
    }

    @Test void aLearnedNonlinearAimRegionChangesTheActualTravelChoice() {
        // The forward endpoint crosses GF bin2; this legal learned region is absent from the analytic prior.
        double[] bins = new double[47]; bins[2] = 12;
        var old = new DuelSurf.Threat(500, 50, 16.4, 98, 90, 90);
        assertEquals(1, DuelSurf.choose(500, 500, 0, 8, 100, 500, 50, List.of(old), 1, 1000, 1000).travelSign());
        var learned = DuelSurf.choose(500, 500, 0, 8, 100, 500, 50, List.of(threat(bins)), 1, 1000, 1000);
        assertEquals(-1, learned.travelSign(), "avoid the hit region that the analytic head-on/linear prior misses");
        assertTrue(DuelSurf.score(500, 500, 0, 8, learned, 100, List.of(threat(bins)), 1000, 1000) > -500);
    }

    @Test void fixedSupportWeightRetainsAtLeastThreeQuartersOfTheAnalyticPrior() {
        double[] bins = new double[47]; bins[23] = 2.99;
        assertEquals(0, DuelSurf.profileWeight(threat(bins)));
        bins[23] = 3;
        assertEquals(3.0 / 27, DuelSurf.profileWeight(threat(bins)), 1e-12);
        assertEquals((1 - 3.0 / 27) * 60, DuelSurf.blendedSafety(threat(bins), 90, 450, 100), 1e-12);
        bins[23] = 8;
        assertEquals(0.25, DuelSurf.profileWeight(threat(bins)), 1e-12);
        assertEquals(45, DuelSurf.blendedSafety(threat(bins), 90, 450, 60), 1e-12);
        assertEquals(15, DuelSurf.blendedSafety(threat(bins), 110, 450, 0), 1e-12);
        bins[23] = 800;
        assertEquals(0.25, DuelSurf.profileWeight(threat(bins)), 1e-12);
    }

    @Test void threatSnapshotsLegalHistoryInsteadOfObservingLaterChanges() {
        double[] bins = new double[47]; bins[23] = 3;
        var snapshot = threat(bins);
        bins[23] = 0; bins[0] = 100;
        assertEquals(1, DuelSurf.profileDanger(snapshot, 90, 450), 1e-12);
        assertEquals(0, DuelSurf.profileDanger(threat(bins), 90, 450), 1e-12);
    }

    @Test void learnedDangerRespectsWaveAndWallSafetyAndRemainsBoundedAtEdges() {
        double[] bins = new double[47]; bins[0] = 3;
        var edge = threat(bins);
        assertTrue(DuelSurf.profileDanger(edge, 30, 450) >= 0);
        assertTrue(DuelSurf.profileDanger(edge, 30, 450) <= 1);
        var passed = new DuelSurf.Threat(500, 50, 16.4, 10, 90, 90, 90, bins);
        assertEquals(DuelSurf.CLEARANCE_CAP,
                DuelSurf.score(500, 500, 0, 8, DuelSurf.command(0, 0, 1), 100, List.of(passed), 1000, 1000));
        assertEquals(-999, DuelSurf.score(780, 300, 0, 8, new DuelSurf.Command(87, -8, -1), 100,
                List.of(edge), 800, 600));
    }
}
