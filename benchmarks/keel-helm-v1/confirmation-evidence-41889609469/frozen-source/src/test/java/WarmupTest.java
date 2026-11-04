import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class WarmupTest {
    private static BotConfig config() {
        return new BotConfig(BotConfig.Mode.TRAIN, 7, Path.of("unused.dat"), 0, null);
    }

    @Test void warmupIsDeterministicAndFinite() {
        double first = Warmup.run(50);
        assertTrue(Double.isFinite(first));
        assertEquals(first, Warmup.run(50), 0.0);
    }

    @Test void warmupDoesNotChangeAgentDecisions() {
        double[] features = new double[14];
        Arrays.fill(features, 0.3);
        DQNAgent before = new DQNAgent(14, 6, config());
        before.startEpisode();
        int[] expected = new int[40];
        for (int i = 0; i < expected.length; i++) expected[i] = before.observe(features, 0.1, false, false);

        Warmup.run(200);
        DQNAgent after = new DQNAgent(14, 6, config());
        after.startEpisode();
        for (int i = 0; i < expected.length; i++)
            assertEquals(expected[i], after.observe(features, 0.1, false, false), "decision " + i);
    }

    @Test void turnLatencyCountersTrackMaximumAndSlowTurnsAndReset() {
        RoundMetrics metrics = new RoundMetrics();
        metrics.recordTurnLatency(1_500);
        metrics.recordTurnLatency(RoundMetrics.SLOW_TURN_MICROS);
        metrics.recordTurnLatency(19_999);
        assertEquals(RoundMetrics.SLOW_TURN_MICROS, metrics.maxTurnMicros);
        assertEquals(1, metrics.slowTurns);
        String json = metrics.json(config(), 1, 10, 1, false, true, false);
        assertTrue(json.contains("\"maxTurnMicros\":20000"), json);
        assertTrue(json.contains("\"slowTurns\":1"), json);
        metrics.reset();
        assertEquals(0, metrics.maxTurnMicros);
        assertEquals(0, metrics.slowTurns);
    }
}
