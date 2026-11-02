import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.*;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

class LearningTest {
    @TempDir Path temp;

    BotConfig config(String mode, long seed) {
        return BotConfig.fromEnvironment(Map.of("AISURVIVOR_MODE", mode, "AISURVIVOR_SEED", "" + seed,
                "AISURVIVOR_MODEL_PATH", temp.resolve("model.dat").toString(), "AISURVIVOR_ACTION", "3"));
    }

    @Test void sameSeedReproducesInitializationExplorationAndReplayTraining() {
        DQNAgent first = new DQNAgent(14, 6, config("train", 123));
        DQNAgent second = new DQNAgent(14, 6, config("train", 123));
        first.startEpisode(); second.startEpisode();
        double[] features = new double[14];
        for (int i = 0; i < 140; i++) {
            features[i % 14] = i / 140.0;
            assertEquals(first.observe(features, i % 7 - 3, false), second.observe(features, i % 7 - 3, false));
        }
        first.finishEpisode(-72); second.finishEpisode(-72);
        assertArrayEquals(first.policy.forward(features), second.policy.forward(features));
        DQNAgent other = new DQNAgent(14, 6, config("train", 124));
        assertFalse(Arrays.equals(first.policy.forward(features), other.policy.forward(features)));
    }

    @Test void replayCreditsTheSelectedActionAndStoresTerminalRewardOnce() {
        DQNAgent agent = new DQNAgent(14, 6, config("train", 123));
        agent.startEpisode();
        int selected = agent.observe(new double[14], 0, false);
        agent.observe(new double[14], 8, false);
        assertEquals(selected, (int) agent.replay.get(0)[14]);
        agent.finishEpisode(4 + 60); // pending damage/kill rewards plus win
        agent.finishEpisode(-999); // duplicate round/game end
        assertEquals(2, agent.replay.size());
        assertEquals(64, agent.replay.get(1)[15]);
        assertEquals(1, agent.replay.get(1)[30]);
        assertEquals(1, agent.totalRounds);
        agent.startEpisode();
        agent.observe(new double[14], 0, false);
        agent.finishEpisode(5 - 84); // pending reward plus death penalty
        assertEquals(-79, agent.replay.get(2)[15]);
        assertEquals(2, agent.totalRounds);
    }

    @Test void evaluationAndFixedModesDoNotChangeModelState() throws Exception {
        for (String mode : List.of("eval", "fixed")) {
            DQNAgent agent = new DQNAgent(14, 6, config(mode, 42));
            Path before = temp.resolve(mode + "-before");
            Path after = temp.resolve(mode + "-after");
            agent.policy.b2[4] = 100;
            ModelStore.save(before, agent);
            agent.startEpisode();
            for (int i = 0; i < 150; i++) assertEquals(mode.equals("eval") ? 4 : 3,
                    agent.observe(new double[14], 100, false));
            agent.finishEpisode(60);
            ModelStore.save(after, agent);
            assertArrayEquals(Files.readAllBytes(before), Files.readAllBytes(after));
            assertTrue(agent.replay.isEmpty());
        }
    }

    @Test void checkpointRoundTripPreservesNetworksStatsAndCounters() throws Exception {
        BotConfig config = config("train", 1);
        DQNAgent saved = new DQNAgent(14, 6, config);
        saved.classGF[2][3][4] = 12.5;
        saved.totalRounds = 3; saved.totalSteps = 100; saved.tau = 0.5;
        saved.target.b2[5] = 4;
        ModelStore.save(config.modelPath(), saved);
        DQNAgent loaded = new DQNAgent(14, 6, config("train", 2));
        assertTrue(ModelStore.load(config.modelPath(), loaded));
        assertArrayEquals(saved.policy.forward(new double[14]), loaded.policy.forward(new double[14]));
        assertArrayEquals(saved.target.forward(new double[14]), loaded.target.forward(new double[14]));
        assertEquals(12.5, loaded.classGF[2][3][4]);
        assertEquals(3, loaded.totalRounds); assertEquals(100, loaded.totalSteps); assertEquals(0.5, loaded.tau);
        assertFalse(ModelStore.load(temp.resolve("missing"), loaded));
    }

    @Test void rejectedCheckpointLeavesAgentUntouchedAndIsBackedUpBeforeReplacement() throws Exception {
        BotConfig config = config("train", 1);
        DQNAgent agent = new DQNAgent(14, 6, config);
        byte[] rejected = {(byte) 0xac, (byte) 0xed, 0, 5}; // legacy Java serialization header
        Files.write(config.modelPath(), rejected);
        double[] before = agent.policy.forward(new double[14]);
        assertThrows(java.io.IOException.class, () -> ModelStore.load(config.modelPath(), agent));
        assertArrayEquals(before, agent.policy.forward(new double[14]));
        ModelStore.save(config.modelPath(), agent);
        try (var paths = Files.list(temp)) {
            Path backup = paths.filter(p -> p.getFileName().toString().contains(".rejected-")).findFirst().orElseThrow();
            assertArrayEquals(rejected, Files.readAllBytes(backup));
        }
        assertTrue(ModelStore.load(config.modelPath(), agent));
    }

    @Test void checkpointRejectsWrongVersionDimensionsNonFiniteAndTruncation() throws Exception {
        BotConfig config = config("train", 1);
        DQNAgent agent = new DQNAgent(14, 6, config);
        ModelStore.save(config.modelPath(), agent);
        byte[] good = Files.readAllBytes(config.modelPath());
        for (int offset : new int[]{4, 8, 28}) {
            byte[] bad = good.clone();
            java.nio.ByteBuffer buffer = java.nio.ByteBuffer.wrap(bad);
            if (offset == 28) buffer.putDouble(offset, Double.NaN);
            else buffer.putInt(offset, 999);
            Files.write(config.modelPath(), bad);
            assertThrows(java.io.IOException.class, () -> ModelStore.load(config.modelPath(), agent));
        }
        Files.write(config.modelPath(), Arrays.copyOf(good, good.length - 1));
        assertThrows(java.io.IOException.class, () -> ModelStore.load(config.modelPath(), agent));
        Files.write(config.modelPath(), good);
        agent.policy.W1[0][0] = Double.NaN;
        assertThrows(java.io.IOException.class, () -> ModelStore.save(config.modelPath(), agent));
        assertArrayEquals(good, Files.readAllBytes(config.modelPath()));
    }

    @Test void configurationFailsFastAndTelemetryUsesActualShotCounts() {
        assertThrows(IllegalArgumentException.class, () -> BotConfig.fromEnvironment(Map.of("AISURVIVOR_MODE", "typo")));
        assertThrows(IllegalArgumentException.class, () -> BotConfig.fromEnvironment(Map.of("AISURVIVOR_SEED", "typo")));
        assertThrows(IllegalArgumentException.class, () -> BotConfig.fromEnvironment(Map.of("AISURVIVOR_ACTION", "6")));
        RoundMetrics metrics = new RoundMetrics();
        metrics.shotsFired = 4; metrics.bulletHits = 1;
        assertTrue(metrics.json(config("eval", 1), 1, 50, 3, true, true, false).contains("\"hitRate\":0.250000"));
        metrics.reset();
        assertTrue(metrics.json(config("fixed", 1), 1, 50, 0, false, false, true).contains("\"action\":3"));
        assertEquals(0, metrics.shotsFired);
    }
}
