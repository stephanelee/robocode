import com.google.gson.JsonParser;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class DefaultDuelPolicyTest {
    private BotConfig config(BotConfig.Mode mode, BotConfig.DuelPolicy policy) {
        return new BotConfig(mode, 20261701, Path.of("unused-duel-policy-test"), 3, null, policy);
    }

    @Test void forcedActionIsSelectedAndCreditedWithoutExplorationDraw() {
        var a = new DQNAgent(14, 6, config(BotConfig.Mode.TRAIN, BotConfig.DuelPolicy.CONTROLLER));
        var control = new DQNAgent(14, 6, config(BotConfig.Mode.TRAIN, BotConfig.DuelPolicy.CONTROLLER));
        a.policy.b2[5] = 1000; // A learned/greedy decision would prefer another action.
        a.startEpisode();
        assertEquals(0, a.observe(new double[14], 0, false, true));
        assertEquals(0, a.lastAction);
        assertEquals(control.rng.nextLong(), a.rng.nextLong(), "forced choice must not sample exploration");
        assertEquals(0, a.observe(new double[14], 7, false, true));
        assertEquals(0, a.replay.get(0)[14]);
        assertEquals(7, a.replay.get(0)[15]);
        assertEquals(1, a.replay.get(0)[31]);
        a.finishEpisode(60); a.finishEpisode(-999);
        assertEquals(2, a.replay.size());
        assertEquals(0, a.replay.get(1)[14]);
        assertEquals(60, a.replay.get(1)[15]);
        assertEquals(1, a.replay.get(1)[30]);
        assertEquals(1, a.totalRounds);
    }

    @Test void nextStateConstraintUsesQ0ButMeleeTransitionKeepsSixActionBootstrap() throws Exception {
        var a = new DQNAgent(14, 6, config(BotConfig.Mode.TRAIN, BotConfig.DuelPolicy.CONTROLLER));
        a.startEpisode();
        int previous = a.observe(new double[14], 0, false, false);
        a.observe(new double[14], 7, false, true);
        a.target = mock(MLP.class);
        when(a.target.forward(any())).thenReturn(new double[]{2, 3, 4, 5, 6, 100});
        a.policy = spy(a.policy);
        doNothing().when(a.policy).trainSingle(any(), anyInt(), anyDouble(), anyDouble());
        Method learn = DQNAgent.class.getDeclaredMethod("replayLearnOne"); learn.setAccessible(true);
        learn.invoke(a);
        verify(a.policy).trainSingle(any(), eq(previous), eq(7 + .95 * 2), anyDouble());
        clearInvocations(a.policy);
        a.observe(new double[14], 4, false, false);
        a.replay.remove(0); // The sole remaining sample leaves the forced duel state for unrestricted play.
        learn.invoke(a);
        verify(a.policy).trainSingle(any(), eq(0), eq(4 + .95 * 100), anyDouble());
    }

    @Test void inactivePathKeepsLegacyActionsRngAndLearning() {
        var a = new DQNAgent(14, 6, config(BotConfig.Mode.TRAIN, BotConfig.DuelPolicy.CONTROLLER));
        var legacy = new DQNAgent(14, 6, config(BotConfig.Mode.TRAIN, BotConfig.DuelPolicy.LEARNED));
        a.startEpisode(); legacy.startEpisode();
        double[] f = new double[14];
        for (int i = 0; i < 75; i++) {
            f[i % 14] = i / 75.0;
            assertEquals(legacy.observe(f, i % 3, false, false), a.observe(f, i % 3, false, false));
        }
        assertArrayEquals(legacy.policy.forward(f), a.policy.forward(f));
        assertEquals(legacy.rng.nextLong(), a.rng.nextLong());
    }

    @Test void evalFixedAndExplicitLearnedFallbackRemainUnforced() {
        for (var mode : new BotConfig.Mode[]{BotConfig.Mode.EVAL, BotConfig.Mode.FIXED}) {
            var a = new DQNAgent(14, 6, config(mode, BotConfig.DuelPolicy.CONTROLLER));
            a.policy.b2[5] = 1000;
            assertEquals(mode == BotConfig.Mode.FIXED ? 3 : 5, a.observe(new double[14], 0, false, true));
        }
        var a = new DQNAgent(14, 6, config(BotConfig.Mode.TRAIN, BotConfig.DuelPolicy.LEARNED));
        a.policy.b2[5] = 1000;
        assertEquals(5, a.observe(new double[14], 0, false, true));
    }

    @Test void environmentAndActionTelemetryAreExplicitAndResetPerRound() {
        assertTrue(BotConfig.fromEnvironment(Map.of()).forceDuel());
        assertFalse(BotConfig.fromEnvironment(Map.of("AISURVIVOR_DUEL_POLICY", "learned")).forceDuel());
        assertThrows(IllegalArgumentException.class, () -> BotConfig.fromEnvironment(Map.of("AISURVIVOR_DUEL_POLICY", "invalid")));
        var metrics = new RoundMetrics();
        metrics.actionCounts[0] = 3; metrics.duelEligibleTurns = metrics.forcedDuelTurns = 3;
        var data = JsonParser.parseString(metrics.json(config(BotConfig.Mode.TRAIN, BotConfig.DuelPolicy.CONTROLLER),
                1, 5, 1, false, false, true)).getAsJsonObject();
        assertEquals("train", data.get("mode").getAsString());
        assertEquals("controller", data.get("duelPolicy").getAsString());
        assertEquals(3, data.getAsJsonArray("actionCounts").get(0).getAsInt());
        assertEquals(3, data.get("forcedDuelTurns").getAsInt());
        metrics.reset();
        assertEquals(0, metrics.actionCounts[0]); assertEquals(0, metrics.forcedDuelTurns);
    }
}
