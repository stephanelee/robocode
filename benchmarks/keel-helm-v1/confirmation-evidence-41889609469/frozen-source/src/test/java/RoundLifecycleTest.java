import dev.robocode.tankroyale.botapi.BotResults;
import dev.robocode.tankroyale.botapi.events.RoundEndedEvent;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class RoundLifecycleTest {
    private RoundEndedEvent rankOneResult() {
        // Results are cumulative game scores; rank 1 does not identify this round's survivor.
        return new RoundEndedEvent(1, 200, new BotResults(1, 100, 0, 0, 0, 0, 0, 100, 0, 0, 0));
    }

    @Test void aliveCumulativeRankOneWithoutWonEventGetsNoWinReward() {
        RoundLifecycle lifecycle = new RoundLifecycle();
        lifecycle.start();
        lifecycle.addReward(4.5);
        RoundLifecycle.Completion completed = lifecycle.finish(rankOneResult());
        assertFalse(completed.won());
        assertFalse(completed.died());
        assertEquals(4.5, completed.reward());
        assertNull(lifecycle.finish(rankOneResult()));
    }

    @Test void winAndDeathKeepFinalCombatRewardsAndCompleteExactlyOnce() {
        RoundLifecycle lifecycle = new RoundLifecycle();
        lifecycle.start();
        lifecycle.addReward(3);
        lifecycle.won();
        lifecycle.addReward(15); // final-tick event delivered after WonRound
        RoundLifecycle.Completion won = lifecycle.finish(rankOneResult());
        assertTrue(won.won());
        assertEquals(78, won.reward());
        assertNull(lifecycle.finish(rankOneResult()));
        lifecycle.start();
        lifecycle.addReward(2);
        lifecycle.died(2);
        lifecycle.addReward(-5); // final-tick event delivered after Death
        RoundLifecycle.Completion died = lifecycle.finish(rankOneResult());
        assertTrue(died.died());
        assertFalse(died.won());
        assertEquals(-87, died.reward());
        assertEquals(0, lifecycle.pendingReward);
        assertNull(lifecycle.finish(rankOneResult()));
    }
}
