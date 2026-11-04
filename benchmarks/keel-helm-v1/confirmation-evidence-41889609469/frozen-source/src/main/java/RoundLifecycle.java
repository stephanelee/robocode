import dev.robocode.tankroyale.botapi.events.RoundEndedEvent;

/** Collects final-tick rewards until round end; score rank never implies a win. */
final class RoundLifecycle {
    record Completion(int round, int turns, double reward, boolean won, boolean died) {}
    double pendingReward;
    private boolean finished = true, won, died;
    private double outcomeReward;

    void start() {
        pendingReward = outcomeReward = 0;
        finished = won = died = false;
    }

    void addReward(double reward) { pendingReward += reward; }
    void won() { won = true; outcomeReward = 60; }
    void died(int enemies) { died = true; outcomeReward = -(60 + 12.0 * enemies); }

    Completion finish(RoundEndedEvent event) {
        if (finished) return null;
        finished = true;
        Completion result = new Completion(event.getRoundNumber(), event.getTurnNumber(),
                pendingReward + outcomeReward, won, died);
        pendingReward = 0;
        return result;
    }
}
