package prism.track;

import prism.Profile;
import prism.core.Rules;

/**
 * Per-enemy record. Learning views live in the gun and movement modules keyed by the same id;
 * this class holds observations, motion timers, energy ledger and counters. Round reset clears
 * transient state; cumulative counters survive the game.
 */
public final class Enemy {
    public final int id;
    public boolean alive = true;
    public final History history = new History(Profile.HISTORY_TICKS, Profile.MAX_INTERP_GAP);
    public final FireInference fire;
    public final PowerModel power = new PowerModel();

    /** Last nonzero movement direction sign (sign of speed). */
    public int lastMoveSign = 0;
    public int lastVelocityChangeTurn = 0;
    public int lastReversalTurn = 0;
    public double acceleration = 0;
    public int lastWallContactTurn = Integer.MIN_VALUE;

    // cumulative game counters
    public double damageToUs = 0;
    public double damageFromUs = 0;
    public int turnsObservedTogether = 0;
    public int scans = 0;

    public Enemy(int id, Rules rules) { this.id = id; this.fire = new FireInference(rules); }

    public void resetRound() {
        alive = true;
        history.clear();
        fire.resetRound();
        lastMoveSign = 0;
        lastVelocityChangeTurn = 0;
        lastReversalTurn = 0;
        acceleration = 0;
        lastWallContactTurn = Integer.MIN_VALUE;
    }

    public History.Sample latest() { return history.latest(); }

    public int scanAge(int turn) {
        History.Sample s = history.latest();
        return s == null ? Integer.MAX_VALUE : turn - s.turn();
    }

    /** Records a scan and updates motion timers. Returns the previous observation or null. */
    public History.Sample observe(int turn, double x, double y, double heading, double speed, double energy) {
        History.Sample prev = history.latest();
        history.add(new History.Sample(turn, x, y, heading, speed, energy, History.Provenance.OBSERVED));
        scans++;
        if (prev != null && turn > prev.turn()) {
            int dt = turn - prev.turn();
            double a = (Math.abs(speed) - Math.abs(prev.speed())) / dt;
            acceleration = a;
            if (Math.abs(speed - prev.speed()) > 0.5) lastVelocityChangeTurn = turn;
        } else if (prev == null) {
            lastVelocityChangeTurn = turn;
            lastReversalTurn = turn;
        }
        int s = speed > 1e-9 ? 1 : speed < -1e-9 ? -1 : 0;
        if (s != 0) {
            if (lastMoveSign != 0 && s != lastMoveSign) lastReversalTurn = turn;
            lastMoveSign = s;
        }
        return prev;
    }
}
