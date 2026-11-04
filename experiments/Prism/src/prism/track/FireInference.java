package prism.track;

import prism.Profile;
import prism.core.Rules;

import java.util.ArrayList;
import java.util.List;

/**
 * Per-enemy energy accountant and gun-heat interval (specification section 6).
 *
 * Visibility contract (Tank Royale 1.4.0 turn order: fire, move, wall, ram, scan, bullets): the
 * enemy's firing cost, wall damage and ram damage on turn k are visible in a scan on turn k;
 * bullet damage and hit bonus on turn k become visible in a scan on turn k+1. Known effects are
 * registered with their expected visible turn and consumed exactly once.
 */
public final class FireInference {
    /** An energy effect with the turn on which a scan first shows it; delta is signed (gain > 0). */
    public record Effect(int visibleTurn, double delta, double uncertainty, String kind) {}

    /** A detected or hypothesised shot. fireTurn interval is inclusive. */
    public record Shot(int fireTurnMin, int fireTurnMax, double power, double powerUncertainty, boolean certain, int hypotheses) {}

    private final Rules rules;
    private final List<Effect> pending = new ArrayList<>();
    private int lastScanTurn = Integer.MIN_VALUE;
    private double lastEnergy = Double.NaN;
    /** Turn on which heat first reaches exactly zero (end of turn), as an interval. */
    private int readyMin, readyMax;
    private int unresolved;

    public FireInference(Rules rules) { this.rules = rules; resetRound(); }

    public void resetRound() {
        pending.clear();
        lastScanTurn = Integer.MIN_VALUE;
        lastEnergy = Double.NaN;
        int r = rules.turnsToCool(Rules.INITIAL_GUN_HEAT);
        readyMin = r;
        readyMax = r;
    }

    public void addEffect(Effect e) { pending.add(e); }

    /** True if a shot on turn t is consistent with the heat interval. */
    public boolean canFireAt(int t) { return t >= readyMin + 1; }
    /** True if heat is certainly zero for a shot on turn t. */
    public boolean certainlyCoolAt(int t) { return t >= readyMax + 1; }
    public int earliestFireTurn() { return readyMin + 1; }
    public int latestReadyTurn() { return readyMax; }
    public int unresolvedCount() { return unresolved; }

    /**
     * Processes a scan. Returns the detected shot or null. A shot is reported only when the
     * residual is a legal power and heat permits; a gap creates a time interval (at most
     * {@link Profile#MAX_SHOT_HYPOTHESES} feasible turns are certain enough to keep exact-turn
     * hypotheses; more becomes an envelope with reduced confidence).
     */
    public Shot onScan(int turn, double energy, double wallDamageMin, double wallDamageMax) {
        Shot result = null;
        if (!Double.isNaN(lastEnergy) && turn > lastScanTurn) {
            double known = 0, unc = 0;
            for (var it = pending.iterator(); it.hasNext(); ) {
                Effect e = it.next();
                if (e.visibleTurn() > lastScanTurn && e.visibleTurn() <= turn) { known += e.delta(); unc += e.uncertainty(); it.remove(); }
                else if (e.visibleTurn() <= lastScanTurn) it.remove();
            }
            double loss = lastEnergy - energy + known;            // energy not explained by known effects
            double costMax = loss - wallDamageMin + unc;
            double costMin = loss - wallDamageMax - unc;
            int gap = turn - lastScanTurn;
            double tol = Profile.ENERGY_TOLERANCE;
            if (costMax >= Rules.MIN_POWER - tol && costMin < Rules.MIN_POWER - tol) {
                // an unknown effect (e.g. wall damage) could explain the loss: uncertainty, not a shot
                unresolved++;
                int lo = Math.max(lastScanTurn + 1, readyMin + 1);
                if (lo <= turn) readyMax = Math.max(readyMax, turn + rules.turnsToCool(Rules.gunHeat(Math.min(Rules.MAX_POWER, costMax))));
            } else if (costMax >= Rules.MIN_POWER - tol && costMin <= Rules.MAX_POWER + tol) {
                int lo = Math.max(lastScanTurn + 1, readyMin + 1);
                int hi = turn;
                if (lo <= hi) {
                    double power = Math.max(Rules.MIN_POWER, Math.min(Rules.MAX_POWER, (costMin + costMax) / 2));
                    double pu = Math.max(0, (costMax - costMin) / 2);
                    int n = hi - lo + 1;
                    boolean certain = gap == 1 && n == 1 && pu < 1e-3 && certainlyCoolAt(turn);
                    result = new Shot(lo, hi, power, pu, certain, Math.min(n, Profile.MAX_SHOT_HYPOTHESES + 1));
                    int cool = rules.turnsToCool(Rules.gunHeat(power));
                    readyMin = lo + cool;
                    readyMax = hi + cool;
                } else {
                    unresolved++; // loss with heat forbidding a shot: unexplained, not a shot
                }
            } else if (costMin > Rules.MAX_POWER + tol && gap > 1) {
                unresolved++; // possibly several hidden shots: widen heat, do not invent exact shots
                readyMin = lastScanTurn + 1;
                readyMax = turn + rules.turnsToCool(Rules.gunHeat(Rules.MAX_POWER));
                result = new Shot(lastScanTurn + 1, turn, Rules.MAX_POWER, 0, false, Profile.MAX_SHOT_HYPOTHESES + 1);
            }
        }
        lastScanTurn = turn;
        lastEnergy = energy;
        return result;
    }
}
