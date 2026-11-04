package prism.radar;

import prism.Profile;
import prism.core.Angles;
import prism.core.Rules;

import java.util.List;

/**
 * Decoupled radar (specification section 11). Engine contract: the scan arc runs from the radar
 * direction at the start of the turn to the new direction, from the bot's end-of-movement
 * position, after all bots have moved. Bearings use the movement stage's predicted next own
 * position and only the latest legal enemy observations.
 */
public final class Radar {
    public record Known(int id, double x, double y, int lastScanTurn) {}

    private int direction = 1;
    private int mismatchTurns;
    private int lastScanAny = Integer.MIN_VALUE;

    public void resetRound() { direction = 1; mismatchTurns = 0; lastScanAny = Integer.MIN_VALUE; }

    public int direction() { return direction; }

    /**
     * @param expected live opponent count from the engine; known: our tracked live opponents.
     * @return radar turn in native radians.
     */
    public double command(int turn, double radarHeading, double nextX, double nextY, int expected, List<Known> known) {
        int maxAge = 0;
        for (Known k : known) maxAge = Math.max(maxAge, turn - k.lastScanTurn());
        if (known.size() != expected) mismatchTurns++; else mismatchTurns = 0;
        boolean forceAcquire = mismatchTurns >= Profile.RADAR_MISMATCH_TURNS;
        if (forceAcquire) mismatchTurns = 0;
        double limit = Math.toRadians(Rules.MAX_RADAR_TURN_DEG);

        if (expected == 1 && known.size() == 1 && !forceAcquire) {
            Known k = known.get(0);
            int age = turn - k.lastScanTurn();
            if (age <= 1) {
                double need = Angles.wrap(Angles.bearing(nextX, nextY, k.x(), k.y()) - radarHeading);
                int s = need >= 0 ? 1 : -1;
                double room = limit - Math.abs(need);
                double over = Math.min(Profile.RADAR_OVERSHOOT, Math.max(0, room));
                if (room >= Profile.RADAR_MIN_OVERSHOOT) over = Math.max(over, Profile.RADAR_MIN_OVERSHOOT);
                direction = s;
                return clamp(need + s * over, limit);
            }
            if (age <= Profile.RADAR_FULL_SWEEP_AFTER) {
                // sweep in the prior direction toward the uncertainty sector
                double need = Angles.wrap(Angles.bearing(nextX, nextY, k.x(), k.y()) - radarHeading);
                if (Math.abs(need) > Math.toRadians(60)) direction = need >= 0 ? 1 : -1;
                return direction * limit;
            }
            return direction * limit;
        }
        if (known.size() < expected || forceAcquire) return direction * limit; // acquisition sweep
        // melee: stalest participant (ties: smaller turn needed, then id)
        Known target = null;
        double targetNeed = 0;
        for (Known k : known) {
            double need = Angles.wrap(Angles.bearing(nextX, nextY, k.x(), k.y()) - radarHeading);
            if (target == null) { target = k; targetNeed = need; continue; }
            int a = turn - k.lastScanTurn(), b = turn - target.lastScanTurn();
            if (a > b || (a == b && (Math.abs(need) < Math.abs(targetNeed) || (Math.abs(need) == Math.abs(targetNeed) && k.id() < target.id())))) {
                target = k; targetNeed = need;
            }
        }
        if (target == null) return direction * limit;
        int s = targetNeed >= 0 ? 1 : -1;
        // keep sweep direction if reversing would repeat a sector just covered
        if (s != direction && Math.abs(targetNeed) < Math.toRadians(30) && turn - target.lastScanTurn() <= 1) s = direction;
        direction = s;
        double over = Math.min(Profile.RADAR_OVERSHOOT, Math.max(0, limit - Math.abs(targetNeed)));
        return clamp(targetNeed + s * over, limit);
    }

    private static double clamp(double v, double lim) { return Math.max(-lim, Math.min(lim, v)); }
}
