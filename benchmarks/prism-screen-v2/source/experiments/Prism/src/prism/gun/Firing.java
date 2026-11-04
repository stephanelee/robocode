package prism.gun;

import prism.Profile;
import prism.core.Angles;
import prism.core.Rules;

/**
 * Power policy and firing-plan validation (specification section 8.3).
 *
 * Engine contract (Tank Royale 1.4.0 GunEngine): the fire phase runs first in a turn and uses the
 * gun direction at the start of that turn, i.e. the heading observed in the tick on which the
 * request is sent. A same-turn gun rotation never applies before the shot. The engine fires only
 * if heat is exactly zero, firepower >= 0.1 and energy > firepower.
 */
public final class Firing {
    private Firing() {}

    /** A prepared shot: created on turn createdTurn for a fire request on turn createdTurn+1. */
    public record Plan(int targetId, double angle, double power, int createdTurn, int expiresTurn, double tolerance,
                       double originX, double originY, long stateVersion, String gun, int targetScanTurn) {}

    /** A fire request submitted to the engine and not yet confirmed. */
    public record Pending(int requestTurn, int expectedConfirmTurn, double power, double angle, double originX, double originY,
                          int targetId, long waveId) {}

    public static double duelPower(double range, double ownEnergy, double targetEnergy) {
        double p = range < Profile.CLOSE_RANGE ? Profile.CLOSE_POWER : Profile.DUEL_POWER;
        if (range > 400 && ownEnergy < 30) p *= Math.max(0.1, Math.pow(ownEnergy / 30, 2));
        return finish(p, ownEnergy, targetEnergy);
    }

    public static double meleePower(double targetRange, double ownEnergy, double targetEnergy) {
        double p = Profile.MELEE_POWER;
        if (targetRange > 500) p = 1.5;
        if (ownEnergy < 20) p = 1.0;
        return finish(p, ownEnergy, targetEnergy);
    }

    /** Kill cap, legal clamp and reserve. Returns 0 when no legal shot is affordable. */
    static double finish(double p, double ownEnergy, double targetEnergy) {
        if (Double.isFinite(targetEnergy)) {
            if (targetEnergy <= 0) p = Rules.MIN_POWER;
            else p = Math.min(p, Rules.killPower(targetEnergy));
        }
        double available = ownEnergy - Profile.ENERGY_RESERVE;
        if (available < Rules.MIN_POWER) return 0;
        p = Math.min(p, available);
        p = Math.max(Rules.MIN_POWER, Math.min(Rules.MAX_POWER, p));
        // engine requires energy strictly greater than firepower
        if (!(ownEnergy > p)) return 0;
        return p;
    }

    public enum Verdict { FIRE, NO_PLAN, EXPIRED, HOT, ENERGY, STALE_TARGET, HEADING, TARGET_DEAD }

    /** Validates a prior plan against actual state at the turn the request would be sent. */
    public static Verdict validate(Plan plan, int turn, double gunHeading, double gunHeat, double energy,
                                   int targetLatestScanTurn, boolean targetAlive, double ownX, double ownY) {
        if (plan == null) return Verdict.NO_PLAN;
        if (turn > plan.expiresTurn() || turn <= plan.createdTurn()) return Verdict.EXPIRED;
        if (!targetAlive) return Verdict.TARGET_DEAD;
        if (gunHeat != 0.0) return Verdict.HOT;
        if (!(plan.power() >= Rules.MIN_POWER) || !(energy > plan.power())) return Verdict.ENERGY;
        if (turn - targetLatestScanTurn > 2) return Verdict.STALE_TARGET;
        if (Math.hypot(ownX - plan.originX(), ownY - plan.originY()) > 1.0) return Verdict.EXPIRED; // incompatible own displacement
        if (Math.abs(Angles.wrap(gunHeading - plan.angle())) > plan.tolerance()) return Verdict.HEADING;
        return Verdict.FIRE;
    }
}
