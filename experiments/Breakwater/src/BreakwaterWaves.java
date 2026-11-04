import java.util.ArrayList;
import java.util.List;

/** Consecutive legal scans and same-tick combat corrections. Carries no state across rounds. */
final class BreakwaterWaves {
    private final EnemyFireTracker energy = new EnemyFireTracker();
    final List<BreakwaterPolicy.Wave> active = new ArrayList<>();
    BreakwaterPolicy.Target latest;
    int owner = -1;
    private final int maxInactivityTurns;
    private int earliestShot;

    BreakwaterWaves(int maxInactivityTurns) {
        this.maxInactivityTurns = maxInactivityTurns;
    }
    private int ownTurn;
    private double ownX, ownY;

    void damage(int id, int turn, double value) { energy.recordBulletDamage(id, turn, value); }
    void bonus(int id, int turn, double power) { energy.recordBulletHitBonus(id, turn, power); }

    void own(int turn, double x, double y) {
        ownTurn = turn;
        ownX = x;
        ownY = y;
        active.removeIf(w -> (turn - w.creation()) * (20 - 3 * w.power()) >
                Math.hypot(x - w.x(), y - w.y()) + 36);
        energy.prune(turn);
    }

    boolean scan(int id, BreakwaterPolicy.Target scan, boolean contact, double width, double height, double cooling) {
        if (id != owner && latest != null) throw new IllegalStateException("Expected one opponent");
        if (latest != null && scan.turn() <= latest.turn()) return false;
        var before = latest;
        latest = scan;
        owner = id;
        if (before == null) return false;
        double power = energy.inferPower(id, before.turn(), scan.turn(), before.energy(), scan.energy());
        // After inactivity can begin, a minimum-power drop is indistinguishable from decay.
        if (scan.turn() >= maxInactivityTurns && power <= .1 + 1e-6) return false;
        if (scan.turn() < Math.ceil(3 / cooling - 1e-9)) return false;
        if (power == 0 || ownTurn != scan.turn() - 1 || scan.turn() < earliestShot || contact ||
                nearWall(before, width, height) || nearWall(scan, width, height)) return false;
        active.add(new BreakwaterPolicy.Wave(scan.turn(), before.x(), before.y(), power,
                CombatMath.bearing(ownX - before.x(), ownY - before.y())));
        earliestShot = scan.turn() + 1 + (int)Math.ceil((1 + power / 5) / cooling - 1e-9);
        return true;
    }

    /** A confirmed projectile identifies one wave only if owner, power, age and actual ray agree. */
    BreakwaterPolicy.Wave match(int turn, int id, double power, double x, double y, double direction) {
        if (id != owner) return null;
        BreakwaterPolicy.Wave matched = null;
        double heading = Math.toRadians(direction);
        for (var w : active) {
            if (turn < w.creation() || Math.abs(power - w.power()) > 1e-5) continue;
            double speed = 20 - 3 * power;
            double dx = x - w.x(), dy = y - w.y();
            double along = dx * Math.cos(heading) + dy * Math.sin(heading);
            double across = Math.abs(dx * Math.sin(heading) - dy * Math.cos(heading));
            // TurnProcessor advances once before publishing either impact event's bullet state.
            double observedRadius = speed * (turn - w.creation() + 1);
            if (across > 1e-3 || Math.abs(along - observedRadius) > 1e-3) continue;
            if (matched != null) return null;
            matched = w;
        }
        if (matched != null) active.remove(matched);
        return matched;
    }

    private static boolean nearWall(BreakwaterPolicy.Target t, double width, double height) {
        return t.x() <= 26 || t.x() >= width - 26 || t.y() <= 26 || t.y() >= height - 26;
    }
}
