import java.util.HashMap;
import java.util.Map;

/** Accounts for observable energy changes before inferring an enemy shot. */
final class EnemyFireTracker {
    private final Map<Integer, Map<Integer, Double>> corrections = new HashMap<>();

    void recordDamage(int botId, int turn, double damage) {
        record(botId, turn, -damage);
    }

    void recordHitBonus(int botId, int turn, double power) {
        record(botId, turn, 3 * power);
    }

    // Pinned 1.4.0 engine scans before applying bullet impacts. These corrections
    // become visible at k+1; firing costs and wall/ram events are not shifted.
    void recordBulletDamage(int botId, int eventTurn, double damage) {
        recordDamage(botId, eventTurn + 1, damage);
    }

    void recordBulletHitBonus(int botId, int eventTurn, double power) {
        recordHitBonus(botId, eventTurn + 1, power);
    }

    private void record(int botId, int turn, double amount) {
        corrections.computeIfAbsent(botId, ignored -> new HashMap<>())
                .merge(turn, amount, Double::sum);
    }

    double inferPower(int botId, int previousTurn, int scanTurn, double previousEnergy,
                      double energy) {
        Map<Integer, Double> history = corrections.get(botId);
        double correction = 0;
        if (history != null) {
            for (var entry : history.entrySet())
                if (entry.getKey() > previousTurn && entry.getKey() <= scanTurn)
                    correction += entry.getValue();
            history.keySet().removeIf(turn -> turn <= scanTurn);
            if (history.isEmpty()) corrections.remove(botId);
        }
        // A gap can hide multiple shots, damage by other bots, and unknown fire times.
        if (scanTurn - previousTurn != 1) return 0;
        double spent = previousEnergy + correction - energy;
        return spent >= 0.1 - 1e-6 && spent <= 3.0 + 1e-6
                ? Math.max(0.1, Math.min(3, spent)) : 0;
    }

    void prune(int turn) {
        corrections.values().forEach(history -> history.keySet().removeIf(t -> t < turn - 80));
        corrections.values().removeIf(Map::isEmpty);
    }

    void clear() { corrections.clear(); }
}
