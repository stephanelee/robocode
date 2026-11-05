import java.util.Map;
import java.util.HashMap;
import java.util.Collection;

/** Per-round legal histories, isolated by observed opponent ID. No synthetic waves. */
final class MeleeTransfer {
    static final class State {
        final Track own = new Track(), enemy = new Track();
        double x, y, energy;
        int turn = -1, scans;
    }
    private final Map<Integer, State> histories = new HashMap<>();
    void clear() { histories.clear(); }
    void died(int id) { histories.remove(id); }
    void observe(int id, int turn, Phys.Pose enemy, double energy, double ownX, double ownY) {
        State s = histories.computeIfAbsent(id, unused -> new State());
        if (turn <= s.turn) return;
        s.enemy.add(turn, enemy, ownX, ownY); s.x = enemy.x(); s.y = enemy.y(); s.energy = energy; s.turn = turn; s.scans++;
    }
    void tick(int turn, Phys.Pose own) {
        for (State s : histories.values()) if (turn - s.turn <= 32) s.own.add(turn, own, s.x, s.y);
    }
    State survivor(int turn, Collection<MeleePolicy.Opponent> live) {
        if (live.size() != 1) return null;
        State s = histories.get(live.iterator().next().id);
        return s != null && turn - s.turn <= 8 ? s : null;
    }
}
