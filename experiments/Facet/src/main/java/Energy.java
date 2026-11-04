import java.util.*;
final class Energy {
  record Effect(int visibleTurn, double loss, String identity) {}
  record Inference(
      int earliest, int latest, double low, double high, double confidence, boolean reliable) {}
  static double cool(double heat, int ticks, double rate) {
    while (ticks-- > 0 && heat > 0) heat = Math.max(0, heat - rate);
    return heat;
  }
  static final class Ledger {
    final Map<String, Effect> effects = new LinkedHashMap<>();
    double energy, heatLow = 3, heatHigh = 3;
    int turn = -1;
    long ambiguous, corrected, duplicates;
    void effect(Effect e) {
      if (effects.putIfAbsent(e.identity, e) != null)
        duplicates++;
      while (effects.size() > 512) effects.remove(effects.keySet().iterator().next());
    }
    Optional<Inference> scan(int t, double next, double cooling, boolean collisionPossible) {
      if (turn < 0) {
        turn = t;
        energy = next;
        heatLow = heatHigh = cool(3, t, cooling);
        if (t > 0 && cool(3, t - 1, cooling) == 0)
          heatHigh = Math.max(heatHigh, 1.6);
        return Optional.empty();
      }
      int old = turn, dt = t - turn;
      if (dt <= 0)
        return Optional.empty();
      double heatBeforeLow = heatLow, heatBeforeHigh = heatHigh;
      heatLow = cool(heatLow, dt, cooling);
      heatHigh = cool(heatHigh, dt, cooling);
      double residual = energy - next;
      for (Effect e : effects.values())
        if (e.visibleTurn > old && e.visibleTurn <= t) {
          residual -= e.loss;
          corrected++;
        }
      effects.values().removeIf(e -> e.visibleTurn <= old);
      turn = t;
      energy = next;
      if (dt > 1 && cool(heatBeforeLow, dt - 1, cooling) == 0)
        heatHigh = Math.max(heatHigh, 1.6);
      if (Math.abs(residual) < 1e-7)
        return Optional.empty();
      boolean possibleCool = cool(heatBeforeLow, dt - 1, cooling) == 0;
      if (residual >= .1 - 1e-7 && residual <= 3 + 1e-7 && possibleCool && !collisionPossible
          && dt == 1 && heatBeforeHigh == 0) {
        double p = Physics.clamp(residual, .1, 3);
        heatLow = heatHigh = 1 + p / 5;
        return Optional.of(new Inference(old, old, p, p, 1, true));
      }
      ambiguous++;
      heatLow = 0;
      heatHigh = Math.max(heatHigh, 1.6);
      if (residual > .1 && possibleCool)
        return Optional.of(new Inference(old, t - 1, .1, 3, .35, false));
      return Optional.empty();
    }
    void reset() {
      effects.clear();
      turn = -1;
      heatLow = heatHigh = 3;
    }
  }
  static double power(double range, double own, double target, boolean melee) {
    double p = melee ? (own < 20 ? 1 : range > 500 ? 1.5 : 2.5) : (range < 180 ? 2.8 : 2);
    if (!melee && range > 400 && own < 30)
      p *= Math.max(.1, Math.pow(own / 30, 2));
    double kill = target <= 4 ? target / 4 : (target + 2) / 6;
    p = Math.min(p, Math.max(.1, kill));
    double available = own - .2;
    if (available < .1)
      return 0;
    return Physics.clamp(Math.min(p, available), .1, 3);
  }
  static double wilsonLower(long hits, long shots) {
    if (hits < 0 || shots < 0 || hits > shots)
      throw new IllegalArgumentException("inconsistent Wilson counts");
    if (shots == 0)
      return 0;
    double z = 1.95996398454, n = shots, p = hits / n;
    return (p + z * z / (2 * n) - z * Math.sqrt((p * (1 - p) + z * z / (4 * n)) / n))
        / (1 + z * z / n);
  }
}
