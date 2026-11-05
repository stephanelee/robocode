import java.io.*;
import java.util.*;

/** Read-only mechanical gate diagnostic against the hash-pinned own Facet JAR. */
public final class GateReplay {
  static double d(String[] a, int i) { return Double.parseDouble(a[i]); }
  static int n(String[] a, int i) { return Integer.parseInt(a[i]); }
  public static void main(String[] args) throws Exception {
    var reader = new BufferedReader(new InputStreamReader(System.in));
    var output = new PrintWriter(System.out, true);
    var rules = new Physics.Rules(800, 600, .1);
    Learner.Enemy enemy = null;
    int count = 0;
    for (String line; (line = reader.readLine()) != null;) {
      if (++count > 100000 || line.length() > 16384) throw new IllegalArgumentException("input cap");
      String[] a = line.split("\\t", -1);
      if (a.length != 34) throw new IllegalArgumentException("input schema " + a.length);
      int turn = n(a, 0), target = n(a, 10);
      var own = new Physics.Body(new Physics.V(d(a, 1), d(a, 2)), 0, 0, d(a, 3), 0, d(a, 4));
      double energy = d(a, 5), power = d(a, 9);
      var original = new Physics.Body(new Physics.V(d(a, 19), d(a, 20)), d(a, 21), d(a, 22), 0, 0, 0);
      var latest = new Physics.Body(new Physics.V(d(a, 28), d(a, 29)), d(a, 30), d(a, 31), 0, 0, 0);
      if (enemy == null || enemy.id != target)
        enemy = new Learner.Enemy(target, new Memory.Budget());
      enemy.latest = new Memory.Sample(1, n(a, 27), latest, d(a, 32), Memory.Quality.OBSERVED, 0);
      enemy.alive = n(a, 33) == 1;
      var fire = new FireControl();
      fire.plan = new FireControl.Plan(d(a, 8), power, target, n(a, 11), n(a, 12), d(a, 13),
          new Physics.V(d(a, 14), d(a, 15)), n(a, 16), new Physics.V(d(a, 17), d(a, 18)), 0,
          original, new FireControl.Policy(n(a, 23), target, d(a, 24), d(a, 25), n(a, 26)), "audit");
      if (n(a, 7) == 1) fire.pending = new FireControl.Pending(0, turn, power, target, own.gun(), own.p());
      boolean valid = fire.valid(own, energy, enemy, turn, rules);
      var predicted = new Physics.Body(Physics.constantPosition(original,
          Math.max(0, enemy.latest.turn() - fire.plan.observationTurn()), rules),
          original.heading(), original.speed(), 0, 0, 0);
      double drift = Math.abs(Physics.wrap(Physics.intercept(own.p(), latest, power, rules)
          - Physics.intercept(own.p(), predicted, power, rules)));
      double half = Math.asin(Math.min(1, 18 / Math.max(18, own.p().distance(latest.p()))));
      double policyPower = Energy.power(fire.plan.policy().melee() ? fire.plan.policy().range()
          : own.p().distance(latest.p()), energy, enemy.latest.energy(), fire.plan.policy().melee());
      var failed = new ArrayList<String>();
      if (fire.pending != null) failed.add("pending");
      if (!enemy.alive) failed.add("targetDead");
      if (drift > half) failed.add("analyticDrift");
      if (turn > fire.plan.expires() || turn < fire.plan.created()) failed.add("planAge");
      if (turn - enemy.latest.turn() > 2) failed.add("scanAge");
      if (latest.p().distance(fire.plan.targetPosition()) > 16.01 * Math.max(1, enemy.latest.turn() - fire.plan.observationTurn())) failed.add("targetDisplacement");
      if (own.p().distance(fire.plan.origin()) > 16.01) failed.add("ownDisplacement");
      if (own.heat() != 0) failed.add("heat");
      if (energy <= power) failed.add("energy");
      if (power > policyPower + 1e-12) failed.add("powerPolicy");
      if (Math.abs(Physics.wrap(own.gun() - fire.plan.angle())) > fire.plan.tolerance()) failed.add("alignment");
      if (valid != failed.isEmpty()) throw new IllegalStateException("predicate/source disagreement");
      output.println(valid + "\t" + drift + "\t" + half + "\t" + policyPower + "\t" + String.join(",", failed));
    }
  }
}
