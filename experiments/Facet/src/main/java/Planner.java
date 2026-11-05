import java.util.*;
final class Planner {
  static final int STEP_LIMIT = 2400, TICK_LIMIT = 160;
  double arenaWidth = 800, arenaHeight = 600, arenaCooling = .1;
  int safetyTurn = -1;
  int steps, previous = -1, stepLimit = STEP_LIMIT;
  final Map<Long, Geometry.Density> densityCache = new HashMap<>();
  long truncations, missingGeometry, emergencies;
  record Choice(Physics.Control command, int action, double cost, boolean complete,
      boolean emergency, List<Physics.Body> path) {}
  record Path(double cost, boolean complete, boolean safe, Physics.Body crossing, int crossingTurn,
      List<Physics.Body> states) {}
  Physics.Control orbit(
      Physics.Body self, Physics.V source, int action, Physics.Rules r, boolean surfing) {
    int direction = action == 0 ? previous : action;
    double range = self.p().distance(source),
           desired = Math.min(600, .65 * Math.min(r.width(), r.height())),
           a = Physics.clamp((range - desired) / desired, -.8, .8),
           angle = source.bearing(self.p()) + direction * (Math.PI / 2 + (surfing ? .5 : 1.2) * a);
    Physics.V destination = self.p();
    for (int i = 0; i <= 180; i++) {
      destination = self.p().add(Physics.vector(angle).mul(140));
      if (r.safe(destination))
        break;
      angle += direction * Math.PI / 180;
    }
    return Physics.toward(self, destination, action == 0);
  }
  Choice choose(Physics.Body self, int turn, List<Waves.Wave> threats,
      Map<Integer, Learner.Enemy> enemies, Physics.Rules r, Map<Integer, Integer> deaths) {
    safetyTurn = turn;
    activeEnemies = enemies.values();
    arenaWidth = r.width();
    arenaHeight = r.height();
    arenaCooling = r.cooling();
    steps = 0;
    densityCache.clear();
    List<Waves.Wave> ordered =
        threats.stream()
            .filter(w
                -> !w.cancelled && (w.role == Waves.Role.INCOMING || w.role == Waves.Role.FORECAST)
                    && w.phase(self.p(), turn) != Waves.Phase.FULLY_PASSED)
            .sorted(Comparator
                    .<Waves.Wave>comparingDouble(w
                        -> Math.max(0,
                            (w.origin.distance(self.p()) - 18 - w.outer(turn))
                                / Physics.bulletSpeed(w.powerLo)))
                    .thenComparingLong(w -> w.id))
            .limit(2)
            .toList();
    if (ordered.isEmpty()) {
      Physics.V source =
          enemies.values()
              .stream()
              .filter(e -> e.alive && e.latest != null)
              .min(Comparator.comparingDouble(e -> self.p().distance(e.latest.body().p())))
              .map(e -> e.latest.body().p())
              .orElse(new Physics.V(r.width() / 2, r.height() / 2));
      Physics.Control c = orbit(self, source, previous, r, false);
      if (firstSafe(self, c, turn, r))
        return new Choice(c, previous, 0, true, false, List.of());
      return emergency(self, r);
    }
    Choice best = null;
    for (int action : new int[] {previous, -previous, 0}) {
      Waves.Wave first = ordered.get(0);
      Physics.Control c = orbit(self, first.origin, action, r, true);
      Path p = simulate(self, turn, first, action, enemies.get(first.owner), r, deaths);
      if (!p.complete || !p.safe)
        continue;
      double cost = p.cost;
      if (ordered.size() > 1) {
        Waves.Wave second = ordered.get(1);
        double next = Double.POSITIVE_INFINITY;
        for (int continuation :
            new int[] {action == 0 ? previous : action, action == 0 ? -previous : -action, 0}) {
          List<Geometry.Arc> prior = new ArrayList<>();
          int contact = p.crossingTurn;
          for (int i = 1; i < p.states.size() && turn + i <= p.crossingTurn; i++) {
            var arcs = second.footprint(p.states.get(i - 1).p(), p.states.get(i).p(), turn + i);
            if (!arcs.isEmpty()) {
              if (prior.isEmpty())
                contact = turn + i;
              prior.addAll(arcs);
            }
          }
          Path q;
          if (second.phase(p.crossing.p(), p.crossingTurn) == Waves.Phase.FULLY_PASSED)
            q = new Path(prior.isEmpty() ? 0
                                         : waveCost(second, Geometry.merge(prior), contact, turn,
                                               enemies.get(second.owner), deaths),
                true, true, p.crossing, p.crossingTurn, List.of(p.crossing));
          else
            q = simulate(p.crossing, p.crossingTurn, second, continuation,
                enemies.get(second.owner), r, deaths, prior, contact, turn);
          if (q.complete && q.safe)
            next = Math.min(next, q.cost);
        }
        if (!Double.isFinite(next))
          continue;
        cost += next;
      }
      if (best == null || cost < best.cost - 1e-6
          || (Math.abs(cost - best.cost) <= 1e-6 && prefer(action, best.action)))
        best = new Choice(c, action, cost, true, false, p.states);
    }
    if (best == null) {
      for (int action : new int[] {previous, -previous}) {
        Physics.Control analytic = orbit(self, ordered.get(0).origin, action, r, false);
        if (firstSafe(self, analytic, turn, r))
          return new Choice(
              analytic, action, Physics.damage(ordered.get(0).powerHi), true, true, List.of());
      }
      return emergency(self, r);
    }
    if (best.action != 0)
      previous = best.action;
    return best;
  }
  boolean prefer(int a, int b) {
    if (a == previous)
      return true;
    if (b == previous)
      return false;
    if (a != 0 && b == 0)
      return true;
    return a == -1 && b != previous;
  }
  Path simulate(Physics.Body initial, int turn, Waves.Wave wave, int action, Learner.Enemy enemy,
      Physics.Rules r, Map<Integer, Integer> deaths) {
    return simulate(initial, turn, wave, action, enemy, r, deaths, List.of(), turn + 1, turn);
  }
  Path simulate(Physics.Body initial, int turn, Waves.Wave wave, int action, Learner.Enemy enemy,
      Physics.Rules r, Map<Integer, Integer> deaths, List<Geometry.Arc> prior, int priorContact,
      int scoringTurn) {
    if (wave.phase(initial.p(), turn) == Waves.Phase.FULLY_PASSED)
      return new Path(prior.isEmpty()
              ? 0
              : waveCost(wave, Geometry.merge(prior), priorContact, scoringTurn, enemy, deaths),
          true, true, initial, turn, List.of(initial));
    Physics.Body s = initial, cross = initial;
    int crossTurn = turn;
    boolean crossed = false, safe = true;
    double score = 0;
    List<Physics.Body> path = new ArrayList<>();
    path.add(s);
    List<Geometry.Arc> union = new ArrayList<>(prior);
    int contact = priorContact;
    for (int t = 1; t <= TICK_LIMIT; t++) {
      if (steps >= stepLimit || Physics.workRemaining() <= 350) {
        truncations++;
        return new Path(score, false, false, cross, crossTurn, List.copyOf(path));
      }
      steps++;
      Physics.Control c = orbit(s, wave.origin, action, r, true);
      Physics.Step next = Physics.step(s, c, r);
      if (t == 1 && (!r.safe(next.body().p()) || collisionRisk(next.body().p(), turn + 1, r) > 0))
        safe = false;
      score += next.damage();
      if (next.wall())
        score += 5;
      for (Learner.Enemy other : activeEnemies) {
        if (other.latest == null || !other.alive || turn + t - other.latest.turn() > 8)
          continue;
        double gap = next.body().p().distance(
            Physics.constantPosition(other.latest.body(), turn + t - other.latest.turn(), r));
        if (gap < 36) {
          score += .6;
          if (t == 1)
            safe = false;
        }
      }
      List<Geometry.Arc> arcs = wave.footprint(s.p(), next.body().p(), turn + t);
      if (!arcs.isEmpty()) {
        if (union.isEmpty())
          contact = turn + t;
        union.addAll(arcs);
      }
      s = next.body();
      path.add(s);
      if (!crossed && wave.inner(turn + t) >= wave.origin.distance(s.p())) {
        cross = s;
        crossTurn = turn + t;
        crossed = true;
      }
      if (wave.phase(s.p(), turn + t) == Waves.Phase.FULLY_PASSED) {
        if (union.isEmpty()) {
          missingGeometry++;
          score += Physics.damage(wave.powerHi) * wave.confidence;
        } else
          score += waveCost(wave, Geometry.merge(union), contact, scoringTurn, enemy, deaths);
        double desired = Math.min(600, .65 * Math.min(r.width(), r.height()));
        score +=
            .05 * Math.pow(Math.max(0, desired / Math.max(1, wave.origin.distance(s.p())) - 1), 2);
        return new Path(score, true, safe, crossed ? cross : s, crossed ? crossTurn : turn + t,
            List.copyOf(path));
      }
    }
    truncations++;
    return new Path(score, false, false, cross, crossTurn, List.copyOf(path));
  }
  Collection<Learner.Enemy> activeEnemies = List.of();
  double waveCost(Waves.Wave w, List<Geometry.Arc> arcs, int contact, int now, Learner.Enemy enemy,
      Map<Integer, Integer> deaths) {
    double density = enemy == null
        ? Math.max(.01, Geometry.width(arcs) / (2 * Physics.escape(w.powerHi)))
        : Geometry.integrate(arcs,
              densityCache.computeIfAbsent(w.id,
                  k
                  -> Learner.prepareDensity(
                      enemy, w, new Physics.Rules(arenaWidth, arenaHeight, arenaCooling))));
    double score = w.confidence * Physics.damage(w.powerHi) * density
        * w.unshadowed(arcs, contact, deaths) / Math.max(1, contact - now);
    return Double.isFinite(score) ? score : Physics.damage(w.powerHi);
  }
  boolean firstSafe(Physics.Body self, Physics.Control c, int turn, Physics.Rules r) {
    var next = Physics.step(self, c, r);
    return r.safe(next.body().p()) && collisionRisk(next.body().p(), turn + 1, r) == 0;
  }
  double collisionRisk(Physics.V point, int nextTurn, Physics.Rules r) {
    double risk = 0;
    for (var e : activeEnemies) {
      if (!e.alive || e.latest == null)
        continue;
      int dt = nextTurn - e.latest.turn();
      if (dt < 1)
        continue;
      // First translation is on the observed heading. Unknown speed control spans
      // the legal velocity image, including the discontinuous instantaneous STOP.
      var b = e.latest.body();
      double lo = Double.POSITIVE_INFINITY, hi = Double.NEGATIVE_INFINITY;
      for (double target : new double[] {-8, -.000001, 0, .000001, 8}) {
        double v = Physics.velocity(b.speed(), target);
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
      var a = b.p().add(Physics.vector(b.heading()).mul(lo));
      var z = b.p().add(Physics.vector(b.heading()).mul(hi));
      double gap = Geometry.segmentDistance(point, a, z) - 8 * Math.max(0, dt - 1);
      if (gap < 36)
        risk += .6;
    }
    return risk;
  }
  Choice emergency(Physics.Body self, Physics.Rules r) {
    emergencies++;
    Choice best = null;
    boolean bestSafe = false;
    int now = safetyTurn >= 0 ? safetyTurn
                              : activeEnemies.stream()
                                    .filter(e -> e.latest != null)
                                    .mapToInt(e -> e.latest.turn())
                                    .max()
                                    .orElse(0);
    for (double speed : new double[] {0, -8, 8})
      for (double turn : new double[] {0, -Math.PI / 2, Math.PI / 2}) {
        Physics.Control c = new Physics.Control(speed, turn, 0, 0);
        Physics.Step s = Physics.step(self, c, r);
        double botDamage = collisionRisk(s.body().p(), now + 1, r), cost = s.damage() + botDamage;
        boolean safe = r.safe(s.body().p()) && botDamage == 0;
        if (best == null || safe && !bestSafe || safe == bestSafe && cost < best.cost) {
          best = new Choice(c, previous, cost, true, true, List.of(s.body()));
          bestSafe = safe;
        }
      }
    return best;
  }
}
