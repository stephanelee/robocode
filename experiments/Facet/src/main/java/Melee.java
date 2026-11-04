import java.util.*;
/** Implemented for isolated tests only. Compile-time gate cannot be bypassed by environment. */
final class Melee {
  static final boolean REVIEW_APPROVED = false;
  final SplittableRandom random;
  final ArrayDeque<Physics.V> recent = new ArrayDeque<>();
  Physics.V retained;
  long unsafeFallbacks;
  Melee(long seed) {
    random = new SplittableRandom(seed);
  }
  static void requireApproval() {
    if (!REVIEW_APPROVED)
      throw new IllegalStateException("Melee gated pending duel semantics and independent review");
  }
  Physics.Control move(Physics.Body self, double energy, int turn,
      Collection<Learner.Enemy> enemies, Physics.Rules r, Planner emergency) {
    if (turn % 8 == 0) {
      recent.addLast(self.p());
      while (recent.size() > 32) recent.removeFirst();
    }
    double nearest = enemies.stream()
                         .filter(e -> e.alive && e.latest != null)
                         .mapToDouble(e -> self.p().distance(e.latest.body().p()))
                         .min()
                         .orElse(180);
    double radius = Math.min(100 + 80 * random.nextDouble(), Math.max(20, nearest));
    LinkedHashSet<Physics.V> destinations = new LinkedHashSet<>();
    for (int i = 0; i < 48; i++)
      destinations.add(r.bound(self.p().add(Physics.vector(i * Physics.TAU / 48).mul(radius)), 19));
    if (retained != null)
      destinations.add(retained);
    destinations.add(r.bound(new Physics.V(r.width() / 2, r.height() / 2), 19));
    List<Physics.V> sorted = new ArrayList<>(destinations);
    sorted.sort(Comparator.comparingDouble(p -> risk(p, self, energy, turn, enemies)));
    for (Physics.V p : sorted) {
      Physics.Body s = self;
      boolean safe = true;
      for (int i = 0; i < 8; i++) {
        if (Physics.workRemaining() <= 50) {
          unsafeFallbacks++;
          return emergency.emergency(self, r).command();
        }
        Physics.Step next = Physics.step(s, Physics.toward(s, p, false), r);
        if (!r.safe(next.body().p()) || next.wall()) {
          safe = false;
          break;
        }
        for (Learner.Enemy e : enemies)
          if (e.alive && e.latest != null && turn - e.latest.turn() <= 8
              && next.body().p().distance(e.latest.body().p())
                  < 36 + Math.min(8, turn - e.latest.turn() + i)) {
            safe = false;
            break;
          }
        if (!safe)
          break;
        s = next.body();
      }
      if (safe) {
        retained = p;
        return Physics.toward(self, p, false);
      }
    }
    unsafeFallbacks++;
    return emergency.emergency(self, r).command();
  }
  double risk(
      Physics.V p, Physics.Body self, double energy, int turn, Collection<Learner.Enemy> enemies) {
    double risk = 0, travel = self.p().bearing(p);
    for (Learner.Enemy e : enemies) {
      if (!e.alive || e.latest == null)
        continue;
      int age = turn - e.latest.turn();
      double closer = 0;
      if (age <= 8)
        for (Learner.Enemy other : enemies)
          if (other != e && other.alive && other.latest != null && turn - other.latest.turn() <= 8
              && e.latest.body().p().distance(other.latest.body().p())
                  < e.latest.body().p().distance(p))
            closer += Math.max(0, 1 - (turn - other.latest.turn()) / 9.0);
      double threat = (e.damageTaken + 5) / Math.max(1, e.observedTogether),
             ratio = Physics.clamp(e.latest.energy() / Math.max(energy, .1), .25, 4),
             align = 1 + Math.abs(Math.cos(e.latest.body().p().bearing(p) - travel));
      risk += threat * ratio * align
          / ((Math.pow(p.distance(e.latest.body().p()), 2) + 36 * 36) * (1 + closer));
    }
    double revisit = 1;
    for (Physics.V old : recent) revisit += 100 / (Math.pow(p.distance(old), 2) + 100);
    return risk * revisit * (p.equals(retained) ? .85 : 1);
  }
  static Optional<Physics.V> project(Memory.Observation label, Physics.Body target, int sign,
      Physics.V origin, double power, Physics.Rules r) {
    if (label.type() != Memory.Label.DISPLACEMENT)
      throw new IllegalArgumentException("typed displacement");
    double heading = Physics.effective(target), x = label.a(), y = label.b() * sign;
    Physics.V velocity = new Physics.V(x * Math.cos(heading) - y * Math.sin(heading),
        x * Math.sin(heading) + y * Math.cos(heading));
    double time = origin.distance(target.p()) / Physics.bulletSpeed(power), previous = -1;
    Physics.V p = target.p();
    for (int i = 0; i < 12; i++) {
      p = target.p().add(velocity.mul(time));
      if (!r.safe(p))
        return Optional.empty();
      double next = origin.distance(p) / Physics.bulletSpeed(power);
      if (Math.abs(next - time) < 1e-6)
        return Optional.of(p);
      if (Math.abs(next - previous) < 1e-6) {
        time = Math.max(next, time);
        p = target.p().add(velocity.mul(time));
        return r.safe(p) ? Optional.of(p) : Optional.empty();
      }
      previous = time;
      time = next;
    }
    return r.safe(p) ? Optional.of(p) : Optional.empty();
  }
  static double aim(
      Physics.Body self, List<Learner.Enemy> targets, double power, int turn, Physics.Rules r) {
    List<Learner.Enemy> fresh =
        targets.stream()
            .filter(e -> e.alive && e.latest != null && turn - e.latest.turn() <= 8)
            .sorted(Comparator.comparingInt(e -> e.id))
            .toList();
    if (fresh.isEmpty())
      return self.gun();
    record Projection(double angle, double range, double weight) {}
    List<Projection> projections = new ArrayList<>();
    int per = Math.max(1, 80 / Math.max(1, fresh.size())), remaining = 80;
    for (Learner.Enemy e : fresh) {
      if (remaining == 0)
        break;
      int k = Math.min(remaining, per);
      remaining -= k;
      Learner.Context c = Learner.context(
          self.p(), e.latest.body(), e, turn, power, r, fresh.size(), fresh.size(), false);
      var neighbors = e.displacement.query(c.features(), Learner.DISPLACEMENT_WEIGHTS, k);
      List<Projection> local = new ArrayList<>();
      double total = 0;
      for (var n : neighbors) {
        var point = project(n.observation(), e.latest.body(), e.sign, self.p(), power, r);
        if (point.isPresent()) {
          double w = n.weight();
          local.add(
              new Projection(self.p().bearing(point.get()), self.p().distance(point.get()), w));
          total += w;
        }
      }
      for (Projection p : local)
        projections.add(new Projection(p.angle, p.range, p.weight / Math.max(1e-9, total)));
    }
    if (projections.isEmpty()) {
      Learner.Enemy e =
          fresh.stream()
              .min(Comparator.comparingDouble(x -> self.p().distance(x.latest.body().p())))
              .orElseThrow();
      return Physics.intercept(self.p(), e.latest.body(), power, r);
    }
    double best = -1, angle = self.gun();
    for (Projection p : projections) {
      double score = 0;
      for (Projection q : projections)
        score += q.weight
            * Geometry.gaussian(Physics.wrap(p.angle - q.angle),
                Math.max(1e-4, 2 * Math.asin(Math.min(1, 18 / Math.max(18, q.range)))))
            / Math.max(1, q.range);
      if (score > best + 1e-12
          || (Math.abs(score - best) <= 1e-12
              && Math.abs(Physics.wrap(p.angle - self.gun()))
                  < Math.abs(Physics.wrap(angle - self.gun())))) {
        best = score;
        angle = p.angle;
      }
    }
    return angle;
  }
}
