import java.util.*;
/** Independent initial melee profile; activation is pending independent melee review. */
final class Melee {
  static final boolean REVIEW_APPROVED = true; // Duel prerequisite passed; not full melee approval.
  final SplittableRandom random;
  final ArrayDeque<Physics.V> recent = new ArrayDeque<>();
  Physics.V retained;
  long unsafeFallbacks;
  long dispatches, quotaFallbacks, unsafeEmergency, candidateChecks, wallRejects, botRejects;
  int selectedSteps;
  String moveContext = "cold";
  int destinations;
  static final class ProjectionWork {
    long samples, iterations, queries, returned;
  }
  final ProjectionWork projectionWork = new ProjectionWork();
  Melee(long seed) {
    random = new SplittableRandom(seed);
  }
  static void requireApproval() {
    if (!REVIEW_APPROVED)
      throw new IllegalStateException("Melee gated pending duel semantics and independent review");
  }
  Physics.Control move(Physics.Body self, double energy, int turn,
      Collection<Learner.Enemy> enemies, Physics.Rules r, Planner emergency) {
    emergency.activeEnemies = enemies;
    dispatches++;
    selectedSteps = 0;
    emergency.safetyTurn = turn;
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
    this.destinations = sorted.size();
    sorted.sort(Comparator.comparingDouble(p -> risk(p, self, energy, turn, enemies)));
    for (Physics.V p : sorted) {
      candidateChecks++;
      Physics.Body s = self;
      boolean safe = true;
      for (int i = 0; i < 8; i++) {
        if (Physics.workRemaining() <= 350) {
          unsafeFallbacks++;
          quotaFallbacks++;
          moveContext = "quota remaining=" + Physics.workRemaining() + " candidate=" + p;
          return emergencyControl(self, r, emergency);
        }
        Physics.Step next = Physics.step(s, Physics.toward(s, p, false), r);
        if (!r.safe(next.body().p()) || next.wall()) {
          wallRejects++;
          safe = false;
          break;
        }
        if (emergency.collisionRisk(next.body().p(), turn + i + 1, r) > 0) {
          botRejects++;
          safe = false;
        }
        if (!safe)
          break;
        s = next.body();
      }
      if (safe) {
        retained = p;
        selectedSteps = 8;
        moveContext = "certified-eight-step destination=" + p;
        return Physics.toward(self, p, false);
      }
    }
    unsafeFallbacks++;
    unsafeEmergency++;
    moveContext = "all-unsafe safest-imminent-emergency";
    return emergencyControl(self, r, emergency);
  }
  Physics.Control emergencyControl(Physics.Body self, Physics.Rules r, Planner planner) {
    var choice = planner.emergency(self, r);
    moveContext += " cost=" + choice.cost() + " control=" + choice.command();
    return choice.command();
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
    return project(label, target, sign, origin, power, r, null);
  }
  static Optional<Physics.V> project(Memory.Observation label, Physics.Body target, int sign,
      Physics.V origin, double power, Physics.Rules r, ProjectionWork work) {
    var total = Memory.displacementWork.get();
    total.projections++;
    if (work != null)
      work.samples++;
    if (label.type() != Memory.Label.DISPLACEMENT)
      throw new IllegalArgumentException("typed displacement");
    double heading = Physics.effective(target), x = label.a(), y = label.b() * sign;
    Physics.V velocity = new Physics.V(x * Math.cos(heading) - y * Math.sin(heading),
        x * Math.sin(heading) + y * Math.cos(heading));
    double time = origin.distance(target.p()) / Physics.bulletSpeed(power), previous = -1;
    Physics.V p = target.p();
    for (int i = 0; i < 12; i++) {
      total.iterations++;
      if (work != null)
        work.iterations++;
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
  record Aim(double angle, int target, Physics.V projection, double power,
      FireControl.Policy policy, int computedTurn, boolean displacement) {}
  Aim cached;
  String cacheKey;
  int lastProjectionTurn = -100;
  long projectionPasses;
  long decisionSamples, decisionIterations, decisionQueries, decisionReturned, decisionPasses;
  void beginDecision() {
    decisionSamples = decisionIterations = decisionQueries = decisionReturned = decisionPasses = 0;
  }
  void resetAim() {
    cached = null;
    cacheKey = null;
    lastProjectionTurn = -100;
  }
  static String key(Physics.Body self, List<Learner.Enemy> targets, double energy, int turn) {
    StringBuilder k = new StringBuilder().append(energy).append(':').append(self.p());
    for (var e : targets)
      k.append('|')
          .append(e.id)
          .append(':')
          .append(e.alive)
          .append(':')
          .append(e.latest)
          .append(':')
          .append(e.displacement.version)
          .append(':')
          .append(e.latest != null && turn - e.latest.turn() <= 2);
    return k.toString();
  }
  Aim scheduledAim(Physics.Body self, List<Learner.Enemy> targets, double energy, int turn,
      int population, Physics.Rules r) {
    beginDecision();
    String k = key(self, targets, energy, turn);
    boolean changed = !k.equals(cacheKey);
    boolean due = self.heat() / r.cooling() <= 4 + 1e-10 || turn < lastProjectionTurn
        || turn - lastProjectionTurn >= 2 || changed && turn == lastProjectionTurn;
    if (due) {
      cached = select(self, targets, energy, turn, population, r);
      cacheKey = k;
      lastProjectionTurn = turn;
    }
    // Invalidated context may track analytically on the intervening hot turn; never reuse its plan.
    return !due && changed ? null : cached;
  }
  Aim select(Physics.Body self, List<Learner.Enemy> targets, double energy, int turn,
      int population, Physics.Rules r) {
    beginDecision();
    var fresh = targets.stream()
                    .filter(e
                        -> e.alive && e.latest != null && e.latest.turn() <= turn
                            && turn - e.latest.turn() <= 2)
                    .toList();
    if (fresh.isEmpty())
      return null;
    var initial = fresh.stream()
                      .min(Comparator.comparingDouble(e -> self.p().distance(e.latest.body().p())))
                      .orElseThrow();
    double power = Energy.power(
        self.p().distance(initial.latest.body().p()), energy, initial.latest.energy(), true);
    long sb = projectionWork.samples, ib = projectionWork.iterations;
    long qb = projectionWork.queries, rb = projectionWork.returned;
    try {
      for (int stage = 0; stage < 3 && power > 0; stage++) {
        // Reserve later stages for queries at the final candidate power/context.
        int quota = stage == 0 ? 26 : 27;
        if (fresh.size() > quota)
          return null;
        projectionPasses++;
        decisionPasses++;
        Aim a = aim(self, fresh, power, turn, r, projectionWork, quota);
        var e = fresh.stream().filter(x -> x.id == a.target()).findFirst().orElseThrow();
        double range = self.p().distance(a.projection());
        double next = Energy.power(range, energy, e.latest.energy(), true);
        if (Math.abs(next - power) < 1e-12)
          return new Aim(a.angle(), e.id, a.projection(), power,
              new FireControl.Policy(population, e.id, range, e.latest.energy(), e.latest.turn()),
              turn, a.displacement());
        power = next;
      }
      // A cycling policy is not a coherent firing solution. Track directly, without a shot plan.
      return null;
    } finally {
      decisionSamples = projectionWork.samples - sb;
      decisionIterations = projectionWork.iterations - ib;
      decisionQueries = projectionWork.queries - qb;
      decisionReturned = projectionWork.returned - rb;
    }
  }
  static Aim aim(
      Physics.Body self, List<Learner.Enemy> targets, double power, int turn, Physics.Rules r) {
    return aim(self, targets, power, turn, r, null);
  }
  static Aim aim(Physics.Body self, List<Learner.Enemy> targets, double power, int turn,
      Physics.Rules r, ProjectionWork work) {
    return aim(self, targets, power, turn, r, work, 80);
  }
  static Aim aim(Physics.Body self, List<Learner.Enemy> targets, double power, int turn,
      Physics.Rules r, ProjectionWork work, int quota) {
    List<Learner.Enemy> fresh = targets.stream()
                                    .filter(e
                                        -> e.alive && e.latest != null && e.latest.turn() <= turn
                                            && turn - e.latest.turn() <= 2)
                                    .sorted(Comparator.comparingInt(e -> e.id))
                                    .toList();
    if (fresh.isEmpty())
      return null;
    record Projection(int target, Physics.V point, double angle, double range, double weight) {}
    List<Projection> projections = new ArrayList<>();
    int per = Math.max(1, quota / Math.max(1, fresh.size())), remaining = quota;
    for (Learner.Enemy e : fresh) {
      if (remaining == 0)
        break;
      int k = Math.min(remaining, per);
      remaining -= k;
      Physics.Body observed = e.latest.body();
      Physics.Body projected =
          new Physics.Body(Physics.constantPosition(observed, turn - e.latest.turn(), r),
              observed.heading(), observed.speed(), 0, 0, 0);
      Learner.Context c = Learner.context(
          self.p(), projected, e, turn, power, r, fresh.size(), fresh.size(), false);
      if (work != null)
        work.queries++;
      var neighbors = e.displacement.query(c.features(), Learner.DISPLACEMENT_WEIGHTS, k);
      if (work != null)
        work.returned += neighbors.size();
      List<Projection> local = new ArrayList<>();
      double total = 0;
      for (var n : neighbors) {
        var point = project(n.observation(), projected, e.sign, self.p(), power, r, work);
        if (point.isPresent()) {
          double w = n.weight();
          local.add(new Projection(
              e.id, point.get(), self.p().bearing(point.get()), self.p().distance(point.get()), w));
          total += w;
        }
      }
      for (Projection p : local)
        projections.add(
            new Projection(p.target, p.point, p.angle, p.range, p.weight / Math.max(1e-9, total)));
    }
    if (projections.isEmpty()) {
      Learner.Enemy e =
          fresh.stream()
              .min(Comparator.comparingDouble(x -> self.p().distance(x.latest.body().p())))
              .orElseThrow();
      var observed = e.latest.body();
      var current = new Physics.Body(Physics.constantPosition(observed, turn - e.latest.turn(), r),
          observed.heading(), observed.speed(), 0, 0, 0);
      Physics.V point = Physics.interceptPoint(self.p(), current, power, r);
      return new Aim(self.p().bearing(point), e.id, point, power, null, turn, false);
    }
    double best = -1, angle = self.gun();
    Projection selected = null;
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
        selected = p;
      }
    }
    return new Aim(angle, selected.target, selected.point, power, null, turn, true);
  }
}
