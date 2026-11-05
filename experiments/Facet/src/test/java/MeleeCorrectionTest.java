import java.util.*;
/** Independent deterministic fixtures, not battle or strength evidence. */
final class MeleeCorrectionTest {
  public static void main(String[] args) {
    quota();
    schedule();
    failures();
    System.out.println("PASS focused global melee quota, scheduling and disable regression");
  }
  static final Physics.Rules R = new Physics.Rules(800, 600, .1);
  static void ok(boolean v, String s) {
    FacetSemanticTest.ok(v, s);
  }
  static void eq(double a, double b, String s) {
    FacetSemanticTest.eq(a, b, 1e-8, s);
  }
  static Physics.Body b(double x, double y, double h, double v) {
    return FacetSemanticTest.body(x, y, h, v);
  }
  static Learner.Enemy enemy(int id, int turn, Physics.Body body, double energy) {
    var e = new Learner.Enemy(id, new Memory.Budget());
    e.observe(new Memory.Sample(1, turn, body, energy, Memory.Quality.OBSERVED, 0),
        new Physics.V(400, 300));
    return e;
  }
  static Memory.Observation displacement(double x, double y, long seq) {
    return new Memory.Observation(
        new double[Learner.DIM], Memory.Label.DISPLACEMENT, x, y, seq, 1, 0, 1, seq, true);
  }
  static void run() {
    var own = b(400, 300, 0, 0);
    var target = enemy(1, 0, b(700, 300, 0, 0), 100);
    var f = new FireControl();
    var policy = new FireControl.Policy(3, 1, 300, 100, 0);
    f.plan = FireControl.prepare(0, 2.5, target, own, 0, 1, policy);
    ok(f.valid(own, 100, target, 0, R), "ordinary melee 2.5 passes submission");
    ok(!f.valid(own, 2.6, target, 0, R), "reserve retained in melee");
    target.latest = new Memory.Sample(1, 0, target.latest.body(), 1, Memory.Quality.OBSERVED, 0);
    ok(!f.valid(own, 100, target, 0, R), "updated melee kill cap rejects");
    target.latest = new Memory.Sample(1, 0, target.latest.body(), 100, Memory.Quality.OBSERVED, 0);
    f.submitted(true, own, 0, 7);
    var pending = f.pending;
    f.plan = FireControl.prepare(0, 1, target, own, 1, 8, policy);
    ok(f.pending == pending && pending.power() == 2.5 && pending.policy() == policy,
        "pending immutable policy across preparation");
    f.reset();
    f.plan = FireControl.prepare(0, 2.5, target, own, 0, 1);
    ok(!f.valid(own, 100, target, 0, R), "duel cap still rejects ordinary 2.5");

    var near = enemy(2, 0, b(500, 300, 0, 0), 100);
    var far = enemy(3, 0, b(750, 400, Math.PI, 0), 4);
    far.displacement.add(displacement(0, 0, 1));
    var m = new Melee(5);
    var selected = m.select(own, List.of(near, far), 100, 0, 2, R);
    ok(selected != null && selected.target() == 3,
        "projected primary differs from nearest cold target");
    eq(selected.power(), 1, "selected low-energy target owns kill cap");
    eq(selected.policy().range(), Math.hypot(350, 100), "selected target owns projected range");
    eq(selected.angle(), Math.atan2(100, 350), "projection identity retained with angle");
    var frozen =
        FireControl.prepare(selected.angle(), selected.power(), far, own, 0, 1, selected.policy());
    ok(frozen.target() == 3 && frozen.policy().targetEnergy() == 4,
        "plan freezes selected target policy");
    far.alive = false;
    ok(m.select(own, List.of(near, far), 100, 0, 2, R).target() == 2, "death excludes target");
    var distant = enemy(5, 0, b(750, 500, 0, 0), 100);
    distant.displacement.add(displacement(0, 0, 3));
    var rangeSelected = m.select(
        b(100, 100, 0, 0), List.of(enemy(6, 0, b(200, 100, 0, 0), 100), distant), 100, 0, 2, R);
    ok(rangeSelected.target() == 5 && rangeSelected.policy().range() > 500,
        "selected projection owns far-range band rather than nearest primary");
    eq(rangeSelected.power(), 1.5, "far-range power re-evaluated coherently");

    // Every candidate and emergency uses the same signed-velocity envelope, including stale scans.
    for (int sign : new int[] {-1, 1})
      for (int age : new int[] {0, 3, 12}) {
        var approaching = enemy(4, 10 - age, b(442, 300, sign > 0 ? Math.PI : 0, sign * 8), 100);
        var p = new Planner();
        p.activeEnemies = List.of(approaching);
        p.safetyTurn = 10;
        ok(p.collisionRisk(new Physics.V(408, 300), 11, R) > 0,
            "approaching signed/stale envelope");
        Physics.beginWork(2400);
        var command = m.move(b(400, 300, 0, 8), 100, 10, List.of(approaching), R, p);
        var next = Physics.step(b(400, 300, 0, 8), command, R);
        var emergency = p.emergency(b(400, 300, 0, 8), R);
        ok(p.collisionRisk(next.body().p(), 11, R) == 0 || emergency.cost() > 0,
            "unsafe first step only when emergency also unsafe");
        ok(Physics.workUsed() <= 2400, "melee shared quota");
        Physics.endWork();
      }
    var surround = new ArrayList<Learner.Enemy>();
    for (int i = 0; i < 8; i++)
      surround.add(enemy(i + 10, 0,
          b(400 + 30 * Math.cos(i * Math.PI / 4), 300 + 30 * Math.sin(i * Math.PI / 4), 0, 0),
          100));
    var p = new Planner();
    p.activeEnemies = surround;
    p.safetyTurn = 0;
    Physics.beginWork(360);
    var command = m.move(own, 100, 0, surround, R, p);
    ok(m.unsafeFallbacks > 0, "all unsafe/quota fallback recorded");
    var safest = p.emergency(own, R).command();
    ok(command.equals(safest), "fallback compares opponent-aware emergency controls");
    Physics.endWork();
    var allUnsafe = new Melee(5);
    Physics.beginWork(2400);
    var unsafeCommand = allUnsafe.move(own, 100, 0, surround, R, p);
    ok(allUnsafe.unsafeEmergency == 1 && allUnsafe.quotaFallbacks == 0,
        "all unsafe distinct from quota");
    ok(unsafeCommand.equals(p.emergency(own, R).command()),
        "all unsafe returns explicit safest emergency");
    ok(allUnsafe.moveContext.contains("all-unsafe") && allUnsafe.moveContext.contains("cost="),
        "all unsafe evidence includes emergency cost");
    Physics.endWork();
    var quota = new Melee(6);
    var safePlanner = new Planner();
    Physics.beginWork(350);
    var quotaCommand = quota.move(own, 100, 0, List.of(), R, safePlanner);
    ok(quota.quotaFallbacks == 1 && quota.unsafeEmergency == 0,
        "quota reason distinct from all unsafe");
    ok(!Physics.step(own, quotaCommand, R).wall()
            && safePlanner.collisionRisk(Physics.step(own, quotaCommand, R).body().p(), 1, R) == 0,
        "quota fallback safe when safety possible");
    ok(Physics.workUsed() <= 350, "quota emergency work included");
    Physics.endWork();
    var unknown = new Learner.Enemy(91, new Memory.Budget());
    Physics.beginWork(2400);
    var unknownCommand = new Melee(1).move(own, 100, 0, List.of(unknown), R, new Planner());
    ok(!Physics.step(own, unknownCommand, R).wall(),
        "unknown target no invented position bounded movement");
    Physics.endWork();
    var stopped = enemy(92, 0, b(440, 300, 0, 0), 100);
    var stoppedPlanner = new Planner();
    stoppedPlanner.activeEnemies = List.of(stopped);
    stoppedPlanner.safetyTurn = 0;
    Physics.beginWork(2400);
    var stoppedCommand =
        new Melee(1).move(b(400, 300, 0, 8), 100, 0, List.of(stopped), R, stoppedPlanner);
    var stoppedNext = Physics.step(b(400, 300, 0, 8), stoppedCommand, R);
    ok(stoppedPlanner.collisionRisk(stoppedNext.body().p(), 1, R) == 0
            || stoppedPlanner.emergency(b(400, 300, 0, 8), R).cost() > 0,
        "stopped opponent safest available first step");
    Physics.endWork();
    for (var corner : List.of(b(781, 580, Math.PI / 4, 8), b(19, 20, Math.PI / 4, -8))) {
      Physics.beginWork(2400);
      var c = m.move(corner, 100, 0, List.of(), R, new Planner());
      ok(!Physics.step(corner, c, R).wall(), "corner first translation safe");
      Physics.endWork();
    }
    var wallMelee = new Melee(7);
    var start = b(740, 540, 0, 8);
    wallMelee.move(start, 100, 0, List.of(), R, new Planner());
    ok(wallMelee.retained != null, "safe candidate retains destination");
    for (int t = 0; t < 8; t++) {
      var step = Physics.step(start, Physics.toward(start, wallMelee.retained, false), R);
      ok(!step.wall() && R.safe(step.body().p()), "full eight-step wall horizon");
      start = step.body();
    }
    quota();
    schedule();
    survivor();
    active();
    normalization();
    boundedActive();
    failures();
  }
  static void quota() {
    var self = b(100, 300, 0, 0);
    var e = enemy(1, 40, b(500, 300, 0, 0), 100);
    for (int i = 0; i < 80; i++) e.displacement.add(displacement(4, 0, i));
    var m = new Melee(49667);
    var a = m.scheduledAim(self, List.of(e), 100, 40, 2, R);
    ok(a != null && a.target() == 1 && a.policy().target() == 1,
        "review reproduction final primary");
    eq(a.power(), 1.5, "review reproduction final legal power");
    eq(a.policy().range(), 539.1304226442027, "review reproduction final power projection");
    ok(m.decisionSamples == 53 && m.decisionReturned == 53 && m.decisionQueries == 2
            && m.decisionPasses == 2,
        "actual 26+27 samples across power change");
    var c = new Core(R, new Diagnostics(), 1);
    c.roundReset(1);
    c.enemies.put(1, e);
    c.decide(self, 100, 40, 2, List.of(), List.of(), 99);
    ok(c.fire.plan != null && c.fire.plan.target() == 1 && c.fire.plan.power() == 1.5,
        "active Core reviewer power-change plan");
    ok(c.melee.decisionSamples <= 80 && c.melee.decisionReturned == c.melee.decisionSamples,
        "active Core actual global budget");
    ok(c.log.counters.get("melee.projectionSamples") == c.melee.decisionSamples
            && c.log.observations.get("meleeDecisionSamples")[3] <= 80,
        "active quota telemetry actual counters");
    long cumulative = m.projectionWork.samples;
    for (int t = 41; t <= 44; t++) {
      e.latest = new Memory.Sample(1, t, e.latest.body(), 100, Memory.Quality.OBSERVED, 0);
      m.scheduledAim(self, List.of(e), 100, t, 2, R);
      ok(m.decisionSamples <= 80 && m.decisionIterations <= 12 * m.decisionSamples
              && m.decisionPasses <= 3,
          "multiple decisions bounded work");
      cumulative += m.decisionSamples;
      ok(m.projectionWork.samples == cumulative, "cumulative samples equal actual deltas");
    }
    var hot = new Physics.Body(self.p(), 0, 0, 0, 0, 1);
    m.resetAim();
    m.scheduledAim(hot, List.of(e), 100, 44, 2, R);
    m.scheduledAim(hot, List.of(e), 100, 45, 2, R);
    ok(m.decisionSamples == 0 && m.decisionQueries == 0 && m.decisionPasses == 0,
        "cached hot decision zero work");
    var cycle = enemy(2, 0, b(500, 300, 0, 0), 100);
    for (int i = 0; i < 80; i++) cycle.displacement.add(displacement(3, 0, i));
    ok(m.select(self, List.of(cycle), 100, 0, 2, R) == null,
        "cycling power conservatively no shot");
    ok(m.decisionSamples == 80 && m.decisionReturned == 80 && m.decisionPasses == 3
            && m.decisionIterations <= 960,
        "cycle actual total80 finite solve");
    var stable = enemy(3, 0, b(300, 300, 0, 0), 100);
    for (int i = 0; i < 80; i++) stable.displacement.add(displacement(0, 0, i));
    ok(m.select(self, List.of(stable), 100, 0, 2, R) != null && m.decisionSamples == 26
            && m.decisionPasses == 1,
        "stable first stage actual quota");
    var empty = enemy(4, 0, b(350, 300, 0, 0), 100);
    var fallback = m.select(self, List.of(empty), 100, 0, 2, R);
    ok(fallback != null && !fallback.displacement() && m.decisionSamples == 0
            && m.decisionReturned == 0 && m.decisionQueries == 1,
        "missing store analytic fallback actual counts");
    m.select(self, List.of(stable, cycle, empty), 100, 0, 3, R);
    ok(m.decisionSamples <= 80 && m.decisionReturned == m.decisionSamples && m.decisionQueries <= 9,
        "multi-target global actual sample bound");
    var crowded = new ArrayList<Learner.Enemy>();
    for (int i = 0; i < 27; i++) crowded.add(enemy(100 + i, 0, b(300, 300, 0, 0), 100));
    ok(m.select(self, crowded, 100, 0, 27, R) == null && m.decisionSamples == 0,
        "unsupported equal split conservatively no shot");
    System.out.println("quota reproduction: power=" + a.power() + " range=" + a.policy().range()
        + " actualSamples=53 stages=2; cycleSamples=80 stages=3");
  }
  static byte[] model(Core c) {
    try {
      var out = new java.io.ByteArrayOutputStream();
      ModelSnapshot.write(c, out);
      return java.security.MessageDigest.getInstance("SHA-256").digest(out.toByteArray());
    } catch (Exception ex) {
      throw new AssertionError(ex);
    }
  }
  static void normalization() {
    var c = new Core(R, new Diagnostics(), 1);
    c.roundReset(1);
    var e = c.enemy(1);
    var self = b(400, 300, 0, 0);
    for (int i = 0; i < 5; i++) e.observe(FacetSemanticTest.sample(0, b(600, 300, 0, 0)), self.p());
    ok(e.observedTogether == 0, "scan callbacks do not count joint decisions");
    byte[] before = model(c);
    e.observedDecision(0);
    e.observedDecision(0);
    ok(e.observedTogether == 1, "one count per own turn");
    ok(!Arrays.equals(before, model(c)), "export hash includes normalizer");
    ok(Arrays.equals(model(c), model(c)), "export hash deterministic");
    e.observedDecision(2);
    e.observedDecision(8);
    e.observedDecision(9);
    ok(e.observedTogether == 3, "sparse decisions count actual fresh turns only");
    e.observe(FacetSemanticTest.sample(20, b(600, 300, 0, 0)), self.p());
    e.observedDecision(19);
    ok(e.observedTogether == 3, "future observation cannot count");
    e.alive = false;
    e.observedDecision(20);
    ok(e.observedTogether == 3, "death cannot count");
    e.reset();
    e.observe(FacetSemanticTest.sample(0, b(600, 300, 0, 0)), self.p());
    e.observedDecision(0);
    ok(e.observedTogether == 4, "round resets bookkeeping and retains game count");
    var cold = new Core(R, new Diagnostics(), 1);
    ok(cold.enemy(1).observedTogether == 0, "cold game denominator");
    c.roundReset(2);
    var scan =
        new dev.robocode.tankroyale.botapi.events.ScannedBotEvent(0, 99, 1, 100, 600, 300, 0, 0);
    c.decide(self, 100, 0, 2, List.of(scan, scan), List.of(), 99);
    long n = e.observedTogether;
    c.observe(self, 100, 0, 2, List.of(scan), List.of(), 99, true);
    Physics.endWork();
    ok(e.observedTogether == n, "terminal replay no increments");
    ok(c.log.counters.getOrDefault("learning.terminalNoLearning", 0L) == 1,
        "terminal label distinct");
    c.decide(self, 100, 5, 2, List.of(), List.of(), 99);
    ok(e.observedTogether == n + 1, "turn gap never synthesizes decisions");
    c.decide(self, 100, 6, 2,
        List.of(new dev.robocode.tankroyale.botapi.events.ScannedBotEvent(
            30, 99, 1, 100, 610, 300, 0, 0)),
        List.of(), 99);
    ok(e.latest.turn() == 0, "future scan excluded from native history");
    double risk = new Melee(1).risk(new Physics.V(400, 400), self, 100, 6, List.of(e));
    double expected =
        (e.damageTaken + 5) / e.observedTogether * (1 + 1 / Math.sqrt(5)) / (50000 + 36 * 36);
    eq(risk, expected, "native enemy-to-destination alignment convention");
  }
  static void boundedActive() {
    var c = new Core(R, new Diagnostics(), 1);
    c.roundReset(1);
    for (int id = 1; id <= 4; id++) {
      var e = c.enemy(id);
      e.observe(FacetSemanticTest.sample(0, b(500 + id * 40, 100 + id * 70, 0, 0)),
          new Physics.V(400, 300));
      for (var s : List.of(e.displacement, e.recentDisplacement[0], e.recentDisplacement[1],
               e.recentDisplacement[2]))
        for (int j = 0; j < s.capacity; j++)
          s.add(displacement(0, (j % 3 - 1) * .1, id * 20000L + j));
    }
    var self = b(400, 300, 0, 0);
    for (int j = 0; j < 8; j++)
      c.waves.add(FacetSemanticTest.wave(j + 1, 1, -8, new Physics.V(100, 100), self, 2));
    c.planner.steps = 321;
    c.decide(self, 100, 0, 4, List.of(), List.of(), 99);
    ok(c.planner.steps == 321, "filled active melee excludes duel planner");
    ok(c.log.observations.get("predictorSteps")[3] <= 2400,
        "filled multi-target waves shared quota");
    ok(c.log.counters.getOrDefault("movement.failure", 0L) == 0
            && c.log.counters.getOrDefault("gun.failure", 0L) == 0,
        "filled active path no subsystem boundary failures");
    ok(c.melee.selectedSteps == 8 || c.melee.quotaFallbacks + c.melee.unsafeEmergency > 0,
        "active eight steps or explicit emergency");
    ok(c.melee.destinations >= 49 && c.melee.destinations <= 50,
        "deduplicated centered active destinations");
    ok(c.melee.projectionWork.samples > 0 && c.melee.projectionWork.samples <= 80,
        "active filled projection sample bound");
    ok(c.melee.projectionWork.iterations <= 12 * c.melee.projectionWork.samples,
        "active displacement iteration bound");
    if (c.melee.selectedSteps == 8) {
      var state = self;
      for (int step = 1; step <= 8; step++) {
        var next = Physics.step(state, Physics.toward(state, c.melee.retained, false), R);
        ok(!next.wall() && R.safe(next.body().p()), "filled active selected eight-step walls");
        ok(c.planner.collisionRisk(next.body().p(), step, R) == 0,
            "filled active selected eight-step bot envelopes");
        state = next.body();
      }
    }
    System.out.println(
        "offline active melee: targets=4 waves=8 filledDisplacementStores=16 payloadBytes="
        + c.budget.bytes + " predictorSteps=" + c.log.observations.get("predictorSteps")[3]
        + " projectionPasses=" + c.melee.projectionPasses + " samples="
        + c.melee.projectionWork.samples + " solveIterations=" + c.melee.projectionWork.iterations
        + " selectedSteps=" + c.melee.selectedSteps + " quotaFallbacks=" + c.melee.quotaFallbacks
        + " allUnsafe=" + c.melee.unsafeEmergency);
    var survivor = c.enemy(1);
    long version = survivor.displacement.version;
    int count = survivor.displacement.size();
    for (int id = 2; id <= 4; id++) c.enemy(id).alive = false;
    c.decide(self, 100, 1, 1, List.of(), List.of(), 99);
    ok(c.fire.plan == null, "transition requires a new survivor scan");
    survivor.crossings = 16;
    survivor.selected = "recent";
    c.decide(self, 100, 2, 1,
        List.of(new dev.robocode.tankroyale.botapi.events.ScannedBotEvent(
            2, 99, 1, 100, 540, 170, 0, 0)),
        List.of(), 99);
    ok(c.fire.plan != null && c.fire.plan.evidence().contains("gun=recent")
            && c.fire.plan.evidence().contains("label=DISPLACEMENT"),
        "fresh survivor actual selected typed plan evidence");
    ok(survivor.displacement.version == version && survivor.displacement.size() == count,
        "transition retains displacement bytes/counts versions");
    var frozen = c.fire.plan;
    c.fire.submitted(true, self, 2, 800);
    ok(c.fire.pending.evidence().equals(frozen.evidenceRow()),
        "pending freezes actual selected plan metadata");
    var confirmed = c.fire.confirm(555, 3, frozen.power());
    ok(confirmed.isPresent() && confirmed.get().evidence().equals(frozen.evidenceRow()),
        "confirmation retains frozen plan identity");
    c.roundReset(2);
    ok(survivor.displacement.size() == count, "round retains learned displacement");
  }
  @SuppressWarnings({"unchecked", "rawtypes"})
  static void failures() {
    for (String subsystem : List.of("movement", "gun")) {
      var c = new Core(R, new Diagnostics(), 1);
      c.roundReset(1);
      for (int id = 1; id <= 2; id++)
        c.enemy(id).observe(
            FacetSemanticTest.sample(0, b(550 + id * 50, 300, 0, 0)), new Physics.V(400, 300));
      if (subsystem.equals("movement"))
        ((ArrayDeque) c.melee.recent).add("injected malformed own history");
      else {
        var e = c.enemy(1);
        double[] extreme = new double[Learner.DIM];
        extreme[1] = Double.MAX_VALUE;
        e.displacement.add(
            new Memory.Observation(extreme, Memory.Label.DISPLACEMENT, 0, 0, 1, 1, 0, 1, 1, true));
      }
      for (int t : new int[] {0, 2, 4}) {
        for (var e : c.enemies.values()) e.latest = FacetSemanticTest.sample(t, e.latest.body());
        c.decide(new Physics.Body(new Physics.V(400, 300), 0, 0, 0, 0, 1), 100, t, 2, List.of(),
            List.of(), 99);
      }
      ok(c.log.counters.getOrDefault(subsystem + ".failure", 0L) == 3,
          "three actual active failures counted " + subsystem);
      ok(c.log.disabled.contains(subsystem), "active subsystem disabled " + subsystem);
      long dispatch = c.melee.dispatches, projections = c.melee.projectionPasses;
      c.decide(new Physics.Body(new Physics.V(400, 300), 0, 0, 0, 0, 1), 100, 6, 2, List.of(),
          List.of(), 99);
      ok(subsystem.equals("movement") ? c.melee.dispatches == dispatch
                                      : c.melee.projectionPasses == projections,
          "disabled melee subsystem does not rerun " + subsystem);
      if (subsystem.equals("gun"))
        ok(c.melee.decisionSamples == 0 && c.melee.decisionReturned == 0
                && c.melee.decisionQueries == 0,
            "disabled actual decision sample counts zero");
      c.melee.recent.clear();
      c.roundReset(2);
      ok(!c.log.disabled.contains(subsystem), "round re-enables subsystem " + subsystem);
    }
  }
  static void schedule() {
    var e = enemy(1, 0, b(600, 300, 0, 0), 100);
    var targets = List.of(e);
    var m = new Melee(1);
    for (int t = 0; t < 4; t++) {
      var self = new Physics.Body(new Physics.V(400, 300), 0, 0, 0, 0, 1);
      m.scheduledAim(self, targets, 100, t, 2, R);
    }
    // Freshness expires at turn3; key invalidation is required then.
    eq(m.projectionPasses, 2, "hot turns 0/2 expensive, stale boundary no projections");
    e.latest = new Memory.Sample(1, 4, e.latest.body(), 100, Memory.Quality.OBSERVED, 0);
    for (int t = 4; t <= 6; t++) {
      var self = new Physics.Body(new Physics.V(400, 300), 0, 0, 0, 0, .4);
      m.scheduledAim(self, targets, 100, t, 2, R);
    }
    eq(m.projectionPasses, 5, "four cooling turns boundary recomputes each turn");
    var hot = new Physics.Body(new Physics.V(400, 300), 0, 0, 0, 0, 1);
    m.scheduledAim(hot, targets, 100, 4, 2, R);
    long n = m.projectionPasses;
    e.displacement.add(displacement(0, 0, 1));
    m.scheduledAim(hot, targets, 100, 4, 2, R);
    ok(m.projectionPasses > n, "displacement version invalidates cache");
    n = m.projectionPasses;
    e.latest = new Memory.Sample(1, 4, b(610, 300, 0, 0), 100, Memory.Quality.OBSERVED, 0);
    m.scheduledAim(hot, targets, 100, 4, 2, R);
    ok(m.projectionPasses > n, "scan update invalidates cache");
    e.alive = false;
    ok(m.scheduledAim(hot, targets, 100, 4, 2, R) == null, "target death invalidates cache");
    e.alive = true;
    e.latest = new Memory.Sample(1, 9, e.latest.body(), 100, Memory.Quality.OBSERVED, 0);
    ok(m.select(hot, targets, 100, 4, 2, R) == null, "future scan cannot enter current context");
    var moving = new Melee(8);
    var live = enemy(2, 0, b(600, 300, 0, 0), 100);
    for (int t = 0; t < 8; t++) {
      live.latest = new Memory.Sample(1, t, b(600 + t, 300, 0, 0), 100, Memory.Quality.OBSERVED, 0);
      var a = moving.scheduledAim(hot, List.of(live), 100, t, 2, R);
      ok(t % 2 == 0 ? a != null && a.computedTurn() == t : a == null,
          "hot scan updates invalidate plans without defeating every-other-turn projection "
              + "schedule");
    }
    eq(moving.projectionPasses, 4, "eight hot moving-scan turns use four projection passes");
  }
  static void survivor() {
    double[] angles = new double[2];
    int i = 0;
    for (String name : List.of("long", "recent")) {
      var c = new Core(R, new Diagnostics(), 1);
      c.roundReset(1);
      c.initialCount = 3;
      c.beganMelee = true;
      var e = c.enemy(1);
      e.observe(FacetSemanticTest.sample(0, b(600, 300, 0, 0)), new Physics.V(400, 300));
      e.crossings = 16;
      e.selected = name;
      e.displacement.add(displacement(0, 3, 1));
      for (var s : e.recentDisplacement) s.add(displacement(0, -3, 2));
      var intent = c.decide(b(400, 300, 0, 0), 100, 0, 1, List.of(), List.of(), 99);
      ok(c.fire.plan != null, "survivor prepares actual selected plan");
      angles[i++] = Physics.wrap(c.fire.plan.angle());
      ok(e.longGun.size() == 0 && Arrays.stream(e.recent).allMatch(s -> s.size() == 0),
          "displacement not mixed into GF stores");
      var stale = c.decide(b(400, 300, 0, 0), 100, 4, 1, List.of(), List.of(), 99);
      ok(c.fire.plan == null && !stale.fire(),
          "stale melee-origin observation not fresh duel target");
    }
    ok(angles[0] > .05 && angles[1] < -.05,
        "selected long/recent typed bearings affect actual plan");
  }
  static void active() {
    var c = new Core(R, new Diagnostics(), 1);
    c.roundReset(1);
    for (int id = 1; id <= 2; id++)
      c.enemy(id).observe(
          FacetSemanticTest.sample(0, b(500 + id * 70, 300, 0, 0)), new Physics.V(400, 300));
    c.planner.steps = 123;
    var intent = c.decide(b(400, 300, 0, 0), 100, 0, 2, List.of(), List.of(), 99);
    ok(c.planner.steps == 123, "active melee does not execute full duel search");
    ok(c.fire.plan != null && c.fire.plan.policy().population() == 2, "active melee plan policy");
    ok(c.fire.plan.power() == 2.5, "active ordinary melee power");
    ok(Double.isFinite(intent.move().speed()), "active movement finite");
    c.roundReset(2);
    ok(c.melee.cached == null, "round clears aim cache");
    var chosen = new Core(R, new Diagnostics(), 1);
    chosen.roundReset(1);
    chosen.enemy(1).observe(
        FacetSemanticTest.sample(0, b(500, 300, 0, 0)), new Physics.V(400, 300));
    var far = chosen.enemy(2);
    far.observe(new Memory.Sample(1, 0, b(750, 400, 0, 0), 4, Memory.Quality.OBSERVED, 0),
        new Physics.V(400, 300));
    far.displacement.add(displacement(0, 0, 1));
    chosen.decide(b(400, 300, 0, 0), 100, 0, 2, List.of(), List.of(), 99);
    var plan = chosen.fire.plan;
    ok(plan.target() == 2 && plan.power() == 1, "actual core plan belongs to projected primary");
    var aligned = new Physics.Body(plan.origin(), 0, 0, plan.angle(), 0, 0);
    var shot = chosen.decide(aligned, 100, 1, 2, List.of(), List.of(), 99);
    ok(shot.fire() && chosen.shotContext.target == 2 && chosen.shotContext.powerHi == 1,
        "submission and physical context use selected primary rather than nearest initial target");
    chosen.fire.submitted(true, aligned, 1, 10);
    ok(chosen.fire.pending.target() == 2 && chosen.fire.pending.policy().target() == 2,
        "submission retains selected policy");
    for (double heat : new double[] {.4, .3}) {
      var duel = new Core(R, new Diagnostics(), 1);
      duel.roundReset(1);
      duel.enemy(1).observe(
          FacetSemanticTest.sample(0, b(600, 300, 0, 0)), new Physics.V(400, 300));
      duel.decide(new Physics.Body(new Physics.V(400, 300), 0, 0, 0, 0, heat), 100, 0, 1, List.of(),
          List.of(), 99);
      ok(duel.melee.projectionPasses == 0, "duel never invokes combined melee projector");
    }
  }
}
