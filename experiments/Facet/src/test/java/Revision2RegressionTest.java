import java.io.*;
import java.util.*;
/** New Facet fixtures: mathematical contracts, never battle evidence. */
final class Revision2RegressionTest {
  static final Physics.Rules R = new Physics.Rules(800, 600, .1);
  static Physics.Body b(double x, double y, double h, double v) {
    return FacetSemanticTest.body(x, y, h, v);
  }
  static Memory.Sample sample(int t, Physics.Body b) {
    return FacetSemanticTest.sample(t, b);
  }
  static void ok(boolean v, String s) {
    FacetSemanticTest.ok(v, s);
  }
  static void eq(double a, double b, String s) {
    FacetSemanticTest.eq(a, b, 1e-9, s);
  }
  static void run() throws RuntimeException {
    // Independent compiled constraint expectation (800x600, corrected reviewer fixture):
    // (781,580) -> raw(786.656854,585.656854) -> (783,582) -> (782,581).
    // Reflect each axis, then swap the axes on a 600x800 field.
    for (boolean swap : new boolean[] {false, true})
      for (int sx : new int[] {-1, 1})
        for (int sy : new int[] {-1, 1}) {
          double x = sx > 0 ? 781 : 19, y = sy > 0 ? 580 : 20, ex = sx > 0 ? 782 : 18,
                 ey = sy > 0 ? 581 : 19;
          var r = swap ? new Physics.Rules(600, 800, .1) : R;
          var body = swap ? b(y, x, Math.atan2(sx, sy), 8) : b(x, y, Math.atan2(sy, sx), 8);
          var step = Physics.step(body, new Physics.Control(8, 0, 0, 0), r);
          eq(step.body().p().x(), swap ? ey : ex, "compiled corner x reflection");
          eq(step.body().p().y(), swap ? ex : ey, "compiled corner y reflection");
          ok(step.wall() && step.body().speed() == 0, "corner stops");
          eq(step.damage(), 3, "corner damage");
          var projection = Physics.constantPosition(body, 100, r);
          ok(projection.x() >= 18 && projection.x() <= r.width() - 18 && projection.y() >= 18
                  && projection.y() <= r.height() - 18,
              "analytic legal bounds");
        }
    var e = new Learner.Enemy(1, new Memory.Budget());
    var source = new Physics.V(300, 300);
    var frozen = b(500, 300, Math.PI / 2, 0);
    for (int t = 0; t <= 40; t++) e.observe(sample(t, frozen), source);
    var before = Learner.context(source, frozen, e, 40, 2, R, 1, 1, false);
    e.observe(sample(41, b(508, 300, 0, 8)), source);
    var after = Learner.context(source, frozen, e, 40, 2, R, 1, 1, false);
    ok(Arrays.equals(before.features(), after.features())
            && before.missingFeatures() == after.missingFeatures(),
        "future append cannot alter old context");
    e.observe(sample(60, frozen), source);
    ok(Learner.context(source, frozen, e, 60, 2, R, 1, 1, false).quality() == .5,
        "gap context missing");
    e.reset();
    ok(e.history.samples.isEmpty() && e.kinematics.isEmpty(), "round clears context chronology");
    var core = new Core(R, new Diagnostics(), 1);
    core.roundReset(1);
    var left = core.enemy(1);
    var right = core.enemy(2);
    left.observe(sample(0, b(200, 300, 0, 0)), source);
    right.observe(sample(0, b(600, 300, 0, 0)), source);
    var moving = b(400, 300, Math.PI / 2, 8);
    var stopped = b(400, 308, Math.PI / 2, 0);
    for (var shooter : List.of(left, right)) {
      core.recordOrbit(shooter, moving, 0);
      core.recordOrbit(shooter, stopped, 1);
    }
    ok(Core.originOrbit(left, 1) == 1 && Core.originOrbit(right, 1) == -1,
        "stopped own sign is shooter specific");
    core.recordOrbit(left, b(400, 308, -Math.PI / 2, 8), 2);
    ok(Core.originOrbit(left, 1) == 1, "future direction preserves old origin");
    var own = b(400, 300, 0, 0);
    var target = new Learner.Enemy(3, new Memory.Budget());
    target.observe(sample(40, b(600, 300, 0, 0)), own.p());
    var fire = new FireControl();
    fire.plan = FireControl.prepare(0, 2, target, own, 40, 1);
    ok(!fire.valid(own, 2.1, target, 41, R), "submission reserves .2");
    target.observe(
        new Memory.Sample(1, 41, target.latest.body(), .4, Memory.Quality.OBSERVED, 0), own.p());
    ok(!fire.valid(own, 100, target, 41, R), "latest kill cap invalidates power");
    fire.plan = FireControl.prepare(0, .1, target, own, 41, 1);
    ok(!fire.valid(new Physics.Body(own.p(), 0, 0, 0, 0, Double.MIN_VALUE), 100, target, 41, R),
        "tiny positive heat illegal");
    fire.submitted(true, own, 41, 1);
    var pending = fire.pending;
    ok(!fire.valid(own, 100, target, 41, R) && fire.pending == pending,
        "pending physical request immutable");
    var heat = new Energy.Ledger();
    heat.scan(40, 98, .1, false);
    ok(heat.heatLow == 0 && heat.heatHigh >= 1.4, "late first scan includes unseen shot heat");
    var early = new Energy.Ledger();
    early.scan(1, 100, .1, false);
    eq(early.heatLow, 2.9, "startup phase decision one cooling");
    eq(early.heatHigh, 2.9, "ordinary early scan heat");
    var farCore = new Core(new Physics.Rules(5000, 5000, .1), new Diagnostics(), 1);
    farCore.roundReset(1);
    farCore.enemy(1);
    var far = b(4982, 4982, 0, 0);
    var wave = FacetSemanticTest.wave(1, 1, 0, new Physics.V(18, 18), far, 3);
    farCore.waves.add(wave);
    farCore.resolve(far, 513);
    ok(wave.phase(far.p(), 513) == Waves.Phase.APPROACHING && farCore.waves.all().contains(wave),
        "unresolved 513-turn real threat retained");
    for (int sign : new int[] {-1, 1})
      for (int speedSign : new int[] {-1, 1}) {
        var opponent = new Learner.Enemy(5, new Memory.Budget());
        opponent.observe(
            sample(40,
                b(400 + sign * 45, 300, (sign > 0 ? Math.PI : 0) + (speedSign < 0 ? Math.PI : 0),
                    8 * speedSign)),
            own.p());
        var self =
            b(400, 300, (sign > 0 ? 0 : Math.PI) + (speedSign < 0 ? Math.PI : 0), 8 * speedSign);
        var planner = new Planner();
        var choice = planner.choose(self, 40, List.of(), Map.of(5, opponent), R, Map.of());
        var next = Physics.step(self, choice.command(), R).body();
        var enemyNext =
            Physics.step(opponent.latest.body(), new Physics.Control(8 * speedSign, 0, 0, 0), R)
                .body();
        ok(next.p().distance(enemyNext.p()) >= 36, "avoidable CV collision rejected both signs");
        ok(planner.collisionRisk(next.p(), 41, R) == 0, "legal unknown speed envelope safe");
        opponent.observe(sample(41, b(400 + sign * 60, 300, sign > 0 ? 0 : Math.PI, 0)), own.p());
        ok(planner.collisionRisk(self.p(), 42, R) == 0,
            "stopped opponent actual speed law avoids unconditional buffer");
      }
    cohorts();
    exports();
    recency();
    var gapCore = new Core(R, new Diagnostics(), 1);
    gapCore.roundReset(1);
    var gapEnemy = gapCore.enemy(1);
    gapEnemy.observe(sample(40, b(600, 300, 0, 0)), own.p());
    for (int t = 0; t <= 40; t++) gapEnemy.ledger.scan(t, 100, .1, false);
    gapCore.ownHistory.add(sample(40, own));
    gapCore.observe(own, 100, 60, 1,
        List.of(new dev.robocode.tankroyale.botapi.events.ScannedBotEvent(
            60, 0, 1, 98, 600, 300, 0, 0)),
        List.of(), 0, false);
    Physics.endWork();
    var envelope = gapCore.waves.all()
                       .stream()
                       .filter(w -> w.role == Waves.Role.INCOMING)
                       .findFirst()
                       .orElseThrow();
    eq(envelope.originError, 160, "gap origin bounds 8dt without 64 cap");
    ok(envelope.start != envelope.end && !envelope.ratingEligible,
        "gap remains unresolved distinct from forecast");
    var missingCore = new Core(R, new Diagnostics(), 1);
    missingCore.roundReset(1);
    missingCore.ownHistory.add(sample(39, b(390, 300, 0, 0)));
    missingCore.ownHistory.add(sample(41, b(410, 300, 0, 0)));
    ok(!missingCore.ownHistory.samples.containsKey(40),
        "missing origin is not a future-assisted observation");
    e = new Learner.Enemy(6, new Memory.Budget());
    for (int t = 0; t <= 40; t++) e.observe(sample(t, frozen), source);
    e.opportunities.add(38);
    eq(Learner.context(source, frozen, e, 40, 2, R, 1, 1, false).features()[14], .25,
        "known opportunity chronology");
    e.opportunities.add(41);
    eq(Learner.context(source, frozen, e, 40, 2, R, 1, 1, false).features()[14], .25,
        "future opportunity excluded");
    eq(Learner.context(source, frozen, e, 40, 2, R, 1, 1, true).features()[14], 0,
        "physical proximity zero");
    var radar = new Radar();
    for (int t = 41; t <= 64; t++) radar.turn(own, List.of(e), t, 2);
    ok(radar.acquiredAfter == 64 && e.history.samples.size() == 41,
        "radar only invalidates acquisition");
    eq(Math.abs(radar.turn(own, List.of(e), 65, 1)), Math.PI / 4, "stale acquisition forces sweep");
    ok(e.recentDisplacement[0].capacity == 128 && e.recentDisplacement[1].capacity == 512
            && e.recentDisplacement[2].capacity == 2048,
        "typed independent recency capacities");
    ok(!Melee.REVIEW_APPROVED, "melee remains gated");
    var closeSource = new Physics.V(344, 300);
    var stationary = b(400, 300, Math.PI / 2, 0);
    var horizonEnemy = new Learner.Enemy(7, new Memory.Budget());
    for (int t = 0; t <= 40; t++) horizonEnemy.observe(sample(t, stationary), closeSource);
    Physics.beginWork(2400);
    var horizon = Learner.context(closeSource, stationary, horizonEnemy, 40, 2, R, 1, 1, false);
    int used = Physics.workUsed();
    Physics.endWork();
    ok(used == 12, "wall context simulates both full ceil1.5H horizons after bullet crossing");
    eq(horizon.features()[7], 1.5, "full clear wall availability");
    Physics.beginWork(405);
    var limited = Learner.context(closeSource, stationary, horizonEnemy, 40, 2, R, 1, 1, false);
    Physics.endWork();
    ok(limited.quality() == .5 && limited.features()[8] == 0,
        "quota missing wall value flagged and zero");
  }
  static void cohorts() {
    var e = new Learner.Enemy(9, new Memory.Budget());
    var target = b(600, 300, 0, 0);
    for (int i = 0; i < 16; i++) {
      var w = FacetSemanticTest.wave(100 + i, 0, 0, new Physics.V(400, 300), target, 2);
      w.ratingEligible = true;
      w.predictions.put("analytic", new Waves.Prediction("analytic", w.id, 0, 0, 2, 0));
      Learner.trainGun(e, w, sample(15, target), i);
    }
    ok(e.crossings == 16 && e.startupRating.count == 16 && e.ratings.get("analytic").count == 0,
        "startup separated from comparison");
    for (int i = 0; i < 8; i++) {
      var w = FacetSemanticTest.wave(200 + i, 0, 0, new Physics.V(400, 300), target, 2);
      w.ratingEligible = true;
      for (String name : List.of("analytic", "long", "recent"))
        w.predictions.put(
            name, new Waves.Prediction(name, w.id, 0, name.equals("analytic") ? .3 : 0, 2, 0));
      Learner.trainGun(e, w, sample(15, target), 20 + i);
      for (var rating : e.ratings.values())
        ok(rating.count == i + 1, "same resolved comparison cohort");
      if (i < 7)
        ok(e.selected.equals("analytic"), "eight common outcomes required");
    }
    ok(!e.selected.equals("analytic"), "raw margin selector transitions after eight");
    var cancelled = FacetSemanticTest.wave(300, 0, 0, new Physics.V(400, 300), target, 2);
    cancelled.cancelled = true;
    cancelled.ratingEligible = true;
    long count = e.ratings.get("long").count;
    Learner.trainGun(e, cancelled, sample(15, target), 40);
    ok(e.ratings.get("long").count == count, "cancelled cohort no outcome leakage");
  }
  static void recency() {
    var e = new Learner.Enemy(10, new Memory.Budget());
    var target = b(500, 300, 0, 0);
    double[] f = new double[16];
    for (int i = 0; i < 6144; i++) {
      var label = new Memory.Observation(
          f, Memory.Label.DISPLACEMENT, 0, i < 4096 ? 1 : -1, i, 1, i, 1, i, true);
      e.displacement.add(label);
      for (var store : e.recentDisplacement) store.add(label);
    }
    var w = new Waves.Wave(1, 0, 10, 1, 0, 0, Waves.Role.VIRTUAL, new Physics.V(300, 300), 0, 2, 2,
        0, 1, 1, 1, 1, f, target, true);
    double longAngle = Physics.wrap(Learner.gun("long", e, w, 0, R)),
           recentAngle = Physics.wrap(Learner.gun("recent", e, w, 0, R));
    ok(longAngle > 0 && recentAngle < 0,
        "survivor long and recent decode distinct bounded displacement evidence");
    ok(e.displacement.size() == 6144 && e.recentDisplacement[0].size() == 128
            && e.recentDisplacement[1].size() == 512 && e.recentDisplacement[2].size() == 2048,
        "survivor views evict independently");
  }
  static byte[] bytes(Core c) {
    try {
      var out = new ByteArrayOutputStream();
      ModelSnapshot.write(c, out);
      return out.toByteArray();
    } catch (IOException ex) {
      throw new RuntimeException(ex);
    }
  }
  static void exports() {
    var c = new Core(R, new Diagnostics(), 1);
    c.roundReset(1);
    var e = c.enemy(1);
    byte[] previous = bytes(c);
    e.shots = 50;
    e.rawHits = 10;
    ok(!Arrays.equals(previous, bytes(c)), "flattener counters change export hash bytes");
    previous = bytes(c);
    e.observedTogether++;
    ok(!Arrays.equals(previous, bytes(c)), "threat counter exported");
    previous = bytes(c);
    e.comparisonEligible = true;
    ok(!Arrays.equals(previous, bytes(c)), "eligibility exported");
    ok(ModelSnapshot.SCHEMA == 2, "export schema version two");
  }
}
