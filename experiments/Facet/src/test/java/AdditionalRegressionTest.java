import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;
import java.util.*;
/** Independently declared analytic geometry; no gameplay parameter search. */
final class AdditionalRegressionTest {
  static void check(boolean v, String s) {
    FacetSemanticTest.ok(v, s);
  }
  static boolean contains(List<Geometry.Arc> arcs, double angle) {
    double a = Physics.norm(angle);
    return arcs.stream().anyMatch(x -> a >= x.lo() && a <= x.hi());
  }
  static double cross(Physics.V a, Physics.V b) {
    return a.x() * b.y() - a.y() * b.x();
  }
  // Independent parametric line/line intersection, followed by the official endpoint predicate.
  static boolean nativePredicate(
      Physics.V o, Physics.V a, Physics.V b, double inner, double outer, double angle) {
    Physics.V u = Physics.vector(angle), c = o.add(u.mul(inner)), e = o.add(u.mul(outer)),
              v = b.sub(a), w = e.sub(c), delta = c.sub(a);
    double det = cross(v, w);
    if (Math.abs(det) < 1e-10)
      return false;
    double t = cross(delta, w) / det, s = cross(delta, v) / det;
    return t >= 0 && t <= 1 && s >= 0 && s <= 1 && b.distance(e) <= 39.4;
  }
  static void certify(Physics.V a, Physics.V b, double inner, double outer, String name) {
    Physics.V o = new Physics.V(0, 0);
    var arcs = Waves.shadowSegment(o, a, b, inner, outer);
    int credited = 0;
    for (int i = 0; i < 20000; i++) {
      double angle = Physics.TAU * (i + .37) / 20000;
      if (contains(arcs, angle)) {
        credited++;
        check(nativePredicate(o, a, b, inner, outer, angle),
            name + " native segment and endpoint certificate");
      }
    }
    check(credited > 0, name + " nonempty certificate");
  }
  static void run() {
    Physics.V o = new Physics.V(0, 0), a = new Physics.V(90, -20), b = new Physics.V(90, 20);
    var partial = Waves.shadowSegment(o, a, b, 92, 100);
    check(!contains(partial, 0) && contains(partial, .21) && !contains(partial, -.21),
        "partial annulus excludes central unsupported sweep");
    certify(new Physics.V(90, -8), new Physics.V(90, 8), 90.2, 100, "legal-speed partial wrap");
    certify(a, b, 92, 100, "partial wrap");
    certify(a, b, 80, 100, "full wrap");
    certify(new Physics.V(-90, 20), new Physics.V(-90, -20), 80, 100, "opposite full");
    check(Waves.shadowSegment(o, a, b, 100, 120).isEmpty(), "disjoint shell");
    check(Waves.shadowSegment(o, a, b, 10, 20).isEmpty(), "wrong tick shell");
    check(Waves.shadowSegment(o, new Physics.V(90, -100), new Physics.V(90, 100), 89, 91).isEmpty(),
        "endpoint broad phase rejects geometric intersections");
    check(Waves.shadowSegment(o, a, a, 80, 100).isEmpty(), "zero segment no credit");
    check(Geometry.width(Geometry.around(0, 0)) == 0, "zero circular interval");
    check(Geometry.coverage(Geometry.around(0, .2), Geometry.around(Math.PI, .2)) == 0,
        "disjoint circular intervals");
    var r = FacetSemanticTest.RULES;
    var own = FacetSemanticTest.body(400, 300, 0, 0);
    Waves.Service service = new Waves.Service();
    var wave = FacetSemanticTest.wave(500, 1, 0, new Physics.V(0, 300), own, 2);
    service.add(wave);
    service.bullets.put(5, new Waves.Bullet(5, new Physics.V(90, 280), Math.PI / 2, 2, 5));
    service.shadows(5, r);
    check(service.shadowWork <= Waves.Service.MAX_SHADOW_WORK
            && service.shadowIntervals <= Waves.Service.MAX_SHADOW_INTERVALS,
        "bounded shadow work and storage");
    wave.shadows.add(new Waves.Shadow(5, 4, Geometry.around(0, .1)));
    wave.shadows.add(new Waves.Shadow(5, 6, Geometry.around(0, .1)));
    service.death(5, 6);
    check(wave.shadows.stream().noneMatch(x -> x.turn() >= 6)
            && wave.shadows.stream().anyMatch(x -> x.turn() == 4),
        "same tick death removes future preserves historical");
    service.death(5, 3);
    check(wave.shadows.isEmpty(), "death before history invalidates");
    Waves.Wave uncertain = new Waves.Wave(501, 1, 0, 1, 0, 0, Waves.Role.INCOMING, o, 1, 2, 2, 0, 1,
        1, 1, 1, new double[16], own, false);
    service.add(uncertain);
    service.shadows(5, r);
    check(uncertain.shadows.isEmpty() && service.shadowFallbacks > 0,
        "continuous uncertain origin explicit zero");
    Waves.Service loaded = new Waves.Service();
    for (int i = 0; i < 256; i++) {
      loaded.add(FacetSemanticTest.wave(800 + i, 1, 0, new Physics.V(100, 300), own, 2));
      loaded.bullets.put(i, new Waves.Bullet(i, new Physics.V(400, 300), 0, 2, 1));
    }
    loaded.shadows(1, r);
    check(loaded.shadowWork <= 4096 && loaded.shadowAllocations <= 524288
            && loaded.shadowIntervals <= 2048,
        "maximum own-bullet/wave solver quotas");
    Waves.Service wall = new Waves.Service();
    wall.add(FacetSemanticTest.wave(1100, 1, 0, o, own, 2));
    wall.bullets.put(1, new Waves.Bullet(1, new Physics.V(799, 300), 0, 2, 1));
    wall.shadows(1, r);
    check(wall.shadowWork <= 2, "future wall termination bounds trajectory");
    Core core = new Core(r, new Diagnostics(), 1);
    core.roundReset(1);
    var enemy = core.enemy(1);
    for (double energy : new double[] {0, .1, Double.NaN})
      check(core.predictPower(enemy, 200, 100, energy) == 0, "no affordable power");
    double legal = core.predictPower(enemy, 200, 100, .11);
    check(legal >= .1 && legal < .11, "strict affordable .11");
    check(Core.coolSteps(0, .1) == 0 && Core.coolSteps(.1, .1) == 1,
        "request-time cool and one-step clocks");
    for (double energy : new double[] {0, .1, .11}) {
      Core forecast = new Core(r, new Diagnostics(), 1);
      forecast.roundReset(1);
      forecast.observe(own, 100, 40, 1,
          List.of(new ScannedBotEvent(40, 0, 1, energy, 600, 300, 0, 0)), List.of(), 0, false);
      Physics.endWork();
      forecast.enemy(1).ledger.heatLow = forecast.enemy(1).ledger.heatHigh = 0;
      forecast.decide(own, 100, 40, 1, List.of(), List.of(), 0);
      var f = forecast.waves.all().stream().filter(x -> x.role == Waves.Role.FORECAST).findFirst();
      check(f.isPresent() == (energy > .1), "forecast affordability before construction");
      if (f.isPresent())
        check(f.get().start == 40 && f.get().end == 40, "cool forecast current request clock");
    }
    Core hot = new Core(r, new Diagnostics(), 1);
    hot.roundReset(1);
    hot.observe(own, 100, 40, 1, List.of(new ScannedBotEvent(40, 0, 1, 100, 600, 300, 0, 0)),
        List.of(), 0, false);
    hot.enemy(1).ledger.heatLow = hot.enemy(1).ledger.heatHigh = .1;
    hot.decide(own, 100, 40, 1, List.of(), List.of(), 0);
    check(hot.waves.all().stream().anyMatch(
              x -> x.role == Waves.Role.FORECAST && x.start == 41 && x.end == 41),
        "one cooling step earliest decision next turn");
    Core unknown = new Core(r, new Diagnostics(), 1);
    unknown.roundReset(1);
    unknown.decide(own, 100, 40, 1, List.of(), List.of(), 0);
    check(unknown.waves.all().isEmpty(), "missing scans invent no shots");
    unknown.fire.pending = new FireControl.Pending(1, 40, 2, 1, 0, own.p());
    unknown.fire.reconcile(42);
    unknown.observe(own, 100, 42, 1, List.of(),
        List.of(new BulletState(6000, 0, 2, 414, 300, 0, null)), 0, false);
    Physics.endWork();
    check(unknown.waves.bullets.isEmpty(),
        "pending cancellation and unconfirmed snapshot no forecast own shadows");
    Core stale = new Core(r, new Diagnostics(), 1);
    stale.roundReset(1);
    stale.observe(own, 100, 40, 1, List.of(new ScannedBotEvent(40, 0, 1, 100, 600, 300, 0, 0)),
        List.of(), 0, false);
    stale.enemy(1).ledger.heatLow = stale.enemy(1).ledger.heatHigh = 0;
    stale.decide(own, 100, 43, 1, List.of(), List.of(), 0);
    check(stale.waves.all().isEmpty(), "stale observation does not invent forecast clock");
    Core ghost = new Core(r, new Diagnostics(), 1);
    ghost.roundReset(1);
    ghost.waves.bullets.put(9, new Waves.Bullet(9, new Physics.V(400, 300), 0, 2, 3));
    var history = FacetSemanticTest.wave(502, 1, 0, o, own, 2);
    ghost.waves.add(history);
    history.shadows.add(new Waves.Shadow(9, 3, Geometry.around(0, .1)));
    history.shadows.add(new Waves.Shadow(9, 5, Geometry.around(0, .1)));
    ghost.observe(own, 100, 8, 1, List.of(), List.of(), 0, false);
    Physics.endWork();
    check(ghost.waves.deaths.get(9) == 4 && history.shadows.stream().allMatch(x -> x.turn() < 4),
        "skipped disappearance invalidates since last alive plus one");
    terminal(own, r);
    controlledTwoWave(r);
  }
  static void controlledTwoWave(Physics.Rules r) {
    // Declared geometry: east of wave1, moving north at legal speed8. Wave2
    // comes from due north, 220 units away, with radius112 now. Its head-on
    // evidence is independent analytic aim toward the frozen self position.
    // Continuing north closes at22/tick; reversing delays overlap. Wave1's
    // contact is earlier by static envelope order, while wave2 overlaps before
    // wave1's center. No profile parameters are changed or searched.
    var self = FacetSemanticTest.body(600, 300, Math.PI / 2, 8);
    var first = FacetSemanticTest.wave(700, 1, -8, new Physics.V(400, 300), self, 2);
    var second = FacetSemanticTest.wave(701, 2, -8, new Physics.V(600, 520), self, 2);
    var budget = new Memory.Budget();
    var enemies = Map.of(1, FacetSemanticTest.enemy(1, 400, 300, budget), 2,
        FacetSemanticTest.enemy(2, 600, 520, budget));
    var one = new Planner().choose(self, 0, List.of(first), enemies, r, Map.of());
    var pair = new Planner().choose(self, 0, List.of(first, second), enemies, r, Map.of());
    System.out.println("controlled native two-wave actions=" + one.action() + "/" + pair.action()
        + " costs=" + one.cost() + "/" + pair.cost());
    check(one.complete() && pair.complete() && !one.emergency() && !pair.emergency(),
        "controlled two-wave paths evaluated safely");
    check(one.action() != pair.action(), "controlled native wave2 changes first action");
    Planner probe = new Planner();
    var path = probe.simulate(self, 0, first, one.action(), null, r, Map.of());
    boolean overlap = false;
    for (int i = 1; i < path.states().size() && i < path.crossingTurn(); i++)
      if (!second.footprint(path.states().get(i - 1).p(), path.states().get(i).p(), i).isEmpty())
        overlap = true;
    check(overlap, "controlled wave2 overlaps before first center history");
    check(one.path().stream().allMatch(s -> r.safe(s.p()))
            && pair.path().stream().allMatch(s -> r.safe(s.p())),
        "both selected first trajectories safe");
  }
  static void terminal(Physics.Body own, Physics.Rules rules) {
    for (boolean reverse : new boolean[] {false, true}) {
      Facet facet = new Facet(java.net.URI.create("ws://localhost:12345"));
      facet.onGameStarted(null);
      facet.core = new Core(rules, facet.diagnostics, 1);
      facet.core.roundReset(1);
      Core core = facet.core;
      core.decide(own, 100, 40, 1, List.of(new ScannedBotEvent(40, 0, 1, 100, 600, 300, 0, 0)),
          List.of(), 0);
      core.decide(own, 100, 41, 1, List.of(new ScannedBotEvent(41, 0, 1, 100, 600, 300, 0, 0)),
          List.of(), 0);
      core.fire.submitted(true, own, 41, 1);
      BulletState bullet = new BulletState(909, 0, 2, 400, 300, 0, null);
      List<BotEvent> events = new ArrayList<>(
          List.of(new BulletFiredEvent(42, bullet), new BulletHitBotEvent(42, 1, bullet, 10, -1),
              new BotDeathEvent(42, 1), new ScannedBotEvent(42, 0, 1, 0, 600, 300, 0, 0)));
      if (reverse)
        Collections.reverse(events);
      facet.batch.addAll(events);
      facet.batch.addAll(events);
      long decisions = facet.diagnostics.counters.getOrDefault("turn.decisions", 0L);
      Runnable finish = () -> facet.finishRound(1, own, 98, 42, 0, List.of(), 0);
      if (reverse) {
        facet.finishGame(finish);
        finish.run();
      } else {
        finish.run();
        facet.finishGame(finish);
      }
      finish.run();
      facet.finishGame(finish);
      check(core.fire.confirmed == 1 && facet.diagnostics.counters.get("shot.hit") == 1,
          "terminal final confirmation/hit once");
      check(!core.enemy(1).alive && facet.batch.isEmpty(), "terminal death and batch drained");
      check(facet.completed.size() == 1 && facet.diagnostics.completed.size() == 1,
          "one completion identity");
      check(facet.diagnostics.counters.get("turn.decisions") == decisions,
          "terminal emits no planned commands");
      check(core.enemy(1).longGun.size() == 0 && core.fire.pending == null,
          "terminal unresolved outcomes never train");
      String game = facet.diagnostics.gameId;
      facet.diagnostics.close();
      facet.onGameStarted(null);
      check(facet.core == null && !facet.gameClosed && facet.completed.isEmpty()
              && !game.equals(facet.diagnostics.gameId),
          "reused API new game cold identity and live diagnostics");
      facet.core = new Core(rules, facet.diagnostics, 1);
      check(facet.core.enemies.isEmpty() && facet.core.fire.confirmed == 0
              && facet.core.budget.bytes == 0,
          "cold models ratings counters");
      facet.core.roundReset(1);
      facet.finishRound(1, own, 100, 1, 1, List.of(), 0);
      check(facet.diagnostics.completed.size() == 1
              && facet.diagnostics.counters.getOrDefault("metrics.ioFailure", 0L) == 0,
          "second game usable metrics writer");
      facet.diagnostics.close();
    }
  }
}
