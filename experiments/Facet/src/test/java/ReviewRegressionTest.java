import dev.robocode.tankroyale.botapi.events.*;
import java.util.*;
final class ReviewRegressionTest {
  static void check(boolean value, String reason) {
    FacetSemanticTest.ok(value, reason);
  }
  static void eq(double a, double b, double t, String reason) {
    FacetSemanticTest.eq(a, b, t, reason);
  }
  static final Physics.Rules R = FacetSemanticTest.RULES;
  static Physics.Body body(double x, double y, double h, double v) {
    return FacetSemanticTest.body(x, y, h, v);
  }
  static void run() {
    // Property key is deliberately lower case. Validation never constructs a Bot or connects.
    check(Facet.resolveServer("ws://localhost:12345", "ws://localhost:7654").getPort() == 12345,
        "official property precedence");
    FacetSemanticTest.rejects(
        ()
            -> Facet.resolveServer("ws://localhost:7654", "ws://localhost:12345"),
        "lowercase unsafe property overrides safe env and fails");
    FacetSemanticTest.rejects(
        () -> Facet.validateServer("ws://localhost"), "explicit port required");
    FacetSemanticTest.rejects(() -> Facet.validateServer("ws://[broken:12345"), "malformed IPv6");
    check(Facet.validateServer("ws://[::1]:12345").getPort() == 12345, "isolated IPv6");
    FacetSemanticTest.rejects(()
                                  -> Facet.resolveServer("", "ws://localhost:12345"),
        "blank explicit property fails closed");
    System.setProperty("SERVER_URL", "ws://localhost:7654");
    check(Facet.resolveServer(null, "ws://localhost:12345").getPort() == 12345,
        "uppercase property ignored");
    System.clearProperty("SERVER_URL");
    for (double v : new double[] {-8, -.25, 0, .25, 8})
      eq(Physics.velocity(v, 0), 0, 0, "STOP all signs");
    eq(Physics.velocity(.5, -8), -.75, 0, "fractional forward reversal");
    eq(Physics.velocity(-.5, 8), .75, 0, "fractional reverse reversal");
    var oblique = body(780, 300, Math.PI / 4, 8);
    var endpoint = Physics.constantPosition(oblique, 10, R);
    eq(endpoint.x(), 782, 1e-9, "first segment contact x");
    eq(endpoint.y(), 302, 1e-9, "no wall sliding");
    var corner = Physics.constantPosition(body(780, 580, Math.PI / 4, 8), 10, R);
    eq(corner.x(), 782, 1e-9, "corner x");
    eq(corner.y(), 582, 1e-9, "corner y");
    // Narrow peaks between the former 81 sample locations need a certified bound.
    double center = Math.PI / 81, sigma = 1e-4, bound = 1 / (Math.sqrt(2 * Math.PI) * sigma);
    Geometry.Density peak = new Geometry.Density() {
      public double at(double a) {
        return Geometry.gaussian(Physics.wrap(a - center), sigma);
      }
      public double upperBound() {
        return bound;
      }
    };
    eq(Geometry.integrate(List.of(new Geometry.Arc(0, 6)), peak), 6 * bound, 1e-8,
        "certified adversarial peak fallback");
    Energy.Ledger heat = new Energy.Ledger();
    heat.scan(0, 100, .1, false);
    heat.heatLow = heat.heatHigh = .1;
    check(heat.scan(1, 98, .1, false).isEmpty(), "cannot fire on turn that only cools to zero");
    Energy.Ledger epsilonHeat = new Energy.Ledger();
    epsilonHeat.scan(40, 100, .1, false);
    epsilonHeat.heatLow = epsilonHeat.heatHigh = 1e-14;
    check(epsilonHeat.scan(41, 98, .1, false).isEmpty(), "strict zero heat before cooling");
    Energy.Ledger cold = new Energy.Ledger();
    cold.scan(40, 100, .1, false);
    var infer = cold.scan(41, 98, .1, false).orElseThrow();
    check(
        infer.earliest() == 40 && infer.latest() == 40 && infer.reliable(), "exact decision clock");
    eq(cold.heatHigh, 1.4, 1e-12, "no creation cooling");
    check(cold.scan(42, 96, .1, false).isEmpty(), "hot gun cannot fire consecutively");
    var budget = new Memory.Budget();
    var enemy = FacetSemanticTest.enemy(1, 600, 300, budget);
    var own = body(400, 300, 0, 0);
    var core = new Core(R, new Diagnostics(), 11);
    core.roundReset(1);
    core.enemy(1).observe(FacetSemanticTest.sample(0, enemy.latest.body()), own.p());
    var w = core.makeOutgoing(own, core.enemy(1), 0, 2, Waves.Role.VIRTUAL, 0, false);
    check(w.originQuality == .5 && w.missingFeatures != 0, "frozen missing origin flags");
    w.predictions.put("analytic", new Waves.Prediction("analytic", w.id, 0, 0, 2, 0));
    Learner.trainGun(core.enemy(1), w, FacetSemanticTest.sample(15, body(600, 300, 0, 0)), 1);
    eq(core.enemy(1).longGun.data.peekFirst().quality(), .5, 0, "origin quality once");
    check(core.enemy(1).ratings.get("analytic").count == 0, "virtual cohort cannot rate");
    var confirmed = core.makeOutgoing(own, core.enemy(1), 0, 2, Waves.Role.OUTGOING, 0, true);
    confirmed.ratingEligible = true;
    for (String name : List.of("analytic", "long", "recent"))
      confirmed.predictions.put(name, new Waves.Prediction(name, confirmed.id, 0, 0, 2, 0));
    Learner.trainGun(
        core.enemy(1), confirmed, FacetSemanticTest.sample(15, body(600, 300, 0, 0)), 2);
    for (var rating : core.enemy(1).ratings.values())
      check(rating.count == 1, "common confirmed denominator");
    // Mirror the prior under negative signed lateral motion; orbit sign must not flip twice.
    var positive = FacetSemanticTest.wave(50, 1, 0, own.p(), body(600, 300, Math.PI / 2, 8), 2);
    var negative = new Waves.Wave(51, 1, 0, 1, 0, 0, Waves.Role.INCOMING, own.p(), 0, 2, 2, 0, -1,
        Physics.escape(2), Physics.escape(2), 1, new double[16], body(600, 300, -Math.PI / 2, 8),
        false);
    var dp = Learner.prepareDensity(enemy, positive, R);
    var dn = Learner.prepareDensity(enemy, negative, R);
    for (double angle : new double[] {.1, .2, .4})
      eq(dp.at(angle), dn.at(-angle), 1e-9, "mirrored native CV prior");
    // Disabled shadow service gives no safety credit, including wrong tick/partial annuli/broad
    // phase.
    var service = new Waves.Service();
    service.add(positive);
    service.bullets.put(4, new Waves.Bullet(4, new Physics.V(350, 310), 0, 2, 0));
    positive.shadows.add(new Waves.Shadow(4, 4, Geometry.around(0, .3)));
    service.shadows(0, R);
    check(positive.shadows.isEmpty() && service.shadowFallbacks == 1,
        "explicit disabled shadow counter");
    eq(positive.unshadowed(Geometry.around(0, .2), 20, Map.of()), 1, 0,
        "disabled service never reduces risk");
    positive.shadows.add(new Waves.Shadow(4, 4, Geometry.around(0, .2)));
    eq(positive.unshadowed(Geometry.around(0, .2), 10, Map.of(4, 4)), 1, 0,
        "same tick death no credit");
    // A reliable center crossing is retained until full clear, then emitted once.
    Core visits = new Core(R, new Diagnostics(), 1);
    visits.roundReset(1);
    var incoming = FacetSemanticTest.wave(60, 1, 0, new Physics.V(0, 300), body(100, 300, 0, 0), 2);
    visits.enemy(1);
    visits.waves.add(incoming);
    for (int t = 0; t <= 10; t++) {
      var state = body(100, 300 + t, 0, 0);
      visits.ownHistory.add(FacetSemanticTest.sample(t, state));
      visits.resolve(state, t);
    }
    check(incoming.center != null, "center retained");
    check(visits.enemy(1).visits.size() == 1, "visit exactly once at clear");
    eq(visits.enemy(1).visits.data.peekFirst().turn(), incoming.center.turn(), 0,
        "center time not clear time");
    check(visits.enemy(1).shots == 1, "resolved real count once");
    FacetSemanticTest.rejects(() -> Energy.wilsonLower(3, 2), "reject impossible rates");
    check(Energy.wilsonLower(10, 80) > 0, "consistent could-hit denominator");
    // Survivor label frame decodes negative speed without treating it as heading/speed swapped.
    var target = body(500, 300, 0, -8);
    var label = new Memory.Observation(
        new double[16], Memory.Label.DISPLACEMENT, 1, .5, 1, 1, 1, 1, 1, true);
    var point = Melee.project(label, target, -1, own.p(), 2, R).orElseThrow();
    check(point.x() < 500 && point.y() > 300,
        "negative effective heading and reflected sign decoded");
    // Incompatible fresh position cancels, compatible fresh scan can fire.
    FireControl fire = new FireControl();
    fire.plan = FireControl.prepare(0, 2, enemy, own, 0, 1);
    enemy.observe(FacetSemanticTest.sample(1, body(600, 300, 0, 0)), own.p());
    check(fire.valid(own, 100, enemy, 1), "fresh compatible plan");
    enemy.observe(FacetSemanticTest.sample(2, body(300, 500, 0, 0)), own.p());
    check(!fire.valid(own, 100, enemy, 2), "major incompatible scan cancels");
    fire.reset();
    fire.plan = FireControl.prepare(0, 2, enemy, own, 2, 1);
    fire.submitted(true, own, 2, 1);
    check(fire.confirm(4, 3, 2, new Physics.V(401, 300), 0).isEmpty(),
        "mismatched actual event origin remains unmatched");
    // Same-batch contact must see scans first, even if contact callback came first.
    Core batch = new Core(R, new Diagnostics(), 5);
    batch.roundReset(1);
    batch.decide(
        own, 100, 40, 1, List.of(new ScannedBotEvent(40, 0, 1, 100, 480, 300, 0, 0)), List.of(), 0);
    var hostile = new dev.robocode.tankroyale.botapi.BulletState(77, 1, 2, 466, 300, 180, null);
    batch.decide(own, 100, 41, 1,
        List.of(new HitByBulletEvent(41, hostile, 10, 90),
            new ScannedBotEvent(41, 0, 1, 98, 480, 300, 0, 0)),
        List.of(), 0);
    check(batch.enemy(1).simple.size() == 1 && batch.waves.unmatched == 0,
        "same-batch scan inference precedes association");
    HitBotEvent contact1 = new HitBotEvent(50, 1, 100, 410, 300, true),
                contact2 = new HitBotEvent(50, 2, 100, 390, 300, true);
    check(batch.dedup(contact1) && batch.dedup(contact2) && !batch.dedup(contact1),
        "distinct same-turn contacts preserved, repeats suppressed");
    // The confirmed cohort retains request origin despite own translation at confirmation.
    Core shotCore = new Core(R, new Diagnostics(), 3);
    shotCore.roundReset(1);
    shotCore.decide(
        own, 100, 1, 1, List.of(new ScannedBotEvent(1, 0, 1, 100, 600, 300, 0, 0)), List.of(), 0);
    var intent = shotCore.decide(
        own, 100, 2, 1, List.of(new ScannedBotEvent(2, 0, 1, 100, 600, 300, 0, 0)), List.of(), 0);
    check(intent.fire(), "consecutive scans analytic shot prepared");
    var predictions = Map.copyOf(shotCore.shotContext.predictions);
    shotCore.fire.submitted(true, own, 2, 99);
    var outgoingBullet =
        new dev.robocode.tankroyale.botapi.BulletState(80, 0, 2, 400, 300, 0, null);
    shotCore.decide(body(408, 300, .1, 8), 98, 3, 1,
        List.of(new BulletFiredEvent(3, outgoingBullet),
            new ScannedBotEvent(3, 0, 1, 100, 600, 300, 0, 0)),
        List.of(), 0);
    var physical = shotCore.waves.all()
                       .stream()
                       .filter(x -> x.role == Waves.Role.OUTGOING)
                       .findFirst()
                       .orElseThrow();
    eq(physical.origin.x(), 400, 0, "confirmed event origin not postmovement state");
    check(physical.start == 2 && physical.end == 2 && physical.creationTurn == 3
            && physical.firstMotionTurn == 3,
        "separate outgoing clocks");
    check(physical.predictions.equals(predictions) && physical.ratingEligible,
        "frozen request cohort retained");
    eq(physical.actualAngle, 0, 0, "actual angle separate");
    check(shotCore.waves.all().stream().noneMatch(
              x -> x.role == Waves.Role.VIRTUAL && x.start == 2 && !x.cancelled),
        "matching virtual opportunity cancelled");
    // Already-cleared second waves charge retained union exactly once and take zero new steps.
    Planner counted = new Planner();
    var passed = FacetSemanticTest.wave(90, 1, -100, new Physics.V(200, 300), own, 2);
    var arcs = List.of(new Geometry.Arc(0, .2), new Geometry.Arc(.1, .3));
    var path = counted.simulate(own, 0, passed, 1, null, R, Map.of(), arcs, -2, -5);
    check(path.complete() && counted.steps == 0, "passed second no extra tick");
    eq(path.cost(), Physics.damage(2) * .3 / (2 * Physics.escape(2)) / 3, 1e-12,
        "prior overlap union charged once");
    // All-real cap uses an explicit radial envelope that cannot train exact labels.
    Waves.Service capped = new Waves.Service();
    for (int i = 0; i < 257; i++)
      capped.add(FacetSemanticTest.wave(100 + i, 1, -i, new Physics.V(200 + i * .1, 300), own, 2));
    check(capped.all().size() == 256 && capped.realOverflow == 1, "real overflow explicit");
    check(capped.all().stream().anyMatch(x -> x.originError > 0 && x.originQuality == 0),
        "overflow conservative origin envelope");
    Core disabledGun = new Core(R, new Diagnostics(), 1);
    disabledGun.roundReset(1);
    disabledGun.log.disabled.add("gun");
    disabledGun.decide(
        own, 100, 1, 1, List.of(new ScannedBotEvent(1, 0, 1, 100, 600, 300, 0, 0)), List.of(), 0);
    check(disabledGun
              .decide(own, 100, 2, 1, List.of(new ScannedBotEvent(2, 0, 1, 100, 600, 300, 0, 0)),
                  List.of(), 0)
              .fire(),
        "disabled learned gun analytic fallback still fires on fresh scans");
    // Disabling each learned subsystem persists through successful fallback until round reset.
    var diagnostics = new Diagnostics();
    for (String subsystem : List.of("gun", "movement", "radar", "inference", "learning")) {
      for (int i = 0; i < 3; i++) diagnostics.failure(subsystem, new IllegalStateException());
      check(diagnostics.disabled.contains(subsystem), "three failures disable " + subsystem);
    }
    diagnostics.roundReset();
    check(diagnostics.disabled.isEmpty(), "round reset failure state");
  }
}
