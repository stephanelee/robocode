import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;
import java.util.*;
import java.nio.file.*;
import java.util.regex.*;
import java.security.*;
public final class FacetSemanticTest {
  static int assertions, groups;
  static final Physics.Rules RULES = new Physics.Rules(800, 600, .1);
  static void ok(boolean b, String reason) {
    assertions++;
    if (!b)
      throw new AssertionError(reason);
  }
  static void eq(double a, double b, double tolerance, String reason) {
    ok(Math.abs(a - b) <= tolerance, reason + " actual=" + a + " expected=" + b);
  }
  static void rejects(Runnable r, String reason) {
    boolean caught = false;
    try {
      r.run();
    } catch (IllegalArgumentException | IllegalStateException e) {
      caught = true;
    }
    ok(caught, reason);
  }
  static Physics.Body body(double x, double y, double h, double speed) {
    return new Physics.Body(new Physics.V(x, y), h, speed, 0, 0, 0);
  }
  static Memory.Sample sample(int turn, Physics.Body s) {
    return new Memory.Sample(1, turn, s, 100, Memory.Quality.OBSERVED, 0);
  }
  static Memory.Observation observation(double[] f, Memory.Label l, double value, long sequence) {
    return new Memory.Observation(f, l, value, 0, sequence, 1, 1, 1, sequence, false);
  }
  static Learner.Enemy enemy(int id, double x, double y, Memory.Budget b) {
    Learner.Enemy e = new Learner.Enemy(id, b);
    e.observe(sample(0, body(x, y, Math.PI / 2, 0)), new Physics.V(600, 300));
    return e;
  }
  static Waves.Wave wave(
      long id, int owner, int start, Physics.V origin, Physics.Body target, double power) {
    return new Waves.Wave(id, owner, 0, 1, start, start, Waves.Role.INCOMING, origin, 0, power,
        power, origin.bearing(target.p()), 1, Physics.escape(power), Physics.escape(power), 1,
        new double[Learner.DIM], target, false);
  }
  static void group(String name, Runnable test) {
    test.run();
    groups++;
    System.out.println("PASS " + name);
  }
  static void angles() {
    eq(new Physics.V(0, 0).bearing(new Physics.V(1, 0)), 0, 1e-12, "east");
    eq(new Physics.V(0, 0).bearing(new Physics.V(0, 1)), Math.PI / 2, 1e-12, "north");
    eq(Physics.wrap(Math.PI), -Math.PI, 1e-12, "wrap pi");
    eq(Physics.wrap(-Physics.TAU + .1), .1, 1e-12, "wrap turn");
    eq(Physics.effective(body(0, 0, 0, -8)), Math.PI, 1e-12, "reverse effective");
    for (int sign : new int[] {-1, 1})
      for (double gf : new double[] {-1, -.2, 0, .7, 1.2})
        eq(Physics.gf(Physics.decode(gf, 6.2, sign, .4, .6), 6.2, sign, .4, .6), gf, 1e-10,
            "asymmetric GF roundtrip");
    eq(Physics.vector(Math.PI / 2).y(), 1, 1e-12, "native CCW");
  }
  static void predictor() {
    Physics.Body s = body(400, 300, 0, 8);
    Physics.Step p = Physics.step(s, new Physics.Control(8, Math.PI / 2, 0, 0), RULES);
    eq(p.body().p().x(), 408, 1e-9, "translation before rotation hypothesis");
    eq(p.body().p().y(), 300, 1e-9, "old heading");
    eq(p.body().heading(), Math.toRadians(4), 1e-9, "turn cap hypothesis");
    eq(Physics.step(s, Physics.Control.stop(), RULES).body().speed(), 0, 1e-9,
        "instantaneous STOP");
    eq(Physics.velocity(-1, 8), .5, 1e-9, "fractional reverse hypothesis");
    Physics.Body edge = body(19, 19, Math.PI, -8);
    ok(RULES.safe(Physics.step(edge, new Physics.Control(-8, Math.PI, 0, 0), RULES).body().p()),
        "first move reverse is inward");
    Physics.Body unsafe = body(19, 19, Math.PI, 8);
    ok(Physics.step(unsafe, new Physics.Control(8, Math.PI, 0, 0), RULES).wall(),
        "inward command cannot fix first translation");
    Planner.Choice c = new Planner().emergency(unsafe, RULES);
    ok(c.emergency(), "all unsafe explicit emergency");
    ok(Double.isFinite(c.cost()), "emergency finite");
    Physics.Control back = Physics.toward(body(400, 300, 0, 0), new Physics.V(300, 300), false);
    ok(back.speed() < 0 && Math.abs(back.turn()) <= Math.PI / 2, "back as front");
    Physics.Step finite = Physics.finiteStep(body(400, 300, 0, 0), 3, 0, RULES);
    eq(finite.remainingDistance(), 2, 1e-9, "finite bookkeeping hypothesis");
    Physics.Body independent = Physics.step(s, new Physics.Control(8, .1, .2, .3), RULES).body();
    eq(independent.gun(), .2, 1e-9, "decoupled gun");
    eq(independent.radar(), .3, 1e-9, "decoupled radar");
  }
  static void recordedMotion() {
    try {
      var fixtures = com.google.gson.JsonParser
                         .parseString(Files.readString(Path.of(
                             "src/test/resources/independent-contracts/motion-fixtures.json")))
                         .getAsJsonArray();
      int count = 0;
      for (var row : fixtures) {
        var f = row.getAsJsonObject();
        var a = f.getAsJsonObject("before");
        var c = f.getAsJsonObject("command");
        var b = f.getAsJsonObject("after");
        Physics.Body before = jsonBody(a);
        Physics.Rules r =
            new Physics.Rules(a.get("width").getAsDouble(), a.get("height").getAsDouble(), .1);
        var step = Physics.step(before,
            new Physics.Control(c.get("targetSpeed").getAsDouble(),
                Math.toRadians(c.get("bodyRate").getAsDouble()),
                Math.toRadians(c.get("gunRate").getAsDouble()),
                Math.toRadians(c.get("radarRate").getAsDouble())),
            r);
        var actual = jsonBody(b);
        eq(step.body().p().x(), actual.p().x(), 1e-6, "trace x");
        eq(step.body().p().y(), actual.p().y(), 1e-6, "trace y");
        eq(step.body().speed(), actual.speed(), 1e-6, "trace speed");
        eq(Math.toDegrees(Physics.wrap(step.body().heading() - actual.heading())), 0, 1e-6,
            "trace body degrees");
        eq(Math.toDegrees(Physics.wrap(step.body().gun() - actual.gun())), 0, 1e-6,
            "trace gun degrees");
        eq(Math.toDegrees(Physics.wrap(step.body().radar() - actual.radar())), 0, 1e-6,
            "trace radar degrees");
        ok(step.wall() == f.get("expectedWall").getAsBoolean(), "trace wall");
        double expectedHeat =
            c.get("fireRequested").getAsBoolean() && c.get("fireAccepted").getAsBoolean()
            ? 1.02
            : step.body().heat();
        eq(expectedHeat, actual.heat(), 1e-9, "trace heat including creation");
        count++;
      }
      ok(count == 433, "all approved Alpha motion transitions");
      var betaRows =
          Files.readAllLines(Path.of("src/test/resources/independent-contracts/Beta.jsonl"))
              .stream()
              .map(x -> com.google.gson.JsonParser.parseString(x).getAsJsonObject())
              .filter(x -> x.get("type").getAsString().equals("decision"))
              .toList();
      int beta = 0;
      for (int i = 1; i < betaRows.size(); i++) {
        var a = betaRows.get(i - 1);
        var b = betaRows.get(i);
        var predicted = Physics
                            .step(jsonBody(a),
                                new Physics.Control(a.get("targetSpeed").getAsDouble(),
                                    Math.toRadians(a.get("bodyRate").getAsDouble()),
                                    Math.toRadians(a.get("gunRate").getAsDouble()),
                                    Math.toRadians(a.get("radarRate").getAsDouble())),
                                new Physics.Rules(a.get("width").getAsDouble(),
                                    a.get("height").getAsDouble(), .1))
                            .body();
        var observed = jsonBody(b);
        eq(predicted.p().x(), observed.p().x(), 1e-6, "Beta x");
        eq(predicted.p().y(), observed.p().y(), 1e-6, "Beta y");
        eq(predicted.speed(), observed.speed(), 1e-6, "Beta speed");
        eq(Math.toDegrees(Physics.wrap(predicted.heading() - observed.heading())), 0, 1e-6,
            "Beta body");
        eq(Math.toDegrees(Physics.wrap(predicted.gun() - observed.gun())), 0, 1e-6, "Beta gun");
        eq(Math.toDegrees(Physics.wrap(predicted.radar() - observed.radar())), 0, 1e-6,
            "Beta radar");
        beta++;
      }
      ok(beta == 433, "all Beta raw motion transitions");
      count += beta;
      var validation = com.google.gson.JsonParser
                           .parseString(Files.readString(
                               Path.of("src/test/resources/independent-contracts/validation.json")))
                           .getAsJsonObject();
      try {
        for (String name : List.of("Alpha.jsonl", "Beta.jsonl", "ContractProbe.java")) {
          String digest =
              java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(
                  Files.readAllBytes(Path.of("src/test/resources/independent-contracts/" + name))));
          ok(digest.equals(validation.getAsJsonObject("hashes").get(name).getAsString()),
              "approved raw hash " + name);
        }
      } catch (java.security.NoSuchAlgorithmException ex) {
        throw new RuntimeException(ex);
      }

      List<String> raw =
          Files.readAllLines(Path.of("src/test/resources/independent-contracts/Alpha.jsonl"));
      var rows = raw.stream()
                     .map(x -> com.google.gson.JsonParser.parseString(x).getAsJsonObject())
                     .toList();
      var shot = rows.stream()
                     .filter(x -> x.get("type").getAsString().equals("fired"))
                     .findFirst()
                     .orElseThrow();
      int creation = shot.get("eventTurn").getAsInt();
      var request = rows.stream()
                        .filter(x
                            -> x.get("type").getAsString().equals("decision")
                                && x.get("turn").getAsInt() == creation - 1)
                        .findFirst()
                        .orElseThrow();
      var motion = rows.stream()
                       .filter(x
                           -> x.get("type").getAsString().equals("bulletState")
                               && x.get("turn").getAsInt() == creation)
                       .findFirst()
                       .orElseThrow();
      double power = shot.get("power").getAsDouble();
      var own = jsonBody(request);
      eq(shot.get("bulletX").getAsDouble(), own.p().x(), 1e-6, "event uses request x");
      eq(shot.get("bulletY").getAsDouble(), own.p().y(), 1e-6, "event uses request y");
      eq(shot.get("bulletDirection").getAsDouble(), request.get("gun").getAsDouble(), 1e-6,
          "request heading");
      eq(shot.get("heat").getAsDouble(), 1 + power / 5, 1e-9, "creation no cooling");
      var projected = own.p().add(Physics.vector(own.gun()).mul(Physics.bulletSpeed(power)));
      eq(motion.get("bulletX").getAsDouble(), projected.x(), 1e-6, "first motion x");
      eq(motion.get("bulletY").getAsDouble(), projected.y(), 1e-6, "first motion y");
      var w = wave(12, 0, creation - 1, own.p(), own, power);
      eq(w.outer(creation), Physics.bulletSpeed(power), 1e-9, "wave clock first radius");
      System.out.println("actual trace replay motion=" + count
          + " physical fired callbacks=1 clocks=" + (creation - 1) + "/" + creation);
    } catch (java.io.IOException ex) {
      throw new RuntimeException(ex);
    }
  }
  static Physics.Body jsonBody(com.google.gson.JsonObject a) {
    return new Physics.Body(new Physics.V(a.get("x").getAsDouble(), a.get("y").getAsDouble()),
        Math.toRadians(a.get("body").getAsDouble()), a.get("speed").getAsDouble(),
        Math.toRadians(a.get("gun").getAsDouble()), Math.toRadians(a.get("radar").getAsDouble()),
        a.get("heat").getAsDouble());
  }
  static void neighbors() {
    Memory.Budget b = new Memory.Budget();
    Memory.Store store = new Memory.Store(100, 3, Memory.Label.PRECISE_GF, b, 0);
    SplittableRandom rng = new SplittableRandom(77);
    List<Memory.Observation> rows = new ArrayList<>();
    for (int i = 0; i < 140; i++) {
      double[] f = {rng.nextDouble(), rng.nextDouble(), i % 5};
      Memory.Observation o = observation(f, Memory.Label.PRECISE_GF, i, i);
      store.add(o);
      rows.add(o);
    }
    rows = rows.subList(40, 140);
    for (double[] w : List.of(new double[] {1, 2, 3}, new double[] {3, 0, .5})) {
      double[] q = {.25, .8, 3};
      List<Memory.Neighbor> result = store.query(q, w, 17);
      List<Memory.Neighbor> truth = new ArrayList<>();
      for (var o : rows) {
        double d = 0;
        double[] f = o.features();
        for (int j = 0; j < 3; j++) d += Math.pow(w[j] * (f[j] - q[j]), 2);
        truth.add(new Memory.Neighbor(o, d));
      }
      truth.sort(Comparator.comparingDouble(Memory.Neighbor::distance)
              .thenComparingLong(n -> n.observation().sequence()));
      for (int i = 0; i < 17; i++) {
        eq(result.get(i).distance(), truth.get(i).distance(), 1e-12, "brute distance");
        ok(result.get(i).observation().sequence() == truth.get(i).observation().sequence(),
            "brute order");
      }
    }
    Memory.Store duplicate = new Memory.Store(2, 1, Memory.Label.POWER, b, 1);
    double[] f = {0};
    Memory.Observation original = observation(f, Memory.Label.POWER, 1, 10);
    f[0] = 99;
    duplicate.add(original);
    duplicate.add(observation(new double[] {0}, Memory.Label.POWER, 2, 11));
    var ns = duplicate.query(new double[] {0}, new double[] {1}, 2);
    eq(ns.get(0).weight(), 1000, 1e-9, "duplicate finite weight");
    ok(ns.get(0).observation().sequence() == 10, "duplicate tie insertion");
    eq(original.features()[0], 0, 0, "immutable copied features");
    double[] copy = original.features();
    copy[0] = 33;
    eq(original.features()[0], 0, 0, "accessor immutable");
    rejects(()
                -> duplicate.add(observation(new double[] {0}, Memory.Label.THEORETICAL_GF, 1, 12)),
        "label separation");
    rejects(()
                -> duplicate.query(new double[] {Double.NaN}, new double[] {1}, 1),
        "nonfinite rejected");
    ok(new Memory.Store(2, 1, Memory.Label.POWER, b, 1)
            .query(new double[] {0}, new double[] {1}, 8)
            .isEmpty(),
        "empty query");
    duplicate.add(observation(new double[] {0}, Memory.Label.POWER, 3, 12));
    ok(duplicate.data.peekFirst().sequence() == 11, "FIFO");
  }
  static void histories() {
    Memory.History h = new Memory.History();
    h.add(sample(1, body(0, 0, Math.toRadians(359), 0)));
    h.add(sample(3, body(2, 2, Math.toRadians(1), 2)));
    var mid = h.at(2).orElseThrow();
    eq(mid.body().p().x(), 1, 1e-9, "interpolation position");
    eq(Physics.wrap(mid.body().heading()), 0, 1e-9, "short angle");
    ok(mid.quality() == Memory.Quality.INTERPOLATED, "quality");
    h.add(sample(20, body(20, 20, 0, 0)));
    ok(h.at(10).isEmpty(), "gap no training interpolation");
    ok(h.at(21).isEmpty(), "future never observation");
    h.add(new Memory.Sample(2, 1, body(1, 1, 0, 0), 100, Memory.Quality.OBSERVED, 0));
    ok(h.samples.size() == 1, "round isolates");
    for (int i = 2; i < 800; i++)
      h.add(new Memory.Sample(2, i, body(i, i, 0, 0), 100, Memory.Quality.OBSERVED, 0));
    ok(h.samples.size() == 512, "history bounded");
  }
  static void geometry() {
    Physics.V origin = new Physics.V(0, 0);
    var crossing = Geometry.swept(origin, new Physics.V(100, -4), new Physics.V(100, 4), 99, 101);
    ok(crossing.size() == 2, "crossing wraps zero");
    ok(Geometry.width(crossing) > .35, "circular footprint width");
    ok(Geometry.swept(origin, new Physics.V(100, 0), new Physics.V(100, 0), 118, 120).size() > 0,
        "tangent included");
    ok(Geometry.swept(origin, new Physics.V(100, 0), new Physics.V(100, 0), 120, 130).isEmpty(),
        "disjoint shell");
    eq(Geometry.width(Geometry.swept(origin, new Physics.V(0, 0), new Physics.V(0, 0), 0, 1)),
        Physics.TAU, 1e-12, "source inside");
    var a = Geometry.around(0, .2);
    var half = Geometry.around(.1, .1);
    eq(Geometry.coverage(a, half), .5, 1e-9, "partial wrap");
    eq(Geometry.coverage(a, a), 1, 1e-9, "complete");
    eq(Geometry.coverage(a, Geometry.around(Math.PI, .1)), 0, 0, "disjoint");
    eq(Geometry.integrate(List.of(new Geometry.Arc(1, 1.4)), x -> 2), .8, 1e-9, "quadrature mass");
    ok(Geometry.integrate(List.of(), x -> 0) > 0, "missing interval costly");
    var w = wave(1, 1, 0, origin, body(100, 0, 0, 0), 2);
    ok(w.phase(new Physics.V(100, 0), 0) == Waves.Phase.APPROACHING, "approach");
    ok(w.phase(new Physics.V(100, 0), 6) == Waves.Phase.OVERLAPPING, "overlap");
    ok(w.phase(new Physics.V(100, 0), 8) == Waves.Phase.CENTER_PASSED, "center passed");
    ok(w.phase(new Physics.V(100, 0), 9) == Waves.Phase.FULLY_PASSED, "passed");
  }
  static void shadows() {
    var w = wave(1, 1, 0, new Physics.V(0, 0), body(100, 0, 0, 0), 2);
    var footprint = Geometry.around(0, .2);
    w.shadows.add(new Waves.Shadow(4, 5, Geometry.around(.1, .1)));
    eq(w.unshadowed(footprint, 6, Map.of()), .625, 1e-9, "inferred partial cap");
    eq(w.unshadowed(footprint, 4, Map.of()), 1, 0, "future shadow no credit");
    eq(w.unshadowed(footprint, 6, Map.of(4, 4)), 1, 0, "death before shadow invalidates");
    w.shadows.add(new Waves.Shadow(5, 5, footprint));
    eq(w.unshadowed(footprint, 6, Map.of()), .25, 1e-9, "75 percent maximum");
    Waves.Service service = new Waves.Service();
    service.add(w);
    service.death(5, 4);
    ok(w.shadows.stream().noneMatch(s -> s.bullet() == 5), "future intervals removed");
    var forecast = new Waves.Wave(2, 1, 0, 1, 0, 0, Waves.Role.FORECAST, new Physics.V(0, 0), 0, 2,
        2, 0, 1, .5, .5, .25, new double[16], body(100, 0, 0, 0), false);
    service.add(forecast);
    service.add(wave(3, 1, 0, new Physics.V(0, 0), body(100, 0, 0, 0), 2));
    ok(service.all().stream().noneMatch(x -> x.role == Waves.Role.FORECAST),
        "real replaces forecast");
  }
  static void energy() {
    Energy.Ledger l = new Energy.Ledger();
    l.scan(1, 100, .1, false);
    l.effect(new Energy.Effect(3, 2, "hit"));
    ok(l.scan(2, 100, .1, false).isEmpty(), "same-turn energy unchanged");
    ok(l.scan(3, 98, .1, false).isEmpty(), "known damage not shot");
    l.effect(new Energy.Effect(4, -1.5, "bonus"));
    ok(l.scan(4, 99.5, .1, false).isEmpty(), "known bonus not shot");
    l.effect(new Energy.Effect(4, -1.5, "bonus"));
    ok(l.duplicates == 1, "effects idempotent");
    Energy.Ledger cool = new Energy.Ledger();
    for (int i = 0; i <= 40; i++) cool.scan(i, 100, .1, false);
    var shot = cool.scan(41, 98, .1, false).orElseThrow();
    ok(shot.reliable(), "cooled consecutive reliable power");
    eq(shot.high(), 2, 1e-9, "power cost");
    var ambiguous = cool.scan(60, 95, .1, false).orElseThrow();
    ok(!ambiguous.reliable(), "missing scan uncertain");
    Energy.Ledger simultaneous = new Energy.Ledger();
    for (int i = 0; i <= 40; i++) simultaneous.scan(i, 100, .1, false);
    simultaneous.effect(new Energy.Effect(41, 2, "known"));
    eq(simultaneous.scan(41, 96, .1, false).orElseThrow().high(), 2, 1e-9,
        "simultaneous damage fire residual");
    eq(Energy.power(100, 100, 4, false), 1, 1e-9, "kill cap inverse");
    eq(Energy.power(100, 100, 10, false), 2, 1e-9, "kill cap above four");
    eq(Energy.power(100, .25, 100, false), 0, 0, "reserve");
    eq(Energy.power(100, 100, 0, false), .1, 1e-9, "zero enemy minimum");
  }
  static double field(String line, String key) {
    Matcher m = Pattern.compile("\\\"" + key + "\\\":(-?[0-9.]+)").matcher(line);
    if (!m.find())
      throw new AssertionError("missing " + key);
    return Double.parseDouble(m.group(1));
  }
  static String type(String line) {
    Matcher m = Pattern.compile("\\\"type\\\":\\\"([^\\\"]+)\\\"").matcher(line);
    if (!m.find())
      throw new AssertionError("missing type");
    return m.group(1);
  }
  static void recordedEnergy() {
    try {
      int windows = 0;
      for (String bot : List.of("Alpha", "Beta")) {
        List<String> raw =
            Files.readAllLines(Path.of("src/test/resources/energy-trace/" + bot + ".jsonl"));
        Map<Integer, Double> scans = new TreeMap<>();
        for (String line : raw)
          if (type(line).equals("scan"))
            scans.put((int) field(line, "turn"), field(line, "energy"));
        for (String line : raw)
          if ((type(line).equals("hit") || type(line).equals("hitBy"))
              && Math.abs(field(line, "power") - .5) < 1e-9) {
            int k = (int) field(line, "turn");
            double effect =
                type(line).equals("hit") ? field(line, "damage") : -3 * field(line, "power");
            eq(scans.get(k), scans.get(k - 1), 1e-9, "actual unchanged hit scan");
            eq(scans.get(k + 1), scans.get(k) - effect, 1e-9, "actual next-scan effect");
            for (boolean reverse : List.of(false, true)) {
              Energy.Ledger ledger = new Energy.Ledger();
              ledger.scan(k - 1, scans.get(k - 1), .1, false);
              if (!reverse)
                ledger.effect(new Energy.Effect(k + 1, effect, "actual"));
              ok(ledger.scan(k, scans.get(k), .1, false).isEmpty(), "actual hit-turn residual");
              if (reverse)
                ledger.effect(new Energy.Effect(k + 1, effect, "actual"));
              ok(ledger.scan(k + 1, scans.get(k + 1), .1, false).isEmpty(),
                  "actual corrected residual");
            }
            windows++;
          }
      }
      ok(windows == 4, "all four recorded isolated visibility windows");
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
  static void fire() {
    Memory.Budget b = new Memory.Budget();
    Learner.Enemy e = enemy(2, 600, 300, b);
    Physics.Body own = body(400, 300, 0, 0);
    FireControl f = new FireControl();
    f.plan = FireControl.prepare(0, 2, e, own, 0, 1);
    ok(f.valid(own, 100, e, 0), "valid current heading");
    Physics.Body wrong = body(400, 300, 0, 0);
    wrong = new Physics.Body(wrong.p(), 0, 0, .1, 0, 0);
    ok(!f.valid(wrong, 100, e, 0), "same-turn rotation never authorizes fire");
    f.submitted(true, own, 0, 10);
    ok(f.confirm(1, 1, 2).isPresent(), "confirmation matches");
    ok(f.confirm(1, 1, 2).isEmpty(), "duplicate confirm");
    ok(f.confirmed == 1, "confirmed once");
    f.plan = FireControl.prepare(0, 2, e, own, 1, 1);
    f.submitted(true, own, 1, 11);
    f.reconcile(3);
    ok(f.pending == null && f.cancelled == 1, "absent confirmation cancels");
    ok(f.confirm(2, 5, 2).isEmpty() && f.unmatched == 1, "late unmatched diagnostic");
    f.plan = FireControl.prepare(0, 2, e, own, 0, 1);
    ok(!f.valid(own, 100, e, 3), "expiry");
    e.alive = false;
    ok(!f.valid(own, 100, e, 1), "death cancellation");
  }
  static void chronology() {
    Memory.Budget b = new Memory.Budget();
    Learner.Enemy e = enemy(1, 500, 300, b);
    Physics.Body own = body(400, 300, 0, 0);
    double[] features = new double[16];
    Waves.Wave w = new Waves.Wave(4, 0, 1, 1, 0, 0, Waves.Role.VIRTUAL, own.p(), 0, 2, 2, 0, 1, .4,
        .6, 1, features, e.latest.body(), false);
    w.ratingEligible = true;
    w.predictions.put("analytic", new Waves.Prediction("analytic", 4, 0, 0, 2, 0));
    w.predictions.put("long", new Waves.Prediction("long", 4, 0, .2, 2, 0));
    Learner.trainGun(e, w, sample(8, body(500, 300, 0, 0)), 1);
    ok(e.longGun.size() == 1 && e.ratings.get("analytic").count == 1, "predictions before labels");
    eq(e.ratings.get("analytic").raw, 1, 1e-9, "perfect prediction");
    ok(e.ratings.get("long").raw < 1, "frozen error rated");
    Learner.trainGun(e, w, sample(8, body(500, 300, 0, 0)), 2);
    ok(e.longGun.size() == 1, "no double training");
    ok(e.hits == 0, "virtual not physical hit");
    Core c = new Core(RULES, new Diagnostics(), 4);
    c.roundReset(1);
    Learner.Enemy stored = c.enemy(1);
    stored.longGun.add(
        observation(new double[Learner.LONG.length], Memory.Label.PRECISE_GF, .2, 1));
    c.waves.add(w);
    c.fire.pending = new FireControl.Pending(1, 1, 2, 1, 0, own.p());
    c.roundReset(2);
    ok(c.enemy(1).longGun.size() == 1, "round learning retained");
    ok(c.waves.all().isEmpty() && c.fire.pending == null, "round pending cleared");
    Core cold = new Core(RULES, new Diagnostics(), 4);
    ok(cold.enemy(1).longGun.size() == 0, "new game cold");
    ScannedBotEvent scan = new ScannedBotEvent(1, 0, 1, 100, 600, 300, 0, 0);
    var intent = c.decide(own, 100, 1, 1, List.of(scan, scan), List.of(), 0);
    ok(c.enemy(1).observedTogether == 1, "batch duplicate scan");
    ok(Double.isFinite(intent.gunTurn()), "finite intent");
    var second = c.decide(
        own, 100, 2, 1, List.of(new ScannedBotEvent(2, 0, 1, 100, 600, 300, 0, 0)), List.of(), 0);
    ok(second.fire(), "fresh consecutive scan does not deadlock prepared shot");
  }
  static void surfing() {
    Physics.Body self = body(600, 300, Math.PI / 2, 8);
    Memory.Budget b = new Memory.Budget();
    Learner.Enemy e = enemy(1, 400, 300, b);
    Map<Integer, Learner.Enemy> enemies = Map.of(1, e);
    Waves.Wave first = wave(1, 1, -8, new Physics.V(400, 300), self, 2);
    Planner p = new Planner();
    var one = p.choose(self, 0, List.of(first), enemies, RULES, Map.of());
    ok(one.complete() && Double.isFinite(one.cost()), "one wave complete");
    ok(p.steps <= 2400, "quota");
    Planner pair = new Planner();
    Waves.Wave second = wave(2, 1, -10, new Physics.V(350, 450), self, 2);
    var both = pair.choose(self, 0, List.of(first, second), enemies, RULES, Map.of());
    ok(pair.steps <= 2400, "two-wave quota");
    ok(!both.emergency() && both.cost() > one.cost(),
        "second wave contributes cost under corrected physics");
    System.out.println("two-wave corrected fixture actions=" + one.action() + "/" + both.action()
        + " costs=" + one.cost() + "/" + both.cost());
    Planner limited = new Planner();
    var far = wave(3, 1, 0, new Physics.V(0, 0), body(100000, 100000, 0, 0), 2);
    var path = limited.simulate(body(600, 300, 0, 0), 0, far, -1, e, RULES, Map.of());
    ok(limited.steps <= 160, "trajectory capped");
    ok(Double.isFinite(path.cost()), "truncation finite");
  }
  static void meleeRadar() {
    rejects(Melee::requireApproval, "review gate locked");
    Memory.Budget b = new Memory.Budget();
    Learner.Enemy e = enemy(1, 200, 200, b);
    Melee m = new Melee(5);
    Physics.Body own = body(400, 300, 0, 0);
    Physics.Control cmd = m.move(own, 100, 0, List.of(e), RULES, new Planner());
    ok(RULES.safe(Physics.step(own, cmd, RULES).body().p()), "melee safe first translation");
    m.move(body(18, 18, Math.PI, 8), 100, 1, List.of(e), RULES, new Planner());
    ok(m.unsafeFallbacks == 1, "all unsafe explicit fallback");
    Memory.Observation displacement = new Memory.Observation(
        new double[16], Memory.Label.DISPLACEMENT, 1, 0, 1, 1, 0, 1, 1, true);
    var projected =
        Melee.project(displacement, body(500, 300, 0, 0), 1, new Physics.V(400, 300), 2, RULES)
            .orElseThrow();
    eq(projected.x(), 400 + 1400.0 / 13, 1e-4, "displacement fixed point independent solution");
    rejects(()
                -> Melee.project(observation(new double[16], Memory.Label.PRECISE_GF, 1, 2), own, 1,
                    own.p(), 2, RULES),
        "label typed projection");
    Radar radar = new Radar();
    eq(radar.turn(own, List.of(e), 0, 2), Math.PI / 4, 1e-12, "unknown acquisition");
    ok(Double.isFinite(radar.turn(own, List.of(e), 20, 1)), "stale sweep");
    e.alive = false;
    ok(Double.isFinite(radar.turn(own, List.of(e), 20, 0)), "last survivor transition defined");
  }
  static void packageBoundary() {
    rejects(() -> Facet.validateServer(null), "no implicit interactive default");
    rejects(() -> Facet.validateServer("ws://localhost:7654"), "interactive port denied");
    rejects(() -> Facet.validateServer("http://localhost:9000"), "websocket only");
    ok(Facet.validateServer("ws://localhost:12345").getPort() == 12345,
        "isolated endpoint parse only");
    Diagnostics d = new Diagnostics();
    for (int i = 0; i < 3; i++) d.failure("gun", new IllegalStateException());
    ok(d.disabled.contains("gun"), "three consecutive failures disable");
    d.roundReset();
    ok(!d.disabled.contains("gun"), "round enables analytic recovery");
    // Prior integrates to one over its theoretical reachable domain.
    Memory.Budget b = new Memory.Budget();
    Learner.Enemy e = enemy(1, 400, 300, b);
    var w = wave(1, 1, 0, new Physics.V(400, 300), body(600, 300, Math.PI / 2, 8), 2);
    Geometry.Density prior = Learner.prepareDensity(e, w);
    double sum = 0, escape = Physics.escape(2);
    for (int i = 0; i < 20000; i++)
      sum += prior.at(w.base - escape + (i + .5) * 2 * escape / 20000) * 2 * escape / 20000;
    eq(sum, 1, 1e-5, "normalized independent prior");
    double bound = Geometry.integrate(List.of(new Geometry.Arc(0, 6)), x -> 0);
    ok(bound >= 60000, "quadrature-cap conservative bound");
  }
  static void deterministicAndExport() {
    Core a = new Core(RULES, new Diagnostics(), 112), b = new Core(RULES, new Diagnostics(), 112);
    a.roundReset(1);
    b.roundReset(1);
    Physics.Body own = body(400, 300, 0, 0);
    for (int t = 1; t <= 240; t++) {
      double x = 550 + 30 * Math.sin(t * .03), y = 300 + 30 * Math.cos(t * .03);
      ScannedBotEvent scan = new ScannedBotEvent(t, 0, 1, 100, x, y, 90, 2);
      Core.Intent ia = a.decide(own, 100, t, 1, List.of(scan), List.of(), 0),
                  ib = b.decide(own, 100, t, 1, List.of(scan), List.of(), 0);
      ok(ia.equals(ib), "deterministic identical legal event stream");
      ok(a.log.observations.get("predictorSteps")[3] <= 2400, "whole turn predictor cap");
      Physics.Control c =
          new Physics.Control(ia.move().speed(), ia.move().turn(), ia.gunTurn(), ia.radarTurn());
      own = Physics.step(own, c, RULES).body();
      ok(RULES.safe(own.p()), "long synthetic movement keeps safe bounds");
    }
    ok(a.enemy(1).crossings > 16, "virtual crossings activate learning");
    ok(a.enemy(1).longGun.size() > 0, "stream actually trains");
    try {
      java.io.ByteArrayOutputStream first = new java.io.ByteArrayOutputStream(),
                                    second = new java.io.ByteArrayOutputStream();
      ModelSnapshot.write(a, first);
      ModelSnapshot.write(b, second);
      ok(Arrays.equals(first.toByteArray(), second.toByteArray()),
          "deterministic export of own derived model");
      java.io.DataInputStream in =
          new java.io.DataInputStream(new java.io.ByteArrayInputStream(first.toByteArray()));
      ok(in.readUTF().equals("FACET-OWN-GAME-EXPORT") && in.readInt() == 1,
          "export schema identity");
      ok(first.size() < ModelSnapshot.MAX_BYTES, "export bounded");
    } catch (java.io.IOException e) {
      throw new RuntimeException(e);
    }
    Physics.beginWork(2);
    Physics.step(own, Physics.Control.stop(), RULES);
    Physics.step(own, Physics.Control.stop(), RULES);
    boolean limited = false;
    try {
      Physics.step(own, Physics.Control.stop(), RULES);
    } catch (Physics.WorkLimit expected) {
      limited = true;
    } finally {
      Physics.endWork();
    }
    ok(limited, "hard predictor cap");
  }
  static void stress() {
    long begin = System.nanoTime();
    Memory.Budget b = new Memory.Budget();
    Learner.Enemy e = enemy(1, 400, 300, b);
    long sequence = 0;
    for (Memory.Store s : List.of(e.longGun, e.displacement, e.recent[0], e.recent[1], e.recent[2],
             e.simple, e.context, e.recentHit, e.visits, e.powers))
      for (int i = 0; i < s.capacity; i++) {
        double[] f = new double[s.dimension];
        Arrays.fill(f, (i % 13) / 13.0);
        s.add(observation(f, s.label, (i % 9 - 4) / 4.0, ++sequence));
      }
    ok(b.bytes < Memory.Budget.LIMIT, "full stores under memory quota");
    Waves.Service waves = new Waves.Service();
    Physics.Body own = body(600, 300, Math.PI / 2, 8);
    for (int i = 0; i < 300; i++) waves.add(wave(i + 1, 1, -8, new Physics.V(400, 300), own, 2));
    ok(waves.all().size() == 256 && waves.realOverflow == 44, "wave cap");
    Planner planner = new Planner();
    List<Long> timings = new ArrayList<>();
    for (int i = 0; i < 20; i++) {
      long start = System.nanoTime();
      var choice = planner.choose(own, 0, waves.all(), Map.of(1, e), RULES, Map.of());
      timings.add(System.nanoTime() - start);
      ok(planner.steps <= 2400, "stress work quota");
      ok(Double.isFinite(choice.cost()), "stress finite");
    }
    timings.sort(Long::compare);
    System.out.println("offline stress: payloadBytes=" + b.bytes + " waves=" + waves.all().size()
        + " iterations=20 p50ns=" + timings.get(10) + " maxns=" + timings.get(19)
        + " totalns=" + (System.nanoTime() - begin));
    // Force global eviction without allocating 128 MiB: accounting-boundary test.
    b.bytes = Memory.Budget.LIMIT + 1;
    b.enforce();
    ok(b.evictions > 0 && b.bytes <= Memory.Budget.LIMIT, "global eviction enforcement");
  }
  public static void main(String[] args) {
    group("actual motion and shot clocks", FacetSemanticTest::recordedMotion);
    group("review regressions", ReviewRegressionTest::run);
    group("native angles and GF", FacetSemanticTest::angles);
    group("native predictor invariants", FacetSemanticTest::predictor);
    group("exact finite neighbors", FacetSemanticTest::neighbors);
    group("history quality and lifecycle", FacetSemanticTest::histories);
    group("circular swept geometry", FacetSemanticTest::geometry);
    group("shadow validity", FacetSemanticTest::shadows);
    group("energy and power", FacetSemanticTest::energy);
    group("actual official energy replay", FacetSemanticTest::recordedEnergy);
    group("fire transaction synthetic", FacetSemanticTest::fire);
    group("learning chronology and integration", FacetSemanticTest::chronology);
    group("two-wave planner", FacetSemanticTest::surfing);
    group("gated melee and radar", FacetSemanticTest::meleeRadar);
    group("package boundary and risk bounds", FacetSemanticTest::packageBoundary);
    group("determinism, online learning and export", FacetSemanticTest::deterministicAndExport);
    group("bounded full-store stress", FacetSemanticTest::stress);
    System.out.println("PASS groups=" + groups + " assertions=" + assertions
        + " motion/shot-clock fixture replay=PASS live adapter review=PENDING battles=0");
  }
}
