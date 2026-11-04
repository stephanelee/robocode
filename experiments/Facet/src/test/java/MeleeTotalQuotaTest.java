import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;
import java.util.*;
/** Whole-decision instrumentation with eligible full views, not strength evidence. */
final class MeleeTotalQuotaTest {
  static void ok(boolean v, String s) {
    FacetSemanticTest.ok(v, s);
  }
  static ScannedBotEvent scan(int t, int id) {
    return new ScannedBotEvent(t, 99, id, 100, 500 + id * 100, 300, 0, 0);
  }
  static void bounded(Core c, String phase) {
    var w = Memory.displacementWork.get();
    ok(w.returned <= 80 && w.projections <= 80, phase + " whole decision total80");
    ok(w.iterations <= 12 * w.projections, phase + " whole decision iterations");
    ok(w.returned == c.melee.decisionReturned && w.projections == c.melee.decisionSamples,
        phase + " no hidden cohort queries/projections");
    ok(w.queries == c.melee.decisionQueries, phase + " independent query counts");
    ok(c.log.observations.get("meleeTotalReturned")[3] <= 80
            && c.log.observations.get("meleeTotalProjections")[3] <= 80,
        phase + " whole-decision telemetry bounded");
    System.out.println("total-quota " + phase + " queries=" + w.queries + " returned=" + w.returned
        + " projections=" + w.projections + " iterations=" + w.iterations
        + " distanceEvaluations=" + w.distances);
  }
  static Waves.Wave physical(Core c, int id) {
    return c.waves.all()
        .stream()
        .filter(w -> w.role == Waves.Role.OUTGOING && w.bullet == id)
        .findFirst()
        .orElseThrow();
  }
  static void cross(Core c, Learner.Enemy e, Waves.Wave w) {
    int t = w.start
        + (int) Math.ceil(w.origin.distance(w.targetOrigin.p()) / Physics.bulletSpeed(w.powerHi));
    for (int tick : new int[] {t - 1, t})
      e.observe(
          new Memory.Sample(1, tick, w.targetOrigin, 100, Memory.Quality.OBSERVED, 0), w.origin);
    c.resolve(w.targetOrigin, t);
    ok(w.labeled, "confirmed physical wave learns at legal observed crossing");
    int n = e.crossings;
    c.resolve(w.targetOrigin, t);
    ok(e.crossings == n, "resolved waves cannot train twice");
  }
  static void run() {
    var c = new Core(new Physics.Rules(800, 600, .1), new Diagnostics(), 49667);
    c.roundReset(1);
    for (int id = 1; id <= 2; id++) {
      var e = c.enemy(id);
      e.crossings = 16;
      e.selected = "recent";
      for (var s : List.of(e.displacement, e.recentDisplacement[0], e.recentDisplacement[1],
               e.recentDisplacement[2]))
        for (int j = 0; j < s.capacity; j++) s.add(MeleeCorrectionTest.displacement(0, 0, j));
    }
    var own = FacetSemanticTest.body(400, 300, 0, 0);
    ok(!c.decide(own, 100, 0, 2, List.of(scan(0, 1), scan(0, 2)), List.of(), 99).fire(),
        "ordinary no-prior-plan decision");
    bounded(c, "no-fire");
    ok(c.fire.plan != null, "filled eligible views prepare combined plan");
    own = FollowupTelemetryFixture.aligned(c.fire.plan);
    ok(c.decide(own, 100, 1, 2, List.of(scan(1, 1), scan(1, 2)), List.of(), 99).fire(),
        "aligned cold valid physical preparation with eligible crossings");
    bounded(c, "physical-preparation");
    ok(c.shotContext.predictions.isEmpty(), "melee physical preparation defers entire duel cohort");
    ok(c.fire.plan.evidence().contains("gun=combinedmelee"), "combined physical metadata retained");
    var e = c.enemy(c.shotContext.target);
    int target = e.id;
    var bullet = FollowupTelemetryFixture.submit(c, own, 1, 901);
    var callback = new BulletFiredEvent(2, bullet);
    var hot = new Physics.Body(own.p(), 0, 0, own.gun(), 0, 1);
    ok(!c.decide(hot, 100, 2, 2, List.of(callback, scan(2, 1), scan(2, 2)), List.of(bullet), 99)
            .fire(),
        "confirmation hot decision does not fire");
    bounded(c, "confirmation");
    var meleeWave = physical(c, 901);
    ok(c.fire.confirmed == 1 && meleeWave.start == 1 && meleeWave.creationTurn == 2
            && meleeWave.firstMotionTurn == 2,
        "confirmed accounting and physical chronology retained");
    ok(meleeWave.predictions.isEmpty() && !meleeWave.ratingEligible,
        "empty melee predictions cannot be a rating cohort");
    long queries = Memory.displacementWork.get().queries;
    long returned = Memory.displacementWork.get().returned;
    long projections = Memory.displacementWork.get().projections;
    long iterations = Memory.displacementWork.get().iterations;
    int crossings = e.crossings;
    long version = e.displacement.version;
    c.observe(hot, 100, 2, 2, List.of(callback), List.of(bullet), 99, false);
    Physics.endWork();
    ok(c.fire.confirmed == 1 && e.crossings == crossings && e.displacement.version == version,
        "duplicate physical callback adds neither confirmation nor training");
    ok(Memory.displacementWork.get().queries == queries
            && Memory.displacementWork.get().returned == returned
            && Memory.displacementWork.get().projections == projections
            && Memory.displacementWork.get().iterations == iterations,
        "duplicate callback adds no queries or projection work");
    cross(c, e, meleeWave);
    ok(e.crossings > crossings && e.displacement.version > version,
        "melee learning and typed long store retained");
    ok(e.startupRating.count == 0 && e.ratings.values().stream().allMatch(r -> r.count == 0),
        "melee labels invent no analytic or empty-cohort ratings");
    ok(e.longGun.size() == 0 && Arrays.stream(e.recent).allMatch(s -> s.size() == 0),
        "melee labels remain separate from precise GF");
    // Resolve remaining legal melee virtual crossings before isolating transition retention.
    for (var w : List.copyOf(c.waves.all()))
      if (w.target == target && w.role == Waves.Role.VIRTUAL && !w.labeled && !w.cancelled)
        cross(c, e, w);
    version = e.displacement.version;
    long[] recentVersions = Arrays.stream(e.recentDisplacement).mapToLong(s -> s.version).toArray();
    int transition = e.latest.turn() + 1;
    for (var other : c.enemies.values())
      if (other != e)
        other.alive = false;
    c.decide(own, 100, transition, 1, List.of(), List.of(), 99);
    ok(c.fire.plan == null, "2-to-1 transition requires fresh survivor scan");
    c.decide(own, 100, transition + 1, 1, List.of(scan(transition + 1, target)), List.of(), 99);
    ok(c.fire.plan != null && c.fire.plan.evidence().contains("gun=recent")
            && c.fire.plan.evidence().contains("label=DISPLACEMENT"),
        "fresh survivor selector resumes");
    ok(e.displacement.version == version
            && Arrays.equals(recentVersions,
                Arrays.stream(e.recentDisplacement).mapToLong(s -> s.version).toArray()),
        "transition preserves long and all recent stores");
    own = FollowupTelemetryFixture.aligned(c.fire.plan);
    int request = transition + 2;
    ok(c.decide(own, 100, request, 1, List.of(scan(request, target)), List.of(), 99).fire(),
        "survivor physical preparation resumes");
    ok(c.shotContext.predictions.keySet().equals(Set.of("analytic", "long", "recent")),
        "eligible survivor freezes common three-gun cohort");
    ok(c.shotContext.predictions.values().stream().allMatch(
           p -> p.version() == e.displacement.version),
        "survivor predictions freeze the typed displacement store version");
    ok(Memory.displacementWork.get().returned >= 175,
        "instrumentation sees full long160 and recent15 queries in survivor");
    var second = FollowupTelemetryFixture.submit(c, own, request, 902);
    c.observe(own, 100, request + 1, 1, List.of(new BulletFiredEvent(request + 1, second)),
        List.of(second), 99, false);
    Physics.endWork();
    var survivorWave = physical(c, 902);
    ok(survivorWave.ratingEligible && survivorWave.predictions.size() == 3,
        "confirmation retains resumed common cohort");
    cross(c, e, survivorWave);
    ok(c.fire.confirmed == 2 && e.comparisonEligible && e.startupRating.count == 1
            && e.ratings.values().stream().allMatch(r -> r.count == 1),
        "survivor confirmed common cohort scores all guns exactly once");
    ok(e.selected.equals("recent"), "minimum8 selector gate preserved");
    var cycle = new Core(new Physics.Rules(800, 600, .1), new Diagnostics(), 49667);
    cycle.roundReset(1);
    var moving = cycle.enemy(1);
    moving.crossings = 16;
    moving.observe(FacetSemanticTest.sample(40, FacetSemanticTest.body(500, 300, 0, 0)),
        new Physics.V(100, 300));
    for (var s : List.of(moving.displacement, moving.recentDisplacement[0],
             moving.recentDisplacement[1], moving.recentDisplacement[2]))
      for (int j = 0; j < s.capacity; j++) s.add(MeleeCorrectionTest.displacement(3, 0, j));
    ok(!cycle.decide(FacetSemanticTest.body(100, 300, 0, 0), 100, 40, 2, List.of(), List.of(), 99)
                .fire()
            && cycle.fire.plan == null,
        "ordinary cycling Core decision conservatively has no fire plan");
    bounded(cycle, "cycling-no-fire");
    ok(Memory.displacementWork.get().returned == 80
            && Memory.displacementWork.get().projections == 80,
        "whole Core reaches actual total80");
  }
  public static void main(String[] args) {
    run();
    System.out.println("PASS focused whole-Core TOTAL80 and survivor cohort proof");
  }
}
