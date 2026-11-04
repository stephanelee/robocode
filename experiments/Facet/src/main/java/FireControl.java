import java.util.*;
final class FireControl {
  /** Forecast provenance and predictions are immutable; physical training context is separate. */
  record Cohort(String selected, int target, boolean meleeOrigin, int preparedTurn,
      int expectedOriginTurn, long sourceWave, Physics.V origin, Physics.Body targetOrigin,
      double[] features, double positive, double negative, double quality, int missingFeatures,
      Map<String, Waves.Prediction> predictions) {
    Cohort {
      features = features.clone();
      Physics.finite(features);
      Physics.finite(positive, negative, quality);
      predictions = Collections.unmodifiableMap(new LinkedHashMap<>(predictions));
      if (preparedTurn < 0 || expectedOriginTurn < preparedTurn || sourceWave < -1
          || positive < 0 || negative < 0 || quality <= 0 || quality > 1)
        throw new IllegalArgumentException("prepared context bounds");
      if (features.length != 0 && features.length != Learner.DIM
          || !(predictions.keySet().equals(Set.of("analytic"))
              || predictions.keySet().equals(Set.of("analytic", "long", "recent")))
          || !predictions.containsKey(selected))
        throw new IllegalArgumentException("prepared cohort schema");
      for (var entry : predictions.entrySet()) {
        var p = entry.getValue();
        if (!Set.of("analytic", "long", "recent").contains(entry.getKey())
            || !entry.getKey().equals(p.gun()) || p.wave() != sourceWave
            || p.turn() != preparedTurn)
          throw new IllegalArgumentException("prepared prediction provenance");
        Physics.finite(p.angle(), p.power());
      }
    }
    @Override public double[] features() { return features.clone(); }
    static Cohort from(Waves.Wave wave, String selected, int turn) {
      return new Cohort(selected, wave.target, wave.melee, turn, wave.start, wave.id, wave.origin,
          wave.targetOrigin, wave.features, wave.positive, wave.negative, wave.originQuality,
          wave.missingFeatures, wave.predictions);
    }
    static Cohort analytic(int target, boolean meleeOrigin, int turn, Physics.V origin,
        Physics.Body targetOrigin, double angle, double power, long version) {
      return new Cohort("analytic", target, meleeOrigin, turn, turn + 1, -1, origin, targetOrigin,
          new double[0], Physics.escape(power), Physics.escape(power), .5, 0xffff,
          Map.of("analytic", new Waves.Prediction("analytic", -1, turn, angle, power, version)));
    }
    Map<String, Waves.Prediction> bind(long physicalWave) {
      Map<String, Waves.Prediction> result = new LinkedHashMap<>();
      for (var p : predictions.values())
        result.put(p.gun(), new Waves.Prediction(
            p.gun(), physicalWave, p.turn(), p.angle(), p.power(), p.version()));
      return Collections.unmodifiableMap(result);
    }
    String evidenceRow() {
      return "preparedTurn=" + preparedTurn + " expectedOriginTurn=" + expectedOriginTurn
          + " preparedWave=" + sourceWave + " selected=" + selected + " origin=" + origin
          + " target=" + target + " meleeOrigin=" + meleeOrigin
          + " targetOrigin=" + targetOrigin + " positive=" + positive + " negative=" + negative
          + " quality=" + quality + " mask=" + missingFeatures
          + " features=" + Arrays.toString(features) + " predictions=" + predictions;
    }
    @Override public String toString() { return evidenceRow(); }
  }
  record Policy(int population, int target, double range, double targetEnergy, int observed) {
    Policy {
      Physics.finite(range, targetEnergy);
      if (population < 1 || range < 0)
        throw new IllegalArgumentException("fire policy");
    }
    boolean melee() {
      return population > 1;
    }
  }
  record Plan(double angle, double power, int target, int created, int expires, double tolerance,
      Physics.V origin, int observationTurn, Physics.V targetPosition, long version,
      Physics.Body targetBody, Policy policy, String evidence, Cohort cohort) {
    Plan {
      if (cohort != null && (cohort.target() != target || cohort.preparedTurn() != created
          || cohort.origin().distance(origin) > 1e-6
          || Math.abs(Physics.wrap(cohort.predictions().get(cohort.selected()).angle() - angle))
              > 1e-12
          || cohort.predictions().values().stream()
              .anyMatch(p -> Math.abs(p.power() - power) > 1e-12)))
        throw new IllegalArgumentException("plan/cohort mismatch");
    }
    Plan(double angle, double power, int target, int created, int expires, double tolerance,
        Physics.V origin, int observationTurn, Physics.V targetPosition, long version,
        Physics.Body targetBody, Policy policy, String evidence) {
      this(angle, power, target, created, expires, tolerance, origin, observationTurn,
          targetPosition, version, targetBody, policy, evidence, null);
    }
    Plan withEvidence(String value) {
      return new Plan(angle, power, target, created, expires, tolerance, origin, observationTurn,
          targetPosition, version, targetBody, policy, value, cohort);
    }
    Plan withCohort(Cohort value) {
      return new Plan(angle, power, target, created, expires, tolerance, origin, observationTurn,
          targetPosition, version, targetBody, policy, evidence, value);
    }
    String evidenceRow() {
      return "plan=" + created + ":" + version + ":" + target + " target=" + target
          + " power=" + power + " angle=" + angle + " policy=" + policy + " " + evidence;
    }
  }
  record Pending(long sequence, int requestedTurn, double power, int target, double heading,
      Physics.V origin, Policy policy, String evidence, Cohort cohort) {
    Pending(long sequence, int requestedTurn, double power, int target, double heading,
        Physics.V origin, Policy policy, String evidence) {
      this(sequence, requestedTurn, power, target, heading, origin, policy, evidence, null);
    }
    Pending(long sequence, int requestedTurn, double power, int target, double heading,
        Physics.V origin) {
      this(sequence, requestedTurn, power, target, heading, origin, null, "fixture");
    }
  }
  Plan plan;
  Pending pending;
  long requested, accepted, confirmed, cancelled, rejected, unmatched;
  final Set<Integer> confirmedIds = new LinkedHashSet<>();
  boolean valid(Physics.Body own, double energy, Learner.Enemy target, int turn) {
    return valid(own, energy, target, turn, new Physics.Rules(800, 600, .1));
  }
  boolean valid(
      Physics.Body own, double energy, Learner.Enemy target, int turn, Physics.Rules rules) {
    if (plan == null || pending != null || target == null || !target.alive || target.latest == null)
      return false;
    int elapsed = Math.max(0, target.latest.turn() - plan.observationTurn);
    Physics.Body predicted =
        new Physics.Body(Physics.constantPosition(plan.targetBody, elapsed, rules),
            plan.targetBody.heading(), plan.targetBody.speed(), 0, 0, 0);
    double drift = Math.abs(
               Physics.wrap(Physics.intercept(own.p(), target.latest.body(), plan.power, rules)
                   - Physics.intercept(own.p(), predicted, plan.power, rules))),
           half = Math.asin(
               Math.min(1, 18 / Math.max(18, own.p().distance(target.latest.body().p()))));
    return drift <= half && target.id == plan.target && turn <= plan.expires && turn >= plan.created
        && turn - target.latest.turn() <= 2
        && target.latest.body().p().distance(plan.targetPosition)
        <= 16.01 * Math.max(1, target.latest.turn() - plan.observationTurn)
        && own.p().distance(plan.origin) <= 16.01 && own.heat() == 0 && energy > plan.power
        && plan.power <= Energy.power(plan.policy.melee()
                                 ? plan.policy.range()
                                 : own.p().distance(target.latest.body().p()),
                             energy, target.latest.energy(), plan.policy.melee())
            + 1e-12
        && Math.abs(Physics.wrap(own.gun() - plan.angle)) <= plan.tolerance;
  }
  void submitted(boolean result, Physics.Body own, int turn, long sequence) {
    requested++;
    if (result) {
      accepted++;
      pending = new Pending(sequence, turn, plan.power, plan.target, own.gun(), own.p(),
          plan.policy, plan.evidenceRow(), plan.cohort);
    } else
      rejected++;
    plan = null;
  }
  Optional<Pending> confirm(int id, int turn, double power) {
    if (!confirmedIds.add(id))
      return Optional.empty();
    while (confirmedIds.size() > 4096) confirmedIds.remove(confirmedIds.iterator().next());
    confirmed++;
    Pending p = pending;
    if (p == null || Math.abs(p.power - power) > .001 || turn != p.requestedTurn + 1) {
      pending = null;
      unmatched++;
      return Optional.empty();
    }
    pending = null;
    return Optional.of(p);
  }
  Optional<Pending> confirm(int id, int turn, double power, Physics.V origin, double heading) {
    Pending p = pending;
    if (p != null
        && (p.origin.distance(origin) > 1e-6
            || Math.abs(Physics.wrap(p.heading - heading)) > 1e-7)) {
      pending = null;
    }
    return confirm(id, turn, power);
  }
  void reconcile(int turn) {
    if (pending != null && turn > pending.requestedTurn + 1) {
      cancelled++;
      pending = null;
    }
  }
  void reset() {
    plan = null;
    pending = null;
    confirmedIds.clear();
  }
  static Plan prepare(
      double angle, double power, Learner.Enemy target, Physics.Body own, int turn, long version) {
    return prepare(angle, power, target, own, turn, version,
        new Policy(1, target.id, own.p().distance(target.latest.body().p()), target.latest.energy(),
            target.latest.turn()));
  }
  static Plan prepare(double angle, double power, Learner.Enemy target, Physics.Body own, int turn,
      long version, Policy policy) {
    if (policy.target() != target.id)
      throw new IllegalArgumentException("policy target");
    double range = own.p().distance(target.latest.body().p()),
           half = Math.asin(Math.min(1, 18 / Math.max(18, range)));
    double tolerance = Physics.clamp(half * .5, Math.toRadians(.05), Math.toRadians(.5));
    return new Plan(Physics.norm(angle), power, target.id, turn, turn + 2, tolerance, own.p(),
        target.latest.turn(), target.latest.body().p(), version, target.latest.body(), policy,
        "unannotated");
  }
}
