import java.util.*;
final class Waves {
  enum Role { OUTGOING, INCOMING, VIRTUAL, FORECAST }
  enum Phase { APPROACHING, OVERLAPPING, CENTER_PASSED, FULLY_PASSED }
  static final class Wave {
    final long id;
    final int owner, target, round, start, end, sign;
    final Role role;
    final Physics.V origin;
    final double originError, powerLo, powerHi, base, positive, negative, confidence;
    final double[] features;
    final Physics.Body targetOrigin;
    final boolean melee;
    int creationTurn = -1, firstMotionTurn = -1;
    double originQuality = 1;
    int missingFeatures;
    boolean ratingEligible;
    double actualAngle = Double.NaN;
    Memory.Sample center;
    Integer bullet;
    boolean labeled, aimLabeled, contact, hit, cancelled;
    final List<Geometry.Arc> crossing = new ArrayList<>();
    final List<Shadow> shadows = new ArrayList<>();
    final Map<String, Prediction> predictions = new LinkedHashMap<>();
    Wave(long id, int owner, int target, int round, int start, int end, Role role, Physics.V origin,
        double originError, double lo, double hi, double base, int sign, double pos, double neg,
        double confidence, double[] features, Physics.Body targetOrigin, boolean melee) {
      this.id = id;
      this.owner = owner;
      this.target = target;
      this.round = round;
      this.start = start;
      this.end = end;
      this.role = role;
      this.origin = origin;
      this.originError = originError;
      powerLo = lo;
      powerHi = hi;
      this.base = base;
      this.sign = sign;
      positive = pos;
      negative = neg;
      this.confidence = confidence;
      this.features = features.clone();
      this.targetOrigin = targetOrigin;
      this.melee = melee;
    }
    double inner(int t) {
      return Math.max(0, Physics.bulletSpeed(powerHi) * Math.max(0, t - end) - originError);
    }
    double outer(int t) {
      return Physics.bulletSpeed(powerLo) * Math.max(0, t - start) + originError;
    }
    Phase phase(Physics.V p, int t) {
      double d = origin.distance(p);
      if (inner(t) > d + 18)
        return Phase.FULLY_PASSED;
      if (inner(t) > d)
        return Phase.CENTER_PASSED;
      if (outer(t) >= d - 18)
        return Phase.OVERLAPPING;
      return Phase.APPROACHING;
    }
    List<Geometry.Arc> footprint(Physics.V from, Physics.V to, int t) {
      return Geometry.swept(origin, from, to, inner(t - 1), outer(t));
    }
    double unshadowed(List<Geometry.Arc> arcs, int contactTurn, Map<Integer, Integer> deaths) {
      List<Geometry.Arc> valid = new ArrayList<>();
      for (Shadow s : shadows)
        if (s.turn <= contactTurn && deaths.getOrDefault(s.bullet, Integer.MAX_VALUE) > s.turn)
          valid.addAll(s.arcs);
      double cap = role == Role.INCOMING ? .75 : 0;
      return 1 - cap * Geometry.coverage(arcs, Geometry.merge(valid));
    }
  }
  record Prediction(String gun, long wave, int turn, double angle, double power, long version) {}
  record Shadow(int bullet, int turn, List<Geometry.Arc> arcs) {}
  record Bullet(int id, Physics.V position, double heading, double power, int observedTurn) {}
  static final class Service {
    final Map<Integer, ArrayList<Wave>> waves = new TreeMap<>();
    final Map<Integer, Bullet> bullets = new TreeMap<>();
    final Map<Integer, Integer> deaths = new HashMap<>();
    long nextId = 1, evictions, unmatched;
    List<Wave> all() {
      return waves.values().stream().flatMap(Collection::stream).toList();
    }
    void add(Wave w) {
      ArrayList<Wave> list = waves.computeIfAbsent(w.owner, k -> new ArrayList<>());
      if (w.role == Role.INCOMING)
        list.removeIf(x -> x.role == Role.FORECAST);
      if (list.size() >= 256) {
        Wave victim = list.stream()
                          .filter(x
                              -> x.role == Role.VIRTUAL || x.role == Role.FORECAST || x.labeled
                                  || x.cancelled)
                          .findFirst()
                          .orElse(null);
        if (victim == null) {
          realOverflow++;
          Wave old = list.remove(list.size() - 1);
          double error = Math.max(old.originError, w.originError + old.origin.distance(w.origin));
          Wave envelope = new Wave(nextId++, w.owner, w.target, w.round,
              Math.min(old.start, w.start), Math.max(old.end, w.end), Role.INCOMING, old.origin,
              error, Math.min(old.powerLo, w.powerLo), Math.max(old.powerHi, w.powerHi), old.base,
              old.sign, old.positive, old.negative, 1, old.features, old.targetOrigin, old.melee);
          envelope.originQuality = 0;
          envelope.missingFeatures = 0xffff;
          list.add(envelope);
          return;
        }
        list.remove(victim);
        evictions++;
      }
      list.add(w);
    }
    void death(int id, int t) {
      deaths.merge(id, t, Math::min);
      bullets.remove(id);
      for (Wave w : all()) w.shadows.removeIf(s -> s.bullet == id && s.turn >= t);
    }
    long shadowFallbacks, realOverflow;
    void shadows(int now, Physics.Rules rules) {
      for (Wave w : all())
        if (w.role == Role.INCOMING) {
          w.shadows.clear();
          shadowFallbacks++;
        }
    }
    Optional<Wave> associate(int owner, double power, Physics.V contact, int turn) {
      return all()
          .stream()
          .filter(w
              -> w.role == Role.INCOMING && w.owner == owner && !w.aimLabeled
                  && power >= w.powerLo - .02 && power <= w.powerHi + .02)
          .filter(w -> {
            double d = w.origin.distance(contact);
            return d >= w.inner(turn) - 2 * Physics.bulletSpeed(power) - w.originError
                && d <= w.outer(turn) + 2 * Physics.bulletSpeed(power) + w.originError;
          })
          .min(Comparator
                  .<Wave>comparingDouble(w
                      -> Math.abs(
                          w.origin.distance(contact) - (w.inner(turn) + w.outer(turn)) * .5))
                  .thenComparingLong(w -> w.id));
    }
    void roundReset() {
      waves.clear();
      bullets.clear();
      deaths.clear();
    }
  }
}
