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
    FireControl.Cohort preparedCohort;
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
    static final int MAX_OWN_BULLETS = 256, MAX_SHADOW_WORK = 4096, MAX_SHADOW_INTERVALS = 2048,
                     SHADOW_HORIZON = 160, MAX_SHADOW_ALLOCATION_UNITS = 524288;
    long shadowFallbacks, realOverflow, shadowWork, shadowAllocations;
    int shadowIntervals;
    void shadows(int now, Physics.Rules rules) {
      shadowWork = shadowAllocations = 0;
      shadowIntervals = 0;
      for (Wave w : all()) {
        // Historical intervals survive only while supported by lifecycle evidence. Future
        // intervals are rebuilt from the current legal own-bullet snapshot every turn.
        w.shadows.removeIf(
            s -> s.turn >= now || deaths.getOrDefault(s.bullet, Integer.MAX_VALUE) <= s.turn);
        shadowIntervals += w.shadows.stream().mapToInt(s -> s.arcs.size()).sum();
        if (w.role != Role.INCOMING || w.cancelled)
          continue;
        if (w.start != w.end || w.originError != 0 || w.powerLo != w.powerHi) {
          shadowFallbacks++;
          continue;
        }
        for (Bullet b : bullets.values()) {
          if (shadowWork >= MAX_SHADOW_WORK || shadowIntervals >= MAX_SHADOW_INTERVALS) {
            shadowFallbacks++;
            break;
          }
          double speed = Physics.bulletSpeed(b.power);
          for (int t = now; t <= now + SHADOW_HORIZON; t++) {
            if (t <= w.start || t <= b.observedTurn && t != now)
              continue;
            if (deaths.getOrDefault(b.id, Integer.MAX_VALUE) <= t)
              break;
            if (shadowWork >= MAX_SHADOW_WORK || shadowIntervals >= MAX_SHADOW_INTERVALS
                || shadowAllocations + 128 > MAX_SHADOW_ALLOCATION_UNITS) {
              shadowFallbacks++;
              break;
            }
            shadowWork++;
            shadowAllocations += 128; // conservative object-allocation charge, not JVM bytes
            Physics.V velocity = Physics.vector(b.heading).mul(speed);
            Physics.V end = b.position.add(velocity.mul(t - b.observedTurn)),
                      from = end.sub(velocity);
            if (from.x() <= 0 || from.y() <= 0 || from.x() >= rules.width()
                || from.y() >= rules.height())
              break;
            // Unknown wall/contact phase: no credit on a segment ending outside the arena.
            if (end.x() <= 0 || end.y() <= 0 || end.x() >= rules.width()
                || end.y() >= rules.height())
              break;
            List<Geometry.Arc> arcs =
                shadowSegment(w.origin, from, end, w.inner(t - 1), w.outer(t));
            if (!arcs.isEmpty() && shadowIntervals + arcs.size() <= MAX_SHADOW_INTERVALS) {
              w.shadows.add(new Shadow(b.id, t, arcs));
              shadowIntervals += arcs.size();
            }
          }
        }
      }
      Set<Integer> retainedIds = new HashSet<>(bullets.keySet());
      for (Wave w : all())
        for (Shadow s : w.shadows) retainedIds.add(s.bullet);
      deaths.keySet().retainAll(retainedIds); // at most256 tracked +2048 interval identities
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
  /**
   * Native swept-line predicate: radial intersection plus endpoint broad phase.
   * Each retained segment is wholly in the annulus; every mapped ray therefore
   * intersects both swept lines. Intersect with the exact endpoint-distance arc.
   * Degenerate/tangent numerical cases deliberately receive zero measure credit.
   */
  static List<Geometry.Arc> shadowSegment(
      Physics.V origin, Physics.V from, Physics.V end, double inner, double outer) {
    if (outer <= inner || inner < 0)
      return List.of();
    Physics.V a = from.sub(origin), d = end.sub(from);
    double aa = d.x() * d.x() + d.y() * d.y();
    if (aa < 1e-12)
      return List.of();
    double[] cuts = new double[6];
    int n = 2;
    cuts[0] = 0;
    cuts[1] = 1;
    for (double radius : new double[] {inner, outer}) {
      double bb = 2 * (a.x() * d.x() + a.y() * d.y()),
             cc = a.x() * a.x() + a.y() * a.y() - radius * radius, disc = bb * bb - 4 * aa * cc;
      if (Math.abs(disc) <= 1e-9)
        return List.of();
      if (disc < 0)
        continue;
      double root = Math.sqrt(disc);
      for (double u : new double[] {(-bb - root) / (2 * aa), (-bb + root) / (2 * aa)})
        if (u > 0 && u < 1)
          cuts[n++] = u;
    }
    Arrays.sort(cuts, 0, n);
    double distance = origin.distance(end);
    if (distance < 1e-9 || outer < 1e-9)
      return List.of();
    double cosine = (outer * outer + distance * distance - 39.4 * 39.4) / (2 * outer * distance);
    if (cosine >= 1)
      return List.of();
    List<Geometry.Arc> broad =
        Geometry.around(origin.bearing(end), cosine <= -1 ? Math.PI : Math.acos(cosine));
    List<Geometry.Arc> out = new ArrayList<>();
    for (int i = 1; i < n; i++) {
      double lo = cuts[i - 1] + 1e-7, hi = cuts[i] - 1e-7;
      if (hi <= lo)
        continue;
      double radius = a.add(d.mul((lo + hi) / 2)).distance(new Physics.V(0, 0));
      if (radius <= inner + 1e-9 || radius >= outer - 1e-9)
        continue;
      Physics.V left = from.add(d.mul(lo)), right = from.add(d.mul(hi));
      if (Geometry.segmentDistance(origin, left, right) < 1e-8)
        continue;
      if (Geometry.segmentDistance(origin, left, right) < inner + 1e-9
          || Math.max(origin.distance(left), origin.distance(right)) > outer - 1e-9)
        continue;
      double angle = origin.bearing(left), delta = Physics.wrap(origin.bearing(right) - angle);
      List<Geometry.Arc> mapped = Geometry.around(angle + delta / 2, Math.abs(delta) / 2);
      for (Geometry.Arc x : mapped)
        for (Geometry.Arc y : broad) {
          double l = Math.max(x.lo(), y.lo()) + 1e-12, h = Math.min(x.hi(), y.hi()) - 1e-12;
          if (h > l)
            out.add(new Geometry.Arc(l, h));
        }
    }
    return Geometry.merge(out);
  }
}
