import java.util.*;
final class Learner {
  static final int DIM = 16;
  static final double[] WEIGHTS = {1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1};
  static final double[] DISPLACEMENT_WEIGHTS = {0, 2, 2, 1, 2, 1, 1, 2, 2, 1, 0, 1, 1, 1, 0, 0};
  static final int[] LONG = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14},
                     RECENT = {1, 2, 3, 4, 5, 6, 7, 8, 9}, SIMPLE = {1, 2, 4},
                     HIT = {1, 2, 4, 5, 6, 7, 8, 9, 11};
  static double[] select(double[] f, int[] index) {
    double[] v = new double[index.length];
    for (int i = 0; i < v.length; i++) v[i] = f[index[i]];
    return v;
  }
  static double[] weights(int[] index) {
    return select(WEIGHTS, index);
  }
  static final class Enemy {
    final int id;
    boolean alive = true;
    Memory.Sample latest;
    final Memory.History history = new Memory.History();
    final Energy.Ledger ledger = new Energy.Ledger();
    int sign = 1, lastDirection = 1, changeTurn, reverseTurn, crossings;
    double acceleration, damageTaken, damageGiven;
    long shots, hits, contacts, rawHits, rawContacts, observedTogether;
    final Memory.Store longGun, displacement, simple, context, recentHit, visits, powers;
    final Memory.Store[] recent = new Memory.Store[3];
    final Map<String, Rating> ratings = new LinkedHashMap<>();
    String selected = "analytic";
    Enemy(int id, Memory.Budget b) {
      this.id = id;
      longGun = new Memory.Store(8192, LONG.length, Memory.Label.PRECISE_GF, b, 0);
      displacement = new Memory.Store(8192, DIM, Memory.Label.DISPLACEMENT, b, 0);
      int[] caps = {128, 512, 2048};
      for (int i = 0; i < 3; i++)
        recent[i] = new Memory.Store(caps[i], RECENT.length, Memory.Label.PRECISE_GF, b, 0);
      simple = new Memory.Store(1024, SIMPLE.length, Memory.Label.THEORETICAL_GF, b, 1);
      context = new Memory.Store(1024, HIT.length, Memory.Label.THEORETICAL_GF, b, 1);
      recentHit = new Memory.Store(64, HIT.length, Memory.Label.THEORETICAL_GF, b, 1);
      visits = new Memory.Store(2048, DIM, Memory.Label.THEORETICAL_GF, b, 0);
      powers = new Memory.Store(512, 3, Memory.Label.POWER, b, 1);
      for (String gun : List.of("analytic", "long", "recent")) ratings.put(gun, new Rating());
    }
    void observe(Memory.Sample s, Physics.V self) {
      if (latest != null) {
        int dt = s.turn() - latest.turn();
        if (dt > 0) {
          acceleration = Physics.clamp(
              (Math.abs(s.body().speed()) - Math.abs(latest.body().speed())) / dt, -2, 1);
          if (Math.abs(s.body().speed() - latest.body().speed()) > .5)
            changeTurn = s.turn();
          int direction = (int) Math.signum(s.body().speed());
          if (direction != 0 && direction != lastDirection) {
            reverseTurn = s.turn();
            lastDirection = direction;
          }
        }
      }
      double lateral = s.body().speed() * Math.sin(s.body().heading() - self.bearing(s.body().p()));
      if (Math.abs(lateral) > 1e-6)
        sign = lateral > 0 ? 1 : -1;
      latest = s;
      history.add(s);
      alive = true;
      observedTogether++;
    }
    void reset() {
      latest = null;
      alive = true;
      history.clear();
      ledger.reset();
      changeTurn = reverseTurn = 0;
      acceleration = 0;
      sign = lastDirection = 1;
    }
  }
  static final class Rating {
    long count;
    double raw, adjusted;
    void score(double value, double difficulty) {
      raw = count == 0 ? value : .95 * raw + .05 * value;
      adjusted = count == 0 ? value * difficulty : .95 * adjusted + .05 * value * difficulty;
      count++;
    }
  }
  record Context(
      double[] features, double positive, double negative, double quality, int missingFeatures) {}
  static Context context(Physics.V source, Physics.Body target, Enemy enemy, int turn, double power,
      Physics.Rules rules, int population, int initial, boolean physical) {
    double range = source.distance(target.p()), base = source.bearing(target.p()),
           flight = range / Physics.bulletSpeed(power), h = Math.max(1, flight),
           relative = Physics.wrap(Physics.effective(target) - base);
    int missing = 0;
    double[] f = new double[DIM];
    f[0] = power / 3;
    f[1] = Math.min(flight, 100) / 100;
    f[2] = target.speed() * Math.sin(target.heading() - base) / 8;
    f[3] = -target.speed() * Math.cos(target.heading() - base) / 8;
    f[4] = (enemy.acceleration + 2) / 3;
    f[5] = Math.sin(relative);
    f[6] = Math.cos(relative);
    double[] escape = new double[2];
    for (int j = 0; j < 2; j++) {
      int direction = j == 0 ? enemy.sign : -enemy.sign;
      Physics.Body s = target;
      int wallTick = (int) Math.ceil(1.5 * h);
      double extremum = 0;
      for (int tick = 1; tick <= Math.min(150, (int) Math.ceil(1.5 * h)); tick++) {
        double a = source.bearing(s.p()) + direction * Math.PI / 2;
        Physics.Control c = Physics.toward(s, s.p().add(Physics.vector(a).mul(140)), false);
        if (Physics.workRemaining() <= 400) {
          missing |= 1 << (7 + j);
          wallTick = tick;
          extremum = Physics.escape(power);
          break;
        }
        Physics.Step n = Physics.step(s, c, rules);
        s = n.body();
        if (n.wall()) {
          wallTick = tick;
          break;
        }
        extremum = Math.max(extremum, direction * Physics.wrap(source.bearing(s.p()) - base));
        if (tick * Physics.bulletSpeed(power) > source.distance(s.p()) + 18)
          break;
      }
      f[7 + j] = Physics.clamp(wallTick / h, 0, 1.5);
      escape[j] = Math.max(1e-6, extremum);
    }
    f[9] = Math.min(Math.max(0, turn - enemy.changeTurn) / h, 1);
    f[10] = Math.min(Math.max(0, turn - enemy.reverseTurn) / h, 1);
    f[11] = enemy.history.displacement(turn, 8) / 64;
    f[12] = enemy.history.displacement(turn, 20) / 160;
    f[13] = enemy.history.displacement(turn, 40) / 320;
    f[14] = physical ? 0 : 1;
    f[15] = population / (double) Math.max(1, initial);
    int[] horizons = {8, 20, 40};
    for (int i = 0; i < 3; i++)
      if (enemy.history.at(turn - horizons[i]).isEmpty()) {
        missing |= 1 << (11 + i);
        f[11 + i] = 0;
      }
    return new Context(f, escape[0], escape[1], missing == 0 ? 1 : .5, missing);
  }
  static double predictPower(Enemy e, double range, double own, double enemy) {
    double[] q = {range / 1000, enemy / 100, own / 100};
    int n = e.powers.size();
    if (n == 0)
      return Math.min(2, Math.max(.1, enemy));
    var neighbors =
        e.powers.query(q, new double[] {1, 1, 1}, Math.min(16, Math.max(1, (int) Math.sqrt(n))));
    double p = neighbors.stream().mapToDouble(x -> x.observation().a()).average().orElse(2);
    return Physics.clamp(Math.min(p, Math.max(.1, enemy)), .1, 3);
  }
  static double gun(String name, Enemy e, Waves.Wave w, double gunHeading, Physics.Rules rules) {
    double analytic = Physics.intercept(w.origin, w.targetOrigin, w.powerHi, rules);
    if (name.equals("analytic"))
      return analytic;
    if (w.melee) {
      var neighbors = e.displacement.query(
          w.features, DISPLACEMENT_WEIGHTS, Math.min(160, Math.max(1, e.displacement.size() / 12)));
      double best = -1, answer = analytic;
      for (var n : neighbors) {
        var point =
            Melee.project(n.observation(), w.targetOrigin, w.sign, w.origin, w.powerHi, rules);
        if (point.isEmpty())
          continue;
        double angle = w.origin.bearing(point.get()), score = 0;
        for (var m : neighbors) {
          var q =
              Melee.project(m.observation(), w.targetOrigin, w.sign, w.origin, w.powerHi, rules);
          if (q.isPresent())
            score += m.weight()
                * Geometry.gaussian(Physics.wrap(angle - w.origin.bearing(q.get())),
                    Math.max(1e-4,
                        2 * Math.asin(Math.min(1, 18 / Math.max(18, w.origin.distance(q.get()))))));
        }
        if (score > best) {
          best = score;
          answer = angle;
        }
      }
      return answer;
    }
    List<List<Memory.Neighbor>> views = new ArrayList<>();
    if (name.equals("long"))
      views.add(e.longGun.query(select(w.features, LONG), weights(LONG),
          Math.min(160, Math.max(1, e.longGun.size() / 12))));
    else
      for (Memory.Store s : e.recent)
        views.add(s.query(select(w.features, RECENT), weights(RECENT), Math.min(5, s.size())));
    if (views.stream().allMatch(List::isEmpty))
      return analytic;
    List<Double> candidates = new ArrayList<>();
    for (int i = 0; i <= 60; i++)
      candidates.add(Physics.decode(-1 + i / 30.0, w.base, w.sign, w.positive, w.negative));
    candidates.add(w.base);
    candidates.add(analytic);
    double bandwidth = Math.max(
        1e-4, 2 * Math.asin(Math.min(1, 18 / Math.max(18, w.origin.distance(w.targetOrigin.p())))));
    double best = -1, angle = analytic;
    for (double a : candidates) {
      double score = 0;
      int populated = 0;
      for (List<Memory.Neighbor> view : views) {
        double sum = 0, total = 0;
        int rank = 0;
        List<Memory.Neighbor> recency = new ArrayList<>(view);
        recency.sort(
            Comparator.comparingLong((Memory.Neighbor n) -> n.observation().sequence()).reversed());
        for (Memory.Neighbor n : recency) {
          double weight = n.weight() * Math.pow(2, -rank++ / 32.0),
                 labelAngle =
                     Physics.decode(n.observation().a(), w.base, w.sign, w.positive, w.negative);
          sum += weight * Geometry.gaussian(Physics.wrap(a - labelAngle), bandwidth);
          total += weight;
        }
        if (total > 0) {
          score += sum / total;
          populated++;
        }
      }
      if (populated > 0)
        score /= populated;
      if (score > best + 1e-12
          || (Math.abs(score - best) <= 1e-12 && tie(a, angle, gunHeading, w.base))) {
        best = score;
        angle = a;
      }
    }
    return angle;
  }
  static boolean tie(double a, double b, double heading, double base) {
    double da = Math.abs(Physics.wrap(a - heading)), db = Math.abs(Physics.wrap(b - heading));
    if (Math.abs(da - db) > 1e-12)
      return da < db;
    da = Math.abs(Physics.wrap(a - base));
    db = Math.abs(Physics.wrap(b - base));
    return Math.abs(da - db) > 1e-12 ? da < db : Physics.wrap(a - base) < Physics.wrap(b - base);
  }
  static void select(Enemy e) {
    Rating current = e.ratings.get(e.selected);
    if (current.count < 8)
      return;
    for (String name : List.of("analytic", "long", "recent")) {
      Rating challenger = e.ratings.get(name);
      if (challenger.count >= 8 && challenger.raw > current.raw + .02) {
        e.selected = name;
        current = challenger;
      }
    }
  }
  static void trainGun(Enemy e, Waves.Wave w, Memory.Sample outcome, long sequence) {
    if (w.labeled || w.cancelled || outcome.quality() == Memory.Quality.PREDICTED)
      return;
    double angle = w.origin.bearing(outcome.body().p()),
           gf = Physics.gf(angle, w.base, w.sign, w.positive, w.negative),
           half = Math.asin(Math.min(1, 18 / Math.max(18, w.origin.distance(outcome.body().p()))));
    // Cohort ratings frozen at origin, consumed before inserting this outcome.
    if (w.ratingEligible)
      for (Waves.Prediction p : w.predictions.values()) {
        double error = Physics.wrap(angle - p.angle()),
               score = Math.exp(-.5 * Math.pow(error / Math.max(half, 1e-4), 2));
        e.ratings.get(p.gun()).score(
            score, Math.min(10, (w.positive + w.negative) / Math.max(2 * half, 1e-4)));
      }
    double quality =
        (outcome.quality() == Memory.Quality.OBSERVED ? 1 : .5) * w.confidence * w.originQuality;
    if (!w.melee)
      e.longGun.add(new Memory.Observation(select(w.features, LONG), Memory.Label.PRECISE_GF, gf, 0,
          w.id, w.round, outcome.turn(), quality, sequence, w.melee));
    if (!w.melee)
      for (Memory.Store s : e.recent)
        s.add(new Memory.Observation(select(w.features, RECENT), Memory.Label.PRECISE_GF, gf, 0,
            w.id, w.round, outcome.turn(), quality, sequence, w.melee));
    double time = Math.max(1, outcome.turn() - w.start), frame = Physics.effective(w.targetOrigin);
    Physics.V delta = outcome.body().p().sub(w.targetOrigin.p()).mul(1 / time);
    double x = delta.x() * Math.cos(frame) + delta.y() * Math.sin(frame),
           y = (-delta.x() * Math.sin(frame) + delta.y() * Math.cos(frame)) * w.sign;
    e.displacement.add(new Memory.Observation(w.features, Memory.Label.DISPLACEMENT, x, y, w.id,
        w.round, outcome.turn(), quality, sequence, w.melee));
    e.crossings++;
    w.labeled = true;
    select(e);
  }
  static void trainHit(Enemy e, Waves.Wave w, double bearing, int turn, double quality,
      long sequence, boolean contact) {
    if (w.aimLabeled)
      return;
    double gf =
        Physics.gf(bearing, w.base, w.sign, Physics.escape(w.powerHi), Physics.escape(w.powerHi));
    for (var spec : List.of(
             Map.entry(e.simple, SIMPLE), Map.entry(e.context, HIT), Map.entry(e.recentHit, HIT)))
      spec.getKey().add(
          new Memory.Observation(select(w.features, spec.getValue()), Memory.Label.THEORETICAL_GF,
              gf, 0, w.id, w.round, turn, quality * w.originQuality, sequence, w.melee));
    w.aimLabeled = true;
    w.contact = contact;
    w.hit = !contact;
    if (contact)
      e.contacts++;
    else
      e.hits++;
  }
  record Kernel(double angle, double weight) {}
  static Geometry.Density prepareDensity(Enemy e, Waves.Wave w) {
    return prepareDensity(e, w, new Physics.Rules(800, 600, .1));
  }
  static Geometry.Density prepareDensity(Enemy e, Waves.Wave w, Physics.Rules rules) {
    double escape = Physics.escape(w.powerHi),
           sigma = Math.max(1e-4,
               2
                   * Math.asin(
                       Math.min(1, 18 / Math.max(18, w.origin.distance(w.targetOrigin.p()))))),
           cv = Physics.intercept(w.origin, w.targetOrigin, w.powerHi, rules);
    List<Memory.Store> stores = new ArrayList<>();
    List<int[]> subsets = new ArrayList<>();
    List<Double> strengths = new ArrayList<>();
    if (e.simple.size() > 0) {
      stores.add(e.simple);
      subsets.add(SIMPLE);
      strengths.add(1.0);
    }
    if (e.context.size() >= 8) {
      stores.add(e.context);
      subsets.add(HIT);
      strengths.add(2.0);
      stores.add(e.recentHit);
      subsets.add(HIT);
      strengths.add(2.0);
    }
    if (e.shots >= 50 && Energy.wilsonLower(e.rawHits, e.shots - e.rawContacts) > .08) {
      stores.add(e.visits);
      subsets.add(null);
      strengths.add(.5);
    }
    List<Kernel> kernels = new ArrayList<>();
    double total = 0;
    for (int j = 0; j < stores.size(); j++) {
      Memory.Store store = stores.get(j);
      int[] subset = subsets.get(j);
      double[] q = subset == null ? w.features : select(w.features, subset),
               weights = subset == null ? WEIGHTS : weights(subset);
      int k = store == e.visits  ? 32
          : store == e.recentHit ? 5
                                 : Math.min(24, Math.max(1, store.size() / 5));
      List<Memory.Neighbor> neighbors = store.query(q, weights, k);
      double norm = neighbors.stream().mapToDouble(Memory.Neighbor::weight).sum();
      if (norm > 0) {
        for (var n : neighbors)
          kernels.add(
              new Kernel(Physics.decode(n.observation().a(), w.base, w.sign, escape, escape),
                  strengths.get(j) * n.weight() / norm));
        total += strengths.get(j);
      }
    }
    final double denominator = total,
                 priorWeight = total == 0 ? 1 : Math.max(.1, 1 - e.simple.size() / 32.0);
    final double cvOffset = Physics.clamp(Physics.wrap(cv - w.base), -escape, escape);
    final double norm0 =
                     Math.max(1e-9, Geometry.cdf(escape / sigma) - Geometry.cdf(-escape / sigma)),
                 normCV = Math.max(1e-9,
                     Geometry.cdf((escape - cvOffset) / sigma)
                         - Geometry.cdf((-escape - cvOffset) / sigma));
    final double circularNorm =
        Math.max(1e-9, Geometry.cdf(Math.PI / sigma) - Geometry.cdf(-Math.PI / sigma));
    double learnedBound = 0;
    for (Kernel k : kernels)
      learnedBound += k.weight / (Math.sqrt(2 * Math.PI) * sigma * circularNorm);
    final double certified =
        priorWeight * (.5 / norm0 + .3 / normCV) / (Math.sqrt(2 * Math.PI) * sigma)
        + priorWeight * .2 / (2 * escape)
        + (1 - priorWeight) * learnedBound / Math.max(1e-9, denominator);
    return new Geometry.Density() {
      public double upperBound() {
        return certified;
      }
      public double at(double angle) {
        double offset = Physics.wrap(angle - w.base),
               prior = .5 * Geometry.truncatedGaussian(offset, 0, sigma, escape)
            + .3 * Geometry.truncatedGaussian(offset, cvOffset, sigma, escape)
            + (Math.abs(offset) <= escape ? .2 / (2 * escape) : 0),
               learned = 0;
        for (Kernel k : kernels)
          learned +=
              k.weight * Geometry.gaussian(Physics.wrap(angle - k.angle), sigma) / circularNorm;
        return priorWeight * prior + (1 - priorWeight) * learned / Math.max(1e-9, denominator);
      }
    };
  }
  static double density(Enemy e, Waves.Wave w, double angle) {
    return prepareDensity(e, w).at(angle);
  }
}
