import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;
import java.util.*;
final class Core {
  final Physics.Rules rules;
  final Diagnostics log;
  final Memory.Budget budget = new Memory.Budget();
  final Map<Integer, Learner.Enemy> enemies = new TreeMap<>();
  final Waves.Service waves = new Waves.Service();
  final FireControl fire = new FireControl();
  final Planner planner = new Planner();
  final Radar radar = new Radar();
  final Melee melee;
  final Memory.History ownHistory = new Memory.History();
  final Learner.Enemy ownContext;
  record Contact(BulletState bullet, int turn, boolean contact) {}
  final LinkedHashSet<String> seen = new LinkedHashSet<>();
  int round, initialCount, lastTurn = -1;
  long sequence;
  boolean beganMelee;
  Waves.Wave shotContext;
  record Intent(Physics.Control move, double gunTurn, double radarTurn, boolean fire) {}
  Core(Physics.Rules r, Diagnostics log, long seed) {
    rules = r;
    this.log = log;
    melee = new Melee(new SplittableRandom(seed).split().nextLong());
    ownContext = new Learner.Enemy(-1, budget);
    RuntimeConfig.verify();
    log.event("profile", 0, 0,
        "independent-v1 seed=" + seed
            + " motion=official-fixture-replay melee=false noCrossGameLoad");
  }
  Learner.Enemy enemy(int id) {
    return enemies.computeIfAbsent(id, k -> new Learner.Enemy(k, budget));
  }
  void roundReset(int round) {
    this.round = round;
    waves.roundReset();
    fire.reset();
    radar.reset();
    planner.previous = -1;
    planner.steps = 0;
    planner.densityCache.clear();
    ownHistory.clear();
    ownContext.reset();
    seen.clear();
    for (Learner.Enemy e : enemies.values()) e.reset();
    initialCount = 0;
    lastTurn = -1;
    shotContext = null;
    melee.recent.clear();
    melee.retained = null;
    log.roundReset();
  }
  boolean dedup(BotEvent e) {
    String key = e.getTurnNumber() + ":" + e.getClass().getSimpleName() + ":" + identity(e);
    if (!seen.add(key)) {
      log.count("duplicate.event");
      return false;
    }
    while (seen.size() > 8192) seen.remove(seen.iterator().next());
    return true;
  }
  String identity(BotEvent e) {
    if (e instanceof ScannedBotEvent x)
      return "" + x.getScannedBotId();
    if (e instanceof BulletFiredEvent x)
      return "" + x.getBullet().getBulletId();
    if (e instanceof BulletHitBotEvent x)
      return "" + x.getBullet().getOwnerId() + ":" + x.getBullet().getBulletId();
    if (e instanceof HitByBulletEvent x)
      return "" + x.getBullet().getOwnerId() + ":" + x.getBullet().getBulletId();
    if (e instanceof BulletHitBulletEvent x)
      return x.getBullet().getBulletId() + ":" + x.getHitBullet().getOwnerId() + ":"
          + x.getHitBullet().getBulletId();
    if (e instanceof BulletHitWallEvent x)
      return "" + x.getBullet().getBulletId();
    if (e instanceof HitBotEvent x)
      return x.getVictimId() + ":" + x.getX() + ":" + x.getY() + ":" + x.isRammed();
    if (e instanceof HitWallEvent)
      return "wall";
    if (e instanceof BotDeathEvent x)
      return "" + x.getVictimId();
    return "";
  }
  Intent decide(Physics.Body own, double energy, int turn, int population, List<BotEvent> events,
      Collection<BulletState> bullets, int myId) {
    long begin = System.nanoTime();
    Physics.beginWork(2400);
    if (initialCount == 0) {
      initialCount = Math.max(1, population);
      beganMelee = population > 1;
    }
    if (lastTurn >= 0 && turn != lastTurn + 1) {
      log.count("turn.gap");
      fire.plan = null;
      fire.pending = null;
    }
    lastTurn = turn;
    ownHistory.add(new Memory.Sample(round, turn, own, energy, Memory.Quality.OBSERVED, 0));
    ownContext.observe(new Memory.Sample(round, turn, own, energy, Memory.Quality.OBSERVED, 0),
        new Physics.V(rules.width() / 2, rules.height() / 2));
    log.event("own.state", round, turn,
        "x=" + own.p().x() + " y=" + own.p().y() + " heading=" + own.heading() + " speed="
            + own.speed() + " gun=" + own.gun() + " radar=" + own.radar() + " heat=" + own.heat()
            + " energy=" + energy + " population=" + population + " version=" + sequence);
    List<Contact> contacts = new ArrayList<>();
    List<BotEvent> fresh = events.stream().filter(this::dedup).toList();
    // Phase 1 lifecycle. Preserve original callback order in chronology.
    for (BotEvent event : fresh) {
      log.event(event.getClass().getSimpleName(), round, event.getTurnNumber(),
          identity(event) + " callbackSequence=" + (++sequence) + " ownStateTurn=" + turn
              + " own=" + own);
      if (event instanceof BotDeathEvent e) {
        enemy(e.getVictimId()).alive = false;
        for (Waves.Wave w : waves.all())
          if (w.target == e.getVictimId()) {
            w.cancelled = true;
            w.predictions.clear();
          }
        if (fire.plan != null && fire.plan.target() == e.getVictimId())
          fire.plan = null;
      }
      if (event instanceof SkippedTurnEvent) {
        log.count("turn.skipped");
        fire.plan = null;
        fire.pending = null;
      }
    }
    // Phase 2 known effects before any scan residual, irrespective of callback order.
    for (BotEvent event : fresh) {
      int t = event.getTurnNumber();
      if (event instanceof BulletHitBotEvent e) {
        Learner.Enemy victim = enemy(e.getVictimId());
        victim.ledger.effect(new Energy.Effect(t + 1, e.getDamage(),
            "damage:" + e.getBullet().getOwnerId() + ":" + e.getBullet().getBulletId() + ":" + t));
        victim.damageGiven += e.getDamage();
        log.observe("energyHitBonus", 3 * e.getBullet().getPower());
        log.count("shot.hit");
        log.observe("damageGiven", e.getDamage());
        waves.death(e.getBullet().getBulletId(), t);
      }
      if (event instanceof HitByBulletEvent e) {
        BulletState b = e.getBullet();
        Learner.Enemy shooter = enemy(b.getOwnerId());
        shooter.ledger.effect(
            new Energy.Effect(t + 1, -3 * b.getPower(), "bonus:" + b.getBulletId() + ":" + t));
        shooter.damageTaken += e.getDamage();
        contacts.add(new Contact(b, t, false));
        log.count("incoming.hit");
        log.observe("damageTaken", e.getDamage());
      }
      if (event instanceof BulletHitBulletEvent e) {
        BulletState b = e.getBullet(), other = e.getHitBullet();
        if (b.getOwnerId() == myId)
          waves.death(b.getBulletId(), t);
        else
          contacts.add(new Contact(b, t, true));
        if (other.getOwnerId() == myId)
          waves.death(other.getBulletId(), t);
        else
          contacts.add(new Contact(other, t, true));
        log.count("bullet.contact");
      }
      if (event instanceof BulletHitWallEvent e)
        waves.death(e.getBullet().getBulletId(), t);
      if (event instanceof HitWallEvent)
        log.count("collision.wall");
      if (event instanceof HitBotEvent)
        log.count("collision.bot");
      if (event instanceof BulletFiredEvent e) {
        BulletState b = e.getBullet();
        log.observe("energyFireCost", b.getPower());
        Optional<FireControl.Pending> pending = fire.confirm(b.getBulletId(), t, b.getPower(),
            new Physics.V(b.getX(), b.getY()), Math.toRadians(b.getDirection()));
        waves.bullets.put(b.getBulletId(),
            new Waves.Bullet(b.getBulletId(), new Physics.V(b.getX(), b.getY()),
                Math.toRadians(b.getDirection()), b.getPower(), t - 1));
        if (pending.isPresent()) {
          var p = pending.get();
          Learner.Enemy target = enemies.get(p.target());
          if (target != null && shotContext != null) {
            Waves.Wave frozen = shotContext;
            for (Waves.Wave old : waves.all())
              if (old.role == Waves.Role.VIRTUAL && old.target == p.target()
                  && old.start == p.requestedTurn())
                old.cancelled = true;
            Waves.Wave w = new Waves.Wave(frozen.id, myId, p.target(), round, p.requestedTurn(),
                p.requestedTurn(), Waves.Role.OUTGOING, new Physics.V(b.getX(), b.getY()), 0,
                p.power(), p.power(), frozen.base, frozen.sign, frozen.positive, frozen.negative,
                frozen.confidence, frozen.features, frozen.targetOrigin, frozen.melee);
            w.originQuality = frozen.originQuality;
            w.missingFeatures = frozen.missingFeatures;
            w.predictions.putAll(frozen.predictions);
            w.creationTurn = t;
            w.firstMotionTurn = t;
            w.bullet = b.getBulletId();
            w.actualAngle = Math.toRadians(b.getDirection());
            w.ratingEligible = true;
            waves.add(w);
            shotContext = null;
          }
        }
        log.event("fire.confirmed", round, t,
            "id=" + b.getBulletId() + " heading=" + b.getDirection() + " position=" + b.getX() + ","
                + b.getY() + " pending=" + pending.isPresent() + " request="
                + pending.map(x -> Integer.toString(x.requestedTurn())).orElse("unmatched")
                + " creation=" + t + " firstMotion=" + t);
      }
    }
    for (BulletState b : bullets) {
      if (b.getOwnerId() == myId && !waves.deaths.containsKey(b.getBulletId())) {
        Waves.Bullet old = waves.bullets.put(b.getBulletId(),
            new Waves.Bullet(b.getBulletId(), new Physics.V(b.getX(), b.getY()),
                Math.toRadians(b.getDirection()), b.getPower(), turn));
        if (old != null && old.position().distance(new Physics.V(b.getX(), b.getY())) > 1e-6)
          log.event("bullet.motion", round, turn, "id=" + b.getBulletId());
      }
    }
    Set<Integer> visibleBullets = new HashSet<>();
    for (BulletState b : bullets)
      if (b.getOwnerId() == myId)
        visibleBullets.add(b.getBulletId());
    for (Waves.Bullet b : List.copyOf(waves.bullets.values()))
      if (b.observedTurn() < turn && !visibleBullets.contains(b.id())) {
        waves.death(b.id(), turn);
        log.count("bullet.absentReconciled");
      }
    while (waves.bullets.size() > 256) {
      waves.bullets.remove(waves.bullets.keySet().iterator().next());
      log.count("ownBullet.eviction");
    }
    long inferenceStart = System.nanoTime();
    for (BotEvent event : fresh)
      if (event instanceof ScannedBotEvent scan) {
        int t = scan.getTurnNumber();
        log.event("scan.observed", round, t,
            "id=" + scan.getScannedBotId() + " x=" + scan.getX() + " y=" + scan.getY()
                + " directionDeg=" + scan.getDirection() + " speed=" + scan.getSpeed()
                + " energy=" + scan.getEnergy());
        Learner.Enemy e = enemy(scan.getScannedBotId());
        if (!e.alive || e.latest != null && t <= e.latest.turn())
          continue;
        Physics.Body observed = new Physics.Body(new Physics.V(scan.getX(), scan.getY()),
            Math.toRadians(scan.getDirection()), scan.getSpeed(), 0, 0, 0);
        Memory.Sample previous = e.latest;
        e.observe(
            new Memory.Sample(round, t, observed, scan.getEnergy(), Memory.Quality.OBSERVED, 0),
            own.p());
        boolean wall = observed.p().x() < 27 || observed.p().y() < 27
            || observed.p().x() > rules.width() - 27 || observed.p().y() > rules.height() - 27;
        boolean ram = population > 1 || own.p().distance(observed.p()) < 60;
        Optional<Energy.Inference> inferred = Optional.empty();
        if (!log.disabled.contains("inference")) {
          try {
            inferred = e.ledger.scan(t, scan.getEnergy(), rules.cooling(), wall || ram);
            log.success("inference");
          } catch (RuntimeException ex) {
            log.failure("inference", ex);
          }
        } else
          log.count("inference.disabled");
        if (inferred.isPresent() && previous != null) {
          Energy.Inference in = inferred.get();
          Physics.Body origin = previous.body();
          Memory.Sample atOrigin = ownHistory.at(in.earliest()).orElse(null);
          Physics.Body targetAtOrigin = atOrigin == null ? own : atOrigin.body();
          Learner.Context context = context(origin.p(), targetAtOrigin, ownContext, in.earliest(),
              in.high(), rules, population, initialCount, false);
          int sign = lateralSign(origin.p(), targetAtOrigin, ownContext.sign);
          Waves.Wave w =
              new Waves.Wave(waves.nextId++, e.id, myId, round, in.earliest(), in.latest(),
                  Waves.Role.INCOMING, origin.p(), in.reliable() ? 0 : 8.0 * (t - previous.turn()),
                  in.low(), in.high(), origin.p().bearing(targetAtOrigin.p()), sign,
                  Physics.escape(in.high()), Physics.escape(in.high()), in.confidence(),
                  context.features(), targetAtOrigin, beganMelee);
          w.originQuality = context.quality() * (atOrigin == null ? .5 : 1);
          w.missingFeatures = context.missingFeatures();
          waves.add(w);
          log.count(in.reliable() ? "inference.reliable" : "inference.ambiguous");
          if (in.reliable()) {
            e.powers.add(new Memory.Observation(new double[] {own.p().distance(origin.p()) / 1000,
                                                    scan.getEnergy() / 100, energy / 100},
                Memory.Label.POWER, in.high(), 0, w.id, round, t, 1, ++sequence, beganMelee));
          }
        }
      }
    for (Contact contact : contacts)
      incomingContact(contact.bullet(), contact.turn(), contact.contact());
    waves.shadows(turn, rules);
    if (waves.realOverflow > 0)
      log.count("waves.realOverflow.conservativeStop");
    log.time("inference", System.nanoTime() - inferenceStart);
    long storageStart = System.nanoTime();
    if (!log.disabled.contains("learning")) {
      try {
        resolve(own, turn);
        log.success("learning");
      } catch (RuntimeException ex) {
        log.failure("learning", ex);
      }
    } else
      log.count("learning.disabled");
    log.time("storage", System.nanoTime() - storageStart);
    fire.reconcile(turn);
    if (fire.pending == null)
      shotContext = null;
    Learner.Enemy target =
        enemies.values()
            .stream()
            .filter(e -> e.alive && e.latest != null)
            .min(Comparator.<Learner.Enemy>comparingInt(e -> turn - e.latest.turn())
                    .thenComparingDouble(e -> own.p().distance(e.latest.body().p()))
                    .thenComparingInt(e -> e.id))
            .orElse(null);
    // At most one cold forecast per enemy, replaced by any actual inference.
    if (target != null && population == 1 && target.ledger.heatHigh <= rules.cooling()
        && waves.all()
                .stream()
                .filter(w -> w.role == Waves.Role.INCOMING || w.role == Waves.Role.FORECAST)
                .count()
            < 2
        && waves.all().stream().noneMatch(
            w -> w.owner == target.id && w.role == Waves.Role.FORECAST)) {
      double p = predictPower(
          target, own.p().distance(target.latest.body().p()), energy, target.latest.energy());
      Learner.Context c = context(target.latest.body().p(), own, ownContext, turn, p, rules,
          population, initialCount, false);
      waves.add(new Waves.Wave(waves.nextId++, target.id, myId, round, turn + 1, turn + 2,
          Waves.Role.FORECAST, target.latest.body().p(), 8, p, p,
          target.latest.body().p().bearing(own.p()),
          lateralSign(target.latest.body().p(), own, ownContext.sign), Physics.escape(p),
          Physics.escape(p), .25, c.features(), own, beganMelee));
    }
    // Shadows require first-bullet-motion engine verification before runtime activation.
    long start = System.nanoTime();
    Physics.Control movement;
    try {
      planner.stepLimit = Math.max(0, Math.min(2400, 2400 - Physics.workUsed() - 350));
      planner.activeEnemies = enemies.values();
      Planner.Choice choice = log.disabled.contains("movement")
          ? planner.emergency(own, rules)
          : planner.choose(own, turn, waves.all(), enemies, rules, waves.deaths);
      movement = choice.command();
      if (choice.emergency())
        log.count("movement.emergency");
      log.success("movement");
    } catch (RuntimeException ex) {
      log.failure("movement", ex);
      movement = planner.emergency(own, rules).command();
    }
    log.time("movement", System.nanoTime() - start);
    boolean meleeMode = population > 1 && Melee.REVIEW_APPROVED;
    if (meleeMode) {
      Melee.requireApproval();
      movement = melee.move(own, energy, turn, enemies.values(), rules, planner);
    }
    if (population > 1 && !meleeMode) {
      log.count("melee.gated");
      fire.plan = null;
    }
    boolean submit = (population == 1 || meleeMode) && fire.valid(own, energy, target, turn, rules);
    if (submit) {
      try {
        shotContext =
            makeOutgoing(own, target, turn, fire.plan.power(), Waves.Role.OUTGOING, myId, true);
        for (String name : List.of("analytic", "long", "recent"))
          if (name.equals("analytic") || (target.crossings >= 16 && !log.disabled.contains("gun")))
            shotContext.predictions.put(name,
                new Waves.Prediction(name, shotContext.id, turn,
                    Learner.gun(name, target, shotContext, own.gun(), rules), shotContext.powerHi,
                    target.longGun.version));
      } catch (RuntimeException ex) {
        log.failure("gun", ex);
        shotContext = null;
        submit = false;
        fire.plan = null;
      }
    }
    if (waves.realOverflow > 0)
      movement = Physics.Control.stop();
    Physics.Body nextOwn = Physics.step(own, movement, rules).body();
    start = System.nanoTime();
    double gunTurn = 0;
    try {
      if (target != null) {
        double power = Energy.power(own.p().distance(target.latest.body().p()), energy,
                   target.latest.energy(), meleeMode),
               angle = nextOwn.p().bearing(target.latest.body().p());
        if (power > 0 && (population == 1 || meleeMode) && turn - target.latest.turn() <= 2) {
          if (meleeMode) {
            angle = Melee.aim(nextOwn, new ArrayList<>(enemies.values()), power, turn, rules);
            if (own.heat() <= 1e-10)
              for (Learner.Enemy living : enemies.values())
                if (living.alive && living.latest != null && turn - living.latest.turn() <= 8)
                  waves.add(makeOutgoing(
                      nextOwn, living, turn + 1, power, Waves.Role.VIRTUAL, myId, false));
          } else if (own.heat() / rules.cooling() <= 3 && !log.disabled.contains("gun")) {
            Waves.Wave virtual = makeOutgoing(nextOwn, target, turn + 1, power, Waves.Role.VIRTUAL,
                myId, own.heat() <= rules.cooling());
            for (String name : List.of("analytic", "long", "recent"))
              if (name.equals("analytic")
                  || (target.crossings >= 16 && !log.disabled.contains("gun")))
                virtual.predictions.put(name,
                    new Waves.Prediction(name, virtual.id, turn,
                        Learner.gun(name, target, virtual, own.gun(), rules), power,
                        target.longGun.version));
            angle = beganMelee
                ? Melee.aim(own, List.of(target), power, turn, rules)
                : virtual.predictions
                      .getOrDefault(target.selected, virtual.predictions.get("analytic"))
                      .angle();
            if (own.heat() <= 1e-10 && !submit)
              waves.add(virtual);
          } else
            angle = Physics.intercept(nextOwn.p(), target.latest.body(), power, rules);
          // Keep prior plan until request has been submitted by boundary.
          if (!submit)
            fire.plan = FireControl.prepare(angle, power, target, nextOwn, turn, sequence);
        } else if (!submit)
          fire.plan = null;
        gunTurn =
            Physics.clamp(Physics.wrap(angle - own.gun()), -Math.toRadians(20), Math.toRadians(20));
      }
      log.success("gun");
    } catch (RuntimeException ex) {
      log.failure("gun", ex);
      fire.plan = null;
      submit = false;
    }
    log.time("gun", System.nanoTime() - start);
    start = System.nanoTime();
    double radarTurn;
    try {
      radarTurn = log.disabled.contains("radar")
          ? Math.PI / 4
          : radar.turn(nextOwn, enemies.values(), turn, population);
      log.success("radar");
    } catch (RuntimeException ex) {
      log.failure("radar", ex);
      radarTurn = Math.PI / 4;
    }
    log.time("radar", System.nanoTime() - start);
    log.event("command.prepared", round, turn,
        "sequence=" + (++sequence) + " speed=" + movement.speed() + " turn=" + movement.turn()
            + " gunTurn=" + gunTurn + " radarTurn=" + radarTurn + " fire=" + submit
            + " plan=" + fire.plan);
    log.count("turn.decisions");
    if (target != null) {
      log.observe("range", own.p().distance(target.latest.body().p()));
      log.observe("scanAge", turn - target.latest.turn());
    }
    log.observe("energy", energy);
    log.observe("shadowFallbacks", waves.shadowFallbacks);
    log.observe("realWaveOverflow", waves.realOverflow);
    log.observe("predictorSteps", Physics.workUsed());
    log.observe("waves", waves.all().size());
    Physics.finite(movement.speed(), movement.turn(), gunTurn, radarTurn);
    log.time("decision", System.nanoTime() - begin);
    Physics.endWork();
    return new Intent(movement, gunTurn, radarTurn, submit);
  }
  Learner.Context context(Physics.V source, Physics.Body target, Learner.Enemy enemy, int turn,
      double power, Physics.Rules rules, int population, int initial, boolean physical) {
    if (!log.disabled.contains("context")) {
      try {
        var value = Learner.context(
            source, target, enemy, turn, power, rules, population, initial, physical);
        if ((value.missingFeatures() & ((1 << 7) | (1 << 8))) != 0)
          log.count("context.quotaFallback");
        log.success("context");
        return value;
      } catch (RuntimeException ex) {
        log.failure("context", ex);
      }
    } else
      log.count("context.disabled");
    double[] f = new double[Learner.DIM];
    f[0] = power / 3;
    f[1] = Math.min(source.distance(target.p()) / Physics.bulletSpeed(power), 100) / 100;
    return new Learner.Context(f, Physics.escape(power), Physics.escape(power), .5, 0xffff);
  }
  double predictPower(Learner.Enemy enemy, double range, double own, double energy) {
    if (!log.disabled.contains("inference")) {
      try {
        return Learner.predictPower(enemy, range, own, energy);
      } catch (RuntimeException ex) {
        log.failure("inference", ex);
      }
    } else
      log.count("power.disabled");
    return Physics.clamp(Math.min(2, energy), .1, 3);
  }
  static int lateralSign(Physics.V source, Physics.Body target, int prior) {
    double v = target.speed() * Math.sin(target.heading() - source.bearing(target.p()));
    return Math.abs(v) < 1e-6 ? prior : v > 0 ? 1 : -1;
  }
  Waves.Wave makeOutgoing(Physics.Body own, Learner.Enemy target, int turn, double p,
      Waves.Role role, int myId, boolean physical) {
    int age = Math.max(0, turn - target.latest.turn());
    Physics.Body frozen = target.latest.body();
    if (age > 0)
      frozen = new Physics.Body(
          Physics.constantPosition(frozen, age, rules), frozen.heading(), frozen.speed(), 0, 0, 0);
    Learner.Context c = context(own.p(), frozen, target, turn, p, rules, 1, initialCount, physical);
    Waves.Wave result = new Waves.Wave(waves.nextId++, myId, target.id, round, turn, turn, role,
        own.p(), 0, p, p, own.p().bearing(frozen.p()), target.sign, c.positive(), c.negative(),
        age > 0 ? .5 : 1, c.features(), frozen, beganMelee);
    result.originQuality = c.quality();
    result.missingFeatures = c.missingFeatures();
    return result;
  }
  void incomingContact(BulletState b, int turn, boolean contact) {
    var found =
        waves.associate(b.getOwnerId(), b.getPower(), new Physics.V(b.getX(), b.getY()), turn);
    if (found.isEmpty()) {
      waves.unmatched++;
      log.count("incoming.unmatched");
      return;
    }
    Waves.Wave w = found.get();
    if (w.start != w.end || w.originError > 0 || w.confidence < .99) {
      log.count("incoming.uncertainContact");
      return;
    }
    if (!log.disabled.contains("learning")) {
      try {
        Learner.trainHit(enemy(b.getOwnerId()), w, Math.toRadians(b.getDirection()), turn,
            contact ? .5 : 1, ++sequence, contact);
      } catch (RuntimeException ex) {
        log.failure("learning", ex);
      }
    }
  }
  void resolve(Physics.Body own, int turn) {
    for (Waves.Wave w : waves.all()) {
      if (w.cancelled)
        continue;
      if (w.role == Waves.Role.INCOMING && w.center == null && w.start == w.end
          && w.originError == 0) {
        var prior = ownHistory.at(turn - 1);
        if (prior.isPresent() && w.inner(turn - 1) < w.origin.distance(prior.get().body().p())
            && w.inner(turn) >= w.origin.distance(own.p()))
          w.center = ownHistory.at(turn).orElse(null);
      }
      if (w.role == Waves.Role.VIRTUAL || w.role == Waves.Role.OUTGOING) {
        Learner.Enemy e = enemies.get(w.target);
        if (e == null || !e.alive) {
          w.cancelled = true;
          continue;
        }
        if (e.latest == null || w.labeled)
          continue;
        // Resolve only at reliable observed/interpolated center crossing, never at a projected
        // future point.
        int from = Math.max(w.end + 1, e.latest.turn() - 8);
        for (int t = from; t <= e.latest.turn(); t++) {
          var sample = e.history.at(t);
          if (sample.isPresent() && e.history.at(t - 1).isPresent()
              && w.inner(t - 1) < w.origin.distance(e.history.at(t - 1).get().body().p())
              && w.inner(t) >= w.origin.distance(sample.get().body().p())) {
            Learner.trainGun(e, w, sample.get(), ++sequence);
            break;
          }
        }
      } else if (w.role == Waves.Role.INCOMING && !w.labeled
          && w.phase(own.p(), turn) == Waves.Phase.FULLY_PASSED) {
        Learner.Enemy e = enemies.get(w.owner);
        if (e != null) {
          if (w.confidence >= .99 && w.start == w.end && w.originError == 0) {
            e.shots++;
            if (w.hit)
              e.rawHits++;
            if (w.contact)
              e.rawContacts++;
            if (w.center != null) {
              var b = w.center;
              double gf = Physics.gf(w.origin.bearing(b.body().p()), w.base, w.sign,
                  Physics.escape(w.powerHi), Physics.escape(w.powerHi));
              e.visits.add(new Memory.Observation(w.features, Memory.Label.THEORETICAL_GF, gf, 0,
                  w.id, round, b.turn(), w.originQuality * .5, ++sequence, w.melee));
            }
          }
          w.labeled = true;
        }
      }
    }
    for (var list : waves.waves.values())
      list.removeIf(w
          -> w.cancelled || w.role == Waves.Role.FORECAST && turn > w.end + 2
              || w.labeled && turn > w.end + 180 || turn > w.end + 512);
  }
}
