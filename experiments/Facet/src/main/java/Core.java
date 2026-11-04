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
  int tracePopulation = -1;
  int survivorSince = -1;
  Waves.Wave shotContext;
  int cohortEvidenceSamples;
  record Intent(Physics.Control move, double gunTurn, double radarTurn, boolean fire) {}
  Core(Physics.Rules r, Diagnostics log, long seed) {
    rules = r;
    this.log = log;
    melee = new Melee(new SplittableRandom(seed).split().nextLong());
    ownContext = new Learner.Enemy(-1, budget);
    RuntimeConfig.verify();
    log.event("profile", 0, 0,
        "independent-v1 seed=" + seed
            + " motion=official-fixture-replay melee=true meleeReview=PENDING noCrossGameLoad");
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
    tracePopulation = -1;
    survivorSince = -1;
    lastTurn = -1;
    shotContext = null;
    cohortEvidenceSamples = 0;
    melee.recent.clear();
    melee.retained = null;
    melee.resetAim();
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
    melee.beginDecision();
    Memory.displacementWork.get().reset();
    long begin = System.nanoTime();
    observe(own, energy, turn, population, events, bullets, myId, false);
    for (var e : enemies.values()) e.observedDecision(turn);
    Learner.Enemy target =
        enemies.values()
            .stream()
            .filter(e -> e.alive && e.latest != null)
            .min(Comparator.<Learner.Enemy>comparingInt(e -> turn - e.latest.turn())
                    .thenComparingDouble(e -> own.p().distance(e.latest.body().p()))
                    .thenComparingInt(e -> e.id))
            .orElse(null);
    if (target != null && own.heat() == 0
        && Energy.power(
               own.p().distance(target.latest.body().p()), energy, target.latest.energy(), false)
            > 0
        && turn - target.latest.turn() <= 2) {
      target.opportunities.add(turn);
      while (target.opportunities.size() > 512) target.opportunities.pollFirst();
    }
    // Fresh observations replace obsolete forecast geometry; missing scans create no new schedule.
    for (var list : waves.waves.values())
      list.removeIf(w
          -> w.role == Waves.Role.FORECAST
              && (turn > w.end + 2
                  || enemies.containsKey(w.owner) && enemies.get(w.owner).latest != null
                      && enemies.get(w.owner).latest.turn() > w.creationTurn));
    // At most one cold forecast per enemy, replaced by any actual inference.
    if (target != null && population == 1 && target.latest.turn() == turn
        && target.latest.energy() > .1 && target.ledger.heatLow <= rules.cooling()
        && waves.all()
                .stream()
                .filter(w -> w.role == Waves.Role.INCOMING || w.role == Waves.Role.FORECAST)
                .count()
            < 2
        && waves.all().stream().noneMatch(
            w -> w.owner == target.id && w.role == Waves.Role.FORECAST)) {
      double p = predictPower(
          target, own.p().distance(target.latest.body().p()), energy, target.latest.energy());
      if (p > 0) {
        int early = turn + coolSteps(target.ledger.heatLow, rules.cooling()),
            late = turn + coolSteps(target.ledger.heatHigh, rules.cooling());
        Learner.Context c = context(target.latest.body().p(), own, ownContext, turn, p, rules,
            population, initialCount, false, originOrbit(target, turn));
        Waves.Wave forecast = new Waves.Wave(waves.nextId++, target.id, myId, round, early, late,
            Waves.Role.FORECAST, target.latest.body().p(), 8, p, p,
            target.latest.body().p().bearing(own.p()), originOrbit(target, turn), Physics.escape(p),
            Physics.escape(p), .25, c.features(), own, beganMelee);
        forecast.creationTurn = turn;
        waves.add(forecast);
      }
    }
    // Certified own-bullet shadows were reconciled before movement scoring.
    boolean meleeMode = population > 1 && Melee.REVIEW_APPROVED;
    long start = System.nanoTime();
    planner.safetyTurn = turn;
    Physics.Control movement;
    try {
      planner.stepLimit = Math.max(0, Math.min(2400, 2400 - Physics.workUsed() - 350));
      planner.activeEnemies = enemies.values();
      if (meleeMode && !log.disabled.contains("movement")) {
        long beforeFallback = melee.unsafeFallbacks;
        movement = melee.move(own, energy, turn, enemies.values(), rules, planner);
        if (melee.unsafeFallbacks != beforeFallback)
          log.count("movement.emergency");
      } else {
        Planner.Choice choice = log.disabled.contains("movement")
            ? planner.emergency(own, rules)
            : planner.choose(own, turn, waves.all(), enemies, rules, waves.deaths);
        movement = choice.command();
        if (choice.emergency())
          log.count("movement.emergency");
      }
      if (meleeMode && waves.realOverflow > 0) {
        movement = planner.emergency(own, rules).command();
        log.count("movement.emergency");
      }
      log.success("movement");
    } catch (RuntimeException ex) {
      log.failure("movement", ex);
      movement = planner.emergency(own, rules).command();
    }
    log.time("movement", System.nanoTime() - start);
    if (population > 1 && !meleeMode) {
      log.count("melee.gated");
      fire.plan = null;
    }
    Learner.Enemy firingTarget = fire.plan == null ? null : enemies.get(fire.plan.target());
    boolean gunFailed = false;
    boolean submit = (population == 1 || meleeMode) && fire.plan != null
        && fire.plan.policy().population() == population
        && (population != 1
            || firingTarget != null && firingTarget.latest != null
                && firingTarget.latest.turn() >= survivorSince)
        && fire.valid(own, energy, firingTarget, turn, rules);
    if (submit) {
      try {
        shotContext = makeOutgoing(
            own, firingTarget, turn, fire.plan.power(), Waves.Role.OUTGOING, myId, true);
        // The combined melee gun learns crossings, but has no duel comparison cohort.
        // Resume the ensemble only in the fresh-scan survivor duel.
        if (population == 1) {
          shotContext.preparedCohort = fire.plan.cohort();
          if (shotContext.preparedCohort != null)
            shotContext.predictions.putAll(shotContext.preparedCohort.bind(shotContext.id));
          else
            log.count("gun.cohort.missing"); // Legacy/manual plans are explicitly unrated.
        }
      } catch (RuntimeException ex) {
        log.failure("gun", ex);
        gunFailed = true;
        shotContext = null;
        submit = false;
        fire.plan = null;
      }
    }
    if (!meleeMode && waves.realOverflow > 0)
      movement = Physics.Control.stop();
    Physics.Body nextOwn = Physics.step(own, movement, rules).body();
    start = System.nanoTime();
    double gunTurn = 0;
    try {
      if (target != null) {
        FireControl.Cohort prepared = null;
        double power = Energy.power(own.p().distance(target.latest.body().p()), energy,
                   target.latest.energy(), meleeMode),
               angle = nextOwn.p().bearing(target.latest.body().p());
        if (power > 0 && (population == 1 || meleeMode) && target.latest.turn() <= turn
            && turn - target.latest.turn() <= 2
            && (population != 1 || target.latest.turn() >= survivorSince)) {
          if (meleeMode && !log.disabled.contains("gun")) {
            Physics.Body aimState = new Physics.Body(nextOwn.p(), nextOwn.heading(),
                nextOwn.speed(), nextOwn.gun(), nextOwn.radar(), own.heat());
            Melee.Aim aim = melee.scheduledAim(
                aimState, new ArrayList<>(enemies.values()), energy, turn, population, rules);
            if (aim != null) {
              angle = aim.angle();
              power = aim.power();
              Learner.Enemy aimTarget = enemies.get(aim.target());
              if (!submit)
                fire.plan = aim.computedTurn() == turn
                    ? FireControl.prepare(
                          angle, power, aimTarget, nextOwn, turn, sequence, aim.policy())
                    : null;
            } else if (!submit)
              fire.plan = null;
            if (own.heat() <= 1e-10)
              for (Learner.Enemy living : enemies.values())
                if (living.alive && living.latest != null && turn - living.latest.turn() <= 8)
                  waves.add(makeOutgoing(
                      nextOwn, living, turn + 1, power, Waves.Role.VIRTUAL, myId, false));
          } else if (population == 1 && own.heat() / rules.cooling() <= 3
              && !log.disabled.contains("gun")) {
            Waves.Wave virtual = makeOutgoing(nextOwn, target, turn + 1, power, Waves.Role.VIRTUAL,
                myId, own.heat() <= rules.cooling());
            for (String name : List.of("analytic", "long", "recent"))
              if (name.equals("analytic")
                  || (target.crossings >= 16 && !log.disabled.contains("gun")))
                virtual.predictions.put(name,
                    new Waves.Prediction(name, virtual.id, turn,
                        Learner.gun(name, target, virtual, own.gun(), rules), power,
                        beganMelee ? target.displacement.version : target.longGun.version));
            angle = virtual.predictions
                        .getOrDefault(target.selected, virtual.predictions.get("analytic"))
                        .angle();
            if (!submit)
              prepared = FireControl.Cohort.from(virtual,
                  virtual.predictions.containsKey(target.selected) ? target.selected : "analytic",
                  turn);
            if (own.heat() <= 1e-10 && !submit)
              waves.add(virtual);
          } else
            angle = Physics.intercept(nextOwn.p(), target.latest.body(), power, rules);
          // Keep prior plan until request has been submitted by boundary.
          if (!submit && !meleeMode) {
            fire.plan = FireControl.prepare(angle, power, target, nextOwn, turn, sequence);
            if (prepared == null)
              prepared = FireControl.Cohort.analytic(target.id, beganMelee, turn, nextOwn.p(),
                  target.latest.body(), angle, power,
                  beganMelee ? target.displacement.version : target.longGun.version);
            fire.plan = fire.plan.withCohort(prepared);
          }
          if (!submit && meleeMode && log.disabled.contains("gun"))
            fire.plan = FireControl.prepare(angle, power, target, nextOwn, turn, sequence,
                new FireControl.Policy(population, target.id,
                    nextOwn.p().distance(
                        Physics.interceptPoint(nextOwn.p(), target.latest.body(), power, rules)),
                    target.latest.energy(), target.latest.turn()));
          if (!submit && fire.plan != null) {
            var actual = enemies.get(fire.plan.target());
            String selected = fire.plan.cohort() != null ? fire.plan.cohort().selected()
                : meleeMode && !log.disabled.contains("gun") ? "combinedmelee"
                : own.heat() / rules.cooling() <= 3 && !log.disabled.contains("gun")
                    && (actual.selected.equals("analytic") || actual.crossings >= 16)
                ? actual.selected
                : "analytic";
            fire.plan = fire.plan.withEvidence("gun=" + selected + " mode="
                + (meleeMode         ? "melee"
                        : beganMelee ? "survivor-duel"
                                     : "duel")
                + " beganMelee=" + beganMelee + " label="
                + (selected.equals("combinedmelee")
                        ? melee.cached != null && melee.cached.displacement() ? "DISPLACEMENT"
                                                                              : "ANALYTIC"
                        : selected.equals("analytic") ? "ANALYTIC"
                        : beganMelee                  ? "DISPLACEMENT"
                                                      : "PRECISE_GF")
                + " " + stores(actual));
            log.event("fire.prepared", round, turn, fire.plan.evidenceRow());
          }
        } else if (!submit)
          fire.plan = null;
        gunTurn =
            Physics.clamp(Physics.wrap(angle - own.gun()), -Math.toRadians(20), Math.toRadians(20));
      }
      if (!gunFailed)
        log.success("gun");
    } catch (RuntimeException ex) {
      if (!gunFailed)
        log.failure("gun", ex);
      else
        log.count("gun.additionalFailureSameDecision");
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
    log.observe("shadowSegmentWork", waves.shadowWork);
    log.observe("shadowAllocationUnits", waves.shadowAllocations);
    log.observe("shadowIntervals", waves.shadowIntervals);
    log.observe("realWaveOverflow", waves.realOverflow);
    log.observe("predictorSteps", Physics.workUsed());
    log.observe("meleeProjectionPasses", melee.projectionPasses);
    log.observe("meleeUnsafeFallbacks", melee.unsafeFallbacks);
    log.counters.merge("melee.projectionSamples", melee.decisionSamples, Long::sum);
    log.counters.merge("melee.queryReturnedSamples", melee.decisionReturned, Long::sum);
    log.counters.merge("melee.neighborQueries", melee.decisionQueries, Long::sum);
    log.counters.merge("melee.projectionPasses", melee.decisionPasses, Long::sum);
    log.observe("meleeDecisionSamples", melee.decisionSamples);
    log.observe("meleeDecisionReturned", melee.decisionReturned);
    log.observe("meleeDecisionQueries", melee.decisionQueries);
    log.observe("meleeDecisionPasses", melee.decisionPasses);
    var displacementWork = Memory.displacementWork.get();
    if (population > 1) {
      log.observe("meleeTotalReturned", displacementWork.returned);
      log.observe("meleeTotalProjections", displacementWork.projections);
      log.observe("meleeTotalIterations", displacementWork.iterations);
      log.observe("meleeTotalDistanceEvaluations", displacementWork.distances);
    }
    if (meleeMode)
      log.event("melee.work", round, turn,
          "dispatches=" + melee.dispatches + " destinations=" + melee.destinations + " candidates="
              + melee.candidateChecks + " selectedSteps=" + melee.selectedSteps + " quota="
              + melee.quotaFallbacks + " allUnsafe=" + melee.unsafeEmergency + " wallReject="
              + melee.wallRejects + " botReject=" + melee.botRejects + " projections="
              + melee.projectionPasses + " samples=" + melee.projectionWork.samples
              + " solveIterations=" + melee.projectionWork.iterations + " decisionSamples="
              + melee.decisionSamples + " decisionReturned=" + melee.decisionReturned
              + " decisionQueries=" + melee.decisionQueries + " decisionPasses="
              + melee.decisionPasses + " decisionIterations=" + melee.decisionIterations
              + " predictor=" + Physics.workUsed() + " totalDisplacementQueries="
              + displacementWork.queries + " totalDisplacementReturned=" + displacementWork.returned
              + " totalDisplacementProjections=" + displacementWork.projections
              + " totalDisplacementIterations=" + displacementWork.iterations
              + " totalDisplacementDistances=" + displacementWork.distances
              + " movementDisabled=" + log.disabled.contains("movement")
              + " gunDisabled=" + log.disabled.contains("gun") + " context=" + melee.moveContext);
    log.observe("waves", waves.all().size());
    Physics.finite(movement.speed(), movement.turn(), gunTurn, radarTurn);
    log.time("decision", System.nanoTime() - begin);
    Physics.endWork();
    return new Intent(movement, gunTurn, radarTurn, submit);
  }
  void observe(Physics.Body own, double energy, int turn, int population, List<BotEvent> events,
      Collection<BulletState> bullets, int myId, boolean terminal) {
    Physics.beginWork(2400);
    String beforeStores = tracePopulation != population ? storeRows() : "";
    if (tracePopulation > 1 && population == 1) {
      survivorSince = turn;
      fire.plan = null;
      melee.resetAim();
    }
    if (initialCount == 0) {
      initialCount = Math.max(1, population);
      beganMelee = population > 1;
    }
    if (lastTurn >= 0 && turn != lastTurn && turn != lastTurn + 1) {
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
    for (var e : enemies.values()) recordOrbit(e, own, turn);
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
            if (p.cohort() != null && p.cohort() == frozen.preparedCohort) {
              w.preparedCohort = p.cohort();
              w.predictions.putAll(p.cohort().bind(w.id));
            } else if (p.cohort() != null || frozen.preparedCohort != null) {
              log.failure("gun.cohort", new IllegalStateException("accepted cohort binding differs"));
            }
            w.creationTurn = t;
            w.firstMotionTurn = t;
            w.bullet = b.getBulletId();
            w.actualAngle = Math.toRadians(b.getDirection());
            w.ratingEligible = !w.predictions.isEmpty();
            if (w.preparedCohort != null) {
              var c = w.preparedCohort;
              double error = Math.abs(Physics.wrap(
                  w.actualAngle - c.predictions().get(c.selected()).angle()));
              log.count("gun.cohort.confirmed");
              log.count(w.predictions.size() == 3 ? "gun.cohort.common" : "gun.cohort.analyticOnly");
              log.observe("gun.cohort.headingErrorRadians", error);
              log.observe("gun.cohort.requestAge", p.requestedTurn() - c.preparedTurn());
              String evidence = "physicalWave=" + w.id + " request=" + p.requestedTurn()
                  + " creation=" + t + " actualAngle=" + w.actualAngle
                  + " actualOrigin=" + w.origin + " physicalMask=" + w.missingFeatures
                  + " physicalFeatures=" + Arrays.toString(w.features) + " " + c.evidenceRow();
              log.event("gun.cohort.confirmed", round, t, evidence);
              if (cohortEvidenceSamples < 16) {
                cohortEvidenceSamples++;
                log.critical("gun.cohort.confirmed", round, t, evidence);
              }
            }
            waves.add(w);
            shotContext = null;
          }
        }
        log.event("fire.confirmed", round, t,
            "id=" + b.getBulletId() + " heading=" + b.getDirection() + " position=" + b.getX() + ","
                + b.getY() + " pending=" + pending.isPresent() + " request="
                + pending.map(x -> Integer.toString(x.requestedTurn())).orElse("unmatched")
                + " creation=" + t + " firstMotion=" + t
                + " selected=" + pending.map(FireControl.Pending::evidence).orElse("unmatched"));
      }
    }
    for (BulletState b : bullets) {
      if (b.getOwnerId() == myId && waves.bullets.containsKey(b.getBulletId())
          && !waves.deaths.containsKey(b.getBulletId())) {
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
        waves.death(b.id(), b.observedTurn() + 1);
        log.count("bullet.absentReconciled");
      }
    while (waves.bullets.size() > Waves.Service.MAX_OWN_BULLETS) {
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
        if (t > turn || !e.alive || e.latest != null && t <= e.latest.turn())
          continue;
        Physics.Body observed = new Physics.Body(new Physics.V(scan.getX(), scan.getY()),
            Math.toRadians(scan.getDirection()), scan.getSpeed(), 0, 0, 0);
        Memory.Sample previous = e.latest;
        e.observe(
            new Memory.Sample(round, t, observed, scan.getEnergy(), Memory.Quality.OBSERVED, 0),
            own.p());
        var ownAtScan = ownHistory.samples.get(t);
        if (ownAtScan != null)
          recordOrbit(e, ownAtScan.body(), t);
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
          Memory.Sample atOrigin = ownHistory.samples.get(in.earliest());
          var prefix = ownHistory.samples.floorEntry(in.earliest());
          Physics.Body targetAtOrigin = atOrigin != null ? atOrigin.body()
              : prefix != null
              ? prefix.getValue().body()
              : new Physics.Body(
                    new Physics.V(rules.width() / 2, rules.height() / 2), 0, 0, 0, 0, 0);
          Learner.Context context = context(origin.p(), targetAtOrigin, ownContext, in.earliest(),
              in.high(), rules, population, initialCount, false, originOrbit(e, in.earliest()));
          int sign = originOrbit(e, in.earliest());
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
    for (Waves.Wave w : waves.all())
      if (w.role == Waves.Role.INCOMING && w.shadows.isEmpty())
        log.event("shadow.zero", round, turn,
            "wave=" + w.id + " reduction=0 reason="
                + (w.start != w.end || w.originError != 0 || w.powerLo != w.powerHi
                        ? "continuous-uncertainty-uncertified"
                        : "no-certified-supported-interval-or-quota"));
    if (waves.realOverflow > 0)
      log.count("waves.realOverflow.conservativeStop");
    log.time("inference", System.nanoTime() - inferenceStart);
    long storageStart = System.nanoTime();
    if (!terminal && !log.disabled.contains("learning")) {
      try {
        resolve(own, turn);
        log.success("learning");
      } catch (RuntimeException ex) {
        log.failure("learning", ex);
      }
    } else
      log.count(terminal ? "learning.terminalNoLearning" : "learning.disabled");
    if (tracePopulation != population) {
      log.event("population.transition", round, turn,
          "from=" + tracePopulation + " to=" + population + " beganMelee=" + beganMelee
              + " before=" + beforeStores + " after=" + storeRows());
      tracePopulation = population;
    }
    log.time("storage", System.nanoTime() - storageStart);
    fire.reconcile(turn);
    if (fire.pending == null)
      shotContext = null;
    if (terminal) {
      fire.plan = null;
      if (fire.pending != null)
        fire.cancelled++;
      fire.pending = null;
      shotContext = null;
      for (Waves.Wave w : waves.all())
        if (!w.labeled) {
          w.cancelled = true;
          w.predictions.clear();
        }
    }
  }
  static String stores(Learner.Enemy e) {
    StringBuilder s = new StringBuilder("target=" + e.id + " alive=" + e.alive
        + " scan=" + (e.latest == null ? -1 : e.latest.turn()) + " long=" + e.displacement.size()
        + ":" + e.displacement.version + " recent=");
    for (var store : e.recentDisplacement)
      s.append(store.size()).append(':').append(store.version).append(',');
    return s.toString();
  }
  String storeRows() {
    StringJoiner s = new StringJoiner(";", "[", "]");
    for (var e : enemies.values()) s.add(stores(e));
    return s.toString();
  }
  Learner.Context context(Physics.V source, Physics.Body target, Learner.Enemy enemy, int turn,
      double power, Physics.Rules rules, int population, int initial, boolean physical) {
    return context(source, target, enemy, turn, power, rules, population, initial, physical,
        Learner.orbitSign(enemy, source, turn));
  }
  Learner.Context context(Physics.V source, Physics.Body target, Learner.Enemy enemy, int turn,
      double power, Physics.Rules rules, int population, int initial, boolean physical, int orbit) {
    if (!log.disabled.contains("context")) {
      try {
        var value = Learner.context(
            source, target, enemy, turn, power, rules, population, initial, physical, orbit);
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
  void recordOrbit(Learner.Enemy e, Physics.Body own, int turn) {
    if (e.latest == null || e.latest.turn() > turn)
      return;
    e.ownOrbit = lateralSign(e.latest.body().p(), own, e.ownOrbit);
    e.ownOrbits.put(turn, e.ownOrbit);
    while (e.ownOrbits.size() > 512) e.ownOrbits.pollFirstEntry();
  }
  static int originOrbit(Learner.Enemy e, int turn) {
    var prior = e.ownOrbits.floorEntry(turn);
    return prior == null ? 1 : prior.getValue();
  }
  static int coolSteps(double heat, double cooling) {
    int n = 0;
    while (heat > 0 && n < 4096) {
      heat = Math.max(0, heat - cooling);
      n++;
    }
    return n;
  }
  double predictPower(Learner.Enemy enemy, double range, double own, double energy) {
    if (!Double.isFinite(energy) || energy <= .1)
      return 0;
    if (!log.disabled.contains("inference")) {
      try {
        return Learner.predictPower(enemy, range, own, energy);
      } catch (RuntimeException ex) {
        log.failure("inference", ex);
      }
    } else
      log.count("power.disabled");
    return Physics.clamp(Math.min(2, Math.nextDown(energy)), .1, 3);
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
    Learner.Context c =
        context(own.p(), frozen, target, turn, p, rules, 1, initialCount, physical, target.sign);
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
              || w.labeled && turn > w.end + 180);
  }
}
