import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;
import java.util.*;
/** Sole official-API boundary. No spectator state, filesystem models or opponent internals. */
public final class Facet extends Bot {
  final ArrayList<BotEvent> batch = new ArrayList<>();
  Core core;
  int roundSeen = -1;
  Diagnostics diagnostics = new Diagnostics();
  int games;
  final Set<Integer> completed = new HashSet<>();
  boolean gameClosed;
  // batch is the single owner monitor for every Core/lifecycle/command transaction.
  final Set<Integer> observedTerminal = new HashSet<>();
  final List<Diagnostics> retired = new ArrayList<>();
  Frame latest;
  int activeRound = -1;
  Thread runOwner;
  int terminalRound = -1, terminalExpectedTurn = -1;
  boolean gameEndRequested;
  java.util.function.Consumer<Core> hashRound = ModelSnapshot::roundHash;
  record Frame(int game, int round, int turn, Physics.Body own, double energy, int population,
      List<BulletState> bullets, int myId) {}
  Frame capture(int round) {
    return new Frame(games, round, getTurnNumber(),
        new Physics.Body(new Physics.V(getX(), getY()), Math.toRadians(getDirection()), getSpeed(),
            Math.toRadians(getGunDirection()), Math.toRadians(getRadarDirection()), getGunHeat()),
        getEnergy(), getEnemyCount(), List.copyOf(getBulletStates()), getMyId());
  }
  void retryEvidence() {
    diagnostics.retryPending();
    if (core != null)
      completed.addAll(diagnostics.completed);
    var it = retired.iterator();
    while (it.hasNext()) {
      Diagnostics old = it.next();
      old.retryPending();
      if (old.pending.isEmpty()) {
        old.close();
        it.remove();
      }
    }
  }
  public Facet() {
    this(resolveServer(System.getProperty("server.url"), System.getenv("SERVER_URL")));
  }
  Facet(java.net.URI endpoint) {
    super(BotInfo.fromResourceFile("/Facet.json"), endpoint);
  }
  public static void main(String[] args) {
    if (args.length == 1 && args[0].equals("--verify-package")) {
      verifyPackage();
      return;
    }
    new Facet().start();
  }
  static java.net.URI resolveServer(String property, String env) {
    return validateServer(property != null ? property : env);
  }
  static java.net.URI validateServer(String value) {
    if (value == null || value.isBlank())
      throw new IllegalArgumentException(
          "Explicit isolated SERVER_URL required; interactive default forbidden");
    java.net.URI uri = java.net.URI.create(value);
    if (!Set.of("ws", "wss").contains(uri.getScheme()) || uri.getHost() == null
        || (uri.getPort() <= 0 || uri.getPort() > 65535 || uri.getPort() == 7654)
        || uri.getUserInfo() != null || uri.getFragment() != null)
      throw new IllegalArgumentException("Isolated ws/wss endpoint required; port 7654 forbidden");
    return uri;
  }
  static void verifyPackage() {
    RuntimeConfig.verify();
    try {
      BotInfo info = BotInfo.fromResourceFile("/Facet.json");
      java.util.Properties p = new java.util.Properties();
      try (var stream = Facet.class.getResourceAsStream("/facet.properties")) {
        if (stream == null)
          throw new IllegalStateException("missing config");
        p.load(stream);
      }
      if (!info.getName().equals("Facet") || !p.getProperty("model.schema").equals("2")
          || !Melee.REVIEW_APPROVED || !p.getProperty("melee.enabled").equals("true")
          || !info.getGameTypes().containsAll(Set.of("1v1", "melee")))
        throw new IllegalStateException("package identity/gate mismatch");
      System.out.println("Facet 0.1.0 package verified; Java17 API1.4.0 melee=true "
          + "motion=approved-fixture-replay meleeReview=PENDING");
    } catch (java.io.IOException e) {
      throw new IllegalStateException(e);
    }
  }
  @Override
  public void onGameStarted(GameStartedEvent e) {
    synchronized (batch) {
      retryEvidence();
      if (games++ > 0) {
        diagnostics.close();
        if (!diagnostics.pending.isEmpty()) {
          if (retired.size() >= 16)
            throw new IllegalStateException("pending diagnostic game capacity exhausted");
          retired.add(diagnostics);
        }
        diagnostics = new Diagnostics();
      }
      completed.clear();
      observedTerminal.clear();
      gameClosed = false;
      diagnostics.gameId = "game-" + games + "-" + java.util.UUID.randomUUID();
      batch.clear();
      latest = null;
      terminalRound = terminalExpectedTurn = -1;
      gameEndRequested = false;
      runOwner = null;
      activeRound = roundSeen = -1;
      diagnostics.event("game.cold", 0, 0, "game=" + games);
      core = null;
    }
  }
  @Override
  public void onRoundStarted(RoundStartedEvent e) {
    synchronized (batch) {
      int round = e.getRoundNumber();
      if (round <= activeRound) {
        diagnostics.critical("lifecycle.staleReset", round, 0, "active=" + activeRound);
        return;
      }
      // Finish from the previous immutable frame before resetting any transients.
      finishRound();
      if (!batch.isEmpty()) {
        diagnostics.critical(
            "lifecycle.unattributedEvents", activeRound, 0, "count=" + batch.size());
        throw new IllegalStateException("cannot reset unobserved terminal events");
      }
      activeRound = round;
      terminalRound = terminalExpectedTurn = -1;
      latest = null;
      runOwner = null;
      if (core != null) {
        core.roundReset(round);
        roundSeen = round;
      }
      retryEvidence();
    }
  }
  @Override
  public void onTick(TickEvent e) {
    acceptTick(e, getMyId());
  }
  void acceptTick(TickEvent e, int myId) {
    synchronized (batch) {
      if (gameClosed || e.getRoundNumber() < activeRound
          || observedTerminal.contains(e.getRoundNumber())) {
        diagnostics.critical(
            "lifecycle.lateTick", e.getRoundNumber(), e.getTurnNumber(), "active=" + activeRound);
        return;
      }
      if (activeRound != -1 && e.getRoundNumber() != activeRound)
        throw new IllegalStateException("tick without ordered round start");
      activeRound = e.getRoundNumber();
      BotState state = e.getBotState();
      Frame frame = new Frame(games, activeRound, e.getTurnNumber(),
          new Physics.Body(new Physics.V(state.getX(), state.getY()),
              Math.toRadians(state.getDirection()), state.getSpeed(),
              Math.toRadians(state.getGunDirection()), Math.toRadians(state.getRadarDirection()),
              state.getGunHeat()),
          state.getEnergy(), state.getEnemyCount(), List.copyOf(e.getBulletStates()), myId);
      if (latest == null || frame.turn() >= latest.turn())
        latest = frame;
      batch.addAll(List.copyOf(e.getEvents()));
      completeRequested();
    }
  }
  void requestRoundFinish(int round, int turn) {
    synchronized (batch) {
      if (core == null || core.round != round || observedTerminal.contains(round))
        return;
      terminalRound = round;
      terminalExpectedTurn = Math.max(terminalExpectedTurn, turn);
      completeRequested();
    }
  }
  void completeRequested() {
    if (core != null && terminalRound == core.round && latest != null && latest.game() == games
        && latest.round() == terminalRound && latest.turn() >= terminalExpectedTurn)
      finishRound(terminalRound);
    if (gameEndRequested && core != null && observedTerminal.contains(core.round))
      finishGame(() -> {});
  }
  void finishRound() {
    synchronized (batch) {
      if (core != null)
        finishRound(core.round);
    }
  }
  void finishRound(int round) {
    synchronized (batch) {
      retryEvidence();
      if (core == null || observedTerminal.contains(round) || round != core.round)
        return;
      Frame frame = latest;
      if (frame == null || frame.game() != games || frame.round() != round
          || terminalRound == round && frame.turn() < terminalExpectedTurn) {
        // No replacement-round getters: absence of a legal frame is explicit.
        diagnostics.count("terminal.missingFrame");
        diagnostics.critical("completion.missingFrame", round, 0,
            "terminal not observed; expectedTurn=" + terminalExpectedTurn + " events="
                + batch.stream()
                    .map(x
                        -> x.getClass().getSimpleName() + ":" + x.getTurnNumber() + ":"
                            + core.identity(x))
                    .toList());
        batch.clear();
        observedTerminal.add(round);
        diagnostics.terminalObservation = "missing-frame";
        diagnostics.terminalHash = "not-attempted";
        diagnostics.snapshot(round, core.fire, core.budget, core.waves, core.planner);
        completed.addAll(diagnostics.completed);
        return;
      }
      finishRound(round, frame.own(), frame.energy(), frame.turn(), frame.population(),
          frame.bullets(), frame.myId());
    }
  }
  void finishRound(int round, Physics.Body own, double energy, int turn, int population,
      Collection<BulletState> bullets, int myId) {
    synchronized (batch) {
      retryEvidence();
      if (core == null || observedTerminal.contains(round) || round != core.round)
        return;
      List<BotEvent> events = List.copyOf(batch);
      // At-most-once attempt: a partially failed observation must not be replayed.
      observedTerminal.add(round);
      batch.clear();
      diagnostics.terminalTurn = turn;
      diagnostics.terminalSequence = core.sequence;
      diagnostics.terminalObservation = "success";
      diagnostics.terminalHash = "success";
      try {
        core.observe(own, energy, turn, population, events, List.copyOf(bullets), myId, true);
      } catch (RuntimeException ex) {
        diagnostics.terminalObservation = "failed-quarantined";
        diagnostics.failure("terminal.observe", ex);
        diagnostics.critical("terminal.quarantined", round, turn,
            "own=" + own + " energy=" + energy + " population=" + population + " events="
                + events.stream()
                    .map(x
                        -> x.getClass().getSimpleName() + ":" + x.getTurnNumber() + ":"
                            + core.identity(x))
                    .toList());
      } finally {
        Physics.endWork();
      }
      try {
        hashRound.accept(core);
      } catch (RuntimeException ex) {
        diagnostics.terminalHash = "failed";
        diagnostics.failure("terminal.hash", ex);
        diagnostics.critical("terminal.hashFailed", round, turn, ex.toString());
      }
      // Freeze cumulative metrics now; future retries touch only immutable row bytes.
      diagnostics.snapshot(round, core.fire, core.budget, core.waves, core.planner);
      completed.addAll(diagnostics.completed);
    }
  }
  /** Public API queue includes undelivered TickEvents. Lifecycle precedes final flush in 1.4.0. */
  void acceptPendingTicks(int round, List<BotEvent> queued, int myId) {
    synchronized (batch) {
      for (BotEvent event : queued)
        if (event instanceof TickEvent tick && tick.getRoundNumber() == round)
          acceptTick(tick, myId);
    }
  }
  void ensureCore(int round) {
    if (core == null) {
      core = new Core(new Physics.Rules(getArenaWidth(), getArenaHeight(), getGunCoolingRate()),
          diagnostics, seed());
      core.roundReset(round);
      roundSeen = activeRound = round;
    }
  }
  void harvestTerminal(int round) {
    // Lifecycle delivery is ordered on the API websocket; a replacement round must never be read.
    if (getRoundNumber() == round)
      acceptPendingTicks(round, List.copyOf(getEvents()), getMyId());
  }
  @Override
  public void onRoundEnded(RoundEndedEvent e) {
    synchronized (batch) {
      if (!gameClosed)
        ensureCore(e.getRoundNumber());
      if (core != null && core.round == e.getRoundNumber()
          && !observedTerminal.contains(e.getRoundNumber()))
        harvestTerminal(e.getRoundNumber());
      requestRoundFinish(e.getRoundNumber(), e.getTurnNumber());
    }
  }
  void finishGame(Runnable finalRound) {
    synchronized (batch) {
      retryEvidence();
      if (gameClosed)
        return;
      finalRound.run();
      if (core != null) {
        try {
          ModelSnapshot.optional(core, games);
        } catch (RuntimeException ex) {
          diagnostics.failure("terminal.export", ex);
        }
      }
      gameClosed = true;
      runOwner = null;
      diagnostics.close();
    }
  }
  @Override
  public void onGameEnded(GameEndedEvent e) {
    synchronized (batch) {
      gameEndRequested = true;
      if (core != null && !observedTerminal.contains(core.round)) {
        int round = core.round, turn = getTurnNumber();
        harvestTerminal(round);
        requestRoundFinish(round, turn);
      } else {
        finishGame(this::finishRound);
      }
    }
  }

  /** Owner transaction used by the real loop and independent concurrency fixtures. */
  void decideFrame(Frame frame, java.util.function.Consumer<Core.Intent> submit) {
    synchronized (batch) {
      if (frame.game() != games || gameClosed || core == null || frame.round() != core.round
          || terminalRound == frame.round() || observedTerminal.contains(frame.round()))
        return;
      List<BotEvent> events = List.copyOf(batch);
      batch.clear();
      try {
        Core.Intent intent = core.decide(frame.own(), frame.energy(), frame.turn(),
            frame.population(), events, frame.bullets(), frame.myId());
        submit.accept(intent);
      } catch (RuntimeException ex) {
        diagnostics.failure("boundary", ex);
        core.fire.plan = null;
        throw ex;
      } finally {
        Physics.endWork();
      }
    }
  }
  @Override
  public void run() {
    final int game;
    final Thread owner = Thread.currentThread();
    synchronized (batch) {
      if (gameClosed)
        return;
      game = games;
      runOwner = owner;
      setAdjustGunForBodyTurn(true);
      setAdjustRadarForBodyTurn(true);
      setAdjustRadarForGunTurn(true);
      setFireAssist(false);
      if (core == null)
        core = new Core(new Physics.Rules(getArenaWidth(), getArenaHeight(), getGunCoolingRate()),
            diagnostics, seed());
      if (roundSeen != getRoundNumber()) {
        core.roundReset(getRoundNumber());
        activeRound = roundSeen = getRoundNumber();
      }
    }
    while (isRunning()) {
      synchronized (batch) {
        if (games != game || runOwner != owner || gameClosed)
          return;
        Frame frame = latest != null ? latest : capture(core.round);
        if (frame.round() != core.round || terminalRound == frame.round()
            || observedTerminal.contains(frame.round()))
          return;
        diagnostics.event("own.commandState", frame.round(), frame.turn(),
            "distanceRemaining=" + getDistanceRemaining() + " bodyRemainingDeg="
                + getTurnRemaining() + " gunRemainingDeg=" + getGunTurnRemaining()
                + " radarRemainingDeg=" + getRadarTurnRemaining()
                + " targetSpeed=" + getTargetSpeed() + " turnRateDeg=" + getTurnRate()
                + " gunRateDeg=" + getGunTurnRate() + " radarRateDeg=" + getRadarTurnRate());
        try {
          setFire(0);
          decideFrame(frame, intent -> {
            setTargetSpeed(intent.move().speed());
            setTurnRate(Math.toDegrees(intent.move().turn()));
            if (intent.fire()) {
              boolean accepted = setFire(core.fire.plan.power());
              diagnostics.event("fire.submitted", frame.round(), frame.turn(),
                  "accepted=" + accepted + " " + core.fire.plan.evidenceRow());
              core.fire.submitted(accepted, frame.own(), frame.turn(), ++core.sequence);
            }
            setGunTurnRate(Math.toDegrees(intent.gunTurn()));
            setRadarTurnRate(Math.toDegrees(intent.radarTurn()));
            diagnostics.event("command.submitted", frame.round(), frame.turn(),
                "sequence=" + (++core.sequence) + " own=" + frame.own() + " speed="
                    + getTargetSpeed() + " turn=" + getTurnRate() + " gun=" + getGunTurnRate()
                    + " radar=" + getRadarTurnRate() + " fire=" + intent.fire() + " firepower="
                    + (intent.fire() && core.fire.pending != null ? core.fire.pending.power() : 0));
          });
        } catch (RuntimeException ex) {
          setFire(0);
          setTargetSpeed(0);
          setTurnRate(0);
          setGunTurnRate(0);
          setRadarTurnRate(45);
        }
        retryEvidence();
      }
      // API sends/waits/dispatches callbacks here: NEVER hold the owner monitor.
      go();
    }
  }
  static long seed() {
    String value = System.getenv("FACET_SEED");
    return value == null ? RuntimeConfig.SEED : Long.parseLong(value);
  }
}
