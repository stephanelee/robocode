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
          || Melee.REVIEW_APPROVED)
        throw new IllegalStateException("package identity/gate mismatch");
      System.out.println("Facet 0.1.0 package verified; Java17 API1.4.0 melee=false "
                         + "motion=approved-fixture-replay adapterReview=PENDING");
    } catch (java.io.IOException e) {
      throw new IllegalStateException(e);
    }
  }
  @Override
  public void onGameStarted(GameStartedEvent e) {
    if (games++ > 0) {
      diagnostics.close();
      diagnostics = new Diagnostics();
    }
    completed.clear();
    gameClosed = false;
    diagnostics.gameId = "game-" + games + "-" + java.util.UUID.randomUUID();
    synchronized (batch) {
      batch.clear();
    }
    diagnostics.event("game.cold", 0, 0, "game=" + games);
    core = null;
    roundSeen = -1;
  }
  @Override
  public void onRoundStarted(RoundStartedEvent e) {
    synchronized (batch) {
      batch.clear();
    }
    roundSeen = -1;
  }
  @Override
  public void onTick(TickEvent e) {
    synchronized (batch) {
      batch.addAll(e.getEvents());
    }
  }
  void finishRound() {
    if (core != null)
      finishRound(core.round);
  }
  void finishRound(int round) {
    synchronized (batch) {
      if (core == null || completed.contains(round) || round != core.round)
        return;
      Physics.Body own = new Physics.Body(new Physics.V(getX(), getY()),
          Math.toRadians(getDirection()), getSpeed(), Math.toRadians(getGunDirection()),
          Math.toRadians(getRadarDirection()), getGunHeat());
      finishRound(
          round, own, getEnergy(), getTurnNumber(), getEnemyCount(), getBulletStates(), getMyId());
    }
  }
  void finishRound(int round, Physics.Body own, double energy, int turn, int population,
      Collection<BulletState> bullets, int myId) {
    synchronized (batch) {
      if (core == null || completed.contains(round))
        return;
      core.observe(own, energy, turn, population, List.copyOf(batch), bullets, myId, true);
      batch.clear();
      Physics.endWork();
      completed.add(round);
      ModelSnapshot.roundHash(core);
      diagnostics.snapshot(round, core.fire, core.budget, core.waves, core.planner);
    }
  }
  @Override
  public void onRoundEnded(RoundEndedEvent e) {
    finishRound(e.getRoundNumber());
  }
  void finishGame(Runnable finalRound) {
    if (gameClosed)
      return;
    finalRound.run();
    if (core != null)
      ModelSnapshot.optional(core, games);
    gameClosed = true;
    diagnostics.close();
  }
  @Override
  public void onGameEnded(GameEndedEvent e) {
    finishGame(this::finishRound);
  }
  @Override
  public void run() {
    setAdjustGunForBodyTurn(true);
    setAdjustRadarForBodyTurn(true);
    setAdjustRadarForGunTurn(true);
    setFireAssist(false);
    if (core == null)
      core = new Core(new Physics.Rules(getArenaWidth(), getArenaHeight(), getGunCoolingRate()),
          diagnostics, seed());
    while (isRunning()) {
      if (roundSeen != getRoundNumber()) {
        core.roundReset(getRoundNumber());
        roundSeen = getRoundNumber();
      }
      Physics.Body own = new Physics.Body(new Physics.V(getX(), getY()),
          Math.toRadians(getDirection()), getSpeed(), Math.toRadians(getGunDirection()),
          Math.toRadians(getRadarDirection()), getGunHeat());
      diagnostics.event("own.commandState", getRoundNumber(), getTurnNumber(),
          "distanceRemaining=" + getDistanceRemaining() + " bodyRemainingDeg=" + getTurnRemaining()
              + " gunRemainingDeg=" + getGunTurnRemaining()
              + " radarRemainingDeg=" + getRadarTurnRemaining()
              + " targetSpeed=" + getTargetSpeed() + " turnRateDeg=" + getTurnRate()
              + " gunRateDeg=" + getGunTurnRate() + " radarRateDeg=" + getRadarTurnRate());
      List<BotEvent> events;
      synchronized (batch) {
        events = List.copyOf(batch);
        batch.clear();
      }
      try {
        setFire(0);
        Core.Intent intent = core.decide(own, getEnergy(), getTurnNumber(), getEnemyCount(), events,
            getBulletStates(), getMyId());
        setTargetSpeed(intent.move().speed());
        setTurnRate(Math.toDegrees(intent.move().turn()));
        if (intent.fire()) {
          boolean accepted = setFire(core.fire.plan.power());
          core.fire.submitted(accepted, own, getTurnNumber(), ++core.sequence);
        }
        setGunTurnRate(Math.toDegrees(intent.gunTurn()));
        setRadarTurnRate(Math.toDegrees(intent.radarTurn()));
        diagnostics.event("command.submitted", getRoundNumber(), getTurnNumber(),
            "sequence=" + (++core.sequence) + " own=" + own + " speed=" + getTargetSpeed()
                + " turn=" + getTurnRate() + " gun=" + getGunTurnRate()
                + " radar=" + getRadarTurnRate() + " fire=" + intent.fire() + " firepower="
                + (intent.fire() && core.fire.pending != null ? core.fire.pending.power() : 0));
      } catch (RuntimeException ex) {
        diagnostics.failure("boundary", ex);
        core.fire.plan = null;
        setFire(0);
        setTargetSpeed(0);
        setTurnRate(0);
        setGunTurnRate(0);
        setRadarTurnRate(45);
      }
      Physics.endWork();
      go();
    }
  }
  static long seed() {
    String value = System.getenv("FACET_SEED");
    return value == null ? RuntimeConfig.SEED : Long.parseLong(value);
  }
}
