import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/** Independent adapter lifecycle invariants. No battle, opponent code, or outcome oracle. */
public final class AdapterConcurrencyTest {
  static int assertions;
  static final Physics.Body OWN = new Physics.Body(new Physics.V(400, 300), 0, 0, 0, 0, 0);
  static final Physics.Rules RULES = new Physics.Rules(800, 600, .1);
  static void check(boolean v, String message) {
    assertions++;
    if (!v)
      throw new AssertionError(message);
  }
  static final class Rows implements Diagnostics.Sink {
    final List<String> rows = new ArrayList<>();
    int failures;
    public void append(String row) throws IOException {
      if (failures > 0) {
        failures--;
        throw new IOException("independent injected append failure");
      }
      rows.add(row);
    }
  }
  static Facet fixture(Rows metrics, Rows critical) {
    Facet f = new Facet(java.net.URI.create("ws://localhost:12345"));
    f.onGameStarted(null);
    f.diagnostics.close();
    f.diagnostics = new Diagnostics(metrics, critical);
    f.diagnostics.gameId = "independent-game-" + f.games;
    f.core = new Core(RULES, f.diagnostics, 1);
    f.core.roundReset(1);
    f.activeRound = f.roundSeen = 1;
    f.latest = frame(f, 1, 40);
    f.batch.add(new ScannedBotEvent(40, 0, 1, 100, 600, 300, 0, 0));
    return f;
  }
  static Facet.Frame frame(Facet f, int round, int turn) {
    return new Facet.Frame(f.games, round, turn, OWN, 100, 1, List.of(), 0);
  }
  static void await(CountDownLatch latch) {
    try {
      if (!latch.await(5, TimeUnit.SECONDS))
        throw new AssertionError("barrier timeout");
    } catch (InterruptedException ex) {
      throw new AssertionError(ex);
    }
  }
  static Thread worker(AtomicReference<Throwable> failure, Runnable body) {
    Thread t = new Thread(() -> {
      try {
        body.run();
      } catch (Throwable ex) {
        failure.compareAndSet(null, ex);
      }
    });
    t.start();
    return t;
  }
  static void blocked(Thread t) {
    long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
    while (t.getState() != Thread.State.BLOCKED && System.nanoTime() < end) Thread.yield();
    check(t.getState() == Thread.State.BLOCKED, "callback contends on real owner monitor");
  }
  static void joined(Thread t, AtomicReference<Throwable> failure) {
    try {
      t.join(5000);
    } catch (InterruptedException ex) {
      throw new AssertionError(ex);
    }
    check(!t.isAlive(), "no monitor deadlock");
    if (failure.get() != null)
      throw new AssertionError("worker failed", failure.get());
  }
  static void concurrency() {
    for (boolean reset : new boolean[] {false, true}) {
      Rows rows = new Rows(), critical = new Rows();
      Facet f = fixture(rows, critical);
      var store = f.core.enemy(1).longGun;
      store.add(new Memory.Observation(
          new double[store.dimension], Memory.Label.PRECISE_GF, .25, 0, 1, 1, 1, 1, 1, false));
      var entered = new CountDownLatch(1);
      var release = new CountDownLatch(1);
      var errors = new AtomicReference<Throwable>();
      AtomicBoolean submitted = new AtomicBoolean();
      Thread decision = worker(errors, () -> f.decideFrame(f.latest, intent -> {
        check(Thread.holdsLock(f.batch), "real submission owns lifecycle monitor");
        entered.countDown();
        await(release);
        submitted.set(true);
      }));
      await(entered);
      Thread callback = worker(errors, () -> {
        if (reset)
          f.onRoundStarted(new RoundStartedEvent(2));
        else
          f.finishRound(1);
      });
      blocked(callback);
      check(
          f.observedTerminal.isEmpty() && rows.rows.isEmpty(), "terminal cannot overlap decision");
      release.countDown();
      joined(decision, errors);
      joined(callback, errors);
      check(submitted.get(), "decision transaction completes");
      check(f.observedTerminal.contains(1) && rows.rows.size() == 1, "terminal captured once");
      if (reset) {
        check(f.core.enemy(1).longGun.size() == 1, "round reset retains independent game learning");
        check(f.core.round == 2 && f.latest == null, "reset after frozen terminal evidence");
        check(rows.rows.get(0).contains("\"round\":1"), "old round completion identity frozen");
        long sequence = f.core.sequence;
        f.finishRound(1, OWN, 1, 999, 0, List.of(), 0);
        check(f.core.round == 2 && f.core.sequence == sequence,
            "late terminal cannot mutate new round");
      }
      f.diagnostics.close();
    }
  }
  static void pendingAndHash() {
    Rows rows = new Rows(), critical = new Rows();
    Facet f = fixture(rows, critical);
    AtomicInteger hashes = new AtomicInteger();
    f.hashRound = core -> {
      hashes.incrementAndGet();
      throw new IllegalStateException("independent hash fault");
    };
    rows.failures = 3;
    f.finishRound(1);
    check(f.observedTerminal.contains(1) && !f.completed.contains(1),
        "failed write never marks completion");
    check(f.diagnostics.pending.size() == 1 && rows.rows.isEmpty(),
        "immutable failed completion retained");
    String frozen = f.diagnostics.pending.get(1);
    check(frozen.contains("\"terminalHash\":\"failed\""), "hash failure explicit in row");
    check(f.diagnostics.counters.get("terminal.hash.failure") == 1, "hash failure counted");
    check(critical.rows.stream().anyMatch(x -> x.contains("terminal.hashFailed")),
        "hash failure independent critical record");
    f.onRoundStarted(new RoundStartedEvent(2));
    f.core.fire.confirmed = 99;
    f.diagnostics.retryPending();
    f.diagnostics.retryPending();
    check(rows.rows.size() == 1 && rows.rows.get(0).equals(frozen),
        "retry persists identical old bytes");
    check(!rows.rows.get(0).contains("\"shotsConfirmed\":99"),
        "pending completion never observes later Core");
    check(hashes.get() == 1, "hash failure does not repeat terminal observation");
    long sequence = f.core.sequence;
    f.finishRound(1, OWN, 0, 300, 0, List.of(), 0);
    check(f.core.sequence == sequence && rows.rows.size() == 1,
        "late callback cannot duplicate completion");
    f.diagnostics.close();
  }
  static void completionOrdersAndGames() {
    for (boolean gameFirst : new boolean[] {false, true}) {
      Rows rows = new Rows(), critical = new Rows();
      Facet f = fixture(rows, critical);
      var bullet = new BulletState(12, 0, 2, 400, 300, 0, null);
      f.batch.add(new BulletFiredEvent(40, bullet));
      f.batch.add(new BulletFiredEvent(40, bullet));
      Runnable finish = () -> f.finishRound(1, OWN, 100, 40, 1, List.of(), 0);
      if (gameFirst) {
        f.finishGame(finish);
        finish.run();
      } else {
        finish.run();
        f.finishGame(finish);
      }
      f.finishGame(finish);
      finish.run();
      check(rows.rows.size() == 1 && f.core.fire.confirmed == 1,
          "both completion orders and duplicate physical confirmation");
      Facet.Frame stale = frame(f, 1, 41);
      f.onGameStarted(null);
      check(f.core == null && f.completed.isEmpty() && f.observedTerminal.isEmpty(),
          "new game cold owner");
      f.diagnostics.close();
      f.diagnostics = new Diagnostics(new Rows(), new Rows());
      f.core = new Core(RULES, f.diagnostics, 1);
      f.core.roundReset(1);
      AtomicBoolean submitted = new AtomicBoolean();
      f.decideFrame(stale, intent -> submitted.set(true));
      check(!submitted.get() && f.core.enemies.isEmpty(), "old-game decision frame rejected");
      f.diagnostics.close();
    }
  }
  static void actualFileRollback() throws Exception {
    Path dir = Files.createTempDirectory("facet-adapter-");
    Path file = dir.resolve("FACET_rows.jsonl");
    Diagnostics.FileSink sink = new Diagnostics.FileSink(file);
    AtomicBoolean once = new AtomicBoolean();
    sink.beforeCommit = () -> {
      if (!once.getAndSet(true))
        throw new IOException("fault after bytes before force");
    };
    Diagnostics d = new Diagnostics(sink, new Rows());
    Core c = new Core(RULES, d, 1);
    c.roundReset(1);
    d.snapshot(1, c.fire, c.budget, c.waves, c.planner);
    check(Files.size(file) == 0 && !d.completed.contains(1), "real partial append rolled back");
    String frozen = d.pending.get(1);
    c.fire.confirmed = 777;
    d.retryPending();
    check(Files.readString(file).equals(frozen) && d.completed.contains(1),
        "real retry forced exact original row");
    d.snapshot(1, c.fire, c.budget, c.waves, c.planner);
    check(Files.readAllLines(file).size() == 1, "duplicate snapshot cannot append another row");
    d.close();
    Files.delete(file);
    Files.delete(dir);
  }
  static void reservedCritical() {
    Rows critical = new Rows();
    Diagnostics d = new Diagnostics(new Rows(), critical);
    d.traceLines = Diagnostics.MAX_TRACE;
    d.failure("independent", new IllegalStateException("trace exhausted"));
    check(critical.rows.size() == 1 && critical.rows.get(0).contains("trace exhausted"),
        "trace cap does not erase failures");
    Core c = new Core(RULES, d, 1);
    c.roundReset(1);
    d.snapshot(1, c.fire, c.budget, c.waves, c.planner);
    check(critical.rows.stream().anyMatch(x -> x.contains("completion.persisted")),
        "completion separate from trace cap");
    d.close();
  }
  static BotState publicState() throws Exception {
    var constructor = BotState.class.getConstructors()[0];
    Class<?>[] types = constructor.getParameterTypes();
    Object[] values = new Object[types.length];
    for (int i = 0; i < types.length; i++) {
      if (types[i] == boolean.class)
        values[i] = false;
      else if (types[i] == double.class)
        values[i] = 0.0;
      else if (types[i] == int.class)
        values[i] = 1;
    }
    return (BotState) constructor.newInstance(values);
  }
  static void queuedTickAndSnapshotFailure() throws Exception {
    Rows rows = new Rows(), critical = new Rows();
    Facet f = fixture(rows, critical);
    BotState state = publicState();
    var bullet = new BulletState(56, 0, 2, 400, 300, 0, null);
    var events = new ArrayList<BotEvent>(List.of(new BulletFiredEvent(41, bullet)));
    var bullets = new ArrayList<BulletState>(List.of(bullet));
    var queued = new TickEvent(41, 1, state, bullets, events);
    // The terminal callback can precede onTick delivery; use only public API queue objects.
    f.acceptPendingTicks(1, List.of(queued), 0);
    var frozen = f.latest;
    events.clear();
    bullets.clear();
    check(frozen.turn() == 41 && frozen.round() == 1 && frozen.game() == f.games,
        "queued terminal frame freezes identity");
    check(frozen.own().p().x() == state.getX() && frozen.bullets().size() == 1,
        "queued state and bullet list copied");
    f.diagnostics.beforeSnapshot = () -> {
      throw new IllegalStateException("independent snapshot fault");
    };
    f.finishRound(1);
    check(f.core.fire.confirmed == 1, "queued terminal physical event consumed before completion");
    check(rows.rows.size() == 1 && rows.rows.get(0).contains("\"snapshotStatus\":\"failed\""),
        "snapshot failure produces explicit completion evidence");
    check(f.diagnostics.counters.get("metrics.captureFailure") == 1, "snapshot failure counted");
    check(critical.rows.stream().anyMatch(x -> x.contains("completion.snapshotFailed")),
        "snapshot failure captured separately");
    f.acceptTick(queued, 0);
    check(f.core.fire.confirmed == 1 && f.batch.isEmpty(),
        "later dispatch cannot replay terminal evidence");
    f.onRoundStarted(new RoundStartedEvent(2));
    f.acceptTick(queued, 0);
    check(f.latest == null && f.core.round == 2, "old round tick cannot relabel/reset new round");
    f.diagnostics.close();
  }
  static void pendingAcrossColdGame() {
    Rows rows = new Rows(), critical = new Rows();
    Facet f = fixture(rows, critical);
    rows.failures = 20;
    f.finishRound(1);
    String frozen = f.diagnostics.pending.get(1);
    Diagnostics old = f.diagnostics;
    f.onGameStarted(null);
    check(f.core == null && f.retired.contains(old),
        "pending old diagnostics retained across cold game");
    rows.failures = 0;
    synchronized (f.batch) {
      f.retryEvidence();
    }
    check(rows.rows.size() == 1 && rows.rows.get(0).equals(frozen),
        "retired retry persists original game identity");
    check(f.retired.isEmpty() && f.completed.isEmpty(),
        "old evidence retry cannot mark new-game completion");
    f.diagnostics.close();
  }
  static void deferredFinalDispatch() throws Exception {
    Rows rows = new Rows(), critical = new Rows();
    Facet f = fixture(rows, critical);
    f.requestRoundFinish(1, 41);
    check(rows.rows.isEmpty() && f.observedTerminal.isEmpty(),
        "completion waits for final expected tick");
    AtomicBoolean submitted = new AtomicBoolean();
    f.decideFrame(f.latest, intent -> submitted.set(true));
    check(!submitted.get(), "pending terminal forbids further command transaction");
    f.gameEndRequested = true;
    var bullet = new BulletState(80, 0, 2, 400, 300, 0, null);
    TickEvent finalTick =
        new TickEvent(41, 1, publicState(), List.of(), List.of(new BulletFiredEvent(41, bullet)));
    f.acceptTick(finalTick, 0);
    check(f.gameClosed && rows.rows.size() == 1 && f.core.fire.confirmed == 1,
        "final dispatch completes deferred round and game after physical evidence");
    check(rows.rows.get(0).contains("\"terminalTurn\":41"), "completion freezes final turn");
    f.acceptTick(finalTick, 0);
    check(rows.rows.size() == 1 && f.core.fire.confirmed == 1,
        "duplicate late dispatch remains idempotent");
  }
  public static void main(String[] args) throws Exception {
    concurrency();
    pendingAndHash();
    completionOrdersAndGames();
    actualFileRollback();
    reservedCritical();
    queuedTickAndSnapshotFailure();
    pendingAcrossColdGame();
    deferredFinalDispatch();
    System.out.println("PASS adapter actual-thread barriers, frozen pending evidence, hash/write "
                       + "faults, lifecycle orders assertions="
        + assertions + " battles=0");
  }
}
