import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;

import java.util.Locale;

/**
 * Tank Royale adapter: connects to the server, forwards events and turns to {@link SurvivorBrain}, and exposes the
 * Bot API as a {@link BotPort}. With {@code AISURVIVOR_TRACE_PATH} set it also writes an exact replay trace
 * (see {@link GameTrace}); tracing only observes and never changes a decision.
 */
public class AISurvivor extends Bot implements BotPort {
    private static final int WARMUP_ITERATIONS = 3000;

    private final SurvivorBrain brain;
    private final GameTrace trace;

    public static void main(String[] args) { new AISurvivor().start(); }

    public AISurvivor() {
        super(BotInfo.fromFile("AISurvivor.json"));
        BotConfig config = BotConfig.fromEnvironment(System.getenv());
        trace = GameTrace.fromEnvironment(System.getenv(), config);
        brain = new SurvivorBrain(this, config, LegalScanRecorder.fromEnvironment(System.getenv()));
        // JIT/class-load the decision code before connecting; pure and gameplay-neutral (see Warmup).
        long started = System.nanoTime();
        double checksum = Warmup.run(WARMUP_ITERATIONS);
        System.out.printf(Locale.ROOT, "[AISurvivor] warmup %d iterations in %d ms (checksum %.3f)%n",
                WARMUP_ITERATIONS, (System.nanoTime() - started) / 1_000_000, checksum);
    }

    @Override
    public void run() {
        trace.tick(this);
        brain.beginRun();
        trace.commands("begin");
        while (isRunning()) {
            // Snapshot the exact state this turn reads: the first turn can run before the first tick is dispatched.
            trace.tick(this);
            brain.turn();
            trace.commands("cmd");
            go();
        }
    }

    // ─── Events: record (when tracing), then forward ─────────────────────────

    @Override public void onTick(TickEvent e) { trace.tick(this); brain.onTick(e); }
    @Override public void onRoundStarted(RoundStartedEvent e) { trace.event(e); brain.onRoundStarted(e); }
    @Override public void onRoundEnded(RoundEndedEvent e) { trace.event(e); brain.onRoundEnded(e); }
    @Override public void onGameEnded(GameEndedEvent e) { trace.event(e); brain.onGameEnded(e); trace.close(); }
    @Override public void onScannedBot(ScannedBotEvent e) { trace.event(e); brain.onScannedBot(e); }
    @Override public void onHitByBullet(HitByBulletEvent e) { trace.event(e); brain.onHitByBullet(e); }
    @Override public void onBulletHit(BulletHitBotEvent e) { trace.event(e); brain.onBulletHit(e); }
    @Override public void onBulletFired(BulletFiredEvent e) { trace.event(e); brain.onBulletFired(e); }
    @Override public void onBotDeath(BotDeathEvent e) { trace.event(e); brain.onBotDeath(e); }
    @Override public void onDeath(DeathEvent e) { trace.event(e); brain.onDeath(e); }
    @Override public void onHitWall(HitWallEvent e) { trace.event(e); brain.onHitWall(e); }
    @Override public void onHitBot(HitBotEvent e) { trace.event(e); brain.onHitBot(e); }
    @Override public void onWonRound(WonRoundEvent e) { trace.event(e); brain.onWonRound(e); }
    @Override public void onSkippedTurn(SkippedTurnEvent e) { trace.event(e); brain.onSkippedTurn(e); }

    // ─── BotPort ────────────────────────────────────────────────────────────

    @Override public int turnNumber() { return getTurnNumber(); }
    @Override public int roundNumber() { return getRoundNumber(); }
    @Override public int enemyCount() { return getEnemyCount(); }
    @Override public int arenaWidth() { return getArenaWidth(); }
    @Override public int arenaHeight() { return getArenaHeight(); }
    @Override public double x() { return getX(); }
    @Override public double y() { return getY(); }
    @Override public double direction() { return getDirection(); }
    @Override public double speed() { return getSpeed(); }
    @Override public double energy() { return getEnergy(); }
    @Override public double gunHeat() { return getGunHeat(); }
    @Override public double gunDirection() { return getGunDirection(); }
    @Override public double radarDirection() { return getRadarDirection(); }
    @Override public double turnRate() { return getTurnRate(); }
    @Override public double targetSpeed() { return getTargetSpeed(); }
    @Override public double gunTurnRate() { return getGunTurnRate(); }
    @Override public double radarTurnRate() { return getRadarTurnRate(); }

    @Override public void commandTurnRate(double rate) { trace.call("turnRate", rate); setTurnRate(rate); }
    @Override public void commandTargetSpeed(double speed) { trace.call("targetSpeed", speed); setTargetSpeed(speed); }
    @Override public void commandGunTurnRate(double rate) { trace.call("gunTurnRate", rate); setGunTurnRate(rate); }
    @Override public void commandRadarTurnRate(double rate) { trace.call("radarTurnRate", rate); setRadarTurnRate(rate); }
    @Override public boolean commandFire(double power) {
        boolean accepted = setFire(power);
        trace.fire(power, accepted);
        return accepted;
    }
    @Override public void adjustGunForBodyTurn(boolean adjust) { trace.flag("adjustGunForBodyTurn", adjust); setAdjustGunForBodyTurn(adjust); }
    @Override public void adjustRadarForGunTurn(boolean adjust) { trace.flag("adjustRadarForGunTurn", adjust); setAdjustRadarForGunTurn(adjust); }
    @Override public void adjustRadarForBodyTurn(boolean adjust) { trace.flag("adjustRadarForBodyTurn", adjust); setAdjustRadarForBodyTurn(adjust); }
}
