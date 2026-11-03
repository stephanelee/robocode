import dev.robocode.tankroyale.botapi.Bot;
import dev.robocode.tankroyale.botapi.BotInfo;
import dev.robocode.tankroyale.botapi.BulletState;
import dev.robocode.tankroyale.botapi.events.*;
import java.nio.file.*;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

public final class Breakwater extends Bot {
    private final long seed = Long.parseLong(System.getenv().getOrDefault("BREAKWATER_SEED", "0"));
    private final BreakwaterPolicy policy = new BreakwaterPolicy(seed);
    private final BreakwaterLedger ledger = new BreakwaterLedger(seed,
            Path.of(System.getenv().getOrDefault("BREAKWATER_METRICS_PATH", "Breakwater-metrics.jsonl")),
            Path.of(System.getenv().getOrDefault("BREAKWATER_MODEL_PATH", "Breakwater-model.json")));
    private BreakwaterWaves waves;
    private final Map<Integer, BreakwaterPolicy.Mode> launched = new HashMap<>();
    private record Request(int creation, BreakwaterPolicy.Decision decision, BreakwaterPolicy.Target scan) {}
    private Request request;
    private int lastTick, lastSkip, maxInactivityTurns;
    private double width, height, cooling;

    @Override public synchronized void onGameStarted(GameStartedEvent event) {
        var setup = event.getGameSetup();
        width = setup.getArenaWidth();
        height = setup.getArenaHeight();
        cooling = setup.getGunCoolingRate();
        maxInactivityTurns = setup.getMaxInactivityTurns();
    }

    public Breakwater() { super(BotInfo.fromFile("Breakwater.json")); }
    public static void main(String[] args) { new Breakwater().start(); }

    @Override public void run() {
        setAdjustGunForBodyTurn(true);
        setAdjustRadarForBodyTurn(true);
        setAdjustRadarForGunTurn(true);
        setFireAssist(false);
        while (isRunning()) {
            command();
            go();
        }
    }

    private synchronized void command() {
        if (!ledger.open(getRoundNumber())) return;
        int now = getTurnNumber();
        var own = new BreakwaterPolicy.Own(now, getX(), getY(), getDirection(), getSpeed(), getGunDirection(),
                getGunHeat(), getEnergy(), getGunCoolingRate(), getArenaWidth(), getArenaHeight());
        var target = waves.latest;
        var decision = policy.decide(own, target, waves.active);
        setTurnRate(decision.turnRate());
        setTargetSpeed(decision.targetSpeed());
        setGunTurnRate(Math.max(-20, Math.min(20, CombatMath.relativeAngle(decision.aim() - getGunDirection()))));
        setRadarTurnRate(target == null ? 45 : DuelRadar.turn(getX(), getY(), getRadarDirection(),
                target.x(), target.y(), now - target.turn(), 1));
        if (target != null && now - target.turn() <= 1) {
            ledger.row.rangeObservedTurns++;
            ledger.row.rangeSum += Math.hypot(target.x() - getX(), target.y() - getY());
        }
        if (decision.solution()) ledger.row.shieldSolutions++;
        if (decision.power() > 0 && setFire(decision.power())) request = new Request(now + 1, decision, target);
    }

    @Override public synchronized void onRoundStarted(RoundStartedEvent e) {
        if (!ledger.begin(e.getRoundNumber())) return;
        policy.startRound();
        waves = new BreakwaterWaves(maxInactivityTurns);
        launched.clear();
        request = null;
        lastTick = lastSkip = -1;
    }

    // Use the bot's round-tagged public tick packet. One pass registers fired IDs before
    // resolving impacts, even when both arrive in the same packet or callbacks have other priorities.
    @Override public synchronized void onTick(TickEvent tick) {
        if (!ledger.open(tick.getRoundNumber()) || tick.getTurnNumber() <= lastTick) return;
        int now = tick.getTurnNumber();
        lastTick = now;
        var events = tick.getEvents();
        for (var event : events)
            if (event instanceof BulletFiredEvent fired) fired(fired.getTurnNumber(), fired.getBullet());
        boolean contact = events.stream().anyMatch(HitBotEvent.class::isInstance);
        for (var event : events) {
            if (event instanceof HitByBulletEvent hit) waves.bonus(hit.getBullet().getOwnerId(), hit.getTurnNumber(), hit.getBullet().getPower());
            if (event instanceof BulletHitBotEvent hit) waves.damage(hit.getVictimId(), hit.getTurnNumber(), hit.getDamage());
            if (event instanceof ScannedBotEvent scan) {
                var target = new BreakwaterPolicy.Target(scan.getTurnNumber(), scan.getX(), scan.getY(),
                        scan.getDirection(), scan.getSpeed(), scan.getEnergy());
                if (waves.scan(scan.getScannedBotId(), target, contact, width, height, cooling))
                    ledger.row.inferredWaves++;
                policy.observe(target);
            }
        }
        var interceptedIds = new HashSet<Integer>();
        for (var event : events) {
            if (event instanceof HitByBulletEvent hit) {
                ledger.row.damageTaken += hit.getDamage();
                learn(hit.getTurnNumber(), hit.getBullet());
            } else if (event instanceof BulletHitBotEvent hit) {
                var mode = launched.get(hit.getBullet().getBulletId());
                ledger.row.bulletHits++;
                ledger.row.damageDealt += hit.getDamage();
                if (mode == BreakwaterPolicy.Mode.SHIELD) {
                    ledger.row.shieldHits++;
                    ledger.row.shieldDamage += hit.getDamage();
                } else if (mode == BreakwaterPolicy.Mode.ATTACK) {
                    ledger.row.attackHits++;
                    ledger.row.attackDamage += hit.getDamage();
                }
            } else if (event instanceof BulletHitBulletEvent collision) {
                var bullet = collision.getBullet();
                if (bullet.getOwnerId() != collision.getHitBullet().getOwnerId() &&
                        launched.get(bullet.getBulletId()) == BreakwaterPolicy.Mode.SHIELD &&
                        interceptedIds.add(bullet.getBulletId())) ledger.row.shieldIntercepts++;
                learn(collision.getTurnNumber(), collision.getHitBullet());
            } else if (event instanceof HitWallEvent) ledger.row.wallCollisions++;
            else if (event instanceof HitBotEvent) ledger.row.botCollisions++;
            else if (event instanceof SkippedTurnEvent skip) skipped(skip.getTurnNumber());
            else if (event instanceof WonRoundEvent) ledger.won = true;
            else if (event instanceof DeathEvent) ledger.died = true;
        }
        for (var event : events) {
            if (event instanceof BulletHitBotEvent hit) launched.remove(hit.getBullet().getBulletId());
            if (event instanceof BulletHitBulletEvent hit) launched.remove(hit.getBullet().getBulletId());
            if (event instanceof BulletHitWallEvent hit) launched.remove(hit.getBullet().getBulletId());
        }
        var own = tick.getBotState();
        waves.own(now, own.getX(), own.getY());
    }

    private void fired(int turn, BulletState bullet) {
        var row = ledger.row;
        row.shotsFired++;
        row.energySpent += bullet.getPower();
        if (request == null || turn != request.creation() ||
                Math.abs(bullet.getPower() - request.decision().power()) > 1e-6) {
            row.unclassifiedShots++;
            return;
        }
        var d = request.decision();
        launched.put(bullet.getBulletId(), d.mode());
        if (d.mode() == BreakwaterPolicy.Mode.SHIELD) {
            row.shieldShots++;
            policy.shieldLaunched(d.waveTurn());
        } else if (d.mode() == BreakwaterPolicy.Mode.ATTACK) {
            row.attackShots++;
            policy.attackLaunched(turn, bullet.getX(), bullet.getY(), bullet.getPower(), request.scan());
        } else row.unclassifiedShots++;
        request = null;
    }

    private void learn(int turn, BulletState bullet) {
        var wave = waves.match(turn, bullet.getOwnerId(), bullet.getPower(), bullet.getX(), bullet.getY(), bullet.getDirection());
        if (wave != null) {
            policy.matched(wave, bullet.getDirection());
            ledger.row.matchedRays++;
        }
    }

    private void skipped(int turn) {
        if (turn > lastSkip) { ledger.row.skippedTurns++; lastSkip = turn; }
    }
    @Override public synchronized void onSkippedTurn(SkippedTurnEvent e) {
        if (ledger.open(getRoundNumber())) skipped(e.getTurnNumber());
    }
    @Override public synchronized void onWonRound(WonRoundEvent e) {
        if (ledger.open(getRoundNumber())) ledger.won = true;
    }
    @Override public synchronized void onDeath(DeathEvent e) {
        if (ledger.open(getRoundNumber())) ledger.died = true;
    }
    @Override public synchronized void onRoundEnded(RoundEndedEvent e) {
        ledger.seal(e.getRoundNumber(), e.getTurnNumber(), policy.corrections(), policy.attackScores(), policy.attackHistory());
    }
}
