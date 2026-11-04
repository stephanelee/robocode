import dev.robocode.tankroyale.botapi.Bot;
import dev.robocode.tankroyale.botapi.BotInfo;
import dev.robocode.tankroyale.botapi.BulletState;
import dev.robocode.tankroyale.botapi.events.*;
import java.nio.file.*;
import java.util.HashMap;
import java.util.Map;

public final class Breakwater extends Bot {
    private final long seed = Long.parseLong(System.getenv().getOrDefault("BREAKWATER_SEED", "0"));
    private final BreakwaterPolicy policy = new BreakwaterPolicy(seed);
    private static final int ENVELOPE = 3;
    private final String gunMode = System.getenv().getOrDefault("BREAKWATER_GUN_MODE", "envelope");
    private final boolean shadow = gunMode.equals("shadow");
    private final BreakwaterLedger ledger = new BreakwaterLedger(seed,
            Path.of(System.getenv().getOrDefault("BREAKWATER_METRICS_PATH", "Breakwater-metrics.jsonl")),
            Path.of(System.getenv().getOrDefault("BREAKWATER_MODEL_PATH", "Breakwater-model.json")), gunMode,
            shadow ? Path.of(System.getenv().getOrDefault("BREAKWATER_SHADOW_PATH", "Breakwater-shadow.jsonl")) : null);
    private final BreakwaterEnvelope envelope = new BreakwaterEnvelope(shadow ? ledger::recordShadow : null);
    private BreakwaterWaves waves;
    private final Map<Integer, Integer> launched = new HashMap<>();
    record GunCommand(double aim, double power, int gun) {}
    static GunCommand gunCommand(boolean shadow, BreakwaterPolicy.Decision baseline, BreakwaterEnvelope.Plan candidate) {
        return shadow ? new GunCommand(baseline.aim(), baseline.power(), baseline.gun())
                      : new GunCommand(candidate.aim(), candidate.power(), ENVELOPE);
    }
    private record Request(int creation, int gun, double power, BreakwaterPolicy.Target scan) {}
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
        var candidate = envelope.decide(own, target, decision);
        var gun = gunCommand(shadow, decision, candidate);
        setTurnRate(decision.turnRate());
        setTargetSpeed(decision.targetSpeed());
        setGunTurnRate(Math.max(-20, Math.min(20, CombatMath.relativeAngle(gun.aim() - getGunDirection()))));
        setRadarTurnRate(target == null ? 45 : DuelRadar.turn(getX(), getY(), getRadarDirection(),
                target.x(), target.y(), now - target.turn(), 1));
        if (target != null && now - target.turn() <= 1) {
            ledger.row.rangeObservedTurns++;
            ledger.row.rangeSum += Math.hypot(target.x() - getX(), target.y() - getY());
        }
        var row = ledger.row;
        row.decisions++;
        if (decision.waves() > 0) row.waveDecisions++;
        if (decision.orbitSign() > 0) row.cwDecisions++;
        else if (decision.orbitSign() < 0) row.ccwDecisions++;
        if (Math.abs(decision.targetSpeed()) > 1e-9) row.movingTargetTurns++;
        if (Math.abs(own.speed()) > 1e-9) row.observedMovingTurns++;
        switch (decision.gun()) {
            case 0 -> row.linearDecisions++;
            case 1 -> row.stoppedDecisions++;
            case 2 -> row.circularDecisions++;
        }
        if (decision.gun() >= 0) row.envelopeDecisions++;
        if (candidate.opportunity()) {
            row.envelopeOpportunities++;
            if (candidate.power() <= 0) row.envelopeHolds++;
        }
        if (gun.power() > 0 && setFire(gun.power())) {
            if (gun.gun() == ENVELOPE) row.envelopeFireRequests++;
            request = new Request(now + 1, gun.gun(), gun.power(), target);
        }
    }

    @Override public synchronized void onRoundStarted(RoundStartedEvent e) {
        if (!ledger.begin(e.getRoundNumber())) return;
        policy.startRound();
        envelope.startRound(e.getRoundNumber());
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
                if (waves.latest == target) { // The shared scan guard accepted this new timestamp.
                    if (shadow) ledger.recordScan(target);
                    envelope.observe(target);
                }
            }
        }
        for (var event : events) {
            if (event instanceof HitByBulletEvent hit) {
                ledger.row.damageTaken += hit.getDamage();
                learn(hit.getTurnNumber(), hit.getBullet());
            } else if (event instanceof BulletHitBotEvent hit) {
                int gun = launched.getOrDefault(hit.getBullet().getBulletId(), -1);
                ledger.row.bulletHits++;
                ledger.row.damageDealt += hit.getDamage();
                if (gun >= 0) {
                    ledger.row.attackHits++;
                    ledger.row.attackDamage += hit.getDamage();
                    switch (gun) {
                        case 0 -> ledger.row.linearHits++;
                        case 1 -> ledger.row.stoppedHits++;
                        case 2 -> ledger.row.circularHits++;
                        case ENVELOPE -> ledger.row.envelopeHits++;
                    }
                }
            } else if (event instanceof BulletHitBulletEvent collision) {
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
        waves.own(now, own.getX(), own.getY(), own.getDirection(), own.getSpeed());
    }

    private void fired(int turn, BulletState bullet) {
        var row = ledger.row;
        row.shotsFired++;
        row.energySpent += bullet.getPower();
        if (request == null || turn != request.creation() ||
                Math.abs(bullet.getPower() - request.power()) > 1e-6) {
            row.unclassifiedShots++;
            return;
        }
        int gun = request.gun();
        launched.put(bullet.getBulletId(), gun);
        if (gun >= 0 && gun <= ENVELOPE) {
            row.attackShots++;
            switch (gun) {
                case 0 -> row.linearShots++;
                case 1 -> row.stoppedShots++;
                case 2 -> row.circularShots++;
                case ENVELOPE -> row.envelopeShots++;
            }
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
        if (ledger.open(e.getRoundNumber())) envelope.endRound(e.getTurnNumber());
        ledger.seal(e.getRoundNumber(), e.getTurnNumber(), e.getResults().getFirstPlaces(),
                policy.corrections(), policy.attackScores(), policy.attackHistory());
    }
}
