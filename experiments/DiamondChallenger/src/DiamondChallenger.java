import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.*;
import java.util.Arrays;
import java.util.Locale;
import java.util.SplittableRandom;

/** Isolated pressure experiment. It neither subclasses AIS nor loads any AIS/DQN checkpoint. */
public class DiamondChallenger extends Bot {
    static final String NAME="DiamondChallenger", VERSION="0.3.0";
    private final long seed=Long.parseLong(System.getenv().getOrDefault("CHALLENGER_SEED","0"));
    private final SplittableRandom random=new SplittableRandom(seed);
    private final Path model=Path.of(System.getenv().getOrDefault("CHALLENGER_MODEL_PATH","DiamondChallenger-model.json")).toAbsolutePath();
    private final Path metrics=Path.of(System.getenv().getOrDefault("CHALLENGER_METRICS_PATH","DiamondChallenger-metrics.jsonl")).toAbsolutePath();
    private final PressureGun gun=new PressureGun();
    private final ChallengerWaves waves=new ChallengerWaves();
    private int surfSign=1,plannerDecisions,profileDecisions,cwDecisions,stopDecisions,ccwDecisions,incomingHits;
    private double arenaWidth=800,arenaHeight=600;
    private int flank=1,nextPhase,roundsCompleted,shots,hits,walls,contacts,skips,rangeTurns,closeTurns,phaseDecisions;
    private double dealt,taken,rangeSum,energySpent;
    private boolean won,died;

    public static void main(String[] args) { new DiamondChallenger().start(); }
    public DiamondChallenger() {
        super(BotInfo.fromFile("DiamondChallenger.json"));
        if(model.equals(metrics) || Files.exists(model) || Files.exists(metrics))
            throw new IllegalStateException("A fresh experiment requires separate, absent model/metrics paths");
        System.out.println("[DiamondChallenger] cold seed="+seed);
    }

    @Override public void run() {
        setAdjustGunForBodyTurn(true); setAdjustRadarForGunTurn(true); setAdjustRadarForBodyTurn(true);
        while(isRunning()) {
            if(getTurnNumber()>=nextPhase) {
                flank=random.nextBoolean()?1:-1;
                nextPhase=getTurnNumber()+PressureMove.phaseDuration(random); phaseDecisions++;
            }
            var target=gun.latest();
            waves.flush(getTurnNumber(),arenaWidth,arenaHeight);
            var decision=WaveSurfMove.choose(getX(),getY(),getDirection(),getSpeed(),getTurnNumber(),
                    target==null?arenaWidth/2:target.x(),target==null?arenaHeight/2:target.y(),
                    waves.threats(),surfSign,arenaWidth,arenaHeight);
            var movement=decision.command();
            if(decision.plannerUsed())plannerDecisions++;
            if(decision.profileUsed())profileDecisions++;
            if(decision.option()>0){cwDecisions++;surfSign=1;}
            else if(decision.option()<0){ccwDecisions++;surfSign=-1;}else stopDecisions++;
            setTurnRate(Math.max(-DuelSurf.maxTurn(getSpeed()),Math.min(DuelSurf.maxTurn(getSpeed()),
                    CombatMath.relativeAngle(movement.bodyHeading()-getDirection()))));
            setTargetSpeed(movement.targetSpeed());
            if(target==null) setRadarTurnRate(45);
            else {
                setRadarTurnRate(DuelRadar.turn(getX(),getY(),getRadarDirection(),target.x(),target.y(),
                        getTurnNumber()-target.turn(),flank));
                if(getTurnNumber()-target.turn()<=1) {
                    double range=Math.hypot(target.x()-getX(),target.y()-getY());
                    rangeTurns++;rangeSum+=range;if(range<180)closeTurns++;
                }
            }
            var aim=gun.decide(getTurnNumber(),getX(),getY(),getGunDirection(),getEnergy(),getGunHeat(),getArenaWidth(),getArenaHeight());
            setGunTurnRate(Math.max(-20,Math.min(20,CombatMath.relativeAngle(aim.direction()-getGunDirection()))));
            if(aim.fire()) setFire(aim.power());
            go();
        }
    }

    @Override public void onRoundStarted(RoundStartedEvent event) {
        gun.startRound();waves.startRound();surfSign=1;
        plannerDecisions=profileDecisions=cwDecisions=stopDecisions=ccwDecisions=incomingHits=0;
        nextPhase=0;shots=hits=walls=contacts=skips=rangeTurns=closeTurns=phaseDecisions=0;
        dealt=taken=rangeSum=energySpent=0;won=died=false;
    }
    @Override public void onGameStarted(GameStartedEvent event) {
        arenaWidth=event.getGameSetup().getArenaWidth();arenaHeight=event.getGameSetup().getArenaHeight();
    }
    @Override public void onTick(TickEvent event) {
        var own=event.getBotState();
        waves.own(new ChallengerWaves.Own(event.getTurnNumber(),own.getX(),own.getY(),own.getDirection(),own.getSpeed()));
    }
    @Override public void onScannedBot(ScannedBotEvent event) {
        var scan=new DuelGun.Scan(event.getScannedBotId(),event.getTurnNumber(),event.getX(),event.getY(),
                event.getDirection(),event.getSpeed(),event.getEnergy());
        gun.onScan(scan);waves.scan(scan);
    }
    @Override public void onBulletFired(BulletFiredEvent event) {
        shots++;var bullet=event.getBullet();energySpent+=bullet.getPower();
        gun.onFired(event.getTurnNumber(),bullet.getX(),bullet.getY(),bullet.getPower());
    }
    @Override public void onBulletHit(BulletHitBotEvent event) {
        hits++;dealt+=event.getDamage();waves.damage(event.getVictimId(),event.getTurnNumber(),event.getDamage());
    }
    @Override public void onHitByBullet(HitByBulletEvent event) {
        taken+=event.getDamage();incomingHits++;var b=event.getBullet();
        waves.bonus(b.getOwnerId(),event.getTurnNumber(),b.getPower());
        waves.hit(new ChallengerWaves.Hit(event.getTurnNumber(),b.getOwnerId(),b.getPower(),b.getX(),b.getY(),b.getDirection()));
    }
    @Override public void onHitWall(HitWallEvent event) { walls++; }
    @Override public void onHitBot(HitBotEvent event) { contacts++;waves.collision(event.getTurnNumber()); }
    @Override public void onSkippedTurn(SkippedTurnEvent event) { skips++; }
    @Override public void onWonRound(WonRoundEvent event) { won=true; }
    @Override public void onDeath(DeathEvent event) { died=true; }

    @Override public void onRoundEnded(RoundEndedEvent event) {
        if(event.getRoundNumber()<=roundsCompleted)return;
        if(event.getRoundNumber()!=roundsCompleted+1)throw new IllegalStateException("Nonsequential experimental rounds");
        waves.flush(event.getTurnNumber(),arenaWidth,arenaHeight);
        roundsCompleted++;
        String state=String.format(Locale.ROOT,
                "{\"schemaVersion\":1,\"name\":\"%s\",\"version\":\"%s\",\"seed\":%d,\"roundsCompleted\":%d,\"checkpointLoaded\":false,\"gunAttempts\":%s,\"gunHits\":%s,\"contextAccepted\":%d,\"contextMemorySize\":%d,\"contextMemory\":%s}%n",
                NAME,VERSION,seed,roundsCompleted,Arrays.toString(gun.attempts()),Arrays.toString(gun.hits()),
                gun.context().acceptedTotal(),gun.context().size(),gun.context().snapshot());
        String row=String.format(Locale.ROOT,
                "{\"schemaVersion\":1,\"name\":\"%s\",\"version\":\"%s\",\"mode\":\"train\",\"seed\":%d,\"round\":%d,\"roundsCompleted\":%d,\"turns\":%d,\"checkpointLoaded\":false,\"shotsFired\":%d,\"bulletHits\":%d,\"damageDealt\":%.9f,\"damageTaken\":%.9f,\"energySpent\":%.9f,\"wallCollisions\":%d,\"botCollisions\":%d,\"skippedTurns\":%d,\"won\":%s,\"died\":%s,\"rangeObservedTurns\":%d,\"rangeSum\":%.9f,\"closeTurns\":%d,\"phaseDecisions\":%d,\"learnedAimDecisions\":%d,\"fallbackAimDecisions\":%d,\"acceptedPassageSamples\":%d,\"confirmedLearnedShots\":%d,\"contextMemorySize\":%d}%n",
                NAME,VERSION,seed,event.getRoundNumber(),roundsCompleted,event.getTurnNumber(),shots,hits,dealt,taken,energySpent,walls,contacts,skips,
                won,died,rangeTurns,rangeSum,closeTurns,phaseDecisions,gun.learnedAimDecisions(),gun.fallbackAimDecisions(),
                gun.context().acceptedRound(),gun.confirmedLearnedShots(),gun.context().size());
        state=state.stripTrailing();
        state=state.substring(0,state.length()-1)+String.format(Locale.ROOT,
                ",\"inferredWavesTotal\":%d,\"matchedHitsTotal\":%d,\"dangerBins\":%s}%n",
                waves.totalInferred(),waves.totalMatched(),Arrays.deepToString(waves.danger()));
        row=row.stripTrailing();
        row=row.substring(0,row.length()-1)+String.format(Locale.ROOT,
                ",\"inferredWaves\":%d,\"matchedHits\":%d,\"unmatchedHits\":%d,\"rejectedScanPairs\":%d,\"incomingHits\":%d,\"plannerDecisions\":%d,\"profileDecisions\":%d,\"cwDecisions\":%d,\"stopDecisions\":%d,\"ccwDecisions\":%d}%n",
                waves.inferred(),waves.matched(),waves.unmatched(),waves.rejected(),incomingHits,
                plannerDecisions,profileDecisions,cwDecisions,stopDecisions,ccwDecisions);
        try {
            Files.createDirectories(model.getParent());Files.createDirectories(metrics.getParent());
            Path temporary=Files.createTempFile(model.getParent(),"challenger-",".tmp");
            Files.writeString(temporary,state);Files.move(temporary,model,StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING);
            Files.writeString(metrics,row,StandardOpenOption.CREATE,StandardOpenOption.APPEND);
        } catch(IOException error) { throw new UncheckedIOException(error); }
    }
}
