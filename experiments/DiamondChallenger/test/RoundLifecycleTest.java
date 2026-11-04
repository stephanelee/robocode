import com.google.gson.JsonParser;
import dev.robocode.tankroyale.botapi.BotInfo;
import dev.robocode.tankroyale.botapi.events.*;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;
import java.io.PrintStream;
import java.nio.file.*;
import java.util.Arrays;
import java.util.concurrent.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class RoundLifecycleTest {
    @TempDir Path directory;
    private final PrintStream out=System.out,err=System.err;
    @AfterEach void restoreStreams(){System.setOut(out);System.setErr(err);}
    static DuelGun.Scan scan(int turn){return new DuelGun.Scan(2,turn,500,300,90,0,100);}
    static PressureGun pendingCohort(PressureGun gun) {
        var context=gun.context();
        for(int i=0;i<32;i++) {
            int t=100+i*100;context.onScan(scan(t-1));context.onScan(scan(t));
            var q=context.query(t,scan(t),scan(t-1),300,300,1.4,0,200,800,600);
            context.confirmed(q,t+1,300,300,1.4);
            for(int age=1;age<=30;age++)context.onScan(scan(t+age));
        }
        gun.onScan(scan(3999));gun.onScan(scan(4000));
        var plan=gun.decide(4000,300,300,0,100,0,800,600);
        assertTrue(plan.fire());assertEquals(1,gun.supportedDecisions());
        gun.onFired(4001,7,300,300,plan.power(),0);
        for(int t=4001;t<=4010;t++)gun.onScan(scan(t));
        assertEquals(1,gun.pairedPending());assertEquals(0,gun.pairedCompletedRound());
        return gun;
    }
    static String state(PressureGun gun) {
        return Arrays.toString(gun.attempts())+Arrays.toString(gun.hits())
                +Arrays.toString(gun.pairedAttempts())+Arrays.toString(gun.pairedHits())
                +gun.pairedPending()+":"+gun.pairedCreatedRound()+":"+gun.pairedCompletedRound()
                +":"+gun.fallbackAimDecisions()+":"+gun.context().snapshot()
                +":"+gun.context().outgoingCount()+":"+gun.context().waveContextQueriesRound()
                +":"+gun.context().outgoingKnownRemovedRound();
    }
    DiamondChallenger bot() {
        return new DiamondChallenger(BotInfo.builder().setName("DiamondChallenger").setVersion("0.6.1")
                .addAuthor("test").build(),1,directory.resolve("model.json"),directory.resolve("metrics.jsonl"),null);
    }
    static PressureGun gun(DiamondChallenger bot)throws Exception {
        var field=DiamondChallenger.class.getDeclaredField("gun");field.setAccessible(true);return (PressureGun)field.get(bot);
    }
    @Test void exactPendingProbeCannotMutateAfterCloseAndReopensWithoutUnresolvedCohort() {
        var gun=pendingCohort(new PressureGun());gun.closeRound();String frozen=state(gun);
        gun.closeRound();gun.onScan(scan(4011));gun.onFired(4012,8,300,300,1.4,0);gun.removed(7);
        assertFalse(gun.decide(4012,300,300,0,100,0,800,600).fire());
        assertNull(gun.context().reference(4010,scan(4010),300,300));
        assertNull(gun.context().query(4011,scan(4011),scan(4010),300,300,1.4,0,200,800,600));
        gun.context().confirmed(null,4012,300,300,1.4);
        assertEquals(frozen,state(gun));
        gun.startRound();assertEquals(0,gun.pairedPending());assertEquals(0,gun.pairedAttempts()[0]);
        gun.onScan(scan(1));gun.onScan(scan(2));
        assertTrue(gun.decide(2,300,300,0,100,0,800,600).fire());
    }
    @Test void closingWavesFlushesTerminalHitExactlyOnceThenFreezesEveryInput() {
        var waves=ChallengerWavesTest.prepared();waves.scan(ChallengerWavesTest.scan(5,99));
        waves.hit(new ChallengerWaves.Hit(5,2,1,117,300,0));waves.closeRound(5,800,600);
        assertEquals(1,waves.matched());String bins=Arrays.deepToString(waves.danger());
        waves.closeRound(5,800,600);waves.scan(ChallengerWavesTest.scan(6,98));
        waves.own(new ChallengerWaves.Own(6,100,100,0,0));waves.damage(2,5,2);waves.bonus(2,5,1);
        waves.hit(new ChallengerWaves.Hit(6,2,Double.NaN,0,0,0));waves.collision(6);waves.flush(6,800,600);
        assertEquals(bins,Arrays.deepToString(waves.danger()));assertEquals(1,waves.matched());assertEquals(0,waves.unmatched());
        waves.startRound();assertEquals(.95,Arrays.stream(waves.danger()[1]).sum(),1e-15);
        assertEquals(0,waves.matched());assertEquals(1,waves.totalMatched());
    }
    @Test void requestedButUnconfirmedShotCannotEnterClosedLearningOrNextRound() {
        var gun=new PressureGun();gun.startRound();gun.onScan(scan(1));
        var plan=gun.decide(1,300,300,0,100,0,800,600);assertTrue(plan.fire());
        gun.closeRound();String frozen=state(gun);
        gun.onFired(2,99,300,300,plan.power(),0);
        assertEquals(frozen,state(gun));assertEquals(0,gun.context().outgoingCreatedRound());
        gun.startRound();assertEquals(0,gun.pairedPending());
        gun.onScan(scan(1));gun.onScan(scan(2));
        assertEquals(0,gun.context().outgoingCount());assertEquals(0,gun.context().acceptedTotal());
    }
    @Test void adapterDrainsTerminalEventsBeforeAtomicSnapshotThenRejectsLateCallbacks()throws Exception {
        var bot=bot();bot.onRoundStarted(new RoundStartedEvent(1));var gun=pendingCohort(gun(bot));
        bot.onRoundEnded(new RoundEndedEvent(1,4011,null));
        assertFalse(Files.exists(directory.resolve("model.json")));assertFalse(bot.performTurn());
        bot.onScannedBot(new ScannedBotEvent(4011,1,2,100,500,300,90,0));
        bot.onWonRound(new WonRoundEvent(4011));bot.onSkippedTurn(new SkippedTurnEvent(4011));
        bot.onGameEnded(new GameEndedEvent(1,null));
        var model=JsonParser.parseString(Files.readString(directory.resolve("model.json"))).getAsJsonObject();
        var row=JsonParser.parseString(Files.readString(directory.resolve("metrics.jsonl"))).getAsJsonObject();
        assertEquals(1,model.getAsJsonArray("pairedAttempts").get(0).getAsInt());
        assertEquals(1,row.get("pairedCompleted").getAsInt());assertTrue(row.get("won").getAsBoolean());
        assertEquals(1,row.get("skippedTurns").getAsInt());
        String modelBytes=Files.readString(directory.resolve("model.json")),rowBytes=Files.readString(directory.resolve("metrics.jsonl"));
        String frozen=state(gun);bot.onScannedBot(new ScannedBotEvent(4012,1,2,100,500,300,90,0));
        bot.onSkippedTurn(new SkippedTurnEvent(4012));bot.onDeath(new DeathEvent(4012));
        bot.onRoundEnded(new RoundEndedEvent(1,4011,null));bot.onGameEnded(new GameEndedEvent(1,null));
        assertFalse(bot.performTurn());assertEquals(frozen,state(gun));
        assertEquals(modelBytes,Files.readString(directory.resolve("model.json")));
        assertEquals(rowBytes,Files.readString(directory.resolve("metrics.jsonl")));
        bot.onRoundStarted(new RoundStartedEvent(2));bot.onRoundStarted(new RoundStartedEvent(2));
        assertEquals(1,gun.pairedAttempts()[0]);assertEquals(0,gun.pairedCompletedRound());
        assertEquals(0,gun.pairedPending());assertEquals(1,Files.readAllLines(directory.resolve("metrics.jsonl")).size());
    }
    @Test void registeredOldRunThreadCannotChangeReopenedRound()throws Exception {
        var bot=spy(bot());doNothing().when(bot).setAdjustGunForBodyTurn(true);
        doNothing().when(bot).setAdjustRadarForGunTurn(true);doNothing().when(bot).setAdjustRadarForBodyTurn(true);
        doReturn(false).when(bot).isRunning();bot.onRoundStarted(new RoundStartedEvent(1));
        var bound=new CountDownLatch(1);var reopened=new CountDownLatch(1);
        var task=CompletableFuture.runAsync(()->{bot.run();bound.countDown();try{reopened.await(5,TimeUnit.SECONDS);}catch(InterruptedException e){throw new RuntimeException(e);}
            bot.onScannedBot(new ScannedBotEvent(1,1,2,100,500,300,90,0));bot.onWonRound(new WonRoundEvent(1));});
        assertTrue(bound.await(5,TimeUnit.SECONDS));bot.onRoundEnded(new RoundEndedEvent(1,1,null));
        bot.onRoundStarted(new RoundStartedEvent(2));reopened.countDown();task.get(5,TimeUnit.SECONDS);
        assertNull(gun(bot).latest());bot.onRoundEnded(new RoundEndedEvent(2,1,null));bot.onGameEnded(new GameEndedEvent(2,null));
        var row=JsonParser.parseString(Files.readAllLines(directory.resolve("metrics.jsonl")).get(1)).getAsJsonObject();
        assertFalse(row.get("won").getAsBoolean());
    }
    @Test void goWaitDoesNotHoldLifecycleMonitor()throws Exception {
        var bot=spy(bot());bot.onRoundStarted(new RoundStartedEvent(1));
        doNothing().when(bot).setAdjustGunForBodyTurn(true);doNothing().when(bot).setAdjustRadarForGunTurn(true);
        doNothing().when(bot).setAdjustRadarForBodyTurn(true);doReturn(true).when(bot).isRunning();
        doReturn(true,false).when(bot).performTurn();
        doAnswer(invocation->{var callback=CompletableFuture.runAsync(()->bot.onRoundEnded(new RoundEndedEvent(1,1,null)));
            callback.get(3,TimeUnit.SECONDS);return null;}).when(bot).go();
        bot.run();verify(bot,times(1)).go();bot.onGameEnded(new GameEndedEvent(1,null));
    }
}
