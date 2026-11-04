import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.*;
import java.util.List;
import dev.robocode.tankroyale.botapi.BulletState;
import dev.robocode.tankroyale.botapi.events.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class PeakLifecycleTest {
    @TempDir Path temp;
    static RoundEndedEvent end(int round){var event=mock(RoundEndedEvent.class);when(event.getRoundNumber()).thenReturn(round);return event;}
    static PeakTelemetry state(MeridianPeak bot)throws Exception{return (PeakTelemetry)PeakCallbackTest.field(bot,"state");}
    static void fire(MeridianPeak bot){bot.onBulletFired(new BulletFiredEvent(4,new BulletState(2,1,1.65,100,100,0,null)));}
    @Test void closingHasNoPrematureEndpointAndFinalSealIncludesTerminalCallbacks()throws Exception {
        var bot=new MeridianPeak(temp.resolve("state").toString());bot.onRoundStarted(new RoundStartedEvent(1));
        var state=state(bot);bot.onRoundEnded(end(1));
        assertFalse(bot.performTurn());assertFalse(Files.exists(temp.resolve("state/endpoint.json")));
        assertFalse(Files.exists(temp.resolve("state/round-001.json")));assertEquals(0,state.completedRounds);
        fire(bot);var skipped=mock(SkippedTurnEvent.class);when(skipped.getTurnNumber()).thenReturn(4);bot.onSkippedTurn(skipped);
        assertEquals(1,state.stats.confirmedShots);assertEquals(1,state.stats.skippedTurns);
        bot.onGameEnded(null);assertEquals(1,state.completedRounds);assertTrue(state.sealed);
        assertTrue(Files.exists(temp.resolve("state/endpoint.json")));String sealed=Files.readString(temp.resolve("state/endpoint.json"));
        bot.onRoundEnded(end(1));bot.onGameEnded(null);fire(bot);
        assertEquals(1,state.completedRounds);assertEquals(1,state.stats.confirmedShots);
        assertEquals(sealed,Files.readString(temp.resolve("state/endpoint.json")));
    }
    @SuppressWarnings("unchecked")
    @Test void nextStartSealsLateModelsBeforeResetAndOldEpochCannotMutateNewRound()throws Exception {
        var bot=new MeridianPeak(temp.resolve("state").toString());bot.onRoundStarted(new RoundStartedEvent(1));
        ((List<HitAngles.Wave>)PeakCallbackTest.field(bot,"waves")).add(new HitAngles.Wave(4,0,0,0,20,2));
        bot.onRoundEnded(end(1));
        var hit=mock(HitByBulletEvent.class);when(hit.getTurnNumber()).thenReturn(4);
        when(hit.getBullet()).thenReturn(new BulletState(1,2,2,14,0,0,null));when(hit.getDamage()).thenReturn(10.0);
        bot.onHitByBullet(hit);fire(bot);
        bot.onRoundStarted(new RoundStartedEvent(2));var state=state(bot);
        assertEquals(1,state.completedRounds);assertEquals(0,state.stats.confirmedShots);assertEquals(1,state.cumulativeAcceptedHits);
        String sealed=Files.readString(temp.resolve("state/round-001.json"));
        assertTrue(sealed.contains("\"confirmedShots\":1"));assertTrue(sealed.contains("\"cumulativeAcceptedHits\":1"));
        var owner=(ThreadLocal<Integer>)PeakCallbackTest.field(bot,"runEpoch");owner.set(1);
        assertFalse(bot.registerRunThread());bot.run();assertFalse(bot.performTurn());fire(bot);bot.onHitByBullet(hit);assertEquals(0,state.stats.confirmedShots);assertEquals(0,state.stats.hitsTaken);
        owner.remove();fire(bot);assertEquals(1,state.stats.confirmedShots);
        bot.onRoundStarted(new RoundStartedEvent(2));assertEquals(1,state.stats.confirmedShots); // duplicate start cannot reset
        bot.onGameEnded(null);assertEquals(1,state.completedRounds); // no RoundEnded: no invented row
        assertFalse(Files.exists(temp.resolve("state/round-002.json")));
    }
    @Test void terminalScanResolvesPendingGunExampleBeforeNextRoundSeal()throws Exception {
        var bot=new MeridianPeak(temp.resolve("state").toString());bot.onRoundStarted(new RoundStartedEvent(1));
        var gun=(CrossingGun)PeakCallbackTest.field(bot,"crossingGun");
        var shot=CrossingGun.launch(4,new Flight.Pose(100,300,0,0),new Flight.Pose(242,300,90,8),2,800,600);
        gun.requested(shot);bot.onBulletFired(new BulletFiredEvent(4,new BulletState(2,1,2,100,300,0,null)));
        for(int turn=12;turn<=13;turn++){
            if(turn==13)bot.onRoundEnded(end(1));
            var scan=mock(ScannedBotEvent.class);when(scan.getTurnNumber()).thenReturn(turn);
            when(scan.getX()).thenReturn(100+142*Flight.cos(20));when(scan.getY()).thenReturn(300+142*Flight.sin(20));
            when(scan.getDirection()).thenReturn(90.0);when(scan.getSpeed()).thenReturn(8.0);when(scan.getEnergy()).thenReturn(100.0);
            bot.onScannedBot(scan);
        }
        assertEquals(1,gun.snapshot().count());assertEquals(0,state(bot).completedRounds);
        bot.onRoundStarted(new RoundStartedEvent(2));
        assertEquals(1,state(bot).cumulativeCrossings);assertEquals(1,gun.snapshot().count());
        assertTrue(Files.readString(temp.resolve("state/round-001.json")).contains("\"crossingLabels\":1"));
    }
    @Test void interruptedUnregisteredEntryCannotClaimReopenedEpoch()throws Exception {
        var bot=new MeridianPeak();bot.onRoundStarted(new RoundStartedEvent(1));bot.onRoundEnded(end(1));
        bot.onRoundStarted(new RoundStartedEvent(2));
        var accepted=new java.util.concurrent.atomic.AtomicBoolean(true);
        var interruptPreserved=new java.util.concurrent.atomic.AtomicBoolean(false);
        var thread=new Thread(()->{
            Thread.currentThread().interrupt();
            accepted.set(bot.registerRunThread());bot.run();
            interruptPreserved.set(Thread.currentThread().isInterrupted());
        });thread.start();thread.join(1000);
        assertFalse(thread.isAlive());assertFalse(accepted.get());assertTrue(interruptPreserved.get());
        assertEquals(0,state(bot).stats.plannerDecisions);assertEquals(1,state(bot).completedRounds);
    }
}
