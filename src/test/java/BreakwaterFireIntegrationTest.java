import dev.robocode.tankroyale.botapi.BulletState;
import dev.robocode.tankroyale.botapi.events.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class BreakwaterFireIntegrationTest {
    @TempDir Path directory;
    record Fixture(SurvivorBrain brain,BotPort port,DuelGun gun,RoundMetrics metrics){}
    Fixture fixture(double targetEnergy,boolean oldFire,boolean v2)throws Exception {
        var p=mock(BotPort.class);when(p.enemyCount()).thenReturn(1);when(p.turnNumber()).thenReturn(10);
        when(p.roundNumber()).thenReturn(1);when(p.arenaWidth()).thenReturn(800);when(p.arenaHeight()).thenReturn(600);
        when(p.x()).thenReturn(400.);when(p.y()).thenReturn(300.);when(p.energy()).thenReturn(100.);
        when(p.gunDirection()).thenReturn(15.);when(p.commandFire(anyDouble())).thenReturn(true);
        var cfg=new BotConfig(BotConfig.Mode.TRAIN,5,directory.resolve("private.dat"),0,null);
        var b=new SurvivorBrain(p,cfg,LegalScanRecorder.fromEnvironment(Map.of()),true,v2);
        b.onRoundStarted(new RoundStartedEvent(1));
        var g=mock(DuelGun.class);TargetingCharacterizationTest.set(b,"duelGun",g);
        when(g.decide(any())).thenReturn(new DuelGun.Plan(30,targetEnergy==0?.1:1.8,oldFire,true,"current",0,0,0));
        b.onScannedBot(new ScannedBotEvent(10,1,7,targetEnergy,600,300,90,8));
        return new Fixture(b,p,g,(RoundMetrics)TargetingCharacterizationTest.get(b,"metrics"));
    }
    @Test void vetoKeepsAimAndPowerButCreatesNoPendingLearningCohort()throws Exception {
        var f=fixture(100,true,true);
        try(var guard=mockStatic(BreakwaterFireGuard.class)) {
            guard.when(()->BreakwaterFireGuard.evaluate(any())).thenReturn(new BreakwaterFireGuard.Result(1,3,false));
            f.brain.turn();verify(f.port).commandGunTurnRate(15);verify(f.port,never()).commandFire(anyDouble());
            var input=org.mockito.ArgumentCaptor.forClass(BreakwaterFireGuard.Input.class);
            guard.verify(()->BreakwaterFireGuard.evaluate(input.capture()));
            assertEquals(15,input.getValue().gunDirection());assertEquals(1.8,input.getValue().power());
            assertEquals(1,f.metrics.breakwaterFireEligibleTurns);assertEquals(1,f.metrics.breakwaterFireBlockedTurns);
            assertEquals(0,f.metrics.breakwaterFirePassedTurns);assertNull(TargetingCharacterizationTest.get(f.brain,"pendingTargetingWave"));
            assertEquals(false,TargetingCharacterizationTest.get(f.brain,"pendingDuelShot"));
            f.brain.onBulletFired(new BulletFiredEvent(11,new BulletState(9,1,1.8,400,300,15,null)));
            verify(f.gun,never()).onFired(anyInt(),anyDouble(),anyDouble(),anyDouble());
        }
    }
    @Test void acceptedGuardStillWaitsForActualFireConfirmation()throws Exception {
        var f=fixture(100,true,true);
        try(var guard=mockStatic(BreakwaterFireGuard.class)) {
            guard.when(()->BreakwaterFireGuard.evaluate(any())).thenReturn(new BreakwaterFireGuard.Result(2,3,true));
            f.brain.turn();verify(f.port).commandFire(1.8);assertEquals(1,f.metrics.breakwaterFirePassedTurns);
            assertEquals(0,f.metrics.breakwaterFireBlockedTurns);assertNotNull(TargetingCharacterizationTest.get(f.brain,"pendingTargetingWave"));
            verify(f.gun,never()).onFired(anyInt(),anyDouble(),anyDouble(),anyDouble());
            f.brain.onBulletFired(new BulletFiredEvent(11,new BulletState(9,1,1.8,400,300,15,null)));
            verify(f.gun).onFired(11,400,300,1.8);
        }
    }
    @Test void zeroEnergyBypassPreservesOldFireGateAndPower()throws Exception {
        for(boolean oldFire:new boolean[]{false,true}) {
            var f=fixture(0,oldFire,true);
            try(var guard=mockStatic(BreakwaterFireGuard.class)) {
                f.brain.turn();guard.verifyNoInteractions();
                assertEquals(oldFire?1:0,f.metrics.breakwaterFireFinisherTurns);
                assertEquals(oldFire?1:0,f.metrics.breakwaterFireEligibleTurns);
                if(oldFire)verify(f.port).commandFire(.1);else verify(f.port,never()).commandFire(anyDouble());
            }
        }
    }
    @Test void historicalMovementOnlyPolicyDoesNotAcquireNewFireGateAndCountersReset()throws Exception {
        var f=fixture(100,true,false);
        try(var guard=mockStatic(BreakwaterFireGuard.class)) {
            f.brain.turn();guard.verifyNoInteractions();verify(f.port).commandFire(1.8);
            assertEquals(0,f.metrics.breakwaterFireEligibleTurns);
            f.metrics.breakwaterFireEligibleTurns=f.metrics.breakwaterFirePassedTurns=f.metrics.breakwaterFireBlockedTurns=f.metrics.breakwaterFireFinisherTurns=9;
            f.brain.onRoundStarted(new RoundStartedEvent(2));
            assertEquals(0,f.metrics.breakwaterFireEligibleTurns);assertEquals(0,f.metrics.breakwaterFirePassedTurns);
            assertEquals(0,f.metrics.breakwaterFireBlockedTurns);assertEquals(0,f.metrics.breakwaterFireFinisherTurns);
        }
    }
}
