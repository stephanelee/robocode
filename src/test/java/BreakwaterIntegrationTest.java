import dev.robocode.tankroyale.botapi.BulletState;
import dev.robocode.tankroyale.botapi.events.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class BreakwaterIntegrationTest {
    @TempDir Path directory;
    BotPort port() {
        var p=mock(BotPort.class);when(p.turnNumber()).thenReturn(4);when(p.roundNumber()).thenReturn(1);
        when(p.enemyCount()).thenReturn(1);when(p.arenaWidth()).thenReturn(800);when(p.arenaHeight()).thenReturn(600);
        when(p.x()).thenReturn(142.);when(p.y()).thenReturn(300.);when(p.direction()).thenReturn(90.);
        when(p.speed()).thenReturn(8.);when(p.energy()).thenReturn(100.);when(p.gunHeat()).thenReturn(1.);
        return p;
    }
    SurvivorBrain brain(BotPort p,BotConfig.Mode mode,BotConfig.DuelPolicy policy) {
        return new SurvivorBrain(p,new BotConfig(mode,7,directory.resolve(mode+"-"+policy+".dat"),0,null,policy),
                LegalScanRecorder.fromEnvironment(Map.of()));
    }
    @Test void onlyDefaultTrainDuelUsesNewControllerWithForcedActionCredit()throws Exception {
        for(var mode:BotConfig.Mode.values())for(var policy:BotConfig.DuelPolicy.values()) {
            var p=port();var b=brain(p,mode,policy);b.onRoundStarted(new RoundStartedEvent(1));
            b.onScannedBot(new ScannedBotEvent(4,1,2,100,500,300,0,0));b.turn();
            var m=(RoundMetrics)TargetingCharacterizationTest.get(b,"metrics");
            int expected=mode==BotConfig.Mode.TRAIN&&policy==BotConfig.DuelPolicy.CONTROLLER?1:0;
            assertEquals(expected,m.breakwaterMovementTurns);assertEquals(expected,m.forcedDuelTurns);
            if(expected==1)assertEquals(1,m.actionCounts[0]);
            verify(p,atLeastOnce()).commandTurnRate(anyDouble());verify(p,atLeastOnce()).commandTargetSpeed(anyDouble());
        }
        var p=port();when(p.enemyCount()).thenReturn(2);var b=brain(p,BotConfig.Mode.TRAIN,BotConfig.DuelPolicy.CONTROLLER);
        b.onRoundStarted(new RoundStartedEvent(1));b.onScannedBot(new ScannedBotEvent(4,1,2,100,500,300,0,0));b.turn();
        assertEquals(0,((RoundMetrics)TargetingCharacterizationTest.get(b,"metrics")).breakwaterMovementTurns);
        when(p.enemyCount()).thenReturn(1);when(p.turnNumber()).thenReturn(5);b.turn();
        var m=(RoundMetrics)TargetingCharacterizationTest.get(b,"metrics");
        assertEquals(1,m.forcedDuelTurns);assertEquals(0,m.breakwaterMovementTurns,"Melee endgame retains accepted movement");
    }
    @Test void zeroBeforeFirstValidStateDoesNotDisableTheDuelController()throws Exception {
        var p=port();when(p.enemyCount()).thenReturn(0);var b=brain(p,BotConfig.Mode.TRAIN,BotConfig.DuelPolicy.CONTROLLER);
        b.onRoundStarted(new RoundStartedEvent(1));b.turn();
        when(p.enemyCount()).thenReturn(1);when(p.turnNumber()).thenReturn(5);
        b.onScannedBot(new ScannedBotEvent(5,1,2,100,500,300,0,0));b.turn();
        var metrics=(RoundMetrics)TargetingCharacterizationTest.get(b,"metrics");
        assertEquals(1,metrics.breakwaterMovementTurns);assertEquals(1,metrics.forcedDuelTurns);
    }
    @Test void incomingHitBeforeInferenceTrainsIndependentStrictLedgerUsingAimTimeState()throws Exception {
        var p=port();var b=brain(p,BotConfig.Mode.TRAIN,BotConfig.DuelPolicy.CONTROLLER);
        b.onRoundStarted(new RoundStartedEvent(1));b.turn(); // records legal aim-time own state4
        b.onScannedBot(new ScannedBotEvent(4,1,2,100,100,300,0,0));
        when(p.turnNumber()).thenReturn(5);when(p.y()).thenReturn(308.);
        b.onScannedBot(new ScannedBotEvent(5,1,2,99,100,300,0,0));
        var hit=new HitByBulletEvent(5,new BulletState(9,2,1,117,300,0,null),4,96);
        b.onHitByBullet(hit);b.turn();
        var movement=(BreakwaterSurf)TargetingCharacterizationTest.get(b,"breakwater");
        assertArrayEquals(new double[]{0},movement.corrections(),1e-12);
        b.onHitByBullet(hit);b.turn();assertEquals(1,movement.corrections().length,"A removed wave cannot train twice");
        b.onRoundStarted(new RoundStartedEvent(2));assertEquals(1,movement.corrections().length,"Legal matched memory persists across rounds");
        assertEquals(0,((RoundMetrics)TargetingCharacterizationTest.get(b,"metrics")).breakwaterMovementTurns);
    }
}
