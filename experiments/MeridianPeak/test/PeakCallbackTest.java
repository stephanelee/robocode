import org.junit.jupiter.api.Test;
import java.lang.reflect.Field;
import java.util.List;
import dev.robocode.tankroyale.botapi.BulletState;
import dev.robocode.tankroyale.botapi.events.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class PeakCallbackTest {
    static Object field(Object object,String name) throws Exception {
        Field f=object.getClass().getDeclaredField(name);f.setAccessible(true);return f.get(object);
    }
    @SuppressWarnings("unchecked")
    @Test void actualLateCallbacksRefreshModelSealButNeverAdvanceIt() throws Exception {
        var bot=new MeridianPeak();bot.onRoundStarted(new RoundStartedEvent(1));
        ((List<HitAngles.Wave>)field(bot,"waves")).add(new HitAngles.Wave(4,0,0,0,20,2));
        var end=mock(RoundEndedEvent.class);when(end.getRoundNumber()).thenReturn(1);bot.onRoundEnded(end);
        var state=(PeakTelemetry)field(bot,"state");long first=state.revision;
        var hit=mock(HitByBulletEvent.class);
        when(hit.getTurnNumber()).thenReturn(4);
        when(hit.getBullet()).thenReturn(new BulletState(1,2,2,14,0,0,null));
        when(hit.getDamage()).thenReturn(10.0);
        bot.onHitByBullet(hit);
        bot.onBulletFired(new BulletFiredEvent(4,new BulletState(2,1,1.65,100,100,0,null)));
        assertEquals(0,state.completedRounds);assertEquals(1,state.round);assertEquals(first,state.revision);
        bot.onGameEnded(null);assertEquals(1,state.completedRounds);assertTrue(state.revision>first);
        assertEquals(1,state.snapshot().models().hitAngles().effectiveTotal());
        assertEquals(1,state.stats.hitsTaken);assertEquals(10,state.stats.damageTaken);
        assertEquals(1,state.stats.confirmedShots);assertEquals(1.65,state.stats.shotEnergy);
        bot.onRoundStarted(new RoundStartedEvent(2));assertEquals(1,state.completedRounds);
        assertEquals(0,state.stats.confirmedShots);assertEquals(1,state.cumulativeAcceptedHits);
    }
}
