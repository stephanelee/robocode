import static org.junit.jupiter.api.Assertions.*;
import dev.robocode.tankroyale.botapi.BotInfo;
import dev.robocode.tankroyale.botapi.events.RoundStartedEvent;
import dev.robocode.tankroyale.botapi.events.BulletFiredEvent;
import dev.robocode.tankroyale.botapi.BulletState;
import java.lang.reflect.Field;
import org.junit.jupiter.api.Test;

/** Invokes the real callback before any tick, without starting a bot or server. */
class RoundLifecycleTest {
    @Test void actualBulletCallbackConfirmsOnlyMeleeShotsAndRoundCallbackClearsPending() throws Exception {
        var bot = new KeelHelm(BotInfo.fromFile("bot/KeelHelm.json"),1);
        var melee = (MeleePolicy)field("melee").get(bot);
        var gun = (MeleeGun)field("meleeGun").get(bot);
        melee.activate();melee.observe(7,10,200,100,0,0,100);
        var o=melee.opponents().iterator().next();
        gun.queue(11,new Phys.Pose(0,100,0,0),1,new double[]{0,0,0},o,2);
        bot.onBulletFired(new BulletFiredEvent(11,new BulletState(1,1,1,0,100,0,null)));
        assertEquals(1,gun.confirmed);gun.observe(7,40,200,100);assertEquals(1,gun.samples(7));
        gun.queue(41,new Phys.Pose(0,100,0,0),1,new double[]{0,0,0},o,2);
        assertDoesNotThrow(() -> bot.onRoundStarted(new RoundStartedEvent(2)));
        melee.activate();bot.onBulletFired(new BulletFiredEvent(41,new BulletState(2,1,1,0,100,0,null)));
        assertEquals(1,gun.confirmed,"pending shot from previous round cannot be confirmed");
        assertThrows(IllegalArgumentException.class,()->new KeelHelm(BotInfo.fromFile("bot/KeelHelm.json"),128));
    }
    private static Field field(String name) throws Exception {
        Field f = KeelHelm.class.getDeclaredField(name); f.setAccessible(true); return f;
    }

    @Test void roundCallbackNeedsNoLiveTickAndClearsPreviousMeleeRound() throws Exception {
        var bot = new KeelHelm(BotInfo.fromFile("bot/KeelHelm.json"));
        assertThrows(RuntimeException.class, bot::getEnemyCount, "no current tick exists in this fixture");
        var melee = (MeleePolicy)field("melee").get(bot);
        for (int round = 1; round <= 3; round++) {
            melee.activate(); melee.observe(7, 10, 100, 100, 0, 8, 100); melee.died(8);
            for (String name : new String[]{"meleeTurns","meleeShots","duelTransitions"}) field(name).setInt(bot,99);
            int nextRound = round;
            assertDoesNotThrow(() -> bot.onRoundStarted(new RoundStartedEvent(nextRound)));
            assertTrue(melee.opponents().isEmpty()); assertFalse(melee.active()); assertFalse(melee.isDead(8));
            for (String name : new String[]{"meleeTurns","meleeShots","duelTransitions"}) assertEquals(0,field(name).getInt(bot));
            assertEquals(round,field("round").getInt(bot));
            melee.activate(); assertTrue(melee.active(), "first live melee tick can activate after reset");
        }
    }
}
