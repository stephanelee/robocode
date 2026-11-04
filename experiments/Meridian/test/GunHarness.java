import dev.robocode.tankroyale.botapi.Bot;
import dev.robocode.tankroyale.botapi.events.*;
import java.lang.reflect.*;
import static org.mockito.Mockito.*;

/** No server, sockets, run loop or hidden opponent inputs: only the gun and scan callbacks. */
final class GunHarness {
    final Bot bot;
    final Method gun;
    int turn, round=1, requests;
    double energy=100, heat, heading, rate, power;
    Flight.Pose own=new Flight.Pose(100,300,0,0);
    Flight.Pose target=new Flight.Pose(500,300,90,8);
    double enemyEnergy=100;
    boolean accepted=true;
    GunHarness(boolean baseline) throws Exception {
        bot=baseline ? spy(new BaselineMeridian()) : spy(new Meridian());
        doAnswer(i->energy).when(bot).getEnergy();
        doAnswer(i->heat).when(bot).getGunHeat();
        doAnswer(i->heading).when(bot).getGunDirection();
        doAnswer(i->own.x()).when(bot).getX();
        doAnswer(i->own.y()).when(bot).getY();
        doAnswer(i->own.h()).when(bot).getDirection();
        doAnswer(i->own.v()).when(bot).getSpeed();
        doAnswer(i->turn).when(bot).getTurnNumber();
        doReturn(800).when(bot).getArenaWidth();
        doReturn(600).when(bot).getArenaHeight();
        doAnswer(i->{rate=i.getArgument(0);return null;}).when(bot).setGunTurnRate(anyDouble());
        doAnswer(i->{requests++; power=i.getArgument(0);return accepted;}).when(bot).setFire(anyDouble());
        gun=bot.getClass().getDeclaredMethod("gun",int.class,Flight.Pose.class,double.class);
        gun.setAccessible(true);
        bot.onRoundStarted(new RoundStartedEvent(1));
    }
    void scan(int t,Flight.Pose p,double e) {
        target=p;enemyEnergy=e;
        bot.onScannedBot(new ScannedBotEvent(t,1,2,e,p.x(),p.y(),p.h(),p.v()));
    }
    void decide(int t) throws Exception {
        turn=t;requests=0;
        gun.invoke(bot,t,own,Math.hypot(target.x()-own.x(),target.y()-own.y()));
        clearInvocations(bot);
    }
    Object field(String name) throws Exception { return field(bot,name); }
    static Object field(Object object,String name) throws Exception {
        Field f=object.getClass().getDeclaredField(name);f.setAccessible(true);return f.get(object);
    }
    void set(String name,Object value) throws Exception {
        Field f=bot.getClass().getDeclaredField(name);f.setAccessible(true);f.set(bot,value);
    }
    void confirm(int t) {
        bot.onBulletFired(new BulletFiredEvent(t,
            new dev.robocode.tankroyale.botapi.BulletState(1,1,power,own.x(),own.y(),heading,null)));
    }
    Meridian.AimPlan plan() throws Exception { return (Meridian.AimPlan)field("aimPlan"); }
    CrossingGun.Launch requested() throws Exception {
        return (CrossingGun.Launch)field(field("crossingGun"),"requested");
    }
}
