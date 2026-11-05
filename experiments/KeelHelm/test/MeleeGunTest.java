import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class MeleeGunTest {
    @Test void crossfireDetectsThirdOpponentEvenWhenNearestPairIsOnOneSide() {
        var self=new Phys.Pose(500,500,0,8);
        var positions=new Phys.Pose[]{new Phys.Pose(600,500,0,0),new Phys.Pose(620,510,0,0),new Phys.Pose(300,500,0,0)};
        assertTrue(MeleePolicy.crossfire(self,positions,new double[]{100,121,200}));
        assertFalse(MeleePolicy.crossfire(self,positions,new double[]{100,121,501}));
    }
    private static MeleePolicy.Opponent target(int id, int t) {
        var o = new MeleePolicy.Opponent(id); o.turn = t; o.x = 200; o.y = 100; o.energy = 100; return o;
    }
    private static void train(MeleeGun gun, int id, int n, double[] angles) {
        for (int i = 0; i < n; i++) {
            int turn = 30 * i + 1;
            var o = target(id, turn - 1);
            gun.queue(turn, new Phys.Pose(0, 100, 0, 0), 1, angles, o, 2);
            gun.confirm(turn, 1); gun.observe(id, turn + 20, 200, 100);
        }
    }
    @Test void confirmedWavesSelectBetterModelPerIdentityWithoutCrossContamination() {
        var gun = new MeleeGun(); train(gun, 1, 16, new double[]{0,0,90});
        assertEquals(16, gun.samples(1)); assertEquals(1, gun.choice(1,true));
        assertEquals(2, gun.choice(2,true)); assertEquals(2, gun.choice(1,false));
        gun.newRound(); assertEquals(1, gun.choice(1,true), "learning stays inside this game across rounds");
    }
    @Test void unconfirmedOrWrongTurnPowerShotsNeverTrain() {
        var gun = new MeleeGun(); var o = target(1,0);
        gun.queue(1,new Phys.Pose(0,100,0,0),1,new double[]{0,0,0},o,2);
        gun.observe(1,20,200,100); assertEquals(0,gun.samples(1));
        gun.confirm(2,1); gun.observe(1,21,200,100); assertEquals(0,gun.samples(1));
        gun.queue(22,new Phys.Pose(0,100,0,0),1,new double[]{0,0,0},o,2);
        gun.confirm(22,2); gun.observe(1,45,200,100); assertEquals(0,gun.samples(1));
    }
    @Test void deathRoundAndDuelTransitionDiscardTransientWaves() {
        for (int operation = 0; operation < 3; operation++) {
            var gun = new MeleeGun(); var o = target(1,0);
            gun.queue(1,new Phys.Pose(0,100,0,0),1,new double[]{0,0,0},o,2); gun.confirm(1,1);
            if (operation == 0) gun.died(1); else if (operation == 1) gun.newRound(); else gun.clearTransient();
            gun.observe(1,20,200,100); assertEquals(0,gun.samples(1));
        }
    }
    @Test void scanGapsResolveOnlyOnceAndNeedMinimumSamples() {
        var gun = new MeleeGun(); train(gun,1,15,new double[]{0,0,90});
        assertEquals(2,gun.choice(1,true)); gun.observe(1,1000,200,100); assertEquals(15,gun.samples(1));
    }
    @Test void inactivePolicyUsesExactInheritedPowerAndCircularAim() {
        var gun = new MeleeGun(); var o = target(1,20);o.speed=5;o.heading=70;o.turnRate=2;
        var self = new Phys.Pose(600,400,30,-4);
        for (double d : new double[]{50,250,499,700}) for (double energy : new double[]{3,19,20,100}) {
            double p = MeleePolicy.power(d,energy,o.energy);
            assertEquals(p,gun.power(o,d,energy,2,false),0);
            var predicted=MeleePolicy.intercept(o,25,self,p,1000,1000);
            assertEquals(Phys.bearing(self.x(),self.y(),predicted.x(),predicted.y()),MeleeGun.angles(o,25,self,p,1000,1000)[2],0);
        }
    }
    @Test void confidencePowerHonorsWarmupAndKillEnergyCaps() {
        var gun = new MeleeGun();var o = target(1,0);
        assertEquals(MeleePolicy.power(200,100,100),gun.power(o,200,100,2,true),0);
        train(gun,1,20,new double[]{90,90,90});
        assertEquals(.7*MeleePolicy.power(200,100,100),gun.power(o,200,100,2,true),1e-12);
        var good = new MeleeGun(); train(good,1,20,new double[]{0,0,0});o.energy=1;
        assertTrue(good.power(o,200,100,2,true)<=.3);
        assertTrue(good.power(o,200,3,2,true)<=.65);
    }
    @Test void futureOpponentPositionsRespectArenaAndForecastMotion() {
        var o = target(1,0);o.x=980;o.y=500;o.speed=8;o.heading=0;
        var forecast=MeleePolicy.forecasts(new MeleePolicy.Opponent[]{o},10,1000,1000);
        for (var row : forecast) {assertTrue(row[0].x()<=982);assertTrue(row[0].y()>=18 && row[0].y()<=982);}
        assertTrue(forecast[0][0].x()>o.x);
    }
    @Test void smartTargetAccountsForGunTurnAndDeadStaleOpponents() {
        var m = new MeleePolicy();m.newRound(3);var gun = new MeleeGun();var self=new Phys.Pose(500,500,0,0);
        m.observe(1,10,700,500,0,0,100);m.observe(2,10,300,500,0,0,100);
        assertEquals(1,m.targetSmart(10,self,0,100,1000,1000,gun,4).id);
        m.died(1);assertEquals(2,m.targetSmart(10,self,0,100,1000,1000,gun,4).id);
        assertNull(m.targetSmart(43,self,0,100,1000,1000,gun,4));
    }
}
