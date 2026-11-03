import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class DuelGunTest {
    private static DuelGun.Scan scan(int turn,double x,double y,double heading,double speed,double energy) {
        return new DuelGun.Scan(2,turn,x,y,heading,speed,energy);
    }
    private static DuelGun.Input input(int turn,DuelGun.Scan s,double gun,double energy,double heat) {
        return new DuelGun.Input(turn,400,300,gun,energy,heat,800,600,s,Double.NaN);
    }
    private static DuelGun.Plan aligned(DuelGun gun,int turn,DuelGun.Scan scan) {
        DuelGun.Plan aim=gun.decide(input(turn,scan,0,100,0));
        return gun.decide(input(turn,scan,aim.direction(),100,0));
    }

    @Test void forecastsMoveOnOldHeadingThenRotateAndStopAtFirstWallContact() {
        var s=scan(0,600,300,0,8,100);
        assertArrayEquals(new double[]{608,300},DuelGun.forecast(s,1,3,30,800,600),1e-9);
        assertArrayEquals(new double[]{608+8*Math.cos(Math.toRadians(4)),300+8*Math.sin(Math.toRadians(4))},
                DuelGun.forecast(s,2,3,30,800,600),1e-9);
        assertArrayEquals(new double[]{600,300},DuelGun.forecast(s,20,1,30,800,600),1e-9);
        assertArrayEquals(new double[]{592,300},DuelGun.forecast(scan(0,600,300,0,-8,100),1,2,0,800,600),1e-9);
        assertArrayEquals(new double[]{782,502},DuelGun.forecast(scan(0,780,500,45,8,100),20,2,0,800,600),1e-9);
        assertArrayEquals(new double[]{782,582},DuelGun.forecast(scan(0,780,580,45,8,100),20,2,0,800,600),1e-9);
    }

    @Test void lowEnergyCanFinishWithoutExhaustingMovementReserve() {
        assertTrue(DuelGun.firepower(11,100,450)>0);
        assertTrue(DuelGun.firepower(11,100,450)<=.8);
        assertEquals(.1,DuelGun.firepower(.6,.2,100),1e-9);
        assertEquals(0,DuelGun.firepower(.3,100,100));
        assertEquals(0,DuelGun.firepower(100,0,100));
        for(double own:new double[]{.5,1,2,7,10,100}) {
            double power=DuelGun.firepower(own,100,100);
            assertTrue(power>=.1 && power<=3 && own-power>=(own<8?.35:2)-1e-9);
        }
    }

    @Test void currentHeadingNotRequestedNewDirectionControlsFire() {
        var gun=new DuelGun(); var s=scan(0,700,300,0,0,100);
        var turned=gun.decide(input(0,s,90,100,0));
        assertEquals(0,turned.direction(),1e-9); assertFalse(turned.shouldFire());
        assertTrue(gun.decide(input(0,s,0,100,0)).shouldFire());
        assertTrue(gun.decide(input(0,s,360,100,0)).shouldFire());
        assertFalse(gun.decide(input(0,s,180,100,0)).shouldFire());
        assertFalse(gun.decide(input(0,s,0,100,.1)).shouldFire());
        assertFalse(gun.decide(input(3,s,0,100,0)).shouldFire());
    }

    @Test void actualCreationTurnCollisionLookaheadAndScanBeforeFiredAreSupported() {
        var gun=new DuelGun();var s=scan(0,445,300,0,0,2);
        var plan=aligned(gun,0,s); assertEquals(.5,plan.power(),1e-9);assertTrue(plan.shouldFire());
        gun.onScan(scan(1,445,300,0,0,2));
        gun.onFired(1,400,300,.5);
        assertEquals(1,gun.attempts("current"));assertEquals(1,gun.hits("current"));
        assertEquals(1,gun.hits("headon"));
        assertEquals(8,DuelGun.segmentDistance(445,300,400,300,0,18.5,37),1e-9);
        assertTrue(DuelGun.segmentDistance(445,300,400,300,0,0,18.5)>18);
    }

    @Test void requestsWrongConfirmationAndMissingScansNeverCreateFalseMisses() {
        var gun=new DuelGun();var s=scan(0,700,300,0,0,100);
        var plan=aligned(gun,0,s);assertTrue(plan.shouldFire());
        gun.onScan(scan(1,700,300,0,0,100));assertEquals(0,gun.attempts("current"));
        gun.onFired(1,400,300,plan.power()+.1);
        for(int t=2;t<50;t++)gun.onScan(scan(t,700,300,0,0,100));
        assertEquals(0,gun.attempts("current"));
        gun.startRound();plan=aligned(gun,0,s);gun.onFired(1,400,300,plan.power());
        gun.onScan(scan(2,700,300,0,0,100)); // Missing creation-turn fresh observation => unknown cohort.
        for(int t=3;t<50;t++)gun.onScan(scan(t,700,300,0,0,100));
        assertEquals(0,gun.attempts("current"));
    }

    @Test void virtualEvidenceSelectsCircularFromLegalPositionsWithoutAnyHitCallbacks() {
        var gun=new DuelGun();var origin=scan(1,650,250,0,8,100);
        for(int cohort=0;cohort<12;cohort++) {
            gun.startRound();
            gun.onScan(scan(0,650-8*Math.cos(Math.toRadians(-4)),250-8*Math.sin(Math.toRadians(-4)),-4,8,100));
            var p=aligned(gun,1,origin);assertTrue(p.shouldFire());
            gun.onFired(2,400,300,p.power());
            for(int t=2;t<90;t++) {
                double[] xy=DuelGun.forecast(origin,t-1,3,4,800,600);
                gun.onScan(scan(t,xy[0],xy[1],4*(t-1),8,100));
            }
        }
        assertEquals(12,gun.attempts("circular"));
        assertTrue(gun.hits("circular")>gun.hits("current"));
        gun.startRound();gun.onScan(scan(0,642.02,250.558,-4,8,100));
        var plan=gun.decide(input(1,origin,0,100,0));
        assertEquals("circular",plan.model());assertFalse(plan.fallback());
        assertEquals(.5,plan.confidence(),1e-9);
        gun.onScan(new DuelGun.Scan(3,2,650,250,0,8,100));
        assertEquals(0,gun.attempts("circular"));
    }
}
