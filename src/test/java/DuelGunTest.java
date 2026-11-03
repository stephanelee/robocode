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
        assertEquals(.1,DuelGun.firepower(100,0,100));
        assertEquals(0,DuelGun.firepower(100,-.01,100));
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
        var plan=aligned(gun,0,s); assertTrue(plan.power()>.5);assertTrue(plan.shouldFire());
        gun.onScan(scan(1,445,300,0,0,2));
        gun.onFired(1,400,300,plan.power());
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

    @Test void actualLegacyGeometryUsesItsGFRotatedFootprintInsteadOfLinearTrajectory() {
        var gun=new DuelGun();var s=scan(0,700,300,90,8,100);
        double power=DuelGun.firepower(100,s.energy(),300);
        var aim=TargetingGeometry.calculate(new TargetingGeometry.Snapshot(
                400,300,800,600,0,0,700,300,90,8,0,0,300,20-3*power,-.6,8));
        var legacy=new DuelGun.LegacyAim(aim.aimAngle(),Math.hypot(aim.x()-400,aim.y()-300),power);
        var plan=gun.decide(new DuelGun.Input(0,400,300,aim.aimAngle(),100,0,800,600,s,legacy));
        assertEquals(aim.aimAngle(),plan.direction(),1e-9);assertTrue(plan.shouldFire());
        var linear=new DuelGun().decide(input(0,s,aim.aimAngle(),100,0));
        assertFalse(linear.shouldFire()); // The newly supplied GF forecast differs from the old linear alias.
        var linearAim=new DuelGun().decide(input(0,s,0,100,0));
        assertFalse(gun.decide(new DuelGun.Input(0,400,300,linearAim.direction(),100,0,800,600,s,legacy)).shouldFire());
        var rawCenter=new DuelGun.LegacyAim(aim.baseAngle(),legacy.range(),power);
        assertFalse(gun.decide(new DuelGun.Input(0,400,300,aim.aimAngle(),100,0,800,600,s,rawCenter)).shouldFire());
    }

    @Test void invalidOrDifferentPowerLegacyHintFallsBackAndVirtualCurrentFreezesHintAngle() {
        var s=scan(0,700,300,0,0,100);var gun=new DuelGun();
        var linear=gun.decide(input(0,s,0,100,0));
        for(var hint:new DuelGun.LegacyAim[]{new DuelGun.LegacyAim(90,300,linear.power()+.1),
                new DuelGun.LegacyAim(90,Double.NaN,linear.power()),new DuelGun.LegacyAim(90,0,linear.power())}) {
            var fallback=gun.decide(new DuelGun.Input(0,400,300,0,100,0,800,600,s,hint));
            assertEquals(linear.direction(),fallback.direction());assertEquals(linear.shouldFire(),fallback.shouldFire());
        }
        var legacy=new DuelGun.LegacyAim(15,300,linear.power());
        var plan=gun.decide(new DuelGun.Input(0,400,300,15,100,0,800,600,s,legacy));
        assertTrue(plan.shouldFire());gun.onFired(1,400,300,plan.power());
        for(int t=1;t<40;t++)gun.onScan(scan(t,700,300,0,0,100));
        assertEquals(1,gun.attempts("current"));assertEquals(0,gun.hits("current"));
        assertEquals(1,gun.hits("linear")); // Actual hit callbacks are neither supplied nor used as labels.
    }

    @Test void legacyFootprintChecksCurrentHeadingRangeAndCreationLookahead() {
        var gun=new DuelGun();var s=scan(0,700,300,0,0,100);
        double p=DuelGun.firepower(100,100,300);
        var far=new DuelGun.LegacyAim(0,300,p);
        assertFalse(gun.decide(new DuelGun.Input(0,400,300,9,100,0,800,600,s,far)).shouldFire());
        var near=new DuelGun.LegacyAim(0,45,p);
        assertTrue(gun.decide(new DuelGun.Input(0,400,300,9,100,0,800,600,s,near)).shouldFire());
        assertFalse(gun.decide(new DuelGun.Input(0,400,300,180,100,0,800,600,s,near)).shouldFire());
        assertTrue(gun.decide(new DuelGun.Input(0,400,300,360,100,0,800,600,s,near)).shouldFire());
    }

    @Test void zeroEnergyIsAliveAndFinishingPowerCrossesTheStrictDeathBoundary() {
        for(double energy:new double[]{0,1e-12,.4,2,3.9999999,4,4.0000001}) {
            double p=DuelGun.firepower(100,energy,100);
            double damage=4*p+2*Math.max(p-1,0);
            assertTrue(damage>energy,"Energy "+energy+" must become negative, not merely zero");
            assertTrue(p>=.1 && p<=3);
        }
        assertEquals(0,DuelGun.firepower(100,-1e-12,100));
        assertEquals(0,DuelGun.firepower(.44,0,100));
        assertEquals(.1,DuelGun.firepower(.45,0,100),1e-12);
        assertTrue(.45-DuelGun.firepower(.45,0,100)>=.35-1e-12);
        assertTrue(new DuelGun().decide(input(0,scan(0,445,300,0,0,0),0,100,0)).shouldFire());
    }
}
