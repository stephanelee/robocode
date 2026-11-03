import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class GuardedGunTest {
    private static DuelGun.Scan scan(int turn){return new DuelGun.Scan(2,turn,500,300,90,0,100);}
    private static PressureGun supportedGun() {return supportedGun(false);}
    private static PressureGun supportedGun(boolean displaced) {
        var gun=new PressureGun();var model=gun.context();
        for(int i=0;i<32;i++) {
            int t=100+i*100;model.onScan(scan(t-1));model.onScan(scan(t));
            var q=model.query(t,scan(t),scan(t-1),300,300,1.4,0,200,800,600);
            model.confirmed(q,t+1,300,300,1.4);
            for(int k=1;k<=30;k++)model.onScan(displaced
                    ?new DuelGun.Scan(2,t+k,500,300+Math.min(4,k)*k,90,Math.min(4,k),100):scan(t+k));
        }
        gun.onScan(scan(3999));gun.onScan(scan(4000));return gun;
    }
    @Test void strictSharedSupportAndMarginRejectSparseUnequalAndTiedEvidence() {
        assertFalse(PressureGun.learnedQualified(new int[]{7,7,7,7},new int[]{0,0,0,7}));
        assertFalse(PressureGun.learnedQualified(new int[]{8,8,8,9},new int[]{0,0,0,8}));
        assertFalse(PressureGun.learnedQualified(new int[]{8,8,8,8},new int[]{4,4,4,4}));
        assertTrue(PressureGun.learnedQualified(new int[]{8,8,8,8},new int[]{0,1,2,3}));
        // At N=48 the one-success advantage is exactly .02: the fixed rule requires STRICTLY more.
        assertFalse(PressureGun.learnedQualified(new int[]{48,48,48,48},new int[]{1,1,1,2}));
    }
    @Test void supportedPredictionDoesNotOverrideLegacyBeforePairedOutcomesExist() {
        var gun=supportedGun();var plan=gun.decide(4000,300,300,0,100,0,800,600);
        assertEquals(1,plan.model());assertTrue(plan.fire());assertEquals(1,gun.supportedDecisions());
        assertEquals(0,gun.gateEligibleDecisions());assertEquals(0,gun.learnedAimDecisions());
        gun.onFired(4001,300,300,plan.power());assertEquals(1,gun.pairedPending());
        assertArrayEquals(new int[4],gun.pairedAttempts(),"confirmation alone cannot leak future skill");
        for(int t=4001;t<4011;t++)gun.onScan(scan(t));
        assertArrayEquals(new int[4],gun.pairedAttempts());
        for(int t=4011;t<=4030;t++)gun.onScan(scan(t));
        assertArrayEquals(new int[]{1,1,1,1},gun.pairedAttempts());
        assertArrayEquals(new int[]{1,1,1,1},gun.pairedHits());assertEquals(1,gun.pairedCompletedRound());
        gun.startRound();assertArrayEquals(new int[]{1,1,1,1},gun.pairedAttempts());
        assertEquals(0,gun.pairedPending());assertEquals(0,gun.pairedCompletedRound());
        gun.onScan(new DuelGun.Scan(3,1,500,300,90,0,100));assertArrayEquals(new int[4],gun.pairedAttempts());
    }
    @Test void gapDiscardsAllFourAndMismatchedPowerNeverCreatesPairedEvidence() {
        var gun=supportedGun();var plan=gun.decide(4000,300,300,0,100,0,800,600);
        gun.onFired(4001,300,300,plan.power());gun.onScan(scan(4002));
        assertEquals(1,gun.pairedDiscardedRound());assertEquals(0,gun.pairedPending());
        assertArrayEquals(new int[4],gun.pairedAttempts());
        gun=supportedGun();plan=gun.decide(4000,300,300,0,100,0,800,600);
        gun.onFired(4001,300,300,plan.power()+.1);
        assertEquals(0,gun.pairedCreatedRound());assertArrayEquals(new int[4],gun.pairedAttempts());
    }
    @Test void unsupportedQueriesKeepOriginalLegacyLearningWithoutPairedPopulation() {
        var gun=new PressureGun();gun.onScan(scan(9));gun.onScan(scan(10));
        var plan=gun.decide(10,300,300,0,100,0,800,600);gun.onFired(11,300,300,plan.power());
        for(int t=11;t<=40;t++)gun.onScan(scan(t));
        assertArrayEquals(new int[]{1,1,1},gun.attempts());assertArrayEquals(new int[4],gun.pairedAttempts());
        assertEquals(0,gun.supportedDecisions());
    }
    @Test void oldGlobalThreeResolveBeforeThePairedFourthMissAndKeepOriginalTiming() {
        var gun=supportedGun(true);var plan=gun.decide(4000,300,300,0,100,0,800,600);
        assertEquals(1,plan.model());assertEquals(1,gun.supportedDecisions());
        gun.onFired(4001,300,300,plan.power());
        for(int t=4001;t<=4011;t++)gun.onScan(scan(t));
        assertArrayEquals(new int[]{1,1,1},gun.attempts());
        assertArrayEquals(new int[4],gun.pairedAttempts(),"fourth miss is unresolved, but old counts must not wait");
        for(int t=4012;t<=4030;t++)gun.onScan(scan(t));
        assertArrayEquals(new int[]{1,1,1,1},gun.pairedAttempts());
        assertArrayEquals(new int[]{1,1,1,0},gun.pairedHits());
    }
    @Test void learnedRayBecomesSelectableOnlyAfterEightResolvedSharedAdvantages() {
        var gun=supportedGun(true);
        for(int i=0;i<8;i++) {
            int t=4000+i*100;
            if(i>0){gun.onScan(scan(t-1));gun.onScan(scan(t));}
            var plan=gun.decide(t,300,300,0,100,0,800,600);
            assertNotEquals(3,plan.model());assertTrue(plan.fire());
            gun.onFired(t+1,300,300,plan.power());
            for(int age=1;age<=30;age++)gun.onScan(new DuelGun.Scan(2,t+age,500,
                    300+Math.min(4,age)*age,90,Math.min(4,age),100));
        }
        assertArrayEquals(new int[]{8,8,8,8},gun.pairedAttempts());
        assertTrue(PressureGun.learnedQualified(gun.pairedAttempts(),gun.pairedHits()));
        gun.onScan(scan(4999));gun.onScan(scan(5000));
        assertEquals(3,gun.decide(5000,300,300,0,100,0,800,600).model());
    }
}
