import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class OutgoingGunWiringTest {
    private static DuelGun.Scan scan(int turn,double y){return new DuelGun.Scan(2,turn,500,y,0,0,100);}
    @Test void queryNullShotUsesDecisionReferenceAndKnownRemovalDoesNotCensorVirtualCohort() {
        var gun=new PressureGun();gun.startRound();gun.onScan(scan(1,300));
        var aim=gun.decide(1,300,300,0,100,0,800,600);assertTrue(aim.fire());
        // A later scan is delivered before the fire callback: it must not replace the decision reference.
        gun.onScan(scan(2,320));gun.onFired(2,99,300,300,aim.power(),0);
        assertEquals(1,gun.context().outgoingCount());assertEquals(1,gun.context().outgoingCreatedRound());
        gun.onScan(scan(3,340));
        var q=gun.context().query(3,scan(3,340),scan(2,320),300,300,aim.power(),0,200,800,600);
        double expected=CombatMath.bearing(200,40)/Math.toDegrees(Math.asin(8/(20-3*aim.power())));
        assertEquals(1,q.features()[7]);assertEquals(expected,q.features()[9],1e-12);
        gun.removed(99);assertEquals(0,gun.context().outgoingCount());
        assertEquals(1,q.features()[7],"The frozen query cannot be edited by a later removal");
        for(int turn=4;turn<=25;turn++)gun.onScan(scan(turn,340));
        assertArrayEquals(new int[]{1,1,1},gun.attempts(),"Known removal affects feature estimates, not existing virtual labels");
    }
}
