import org.junit.jupiter.api.Test;
import java.util.SplittableRandom;
import static org.junit.jupiter.api.Assertions.*;

class PressureStrategyTest {
    private static DuelGun.Scan scan(int turn) {return new DuelGun.Scan(2,turn,400,300,0,0,100);}

    @Test void nearWallMovementCannotProjectInstantlyOntoTheDesiredHeading() {
        var command=PressureMove.choose(780,300,0,8,100,new DuelGun.Scan(2,100,500,300,0,0,100),0,1,800,600);
        assertEquals(0,command.targetSpeed(),1e-12);
        assertTrue(780+DuelSurf.nextSpeed(8,command.targetSpeed())<=782);
    }
    @Test void openArenaPursuitClosesRangeAndPhaseTimingIsSeeded() {
        var command=PressureMove.choose(100,300,0,0,100,new DuelGun.Scan(2,100,600,300,0,0,100),0,1,800,600);
        assertEquals(8,command.targetSpeed(),1e-12);
        assertTrue(Math.abs(CombatMath.relativeAngle(command.bodyHeading()))<45);
        var a=new SplittableRandom(45);var b=new SplittableRandom(45);
        for(int i=0;i<100;i++){int value=PressureMove.phaseDuration(a);assertEquals(value,PressureMove.phaseDuration(b));assertTrue(value>=20&&value<=45);}
    }
    @Test void powerConservesApproachEnergyAndKillsRatherThanOnlyDisabling() {
        assertEquals(0.6,PressureGun.power(100,100,500),1e-12);
        assertEquals(3,PressureGun.power(100,100,100),1e-12);
        assertEquals(0.1,PressureGun.power(10,0,100),1e-12);
        double power=PressureGun.power(100,4,100);
        assertTrue(4*power+Math.max(0,2*(power-1))>4);
        assertEquals(0,PressureGun.power(0.3,100,100),1e-12);
    }
    @Test void firingGateUsesTheCurrentGunRay() {
        var gun=new PressureGun();gun.onScan(scan(10));
        assertFalse(gun.decide(10,300,300,180,100,0,800,600).fire());
        assertTrue(gun.decide(10,300,300,0,100,0,800,600).fire());
        assertFalse(gun.decide(10,300,300,0,100,1,800,600).fire());
    }
    @Test void virtualLearningNeedsConfirmedFireAndFreshConsecutiveScans() {
        var gun=new PressureGun();gun.onScan(scan(10));gun.decide(10,300,300,0,100,0,800,600);
        for(int turn=11;turn<=22;turn++)gun.onScan(scan(turn));
        assertArrayEquals(new int[3],gun.attempts());
        gun.startRound();gun.onScan(scan(10));gun.decide(10,300,300,0,100,0,800,600);
        gun.onFired(11,300,300,3);
        for(int turn=11;turn<=22;turn++)gun.onScan(scan(turn));
        assertArrayEquals(new int[]{1,1,1},gun.attempts());assertArrayEquals(new int[]{1,1,1},gun.hits());
        gun.startRound();gun.onScan(scan(10));gun.decide(10,300,300,0,100,0,800,600);gun.onFired(11,300,300,3);
        gun.onScan(scan(13));
        assertArrayEquals(new int[]{1,1,1},gun.attempts(),"a missing observation is not a miss");
    }
}
