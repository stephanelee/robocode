import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class BreakwaterFireGuardTest {
    private static BreakwaterFireGuard.Input input(double gun,DuelGun.Scan target) {
        return new BreakwaterFireGuard.Input(10,400,300,800,600,gun,1.2,target);
    }
    @Test void actualCurrentRayMustBeCoveredEvenWhenADifferentAimWouldHaveAgreement() {
        var target=new DuelGun.Scan(2,10,450,300,90,0,100);
        var aligned=BreakwaterFireGuard.evaluate(input(0,target));
        assertEquals(3,aligned.coveredPaths());assertTrue(aligned.allow());
        var opposite=BreakwaterFireGuard.evaluate(input(180,target));
        assertEquals(0,opposite.coveredPaths());assertFalse(opposite.allow());
    }
    @Test void repeatedFootprintsFromOnePathCannotBecomeTwoPathAgreement() {
        var target=new DuelGun.Scan(2,10,700,300,90,0,100);
        var result=BreakwaterFireGuard.evaluate(new BreakwaterFireGuard.Input(10,100,300,800,600,0,1.2,target));
        assertEquals(1,result.coveredPaths());assertEquals(3,result.totalPaths());assertFalse(result.allow());
    }
    @Test void footprintUsesEntireBulletAnnulusIncludingTangentAndWraparound() {
        assertEquals(Math.toDegrees(Math.asin(.18)),BreakwaterFireGuard.halfFootprint(100,95,105),1e-12);
        assertEquals(0,BreakwaterFireGuard.halfFootprint(100,118,130),1e-12);
        assertTrue(Double.isNaN(BreakwaterFireGuard.halfFootprint(100,0,81)));
        var target=new DuelGun.Scan(2,10,350,300,270,0,100);
        assertEquals(BreakwaterFireGuard.evaluate(input(180,target)),BreakwaterFireGuard.evaluate(input(-180,target)));
        assertTrue(BreakwaterFireGuard.evaluate(input(-180,target)).allow());
    }
    @Test void oldHeadingMovementPrecedesLimitedTurnAndWallIntersectionStopsSpeed() {
        var before=new BreakwaterFireGuard.Pose(780,300,0,8);
        var reverse=BreakwaterFireGuard.step(before,4,-8);
        assertEquals(783,reverse.x(),1e-12);assertEquals(300,reverse.y(),1e-12);
        assertEquals(3,reverse.speed(),1e-12);assertEquals(4,reverse.heading(),1e-12);
        var stopped=BreakwaterFireGuard.step(before,0,0);assertEquals(780,stopped.x(),1e-12);
        var wall=BreakwaterFireGuard.wall(before,reverse,800,600);
        assertEquals(782,wall.x(),1e-12);assertEquals(300,wall.y(),1e-12);assertEquals(0,wall.speed());
        var corner=BreakwaterFireGuard.wall(new BreakwaterFireGuard.Pose(780,580,45,8),
                new BreakwaterFireGuard.Pose(786,586,49,8),800,600);
        assertEquals(782,corner.x(),1e-12);assertEquals(582,corner.y(),1e-12);assertEquals(0,corner.speed());
    }
    @Test void missingStaleFutureAndInvalidInputsCannotInventCoverage() {
        assertFalse(BreakwaterFireGuard.evaluate(input(0,null)).allow());
        assertFalse(BreakwaterFireGuard.evaluate(input(0,new DuelGun.Scan(2,8,450,300,90,0,100))).allow());
        assertFalse(BreakwaterFireGuard.evaluate(input(0,new DuelGun.Scan(2,11,450,300,90,0,100))).allow());
        assertFalse(BreakwaterFireGuard.evaluate(input(Double.NaN,new DuelGun.Scan(2,10,450,300,90,0,100))).allow());
        assertTrue(BreakwaterFireGuard.evaluate(input(0,new DuelGun.Scan(2,9,450,300,90,0,100))).allow());
    }
}
