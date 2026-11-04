import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class BreakwaterPolicyTest {
    private static BreakwaterPolicy.Own own(int turn,double x,double y,double angle,double speed,double gun,double heat) {
        return new BreakwaterPolicy.Own(turn,x,y,angle,speed,gun,heat,100,.1,800,600);
    }
    private static BreakwaterPolicy.Target target(int turn,double x,double y,double heading,double speed) {
        return new BreakwaterPolicy.Target(turn,x,y,heading,speed,100);
    }

    @Test void collisionRequiresSameTickSegments() {
        assertTrue(BreakwaterPolicy.collide(110,100,125,100,116,95,116,105));
        assertFalse(BreakwaterPolicy.collide(125,100,140,100,116,95,116,105));
        assertTrue(BreakwaterPolicy.collide(100,100,115,100,120,100,105,100));
        assertFalse(BreakwaterPolicy.collide(100,100,115,100,120,100.00001,105,100.00001));
    }

    @Test void inferredHeadOnWaveProducesLegalSidestepAndAnActualHeadingCollision() {
        BreakwaterPolicy p = new BreakwaterPolicy(2);
        var w = new BreakwaterPolicy.Wave(100,120,300,1.95,0);
        double x=620,y=300,h=90,v=0,gun=177;
        for (int now=100;now<130;now++) {
            var d=p.decide(own(now,x,y,h,v,gun,0),target(now,120,300,90,5),List.of(w));
            assertEquals(BreakwaterPolicy.Mode.SHIELD,d.mode());
            if (d.power()>0) {
                assertEquals(.1,d.power());
                assertTrue(Math.abs(y-300)>=2);
                boolean crosses=false;
                for(int t=now+1;t<145;t++) {
                    double in=19.7*(t-(now+1)+1), a=Math.toRadians(gun), en=w.speed()*(t-100+1);
                    crosses |= BreakwaterPolicy.collide(x+in*Math.cos(a),y+in*Math.sin(a),
                        x+(in+19.7)*Math.cos(a),y+(in+19.7)*Math.sin(a),120+en,300,120+en+w.speed(),300);
                }
                assertTrue(crosses);
                p.shieldLaunched(w.creation());
                var next=p.decide(own(now+1,x,y,h,v,gun,1),target(now+1,120,300,90,5),List.of(w));
                assertNotEquals(BreakwaterPolicy.Mode.SHIELD,next.mode());
                return;
            }
            gun+=Math.max(-20,Math.min(20,CombatMath.relativeAngle(d.aim()-gun)));
            v=DuelSurf.nextSpeed(v,d.targetSpeed()); x+=v*Math.cos(Math.toRadians(h)); y+=v*Math.sin(Math.toRadians(h));
            h+=Math.max(-DuelSurf.maxTurn(v),Math.min(DuelSurf.maxTurn(v),d.turnRate()));
        }
        fail("Shield never launched");
    }

    @Test void hotGunDoesNotMakeBaitWaitForAnImpossibleShield() {
        var p=new BreakwaterPolicy(5);
        var d=p.decide(own(90,610,300,90,0,180,4),target(90,120,300,90,0),
                       List.of(new BreakwaterPolicy.Wave(69,120,300,2,0)));
        assertEquals(BreakwaterPolicy.Mode.EVADE,d.mode());
        assertNotEquals(0,d.targetSpeed()); assertEquals(0,d.power());
    }

    @Test void staleScanEvadesAndWallFallbackDoesNotContinueIntoTheWall() {
        var p=new BreakwaterPolicy(5);
        var d=p.decide(own(13,19,300,180,8,0,0),target(9,430,300,90,0),List.of());
        assertEquals(BreakwaterPolicy.Mode.EVADE,d.mode());
        double v=DuelSurf.nextSpeed(8,d.targetSpeed());
        assertTrue(19-v>18); assertEquals(0,d.power());
    }

    @Test void attackRequiresLineAgreementOrAFinisherAndActualGunHeading() {
        var p=new BreakwaterPolicy(3);
        assertEquals(0,p.decide(own(20,100,300,90,0,0,0),target(20,620,300,90,8),List.of()).power());
        assertEquals(0,p.decide(own(20,100,300,90,0,25,0),target(20,620,300,90,0),List.of()).power());
        assertEquals(1,p.decide(own(20,100,300,90,0,0,0),target(20,620,300,90,0),List.of()).power());
    }

    @Test void confirmationCallbacksAndRoundBoundariesKeepOnlyResolvedLearning() {
        var p=new BreakwaterPolicy(2);
        var t=target(20,180,300,90,0);
        p.decide(own(20,100,300,90,0,0,0),t,List.of());
        p.observe(target(21,180,300,90,0)); // scan before fired event on same tick
        p.attackLaunched(21,100,300,1,t);
        p.observe(target(22,180,300,90,0)); p.observe(target(23,180,300,90,0));
        p.observe(target(24,180,300,90,0));
        assertArrayEquals(new int[]{1,1,1},p.attackScores());
        p.attackLaunched(26,100,300,1,target(25,180,300,90,0));
        p.observe(target(27,180,300,90,0)); // gap masks both rays
        p.matched(new BreakwaterPolicy.Wave(1,0,0,2,358),2);
        p.startRound(); p.observe(target(1,180,300,90,0));
        assertArrayEquals(new int[]{1,1,1},p.attackScores());
        assertArrayEquals(new double[]{4},p.corrections());
        var cold=new BreakwaterPolicy(2);
        assertArrayEquals(new int[]{0,0,0},cold.attackScores()); assertEquals(0,cold.corrections().length);
    }

    @Test void earliestThreatOwnsTheGunAndEightResolvedCorrectionsAreBounded() {
        var p=new BreakwaterPolicy(1);
        var w=new BreakwaterPolicy.Wave(20,100,300,1,0);
        for(int i=0;i<12;i++) p.matched(w,i==11?-40:i);
        assertArrayEquals(new double[]{4,5,6,7,8,9,10,-12},p.corrections());
        var d=p.decide(own(21,610,305,90,0,180,0),target(21,100,300,90,0),
                List.of(new BreakwaterPolicy.Wave(21,100,300,3,0),w));
        assertEquals(20,d.waveTurn()); assertTrue(d.solution());
    }
}
