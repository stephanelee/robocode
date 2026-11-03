import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class BreakwaterSurfTest {
    @Test void firstStepUsesOldHeadingAndEngineStopReversalRules() {
        var state=new BreakwaterSurf.Pose(780,300,0,8);
        var safe=BreakwaterSurf.travel(state,500,300,1,800,600);
        assertEquals(0,safe.targetSpeed());assertEquals(780,BreakwaterSurf.step(state,safe).x(),1e-12);
        var reversed=BreakwaterSurf.step(state,new BreakwaterSurf.Command(4,-8,-1));
        assertEquals(783,reversed.x(),1e-12);assertEquals(300,reversed.y(),1e-12);
        assertEquals(3,reversed.speed(),1e-12);assertEquals(4,reversed.heading(),1e-12);
    }
    @Test void backAsFrontMovesAlongTheChosenOrbitWithoutInstantHeadingProjection() {
        var state=new BreakwaterSurf.Pose(300,300,270,0);
        var motion=BreakwaterSurf.travel(state,730,300,1,1000,800);
        assertEquals(-8,motion.targetSpeed());assertEquals(0,motion.turnRate(),1e-12);
        var next=BreakwaterSurf.step(state,motion);
        assertEquals(300,next.x(),1e-12);assertEquals(301,next.y(),1e-12);
    }
    @Test void seededPoliciesReplayAndMovingTargetsStayWithinPhysicalBounds() {
        var a=new BreakwaterSurf(917);var b=new BreakwaterSurf(917);
        var state=new BreakwaterSurf.Pose(780,580,45,8);
        for(int turn=1;turn<=250;turn++) {
            var own=new BreakwaterSurf.Own(turn,state.x(),state.y(),state.heading(),state.speed(),800,600);
            double ex=400+100*Math.sin(turn/30.0),ey=300+80*Math.cos(turn/40.0);
            var x=a.choose(own,ex,ey,List.of());assertEquals(x,b.choose(own,ex,ey,List.of()));
            assertTrue(Math.abs(x.turnRate())<=DuelSurf.maxTurn(state.speed())+1e-12);
            assertTrue(Math.abs(x.targetSpeed())==8||x.targetSpeed()==0);
            state=BreakwaterSurf.step(state,x);
            assertTrue(state.x()>=18&&state.x()<=782&&state.y()>=18&&state.y()<=582);
        }
    }
    @Test void matchedRayMemoryIsSignedBoundedAndRoundPersistentWithoutLearningUnknownWaves() {
        var policy=new BreakwaterSurf(3);
        var wave=new BreakwaterSurf.Threat(1,100,300,14,0,-8);
        policy.matched(new BreakwaterSurf.Threat(1,100,300,14,0,.5),20);
        policy.matched(wave,Double.NaN);assertEquals(0,policy.corrections().length);
        for(int i=0;i<40;i++)policy.matched(wave,-10);
        assertEquals(32,policy.corrections().length);assertTrue(policy.corrections()[0]>0);
        double[] before=policy.corrections();before[0]=999;
        assertNotEquals(999,policy.corrections()[0]);
        policy.startRound();assertEquals(32,policy.corrections().length);
    }
    @Test void expiredAndFutureWavesHaveTheSamePolicyAsNoKnownWave() {
        var own=new BreakwaterSurf.Own(100,400,300,90,8,800,600);
        var a=new BreakwaterSurf(9);var b=new BreakwaterSurf(9);
        var invalidTime=List.of(new BreakwaterSurf.Threat(101,100,300,14,0,8),
                new BreakwaterSurf.Threat(1,100,300,14,0,8));
        assertEquals(a.choose(own,100,300,List.of()),b.choose(own,100,300,invalidTime));
    }
    @Test void onlyTheFirstTwoIncomingWaveIdentitiesAffectPolicySelection() {
        var own=new BreakwaterSurf.Own(30,400,300,90,8,800,600);
        var first=new BreakwaterSurf.Threat(20,100,300,14,0,8);
        var second=new BreakwaterSurf.Threat(25,100,300,14,0,8);
        var later=new BreakwaterSurf.Threat(30,100,300,14,0,-8);
        var a=new BreakwaterSurf(11);var b=new BreakwaterSurf(11);
        assertEquals(a.choose(own,100,300,List.of(first,second)),b.choose(own,100,300,List.of(later,second,first)));
    }
}
