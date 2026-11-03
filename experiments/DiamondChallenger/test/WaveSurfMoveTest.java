import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class WaveSurfMoveTest {
    private static WaveSurfMove.Threat wave(double[] bins) {
        return new WaveSurfMove.Threat(100,300,14,10,0,1,4,bins);
    }
    @Test void realFirstStepUsesOldHeadingAndImmediateStopOrEngineReversal() {
        var state=new WaveSurfMove.State(780,300,0,8);
        var stopped=WaveSurfMove.step(state,new DuelSurf.Command(90,0,0));
        assertEquals(780,stopped.x(),1e-12);assertEquals(0,stopped.speed(),1e-12);
        assertEquals(4,stopped.heading(),1e-12,"request limited at old speed even when new speed is zero");
        var reversed=WaveSurfMove.step(state,new DuelSurf.Command(90,-8,-1));
        assertEquals(783,reversed.x(),1e-12);assertEquals(300,reversed.y(),1e-12);
        assertEquals(3,reversed.speed(),1e-12);assertEquals(4,reversed.heading(),1e-12);
    }
    @Test void annulusIncludesTangentBetweenBulletSegmentEndpoints() {
        assertEquals(Math.asin(18.0/100),WaveSurfMove.angularFootprint(100,95,105),1e-12);
        assertTrue(Double.isNaN(WaveSurfMove.angularFootprint(100,0,81)));
        assertEquals(0,WaveSurfMove.angularFootprint(100,118,130),1e-12);
    }
    @Test void sparseAndInvalidProfilesUseIdenticalPriorAndSnapshotCannotBeMutated() {
        double[] bins=new double[47];bins[0]=2.99;
        var prior=WaveSurfMove.density(wave(null));
        assertArrayEquals(prior,WaveSurfMove.density(wave(bins)),1e-12);
        bins[0]=Double.NaN;assertArrayEquals(prior,WaveSurfMove.density(wave(bins)),1e-12);
        bins[0]=3;var t=wave(bins);bins[0]=100;
        assertEquals(3,t.matchedBins47()[0]);t.matchedBins47()[0]=55;
        assertEquals(3,t.matchedBins47()[0]);
        assertEquals(1,java.util.Arrays.stream(WaveSurfMove.density(t)).sum(),1e-12);
        assertTrue(WaveSurfMove.density(t)[0]>prior[0]);
    }
    @Test void edgeSmoothingAndLargeProfilesRetainFiniteNormalizedMass() {
        double[] bins=new double[47];bins[46]=320;
        var d=WaveSurfMove.density(wave(bins));
        assertEquals(1,java.util.Arrays.stream(d).sum(),1e-12);
        assertEquals(1,WaveSurfMove.integrate(d,-2,2),1e-12);
        assertEquals(0,WaveSurfMove.integrate(d,2,3),1e-12);
        assertTrue(d[46]>d[45]);
    }
    @Test void noThreatAndAlreadyPassedThreatUseAcceptedWallSafeFallback() {
        var accepted=DuelSurf.choose(780,300,0,8,100,500,300,List.of(),1,800,600);
        var noWave=WaveSurfMove.choose(780,300,0,8,100,500,300,List.of(),1,800,600);
        var passed=WaveSurfMove.choose(780,300,0,8,100,500,300,List.of(wave(null)),1,800,600);
        assertEquals(accepted,noWave.command());assertEquals(accepted,passed.command());
        assertFalse(passed.plannerUsed());assertEquals(0,noWave.command().targetSpeed(),1e-12);
    }
    @Test void thirdLaterWaveCannotChangeSelectedPlanOrProfileDiagnostic() {
        var a=new WaveSurfMove.Threat(100,300,14,10,0,1,4,null);
        var b=new WaveSurfMove.Threat(150,300,14,10,0,1,4,null);
        double[] bins=new double[47];bins[2]=32;
        var later=new WaveSurfMove.Threat(0,300,14,10,0,1,4,bins);
        var pair=WaveSurfMove.choose(400,300,90,0,10,100,300,List.of(a,b),1,800,600);
        var triple=WaveSurfMove.choose(400,300,90,0,10,100,300,List.of(a,b,later),1,800,600);
        assertEquals(pair,triple);assertTrue(pair.plannerUsed());assertFalse(pair.profileUsed());
    }
    @Test void unresolvedDistantWaveRetainsCurrentOptionWithoutPretendingNoPlanner() {
        var distant=new WaveSurfMove.Threat(100,5000,11,10,0,-1,8,null);
        var choice=WaveSurfMove.choose(9000,5000,90,0,10,8550,5000,List.of(distant),-1,10000,10000);
        assertTrue(choice.plannerUsed());assertEquals(-1,choice.option());
        assertFalse(choice.profileUsed());
    }
    @Test void collisionSegmentUsesEngineTickIncrementAndPredictedOwnPosition() {
        var t=new WaveSurfMove.Threat(0,300,14,10,0,1,0,null);
        assertEquals(14,WaveSurfMove.segmentStart(t,10),1e-12,"creation checks [v,2v]");
        assertEquals(28,WaveSurfMove.segmentStart(t,11),1e-12);
        assertFalse(Double.isNaN(WaveSurfMove.angularFootprint(55,28,42)));
        var moved=WaveSurfMove.step(new WaveSurfMove.State(55,300,0,8),new DuelSurf.Command(0,8,1));
        assertEquals(63,moved.x(),1e-12);
        assertTrue(Double.isNaN(WaveSurfMove.angularFootprint(moved.x(),28,42)));
    }
    @Test void impossibleMovingPathsFallBackToImmediateSafeStop() {
        var imminent=new WaveSurfMove.Threat(500,300,14,10,0,1,0,null);
        var choice=WaveSurfMove.choose(780,300,0,8,10,500,300,List.of(imminent),1,800,600);
        assertEquals(0,choice.command().targetSpeed(),1e-12);
    }
}
