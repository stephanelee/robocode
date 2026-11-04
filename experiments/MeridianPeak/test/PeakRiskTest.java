import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class PeakRiskTest {
    @Test void finiteSegmentUsesCentreDistanceAtInteriorAndEndpoints() {
        assertEquals(9,PeakRisk.distanceSquared(5,3,0,0,10,0));
        assertEquals(13,PeakRisk.distanceSquared(-2,3,0,0,10,0));
        assertEquals(13,PeakRisk.distanceSquared(12,3,0,0,10,0));
        assertEquals(40,PeakRisk.kernel(0));
        assertEquals(20,PeakRisk.kernel(500));
    }
    @Test void rayPeaksPrecedeMixtureInsteadOfMixturePeak() {
        assertEquals(80,PeakRisk.weightedPeaks(new double[]{1,1},new double[][]{{40,0},{0,40}}));
        assertEquals(40,Math.max(40+0,0+40));
    }
    @Test void firstSweepAndFutureClockAreExplicit() {
        var wave=new HitAngles.Wave(7,100,100,0,0,2);
        assertEquals(14,PeakRisk.nearRadius(wave,6,1)); // launch sweep [v,2v]
        assertEquals(28,PeakRisk.nearRadius(wave,7,1)); // first future sweep
        assertEquals(42,PeakRisk.nearRadius(wave,7,2));
    }
    @Test void onlyNearEndpointKillsRayAndBoundRemainsEighty() {
        var own=new Flight.Pose(700,300,0,0);var profile=new HitAngles();
        var wave=new HitAngles.Wave(0,770,300,0,0,3);
        var result=PeakRisk.evaluate(own,0,1,1,770,300,List.of(wave),profile,800,600);
        assertEquals(2,result.kernelEvaluations()); // first near792/far803 survives; next near803 dies
        assertTrue(result.maximumWaveRisk()>0 && result.maximumWaveRisk()<=80+1e-9);
        var outside=new HitAngles.Wave(0,799,300,0,0,3);
        assertEquals(0,PeakRisk.evaluate(own,0,1,1,799,300,List.of(outside),profile,800,600).kernelEvaluations());
        assertTrue(PeakRisk.outside(800,200,800,600));
        assertTrue(PeakRisk.outside(0,200,800,600));
    }
    @Test void rayMixturePreservesOriginalRiskAndFallbackBoundary() {
        var profile=new HitAngles();var wave=new HitAngles.Wave(0,0,0,0,20,2);
        assertEquals(2,profile.rays(wave).size());
        assertEquals(HitAngles.Label.ACCEPTED,profile.observe(List.of(wave),0,14,0,0,2));
        var rays=profile.rays(wave);assertEquals(3,rays.size());
        assertEquals(2,rays.stream().mapToDouble(HitAngles.Ray::weight).sum(),1e-12);
        double reconstructed=0;
        for(var ray:rays) {
            double miss=30*Flight.sin(ray.heading())-7*Flight.cos(ray.heading());
            reconstructed+=ray.weight()*40/(1+miss*miss/500);
        }
        assertEquals(profile.risk(wave,30,7),reconstructed,1e-12);
        assertEquals(2,profile.rays(new HitAngles.Wave(0,0,0,0,.999,2)).size());
        assertEquals(3,profile.rays(new HitAngles.Wave(0,0,0,0,1,2)).size());
        var copy=profile.snapshot();copy.weights()[10]=10000;
        assertEquals(1,profile.snapshot().weights()[10]);
    }
}
