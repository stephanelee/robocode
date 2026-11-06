import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MeleeDisplacementGunTest {
    private void stationary(MeleeDisplacementGun g, int id, int count) {
        for (int t=1;t<=count;t+=4) g.observe(id,t,500,500,0,0,100,100,1000,1000);
    }
    @Test void coldSparseAndOtherIdsUseCircular() {
        var g=new MeleeDisplacementGun();g.newRound();stationary(g,2,40);
        var self=new Phys.Pose(100,100,0,0);var circular=new Phys.Pose(520,510,0,0);
        assertFalse(g.predict(2,37,self,1,circular,1000,1000).learned());
        assertFalse(g.predict(3,37,self,1,circular,1000,1000).learned());
    }
    @Test void learnsActualStationaryEndpointsAndResetsTransientsNotModel() {
        var g=new MeleeDisplacementGun();g.newRound();stationary(g,2,800);
        assertEquals(1024,g.sampleCount(2));
        var self=new Phys.Pose(100,100,0,0);var circular=new Phys.Pose(520,510,0,0);
        var aim=g.predict(2,797,self,1,circular,1000,1000);
        assertTrue(aim.learned());assertEquals(500,aim.x(),1e-8);assertEquals(500,aim.y(),1e-8);
        long queries=g.queries;g.predict(2,797,self,1,circular,1000,1000);assertEquals(queries,g.queries);
        g.clearTransient();assertEquals(1024,g.sampleCount(2));
        assertFalse(g.predict(2,797,self,1,circular,1000,1000).learned());
        g.newRound();g.observe(2,1,500,500,0,0,100,100,1000,1000);
        assertEquals(1024,g.sampleCount(2));
        assertFalse(g.predict(2,1,self,1,circular,1000,1000).learned());
    }
    @Test void rejectsIllegalGapsAndTeleportLabels() {
        var g=new MeleeDisplacementGun();g.newRound();
        g.observe(2,1,100,100,0,8,200,200,1000,1000);
        g.observe(2,5,132,100,0,8,200,200,1000,1000);
        g.observe(2,13,900,900,0,8,200,200,1000,1000);
        assertEquals(0,g.sampleCount(2));
        g.died(2);g.observe(2,17,900,900,0,8,200,200,1000,1000);
        assertEquals(0,g.sampleCount(2));
        g.newRound();g.observe(2,1,100,100,0,8,200,200,1000,1000);
        g.observe(2,30,100,100,0,8,200,200,1000,1000);
        g.observe(2,38,100,100,0,8,200,200,1000,1000);
        assertEquals(0,g.sampleCount(2));
    }
    @Test void actualShotExposureRequiresActualDirectionAndSurvivesTransition() {
        var g=new MeleeDisplacementGun();g.newRound();
        g.pending(10,2,1,45,40,true);g.confirm(11,1,1,40);
        assertTrue(g.stats().contains("learnedConfirmed=0"));
        g.pending(12,2,1,45,40,true);g.confirm(13,2,1,44);
        assertTrue(g.stats().contains("learnedConfirmed=1 learnedTargets=2:1"));
        g.clearTransient();g.hit(2,2,4);g.wall(1);
        assertTrue(g.stats().contains("confirmedHits=1 confirmedWalls=1"));
        assertTrue(g.stats().contains("confirmedCensored=0"));
        g.pending(20,3,1,45,40,true);g.confirm(22,3,1,45);
        assertTrue(g.stats().contains("confirmedMelee=2"));
        assertEquals(2,g.trace().split(";").length);
    }
    @Test void boundsShotLedgerAndReportsTraceOverflow() {
        var g=new MeleeDisplacementGun();g.newRound();
        for(int i=0;i<1030;i++){g.pending(i*2,2,1,45,40,true);g.confirm(i*2+1,i,1,45);}
        assertTrue(g.stats().contains("confirmedCensored=1030"));
        assertTrue(g.stats().contains("traceOverflow=6"));
        assertEquals(1024,g.trace().split(";").length);
    }
    @Test void wallRaysAndBodyRelativeFeatures() {
        assertEquals(.25,MeleeDisplacementGun.ray(118,118,180,1000,1000),1e-8);
        var o=new MeleeDisplacementGun.Observation(1,1,118,118,0,-8,218,118,1000,1000,10,-2,true);
        var f=MeleeDisplacementGun.features(o,48);
        assertEquals(0,f[0]);assertEquals(1,f[1]);assertEquals(0,f[2]);assertEquals(.5,f[4]);assertEquals(1,f[9]);
    }
    @Test void cachedQueryGuardUsesTheOriginalNeighborHorizon() throws Exception {
        var g=new MeleeDisplacementGun();g.newRound();
        for(int t=1;t<=800;t+=4) g.observe(2,t,800,500,0,0,100,500,2000,1000);
        var self=new Phys.Pose(100,500,0,0);
        // Install a consistent legal mean to isolate the cache guard from sample fitting.
        var field=MeleeDisplacementGun.class.getDeclaredField("models");field.setAccessible(true);
        Object model=((java.util.Map<?,?>)field.get(g)).get(2);
        for(String name:new String[]{"cachedTurn","cachedPower","cachedHorizon","cached"}) {
            var f=model.getClass().getDeclaredField(name);f.setAccessible(true);
            switch(name) {
                case "cachedTurn" -> f.set(model,797);
                case "cachedPower" -> f.set(model,1.0);
                case "cachedHorizon" -> f.set(model,40.0);
                case "cached" -> f.set(model,new MeleeDisplacementGun.Estimate(0.55,0,0));
            }
        }
        // Current circular horizon43.8 reuses query40. The outward learned forecast
        // refines beyond52, so it must fall back even if within12 of current43.8.
        var circular=new Phys.Pose(844.6,500,0,0);
        var aim=g.predict(2,797,self,1,circular,2000,1000);
        assertEquals(0,g.queries);assertFalse(aim.learned());
    }
}
