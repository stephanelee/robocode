import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ContextGunTest {
    private static DuelGun.Scan scan(int turn,double x,double y,double direction,double speed) {
        return new DuelGun.Scan(2,turn,x,y,direction,speed,100);
    }
    private static ContextGun.Query query(ContextGun model,int turn) {
        var before=scan(turn-1,500,300,90,0);var now=scan(turn,500,300,90,0);
        model.onScan(before);model.onScan(now);
        return model.query(turn,now,before,300,300,2,0,200,800,600);
    }
    private static void teach(ContextGun model,int turn) {
        var q=query(model,turn);model.confirmed(q,turn+1,300,300,2);
        double y=300;
        for(int age=1;age<=24;age++) {
            double speed=Math.min(4,age);y+=speed;
            model.onScan(scan(turn+age,500,y,90,speed));
        }
    }

    @Test void onlyConfirmedShotsAndFreshCenterPassagesCreateCausalLabels() {
        var model=new ContextGun();var q=query(model,10);
        for(int turn=11;turn<=40;turn++)model.onScan(scan(turn,500,300,90,0));
        assertEquals(0,model.size());
        q=query(model,100);model.confirmed(q,101,300,300,2);
        for(int turn=101;turn<115;turn++)model.onScan(scan(turn,500,300,90,0));
        assertEquals(0,model.size()); // radius14*(114-101+1)=196<200
        model.onScan(scan(115,500,300,90,0));
        assertEquals(1,model.size());var sample=model.samples().get(0);
        assertEquals(0,sample.guessFactor(),1e-12);assertEquals(115,sample.observedTurn());
        assertTrue(sample.observedSequence()>sample.createdSequence());
    }
    @Test void gapsRoundBoundariesAndTargetChangesMaskUnresolvedLabels() {
        var model=new ContextGun();var q=query(model,10);model.confirmed(q,11,300,300,2);
        model.onScan(scan(12,500,300,90,0));
        for(int turn=13;turn<=40;turn++)model.onScan(scan(turn,500,300,90,0));assertEquals(0,model.size());
        teach(model,100);assertEquals(1,model.size());
        q=query(model,200);model.confirmed(q,201,300,300,2);model.startRound();
        for(int turn=1;turn<=40;turn++)model.onScan(scan(turn,500,300,90,0));assertEquals(1,model.size());
        model.onScan(new DuelGun.Scan(3,41,500,300,90,0,100));assertEquals(0,model.size());
    }
    @Test void confirmedMovingOriginUsesTheSameFrozenTargetReferencePoint() {
        var model=new ContextGun();var before=scan(9,500,400,90,8);var now=scan(10,500,400,90,8);
        model.onScan(before);model.onScan(now);
        var q=model.query(10,now,before,300,300,2,26.565,224,800,600);
        assertEquals(1,q.sign());model.confirmed(q,11,308,300,2);
        for(int turn=11;turn<=40;turn++)model.onScan(scan(turn,500,400,90,0));
        assertEquals(1,model.size());var sample=model.samples().get(0);
        assertEquals(0,sample.guessFactor(),1e-12,"unchanged target point remains GF0 even when shooter moves");
        assertEquals(308,sample.originX());assertEquals(500,sample.referenceX());assertEquals(400,sample.referenceY());
    }
    @Test void sufficientPastContextLearnsConditionalDisplacementWithoutFutureQueryLeakage() {
        var model=new ContextGun();var early=query(model,10);assertTrue(Double.isNaN(model.predict(early)));
        for(int i=0;i<32;i++)teach(model,100+i*100);
        assertTrue(Double.isNaN(model.predict(early)),"later labels cannot make an earlier query supported");
        var fresh=query(model,4000);double angle=model.predict(fresh);
        assertTrue(Double.isFinite(angle));assertTrue(angle>2,"learned passage should differ from linear GF0");
        var before=scan(4099,500,300,170,7);var now=scan(4100,500,300,180,8);
        model.onScan(before);model.onScan(now);
        var unsupported=model.query(4100,now,before,300,300,2,0,200,800,600);
        assertTrue(Double.isNaN(model.predict(unsupported)),"distant feature context falls back");
    }
    @Test void memoryIsFifoBoundedAndFeaturesRespectMissingAndMirrorConventions() {
        var model=new ContextGun();for(int i=0;i<520;i++)teach(model,100+i*100);
        assertEquals(512,model.size());assertEquals(9,model.samples().get(0).insertionSequence());
        assertEquals(520,model.acceptedTotal());
        var before=scan(60000,500,300,90,0);var now=scan(60001,500,300,90,0);
        model.onScan(before);model.onScan(now);var q=model.query(60001,now,before,300,300,2,0,200,800,600);
        assertEquals(1,q.sign());assertEquals(1,q.features()[6]);
        assertNull(model.query(60002,now,before,300,300,2,0,200,800,600),"stale feature input is missing");
    }
    @Test void mismatchedConfirmedPowerDoesNotTrainThePlannedContext() {
        var gun=new PressureGun();gun.onScan(scan(9,500,300,90,0));gun.onScan(scan(10,500,300,90,0));
        var plan=gun.decide(10,300,300,0,100,0,800,600);assertTrue(plan.fire());
        gun.onFired(11,300,300,plan.power()+0.1);
        for(int turn=11;turn<=50;turn++)gun.onScan(scan(turn,500,300,90,0));
        assertEquals(0,gun.context().size());assertArrayEquals(new int[3],gun.attempts());
        var query=gun.context().query(50,scan(50,500,300,90,0),scan(49,500,300,90,0),300,300,1.4,0,200,800,600);
        assertEquals(39/40.0,query.features()[6],1e-12,"every real shot still updates the past-shot clock");
    }
}
