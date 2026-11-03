import org.junit.jupiter.api.Test;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.*;

class OutgoingContextTest {
    private static DuelGun.Scan scan(int turn){return new DuelGun.Scan(2,turn,500,300,90,0,100);}
    private static ContextGun model() {var m=new ContextGun();m.onScan(scan(9));m.onScan(scan(10));return m;}
    private static ContextGun.Query query(ContextGun m,int turn) {
        m.onScan(scan(turn-1));m.onScan(scan(turn));
        return m.query(turn,scan(turn),scan(turn-1),300,300,2,0,200,800,600);
    }
    @Test void queryCannotSeeCurrentFutureShotAndStoredSampleRetainsFrozenThirteenFeatures() {
        var m=model();var before=query(m,10);var ref=m.reference(10,scan(10),300,300);
        assertEquals(13,before.features().length);assertArrayEquals(new double[6],Arrays.copyOfRange(before.features(),7,13));
        m.confirmed(before,11,308,308,2,7,ref);
        var after=query(m,12);assertEquals(1,after.features()[7]);assertEquals(0,after.features()[9],1e-12,
                "frozen target point is reanchored to the same actual confirmed origin");
        double range=Math.hypot(500-308,300-308);
        assertEquals((range-28)/14/40,after.features()[8],1e-12);
        assertArrayEquals(new double[6],Arrays.copyOfRange(before.features(),7,13));
        for(int turn=13;turn<=40;turn++)m.onScan(scan(turn));
        assertEquals(1,m.size());assertArrayEquals(before.features(),m.samples().get(0).features());
    }
    @Test void nearestTwoUseRawCenterEtaRatherThanCreationOrderAndKnownRemovalIsExact() {
        var m=model();m.confirmed(null,11,300,300,3,1,m.reference(10,scan(10),300,300));
        m.onScan(scan(11));m.confirmed(null,12,300,300,.1,2,m.reference(11,scan(11),300,300));
        m.onScan(scan(12));m.confirmed(null,13,300,300,2,3,m.reference(12,scan(12),300,300));
        var q=query(m,14);
        assertEquals(1,q.features()[7]);assertEquals(1,q.features()[10]);
        assertEquals((200-19.7*3)/19.7/40,q.features()[8],1e-12);
        assertEquals((200.0-14*2)/14/40,q.features()[11],1e-12);
        m.removed(999);assertEquals(3,m.outgoingCount());m.removed(2);
        q=query(m,14);assertEquals((200.0-14*2)/14/40,q.features()[8],1e-12);
        assertEquals((200.0-11*4)/11/40,q.features()[11],1e-12);
        assertEquals(2,m.outgoingKnownRemovedRound());assertEquals(3,m.outgoingCreatedRound());
        assertEquals(2,m.waveContextQueriesRound());assertEquals(2,m.waveContextTwoRound());
    }
    @Test void missingReferenceAndMissingIdentityAreAbsentNotInventedAndRoundTargetResetClearsEstimates() {
        var m=model();assertNull(m.reference(11,scan(10),300,300));
        m.confirmed(null,11,300,300,2,1,null);assertEquals(0,m.outgoingCount());
        m.confirmed(null,11,300,300,2,-1,m.reference(10,scan(10),300,300));assertEquals(0,m.outgoingCount());
        var old=m.reference(10,scan(10),300,300);m.confirmed(null,11,300,300,2,1,old);
        assertEquals(1,m.outgoingCount());m.startRound();assertEquals(0,m.outgoingCount());
        m.onScan(scan(9));m.onScan(scan(10));m.confirmed(null,11,300,300,2,1,old);
        assertEquals(0,m.outgoingCount(),"a previous round reference is unavailable");
        m.confirmed(null,11,300,300,2,1,m.reference(10,scan(10),300,300));
        m.onScan(new DuelGun.Scan(3,11,500,300,90,0,100));assertEquals(0,m.outgoingCount());
    }
    @Test void contactingNegativeCenterEtaClipsToZeroAndThenPassedWaveBecomesAbsent() {
        var m=model();m.confirmed(null,11,300,300,2,1,m.reference(10,scan(10),300,300));
        var q=query(m,25); // radius210, targetcenter200: still within the18 body envelope.
        assertEquals(1,q.features()[7]);assertEquals(0,q.features()[8]);
        q=query(m,26);assertArrayEquals(new double[6],Arrays.copyOfRange(q.features(),7,13));
        assertEquals(0,m.outgoingCount());
    }
    @Test void fixedLedgerCapAndMirroredDisplacementRemainFinite() {
        var m=model();
        for(int i=10;i<70;i++) {
            m.onScan(scan(i));m.confirmed(null,i+1,300,300,2,i,m.reference(i,scan(i),300,300));
        }
        assertEquals(48,m.outgoingCount());
        m.startRound();m.onScan(new DuelGun.Scan(2,9,500,300,270,8,100));
        var current=new DuelGun.Scan(2,10,500,300,270,8,100);m.onScan(current);
        var ref=m.reference(10,current,300,300);assertEquals(-1,ref.sign());
        m.confirmed(null,11,300,300,2,1,ref);
        var before=new DuelGun.Scan(2,11,500,292,270,8,100);current=new DuelGun.Scan(2,12,500,284,270,8,100);
        m.onScan(before);m.onScan(current);
        var q=m.query(12,current,before,300,300,2,-5,200,800,600);
        assertTrue(q.features()[9]>0);for(double f:q.features())assertTrue(Double.isFinite(f));
    }
    @Test void removalBeforeFireCallbackCannotResurrectKnownRemovedEstimate() {
        var m=model();var ref=m.reference(10,scan(10),300,300);
        m.removed(7);m.confirmed(null,11,300,300,2,7,ref);
        assertEquals(0,m.outgoingCount());assertEquals(1,m.outgoingOmittedRound());
        assertEquals(1,m.outgoingKnownRemovedRound());m.removed(7);
        assertEquals(1,m.outgoingKnownRemovedRound(),"duplicate removal is not a second callback outcome");
        m.startRound();m.onScan(scan(9));m.onScan(scan(10));
        m.confirmed(null,11,300,300,2,7,m.reference(10,scan(10),300,300));
        assertEquals(1,m.outgoingCount(),"round-local IDs can be reused after reset");
    }
}
