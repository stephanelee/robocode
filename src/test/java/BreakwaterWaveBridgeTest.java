import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class BreakwaterWaveBridgeTest {
    private static EnemyWave wave(){return new EnemyWave(100,300,14,99,9,2);}
    @Test void actualCreationAndReferenceUseSavedAimStateInsteadOfLaterCallbackPosition() {
        var wave=wave();var own=new AimCorridor.OwnState(9,300,300,90,8);
        var bridge=BreakwaterWaveBridge.from(wave,own);
        assertEquals(10,bridge.creationTurn());assertEquals(0,bridge.reference(),1e-12);
        assertEquals(8,bridge.lateral(),1e-12);assertEquals(99,wave.bearingToUs);
        assertNull(BreakwaterWaveBridge.from(wave,new AimCorridor.OwnState(10,300,308,90,8)));
        assertNull(BreakwaterWaveBridge.from(wave,null));
        var mirrored=BreakwaterWaveBridge.from(wave,new AimCorridor.OwnState(9,300,300,270,8));
        assertEquals(-8,mirrored.lateral(),1e-12);
    }
    @Test void creationTickRayHasOneSpeedOfTravelAndCorrectOwnerPower() {
        var wave=wave();
        assertSame(wave,BreakwaterWaveBridge.uniqueMatch(List.of(wave),2,2,10,114,300,0));
        assertNull(BreakwaterWaveBridge.uniqueMatch(List.of(wave),2,2,10,128,300,0));
        assertNull(BreakwaterWaveBridge.uniqueMatch(List.of(wave),3,2,10,114,300,0));
        assertNull(BreakwaterWaveBridge.uniqueMatch(List.of(wave),2,1.9,10,114,300,0));
        assertNull(BreakwaterWaveBridge.uniqueMatch(List.of(wave),2,2,9,100,300,0));
    }
    @Test void broadRadialMatchCannotTrainARayWithWrongCollinearityOrDirection() {
        var wave=wave();
        assertTrue(Double.isFinite(wave.matchError(2,2,240,301,19)),"legacy broad radial match accepts it");
        assertNull(BreakwaterWaveBridge.uniqueMatch(List.of(wave),2,2,19,240,301,0));
        assertNull(BreakwaterWaveBridge.uniqueMatch(List.of(wave),2,2,19,240,300,180));
        assertSame(wave,BreakwaterWaveBridge.uniqueMatch(List.of(wave),2,2,19,240,300,360));
    }
    @Test void duplicatePlausibleRaysAreUnknownAndMatchingDoesNotMutateLegacyList() {
        var a=wave();var duplicate=wave();var list=new java.util.ArrayList<>(List.of(a,duplicate));
        assertNull(BreakwaterWaveBridge.uniqueMatch(list,2,2,19,240,300,0));assertEquals(2,list.size());
        assertSame(a,BreakwaterWaveBridge.uniqueMatch(List.of(a),2,2,19,240,300,0));
        assertNull(BreakwaterWaveBridge.uniqueMatch(List.of(a),2,2,19,Double.NaN,300,0));
    }
}
