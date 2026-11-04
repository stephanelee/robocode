import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.*;
import com.google.gson.*;
import static org.junit.jupiter.api.Assertions.*;

class PeakTelemetryTest {
    @TempDir Path temp;
    @Test void lateCallbacksRewriteSealedModelsWithoutDoubleCountingRounds() throws Exception {
        var hits=new HitAngles();var gun=new CrossingGun();
        var journal=new PeakTelemetry(temp.resolve("state").toString(),17,
            ()->new PeakTelemetry.Models(hits.snapshot(),gun.snapshot(),new double[]{.2,.3,.25},java.util.List.of()));
        var cold=JsonParser.parseString(Files.readString(temp.resolve("state/cold.json"))).getAsJsonObject();
        assertEquals(0,cold.get("round").getAsInt());assertEquals(0,cold.get("completedRounds").getAsInt());
        assertFalse(cold.get("checkpointLoaded").getAsBoolean());assertTrue(cold.get("coldStateSHA256").isJsonNull());
        journal.start(1);journal.end(1);
        long firstRevision=journal.revision;
        var wave=new HitAngles.Wave(0,0,0,0,20,2);
        journal.hitLabel(hits.observe(java.util.List.of(wave),0,14,0,0,2));
        journal.stats.confirmedShots++;journal.stats.shotEnergy+=2;journal.changed();
        var late=JsonParser.parseString(Files.readString(temp.resolve("state/endpoint.json"))).getAsJsonObject();
        assertEquals(1,late.get("completedRounds").getAsInt());assertEquals(1,late.get("round").getAsInt());
        assertTrue(late.get("revision").getAsLong()>firstRevision);
        assertEquals(1,late.get("cumulativeAcceptedHits").getAsInt());
        assertEquals(1,late.getAsJsonObject("models").getAsJsonObject("hitAngles").get("effectiveTotal").getAsDouble());
        assertEquals(Files.readString(temp.resolve("state/endpoint.json")),Files.readString(temp.resolve("state/round-001.json")));
        assertThrows(IllegalStateException.class,()->journal.end(1));
        journal.start(2);assertEquals(0,journal.stats.confirmedShots);assertEquals(1,journal.cumulativeAcceptedHits);
        journal.end(2);assertEquals(2,journal.completedRounds);
        assertThrows(java.io.UncheckedIOException.class,()->new PeakTelemetry(temp.resolve("state").toString(),17,
            ()->new PeakTelemetry.Models(hits.snapshot(),gun.snapshot(),new double[3],java.util.List.of())));
    }
    @Test void cappedGunSnapshotIsReadOnlyAndCold() {
        var state=new CrossingGun().snapshot();assertEquals(192,state.capacity());assertEquals(0,state.count());
        assertEquals(0,state.next());assertTrue(state.examples().isEmpty());assertTrue(state.inFlight().isEmpty());assertNull(state.requested());
        assertThrows(UnsupportedOperationException.class,()->state.examples().add(new CrossingGun.ExampleSnapshot(0,new CrossingGun.Context(0,0,0,0),0)));
    }
}
