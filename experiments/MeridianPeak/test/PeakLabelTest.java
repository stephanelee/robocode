import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.*;
import java.util.List;
import com.google.gson.JsonParser;
import static org.junit.jupiter.api.Assertions.*;

class PeakLabelTest {
    @TempDir Path temp;
    private static CrossingGun gun() {
        var gun=new CrossingGun();
        var shot=CrossingGun.launch(10,new Flight.Pose(100,300,0,0),new Flight.Pose(242,300,90,8),2,800,600);
        gun.requested(shot);gun.fired(10,100,300,2);return gun;
    }
    private static Flight.Pose at(double angle,double range) {
        return new Flight.Pose(100+range*Flight.cos(angle),300+range*Flight.sin(angle),90,8);
    }
    @Test void passiveLabelsRetainEveryRemovedLaunchAndDoNotMutateLearner() {
        var reasons=new CrossingGun.Reason[]{CrossingGun.Reason.RESOLVED,CrossingGun.Reason.MISSING_BRACKET,
            CrossingGun.Reason.INSIDE_PREVIOUS_FRONT,CrossingGun.Reason.OUTSIDE_OFFSET};
        for(var reason:reasons) {
            var gun=gun();
            var enemy=at(reason==CrossingGun.Reason.OUTSIDE_OFFSET?90:20,142);
            var previous=at(19,reason==CrossingGun.Reason.INSIDE_PREVIOUS_FRONT?140:142);
            int previousTurn=reason==CrossingGun.Reason.MISSING_BRACKET?17:18;
            assertTrue(gun.labelsFor(18,enemy,17,previous).isEmpty());
            var labels=gun.labelsFor(19,enemy,previousTurn,previous);
            assertEquals(1,labels.size());assertEquals(reason,labels.get(0).reason());
            assertEquals(1,gun.snapshot().inFlight().size());assertEquals(0,gun.snapshot().count());
            assertEquals(reason==CrossingGun.Reason.RESOLVED?1:0,gun.observe(19,enemy,previousTurn,previous));
            assertTrue(gun.snapshot().inFlight().isEmpty());
        }
    }
    @Test void maskedTrialAndCrossingJournalReconcilesWithSealedCounters() throws Exception {
        var gun=gun();var hits=new HitAngles();
        var telemetry=new PeakTelemetry(temp.resolve("state").toString(),0,
            ()->new PeakTelemetry.Models(hits.snapshot(),gun.snapshot(),new double[]{.2,.3,.25},List.of()));
        telemetry.start(1);var labels=gun.labelsFor(19,at(20,142),17,at(19,142));
        telemetry.crossingLabels(labels);telemetry.crossings(gun.observe(19,at(20,142),17,at(19,142)));
        telemetry.trial(20,new PeakTelemetry.TrialSnapshot(20,new double[]{0,10,20},100,300),at(20,142));
        telemetry.end(1);
        telemetry.trial(22,new PeakTelemetry.TrialSnapshot(21,new double[]{0,10,20},100,300),at(20,142));telemetry.changed();
        var rows=Files.readAllLines(temp.resolve("state/labels.jsonl"));assertEquals(3,rows.size());
        assertEquals("MISSING_BRACKET",JsonParser.parseString(rows.get(0)).getAsJsonObject().getAsJsonObject("label").get("reason").getAsString());
        assertEquals(1,telemetry.stats.crossingMaskedMissingBracket);assertEquals(0,telemetry.stats.crossingLabels);
        assertEquals(1,telemetry.stats.trialScored);assertEquals(1,telemetry.stats.trialMissingScan);
        assertEquals(1,telemetry.completedRounds);
        String endpoint=Files.readString(temp.resolve("state/endpoint.json"));
        assertFalse(endpoint.contains("NaN"));assertFalse(endpoint.contains("Infinity"));
    }
    @Test void cumulativeLabelsRemainDistinctFromCappedBankAndDecayedHistogram() {
        var gun=new CrossingGun();var hits=new HitAngles();
        var telemetry=new PeakTelemetry(null,0,()->new PeakTelemetry.Models(hits.snapshot(),gun.snapshot(),new double[3],List.of()));
        telemetry.start(1);
        var shot=CrossingGun.launch(10,new Flight.Pose(100,300,0,0),new Flight.Pose(242,300,90,8),2,800,600);
        var wave=new HitAngles.Wave(0,0,0,0,20,2);
        for(int i=0;i<201;i++) {
            gun.requested(shot);gun.fired(10,100,300,2);
            telemetry.crossings(gun.observe(19,at(20,142),18,at(19,142)));
            telemetry.hitLabel(hits.observe(List.of(wave),0,14,0,0,2));
        }
        telemetry.end(1);
        assertEquals(201,telemetry.cumulativeCrossings);assertEquals(201,telemetry.stats.crossingLabels);
        assertEquals(192,gun.snapshot().count());assertEquals(9,gun.snapshot().next());assertEquals(192,gun.snapshot().examples().size());
        assertEquals(201,telemetry.cumulativeAcceptedHits);assertTrue(hits.snapshot().effectiveTotal()<20);
        assertEquals(hits.snapshot().effectiveTotal(),java.util.Arrays.stream(hits.snapshot().weights()).sum(),1e-12);
    }
}
