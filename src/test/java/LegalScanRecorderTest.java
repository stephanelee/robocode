import com.google.gson.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.*;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

class LegalScanRecorderTest {
    @TempDir Path temp;
    static final LegalScanRecorder.Own OWN = new LegalScanRecorder.Own(400, 300, 20, -2, 35, 40.0, 100.0);
    static LegalScanRecorder.Target target(int id, int scanTurn) {
        return new LegalScanRecorder.Target(id, new LegalScanRecorder.Scan(scanTurn, 500, 300, 90, 8, 99.0));
    }
    static LegalScanRecorder.Intent intent(int turn) {
        return new LegalScanRecorder.Intent(turn, 8, 3, 15, -25, null, 0);
    }
    static LegalScanRecorder.Shot shot(int turn) {
        return new LegalScanRecorder.Shot(turn, 7, 0.5, 35, 400, 300);
    }
    JsonObject status(Path path) throws Exception {
        return JsonParser.parseString(Files.readString(Path.of(path + ".status.json"))).getAsJsonObject();
    }
    List<JsonObject> rows(Path path) throws Exception {
        return Files.readAllLines(path).stream().map(line -> JsonParser.parseString(line).getAsJsonObject()).toList();
    }

    @Test void disabledHasNoPathAndExplicitCaptureRefusesStaleFiles() throws Exception {
        assertFalse(LegalScanRecorder.fromEnvironment(Map.of()).enabled());
        assertThrows(IllegalArgumentException.class, () -> LegalScanRecorder.fromEnvironment(Map.of("AISURVIVOR_DATASET_PATH", temp.resolve("missing-id").toString())));
        Path path = temp.resolve("existing.jsonl"); Files.writeString(path, "preserve");
        assertThrows(Exception.class, () -> new LegalScanRecorder(path, "game-1", 100));
        assertEquals("preserve", Files.readString(path));
    }
    @Test void buffersConsistentPreDecisionSnapshotsAndOnlyConfirmedShots() throws Exception {
        Path path = temp.resolve("events.jsonl"); var recorder = new LegalScanRecorder(path, "game-1", 100);
        recorder.startRound(1); recorder.submitted(intent(0));
        recorder.frame(1, 1, OWN, List.of(target(1, 1), target(2, 1)));
        recorder.confirmed(shot(2)); recorder.confirmed(shot(2)); // Same event is not another shot.
        recorder.submitted(intent(1)); recorder.frame(1, 2, OWN, List.of(target(1, 1), target(2, 2)));
        assertEquals("", Files.readString(path), "No file append on the decision path");
        recorder.finishRound(1); recorder.finishRound(1); recorder.finishGame();
        var rows = rows(path); assertEquals(4, rows.size());
        assertEquals(rows.get(0).get("own"), rows.get(1).get("own"));
        assertEquals(rows.get(2).get("confirmedShots"), rows.get(3).get("confirmedShots"));
        assertEquals(0, rows.get(0).getAsJsonArray("confirmedShots").size());
        assertEquals(1, rows.get(2).getAsJsonArray("confirmedShots").size());
        assertTrue(rows.get(2).getAsJsonObject("previousIntent").get("submittedTurn").getAsInt() < 2);
        assertEquals("complete", status(path).get("status").getAsString());
        Path fixture = Path.of("tools/fixtures/legal-recorder/synthetic-events.jsonl");
        if ("1".equals(System.getenv("CAPTURE_RECORDER_FIXTURE"))) {
            Files.createDirectories(fixture.getParent());
            Files.copy(path, fixture); // Never overwrite an existing recorded producer fixture.
        }
        assertEquals(Files.readString(fixture).replace("\r\n", "\n"), Files.readString(path));
    }
    @Test void resetsAllHistoryAndBulletIdentityAtRoundBoundary() throws Exception {
        Path path = temp.resolve("rounds.jsonl"); var recorder = new LegalScanRecorder(path, "game-2", 100);
        recorder.startRound(1); recorder.confirmed(shot(1)); recorder.frame(1, 1, OWN, List.of(target(1, 1))); recorder.finishRound(1);
        recorder.startRound(2); recorder.frame(2, 1, OWN, List.of(target(1, 1))); recorder.finishRound(2); recorder.finishGame();
        var next = rows(path).get(1); assertTrue(next.get("previousIntent").isJsonNull());
        assertEquals(0, next.getAsJsonArray("confirmedShots").size());
        assertEquals(2, status(path).getAsJsonArray("rounds").size());
    }
    @Test void rejectsFutureIntentScanAndBulletWithoutInfluencingCommands() throws Exception {
        for (int type = 0; type < 3; type++) {
            Path path = temp.resolve("future-" + type); var recorder = new LegalScanRecorder(path, "game-3", 100);
            recorder.startRound(1);
            if (type == 0) recorder.submitted(intent(1));
            if (type == 2) recorder.confirmed(shot(2));
            recorder.frame(1, 1, OWN, List.of(target(1, type == 1 ? 2 : 1)));
            assertFalse(recorder.enabled()); assertEquals("", Files.readString(path));
            assertEquals("recording", status(path).get("status").getAsString());
            recorder.finishRound(1);
            assertEquals("failed", status(path).get("status").getAsString());
        }
    }
    @Test void rowBoundPreservesPartialEvidenceAndMakesCaptureInvalid() throws Exception {
        Path path = temp.resolve("bounded"); var recorder = new LegalScanRecorder(path, "game-4", 1);
        recorder.startRound(1); recorder.frame(1, 1, OWN, List.of(target(1, 1)));
        recorder.frame(1, 2, OWN, List.of(target(1, 2)));
        assertFalse(recorder.enabled()); assertEquals("", Files.readString(path));
        recorder.finishRound(1); assertEquals(1, rows(path).size());
        assertEquals("failed", status(path).get("status").getAsString());
    }
    @Test void absentRoundEndIsInvalidRatherThanSilentlyDiscardingRows() throws Exception {
        Path path = temp.resolve("incomplete"); var recorder = new LegalScanRecorder(path, "game-5", 100);
        recorder.startRound(1); recorder.frame(1, 1, OWN, List.of(target(1, 1))); recorder.finishGame();
        assertFalse(recorder.enabled()); assertEquals(1, rows(path).size());
        assertEquals("failed", status(path).get("status").getAsString());
    }
    @Test void enabledPreDecisionHookPreservesAcceptedCommandsAndState() throws Exception {
        var scenarios = TargetingCharacterizationTest.scenarios();
        for (int index = 0; index < 32; index++) {
            Path path = temp.resolve("neutral-" + index + ".jsonl");
            var recorder = new LegalScanRecorder(path, "neutral-" + index, 100);
            recorder.startRound(1);
            assertEquals(TargetingCharacterizationTest.record(scenarios.get(index)),
                    TargetingCharacterizationTest.record(scenarios.get(index), recorder));
            recorder.finishRound(1); recorder.finishGame();
            assertEquals("complete", status(path).get("status").getAsString());
        }
    }
}
