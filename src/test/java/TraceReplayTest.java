import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * The committed golden traces were recorded from real local games of the default bot. Any change that claims
 * behavioral parity must replay them identically; any decision change makes this test fail with the first mismatch.
 */
class TraceReplayTest {
    static final List<String> GOLDENS = List.of("retcon-1v1-3r", "melee8-3r", "walls-1v1-3r");

    static Path golden(String name) { return Path.of("benchmarks/replay-goldens/" + name + ".jsonl.gz"); }

    @Test void goldenTracesReplayIdentically(@TempDir Path temp) throws IOException {
        for (String name : GOLDENS) {
            TraceReplay.Result result = TraceReplay.replay(golden(name), temp.resolve(name + ".dat"));
            assertTrue(result.identical(), name + ": " + result.mismatch());
            assertTrue(result.batches() > 500, name + " replayed only " + result.batches() + " batches");
        }
    }

    @Test void coherentMeleeChangesMeleeDecisionsOnlyWhereIntended(@TempDir Path temp) throws IOException {
        // The pre-change melee golden must diverge at its first DQN-selected melee movement; 1v1 goldens stay identical.
        TraceReplay.Result old = TraceReplay.replay(
                Path.of("benchmarks/replay-goldens/superseded/melee8-3r-before-coherent-melee.jsonl.gz"), temp.resolve("o.dat"));
        assertFalse(old.identical());
        assertTrue(old.mismatch().startsWith("line 15 turn 3 round 1:"), old.mismatch());
    }

    @Test void alteredRecordedCommandIsDetected(@TempDir Path temp) throws IOException {
        List<String> lines = read(golden("retcon-1v1-3r"));
        int index = -1;
        for (int i = 0; i < lines.size(); i++)
            if (lines.get(i).startsWith("{\"k\":\"cmd\"") && lines.get(i).contains("\"turnRate\"") && i > 200) { index = i; break; }
        assertTrue(index > 0);
        lines.set(index, lines.get(index).replaceFirst("\\[\"turnRate\",(-?[0-9.Ee+-]+)\\]",
                "[\"turnRate\",123.456]"));
        Path altered = write(temp.resolve("altered.jsonl.gz"), lines);
        TraceReplay.Result result = TraceReplay.replay(altered, temp.resolve("m.dat"));
        assertFalse(result.identical());
        assertTrue(result.mismatch().contains("123.456"), result.mismatch());
    }

    @Test void differentSeedChangesMeleeDecisions(@TempDir Path temp) throws IOException {
        // Melee uses seeded DQN Boltzmann sampling, so a different seed must be caught as a behavior change.
        List<String> lines = read(golden("melee8-3r"));
        lines.set(0, lines.get(0).replaceFirst("\"seed\":-?[0-9]+", "\"seed\":999"));
        Path reseeded = write(temp.resolve("reseeded.jsonl.gz"), lines);
        assertFalse(TraceReplay.replay(reseeded, temp.resolve("m.dat")).identical());
    }

    @Test void tracingAddsNoDecisionCalls() throws IOException {
        // The 'begin' batch must contain exactly the two run-start adjustments the brain makes.
        List<String> lines = read(golden("walls-1v1-3r"));
        String begin = lines.stream().filter(l -> l.startsWith("{\"k\":\"begin\"")).findFirst().orElseThrow();
        assertEquals("{\"k\":\"begin\",\"calls\":[[\"adjustGunForBodyTurn\",true],[\"adjustRadarForGunTurn\",true]]}", begin);
    }

    private static List<String> read(Path path) throws IOException {
        try (var reader = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(path)),
                StandardCharsets.UTF_8))) {
            return reader.lines().collect(java.util.stream.Collectors.toList());
        }
    }

    private static Path write(Path path, List<String> lines) throws IOException {
        try (var writer = new BufferedWriter(new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(path)),
                StandardCharsets.UTF_8))) {
            for (String line : lines) { writer.write(line); writer.write('\n'); }
        }
        return path;
    }
}
