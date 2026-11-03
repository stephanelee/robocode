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
    /** Exact coverage of each complete golden game: command batches and completed rounds. */
    static final java.util.Map<String, int[]> COVERAGE = java.util.Map.of(
            "retcon-1v1-3r", new int[]{3044, 3}, "melee8-3r", new int[]{3007, 3}, "walls-1v1-3r", new int[]{939, 3});

    static Path golden(String name) { return Path.of("benchmarks/replay-goldens/" + name + ".jsonl.gz"); }

    @Test void goldenTracesReplayIdentically(@TempDir Path temp) throws IOException {
        for (String name : GOLDENS) {
            TraceReplay.Result result = TraceReplay.replay(golden(name), temp.resolve(name + ".dat"));
            assertTrue(result.identical(), name + ": " + result.mismatch());
            assertEquals(COVERAGE.get(name)[0], result.batches(), name + " batches");
            assertEquals(COVERAGE.get(name)[1], result.rounds(), name + " rounds");
        }
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

    @Test void signedZeroIsNotEqualToZero(@TempDir Path temp) throws IOException {
        // Bit-exact contract: a recorded 0.0 argument changed to -0.0 must be reported (numeric equality would hide it).
        for (String name : List.of("retcon-1v1-3r", "walls-1v1-3r")) {
            List<String> lines = read(golden(name));
            int index = -1;
            for (int i = 0; i < lines.size(); i++)
                if (lines.get(i).startsWith("{\"k\":\"cmd\"") && lines.get(i).contains(",0.0]")) { index = i; break; }
            assertTrue(index > 0, name + " has no zero-valued argument to mutate");
            lines.set(index, lines.get(index).replaceFirst(",0\\.0\\]", ",-0.0]"));
            TraceReplay.Result result = TraceReplay.replay(write(temp.resolve(name + "-z.jsonl.gz"), lines), temp.resolve("z.dat"));
            assertFalse(result.identical(), name + " accepted -0.0 for 0.0");
            assertTrue(result.mismatch().contains("-0.0"), result.mismatch());
        }
        assertFalse(TraceReplay.sameCalls(calls(0.0), calls(-0.0)));
        assertTrue(TraceReplay.sameCalls(calls(1.5), calls(1.5)));
    }

    @Test void incompleteTracesAreNotSuccessful(@TempDir Path temp) throws IOException {
        List<String> lines = read(golden("retcon-1v1-3r"));
        Path headerOnly = write(temp.resolve("header.jsonl.gz"), lines.subList(0, 1));
        TraceReplay.Result empty = TraceReplay.replay(headerOnly, temp.resolve("h.dat"));
        assertFalse(empty.identical());
        assertTrue(empty.mismatch().startsWith("incomplete trace"), empty.mismatch());
        Path truncated = write(temp.resolve("truncated.jsonl.gz"), lines.subList(0, 6000));
        TraceReplay.Result partial = TraceReplay.replay(truncated, temp.resolve("t.dat"));
        assertFalse(partial.identical(), "a truncated game must not count as a successful replay");
        assertTrue(partial.mismatch().startsWith("incomplete trace"), partial.mismatch());
    }

    private static com.google.gson.JsonArray calls(double value) {
        com.google.gson.JsonArray call = new com.google.gson.JsonArray();
        call.add("turnRate"); call.add(value);
        com.google.gson.JsonArray all = new com.google.gson.JsonArray();
        all.add(call);
        return all;
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
