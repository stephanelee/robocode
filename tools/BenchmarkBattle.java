import dev.robocode.tankroyale.runner.*;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;

/** Thin command-line adapter for the official Battle Runner. */
public class BenchmarkBattle {
    public static void main(String[] args) throws Exception {
        if (args.length < 5) throw new IllegalArgumentException("rounds timeoutSeconds output.json botDir opponentDir...");
        int rounds = Integer.parseInt(args[0]);
        int timeout = Integer.parseInt(args[1]);
        var bots = new ArrayList<BotEntry>();
        for (int i = 3; i < args.length; i++) bots.add(BotEntry.of(args[i]));
        try (var runner = BattleRunner.create(b -> b.embeddedServer()
                .suppressServerOutput().botConnectTimeout(Duration.ofSeconds(45)))) {
            Runtime.getRuntime().addShutdownHook(new Thread(runner::close));
            var setup = bots.size() == 2
                    ? BattleSetup.oneVsOne(s -> { s.setNumberOfRounds(rounds); s.setTurnTimeoutMicros(100000); })
                    : BattleSetup.melee(s -> { s.setNumberOfRounds(rounds); s.setMinNumberOfParticipants(bots.size()); s.setMaxNumberOfParticipants(bots.size()); s.setTurnTimeoutMicros(100000); });
            try (var handle = runner.startBattleAsync(setup, bots)) {
                var executor = Executors.newSingleThreadExecutor();
                try {
                    var future = executor.submit(handle::awaitResults);
                    var results = future.get(timeout, TimeUnit.SECONDS);
                    awaitRetconStats(results, rounds);
                    var rows = new ArrayList<String>();
                    for (var r : results.getResults()) rows.add(String.format(Locale.ROOT,
                            "{\"id\":%d,\"name\":%s,\"version\":%s,\"rank\":%d,\"totalScore\":%d,\"survival\":%d,\"bulletDamage\":%d,\"firstPlaces\":%d}",
                            r.getId(), quote(r.getName()), quote(r.getVersion()), r.getRank(), r.getTotalScore(), r.getSurvival(), r.getBulletDamage(), r.getFirstPlaces()));
                    Files.writeString(Path.of(args[2]), "{\"numberOfRounds\":" + results.getNumberOfRounds()
                            + ",\"results\":[" + String.join(",", rows) + "]}\n");
                    awaitChallengerCompletion(rounds);
                    awaitLegalDataset();
                } catch (TimeoutException e) {
                    handle.stop();
                    throw new IllegalStateException("Battle exceeded " + timeout + " seconds", e);
                } finally { executor.shutdownNow(); }
            }
        }
    }
    private static void awaitChallengerCompletion(int rounds) throws Exception {
        String model = System.getenv("CHALLENGER_MODEL_PATH");
        String metrics = System.getenv("CHALLENGER_METRICS_PATH");
        if (model == null && metrics == null) return;
        if (model == null || model.isBlank() || metrics == null || metrics.isBlank())
            throw new IllegalArgumentException("Challenger completion requires both model and metrics paths");
        awaitChallengerCompletion(Path.of(model), Path.of(metrics), rounds, Duration.ofSeconds(3));
    }
    // Readiness only: the experiment's existing strict validators still check every field and row.
    static void awaitChallengerCompletion(Path model, Path metrics, int rounds, Duration timeout) throws Exception {
        if (!model.isAbsolute() || !metrics.isAbsolute() || model.normalize().equals(metrics.normalize()) || rounds < 1)
            throw new IllegalArgumentException("Challenger outputs must be distinct absolute paths and rounds positive");
        var coverage = java.util.regex.Pattern.compile("\"roundsCompleted\"\\s*:\\s*" + rounds + "\\s*[,}]");
        long deadline = System.nanoTime() + timeout.toNanos();
        while (System.nanoTime() < deadline) {
            if (Files.isRegularFile(model, LinkOption.NOFOLLOW_LINKS)
                    && Files.isRegularFile(metrics, LinkOption.NOFOLLOW_LINKS)) {
                if (Files.size(model) > 2_000_000 || Files.size(metrics) > 32_000_000)
                    throw new IllegalStateException("Challenger completion output exceeds size limit");
                // The model is replaced atomically; the final newline marks a fully appended row.
                String state = Files.readString(model), raw = Files.readString(metrics);
                var rows = raw.lines().filter(line -> !line.isBlank()).toList();
                if (coverage.matcher(state).find() && raw.endsWith("\n") && rows.size() == rounds
                        && coverage.matcher(rows.get(rows.size() - 1)).find()) return;
            }
            Thread.sleep(20);
        }
        throw new IllegalStateException("Challenger final model/metrics did not finish within bounded flush wait");
    }
    private static void awaitLegalDataset() throws Exception {
        String configured = System.getenv("AISURVIVOR_DATASET_PATH");
        if (configured == null || configured.isBlank()) return;
        Path status = Path.of(configured + ".status.json");
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while (System.nanoTime() < deadline) {
            if (Files.isRegularFile(status) && Files.size(status) < 1_000_000) {
                String json = Files.readString(status);
                if (json.contains("\"status\":\"complete\"")) return;
                if (json.contains("\"status\":\"failed\"")) throw new IllegalStateException("Legal dataset capture failed: " + json);
            }
            Thread.sleep(20);
        }
        throw new IllegalStateException("Legal dataset did not finish within bounded flush wait");
    }
    private static void awaitRetconStats(BattleResults results, int rounds) throws Exception {
        var configured = System.getenv("RETCON_STATS_DIR");
        if (configured == null || configured.isBlank()) return;
        var directory = Path.of(configured).toAbsolutePath().normalize();
        var temporary = Path.of(System.getProperty("java.io.tmpdir")).toAbsolutePath().normalize();
        if (!directory.startsWith(temporary) || directory.equals(temporary))
            throw new IllegalArgumentException("Retcon capture must be a Java-temp child");
        var opponent = results.getResults().stream().filter(r -> r.getName().equals("Retcon")).findFirst();
        if (opponent.isEmpty()) return;
        var filename = java.util.regex.Pattern.compile("Retcon-" + opponent.get().getId() + "--?[0-9]+\\.json");
        var coverage = java.util.regex.Pattern.compile("\"roundsRecorded\"\\s*:\\s*" + rounds + "\\s*[,}]");
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while (System.nanoTime() < deadline) {
            try (var files = Files.list(directory)) {
                for (var file : files.filter(p -> filename.matcher(p.getFileName().toString()).matches()).toList()) {
                    if (!Files.isSymbolicLink(file) && Files.isRegularFile(file) && Files.size(file) <= 2_000_000
                            && coverage.matcher(Files.readString(file)).find()) return;
                }
            }
            Thread.sleep(20);
        }
        System.err.println("Retcon final stats did not cover every round within the bounded flush wait");
    }
    private static String quote(String value) {
        var out = new StringBuilder("\"");
        for (char c : value.toCharArray()) {
            if (c == '\\' || c == '"') out.append('\\').append(c);
            else if (c < 32) out.append(String.format("\\u%04x", (int)c));
            else out.append(c);
        }
        return out.append('"').toString();
    }
}
