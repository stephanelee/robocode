import dev.robocode.tankroyale.runner.*;
import java.nio.file.*;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.*;

/** Observer only: one 1v1 game on an isolated embedded server; each bot runs in its own process. */
public class KeelPrismBattle {
    public static void main(String[] args) throws Exception {
        int rounds = Integer.parseInt(args[0]);
        Path output = Path.of(args[1]), markers = output.resolve("markers.txt"), prism = output.resolve("prism.jsonl");
        var setup = BattleSetup.oneVsOne(s -> { s.setNumberOfRounds(rounds); s.setTurnTimeoutMicros(100000); });
        try (var runner = BattleRunner.create(b -> b.embeddedServer().enableRecording(output).suppressServerOutput()
                .botConnectTimeout(Duration.ofSeconds(80)));
             var battle = runner.startBattleAsync(setup, List.of(BotEntry.of(args[2]), BotEntry.of(args[3])))) {
            var pool = Executors.newSingleThreadExecutor();
            try {
                var result = pool.submit(battle::awaitResults).get(3000, TimeUnit.SECONDS);
                if (result.getNumberOfRounds() != rounds) throw new IllegalStateException("Incomplete game");
                for (var row : result.getResults())
                    System.out.printf("%s %s %d %d%n", row.getName(), row.getVersion(), row.getTotalScore(), row.getFirstPlaces());
                long until = System.nanoTime() + TimeUnit.SECONDS.toNanos(15);
                while (!Files.isRegularFile(markers) || !Files.readAllLines(markers).contains(rounds + " end")
                        || !Files.isRegularFile(prism) || Files.readAllLines(prism).stream().noneMatch(s -> s.contains("\"event\":\"game\""))) {
                    if (System.nanoTime() > until) throw new IllegalStateException("Final callback did not flush");
                    Thread.sleep(25);
                }
            } finally { pool.shutdownNow(); }
        }
    }
}
