import dev.robocode.tankroyale.runner.*;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;

/**
 * Runs two bot directories in one isolated embedded-server 1v1 battle through the official
 * Tank Royale 1.4.0 Battle Runner (ephemeral port; the interactive server on 7654 is never
 * contacted). Used for the engine-contract probe trace and the passive lifecycle smoke; it is not
 * a measurement harness. PRISM_ROUNDS sets the round count (default 1).
 * Usage: java -Dfile.encoding=UTF-8 -cp runner.jar PrismTraceBattle.java botDirA botDirB results.json
 */
public class PrismTraceBattle {
    public static void main(String[] args) throws Exception {
        var bots = List.of(BotEntry.of(args[0]), BotEntry.of(args[1]));
        try (var runner = BattleRunner.create(b -> b.embeddedServer().suppressServerOutput().botConnectTimeout(Duration.ofSeconds(60)))) {
            var setup = BattleSetup.oneVsOne(s -> { s.setNumberOfRounds(Integer.parseInt(System.getenv().getOrDefault("PRISM_ROUNDS","1"))); s.setTurnTimeoutMicros(100000); });
            try (var handle = runner.startBattleAsync(setup, bots)) {
                var ex = Executors.newSingleThreadExecutor();
                try {
                    var results = ex.submit(handle::awaitResults).get(600, TimeUnit.SECONDS);
                    var rows = new ArrayList<String>();
                    for (var r : results.getResults()) rows.add("{\"name\":\"" + r.getName() + "\",\"rank\":" + r.getRank() + ",\"totalScore\":" + r.getTotalScore() + "}");
                    Files.writeString(Path.of(args[2]), "{\"numberOfRounds\":" + results.getNumberOfRounds() + ",\"results\":[" + String.join(",", rows) + "]}\n");
                } finally { ex.shutdownNow(); }
            }
        }
    }
}
