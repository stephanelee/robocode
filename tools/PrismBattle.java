import dev.robocode.tankroyale.runner.*;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;

/**
 * One isolated 1v1 game for the registered Prism screen through the official Tank Royale 1.4.0
 * Battle Runner with an embedded server on an ephemeral port (never the interactive 7654).
 * Usage: java -cp runner.jar tools/PrismBattle.java rounds timeoutSeconds results.json botDirA botDirB
 */
public class PrismBattle {
    public static void main(String[] args) throws Exception {
        int rounds = Integer.parseInt(args[0]);
        int timeout = Integer.parseInt(args[1]);
        var bots = List.of(BotEntry.of(args[3]), BotEntry.of(args[4]));
        try (var runner = BattleRunner.create(b -> b.embeddedServer().suppressServerOutput().botConnectTimeout(Duration.ofSeconds(90)))) {
            Runtime.getRuntime().addShutdownHook(new Thread(runner::close));
            var setup = BattleSetup.oneVsOne(s -> { s.setNumberOfRounds(rounds); s.setTurnTimeoutMicros(100000); });
            try (var handle = runner.startBattleAsync(setup, bots)) {
                var ex = Executors.newSingleThreadExecutor();
                try {
                    var results = ex.submit(handle::awaitResults).get(timeout, TimeUnit.SECONDS);
                    var rows = new ArrayList<String>();
                    for (var r : results.getResults()) rows.add(String.format(Locale.ROOT,
                            "{\"id\":%d,\"name\":%s,\"version\":%s,\"rank\":%d,\"totalScore\":%d,\"survival\":%d,\"bulletDamage\":%d,\"ramDamage\":%d,\"firstPlaces\":%d,\"secondPlaces\":%d}",
                            r.getId(), quote(r.getName()), quote(r.getVersion()), r.getRank(), r.getTotalScore(), r.getSurvival(),
                            r.getBulletDamage(), r.getRamDamage(), r.getFirstPlaces(), r.getSecondPlaces()));
                    Files.writeString(Path.of(args[2]), "{\"numberOfRounds\":" + results.getNumberOfRounds() + ",\"results\":[" + String.join(",", rows) + "]}\n");
                } catch (TimeoutException e) {
                    handle.stop();
                    throw new IllegalStateException("Battle exceeded " + timeout + " seconds", e);
                } finally { ex.shutdownNow(); }
            }
        }
    }

    private static String quote(String v) {
        var b = new StringBuilder("\"");
        for (char c : v.toCharArray()) { if (c == '"' || c == '\\') b.append('\\'); if (c >= 32) b.append(c); }
        return b.append('"').toString();
    }
}
