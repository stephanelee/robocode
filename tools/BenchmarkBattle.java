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
                    var rows = new ArrayList<String>();
                    for (var r : results.getResults()) rows.add(String.format(Locale.ROOT,
                            "{\"name\":%s,\"rank\":%d,\"totalScore\":%d,\"survival\":%d,\"bulletDamage\":%d,\"firstPlaces\":%d}",
                            quote(r.getName()), r.getRank(), r.getTotalScore(), r.getSurvival(), r.getBulletDamage(), r.getFirstPlaces()));
                    Files.writeString(Path.of(args[2]), "{\"numberOfRounds\":" + results.getNumberOfRounds()
                            + ",\"results\":[" + String.join(",", rows) + "]}\n");
                } catch (TimeoutException e) {
                    handle.stop();
                    throw new IllegalStateException("Battle exceeded " + timeout + " seconds", e);
                } finally { executor.shutdownNow(); }
            }
        }
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
