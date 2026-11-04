import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;
import java.util.List;

/** Run directly on Java 17 with only the releasable fat JAR on the classpath; no server connection. */
class ReleasedLifecycle {
    private static TickEvent tick(int round, int turn, BotEvent... events) {
        var own = new BotState(false, 100, 600, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                null, null, null, null, null, null, null, false);
        return new TickEvent(turn, round, own, List.of(), List.of(events));
    }
    public static void main(String[] args) {
        Breakwater bot = new Breakwater();
        bot.onGameStarted(new GameStartedEvent(1, null, new GameSetup("1v1", 800, 600, 2, .1, 450, 100000, 1000000)));
        bot.onRoundStarted(new RoundStartedEvent(1));
        bot.onTick(tick(1, 29, new ScannedBotEvent(29, 1, 2, 100, 180, 200, 0, 0)));
        bot.onTick(tick(1, 30, new ScannedBotEvent(30, 1, 2, 98.05, 180, 200, 0, 0)));
        double aim = Math.toRadians(2);
        var bullet = new BulletState(7, 2, 1.95, 180 + 28 * 14.15 * Math.cos(aim),
                200 + 28 * 14.15 * Math.sin(aim), 2, null);
        var impact = new HitByBulletEvent(57, bullet, 9.7, 90.3);
        bot.onTick(tick(1, 57, impact));
        bot.onRoundEnded(new RoundEndedEvent(1, 70, null));
        bot.onTick(tick(1, 71, impact));
        bot.onRoundEnded(new RoundEndedEvent(1, 71, null));
        bot.onRoundStarted(new RoundStartedEvent(2));
        bot.onTick(tick(1, 72, impact));
        bot.onTick(tick(2, 1));
        bot.onRoundEnded(new RoundEndedEvent(2, 5, null));
    }
}
