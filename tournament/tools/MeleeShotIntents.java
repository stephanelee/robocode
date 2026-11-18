import com.google.gson.*;
import java.nio.file.Path;
import java.util.*;

/** Own legal shot contexts from the measured bot. No opponent-private observations. */
public final class MeleeShotIntents extends CrossfireReplay {
    record Context(int round, int fireTurn, int target, int scanTurn, double range, double power,
                   double actualRay, double desiredError, double x, double y,
                   double targetX, double targetY, double heading, double speed, double turnRate) {}
    final List<Context> shots = new ArrayList<>();
    MeleeShotIntents() { super(false); }

    @Override MeleeFire.Aim decide(int t, JsonObject state, Counts counts) {
        var shot = super.decide(t, state, counts);
        if (shot != null) {
            var o = shot.target();
            shots.add(new Context(round,t+1,o.id,o.turn,shot.distance(),shot.power(),number(state,"gunDirection"),
                    shot.error(),number(state,"x"),number(state,"y"),o.x,o.y,o.heading,o.speed,o.turnRate));
        }
        return shot;
    }

    public static void main(String[] args) throws Exception {
        var games = readGames(Path.of(args[0]), MeleeShotIntents::new);
        var output = new TreeMap<String,List<Context>>();
        games.forEach((id, replay) -> output.put(id,replay.shots));
        if (output.values().stream().mapToInt(List::size).sum() != 4042)
            throw new IllegalStateException("Incomplete request reconstruction");
        System.out.println(new Gson().toJson(output));
    }
}
