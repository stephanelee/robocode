import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;
import java.util.*;
/** Synthetic legal-event adapter fixture, never an engine or battle-strength oracle. */
public final class FacetTelemetryFixture {
  public static void main(String[] args) {
    Diagnostics d = new Diagnostics();
    Core core = new Core(new Physics.Rules(800, 600, .1), d, 49667);
    core.roundReset(1);
    Physics.Body own = new Physics.Body(new Physics.V(400, 300), 0, 0, 0, 0, 0);
    BulletState bullet = new BulletState(1, 0, 2, 400, 300, 0, null);
    for (int t = 1; t <= 12; t++) {
      List<BotEvent> events = new ArrayList<>();
      events.add(new ScannedBotEvent(t, 0, 1, t <= 5 ? 100 : 90, 600, 300, 0, 0));
      if (t == 3) {
        BulletFiredEvent fired = new BulletFiredEvent(t, bullet);
        events.add(fired);
        events.add(fired);
      }
      if (t == 5)
        events.add(new BulletHitBotEvent(t, 1, bullet, 10, 90));
      Core.Intent intent =
          core.decide(own, 100, t, 1, events, t == 3 || t == 4 ? List.of(bullet) : List.of(), 0);
      if (t == 2) {
        if (!intent.fire())
          throw new AssertionError("prepared firing opportunity absent");
        core.fire.submitted(true, own, t, ++core.sequence);
      }
    }
    if (core.fire.confirmed != 1 || d.counters.getOrDefault("shot.hit", 0L) != 1)
      throw new AssertionError("physical counters");
    ModelSnapshot.optional(core);
    d.snapshot(1, core.fire, core.budget, core.waves, core.planner);
    d.close();
    System.out.println("PASS synthetic private telemetry/model fixture requested=1 confirmed=1 "
                       + "hits=1 duplicateCallbacks=1");
  }
}
