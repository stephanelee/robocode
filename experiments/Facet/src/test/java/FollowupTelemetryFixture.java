import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;
import java.util.*;
/** Synthetic legal callback chronology only; no engine launch or live acceptance. */
public final class FollowupTelemetryFixture {
  static ScannedBotEvent scan(int turn, int id) {
    return new ScannedBotEvent(turn, 99, id, 100, id == 1 ? 600 : 700, id == 1 ? 300 : 400, 0, 0);
  }
  static Physics.Body aligned(FireControl.Plan p) {
    return new Physics.Body(p.origin(), 0, 0, p.angle(), 0, 0);
  }
  static BulletState submit(Core c, Physics.Body own, int t, int id) {
    c.log.event("fire.submitted", 1, t, "accepted=true " + c.fire.plan.evidenceRow());
    c.fire.submitted(true, own, t, ++c.sequence);
    var p = c.fire.pending;
    return new BulletState(
        id, 99, p.power(), p.origin().x(), p.origin().y(), Math.toDegrees(p.heading()), null);
  }
  public static void main(String[] args) {
    var d = new Diagnostics();
    var c = new Core(new Physics.Rules(800, 600, .1), d, 49667);
    c.roundReset(1);
    for (int id = 1; id <= 2; id++) {
      var e = c.enemy(id);
      e.crossings = 16;
      e.selected = "recent";
      var sample = new Memory.Observation(
          new double[Learner.DIM], Memory.Label.DISPLACEMENT, 0, 1, id, 1, 0, 1, id, true);
      e.displacement.add(sample);
      for (var s : e.recentDisplacement) s.add(sample);
    }
    var own = new Physics.Body(new Physics.V(400, 300), 0, 0, 0, 0, 0);
    c.decide(own, 100, 0, 2, List.of(scan(0, 1), scan(0, 2)), List.of(), 99);
    own = aligned(c.fire.plan);
    if (!c.decide(own, 100, 1, 2, List.of(scan(1, 1), scan(1, 2)), List.of(), 99).fire())
      throw new AssertionError("melee submission");
    var bullet = submit(c, own, 1, 101);
    c.decide(own, 100, 2, 1,
        List.of(new BulletFiredEvent(2, bullet), new BotDeathEvent(2, 2), scan(2, 1)),
        List.of(bullet), 99);
    if (c.fire.plan == null || !c.fire.plan.evidence().contains("gun=recent"))
      throw new AssertionError("survivor selected path");
    own = aligned(c.fire.plan);
    if (!c.decide(own, 100, 3, 1, List.of(scan(3, 1)), List.of(bullet), 99).fire())
      throw new AssertionError("survivor submission");
    var second = submit(c, own, 3, 102);
    c.observe(own, 100, 4, 1, List.of(new BulletFiredEvent(4, second), scan(4, 1)),
        List.of(bullet, second), 99, true);
    Physics.endWork();
    if (c.fire.confirmed != 2 || c.fire.unmatched != 0)
      throw new AssertionError("confirmed selected plans");
    ModelSnapshot.roundHash(c);
    d.snapshot(1, c.fire, c.budget, c.waves, c.planner);
    d.close();
    System.out.println("PASS synthetic population2/1 confirmed selected plans; live transition "
        + "acceptance remains pending");
  }
}
