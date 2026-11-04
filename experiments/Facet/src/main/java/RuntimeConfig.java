import java.util.*;
/** Immutable independent-v1 profile. Reject unsupported edits rather than silently ignore them. */
final class RuntimeConfig {
  static final long SEED = 49667;
  static final Map<String, String> PINNED =
      Map.ofEntries(Map.entry("config.schema", "2"), Map.entry("model.schema", "2"),
          Map.entry("feature.schema", "1"), Map.entry("profile", "independent-v1"),
          Map.entry("api", "1.4.0"), Map.entry("motion.contract", "official-1.4.0-fixture-replay"),
          Map.entry("melee.enabled", "true"), Map.entry("cross.game.load", "false"),
          Map.entry("seed", Long.toString(SEED)), Map.entry("wave.cap.per.enemy", "256"),
          Map.entry("history.cap", "512"), Map.entry("neighbor.payload.cap.bytes", "134217728"),
          Map.entry("predictor.steps.per.turn", "2400"), Map.entry("trajectory.ticks", "160"));
  static void verify() {
    try (var in = RuntimeConfig.class.getResourceAsStream("/facet.properties")) {
      if (in == null)
        throw new IllegalStateException("missing runtime profile");
      Properties p = new Properties();
      p.load(in);
      if (!p.equals(PINNED))
        throw new IllegalStateException(
            "Unsupported runtime profile; semantic changes require a new schema/profile");
    } catch (java.io.IOException ex) {
      throw new IllegalStateException(ex);
    }
  }
}
