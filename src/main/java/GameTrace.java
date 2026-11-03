import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import dev.robocode.tankroyale.botapi.BulletState;
import dev.robocode.tankroyale.botapi.events.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.zip.GZIPOutputStream;

/**
 * Optional exact replay trace of one bot process: a header, then in arrival order every turn-state snapshot ("tick"),
 * every event the brain receives ("ev"), and every command batch ("begin" and one "cmd" per turn) with each
 * intent-setter call and argument in order. Enabled only by {@code AISURVIVOR_TRACE_PATH}; a {@code .gz} suffix
 * compresses it. It observes only. {@link TraceReplay} reads it back.
 */
final class GameTrace {
    static final int VERSION = 1;
    private static final Gson JSON = new GsonBuilder().serializeSpecialFloatingPointValues().create();
    private static final GameTrace OFF = new GameTrace(null);

    private final Writer out;
    private JsonArray pending = new JsonArray();

    private GameTrace(Writer out) { this.out = out; }

    static GameTrace fromEnvironment(Map<String, String> env, BotConfig config) {
        String value = env.get("AISURVIVOR_TRACE_PATH");
        if (value == null) return OFF;
        if (value.isBlank()) throw new IllegalArgumentException("AISURVIVOR_TRACE_PATH must not be blank");
        try {
            return open(Path.of(value), config);
        } catch (IOException ex) {
            throw new UncheckedIOException("Cannot open trace " + value, ex);
        }
    }

    static GameTrace open(Path path, BotConfig config) throws IOException {
        Path absolute = path.toAbsolutePath();
        if (absolute.getParent() != null) Files.createDirectories(absolute.getParent());
        OutputStream stream = Files.newOutputStream(absolute);
        if (absolute.toString().endsWith(".gz")) stream = new GZIPOutputStream(stream);
        GameTrace trace = new GameTrace(new BufferedWriter(new OutputStreamWriter(stream, StandardCharsets.UTF_8)));
        JsonObject header = new JsonObject();
        header.addProperty("k", "header");
        header.addProperty("version", VERSION);
        header.addProperty("mode", config.mode().name());
        header.addProperty("seed", config.seed());
        header.addProperty("fixedAction", config.fixedAction());
        header.addProperty("duelPolicy", config.duelPolicy().name());
        trace.write(header);
        return trace;
    }

    boolean enabled() { return out != null; }

    void tick(BotPort p) {
        if (!enabled()) return;
        JsonObject o = new JsonObject();
        o.addProperty("k", "tick");
        o.addProperty("turn", p.turnNumber());
        o.addProperty("round", p.roundNumber());
        o.addProperty("enemies", p.enemyCount());
        o.addProperty("w", p.arenaWidth());
        o.addProperty("h", p.arenaHeight());
        o.addProperty("x", p.x());
        o.addProperty("y", p.y());
        o.addProperty("dir", p.direction());
        o.addProperty("speed", p.speed());
        o.addProperty("energy", p.energy());
        o.addProperty("gunHeat", p.gunHeat());
        o.addProperty("gunDir", p.gunDirection());
        o.addProperty("radarDir", p.radarDirection());
        o.addProperty("turnRate", p.turnRate());
        o.addProperty("targetSpeed", p.targetSpeed());
        o.addProperty("gunTurnRate", p.gunTurnRate());
        o.addProperty("radarTurnRate", p.radarTurnRate());
        write(o);
    }

    void event(BotEvent e) {
        if (!enabled()) return;
        JsonObject o = new JsonObject();
        o.addProperty("k", "ev");
        o.addProperty("type", e.getClass().getSimpleName());
        o.addProperty("turn", e.getTurnNumber());
        if (e instanceof ScannedBotEvent s) {
            o.addProperty("scannedBy", s.getScannedByBotId());
            o.addProperty("id", s.getScannedBotId());
            o.addProperty("energy", s.getEnergy());
            o.addProperty("x", s.getX());
            o.addProperty("y", s.getY());
            o.addProperty("dir", s.getDirection());
            o.addProperty("speed", s.getSpeed());
        } else if (e instanceof HitByBulletEvent h) {
            o.add("bullet", bullet(h.getBullet()));
            o.addProperty("damage", h.getDamage());
            o.addProperty("energy", h.getEnergy());
        } else if (e instanceof BulletHitBotEvent h) {
            o.addProperty("victim", h.getVictimId());
            o.add("bullet", bullet(h.getBullet()));
            o.addProperty("damage", h.getDamage());
            o.addProperty("energy", h.getEnergy());
        } else if (e instanceof BulletFiredEvent f) {
            o.add("bullet", bullet(f.getBullet()));
        } else if (e instanceof BotDeathEvent d) {
            o.addProperty("victim", d.getVictimId());
        } else if (e instanceof HitBotEvent h) {
            o.addProperty("victim", h.getVictimId());
            o.addProperty("energy", h.getEnergy());
            o.addProperty("x", h.getX());
            o.addProperty("y", h.getY());
            o.addProperty("rammed", h.isRammed());
        }
        write(o);
    }

    /** RoundStarted and GameEnded are not BotEvents in the 1.4.0 API; record their identity separately. */
    void event(RoundStartedEvent e) {
        if (!enabled()) return;
        JsonObject o = new JsonObject();
        o.addProperty("k", "ev");
        o.addProperty("type", "RoundStartedEvent");
        o.addProperty("round", e.getRoundNumber());
        write(o);
    }

    void event(RoundEndedEvent e) {
        if (!enabled()) return;
        JsonObject o = new JsonObject();
        o.addProperty("k", "ev");
        o.addProperty("type", "RoundEndedEvent");
        o.addProperty("round", e.getRoundNumber());
        o.addProperty("turn", e.getTurnNumber());
        write(o);
    }

    void event(GameEndedEvent e) {
        if (!enabled()) return;
        JsonObject o = new JsonObject();
        o.addProperty("k", "ev");
        o.addProperty("type", "GameEndedEvent");
        o.addProperty("rounds", e.getNumberOfRounds());
        write(o);
    }

    void call(String name, double value) {
        if (!enabled()) return;
        JsonArray c = new JsonArray();
        c.add(name);
        c.add(value);
        pending.add(c);
    }

    void flag(String name, boolean value) {
        if (!enabled()) return;
        JsonArray c = new JsonArray();
        c.add(name);
        c.add(value);
        pending.add(c);
    }

    void fire(double power, boolean accepted) {
        if (!enabled()) return;
        JsonArray c = new JsonArray();
        c.add("fire");
        c.add(power);
        c.add(accepted);
        pending.add(c);
    }

    /** Writes the intent calls made since the previous batch, then starts a new batch. */
    void commands(String kind) {
        if (!enabled()) return;
        JsonObject o = new JsonObject();
        o.addProperty("k", kind);
        o.add("calls", pending);
        pending = new JsonArray();
        write(o);
    }

    void close() {
        if (!enabled()) return;
        try { out.close(); } catch (IOException ex) { System.out.println("[AISurvivor] Trace close failed: " + ex.getMessage()); }
    }

    private static JsonObject bullet(BulletState b) {
        JsonObject o = new JsonObject();
        o.addProperty("id", b.getBulletId());
        o.addProperty("owner", b.getOwnerId());
        o.addProperty("power", b.getPower());
        o.addProperty("x", b.getX());
        o.addProperty("y", b.getY());
        o.addProperty("dir", b.getDirection());
        return o;
    }

    private void write(JsonObject o) {
        try {
            out.write(JSON.toJson(o));
            out.write('\n');
        } catch (IOException ex) {
            throw new UncheckedIOException("Trace write failed", ex);
        }
    }
}
