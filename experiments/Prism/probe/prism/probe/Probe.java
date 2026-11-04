package prism.probe;

import dev.robocode.tankroyale.botapi.Bot;
import dev.robocode.tankroyale.botapi.BotInfo;
import dev.robocode.tankroyale.botapi.BotState;
import dev.robocode.tankroyale.botapi.BulletState;
import dev.robocode.tankroyale.botapi.events.*;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Locale;

/**
 * Engine-contract probe (not a gameplay bot). Role "mover" executes a fixed command script that
 * exercises acceleration, instant stop, reversal from high and low speed, maximum turns at several
 * speeds, wall and corner contact, and one shot. Role "shooter" stays still, locks radar, and fires
 * low-power shots while requesting a simultaneous gun rotation. Both log every tick's own state, the
 * exact command sent after it, and every delivered event as JSONL to $PRISM_PROBE_DIR/<role>.jsonl.
 * The recorded files are the evidence; tests replay them through Prism's predictor.
 */
public final class Probe extends Bot {
    private final String role;
    private final Path out;
    private TickEvent tick;
    private int phaseTurn;
    private int phase;
    private int wallTurns;
    private boolean fired;

    Probe(String role) {
        super(BotInfo.fromFile(role.equals("mover") ? "PrismProbeMover.json" : "PrismProbeShooter.json"));
        this.role = role;
        String dir = System.getenv("PRISM_PROBE_DIR");
        this.out = Path.of(dir == null ? "." : dir, role + ".jsonl");
    }

    public static void main(String[] args) { new Probe(args.length > 0 ? args[0] : "mover").start(); }

    @Override public void onTick(TickEvent e) { tick = e; }

    @Override
    public void run() {
        setAdjustGunForBodyTurn(true);
        setAdjustRadarForGunTurn(true);
        setAdjustRadarForBodyTurn(true);
        int last = -1;
        while (isRunning()) {
            TickEvent t = tick;
            if (t != null && t.getTurnNumber() != last) {
                last = t.getTurnNumber();
                double[] cmd = role.equals("mover") ? mover(t) : shooter(t);
                setTargetSpeed(cmd[0]);
                setTurnRate(cmd[1]);
                setGunTurnRate(cmd[2]);
                setRadarTurnRate(cmd[3]);
                boolean accepted = false;
                if (cmd[4] > 0) accepted = setFire(cmd[4]);
                else if (getGunHeat() == 0) setFire(0);
                log(t, cmd, accepted);
            }
            go();
        }
    }

    private static final double[][] SCRIPT = {
            // {turns, targetSpeed, turnRateDeg}
            {10, 8, 0}, {3, 0, 0}, {10, 8, 0}, {6, -8, 0}, {4, -8, 0}, {2, 1, 0}, {3, 8, 0}, {4, -8, 0},
            {5, 8, 10}, {5, 8, -10}, {4, 3, 10}, {3, 0, 10}, {3, -5, -7}, {2, 0.5, 0}, {3, 2, 0}, {2, -1, 0},
    };

    private double[] mover(TickEvent t) {
        BotState s = t.getBotState();
        double gun = 0, radar = 45, fire = 0;
        if (!fired && s.getGunHeat() == 0) { fire = 1.0; fired = true; }
        if (phase < SCRIPT.length) {
            double[] p = SCRIPT[phase];
            if (++phaseTurn >= p[0]) { phase++; phaseTurn = 0; }
            return new double[] {p[1], p[2], gun, radar, fire};
        }
        // wall phase: drive straight at the nearest wall, keep pushing a few turns, then a corner
        int w = getArenaWidth(), h = getArenaHeight();
        double x = s.getX(), y = s.getY();
        if (phase == SCRIPT.length) {
            double target = nearestWallHeading(x, y, w, h);
            double turn = clampDeg(norm(target - s.getDirection()), 10);
            for (BotEvent e : t.getEvents()) if (e instanceof HitWallEvent && wallTurns == 0) wallTurns = 1;
            if (wallTurns > 0 && ++wallTurns > 4) { phase++; wallTurns = 0; }
            return new double[] {Math.abs(turn) < 1 ? 8 : 2, turn, gun, radar, fire};
        }
        if (phase == SCRIPT.length + 1) { // back off
            if (++phaseTurn > 12) { phase++; phaseTurn = 0; }
            return new double[] {-6, 0, gun, radar, fire};
        }
        if (phase == SCRIPT.length + 2) { // go to a staging point 100 units diagonally inside the nearest corner
            double cx = x < w / 2.0 ? 18 : w - 18, cy = y < h / 2.0 ? 18 : h - 18;
            double px = cx + (cx < w / 2.0 ? 100 : -100), py = cy + (cy < h / 2.0 ? 100 : -100);
            double d = Math.hypot(px - x, py - y);
            double target = Math.toDegrees(Math.atan2(py - y, px - x));
            double turn = clampDeg(norm(target - s.getDirection()), 10);
            if (d < 3) { phase++; return new double[] {0, 0, gun, radar, fire}; }
            return new double[] {Math.abs(turn) < 5 ? Math.min(8, Math.max(1, d / 3)) : 0, turn, gun, radar, fire};
        }
        if (phase == SCRIPT.length + 3) { // align exactly to the 45 degree diagonal, stopped
            double cx = x < w / 2.0 ? 0 : w, cy = y < h / 2.0 ? 0 : h;
            double target = Math.toDegrees(Math.atan2(cy - y, cx - x));
            double turn = clampDeg(norm(target - s.getDirection()), 10);
            if (Math.abs(turn) < 1e-9) phase++;
            return new double[] {0, turn, gun, radar, fire};
        }
        if (phase == SCRIPT.length + 4) { // drive into the corner and keep pushing
            for (BotEvent e : t.getEvents()) if (e instanceof HitWallEvent && wallTurns == 0) wallTurns = 1;
            if (wallTurns > 0 && ++wallTurns > 6) phase++;
            return new double[] {8, 0, gun, radar, fire};
        }
        return new double[] {0, 0, gun, radar, fire};
    }

    private double[] shooter(TickEvent t) {
        BotState s = t.getBotState();
        double radar = 45, gun = 15, fire = 0;
        ScannedBotEvent scan = null;
        for (BotEvent e : t.getEvents()) if (e instanceof ScannedBotEvent sc) scan = sc;
        if (scan != null) {
            double b = Math.toDegrees(Math.atan2(scan.getY() - s.getY(), scan.getX() - s.getX()));
            double need = norm(b - s.getRadarDirection());
            radar = clampDeg(need + Math.signum(need) * 20, 45);
            // alternate: aim at the target, or deliberately request a 15 degree rotation while firing
            gun = clampDeg(norm(b - s.getGunDirection()), 20);
        }
        if (s.getGunHeat() == 0 && t.getTurnNumber() > 40) { fire = t.getTurnNumber() > 420 ? 3.0 : 0.3; gun = 15; }
        return new double[] {0, 0, gun, radar, fire};
    }

    private static double nearestWallHeading(double x, double y, int w, int h) {
        double[] d = {x, w - x, y, h - y};
        double[] hd = {180, 0, 270, 90};
        int k = 0;
        for (int i = 1; i < 4; i++) if (d[i] < d[k]) k = i;
        return hd[k];
    }

    private static double norm(double a) { a = (a + 180) % 360; if (a < 0) a += 360; return a - 180; }
    private static double clampDeg(double a, double lim) { return Math.max(-lim, Math.min(lim, a)); }

    private void log(TickEvent t, double[] cmd, boolean accepted) {
        BotState s = t.getBotState();
        StringBuilder b = new StringBuilder();
        b.append(String.format(Locale.ROOT,
                "{\"round\":%d,\"turn\":%d,\"x\":%.17g,\"y\":%.17g,\"dir\":%.17g,\"gunDir\":%.17g,\"radarDir\":%.17g,\"speed\":%.17g,\"energy\":%.17g,\"gunHeat\":%.17g,\"enemies\":%d,"
                        + "\"cmd\":{\"targetSpeed\":%.17g,\"turnRate\":%.17g,\"gunTurnRate\":%.17g,\"radarTurnRate\":%.17g,\"fire\":%.17g,\"accepted\":%b},\"events\":[",
                t.getRoundNumber(), t.getTurnNumber(), s.getX(), s.getY(), s.getDirection(), s.getGunDirection(), s.getRadarDirection(),
                s.getSpeed(), s.getEnergy(), s.getGunHeat(), s.getEnemyCount(), cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], accepted));
        boolean first = true;
        for (BotEvent e : t.getEvents()) {
            String j = event(e);
            if (j == null) continue;
            if (!first) b.append(',');
            first = false;
            b.append(j);
        }
        b.append("]}\n");
        try {
            Files.writeString(out, b.toString(), StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException ex) { throw new RuntimeException(ex); }
    }

    private static String bullet(BulletState b) {
        return String.format(Locale.ROOT, "{\"id\":%d,\"owner\":%d,\"power\":%.17g,\"x\":%.17g,\"y\":%.17g,\"dir\":%.17g}",
                b.getBulletId(), b.getOwnerId(), b.getPower(), b.getX(), b.getY(), b.getDirection());
    }

    private static String event(BotEvent e) {
        int n = e.getTurnNumber();
        if (e instanceof ScannedBotEvent s) return String.format(Locale.ROOT,
                "{\"type\":\"scan\",\"turn\":%d,\"id\":%d,\"energy\":%.17g,\"x\":%.17g,\"y\":%.17g,\"dir\":%.17g,\"speed\":%.17g}",
                n, s.getScannedBotId(), s.getEnergy(), s.getX(), s.getY(), s.getDirection(), s.getSpeed());
        if (e instanceof BulletFiredEvent f) return "{\"type\":\"fired\",\"turn\":" + n + ",\"bullet\":" + bullet(f.getBullet()) + "}";
        if (e instanceof BulletHitBotEvent h) return String.format(Locale.ROOT, "{\"type\":\"bulletHit\",\"turn\":%d,\"victim\":%d,\"damage\":%.17g,\"energy\":%.17g,\"bullet\":%s}",
                n, h.getVictimId(), h.getDamage(), h.getEnergy(), bullet(h.getBullet()));
        if (e instanceof HitByBulletEvent h) return String.format(Locale.ROOT, "{\"type\":\"hitByBullet\",\"turn\":%d,\"damage\":%.17g,\"energy\":%.17g,\"bullet\":%s}",
                n, h.getDamage(), h.getEnergy(), bullet(h.getBullet()));
        if (e instanceof BulletHitWallEvent w) return "{\"type\":\"bulletWall\",\"turn\":" + n + ",\"bullet\":" + bullet(w.getBullet()) + "}";
        if (e instanceof BulletHitBulletEvent h) return "{\"type\":\"bulletBullet\",\"turn\":" + n + "}";
        if (e instanceof HitWallEvent) return "{\"type\":\"hitWall\",\"turn\":" + n + "}";
        if (e instanceof HitBotEvent h) return "{\"type\":\"hitBot\",\"turn\":" + n + ",\"victim\":" + h.getVictimId() + "}";
        if (e instanceof BotDeathEvent d) return "{\"type\":\"botDeath\",\"turn\":" + n + ",\"victim\":" + d.getVictimId() + "}";
        if (e instanceof DeathEvent) return "{\"type\":\"death\",\"turn\":" + n + "}";
        if (e instanceof WonRoundEvent) return "{\"type\":\"won\",\"turn\":" + n + "}";
        return "{\"type\":\"" + e.getClass().getSimpleName() + "\",\"turn\":" + n + "}";
    }
}
