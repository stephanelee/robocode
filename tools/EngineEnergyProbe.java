import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;
import java.nio.file.*;
import java.io.*;
import java.util.Locale;

/** Stationary diagnostic probes, never used by AISurvivor or Retcon. */
public class EngineEnergyProbe extends Bot {
    private final String label;
    private final PrintWriter trace;
    private final PrintWriter startup;
    private double bearing;
    private boolean seen, fired;
    private boolean tickRecorded;

    EngineEnergyProbe(String label) throws IOException {
        super(BotInfo.fromFile(label + ".json"));
        this.label = label;
        Path output = Path.of(System.getenv("ENGINE_TRACE_DIR"));
        trace = new PrintWriter(Files.newBufferedWriter(output.resolve(label + ".jsonl")));
        startup = new PrintWriter(Files.newBufferedWriter(output.resolve(label + "-startup.jsonl")), true);
        startup("processStarted", "");
    }
    public static void main(String[] args) throws Exception { new EngineEnergyProbe(args[0]).start(); }
    private synchronized void startup(String phase, String fields) {
        startup.printf(Locale.ROOT, "{\"phase\":\"%s\",\"timeMillis\":%d%s}%n",
                phase, System.currentTimeMillis(), fields);
    }
    @Override public void onConnected(ConnectedEvent e) { startup("connected", ""); }
    @Override public void onDisconnected(DisconnectedEvent e) { startup("disconnected", ""); }
    @Override public void onGameStarted(GameStartedEvent e) {
        startup("gameStarted", ",\"id\":" + e.getMyId());
    }
    @Override public void onRoundStarted(RoundStartedEvent e) {
        startup("roundStarted", ",\"round\":" + e.getRoundNumber());
    }
    @Override public void onTick(TickEvent e) {
        if (!tickRecorded) {
            tickRecorded = true;
            startup("firstTick", ",\"round\":" + e.getRoundNumber() + ",\"turn\":" + e.getTurnNumber());
        }
    }
    private static double normal(double angle) { return (angle % 360 + 540) % 360 - 180; }
    private void record(String type, int turn, String fields) {
        trace.printf(Locale.ROOT, "{\"type\":\"%s\",\"label\":\"%s\",\"turn\":%d,\"self\":%d,\"selfEnergy\":%.12f%s}%n",
                type, label, turn, getMyId(), getEnergy(), fields);
        trace.flush();
    }
    @Override public void run() {
        setAdjustGunForBodyTurn(true);
        setAdjustRadarForGunTurn(true);
        setAdjustRadarForBodyTurn(true);
        setFireAssist(false);
        while (isRunning()) {
            setTargetSpeed(0);
            setTurnRate(0);
            setGunTurnRate(seen ? normal(bearing - getGunDirection()) : 0);
            // Cross the stationary target every turn once acquired; no movement or damage inference.
            setRadarTurnRate(seen ? normal(bearing + (getTurnNumber() % 2 == 0 ? 2 : -2) - getRadarDirection()) : 45);
            int shotTurn = label.equals("Alpha") ? 50 : 200;
            if (seen && !fired && getTurnNumber() >= shotTurn && Math.abs(normal(bearing - getGunDirection())) < 0.001) {
                fired = setFire(0.5);
            }
            // Finish the one-round diagnostic after both isolated shots have landed.
            if (label.equals("Alpha") && getTurnNumber() >= 400 && seen && getGunHeat() == 0) setFire(3);
            go();
        }
    }
    @Override public void onScannedBot(ScannedBotEvent e) {
        bearing = Math.toDegrees(Math.atan2(e.getY() - getY(), e.getX() - getX()));
        seen = true;
        record("scan", e.getTurnNumber(), String.format(Locale.ROOT,
                ",\"enemy\":%d,\"energy\":%.12f,\"x\":%.12f,\"y\":%.12f", e.getScannedBotId(), e.getEnergy(), e.getX(), e.getY()));
    }
    @Override public void onBulletFired(BulletFiredEvent e) {
        record("fired", e.getTurnNumber(), String.format(Locale.ROOT,
                ",\"bullet\":%d,\"power\":%.12f", e.getBullet().getBulletId(), e.getBullet().getPower()));
    }
    @Override public void onBulletHit(BulletHitBotEvent e) {
        record("hit", e.getTurnNumber(), String.format(Locale.ROOT,
                ",\"enemy\":%d,\"bullet\":%d,\"power\":%.12f,\"damage\":%.12f,\"victimEnergy\":%.12f",
                e.getVictimId(), e.getBullet().getBulletId(), e.getBullet().getPower(), e.getDamage(), e.getEnergy()));
    }
    @Override public void onHitByBullet(HitByBulletEvent e) {
        record("hitBy", e.getTurnNumber(), String.format(Locale.ROOT,
                ",\"enemy\":%d,\"bullet\":%d,\"power\":%.12f,\"damage\":%.12f,\"victimEnergy\":%.12f",
                e.getBullet().getOwnerId(), e.getBullet().getBulletId(), e.getBullet().getPower(), e.getDamage(), e.getEnergy()));
    }
}
