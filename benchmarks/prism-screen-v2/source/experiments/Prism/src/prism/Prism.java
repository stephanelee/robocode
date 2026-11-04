package prism;

import dev.robocode.tankroyale.botapi.Bot;
import dev.robocode.tankroyale.botapi.BotInfo;
import dev.robocode.tankroyale.botapi.BotState;
import dev.robocode.tankroyale.botapi.BulletState;
import dev.robocode.tankroyale.botapi.GameSetup;
import dev.robocode.tankroyale.botapi.events.*;
import prism.core.Command;
import prism.core.Obs;
import prism.core.Rules;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Thin Tank Royale 1.4.0 adapter. Event handlers only capture data; all decisions run in
 * {@link Brain#turn} once per tick from {@link #run()}, after {@code go()} has returned and the
 * Bot API has dispatched that tick's events. Every command field is set every turn, including an
 * explicit zero firepower whenever the gun is cool, so a stale request can never fire.
 */
public final class Prism extends Bot {
    public static final String VERSION = "0.2.0";

    private Brain brain;
    private Telemetry tel;
    private GameSetup setup;
    private TickEvent tick;
    private int processedTurn = -1, processedRound = -1;
    private final List<Obs.Event> skipped = new ArrayList<>();

    Prism() { super(BotInfo.fromFile("Prism.json")); }

    public static void main(String[] args) { new Prism().start(); }

    @Override
    public void onGameStarted(GameStartedEvent e) {
        setup = e.getGameSetup();
        String tp = System.getenv("PRISM_TELEMETRY_PATH");
        tel = new Telemetry(tp == null || tp.isBlank() ? null : Path.of(tp));
        brain = null; // a new game is cold
    }

    @Override public void onTick(TickEvent e) { tick = e; }

    @Override public void onSkippedTurn(SkippedTurnEvent e) { skipped.add(new Obs.Skipped(e.getTurnNumber())); }

    @Override
    public void run() {
        setAdjustGunForBodyTurn(true);
        setAdjustRadarForGunTurn(true);
        setAdjustRadarForBodyTurn(true);
        while (isRunning()) {
            TickEvent t = tick;
            if (t != null && (t.getTurnNumber() != processedTurn || t.getRoundNumber() != processedRound)) {
                processedTurn = t.getTurnNumber();
                processedRound = t.getRoundNumber();
                apply(decide(t));
            } else if (t == null) {
                setTargetSpeed(0);
                setRadarTurnRate(45);
            }
            go();
        }
    }

    private Command decide(TickEvent t) {
        BotState s = t.getBotState();
        if (brain == null) {
            double cooling = setup == null ? 0.1 : setup.getGunCoolingRate();
            int w = setup == null ? getArenaWidth() : setup.getArenaWidth();
            int h = setup == null ? getArenaHeight() : setup.getArenaHeight();
            Rules rules = new Rules(w, h, cooling, Math.max(1, s.getEnemyCount()));
            long seed = parseSeed(System.getenv("PRISM_SEED"));
            if (tel == null) tel = new Telemetry(null);
            brain = new Brain(rules, seed, tel);
        }
        List<Obs.Event> events = new ArrayList<>(skipped);
        skipped.clear();
        int me = getMyId();
        for (BotEvent e : t.getEvents()) {
            Obs.Event o = convert(e, me);
            if (o != null) events.add(o);
        }
        Obs.Own own = new Obs.Own(t.getTurnNumber(), s.getX(), s.getY(), Math.toRadians(s.getDirection()),
                Math.toRadians(s.getGunDirection()), Math.toRadians(s.getRadarDirection()), s.getSpeed(), s.getEnergy(),
                s.getGunHeat(), s.getEnemyCount());
        return brain.turn(new Obs.Turn(t.getRoundNumber(), own, events));
    }

    private static Obs.Event convert(BotEvent e, int me) {
        int turn = e.getTurnNumber();
        if (e instanceof ScannedBotEvent s)
            return new Obs.Scan(turn, s.getScannedBotId(), s.getEnergy(), s.getX(), s.getY(), Math.toRadians(s.getDirection()), s.getSpeed());
        if (e instanceof BulletFiredEvent f) {
            BulletState b = f.getBullet();
            return new Obs.BulletFired(turn, b.getBulletId(), b.getPower(), b.getX(), b.getY(), Math.toRadians(b.getDirection()));
        }
        if (e instanceof BulletHitBotEvent h)
            return new Obs.BulletHit(turn, h.getBullet().getBulletId(), h.getVictimId(), h.getDamage(), h.getEnergy(), h.getBullet().getPower());
        if (e instanceof HitByBulletEvent h) {
            BulletState b = h.getBullet();
            return new Obs.HitByBullet(turn, b.getBulletId(), b.getOwnerId(), b.getPower(), b.getX(), b.getY(), Math.toRadians(b.getDirection()), h.getDamage(), h.getEnergy());
        }
        if (e instanceof BulletHitBulletEvent h) {
            BulletState o = h.getHitBullet();
            return new Obs.BulletHitBullet(turn, h.getBullet().getBulletId(), o.getBulletId(), o.getOwnerId(), o.getPower(), o.getX(), o.getY(), Math.toRadians(o.getDirection()));
        }
        if (e instanceof BulletHitWallEvent w) return new Obs.BulletHitWall(turn, w.getBullet().getBulletId());
        if (e instanceof HitWallEvent) return new Obs.HitWall(turn);
        if (e instanceof HitBotEvent h) return new Obs.HitBot(turn, h.getVictimId(), h.getEnergy(), h.getX(), h.getY(), h.isRammed());
        if (e instanceof DeathEvent) return new Obs.Death(turn);
        if (e instanceof BotDeathEvent d) return d.getVictimId() == me ? new Obs.Death(turn) : new Obs.BotDeath(turn, d.getVictimId());
        if (e instanceof WonRoundEvent) return new Obs.Won(turn);
        return null;
    }

    private void apply(Command c) {
        setTargetSpeed(c.targetSpeed());
        setTurnRate(Math.toDegrees(c.bodyTurn()));
        setGunTurnRate(Math.toDegrees(c.gunTurn()));
        setRadarTurnRate(Math.toDegrees(c.radarTurn()));
        if (c.firepower() > 0) {
            if (!setFire(c.firepower())) brain.fireRejected();
        } else if (getGunHeat() == 0) {
            setFire(0); // clear any persisted firepower: a stale plan never fires
        }
    }

    @Override
    public void onRoundEnded(RoundEndedEvent e) {
        if (tel != null && brain != null) {
            tel.writeLine(tel.json("\"bot\":\"Prism\",\"version\":\"" + VERSION + "\",\"profile\":\"" + Profile.VERSION
                    + "\",\"engine\":\"" + Rules.ENGINE_PROFILE + "\",\"event\":\"round\",\"round\":" + e.getRoundNumber()
                    + ",\"seed\":" + brain.seed() + ","));
        }
    }

    @Override
    public void onGameEnded(GameEndedEvent e) {
        if (tel != null && brain != null) {
            tel.writeLine(tel.json("\"bot\":\"Prism\",\"version\":\"" + VERSION + "\",\"profile\":\"" + Profile.VERSION
                    + "\",\"event\":\"game\",\"seed\":" + brain.seed() + ","));
        }
        brain = null;
    }

    static long parseSeed(String s) {
        if (s == null || s.isBlank()) return 0x5052_4953_4dL; // fixed default: deterministic
        try { return Long.parseLong(s.trim()); } catch (NumberFormatException ex) { return 0x5052_4953_4dL; }
    }
}
