import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Keel: experimental 1v1 duelist. Precise-intersection wave surfing over engine-exact paths (including
 * Tank Royale's instant stop), KNN danger, and KNN guess-factor guns. Uses only its own legal scans,
 * state, commands and events; never reads opponent internals.
 */
public final class KeelHelm extends Bot {
    private final Surfer surfer = new Surfer();
    private final Gun gun = new Gun();
    private final Track me = new Track(), enemy = new Track();
    private final Map<Integer, Double> corrections = new HashMap<>();
    private int scanT = -1, round, orbitDir = 1, radarSign = 1, stuck = 0;
    private double ex, ey, eEnergy = 100;
    private Phys.Pose ePrev;
    private double ePrevEnergy;
    private int ePrevT = -1;
    private final String telemetry = System.getenv("KEEL_HELM_METRICS");
    private final long seed = Long.parseLong(System.getenv().getOrDefault("KEEL_HELM_SEED", "0"));
    long maxNanos = 0;
    /** Energy lead over the opponent above which Keel fires normal 1.2-power shots instead of 0.1 tokens. */
    static final double LEAD = 15, BASE_POWER = 1.2;

    KeelHelm() { super(BotInfo.fromFile("KeelHelm.json")); }
    public static void main(String[] args) { new KeelHelm().start(); }

    private synchronized void mark(String value) {
        if (telemetry == null) return;
        try { Files.writeString(Path.of(telemetry), round + " " + value + "\n", StandardOpenOption.CREATE, StandardOpenOption.APPEND); }
        catch (IOException e) { throw new UncheckedIOException(e); }
    }

    @Override public void onRoundStarted(RoundStartedEvent e) {
        round = e.getRoundNumber();
        surfer.newRound(); gun.newRound(); me.clear(); enemy.clear(); corrections.clear();
        scanT = -1; ePrevT = -1; ePrev = null; stuck = 0; maxNanos = 0;
        orbitDir = new java.util.SplittableRandom(seed * 31 + round).nextBoolean() ? 1 : -1;
        radarSign = orbitDir;
        mark("start");
    }
    @Override public void onRoundEnded(RoundEndedEvent e) {
        mark("stats shots=" + surfer.enemyShots + " hitsTaken=" + surfer.enemyHits + " gun=" + gun.lastChoice
                + " score=" + String.format("%.3f/%.3f", gun.score[0], gun.score[1]) + " maxMs=" + maxNanos / 1_000_000
                + " stoppedRolloutSteps=" + surfer.stoppedRolloutSteps
                + " headingChangedSteps=" + surfer.headingChangedSteps
                + " secondWaveEvaluations=" + surfer.secondWaveEvaluations
                + " executedStopNow=" + surfer.executedStopNow);
        mark("end");
    }
    @Override public void onSkippedTurn(SkippedTurnEvent e) { mark("skip " + e.getTurnNumber()); }
    @Override public void onBulletHit(BulletHitBotEvent e) {
        corrections.merge(e.getTurnNumber(), -e.getDamage(), Double::sum);
        dropOwn(e.getBullet().getBulletId());
    }
    @Override public void onBulletHitWall(BulletHitWallEvent e) { dropOwn(e.getBullet().getBulletId()); }
    @Override public void onBulletFired(BulletFiredEvent e) {
        var b = e.getBullet();
        Phys.Pose o = me.get(e.getTurnNumber() - 1);
        if (o == null) return;
        surfer.bullets.add(new Surfer.Own(b.getBulletId(), e.getTurnNumber(), o.x(), o.y(), b.getDirection(), Phys.bulletSpeed(b.getPower())));
        surfer.reshade(getArenaWidth(), getArenaHeight());
    }
    private void dropOwn(int id) {
        if (surfer.bullets.removeIf(b -> b.id() == id)) surfer.reshade(getArenaWidth(), getArenaHeight());
    }
    @Override public void onHitBot(HitBotEvent e) { corrections.merge(e.getTurnNumber(), -0.6, Double::sum); }
    @Override public void onHitByBullet(HitByBulletEvent e) {
        var b = e.getBullet();
        corrections.merge(e.getTurnNumber(), 3 * b.getPower(), Double::sum);
        surfer.hitBy(e.getTurnNumber(), b.getX(), b.getY(), b.getDirection(), b.getPower());
    }
    @Override public void onBulletHitBullet(BulletHitBulletEvent e) {
        dropOwn(e.getBullet().getBulletId());
        var b = e.getHitBullet();
        surfer.bulletCollision(e.getTurnNumber(), b.getX(), b.getY(), b.getDirection(), b.getPower());
    }

    @Override public void onScannedBot(ScannedBotEvent e) {
        int t = e.getTurnNumber();
        if (t <= scanT) return;
        Phys.Pose p = new Phys.Pose(e.getX(), e.getY(), e.getDirection(), e.getSpeed());
        enemy.add(t, p, getX(), getY());
        double w = getArenaWidth(), h = getArenaHeight();
        if (ePrev != null && ePrevT == t - 1) {
            double corr = corrections.getOrDefault(t - 1, 0.0);
            boolean wall = Math.abs(p.v()) < 0.01 && Math.abs(ePrev.v()) > 2
                    && (p.x() < 18.5 || p.y() < 18.5 || p.x() > w - 18.5 || p.y() > h - 18.5);
            if (wall) corr -= Math.max(0, (Math.abs(ePrev.v()) + 1) / 2 - 1);
            double power = ePrevEnergy - e.getEnergy() + corr;
            if (power >= 0.0999 && power <= 3.0001) {
                Phys.Pose aimed = me.atOrBefore(t - 2, 2);
                if (aimed != null) {
                    double s = Phys.bulletSpeed(power);
                    double[] f = me.features(me.get(t - 2) != null ? t - 2 : me.latest(), ePrev.x(), ePrev.y(), s, w, h);
                    if (f != null) {
                        double direct = Phys.bearing(ePrev.x(), ePrev.y(), aimed.x(), aimed.y());
                        int lat = latDir(aimed, ePrev.x(), ePrev.y(), me.latDir(t - 2));
                        surfer.waves.add(new Surfer.Wave(t, ePrev.x(), ePrev.y(), power, direct, lat, f, surfer.density(f)));
                        surfer.enemyShots++;
                        surfer.reshade(w, h);
                    }
                }
            }
        }
        corrections.keySet().removeIf(k -> k < t - 1);
        gun.observe(t, p.x(), p.y());
        ePrev = p; ePrevEnergy = e.getEnergy(); ePrevT = t;
        ex = p.x(); ey = p.y(); eEnergy = e.getEnergy(); scanT = t;
    }

    private static int latDir(Phys.Pose target, double sx, double sy, int fallback) {
        double lat = target.v() * Phys.sin(target.h() - Phys.bearing(sx, sy, target.x(), target.y()));
        return Math.abs(lat) > 0.05 ? (lat > 0 ? 1 : -1) : fallback;
    }

    private double power(double d) {
        double my = getEnergy();
        // Energy-conserving fire: token shots keep gun data flowing while the opponent spends its energy.
        double p = d < 200 ? 2.5 : my - eEnergy > LEAD ? BASE_POWER : 0.1;
        if (my < 30) p = Math.min(p, 0.4 + my / 30);
        if (my < 8) p = Math.min(p, my / 8);
        double kill = eEnergy <= 4 ? eEnergy / 4 : (eEnergy + 2) / 6;
        p = Math.min(p, kill + 0.05);
        return Phys.clip(p, 0.1, 3);
    }

    private void aimAndFire(int t, Phys.Pose self) {
        double d = Phys.distance(self.x(), self.y(), ex, ey);
        double power = power(d);
        double s = Phys.bulletSpeed(power);
        double[] f = enemy.features(t, self.x(), self.y(), s, getArenaWidth(), getArenaHeight());
        if (f == null) return;
        double direct = Phys.bearing(self.x(), self.y(), ex, ey);
        int lat = enemy.latDir(t);
        double mea = Phys.mea(power);
        double hwDeg = Math.toDegrees(Math.atan(Phys.BOT_RADIUS / d));
        double[] pred = gun.predictions(f, hwDeg / mea);
        int g = gun.choice();
        double gf = Double.isNaN(pred[g]) ? (Double.isNaN(pred[0]) ? 0 : pred[0]) : pred[g];
        double aim = direct + gf * mea * lat;
        boolean real = false;
        double offset = Math.abs(Phys.norm(aim - getGunDirection()));
        if (getGunHeat() == 0 && offset <= hwDeg * 0.9 && getEnergy() > power + 0.1 && (getEnergy() > 1.5 || eEnergy < 0.5)) {
            real = setFire(power);
        }
        gun.launch(new Gun.Wave(t + 1, self.x(), self.y(), power, direct, lat, f, real, pred));
        setGunTurnRate(Phys.norm(aim - getGunDirection()));
    }

    @Override public void run() {
        setAdjustGunForBodyTurn(true); setAdjustRadarForGunTurn(true); setAdjustRadarForBodyTurn(true); setFireAssist(false);
        while (isRunning()) {
            long start = System.nanoTime();
            int t = getTurnNumber();
            double w = getArenaWidth(), h = getArenaHeight();
            Phys.Pose self = new Phys.Pose(getX(), getY(), getDirection(), getSpeed());
            me.add(t, self, scanT >= 0 ? ex : w / 2, scanT >= 0 ? ey : h / 2);
            surfer.update(t, self);
            if (scanT < 0 || t - scanT > 4) {
                setRadarTurnRate(45 * radarSign);
                setTurnRate(Phys.norm(Phys.bearing(getX(), getY(), w / 2, h / 2) - getDirection()));
                setTargetSpeed(scanT < 0 ? 2 : 6);
                setGunTurnRate(0);
            } else {
                double bearing = Phys.bearing(getX(), getY(), ex, ey);
                double rd = Phys.norm(bearing - getRadarDirection());
                if (Math.abs(rd) > 0.01) radarSign = rd > 0 ? 1 : -1;
                setRadarTurnRate(rd + radarSign * 20);
                Surfer.Plan plan = surfer.plan(self, t, ex, ey, w, h, getEnergy());
                if (plan != null) {
                    orbitDir = plan.dir();
                    setTurnRate(plan.turn()); setTargetSpeed(plan.speed());
                    if (plan.speed() == 0) surfer.executedStopNow++;
                } else {
                    if (Math.abs(getSpeed()) < 0.5) stuck++; else stuck = 0;
                    if (stuck > 6) { orbitDir = -orbitDir; stuck = 0; }
                    double[] c = Surfer.orbit(self, ex, ey, orbitDir, w, h);
                    setTurnRate(c[0]); setTargetSpeed(c[1]);
                }
                if (scanT == t) aimAndFire(t, self);
            }
            maxNanos = Math.max(maxNanos, System.nanoTime() - start);
            go();
        }
    }
}
