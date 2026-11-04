import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Aegis: an isolated 1v1 bullet-shield experiment. It infers each enemy shot from legal scan energy drops and
 * models it as a head-on shot at our position predicted from the scan two turns before creation. It moves along
 * one axis so every enemy wave (fired or not) breaks where we were when it was launched, keeping a learning gun's
 * labels at zero, while fast passes before each predicted enemy shot give 0.1-power interceptors a crossing angle.
 * Only legal scans, own state and confirmed events are used; there is no persisted model or cross-round state.
 */
public class Aegis extends Bot {
    static final String NAME = "Aegis", VERSION = "0.1.0";
    static final int HORIZON = 26;
    static final double SPAN = 70, SHIELD_OFFSET = 30, MIN_SHIELD_POWER = 0.09, WALL_MARGIN = 22;
    static final int PREP = 4;
    static final double REPLAY = 10, OFFSET_WEIGHT = 0.5, SEED_AMPLITUDE = 50, SEED_WEIGHT = 1;
    static final int SEED_PERIOD = 22;
    static final int RELOAD = 12, SEARCH_LEAD = 10;
    static final double DODGE_CLEARANCE = 26, DODGE_WEIGHT = 60; static final int GOOD_COVERAGE = 5;
    static final double FAST_SPEED = 8, FAST_WEIGHT = 40, LOCK = 1000;
    private int lastFireTurn = 16;

    private final long seed = Long.parseLong(System.getenv().getOrDefault("AEGIS_SEED", "0"));
    private final Path metrics = path("AEGIS_METRICS_PATH", "Aegis-metrics.jsonl");
    private final Path tracePath = System.getenv("AEGIS_TRACE_PATH") == null ? null : Path.of(System.getenv("AEGIS_TRACE_PATH")).toAbsolutePath();
    private PrintWriter trace;

    // round state
    private final Map<Integer, double[]> ownHistory = new HashMap<>();
    private final Map<Integer, double[]> enemyHistory = new HashMap<>();
    private final Map<Integer, Double> energyCorrection = new HashMap<>();
    private final List<Shot> shots = new ArrayList<>();
    private double enemyEnergy = -1, lastOwnEnergy = -1, ownFiredThisTurnPower;
    private int enemyScanTurn = -1, enemyId = -1;
    private double ex, ey, enemyHeading, enemySpeed;
    private boolean axisReady;
    private double axisHeading, homeX, homeY, minS, maxS, arenaW = 800, arenaH = 600;
    private LateralPlanner.Result lastPlan; private LateralPlanner planner;
    private int planTurn = -1;
    private Shot pendingShot; private int pendingFireTurn = -1, pendingIntercept = -1; private double pendingDirection;
    private final Map<Integer, Double> lockedS = new HashMap<>();
    private Shot intentShot; private int intentFire = -1;
    private final Map<Integer, Double> ownHitBonus = new HashMap<>(), ownDamage = new HashMap<>();

    // metrics
    private int roundsCompleted, detected, shieldsFired, blocked, hitsTaken, skips, walls, botHits, shotsFired, ourHits, unmatchedRemovals, inactivityTurns;
    private double taken, dealt, energySpent, labelErrorSum; private int labelSamples;
    private boolean won, died; private long maxMicros; private int slowTurns;
    private StringJoiner skippedTurnList = new StringJoiner(",", "[", "]");

    public static void main(String[] args) { warmUp(); new Aegis().start(); }

    /** Gameplay-neutral JIT warm-up of the planner and solver on synthetic inputs before connecting. */
    static void warmUp() {
        long seedBits = 12345;
        for (int round = 0; round < 60; round++) {
            LateralPlanner planner = new LateralPlanner(-SPAN, SPAN, HORIZON);
            for (int i = 0; i < planner.states(); i++) planner.addPositionCost(1 + (i % HORIZON), i, (i * 7 + round) % 13);
            planner.plan(round % 7 - 3, (round % 5) - 2);
            Shot shot = new Shot(100, 1.95, 100, 100, 500, 300 + round % 9);
            final double sx = 500 + round % 11;
            ShieldSolver.Positions own = new ShieldSolver.Positions() {
                public double x(int turn) { return sx + 0.3 * (turn % 5); }
                public double y(int turn) { return 300 + 0.5 * (turn % 7); }
            };
            ShieldSolver.Lines lines = ShieldSolver.lines(shot, own);
            for (int f = shot.contactTurn() - 4; f <= shot.contactTurn(); f++) ShieldSolver.best(shot, f, own, lines);
            seedBits ^= Double.doubleToLongBits(DiamondLabel.offset(100, 100, 99, 14.15, 120, new double[]{480, 485, 490}, new double[]{300, 300, 300}, 500, 300));
        }
        if (seedBits == 42) System.out.println("[Aegis] warm-up checksum");
    }

    Aegis() {
        super(BotInfo.fromFile("Aegis.json"));
        if (Files.exists(metrics)) throw new IllegalStateException("Metrics path must be fresh");
        try { if (tracePath != null) trace = new PrintWriter(Files.newBufferedWriter(tracePath)); } catch (IOException e) { throw new UncheckedIOException(e); }
        System.out.println("[Aegis] cold seed=" + seed);
    }

    private static Path path(String key, String fallback) { return Path.of(System.getenv().getOrDefault(key, fallback)).toAbsolutePath(); }

    @Override public void run() {
        setAdjustGunForBodyTurn(true); setAdjustRadarForGunTurn(true); setAdjustRadarForBodyTurn(true);
        while (isRunning()) {
            int t = getTurnNumber();
            ownHistory.put(t, new double[]{getX(), getY(), getSpeed(), getDirection()});
            if (axisReady) emit("s", t, "s", sOf(getX(), getY()));
            long start = System.nanoTime();
            accountOwnEnergy(t);
            radar(t);
            move(t);
            gun(t);
            long micros = (System.nanoTime() - start) / 1000;
            maxMicros = Math.max(maxMicros, micros); if (micros > 20000) slowTurns++;
            go();
        }
    }

    // ---------------------------------------------------------------- radar
    private void radar(int t) {
        if (enemyScanTurn < 0 || t - enemyScanTurn > 2) { setRadarTurnRate(45); return; }
        double b = Physics.bearing(getX(), getY(), ex, ey), r = Physics.normalize(b - getRadarDirection());
        r += Math.signum(r == 0 ? 1 : r) * 22.5;
        setRadarTurnRate(Math.max(-45, Math.min(45, r)));
    }

    // ---------------------------------------------------------------- own energy / inactivity
    private void accountOwnEnergy(int t) {
        double e = getEnergy();
        if (lastOwnEnergy >= 0) {
            double expected = lastOwnEnergy - ownFiredThisTurnPower - ownDamage.getOrDefault(t, 0.0) + ownHitBonus.getOrDefault(t, 0.0);
            double unexplained = expected - e;
            if (Math.abs(unexplained - 0.1) < 1e-6) { inactivityTurns++; energyCorrection.merge(t, -0.1, Double::sum); }
        }
        lastOwnEnergy = e; ownFiredThisTurnPower = 0;
    }

    // ---------------------------------------------------------------- movement
    private double ax() { return Math.cos(Math.toRadians(axisHeading)); }
    private double ay() { return Math.sin(Math.toRadians(axisHeading)); }
    private double sOf(double x, double y) { return (x - homeX) * ax() + (y - homeY) * ay(); }

    private void move(int t) {
        if (!axisReady) {
            if (enemyScanTurn < 0) { setTargetSpeed(0); return; }
            double cx = Math.max(110, Math.min(arenaW - 110, getX())), cy = Math.max(110, Math.min(arenaH - 110, getY()));
            if (Math.hypot(cx - getX(), cy - getY()) > 4 && t < 40) {
                double b = Physics.bearing(getX(), getY(), cx, cy), turn = Physics.normalize(b - getDirection()), dir = 1;
                if (Math.abs(turn) > 90) { turn = Physics.normalize(turn + 180); dir = -1; }
                setTurnRate(turn); double d = Math.hypot(cx - getX(), cy - getY());
                setTargetSpeed(Math.abs(turn) > 30 ? 0 : dir * Math.min(8, Math.max(1, Math.sqrt(2 * d))));
                return;
            }
            double toEnemy = Physics.bearing(getX(), getY(), ex, ey);
            double want = Physics.normalize(toEnemy + 90), err = Physics.normalize(want - getDirection());
            if (Math.abs(err) > 90) { want = Physics.normalize(want + 180); err = Physics.normalize(want - getDirection()); }
            if (Math.abs(getSpeed()) > 0) { setTargetSpeed(0); setTurnRate(err); return; }
            if (Math.abs(err) > 3) { setTurnRate(err); setTargetSpeed(0); return; }
            axisReady = true; axisHeading = getDirection(); homeX = getX(); homeY = getY();
            double lo = -SPAN, hi = SPAN;
            for (double s = 0; s >= -SPAN; s -= 0.5) if (!inside(s)) { lo = s + 0.5; break; }
            for (double s = 0; s <= SPAN; s += 0.5) if (!inside(s)) { hi = s - 0.5; break; }
            minS = lo; maxS = hi;
        }
        setTurnRate(0);
        double s0 = sOf(getX(), getY());
        if (planner == null) planner = new LateralPlanner(minS, maxS, HORIZON); else planner.reset();
        LateralPlanner planner = this.planner;
        int ns = planner.states();
        for (int i = 0; i < ns; i++) { double s = planner.position(i); for (int k = 1; k <= HORIZON; k++) planner.addPositionCost(k, i, 0.0001 * s * s); }
        // Replay cost: every enemy wave launched at tau breaks about F turns later; being laterally where we were at tau
        // keeps its label at GF 0. Uses only our own history and legal enemy scan positions.
        double waveSpeed = Physics.bulletSpeed(lastEnemyPower);
        for (int k = 1; k <= HORIZON; k++) {
            int turn = t + k;
            double d = Math.hypot(ex - getX(), ey - getY()), tau = turn - (d / waveSpeed - 0.5);
            for (int iter = 0; iter < 2; iter++) {
                double[] e0 = enemyAt(tau); if (e0 == null) break;
                d = Math.hypot(e0[0] - getX(), e0[1] - getY()); tau = turn - (d / waveSpeed - 0.5);
            }
            double[] ref = ownAt(tau), src = enemyAt(tau);
            if (ref == null || src == null || tau > t) { // no history yet: seed a periodic swing
                double phase = 2 * Math.PI * turn / SEED_PERIOD;
                for (int i = 0; i < ns; i++) { double ds = planner.position(i) - SEED_AMPLITUDE * Math.sin(phase); planner.addPositionCost(k, i, SEED_WEIGHT * ds * ds); }
                continue;
            }
            double lx = ref[0] - src[0], ly = ref[1] - src[1], ll = Math.hypot(lx, ly), nx = -ly / ll, ny = lx / ll;
            for (int i = 0; i < ns; i++) {
                double sp = planner.position(i), px = homeX + ax() * sp, py = homeY + ay() * sp;
                double lat = (px - ref[0]) * nx + (py - ref[1]) * ny;
                planner.addPositionCost(k, i, REPLAY * lat * lat);
            }
        }
        // Shield geometry is fixed when the enemy fires (replay repeats that motion at contact): be fast then.
        if (lastFireTurn > 0) {
            int period = (int) Math.ceil(Physics.gunHeat(lastEnemyPower) / 0.1 - 1e-9) + 1;
            for (int next = lastFireTurn + period; next - t <= HORIZON + 4; next += period) {
                for (int turn = next - 4; turn <= next - 1; turn++) {
                    if (turn <= t || turn - t > HORIZON) continue;
                    for (int i = 0; i < ns; i++) for (int v = 0; v < LateralPlanner.NV; v++) {
                        double miss = Math.max(0, FAST_SPEED - Math.abs(LateralPlanner.speedOf(v)));
                        planner.addCost(turn - t, i, v, FAST_WEIGHT * miss * miss);
                    }
                }
            }
        }
        for (Shot shot : shots) {
            if (shot.resolved || !shot.dodge) continue;
            double nx = -Math.sin(Math.toRadians(shot.heading)), ny = Math.cos(Math.toRadians(shot.heading));
            for (int turn = Math.max(t + 1, shot.contactTurn() - 2); turn <= shot.passTurn() && turn - t <= HORIZON; turn++)
                for (int i = 0; i < ns; i++) {
                    double sp = planner.position(i), px = homeX + ax() * sp, py = homeY + ay() * sp;
                    double off = Math.abs((px - shot.tx) * nx + (py - shot.ty) * ny), miss = Math.max(0, DODGE_CLEARANCE - off);
                    planner.addPositionCost(turn - t, i, DODGE_WEIGHT * miss * miss);
                }
        }
        for (Shot shot : shots) {
            if (shot.resolved || shot.shielded || shot.power < MIN_SHIELD_POWER) continue;
            double nx = -Math.sin(Math.toRadians(shot.heading)), ny = Math.cos(Math.toRadians(shot.heading));
            int prep = shot.contactTurn() - PREP;
            if (prep > t && prep - t <= HORIZON) {
                for (int i = 0; i < ns; i++) {
                    double sp = planner.position(i), px = homeX + ax() * sp, py = homeY + ay() * sp;
                    double off = Math.abs((px - shot.tx) * nx + (py - shot.ty) * ny), miss = Math.max(0, SHIELD_OFFSET - off);
                    planner.addPositionCost(prep - t, i, OFFSET_WEIGHT * miss * miss);
                }
            }
        }
        for (var lock : lockedS.entrySet()) {
            int k = lock.getKey() - t; if (k < 1 || k > HORIZON) continue;
            for (int i = 0; i < ns; i++) { double ds = planner.position(i) - lock.getValue(); planner.addPositionCost(k, i, LOCK * ds * ds); }
        }
        lockedS.keySet().removeIf(turn -> turn <= t);
        lastPlan = planner.plan(s0, getSpeed()); planTurn = t;
        setTargetSpeed(lastPlan.targetSpeed());
    }

    static final int PASS_STATES = 8;
    private double lastEnemyPower = 1.95;

    /** Linear interpolation of our end-of-turn history at a fractional turn. */
    private double[] ownAt(double tau) { return interpolate(ownHistory, tau); }
    private double[] enemyAt(double tau) { return interpolate(enemyHistory, tau); }
    private static double[] interpolate(Map<Integer, double[]> h, double tau) {
        int a = (int) Math.floor(tau); double w = tau - a; double[] p = h.get(a), q = h.get(a + 1);
        if (p == null) return null; if (q == null) return w < 1e-9 ? p : null;
        return new double[]{p[0] + (q[0] - p[0]) * w, p[1] + (q[1] - p[1]) * w};
    }
    private boolean inside(double s) {
        double x = homeX + ax() * s, y = homeY + ay() * s;
        return x > WALL_MARGIN && y > WALL_MARGIN && x < arenaW - WALL_MARGIN && y < arenaH - WALL_MARGIN;
    }

    private ShieldSolver.Positions planned(int t) {
        return new ShieldSolver.Positions() {
            double[] at(int turn) {
                if (turn <= t) { double[] h = ownHistory.get(turn); return h == null ? new double[]{getX(), getY()} : h; }
                if (lastPlan == null || planTurn != t) return new double[]{getX(), getY()};
                int k = Math.min(turn - t, HORIZON); double s = lastPlan.positions()[k];
                return new double[]{homeX + ax() * s, homeY + ay() * s};
            }
            public double x(int turn) { return at(turn)[0]; }
            public double y(int turn) { return at(turn)[1]; }
        };
    }

    // ---------------------------------------------------------------- shield gun
    private void gun(int t) {
        if (pendingShot != null && pendingFireTurn == t + 1) {
            boolean aligned = Math.abs(Physics.normalize(getGunDirection() - pendingDirection)) < 1e-6;
            if (!(aligned && getGunHeat() == 0 && !pendingShot.resolved)) emit("abort", t, "fireTurn", pendingShot.fireTurn, "aligned", aligned, "heat", getGunHeat(), "resolved", pendingShot.resolved);
            if (aligned && getGunHeat() == 0 && !pendingShot.resolved) {
                setFire(ShieldSolver.POWER);
                pendingShot.shielded = true; pendingShot.shieldTurn = t + 1; shieldsFired++;
                emit("shield", t, "fireTurn", pendingShot.fireTurn, "dir", pendingDirection);
            }
            if (!pendingShot.shielded) for (int turn = t + 1; turn <= pendingIntercept + 1; turn++) lockedS.remove(turn);
            pendingShot = null; pendingFireTurn = -1;
            setGunTurnRate(0);
            return;
        }
        pendingShot = null;
        Shot target = null;
        for (Shot s : shots) if (!s.resolved && !s.shielded && s.power >= MIN_SHIELD_POWER && s.contactTurn() > t + 2
                && (target == null || s.contactTurn() < target.contactTurn())) target = s;
        if (target == null || !axisReady) { if (enemyScanTurn >= 0) setGunTurnRate(clip20(Physics.normalize(Physics.bearing(getX(), getY(), ex, ey) - getGunDirection()))); return; }
        double heat = getGunHeat(); int cool = 0; while (heat > 1e-12) { heat = Math.max(0, heat - 0.1); cool++; }
        int earliest = Math.max(t + 2, t + cool + 1);
        ShieldSolver.Positions own = planned(t);
        if (target.contactTurn() - t > SEARCH_LEAD + 4) { setGunTurnRate(clip20(Physics.normalize(Physics.bearing(getX(), getY(), target.ox, target.oy) - 25 - getGunDirection()))); return; }
        ShieldSolver.Lines lines = ShieldSolver.lines(target, own);
        Shot following = null;
        for (Shot s : shots) if (s != target && !s.resolved && !s.shielded && s.power >= MIN_SHIELD_POWER && s.contactTurn() >= target.contactTurn()
                && (following == null || s.contactTurn() < following.contactTurn())) following = s;
        int latestFree = following == null ? Integer.MAX_VALUE : following.contactTurn() - 3 - RELOAD;
        ShieldSolver.Plan best = null, bestFree = null;
        if (intentShot == target && intentFire >= earliest && intentFire <= target.contactTurn()) {
            best = ShieldSolver.best(target, intentFire, own, lines); // keep the chosen fire turn; trajectory is locked to it
        } else {
            for (int f = Math.max(earliest, target.contactTurn() - SEARCH_LEAD); f <= target.contactTurn(); f++) {
                ShieldSolver.Plan p = ShieldSolver.best(target, f, own, lines);
                if (best == null || p.score() > best.score() + 1e-9) best = p;
                if (f <= latestFree && p.covered() > 0 && (bestFree == null || p.score() > bestFree.score() + 1e-9)) bestFree = p;
            }
            if (bestFree != null && bestFree.score() >= 0.6 * best.score()) best = bestFree;
            if (best != null && best.covered() > 0) {
                intentShot = target; intentFire = best.fireTurn();
                for (int turn = t + 1; turn <= best.interceptTurn() + 1 && turn - t <= HORIZON; turn++) lockedS.put(turn, lastPlan.positions()[turn - t]);
            }
        }
        if (best == null || best.covered() == 0) { target.dodge = true; setGunTurnRate(0); emit("noplan", t, "fireTurn", target.fireTurn, "earliest", earliest, "contact", target.contactTurn(), "heat", getGunHeat()); return; }
        double diff = Physics.normalize(best.direction() - getGunDirection());
        if (best.fireTurn() == t + 2 && Math.abs(diff) <= 20) {
            if (best.covered() < GOOD_COVERAGE) target.dodge = true;
            intentShot = null; setGunTurnRate(diff); pendingShot = target; pendingFireTurn = t + 2; pendingDirection = best.direction(); pendingIntercept = best.interceptTurn();
            for (int turn = t + 1; turn <= best.interceptTurn() + 1 && turn - t <= HORIZON; turn++) lockedS.put(turn, lastPlan.positions()[turn - t]);
            emit("commit", t, "fireTurn", target.fireTurn, "f", best.fireTurn(), "covered", best.covered(), "dir", best.direction(), "intercept", best.interceptTurn(), "contact", target.contactTurn(), "mask", "\"" + Integer.toBinaryString(best.mask() | 512).substring(1) + "\"", "plan", java.util.Arrays.toString(java.util.Arrays.copyOf(lastPlan.positions(), 8)).replace(" ", ""), "s0", sOf(getX(), getY()));
        } else { setGunTurnRate(clip20(diff)); if (best.fireTurn() == t + 2) emit("unreachable", t, "fireTurn", target.fireTurn, "diff", diff); }
    }

    private static double clip20(double v) { return Math.max(-20, Math.min(20, v)); }

    // ---------------------------------------------------------------- events
    @Override public void onScannedBot(ScannedBotEvent e) {
        int t = e.getTurnNumber(); enemyId = e.getScannedBotId();
        if (enemyScanTurn == t - 1 && enemyEnergy >= 0) {
            double power = inferredPower(enemyEnergy, e.getEnergy(), energyCorrection.getOrDefault(t - 1, 0.0));
            double[] aim = ownHistory.get(t - 2), ref = ownHistory.get(t - 1);
            if (!Double.isNaN(power) && aim != null && ref != null) {
                double h = Math.toRadians(aim[3]);
                Shot shot = new Shot(t, power, ex, ey, aim[0] + Math.cos(h) * aim[2], aim[1] + Math.sin(h) * aim[2]);
                shot.rx = ref[0]; shot.ry = ref[1];
                double[] e0 = enemyHistory.get(t - 2);
                if (e0 != null) { double cv = classicSpeed(e0[4], enemySpeed); shot.predictedSource(e0[0] + Math.cos(Math.toRadians(enemyHeading)) * cv, e0[1] + Math.sin(Math.toRadians(enemyHeading)) * cv); }
                shots.add(shot); detected++; lastEnemyPower = power; lastFireTurn = t;
                emit("fire", t, "power", power, "ox", ex, "oy", ey, "tx", shot.tx, "ty", shot.ty, "psx", shot.psx, "psy", shot.psy);
            }
        }
        enemyHistory.put(t, new double[]{e.getX(), e.getY(), e.getEnergy(), e.getDirection(), e.getSpeed()});
        enemyHeading = e.getDirection(); enemySpeed = e.getSpeed();
        emit("scan", t, "x", e.getX(), "y", e.getY(), "dir", e.getDirection(), "v", e.getSpeed());
        ex = e.getX(); ey = e.getY(); enemyEnergy = e.getEnergy(); enemyScanTurn = t;
    }
    /**
     * Fire power implied by consecutive legal scans. {@code correction} holds the previous turn's known non-fire
     * changes (our bullet damage negative, the shooter's hit bonus positive, shared inactivity damage negative),
     * which the pinned engine reports one scan late. Returns NaN when the drop is not a legal fire power.
     */
    static double inferredPower(double previousEnergy, double energy, double correction) {
        double drop = previousEnergy - energy + correction;
        return drop >= 0.0999 && drop <= 3.0001 ? Math.round(drop * 1000) / 1000.0 : Double.NaN;
    }

    /**
     * Speed the shooter's classic-order predictor expected after one turn, inferred from observed speeds:
     * the pinned engine stops or reverses faster than classic Robocode (which decelerates by 2).
     */
    static double classicSpeed(double v0, double v1) {
        double sign = v0 > 0 ? 1 : -1;
        if (Math.abs(v0) > 2.0001 && (Math.abs(v1) < 1e-9 || (Math.abs(Math.abs(v1) - (Math.abs(v0) / 2 - 1)) < 1e-6 && Math.abs(v0) - Math.abs(v1) > 2.0001)))
            return v0 - 2 * sign;
        return v1;
    }

    @Override public void onHitByBullet(HitByBulletEvent e) {
        var b = e.getBullet(); hitsTaken++; taken += e.getDamage();
        energyCorrection.merge(e.getTurnNumber(), 3 * b.getPower(), Double::sum);
        ownDamage.merge(e.getTurnNumber(), e.getDamage(), Double::sum);
        Shot s = match(b.getPower(), b.getX(), b.getY(), b.getDirection());
        if (s != null) { s.resolved = true; s.outcome = "hit"; }
        double lineError = s == null ? Double.NaN : -Math.sin(Math.toRadians(b.getDirection())) * (s.tx - b.getX()) + Math.cos(Math.toRadians(b.getDirection())) * (s.ty - b.getY());
        double solverError = s == null ? Double.NaN : solverError(s, b.getX(), b.getY(), b.getDirection());
        emit("hitBy", e.getTurnNumber(), "power", b.getPower(), "bx", b.getX(), "by", b.getY(), "dir", b.getDirection(), "matched", s != null ? s.fireTurn : -1,
                "shielded", s != null && s.shielded, "lineError", lineError, "solverError", solverError, "dodge", s != null && s.dodge);
    }
    @Override public void onBulletHit(BulletHitBotEvent e) {
        ourHits++; dealt += e.getDamage();
        energyCorrection.merge(e.getTurnNumber(), -e.getDamage(), Double::sum);
        ownHitBonus.merge(e.getTurnNumber(), 3 * e.getBullet().getPower(), Double::sum);
    }
    @Override public void onBulletHitBullet(BulletHitBulletEvent e) {
        var b = e.getHitBullet();
        Shot s = match(b.getPower(), b.getX(), b.getY(), b.getDirection());
        if (s != null) { s.resolved = true; s.outcome = "blocked"; blocked++; } else unmatchedRemovals++;
        double lineError = s == null ? Double.NaN : -Math.sin(Math.toRadians(b.getDirection())) * (s.tx - b.getX()) + Math.cos(Math.toRadians(b.getDirection())) * (s.ty - b.getY());
        emit("blocked", e.getTurnNumber(), "power", b.getPower(), "matched", s != null ? s.fireTurn : -1, "lineError", lineError, "solverError", s == null ? Double.NaN : solverError(s, b.getX(), b.getY(), b.getDirection()));
    }
    @Override public void onBulletFired(BulletFiredEvent e) {
        double[] prev = ownHistory.get(e.getTurnNumber() - 1);
        emit("ownBullet", e.getTurnNumber(), "id", e.getBullet().getBulletId(), "x", e.getBullet().getX(), "y", e.getBullet().getY(), "dir", e.getBullet().getDirection(),
             "px", prev == null ? -1.0 : prev[0], "py", prev == null ? -1.0 : prev[1]);
        shotsFired++; energySpent += e.getBullet().getPower(); ownFiredThisTurnPower += e.getBullet().getPower(); }
    @Override public void onHitWall(HitWallEvent e) { walls++; }
    @Override public void onHitBot(HitBotEvent e) { botHits++; }
    @Override public void onSkippedTurn(SkippedTurnEvent e) { skips++; if (skippedTurnList.length() < 200) skippedTurnList.add(String.valueOf(e.getTurnNumber())); }
    @Override public void onWonRound(WonRoundEvent e) { won = true; }
    @Override public void onDeath(DeathEvent e) { died = true; }

    private Shot match(double power, double x, double y, double direction) {
        Shot best = null; double bestErr = 2.0;
        for (Shot s : shots) {
            if (s.resolved || Math.abs(s.power - power) > 0.01) continue;
            double h = Math.toRadians(direction), err = Math.abs(-Math.sin(h) * (s.ox - x) + Math.cos(h) * (s.oy - y));
            if (err < bestErr) { bestErr = err; best = s; }
        }
        return best;
    }
    /** Lateral offset, at the aim range, of the actual bullet line from the solver's centre line. */
    static double solverError(Shot s, double bx, double by, double direction) {
        double h = Math.toRadians(s.heading), px = s.ox + Math.cos(h) * s.range, py = s.oy + Math.sin(h) * s.range, d = Math.toRadians(direction);
        return -Math.sin(d) * (px - bx) + Math.cos(d) * (py - by);
    }

    @Override public void onTick(TickEvent e) {
        int t = e.getTurnNumber();
        for (Shot s : shots) {
            if (!s.resolved && t > s.passTurn() + 2) { s.resolved = true; s.outcome = s.shielded ? "shieldMissedEnemyMissed" : "missed"; }
            if (t == s.breakLast() + 8) {
                int t0 = s.breakFirst() - 1; double[] xs = new double[PASS_STATES], ys = new double[PASS_STATES]; boolean ok = true;
                for (int k = 0; k < PASS_STATES; k++) { double[] h = k + t0 == t ? new double[]{e.getBotState().getX(), e.getBotState().getY()} : ownHistory.get(t0 + k); if (h == null) { ok = false; break; } xs[k] = h[0]; ys[k] = h[1]; }
                if (ok && t0 + PASS_STATES - 1 <= t) { double off = DiamondLabel.offset(s.ox, s.oy, s.fireTurn - 1, s.speed, t0, xs, ys, s.rx, s.ry);
                    if (!Double.isNaN(off)) { double px = Math.toRadians(off) * s.range; labelErrorSum += Math.abs(px); labelSamples++; emit("label", t, "fireTurn", s.fireTurn, "px", px); } }
            }
        }
    }

    @Override public void onRoundStarted(RoundStartedEvent e) {
        ownHistory.clear(); enemyHistory.clear(); lockedS.clear(); intentShot = null; energyCorrection.clear(); shots.clear(); ownHitBonus.clear(); ownDamage.clear();
        enemyEnergy = -1; lastOwnEnergy = -1; enemyScanTurn = -1; axisReady = false; lastPlan = null; planner = null; pendingShot = null; lastEnemyPower = 1.95; lastFireTurn = 16; pendingFireTurn = -1;
        detected = shieldsFired = blocked = hitsTaken = skips = walls = botHits = shotsFired = ourHits = unmatchedRemovals = inactivityTurns = 0;
        skippedTurnList = new StringJoiner(",", "[", "]"); taken = dealt = energySpent = labelErrorSum = 0; maxMicros = 0; slowTurns = 0; labelSamples = 0; won = died = false;
        emit("round", 0, "round", e.getRoundNumber());
    }
    @Override public void onGameStarted(GameStartedEvent e) { arenaW = e.getGameSetup().getArenaWidth(); arenaH = e.getGameSetup().getArenaHeight(); }

    @Override public void onRoundEnded(RoundEndedEvent e) {
        if (e.getRoundNumber() <= roundsCompleted) return;
        roundsCompleted++;
        Map<String, Integer> outcomes = new TreeMap<>();
        for (Shot s : shots) outcomes.merge(s.outcome, 1, Integer::sum);
        String row = String.format(Locale.ROOT,
            "{\"schemaVersion\":1,\"name\":\"%s\",\"version\":\"%s\",\"mode\":\"train\",\"seed\":%d,\"round\":%d,\"roundsCompleted\":%d,\"turns\":%d,\"checkpointLoaded\":false,"
          + "\"shotsFired\":%d,\"bulletHits\":%d,\"damageDealt\":%.6f,\"damageTaken\":%.6f,\"energySpent\":%.6f,\"wallCollisions\":%d,\"botCollisions\":%d,\"skippedTurns\":%d,\"won\":%s,\"died\":%s,"
          + "\"enemyShotsDetected\":%d,\"shieldsFired\":%d,\"bulletsBlocked\":%d,\"hitsTaken\":%d,\"unmatchedRemovals\":%d,\"inactivityTurns\":%d,\"meanLabelError\":%.6f,\"maxDecisionMicros\":%d,\"slowTurns\":%d,\"skippedTurnNumbers\":%s,\"shotOutcomes\":%s}%n",
            NAME, VERSION, seed, e.getRoundNumber(), roundsCompleted, e.getTurnNumber(), shotsFired, ourHits, dealt, taken, energySpent, walls, botHits, skips, won, died,
            detected, shieldsFired, blocked, hitsTaken, unmatchedRemovals, inactivityTurns, labelSamples == 0 ? 0 : labelErrorSum / labelSamples, maxMicros, slowTurns, skippedTurnList, json(outcomes));
        try { if (metrics.getParent() != null) Files.createDirectories(metrics.getParent()); Files.writeString(metrics, row, StandardOpenOption.CREATE, StandardOpenOption.APPEND); } catch (IOException x) { throw new UncheckedIOException(x); }
        if (trace != null) trace.flush();
    }
    private static String json(Map<String, Integer> m) {
        StringJoiner j = new StringJoiner(",", "{", "}"); m.forEach((k, v) -> j.add("\"" + k + "\":" + v)); return j.toString();
    }

    private void emit(String event, int turn, Object... kv) {
        if (trace == null) return;
        StringBuilder b = new StringBuilder("{\"ev\":\"").append(event).append("\",\"turn\":").append(turn);
        for (int i = 0; i < kv.length; i += 2) {
            b.append(",\"").append(kv[i]).append("\":");
            Object v = kv[i + 1];
            if (v instanceof Double d) b.append(Double.isFinite(d) ? String.format(Locale.ROOT, "%.4f", d) : "null"); else b.append(v);
        }
        trace.println(b.append('}'));
    }
}
