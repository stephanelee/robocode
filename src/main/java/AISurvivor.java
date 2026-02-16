import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;

import java.io.*;
import java.util.*;

/**
 * AISurvivor — Robocode Tank Royale melee bot.
 *
 *  Q-Learning (TD-λ) with sparse eligibility traces
 *    Selects between anti-gravity, circling, fleeing, dodging, and hunting.
 *    Uses sparse eligibility traces (λ=0.65) — only visited state-action
 *    pairs are updated, avoiding a full sweep of the Q-table each turn.
 *    Persistent across battles via serialised QAgent on disk.
 *
 *  GuessFactor Targeting with per-round decay
 *    Records where the enemy actually is each time a targeting wave passes.
 *    Segmented by distance × lateral velocity (9 segments × 47 bins per enemy).
 *    Stats decay by 0.92 each round so recent behaviour is weighted higher.
 *
 *  Bullet-Aware Anti-Gravity Movement
 *    Wall repulsion + enemy repulsion + centre attraction + incoming bullet
 *    repulsion (perpendicular to bullet path for evasion).
 *
 *  Melee Radar Sweep / 1v1 Narrow Lock
 *    In melee, spins radar for full coverage; in 1v1, tight oscillating lock.
 *
 *  Wall Smoothing
 *    Every movement strategy adjusts its target heading away from walls.
 */
public class AISurvivor extends Bot {

    // ─── GuessFactor constants ────────────────────────────────────────────────
    private static final int    GF_BINS   = 47;
    private static final int    GF_MID    = GF_BINS / 2;
    private static final int    DIST_SEGS = 3;
    private static final int    LAT_SEGS  = 3;
    private static final int    NUM_SEGS  = DIST_SEGS * LAT_SEGS;
    private static final double WALL_MARGIN = 80.0;
    private static final double GF_DECAY  = 0.92;

    // ─── Actions ──────────────────────────────────────────────────────────────
    private static final int NUM_ACTIONS   = 6;
    private static final int A_ANTIGRAVITY = 0;
    private static final int A_CIRCLE_CW   = 1;
    private static final int A_CIRCLE_CCW  = 2;
    private static final int A_FLEE        = 3;
    private static final int A_DODGE       = 4;
    private static final int A_HUNT        = 5;

    // ─── State space (unchanged for Q-table compatibility) ───────────────────
    private static final int SS_ENERGY = 5;
    private static final int SS_DIST   = 4;
    private static final int SS_ADVAN  = 3;
    private static final int SS_COUNT  = 4;
    private static final int SS_WALL   = 3;
    private static final int NUM_STATES = SS_ENERGY * SS_DIST * SS_ADVAN * SS_COUNT * SS_WALL;

    // ─── Anti-gravity forces ─────────────────────────────────────────────────
    private static final double WALL_REPULSE   = 30_000.0;
    private static final double ENEMY_REPULSE  = 10_000.0;
    private static final double CENTER_ATTRACT =    600.0;
    private static final double BULLET_REPULSE = 15_000.0;

    // ─── Misc constants ─────────────────────────────────────────────────────
    private static final double SURVIVAL_BONUS = 0.1;
    private static final int    STALE_TURNS    = 40;
    private static final String AGENT_FILE     = "AISurvivor-qtable.dat";

    // ═════════════════════════════════════════════════════════════════════════
    //  Inner classes
    // ═════════════════════════════════════════════════════════════════════════

    private static final class Enemy {
        final int id;
        double x, y;
        double direction;
        double speed;
        double energy;
        double lateralVelocity;
        double prevEnergy;
        double turnRate;
        int    lastScanTurn;
        boolean justFired;
        double  firedPower;

        Enemy(int id, double x, double y, double dir, double spd,
              double en, double latVel, double prevEn, double tr, int turn) {
            this.id = id; this.x = x; this.y = y;
            this.direction = dir; this.speed = spd; this.energy = en;
            this.lateralVelocity = latVel; this.prevEnergy = prevEn;
            this.turnRate = tr; this.lastScanTurn = turn;

            double drop = prevEn - en;
            if (drop >= 0.09 && drop <= 3.01) {
                justFired = true;
                firedPower = Math.round(drop * 10.0) / 10.0;
            }
        }
    }

    private static final class TargetingWave {
        double   originX, originY;
        double   baseAngle;
        double   bulletSpeed;
        int      fireTurn;
        int      targetId;
        double   lateralDir;
        int      segment;
        double[] bins;
    }

    /**
     * Q-Learning agent with sparse TD(λ) eligibility traces.
     *
     * <p>Only visited state-action pairs carry traces, stored in a HashMap.
     * Entries below TRACE_MIN are pruned each step, keeping the update cost
     * proportional to the number of recently visited pairs (~10-30) rather
     * than the full state-action space (4,320).
     */
    static final class QAgent implements Serializable {
        private static final long serialVersionUID = 3L;

        private static final double ALPHA     = 0.15;
        private static final double GAMMA     = 0.95;
        private static final double LAMBDA    = 0.65;
        private static final double EPS_INIT  = 0.40;
        private static final double EPS_MIN   = 0.05;
        private static final double EPS_DECAY = 0.992;
        private static final double TRACE_MIN = 1e-4;

        final int numStates, numActions;
        double[][] Q;
        double     epsilon;
        int        totalRounds;

        transient HashMap<Long, Double> traces;
        transient int lastState  = -1;
        transient int lastAction = -1;

        QAgent(int states, int actions) {
            numStates = states; numActions = actions;
            Q = new double[states][actions];
            epsilon = EPS_INIT;
        }

        void startEpisode() {
            if (traces == null) traces = new HashMap<>();
            else traces.clear();
            lastState = lastAction = -1;
        }

        int observe(int state, double reward) {
            if (lastState >= 0) {
                double maxNext = Q[state][0];
                for (int a = 1; a < numActions; a++)
                    if (Q[state][a] > maxNext) maxNext = Q[state][a];

                double delta = reward + GAMMA * maxNext - Q[lastState][lastAction];

                traces.merge(traceKey(lastState, lastAction), 1.0, Double::sum);

                Iterator<Map.Entry<Long, Double>> it = traces.entrySet().iterator();
                while (it.hasNext()) {
                    Map.Entry<Long, Double> entry = it.next();
                    long k = entry.getKey();
                    int s = (int)(k >> 16);
                    int a = (int)(k & 0xFFFF);
                    double trace = entry.getValue();

                    Q[s][a] += ALPHA * delta * trace;
                    trace *= GAMMA * LAMBDA;

                    if (trace < TRACE_MIN) it.remove();
                    else entry.setValue(trace);
                }
            }

            int action;
            if (Math.random() < epsilon) {
                action = (int)(Math.random() * numActions);
            } else {
                action = 0;
                for (int a = 1; a < numActions; a++)
                    if (Q[state][a] > Q[state][action]) action = a;
            }

            lastState  = state;
            lastAction = action;
            return action;
        }

        void onDeath() {
            if (lastState < 0) return;
            traces.merge(traceKey(lastState, lastAction), 1.0, Double::sum);
            double delta = -100.0 - Q[lastState][lastAction];
            for (Map.Entry<Long, Double> entry : traces.entrySet()) {
                long k = entry.getKey();
                int s = (int)(k >> 16);
                int a = (int)(k & 0xFFFF);
                Q[s][a] += ALPHA * delta * entry.getValue();
            }
            traces.clear();
        }

        void endEpisode() {
            totalRounds++;
            epsilon = Math.max(EPS_MIN, epsilon * EPS_DECAY);
        }

        private static long traceKey(int state, int action) {
            return ((long)state << 16) | action;
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Fields
    // ═════════════════════════════════════════════════════════════════════════

    private final Map<Integer, Enemy>        enemies        = new HashMap<>();
    private final Map<Integer, double[][]>   gfStats        = new HashMap<>();
    private final List<TargetingWave>        targetingWaves = new ArrayList<>();
    private final List<BulletState>          incomingBullets = new ArrayList<>();

    private int    radarTargetId = -1;
    private int    radarSign     = 1;
    private int    dodgeSign     = 1;
    private double pendingReward = 0.0;

    private QAgent agent;

    // ═════════════════════════════════════════════════════════════════════════
    //  Entry point
    // ═════════════════════════════════════════════════════════════════════════

    public static void main(String[] args) { new AISurvivor().start(); }

    public AISurvivor() {
        super(BotInfo.fromFile("AISurvivor.json"));
        loadAgent();
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Main run loop  (one iteration = one game turn)
    // ═════════════════════════════════════════════════════════════════════════

    @Override
    public void run() {
        setAdjustGunForBodyTurn(true);
        setAdjustRadarForGunTurn(true);

        while (isRunning()) {
            pruneStaleEnemies();
            advanceTargetingWaves();

            int state  = encodeState();
            int action = agent.observe(state, pendingReward + SURVIVAL_BONUS);
            pendingReward = 0.0;

            executeMovement(action);
            updateRadar();
            aimGF();
            go();
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Lifecycle
    // ═════════════════════════════════════════════════════════════════════════

    @Override
    public void onRoundStarted(RoundStartedEvent e) {
        enemies.clear();
        targetingWaves.clear();
        incomingBullets.clear();
        pendingReward = 0.0;
        dodgeSign     = 1;
        radarTargetId = -1;
        radarSign     = 1;
        agent.startEpisode();

        // Decay GF stats so recent rounds weigh more
        for (double[][] segs : gfStats.values())
            for (double[] bins : segs)
                for (int i = 0; i < bins.length; i++)
                    bins[i] *= GF_DECAY;
    }

    @Override
    public void onRoundEnded(RoundEndedEvent e) {
        agent.endEpisode();
        saveAgent();
    }

    @Override
    public void onGameEnded(GameEndedEvent e) { saveAgent(); }

    @Override
    public void onTick(TickEvent e) {
        incomingBullets.clear();
        for (BulletState b : e.getBulletStates())
            if (enemies.containsKey(b.getOwnerId())) incomingBullets.add(b);
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Combat events
    // ═════════════════════════════════════════════════════════════════════════

    @Override
    public void onScannedBot(ScannedBotEvent e) {
        Enemy prev = enemies.get(e.getScannedBotId());
        double prevEn   = (prev != null) ? prev.energy    : e.getEnergy();
        double turnRate = (prev != null)
                ? normalizeAngle(e.getDirection() - prev.direction) : 0.0;

        double bearingToEnemy = directionTo(e.getX(), e.getY());
        double latVel = e.getSpeed() *
                Math.sin(Math.toRadians(e.getDirection() - bearingToEnemy));

        Enemy info = new Enemy(
                e.getScannedBotId(), e.getX(), e.getY(),
                e.getDirection(), e.getSpeed(), e.getEnergy(),
                latVel, prevEn, turnRate, getTurnNumber());
        enemies.put(info.id, info);
        gfStats.computeIfAbsent(info.id, k -> new double[NUM_SEGS][GF_BINS]);
    }

    @Override
    public void onHitByBullet(HitByBulletEvent e) {
        pendingReward -= e.getDamage() * 0.8;
        dodgeSign = -dodgeSign;
    }

    @Override
    public void onBulletHit(BulletHitBotEvent e) {
        pendingReward += e.getDamage() * 0.6;
        if (e.getEnergy() <= 0) {
            enemies.remove(e.getVictimId());
            pendingReward += 30.0;
        }
    }

    @Override
    public void onBotDeath(BotDeathEvent e) {
        enemies.remove(e.getVictimId());
        pendingReward += 15.0;
        if (e.getVictimId() == radarTargetId) radarTargetId = -1;
    }

    @Override
    public void onDeath(DeathEvent e) { agent.onDeath(); }

    @Override
    public void onHitWall(HitWallEvent e) {
        pendingReward -= 3.0;
        dodgeSign = -dodgeSign;
    }

    @Override
    public void onHitBot(HitBotEvent e) {
        pendingReward -= e.isRammed() ? 0.5 : 2.0;
        dodgeSign = -dodgeSign;
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  State encoding
    // ═════════════════════════════════════════════════════════════════════════

    private int encodeState() {
        double myEnergy = getEnergy();
        int eBucket = myEnergy < 15 ? 0 : myEnergy < 30 ? 1 : myEnergy < 50 ? 2 : myEnergy < 70 ? 3 : 4;

        Enemy t = bestTarget();
        int dBucket, advBucket;
        if (t == null) {
            dBucket = 3; advBucket = 1;
        } else {
            double d = distanceTo(t.x, t.y);
            dBucket   = d < 150 ? 0 : d < 300 ? 1 : d < 500 ? 2 : 3;
            double adv = myEnergy - t.energy;
            advBucket = adv < -15 ? 0 : adv > 15 ? 2 : 1;
        }

        int cnt = getEnemyCount();
        int cBucket = cnt <= 1 ? 0 : cnt == 2 ? 1 : cnt == 3 ? 2 : 3;

        double wall = minWallDist();
        int wBucket = wall < 70 ? 2 : wall < 160 ? 1 : 0;

        return eBucket
             + SS_ENERGY * (dBucket
             + SS_DIST   * (advBucket
             + SS_ADVAN  * (cBucket
             + SS_COUNT  * wBucket)));
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  GuessFactor Targeting
    // ═════════════════════════════════════════════════════════════════════════

    private void advanceTargetingWaves() {
        Iterator<TargetingWave> it = targetingWaves.iterator();
        while (it.hasNext()) {
            TargetingWave w = it.next();
            double traveled = (getTurnNumber() - w.fireTurn) * w.bulletSpeed;

            Enemy t = enemies.get(w.targetId);
            if (t == null) { it.remove(); continue; }

            double dist = Math.sqrt(sq(t.x - w.originX) + sq(t.y - w.originY));
            if (traveled < dist - w.bulletSpeed) continue;

            double absAngle = Math.toDegrees(Math.atan2(t.x - w.originX, t.y - w.originY));
            double latAngle = normalizeAngle(absAngle - w.baseAngle) * w.lateralDir;
            double maxAngle = Math.toDegrees(Math.asin(Math.min(1.0, 8.0 / w.bulletSpeed)));
            double gf       = clamp(latAngle / maxAngle, -1.0, 1.0);
            w.bins[gfBin(gf)] += 1.0;
            it.remove();
        }
    }

    private void aimGF() {
        Enemy t = bestTarget();
        if (t == null) return;

        double dist = distanceTo(t.x, t.y);
        double fp   = firePower(dist);
        double bspd = calcBulletSpeed(fp > 0 ? fp : 1.0);

        int        seg     = segment(dist, Math.abs(t.lateralVelocity));
        double[][] st      = gfStats.get(t.id);
        int        bestBin = GF_MID;
        if (st != null) {
            double[] bins = st[seg];
            double   max  = -1;
            for (int i = 0; i < GF_BINS; i++) if (bins[i] > max) { max = bins[i]; bestBin = i; }
        }

        double gf       = (bestBin / (double)(GF_BINS - 1)) * 2.0 - 1.0;
        double maxAngle = Math.toDegrees(Math.asin(Math.min(1.0, 8.0 / bspd)));
        double latDir   = Math.signum(t.lateralVelocity);
        if (latDir == 0) latDir = 1;

        double ticks = dist / bspd;
        double predX, predY;
        if (Math.abs(t.turnRate) < 0.1) {
            predX = t.x + Math.sin(Math.toRadians(t.direction)) * t.speed * ticks;
            predY = t.y + Math.cos(Math.toRadians(t.direction)) * t.speed * ticks;
        } else {
            double trRad  = Math.toRadians(t.turnRate);
            double radius = t.speed / trRad;
            double angle  = Math.toRadians(t.direction);
            predX = t.x + radius * (Math.sin(angle + trRad * ticks) - Math.sin(angle));
            predY = t.y + radius * (Math.cos(angle) - Math.cos(angle + trRad * ticks));
        }
        predX = clamp(predX, 18, getArenaWidth()  - 18);
        predY = clamp(predY, 18, getArenaHeight() - 18);
        double baseAngle = directionTo(predX, predY);
        double aimAngle  = baseAngle + latDir * gf * maxAngle;

        double gunTurn = normalizeAngle(aimAngle - getGunDirection());
        setGunTurnRate(clamp(gunTurn, -20.0, 20.0));

        if (fp > 0 && getGunHeat() == 0 && Math.abs(gunTurn) < 10.0) {
            setFire(fp);
            if (st != null) {
                TargetingWave tw = new TargetingWave();
                tw.originX    = getX();    tw.originY    = getY();
                tw.baseAngle  = baseAngle; tw.bulletSpeed = bspd;
                tw.fireTurn   = getTurnNumber(); tw.targetId = t.id;
                tw.lateralDir = latDir;    tw.segment    = seg;
                tw.bins       = st[seg];
                targetingWaves.add(tw);
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Movement
    // ═════════════════════════════════════════════════════════════════════════

    private void executeMovement(int action) {
        switch (action) {
            case A_ANTIGRAVITY -> doAntiGravity(1.0);
            case A_CIRCLE_CW   -> doCircle(true);
            case A_CIRCLE_CCW  -> doCircle(false);
            case A_FLEE        -> doAntiGravity(2.5);
            case A_DODGE       -> doDodge();
            case A_HUNT        -> doHunt();
            default            -> doAntiGravity(1.0);
        }
    }

    private void doAntiGravity(double repulseScale) {
        double[] force   = antiGravForce(repulseScale);
        double targetDir = Math.toDegrees(Math.atan2(force[0], force[1]));
        double smoothed  = wallSmooth(getX(), getY(), normDir(targetDir), 1);
        steerTo(smoothed, 8.0);
    }

    private double[] antiGravForce(double repulseScale) {
        double x = getX(), y = getY();
        double fx = 0, fy = 0;

        // Wall repulsion
        fx += WALL_REPULSE / sq(x + 1);
        fx -= WALL_REPULSE / sq(getArenaWidth()  - x + 1);
        fy += WALL_REPULSE / sq(y + 1);
        fy -= WALL_REPULSE / sq(getArenaHeight() - y + 1);

        // Enemy repulsion
        for (Enemy e : enemies.values()) {
            double dx = x - e.x, dy = y - e.y;
            double d  = Math.sqrt(dx * dx + dy * dy) + 1.0;
            double mag = ENEMY_REPULSE * repulseScale / (d * d);
            fx += mag * (dx / d); fy += mag * (dy / d);
        }

        // Incoming bullet repulsion — push perpendicular to bullet path
        for (BulletState b : incomingBullets) {
            double dx = x - b.getX(), dy = y - b.getY();
            double d = Math.sqrt(dx * dx + dy * dy) + 1.0;
            if (d < 250) {
                double bearToMe  = Math.toDegrees(Math.atan2(dx, dy));
                double angleDiff = Math.abs(normalizeAngle(b.getDirection() - bearToMe));
                if (angleDiff < 60.0) {
                    double mag = BULLET_REPULSE / (d * d);
                    double perpDir = Math.toRadians(b.getDirection() + 90.0 * dodgeSign);
                    fx += mag * Math.sin(perpDir);
                    fy += mag * Math.cos(perpDir);
                }
            }
        }

        // Center attraction
        double cx = getArenaWidth() / 2.0, cy = getArenaHeight() / 2.0;
        double cdx = cx - x, cdy = cy - y;
        double cd  = Math.sqrt(cdx * cdx + cdy * cdy) + 1.0;
        double cMag = CENTER_ATTRACT / (cd + 100.0);
        fx += cMag * (cdx / cd); fy += cMag * (cdy / cd);
        return new double[]{fx, fy};
    }

    private void doCircle(boolean cw) {
        Enemy t = nearestEnemy();
        if (t == null) { doAntiGravity(1.0); return; }
        double absOrbit = directionTo(t.x, t.y) + (cw ? 90.0 : -90.0);
        double smoothed = wallSmooth(getX(), getY(), normDir(absOrbit), cw ? 1 : -1);
        double speed = 8.0 * ((getTurnNumber() % 11 < 7) ? 1.0 : 0.5);
        steerTo(smoothed, speed);
    }

    private void doDodge() {
        BulletState threat = nearestThreat();
        if (threat != null) {
            double bulletDir = threat.getDirection();
            double perp1 = normDir(bulletDir + 90.0);
            double perp2 = normDir(bulletDir - 90.0);
            double h1 = wallSmooth(getX(), getY(), perp1,  1);
            double h2 = wallSmooth(getX(), getY(), perp2, -1);
            double t1 = Math.abs(normalizeAngle(h1 - getDirection()));
            double t2 = Math.abs(normalizeAngle(h2 - getDirection()));
            steerTo((t1 <= t2) ? h1 : h2, 8.0);
        } else {
            setTurnRate(maxBodyTurn() * dodgeSign);
            setTargetSpeed(8.0 * ((getTurnNumber() % 7 < 4) ? 1 : -1));
        }
    }

    private BulletState nearestThreat() {
        BulletState nearest = null;
        double minDist = Double.MAX_VALUE;
        double myX = getX(), myY = getY();
        for (BulletState b : incomingBullets) {
            double dx = myX - b.getX(), dy = myY - b.getY();
            double dist = Math.sqrt(dx * dx + dy * dy);
            double bearToMe  = Math.toDegrees(Math.atan2(dx, dy));
            double angleDiff = Math.abs(normalizeAngle(b.getDirection() - bearToMe));
            if (angleDiff < 45.0 && dist < minDist) { minDist = dist; nearest = b; }
        }
        return nearest;
    }

    private void doHunt() {
        Enemy t = weakestEnemy();
        if (t == null) { doAntiGravity(1.0); return; }
        double smoothed = wallSmooth(getX(), getY(), normDir(directionTo(t.x, t.y)), 1);
        steerTo(smoothed, 8.0);
    }

    /** Steer toward an absolute heading at the given speed. */
    private void steerTo(double heading, double speed) {
        double turn = normalizeAngle(heading - getDirection()) * 2.0;
        setTurnRate(clamp(turn, -maxBodyTurn(), maxBodyTurn()));
        setTargetSpeed(speed);
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Radar
    // ═════════════════════════════════════════════════════════════════════════

    private void updateRadar() {
        if (getEnemyCount() > 1) {
            // Melee: spin radar for coverage, slow near gun target
            Enemy target = bestTarget();
            if (target != null) {
                double radarTurn = normalizeAngle(
                        directionTo(target.x, target.y) - getRadarDirection());
                int staleness = getTurnNumber() - target.lastScanTurn;
                if (staleness <= 2 && Math.abs(radarTurn) < 20) {
                    setRadarTurnRate(45.0 * radarSign);
                } else {
                    setRadarTurnRate(clamp(radarTurn + Math.signum(radarTurn) * 15.0, -45.0, 45.0));
                }
            } else {
                setRadarTurnRate(45.0);
            }
        } else {
            // 1v1: tight narrow-beam lock
            Enemy target = enemies.get(radarTargetId);
            if (target == null) {
                target = bestTarget();
                radarTargetId = (target != null) ? target.id : -1;
            }
            if (target == null) { setRadarTurnRate(45.0); return; }

            int staleness = getTurnNumber() - target.lastScanTurn;
            double radarTurn = normalizeAngle(
                    directionTo(target.x, target.y) - getRadarDirection());

            if (staleness == 0) {
                radarSign = -radarSign;
                setRadarTurnRate(clamp(radarTurn + radarSign * 8.0, -45.0, 45.0));
            } else {
                double overshoot = Math.min(10.0 + staleness * 7.0, 45.0);
                setRadarTurnRate(clamp(radarTurn + Math.signum(radarTurn) * overshoot, -45.0, 45.0));
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Persistence
    // ═════════════════════════════════════════════════════════════════════════

    private void loadAgent() {
        File f = new File(AGENT_FILE);
        if (f.exists()) {
            try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(f))) {
                Object obj = in.readObject();
                if (obj instanceof QAgent loaded
                        && loaded.numStates  == NUM_STATES
                        && loaded.numActions == NUM_ACTIONS) {
                    agent = loaded;
                    System.out.printf("[AISurvivor] Agent loaded. Rounds: %d  ε=%.3f%n",
                            agent.totalRounds, agent.epsilon);
                    return;
                }
                System.out.println("[AISurvivor] Agent shape mismatch — starting fresh.");
            } catch (Exception ex) {
                System.out.println("[AISurvivor] Agent load failed: " + ex.getMessage());
            }
        } else {
            System.out.println("[AISurvivor] No agent file — starting fresh.");
        }
        agent = new QAgent(NUM_STATES, NUM_ACTIONS);
    }

    private void saveAgent() {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(AGENT_FILE))) {
            out.writeObject(agent);
            System.out.printf("[AISurvivor] Agent saved. Rounds: %d  ε=%.3f%n",
                    agent.totalRounds, agent.epsilon);
        } catch (Exception ex) {
            System.out.println("[AISurvivor] Agent save failed: " + ex.getMessage());
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Enemy selection helpers
    // ═════════════════════════════════════════════════════════════════════════

    private Enemy nearestEnemy() {
        Enemy best = null; double minD = Double.MAX_VALUE;
        for (Enemy e : enemies.values()) {
            double d = distanceTo(e.x, e.y);
            if (d < minD) { minD = d; best = e; }
        }
        return best;
    }

    private Enemy weakestEnemy() {
        Enemy best = null; double minEn = Double.MAX_VALUE;
        for (Enemy e : enemies.values()) if (e.energy < minEn) { minEn = e.energy; best = e; }
        return best;
    }

    private Enemy bestTarget() {
        Enemy best = null; double minScore = Double.MAX_VALUE;
        for (Enemy e : enemies.values()) {
            double score = distanceTo(e.x, e.y) * 0.7 + e.energy * 1.5;
            if (score < minScore) { minScore = score; best = e; }
        }
        return best;
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Utilities
    // ═════════════════════════════════════════════════════════════════════════

    private void pruneStaleEnemies() {
        int now = getTurnNumber();
        enemies.values().removeIf(e -> now - e.lastScanTurn > STALE_TURNS);
    }

    private double minWallDist() {
        double x = getX(), y = getY();
        return Math.min(Math.min(x, getArenaWidth() - x), Math.min(y, getArenaHeight() - y));
    }

    private double maxBodyTurn() { return 10.0 - 0.75 * Math.abs(getSpeed()); }

    private double wallSmooth(double x, double y, double heading, int rotDir) {
        double h = heading;
        for (int i = 0; i < 36; i++) {
            double tx = x + Math.sin(Math.toRadians(h)) * WALL_MARGIN;
            double ty = y + Math.cos(Math.toRadians(h)) * WALL_MARGIN;
            if (tx > WALL_MARGIN && tx < getArenaWidth()  - WALL_MARGIN &&
                ty > WALL_MARGIN && ty < getArenaHeight() - WALL_MARGIN) break;
            h = normDir(h + rotDir * 5.0);
        }
        return h;
    }

    /** Normalize direction to [0, 360). */
    private static double normDir(double d) { return ((d % 360) + 360) % 360; }

    private static int segment(double dist, double latSpeed) {
        int dBucket = dist < 200 ? 0 : dist < 500 ? 1 : 2;
        int vBucket = latSpeed < 2.0 ? 0 : latSpeed < 5.0 ? 1 : 2;
        return dBucket * LAT_SEGS + vBucket;
    }

    private static int gfBin(double gf) {
        return (int) clamp(Math.round((gf + 1.0) / 2.0 * (GF_BINS - 1)), 0, GF_BINS - 1);
    }

    private double firePower(double dist) {
        double en = getEnergy();
        if (en < 12.0) return 0.0;
        double fp = clamp(700.0 / (dist + 50.0), 0.5, 3.0);
        if (en < 25.0)      fp = Math.min(fp, 0.8);
        else if (en < 50.0) fp = Math.min(fp, 1.5);
        if (getEnemyCount() > 3) fp = Math.min(fp, 1.5);
        return fp;
    }

    private static double normalizeAngle(double a) {
        while (a >  180.0) a -= 360.0;
        while (a < -180.0) a += 360.0;
        return a;
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static double sq(double v) { return v * v; }
}
