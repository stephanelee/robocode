import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;

import java.io.*;
import java.util.*;

/**
 * AISurvivor — Robocode Tank Royale melee bot.
 *
 *  Q-Learning (TD-λ)
 *    Selects between anti-gravity, circling, fleeing, dodging, and hunting.
 *    Uses eligibility traces (λ=0.65) so rewards propagate back through
 *    the last ~10 decisions, not just the immediately preceding one.
 *    Persistent across battles via serialised QAgent on disk.
 *
 *  GuessFactor Targeting
 *    Records where the enemy actually is each time a targeting wave passes
 *    (hits and misses).  Segmented by distance × lateral velocity
 *    (9 segments × 47 bins per enemy).
 *
 *  Wall Smoothing
 *    Every movement strategy adjusts its target heading away from walls
 *    before committing, preventing corner-grinding.
 *
 *  Radar Lock
 *    Locks on the current gun target with adaptive overshoot; sweeps when
 *    no target is known.
 */
public class AISurvivor extends Bot {

    // ─── GuessFactor constants ────────────────────────────────────────────────
    private static final int    GF_BINS   = 47;
    private static final int    GF_MID    = GF_BINS / 2;
    private static final int    DIST_SEGS = 3;
    private static final int    LAT_SEGS  = 3;
    private static final int    NUM_SEGS  = DIST_SEGS * LAT_SEGS; // 9 per enemy
    private static final double WALL_MARGIN = 80.0;

    // ─── Actions ──────────────────────────────────────────────────────────────
    private static final int NUM_ACTIONS   = 6;
    private static final int A_ANTIGRAVITY = 0;
    private static final int A_CIRCLE_CW   = 1;
    private static final int A_CIRCLE_CCW  = 2;
    private static final int A_FLEE        = 3;
    private static final int A_DODGE       = 4;
    private static final int A_HUNT        = 5;

    // ─── State space ──────────────────────────────────────────────────────────
    // Features: own energy (5) × dist to target (4) × energy advantage (3)
    //           × enemy count (4) × wall proximity (3) = 720 states
    private static final int SS_ENERGY = 5;
    private static final int SS_DIST   = 4;
    private static final int SS_ADVAN  = 3;  // energy advantage vs target
    private static final int SS_COUNT  = 4;
    private static final int SS_WALL   = 3;
    private static final int NUM_STATES = SS_ENERGY * SS_DIST * SS_ADVAN * SS_COUNT * SS_WALL;

    // ─── Misc constants ───────────────────────────────────────────────────────
    private static final double WALL_REPULSE    = 30_000.0;
    private static final double ENEMY_REPULSE   = 10_000.0;
    private static final double CENTER_ATTRACT  =    600.0;
    private static final double SURVIVAL_BONUS  =      0.1; // per turn reward for staying alive
    private static final int    STALE_TURNS     = 40;
    private static final String AGENT_FILE      = "AISurvivor-qtable.dat";

    // ═════════════════════════════════════════════════════════════════════════
    //  Inner classes
    // ═════════════════════════════════════════════════════════════════════════

    /** Last known state of one enemy bot. */
    private static final class EnemyInfo {
        int    id;
        double x, y, direction, speed, energy;
        double lateralVelocity;
        double prevEnergy;
        int    lastScanTurn;

        EnemyInfo(int id, double x, double y, double dir, double spd,
                  double en, double latVel, double prevEn, int turn) {
            this.id = id; this.x = x; this.y = y;
            this.direction = dir; this.speed = spd; this.energy = en;
            this.lateralVelocity = latVel; this.prevEnergy = prevEn;
            this.lastScanTurn = turn;
        }
    }

    /** A bullet WE fired; used to record GF statistics when it reaches the enemy. */
    private static final class TargetingWave {
        double   originX, originY;
        double   baseAngle;    // predicted bearing at fire time (linear prediction)
        double   bulletSpeed;
        int      fireTurn;
        int      targetId;
        double   lateralDir;
        int      segment;
        double[] bins;         // reference into gfStats — mutated on arrival
    }

    /**
     * Self-contained Q-Learning agent with TD(λ) eligibility traces.
     *
     * <p>Eligibility traces propagate the TD error backward through the
     * last ~10 state-action pairs, greatly speeding up credit assignment
     * compared to plain TD(0).
     *
     * <p>Serialisable so the Q-table and epsilon survive across battles.
     */
    static final class QAgent implements Serializable {
        private static final long serialVersionUID = 3L;

        private static final double ALPHA     = 0.15;  // learning rate
        private static final double GAMMA     = 0.95;  // discount factor
        private static final double LAMBDA    = 0.65;  // trace decay  (0 = TD(0), 1 = MC)
        private static final double EPS_INIT  = 0.40;
        private static final double EPS_MIN   = 0.05;
        private static final double EPS_DECAY = 0.992;

        final int numStates, numActions;
        double[][] Q;
        double     epsilon;
        int        totalRounds;

        // Transient — reset each episode, not persisted
        transient double[][] traces;
        transient int        lastState  = -1;
        transient int        lastAction = -1;

        QAgent(int states, int actions) {
            numStates = states; numActions = actions;
            Q         = new double[states][actions];
            epsilon   = EPS_INIT;
        }

        /** Reset eligibility traces at the start of each episode. */
        void startEpisode() {
            if (traces == null) traces = new double[numStates][numActions];
            else for (double[] row : traces) Arrays.fill(row, 0.0);
            lastState = lastAction = -1;
        }

        /**
         * Observe the current state + reward from the last action, run a TD(λ)
         * update, then select and return the next action (ε-greedy).
         *
         * <p>Call once per turn in the run loop.
         */
        int observe(int state, double reward) {
            if (lastState >= 0) {
                // Max Q-value for the current state (Q-learning target)
                double maxNext = Q[state][0];
                for (int a = 1; a < numActions; a++) if (Q[state][a] > maxNext) maxNext = Q[state][a];

                double delta = reward + GAMMA * maxNext - Q[lastState][lastAction];

                // Accumulate trace for the state-action we just left
                traces[lastState][lastAction] += 1.0;

                // Propagate TD error through all active traces
                for (int s = 0; s < numStates; s++) {
                    for (int a = 0; a < numActions; a++) {
                        if (traces[s][a] > 1e-9) {
                            Q[s][a]      += ALPHA * delta * traces[s][a];
                            traces[s][a] *= GAMMA * LAMBDA;
                        }
                    }
                }
            }

            // ε-greedy action selection
            int action;
            if (Math.random() < epsilon) {
                action = (int)(Math.random() * numActions);
            } else {
                action = 0;
                for (int a = 1; a < numActions; a++) if (Q[state][a] > Q[state][action]) action = a;
            }

            lastState  = state;
            lastAction = action;
            return action;
        }

        /** Apply a terminal penalty on death and zero out all traces. */
        void onDeath() {
            if (lastState < 0) return;
            traces[lastState][lastAction] += 1.0;
            double delta = -100.0 - Q[lastState][lastAction];
            for (int s = 0; s < numStates; s++)
                for (int a = 0; a < numActions; a++)
                    if (traces[s][a] > 1e-9) {
                        Q[s][a] += ALPHA * delta * traces[s][a];
                        traces[s][a] = 0.0;
                    }
        }

        /** Decay epsilon and increment round counter at the end of each episode. */
        void endEpisode() {
            totalRounds++;
            epsilon = Math.max(EPS_MIN, epsilon * EPS_DECAY);
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Fields
    // ═════════════════════════════════════════════════════════════════════════

    private final Map<Integer, EnemyInfo>   enemies        = new HashMap<>();
    private final Map<Integer, double[][]>  gfStats        = new HashMap<>();
    private final List<TargetingWave>       targetingWaves = new ArrayList<>();
    private final List<BulletState>         incomingBullets = new ArrayList<>();

    private int    radarTargetId = -1;
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

            // Observe current state + accumulated reward, get next action
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
        agent.startEpisode();
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
        EnemyInfo prev = enemies.get(e.getScannedBotId());
        double prevEn  = (prev != null) ? prev.energy : e.getEnergy();

        double bearingToEnemy = directionTo(e.getX(), e.getY());
        double latVel = e.getSpeed() *
                Math.sin(Math.toRadians(e.getDirection() - bearingToEnemy));

        EnemyInfo info = new EnemyInfo(
                e.getScannedBotId(), e.getX(), e.getY(),
                e.getDirection(), e.getSpeed(), e.getEnergy(),
                latVel, prevEn, getTurnNumber());
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

    /**
     * Encodes the current game state into a single integer index.
     *
     * Features (in order of least-to-most significant):
     *   eBucket   — own energy level (5 levels: critical → full)
     *   dBucket   — distance to primary target (4 levels)
     *   advBucket — energy advantage vs primary target (losing / even / winning)
     *   cBucket   — number of enemies alive (1 / 2 / 3 / 4+)
     *   wBucket   — wall proximity (far / medium / close)
     */
    private int encodeState() {
        double myEnergy = getEnergy();
        int eBucket = myEnergy < 15 ? 0 : myEnergy < 30 ? 1 : myEnergy < 50 ? 2 : myEnergy < 70 ? 3 : 4;

        EnemyInfo t = bestTarget();
        int dBucket, advBucket;
        if (t == null) {
            dBucket = 3; advBucket = 1; // no target: treat as far, even
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

            EnemyInfo t = enemies.get(w.targetId);
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
        EnemyInfo t = bestTarget();
        if (t == null) return;

        double dist = distanceTo(t.x, t.y);
        double fp   = firePower(dist);
        double bspd = calcBulletSpeed(fp > 0 ? fp : 1.0);

        int        seg    = segment(dist, Math.abs(t.lateralVelocity));
        double[][] st     = gfStats.get(t.id);
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

        // Linear prediction as the base; GF bins learn residual correction only
        double ticks   = dist / bspd;
        double predX   = clamp(t.x + Math.sin(Math.toRadians(t.direction)) * t.speed * ticks,
                18, getArenaWidth()  - 18);
        double predY   = clamp(t.y + Math.cos(Math.toRadians(t.direction)) * t.speed * ticks,
                18, getArenaHeight() - 18);
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
        double[] force    = antiGravForce(repulseScale);
        double targetDir  = Math.toDegrees(Math.atan2(force[0], force[1]));
        double smoothed   = wallSmooth(getX(), getY(), ((targetDir % 360) + 360) % 360, 1);
        setTurnRate(clamp(normalizeAngle(smoothed - getDirection()) * 2.0, -maxBodyTurn(), maxBodyTurn()));
        setTargetSpeed(8.0);
    }

    private double[] antiGravForce(double repulseScale) {
        double x = getX(), y = getY();
        double fx = 0, fy = 0;

        fx += WALL_REPULSE / sq(x + 1);
        fx -= WALL_REPULSE / sq(getArenaWidth()  - x + 1);
        fy += WALL_REPULSE / sq(y + 1);
        fy -= WALL_REPULSE / sq(getArenaHeight() - y + 1);

        for (EnemyInfo e : enemies.values()) {
            double dx = x - e.x, dy = y - e.y;
            double d  = Math.sqrt(dx * dx + dy * dy) + 1.0;
            double mag = ENEMY_REPULSE * repulseScale / (d * d);
            fx += mag * (dx / d); fy += mag * (dy / d);
        }

        double cx = getArenaWidth() / 2.0, cy = getArenaHeight() / 2.0;
        double cdx = cx - x, cdy = cy - y;
        double cd  = Math.sqrt(cdx * cdx + cdy * cdy) + 1.0;
        double cMag = CENTER_ATTRACT / (cd + 100.0);
        fx += cMag * (cdx / cd); fy += cMag * (cdy / cd);
        return new double[]{fx, fy};
    }

    private void doCircle(boolean cw) {
        EnemyInfo t = nearestEnemy();
        if (t == null) { doAntiGravity(1.0); return; }
        double absOrbit = directionTo(t.x, t.y) + (cw ? 90.0 : -90.0);
        double smoothed = wallSmooth(getX(), getY(), ((absOrbit % 360) + 360) % 360, cw ? 1 : -1);
        setTurnRate(clamp(normalizeAngle(smoothed - getDirection()) * 2.0, -maxBodyTurn(), maxBodyTurn()));
        setTargetSpeed(8.0);
    }

    private void doDodge() {
        BulletState threat = nearestThreat();
        if (threat != null) {
            double bulletDir = threat.getDirection();
            double perp1 = ((bulletDir + 90.0) % 360 + 360) % 360;
            double perp2 = ((bulletDir - 90.0) % 360 + 360) % 360;
            double h1 = wallSmooth(getX(), getY(), perp1,  1);
            double h2 = wallSmooth(getX(), getY(), perp2, -1);
            double t1 = Math.abs(normalizeAngle(h1 - getDirection()));
            double t2 = Math.abs(normalizeAngle(h2 - getDirection()));
            double heading = (t1 <= t2) ? h1 : h2;
            setTurnRate(clamp(normalizeAngle(heading - getDirection()) * 2.0, -maxBodyTurn(), maxBodyTurn()));
        } else {
            setTurnRate(maxBodyTurn() * dodgeSign);
        }
        setTargetSpeed(8.0 * ((getTurnNumber() % 7 < 4) ? 1 : -1));
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
        EnemyInfo t = weakestEnemy();
        if (t == null) { doAntiGravity(1.0); return; }
        double relBear = normalizeAngle(directionTo(t.x, t.y) - getDirection());
        setTurnRate(clamp(relBear * 2.0, -maxBodyTurn(), maxBodyTurn()));
        setTargetSpeed(8.0);
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Radar
    // ═════════════════════════════════════════════════════════════════════════

    private void updateRadar() {
        EnemyInfo target = enemies.get(radarTargetId);
        if (target == null) {
            target = bestTarget();
            radarTargetId = (target != null) ? target.id : -1;
        }
        if (target == null) { setRadarTurnRate(45.0); return; }

        int    staleness = getTurnNumber() - target.lastScanTurn;
        double radarTurn = normalizeAngle(directionTo(target.x, target.y) - getRadarDirection());
        double overshoot = (staleness == 0) ? 5.0 : Math.min(10.0 + staleness * 7.0, 45.0);
        setRadarTurnRate(clamp(radarTurn + Math.signum(radarTurn) * overshoot, -45.0, 45.0));
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

    private EnemyInfo nearestEnemy() {
        EnemyInfo best = null; double minD = Double.MAX_VALUE;
        for (EnemyInfo e : enemies.values()) {
            double d = distanceTo(e.x, e.y);
            if (d < minD) { minD = d; best = e; }
        }
        return best;
    }

    private EnemyInfo weakestEnemy() {
        EnemyInfo best = null; double minEn = Double.MAX_VALUE;
        for (EnemyInfo e : enemies.values()) if (e.energy < minEn) { minEn = e.energy; best = e; }
        return best;
    }

    private EnemyInfo bestTarget() {
        EnemyInfo best = null; double minScore = Double.MAX_VALUE;
        for (EnemyInfo e : enemies.values()) {
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
            h = ((h + rotDir * 5.0) % 360 + 360) % 360;
        }
        return h;
    }

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
