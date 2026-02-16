import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;

import java.io.*;
import java.util.*;

/**
 * AISurvivor — Robocode Tank Royale bot.
 *
 * Uses Q-Learning (tabular reinforcement learning) to improve its battle
 * strategy across rounds and battles, combined with two proven melee
 * heuristics that provide a strong baseline from the very first turn:
 *
 *   • Anti-gravity movement  — naturally repels the bot away from walls and
 *     enemies while weakly pulling it toward the arena centre, producing
 *     smooth, unpredictable evasion without hard-coded rules.
 *
 *   • Linear predictive targeting — leads moving targets so bullets arrive
 *     at the predicted position rather than the stale scan position.
 *
 * Q-Learning layer
 * ─────────────────
 * State space  : 1 920 states = energy(4) × dist(5) × bearing(8)
 *                               × enemyCount(4) × wallProximity(3)
 * Action space : 6 movement strategies (anti-gravity, circle CW/CCW,
 *                flee, dodge, hunt).
 * Reward       : +0.05/turn survival, +(bullet damage × 0.6) on hit,
 *                +30 kill, +15 enemy death, −(damage × 0.8) when hit,
 *                −100 own death, −3 wall hit.
 * Persistence  : Q-table is serialised to AISurvivor-qtable.dat in the
 *                bot's working directory and reloaded each battle, so
 *                the bot accumulates knowledge indefinitely.
 * Exploration  : ε-greedy, ε decays from 0.40 → 0.05 at rate 0.992/round.
 */
public class AISurvivor extends Bot {

    // ── Q-Learning hyperparameters ────────────────────────────────────────────
    private static final double LEARNING_RATE = 0.15;
    private static final double DISCOUNT      = 0.95;
    private static final double EPSILON_INIT  = 0.40;
    private static final double EPSILON_MIN   = 0.05;
    private static final double EPSILON_DECAY = 0.992;   // per-round
    private static final String QTABLE_FILE   = "AISurvivor-qtable.dat";

    // ── State-space dimensions ────────────────────────────────────────────────
    private static final int S_ENERGY  = 4;   // own energy: critical/low/medium/high
    private static final int S_DIST    = 5;   // nearest enemy distance (5 buckets)
    private static final int S_BEARING = 8;   // nearest enemy bearing (8 × 45°)
    private static final int S_COUNT   = 4;   // live enemies: 1 / 2 / 3 / 4+
    private static final int S_WALL    = 3;   // wall proximity: safe/caution/danger
    private static final int NUM_STATES =
            S_ENERGY * S_DIST * S_BEARING * S_COUNT * S_WALL;  // 1 920

    // ── Action indices ────────────────────────────────────────────────────────
    private static final int NUM_ACTIONS   = 6;
    private static final int A_ANTIGRAVITY = 0; // anti-gravity vector movement
    private static final int A_CIRCLE_CW   = 1; // orbit nearest enemy CW
    private static final int A_CIRCLE_CCW  = 2; // orbit nearest enemy CCW
    private static final int A_FLEE        = 3; // amplified anti-gravity (defensive)
    private static final int A_DODGE       = 4; // erratic perpendicular evasion
    private static final int A_HUNT        = 5; // charge weakest enemy

    // ── Anti-gravity force constants ──────────────────────────────────────────
    private static final double WALL_REPULSE   = 30_000.0;
    private static final double ENEMY_REPULSE  = 10_000.0;
    private static final double CENTER_ATTRACT =    600.0;

    // ── Enemy tracking ────────────────────────────────────────────────────────
    private static final int STALE_TURNS = 40; // discard scans older than this

    // ── Inner record: last known state of one enemy bot ──────────────────────
    private static final class EnemyInfo {
        final int id;
        double x, y, direction, speed, energy;
        int    lastScanTurn;

        EnemyInfo(int id, double x, double y,
                  double dir, double spd, double en, int turn) {
            this.id = id;
            this.x = x; this.y = y;
            this.direction = dir; this.speed = spd; this.energy = en;
            this.lastScanTurn = turn;
        }
    }

    // ── Q-Learning state ──────────────────────────────────────────────────────
    private double[][] qTable;
    private double     epsilon;
    private int        totalRounds;
    private int        prevState  = -1;
    private int        prevAction = -1;
    private double     pendingReward = 0.0;

    // ── Per-round game state ──────────────────────────────────────────────────
    private final Map<Integer, EnemyInfo> enemies = new HashMap<>();
    private int dodgeSign = 1; // flips on every hit to break attacker's aim lock

    // ═════════════════════════════════════════════════════════════════════════
    //  Entry point
    // ═════════════════════════════════════════════════════════════════════════

    public static void main(String[] args) {
        new AISurvivor().start();
    }

    public AISurvivor() {
        super(BotInfo.fromFile("AISurvivor.json"));
        loadQTable();
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Main run loop  (one iteration = one game turn)
    // ═════════════════════════════════════════════════════════════════════════

    @Override
    public void run() {
        // Decouple component rotations so gun/radar move independently of body
        setAdjustGunForBodyTurn(true);
        setAdjustRadarForGunTurn(true);

        while (isRunning()) {

            // 1. Drop enemy entries that are too old to be reliable
            pruneStaleEnemies();

            // 2. Q-Learning: update Q-value for the action taken last turn
            int state = computeState();
            if (prevState >= 0) {
                updateQ(prevState, prevAction, pendingReward + 0.05, state);
                pendingReward = 0.0;
            }

            // 3. Choose this turn's action (ε-greedy)
            int action = selectAction(state);
            prevState  = state;
            prevAction = action;

            // 4. Execute movement strategy
            executeMovement(action);

            // 5. Radar: lock on nearest known enemy; sweep only when no target
            updateRadar();

            // 6. Aim gun and fire if aligned
            aimAndFire();

            // 7. Commit all queued commands for this turn
            go();
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Round / game lifecycle
    // ═════════════════════════════════════════════════════════════════════════

    @Override
    public void onRoundStarted(RoundStartedEvent e) {
        enemies.clear();
        pendingReward  = 0.0;
        prevState = prevAction = -1;
        dodgeSign = 1;
    }

    @Override
    public void onRoundEnded(RoundEndedEvent e) {
        totalRounds++;
        epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);
        saveQTable();
    }

    @Override
    public void onGameEnded(GameEndedEvent e) {
        saveQTable();
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Combat events — update enemy map and accumulate rewards
    // ═════════════════════════════════════════════════════════════════════════

    @Override
    public void onScannedBot(ScannedBotEvent e) {
        enemies.put(e.getScannedBotId(), new EnemyInfo(
                e.getScannedBotId(),
                e.getX(), e.getY(),
                e.getDirection(), e.getSpeed(), e.getEnergy(),
                getTurnNumber()
        ));
    }

    @Override
    public void onHitByBullet(HitByBulletEvent e) {
        pendingReward -= bulletDamage(e.getBullet().getPower()) * 0.8;
        dodgeSign = -dodgeSign; // break the attacker's radar/gun lock
    }

    @Override
    public void onBulletHit(BulletHitBotEvent e) {
        pendingReward += bulletDamage(e.getBullet().getPower()) * 0.6;
        if (e.getEnergy() <= 0) {
            enemies.remove(e.getVictimId());
            pendingReward += 30.0; // kill bonus
        }
    }

    @Override
    public void onBotDeath(BotDeathEvent e) {
        enemies.remove(e.getVictimId());
        pendingReward += 15.0; // surviving as the field narrows is rewarded
    }

    @Override
    public void onDeath(DeathEvent e) {
        // Terminal Q-update: no future state, apply large penalty
        if (prevState >= 0 && prevAction >= 0) {
            double old = qTable[prevState][prevAction];
            qTable[prevState][prevAction] = old + LEARNING_RATE * (-100.0 - old);
        }
    }

    @Override
    public void onHitWall(HitWallEvent e) {
        pendingReward -= 3.0;
        dodgeSign = -dodgeSign;
    }

    @Override
    public void onHitBot(HitBotEvent e) {
        pendingReward -= 1.5;
        dodgeSign = -dodgeSign;
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Q-Learning core
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Encodes the current game state into a single integer index into qTable.
     *
     * Encoding (base-mixed):
     *   index = eBucket
     *         + S_ENERGY  * (distBucket
     *         + S_DIST    * (bearBucket
     *         + S_BEARING * (cntBucket
     *         + S_COUNT   * wallBucket)))
     */
    private int computeState() {
        // Own energy bucket (0 = critical, 3 = healthy)
        double en = getEnergy();
        int eBucket = en < 20 ? 0 : en < 40 ? 1 : en < 70 ? 2 : 3;

        // Nearest scanned enemy: distance and relative bearing
        EnemyInfo nearest = nearestEnemy();
        int distBucket, bearBucket;
        if (nearest == null) {
            distBucket = 4; // "no data" sentinel bucket
            bearBucket = 0;
        } else {
            double d = distanceTo(nearest.x, nearest.y);
            distBucket = d < 100 ? 0
                       : d < 220 ? 1
                       : d < 380 ? 2
                       : d < 580 ? 3 : 4;

            // Relative bearing: absolute direction minus body heading
            double rel = normalizeAngle(directionTo(nearest.x, nearest.y) - getDirection());
            if (rel < 0) rel += 360.0;
            bearBucket = ((int)(rel / 45.0)) % 8;
        }

        // Live enemy count bucket
        int cnt     = getEnemyCount();
        int cBucket = cnt <= 1 ? 0 : cnt == 2 ? 1 : cnt == 3 ? 2 : 3;

        // Wall proximity bucket
        double wall  = minWallDist();
        int wBucket  = wall < 70 ? 2 : wall < 160 ? 1 : 0;

        return eBucket
             + S_ENERGY  * (distBucket
             + S_DIST    * (bearBucket
             + S_BEARING * (cBucket
             + S_COUNT   * wBucket)));
    }

    /** ε-greedy action selection. */
    private int selectAction(int state) {
        if (Math.random() < epsilon) {
            return (int)(Math.random() * NUM_ACTIONS);
        }
        int best = 0;
        for (int a = 1; a < NUM_ACTIONS; a++) {
            if (qTable[state][a] > qTable[state][best]) best = a;
        }
        return best;
    }

    /** Standard TD(0) Q-update. */
    private void updateQ(int s, int a, double reward, int ns) {
        double maxNext = qTable[ns][0];
        for (int i = 1; i < NUM_ACTIONS; i++) {
            if (qTable[ns][i] > maxNext) maxNext = qTable[ns][i];
        }
        double tdTarget = reward + DISCOUNT * maxNext;
        qTable[s][a] += LEARNING_RATE * (tdTarget - qTable[s][a]);
    }

    // ── Q-table persistence ───────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private void loadQTable() {
        epsilon     = EPSILON_INIT;
        totalRounds = 0;
        qTable      = new double[NUM_STATES][NUM_ACTIONS];  // zero-initialised

        File f = new File(QTABLE_FILE);
        if (!f.exists()) {
            System.out.println("[AISurvivor] No prior Q-table found — starting fresh.");
            return;
        }
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(f))) {
            double[][] loaded = (double[][]) in.readObject();
            epsilon     = in.readDouble();
            totalRounds = in.readInt();
            if (loaded.length == NUM_STATES && loaded[0].length == NUM_ACTIONS) {
                qTable = loaded;
                System.out.printf("[AISurvivor] Q-table loaded. Rounds trained: %d  ε=%.3f%n",
                        totalRounds, epsilon);
            } else {
                System.out.println("[AISurvivor] Q-table shape mismatch — starting fresh.");
            }
        } catch (Exception ex) {
            System.out.println("[AISurvivor] Could not load Q-table: " + ex.getMessage());
        }
    }

    private void saveQTable() {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(QTABLE_FILE))) {
            out.writeObject(qTable);
            out.writeDouble(epsilon);
            out.writeInt(totalRounds);
            System.out.printf("[AISurvivor] Q-table saved. Rounds: %d  ε=%.3f%n",
                    totalRounds, epsilon);
        } catch (Exception ex) {
            System.out.println("[AISurvivor] Could not save Q-table: " + ex.getMessage());
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Movement strategies
    // ═════════════════════════════════════════════════════════════════════════

    private void executeMovement(int action) {
        switch (action) {
            case A_ANTIGRAVITY -> doAntiGravity(1.0);
            case A_CIRCLE_CW   -> doCircle(true);
            case A_CIRCLE_CCW  -> doCircle(false);
            case A_FLEE        -> doAntiGravity(2.5);  // amplified repulsion
            case A_DODGE       -> doDodge();
            case A_HUNT        -> doHunt();
            default            -> doAntiGravity(1.0);
        }
    }

    /**
     * Anti-gravity movement.
     *
     * Computes a resultant force from wall repulsion, enemy repulsion, and
     * weak centre attraction, then steers the bot along that force vector.
     * The {@code repulseScale} parameter amplifies the enemy component,
     * allowing the same method to serve both normal and fleeing modes.
     */
    private void doAntiGravity(double repulseScale) {
        double[] force      = antiGravForce(repulseScale);
        // atan2(Fx, Fy) gives angle from North because North = +Y in Tank Royale
        double   targetDir  = Math.toDegrees(Math.atan2(force[0], force[1]));
        double   turn       = normalizeAngle(targetDir - getDirection());
        setTurnRate(clamp(turn * 2.0, -maxBodyTurn(), maxBodyTurn()));
        setTargetSpeed(8.0);
    }

    private double[] antiGravForce(double repulseScale) {
        double x  = getX(),  y  = getY();
        double cx = getArenaWidth()  / 2.0;
        double cy = getArenaHeight() / 2.0;
        double fx = 0.0, fy = 0.0;

        // Wall repulsion — each wall pushes the bot away with 1/d² decay
        fx += WALL_REPULSE / sq(x + 1);                         // left wall  → +x
        fx -= WALL_REPULSE / sq(getArenaWidth()  - x + 1);      // right wall → -x
        fy += WALL_REPULSE / sq(y + 1);                         // bottom     → +y
        fy -= WALL_REPULSE / sq(getArenaHeight() - y + 1);      // top        → -y

        // Enemy repulsion — each known enemy pushes the bot away
        for (EnemyInfo e : enemies.values()) {
            double dx   = x - e.x,  dy = y - e.y;
            double dist = Math.sqrt(dx * dx + dy * dy) + 1.0;
            double mag  = ENEMY_REPULSE * repulseScale / (dist * dist);
            fx += mag * (dx / dist);
            fy += mag * (dy / dist);
        }

        // Weak centre attraction — prevents camping in a corner
        double cdx  = cx - x,  cdy = cy - y;
        double cDist = Math.sqrt(cdx * cdx + cdy * cdy) + 1.0;
        double cMag  = CENTER_ATTRACT / (cDist + 100.0);
        fx += cMag * (cdx / cDist);
        fy += cMag * (cdy / cDist);

        return new double[]{fx, fy};
    }

    /** Orbit the nearest enemy perpendicularly to maintain engagement range. */
    private void doCircle(boolean clockwise) {
        EnemyInfo t = nearestEnemy();
        if (t == null) { doAntiGravity(1.0); return; }

        double relBear = normalizeAngle(directionTo(t.x, t.y) - getDirection());
        double orbit   = relBear + (clockwise ? 90.0 : -90.0);
        setTurnRate(clamp(orbit, -maxBodyTurn(), maxBodyTurn()));
        setTargetSpeed(8.0);
    }

    /** Erratic movement that is hard for enemy guns to track. */
    private void doDodge() {
        setTurnRate(maxBodyTurn() * dodgeSign);
        // Alternate forward/reverse on a prime-number cycle to be unpredictable
        setTargetSpeed(8.0 * ((getTurnNumber() % 7 < 4) ? 1 : -1));
    }

    /** Close in on the lowest-energy enemy to finish it off quickly. */
    private void doHunt() {
        EnemyInfo t = weakestEnemy();
        if (t == null) { doAntiGravity(1.0); return; }

        double relBear = normalizeAngle(directionTo(t.x, t.y) - getDirection());
        setTurnRate(clamp(relBear * 2.0, -maxBodyTurn(), maxBodyTurn()));
        setTargetSpeed(8.0);
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Targeting
    // ═════════════════════════════════════════════════════════════════════════

    // ═════════════════════════════════════════════════════════════════════════
    //  Radar
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Focused radar: lock on the nearest known enemy so we get a fresh scan
     * every 1-2 turns instead of once every ~8 turns with a full spin.
     * When no enemies are in the map (start of round or all stale), sweep at
     * max speed until something is found.
     */
    private void updateRadar() {
        EnemyInfo target = nearestEnemy();
        if (target == null) {
            // No data — sweep at full speed to acquire a target
            setRadarTurnRate(45.0);
        } else {
            // Turn radar toward the last known position of the nearest enemy.
            // Add a small overshoot in the same direction so the radar
            // oscillates back and forth over the target, keeping it in the
            // scan arc every turn.
            double radarTurn = normalizeAngle(
                    directionTo(target.x, target.y) - getRadarDirection());
            double overshoot = Math.signum(radarTurn) * 10.0;
            setRadarTurnRate(clamp(radarTurn + overshoot, -45.0, 45.0));
        }
    }

    /**
     * Selects the best target, turns the gun toward the predicted impact
     * point using linear targeting, and fires when aligned.
     */
    private void aimAndFire() {
        EnemyInfo t = bestTarget();
        if (t == null) return;

        double fp = firePower(distanceTo(t.x, t.y));

        // Always track the target even when the gun is cooling
        turnGunToward(t, fp > 0 ? fp : 1.0);

        if (fp > 0 && getGunHeat() == 0) {
            // Fire only when the gun error is within a reasonable cone
            double gunErr = normalizeAngle(
                    directionTo(t.x, t.y) - getGunDirection());
            if (Math.abs(gunErr) < 9.0) {
                setFire(fp);
            }
        }
    }

    /**
     * Linear targeting: predicts where the enemy will be when the bullet
     * arrives and turns the gun toward that predicted position.
     */
    private void turnGunToward(EnemyInfo t, double fp) {
        double bulletSpd = calcBulletSpeed(fp);
        double dist      = distanceTo(t.x, t.y);
        double ticks     = dist / bulletSpd;

        // First-order linear prediction (one iteration is sufficient for this speed range)
        double predX = t.x + Math.sin(Math.toRadians(t.direction)) * t.speed * ticks;
        double predY = t.y + Math.cos(Math.toRadians(t.direction)) * t.speed * ticks;

        // Clamp to arena so we don't aim at a phantom position outside the walls
        predX = clamp(predX, 18, getArenaWidth()  - 18);
        predY = clamp(predY, 18, getArenaHeight() - 18);

        double absAim  = directionTo(predX, predY);
        double gunTurn = normalizeAngle(absAim - getGunDirection());
        setGunTurnRate(clamp(gunTurn, -20.0, 20.0));
    }

    /**
     * Computes firepower scaled by distance and own energy.
     * Returns 0 when energy is critically low (preserve for survival).
     */
    private double firePower(double dist) {
        double en = getEnergy();
        if (en < 12.0) return 0.0;                     // near-death: don't fire

        // Distance-scaled: 3.0 up close → 0.5 at range
        double fp = clamp(700.0 / (dist + 50.0), 0.5, 3.0);

        // Energy conservation tiers
        if (en < 25.0)      fp = Math.min(fp, 0.8);
        else if (en < 50.0) fp = Math.min(fp, 1.5);

        // In crowded melee, moderate firepower stretches energy further
        if (getEnemyCount() > 3) fp = Math.min(fp, 1.5);

        return fp;
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Enemy selection helpers
    // ═════════════════════════════════════════════════════════════════════════

    private EnemyInfo nearestEnemy() {
        EnemyInfo best = null;
        double minD = Double.MAX_VALUE;
        for (EnemyInfo e : enemies.values()) {
            double d = distanceTo(e.x, e.y);
            if (d < minD) { minD = d; best = e; }
        }
        return best;
    }

    private EnemyInfo weakestEnemy() {
        EnemyInfo best = null;
        double minEn = Double.MAX_VALUE;
        for (EnemyInfo e : enemies.values()) {
            if (e.energy < minEn) { minEn = e.energy; best = e; }
        }
        return best;
    }

    /** Combines distance and energy to rank targets: closer + weaker = better. */
    private EnemyInfo bestTarget() {
        EnemyInfo best = null;
        double minScore = Double.MAX_VALUE;
        for (EnemyInfo e : enemies.values()) {
            double score = distanceTo(e.x, e.y) * 0.7 + e.energy * 1.5;
            if (score < minScore) { minScore = score; best = e; }
        }
        return best;
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Utilities
    // ═════════════════════════════════════════════════════════════════════════

    /** Remove enemy entries not refreshed within STALE_TURNS. */
    private void pruneStaleEnemies() {
        int now = getTurnNumber();
        enemies.values().removeIf(e -> now - e.lastScanTurn > STALE_TURNS);
    }

    /** Distance to the nearest arena wall. */
    private double minWallDist() {
        double x = getX(), y = getY();
        return Math.min(
                Math.min(x, getArenaWidth()  - x),
                Math.min(y, getArenaHeight() - y));
    }

    /**
     * Maximum body turn rate this turn (speed-dependent per game physics).
     * Formula: 10 − 0.75 × |speed|
     */
    private double maxBodyTurn() {
        return 10.0 - 0.75 * Math.abs(getSpeed());
    }

    /** Normalise angle to (−180, +180]. */
    private static double normalizeAngle(double a) {
        while (a >  180.0) a -= 360.0;
        while (a < -180.0) a += 360.0;
        return a;
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static double sq(double v) { return v * v; }

    /** Bullet damage formula from the game physics spec. */
    private static double bulletDamage(double power) {
        return 4.0 * power + (power > 1.0 ? 2.0 * (power - 1.0) : 0.0);
    }
}
