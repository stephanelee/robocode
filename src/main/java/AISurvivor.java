import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;

import java.io.*;
import java.util.*;

/**
 * AISurvivor v2 — Robocode Tank Royale melee bot.
 *
 * Improvements over v1
 * ─────────────────────
 *  GuessFactor Targeting
 *    Records the actual enemy position every time a targeting wave
 *    passes through it (not just on hits) to build a statistical model
 *    of how each enemy moves.  Segmented by distance × lateral velocity
 *    (9 segments × 47 bins per enemy).
 *
 *  Wave Surfing
 *    Detects enemy bullet fires via energy drops each scan.  Creates
 *    EnemyWave objects that carry a danger profile copied from that
 *    enemy's observed shooting pattern.  Each turn, evaluates 24
 *    candidate headings and steers toward the lowest-danger position
 *    across all active waves.  Danger profiles improve when hit.
 *
 *  Wall Smoothing
 *    Every movement strategy adjusts its target heading away from
 *    walls before committing, preventing the bot from grinding corners.
 *
 *  Q-Learning (kept from v1)
 *    Selects between wave surfing, anti-gravity, circling, fleeing, and
 *    hunting.  Action slot 4 (was "dodge") is now "wave surf"; existing
 *    Q-tables are forward-compatible.  Persistent across battles.
 *
 *  Focused radar (from v1)
 *    Locks on nearest enemy with overshoot oscillation; sweeps when
 *    no target is known.
 */
public class AISurvivor extends Bot {

    // ─── GuessFactor constants ────────────────────────────────────────────────
    private static final int    GF_BINS   = 47;
    private static final int    GF_MID    = GF_BINS / 2;   // 23 — direct aim
    private static final int    DIST_SEGS = 3;              // distance buckets
    private static final int    LAT_SEGS  = 3;              // lateral-vel buckets
    private static final int    NUM_SEGS  = DIST_SEGS * LAT_SEGS; // 9 per enemy
    private static final double WALL_MARGIN = 80.0;

    // ─── Q-Learning constants ─────────────────────────────────────────────────
    private static final double LEARNING_RATE = 0.15;
    private static final double DISCOUNT      = 0.95;
    private static final double EPSILON_INIT  = 0.40;
    private static final double EPSILON_MIN   = 0.05;
    private static final double EPSILON_DECAY = 0.992;
    private static final String QTABLE_FILE   = "AISurvivor-qtable.dat";

    // ─── State space ──────────────────────────────────────────────────────────
    private static final int S_ENERGY  = 4;
    private static final int S_DIST    = 5;
    private static final int S_BEARING = 8;
    private static final int S_COUNT   = 4;
    private static final int S_WALL    = 3;
    private static final int NUM_STATES =
            S_ENERGY * S_DIST * S_BEARING * S_COUNT * S_WALL; // 1 920

    // ─── Actions ─────────────────────────────────────────────────────────────
    private static final int NUM_ACTIONS   = 6;
    private static final int A_ANTIGRAVITY = 0;
    private static final int A_CIRCLE_CW   = 1;
    private static final int A_CIRCLE_CCW  = 2;
    private static final int A_FLEE        = 3;
    private static final int A_WAVE_SURF   = 4; // replaces v1 A_DODGE — same slot
    private static final int A_HUNT        = 5;

    // ─── Anti-gravity constants ───────────────────────────────────────────────
    private static final double WALL_REPULSE   = 30_000.0;
    private static final double ENEMY_REPULSE  = 10_000.0;
    private static final double CENTER_ATTRACT =    600.0;

    private static final int STALE_TURNS = 40;

    // ═════════════════════════════════════════════════════════════════════════
    //  Inner classes
    // ═════════════════════════════════════════════════════════════════════════

    /** Last known state of one enemy bot. */
    private static final class EnemyInfo {
        int    id;
        double x, y, direction, speed, energy;
        double lateralVelocity; // signed: component of velocity perpendicular to bearing
        double prevEnergy;      // energy on the previous scan (for bullet-fire detection)
        int    lastScanTurn;

        EnemyInfo(int id, double x, double y, double dir, double spd,
                  double en, double latVel, double prevEn, int turn) {
            this.id = id; this.x = x; this.y = y;
            this.direction = dir; this.speed = spd; this.energy = en;
            this.lateralVelocity = latVel; this.prevEnergy = prevEn;
            this.lastScanTurn = turn;
        }
    }

    /**
     * A bullet WE fired — travels toward the enemy.
     * When the wave radius reaches the enemy's position we record the actual
     * GuessFactor to update our targeting statistics.
     */
    private static final class TargetingWave {
        double   originX, originY;   // our position when we fired
        double   directAngle;        // absolute angle from origin to enemy at fire time
        double   bulletSpeed;
        int      fireTurn;
        int      targetId;
        double   lateralDir;         // sign of enemy's lateral velocity at fire time
        int      segment;            // which stats segment this belongs to
        double[] bins;               // reference to gfStats[targetId][segment] — mutated on hit
    }

    /**
     * A bullet AN ENEMY fired — travels toward us.
     * Carries a snapshot of that enemy's danger profile at the time of firing.
     * Used to evaluate where we should move to avoid the bullet.
     */
    private static final class EnemyWave {
        double   originX, originY;   // enemy position when they fired
        double   directAngle;        // absolute angle from origin to OUR position at fire time
        double   bulletSpeed;
        int      fireTurn;
        int      enemyId;
        double[] dangerBins;         // copy of surfStats[enemyId] at wave creation time
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Fields
    // ═════════════════════════════════════════════════════════════════════════

    // Enemy tracking
    private final Map<Integer, EnemyInfo> enemies = new HashMap<>();

    // GF targeting statistics: gfStats.get(id)[segment][gfBin]
    private final Map<Integer, double[][]> gfStats   = new HashMap<>();

    // Wave surfing danger profiles: surfStats.get(id)[gfBin]
    // Records which GF offsets each enemy tends to fire at us with.
    private final Map<Integer, double[]>   surfStats = new HashMap<>();

    // Active waves
    private final List<TargetingWave> targetingWaves = new ArrayList<>();
    private final List<EnemyWave>     enemyWaves     = new ArrayList<>();

    // Q-Learning
    private double[][] qTable;
    private double     epsilon;
    private int        totalRounds;
    private int        prevState  = -1;
    private int        prevAction = -1;
    private double     pendingReward = 0.0;
    private int        dodgeSign = 1;

    // ═════════════════════════════════════════════════════════════════════════
    //  Entry point
    // ═════════════════════════════════════════════════════════════════════════

    public static void main(String[] args) { new AISurvivor().start(); }

    public AISurvivor() {
        super(BotInfo.fromFile("AISurvivor.json"));
        loadQTable();
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
            advanceTargetingWaves();   // update GF stats
            advanceEnemyWaves();       // remove waves that passed us

            // Q-Learning update
            int state = computeState();
            if (prevState >= 0) {
                updateQ(prevState, prevAction, pendingReward + 0.05, state);
                pendingReward = 0.0;
            }
            int action = selectAction(state);
            prevState  = state;
            prevAction = action;

            executeMovement(action);
            updateRadar();
            aimGF();       // GuessFactor gun replaces linear targeting
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
        enemyWaves.clear();
        pendingReward = 0.0;
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
    public void onGameEnded(GameEndedEvent e) { saveQTable(); }

    // ═════════════════════════════════════════════════════════════════════════
    //  Combat events
    // ═════════════════════════════════════════════════════════════════════════

    @Override
    public void onScannedBot(ScannedBotEvent e) {
        EnemyInfo prev = enemies.get(e.getScannedBotId());
        double prevEn = (prev != null) ? prev.energy : e.getEnergy();

        // Lateral velocity: component of enemy movement perpendicular to the
        // line between us.  Positive = moving right relative to the bearing.
        double bearingToEnemy = directionTo(e.getX(), e.getY());
        double latVel = e.getSpeed() *
                Math.sin(Math.toRadians(e.getDirection() - bearingToEnemy));

        EnemyInfo info = new EnemyInfo(
                e.getScannedBotId(), e.getX(), e.getY(),
                e.getDirection(), e.getSpeed(), e.getEnergy(),
                latVel, prevEn, getTurnNumber());
        enemies.put(info.id, info);

        // Initialise statistics maps for new enemies
        gfStats  .computeIfAbsent(info.id, k -> new double[NUM_SEGS][GF_BINS]);
        surfStats.computeIfAbsent(info.id, k -> new double[GF_BINS]);

        // Detect bullet fire: an energy drop of 0.1–3.0 that isn't explained
        // by normal damage means the enemy fired a bullet of that power.
        double drop = prevEn - e.getEnergy();
        if (drop >= 0.1 && drop <= 3.01) {
            double bspeed = calcBulletSpeed(drop);

            // Angle from enemy's firing position toward our position right now
            double directAngle = Math.toDegrees(
                    Math.atan2(getX() - e.getX(), getY() - e.getY()));

            EnemyWave wave    = new EnemyWave();
            wave.originX      = e.getX();
            wave.originY      = e.getY();
            wave.directAngle  = directAngle;
            wave.bulletSpeed  = bspeed;
            wave.fireTurn     = getTurnNumber();
            wave.enemyId      = info.id;
            wave.dangerBins   = surfStats.get(info.id).clone(); // snapshot
            enemyWaves.add(wave);
        }
    }

    @Override
    public void onHitByBullet(HitByBulletEvent e) {
        pendingReward -= bulletDamage(e.getBullet().getPower()) * 0.8;
        dodgeSign = -dodgeSign;

        // Find the enemy wave closest to us right now from the shooter,
        // then record which GF bin the bullet came from to update the
        // danger profile so future surfing is more accurate.
        int shooterId = e.getBullet().getOwnerId();
        double[] stats = surfStats.get(shooterId);
        if (stats == null) return;

        EnemyWave closest = null;
        double    minDiff = Double.MAX_VALUE;
        for (EnemyWave w : enemyWaves) {
            if (w.enemyId != shooterId) continue;
            double traveled = (getTurnNumber() - w.fireTurn) * w.bulletSpeed;
            double myDist   = Math.sqrt(sq(getX() - w.originX) + sq(getY() - w.originY));
            double diff = Math.abs(traveled - myDist);
            if (diff < minDiff) { minDiff = diff; closest = w; }
        }
        if (closest == null) return;

        double absAngle     = Math.toDegrees(
                Math.atan2(getX() - closest.originX, getY() - closest.originY));
        double latAngle     = normalizeAngle(absAngle - closest.directAngle);
        double maxAngle     = Math.toDegrees(
                Math.asin(Math.min(1.0, 8.0 / closest.bulletSpeed)));
        double gf           = clamp(latAngle / maxAngle, -1.0, 1.0);
        int    bin          = gfBin(gf);

        // Decay all bins slightly so recent hits matter more, then increment
        for (int i = 0; i < stats.length; i++) stats[i] *= 0.97;
        stats[bin] += 1.0;
    }

    @Override
    public void onBulletHit(BulletHitBotEvent e) {
        pendingReward += bulletDamage(e.getBullet().getPower()) * 0.6;
        if (e.getEnergy() <= 0) {
            enemies.remove(e.getVictimId());
            pendingReward += 30.0;
        }
    }

    @Override
    public void onBotDeath(BotDeathEvent e) {
        enemies.remove(e.getVictimId());
        pendingReward += 15.0;
    }

    @Override
    public void onDeath(DeathEvent e) {
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
    //  GuessFactor Targeting
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * For each active targeting wave, check whether it has reached the enemy.
     * When it does, compute the actual GuessFactor (where the enemy really
     * was vs. where we aimed) and increment the corresponding bin.
     * This records data on every shot — hits AND misses.
     */
    private void advanceTargetingWaves() {
        Iterator<TargetingWave> it = targetingWaves.iterator();
        while (it.hasNext()) {
            TargetingWave w = it.next();
            double traveled = (getTurnNumber() - w.fireTurn) * w.bulletSpeed;

            EnemyInfo t = enemies.get(w.targetId);
            if (t == null) { it.remove(); continue; }

            double dist = Math.sqrt(sq(t.x - w.originX) + sq(t.y - w.originY));
            if (traveled < dist - w.bulletSpeed) continue; // not there yet

            // Wave has reached (or passed) the enemy — record actual GF
            double absAngle  = Math.toDegrees(
                    Math.atan2(t.x - w.originX, t.y - w.originY));
            double latAngle  = normalizeAngle(absAngle - w.directAngle) * w.lateralDir;
            double maxAngle  = Math.toDegrees(
                    Math.asin(Math.min(1.0, 8.0 / w.bulletSpeed)));
            double gf        = clamp(latAngle / maxAngle, -1.0, 1.0);
            w.bins[gfBin(gf)] += 1.0;
            it.remove();
        }
    }

    /**
     * Aim using the GF bin with the highest recorded hit count for this
     * enemy and segment.  Falls back to middle bin (= direct aim + linear
     * prediction) when no data exists yet.
     * Creates a new TargetingWave each time we fire.
     */
    private void aimGF() {
        EnemyInfo t = bestTarget();
        if (t == null) return;

        double dist  = distanceTo(t.x, t.y);
        double fp    = firePower(dist);
        double bspd  = calcBulletSpeed(fp > 0 ? fp : 1.0);

        // Pick the best-performing GF bin for this enemy and segment
        int    seg    = segment(dist, Math.abs(t.lateralVelocity));
        double[][] st = gfStats.get(t.id);
        int bestBin   = GF_MID; // default: direct aim
        if (st != null) {
            double[] bins = st[seg];
            double   max  = -1;
            for (int i = 0; i < GF_BINS; i++) {
                if (bins[i] > max) { max = bins[i]; bestBin = i; }
            }
        }

        // Convert best bin → aim angle offset
        double gf          = (bestBin / (double)(GF_BINS - 1)) * 2.0 - 1.0;
        double maxAngle    = Math.toDegrees(Math.asin(Math.min(1.0, 8.0 / bspd)));
        double latDir      = Math.signum(t.lateralVelocity);
        if (latDir == 0) latDir = 1;

        // Base the aim on a linear prediction of where the enemy will be
        double ticks = dist / bspd;
        double predX = clamp(t.x + Math.sin(Math.toRadians(t.direction)) * t.speed * ticks,
                18, getArenaWidth()  - 18);
        double predY = clamp(t.y + Math.cos(Math.toRadians(t.direction)) * t.speed * ticks,
                18, getArenaHeight() - 18);

        double aimAngle = directionTo(predX, predY) + latDir * gf * maxAngle;
        double gunTurn  = normalizeAngle(aimAngle - getGunDirection());
        setGunTurnRate(clamp(gunTurn, -20.0, 20.0));

        if (fp > 0 && getGunHeat() == 0 && Math.abs(gunTurn) < 10.0) {
            setFire(fp);

            // Record this targeting wave so we can learn from it
            if (st != null) {
                TargetingWave tw = new TargetingWave();
                tw.originX     = getX();
                tw.originY     = getY();
                tw.directAngle = directionTo(t.x, t.y);
                tw.bulletSpeed = bspd;
                tw.fireTurn    = getTurnNumber();
                tw.targetId    = t.id;
                tw.lateralDir  = latDir;
                tw.segment     = seg;
                tw.bins        = st[seg];
                targetingWaves.add(tw);
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Wave Surfing
    // ═════════════════════════════════════════════════════════════════════════

    /** Remove waves that have traveled well past our current position. */
    private void advanceEnemyWaves() {
        double myX = getX(), myY = getY();
        int    now = getTurnNumber();
        enemyWaves.removeIf(w -> {
            double traveled = (now - w.fireTurn) * w.bulletSpeed;
            double myDist   = Math.sqrt(sq(myX - w.originX) + sq(myY - w.originY));
            return traveled > myDist + 60;
        });
    }

    /**
     * Wave surfing movement.
     * Evaluates 24 candidate absolute headings (every 15°), applies wall
     * smoothing to each, sums danger across all active enemy waves, and
     * steers toward the safest one.
     * Falls back to anti-gravity when no waves are active.
     */
    private void doWaveSurf() {
        if (enemyWaves.isEmpty()) { doAntiGravity(1.0); return; }

        double myX = getX(), myY = getY();
        double bestDanger  = Double.MAX_VALUE;
        double bestHeading = getDirection();

        for (int i = 0; i < 24; i++) {
            double heading  = i * 15.0;
            double smoothed = wallSmooth(myX, myY, heading, 1);
            double danger   = totalDanger(myX, myY, smoothed);
            if (danger < bestDanger) { bestDanger = danger; bestHeading = smoothed; }
        }

        double turn = normalizeAngle(bestHeading - getDirection());
        setTurnRate(clamp(turn * 2.0, -maxBodyTurn(), maxBodyTurn()));
        setTargetSpeed(8.0);
    }

    /**
     * Sums the GF danger across all active enemy waves for a bot that moves
     * in {@code heading} from ({@code myX}, {@code myY}).
     * The bot's position is simulated forward until the wave would intersect.
     */
    private double totalDanger(double myX, double myY, double heading) {
        int    now    = getTurnNumber();
        double sinH   = Math.sin(Math.toRadians(heading));
        double cosH   = Math.cos(Math.toRadians(heading));
        double danger = 0;

        for (EnemyWave w : enemyWaves) {
            double traveled = (now - w.fireTurn) * w.bulletSpeed;
            double myDist   = Math.sqrt(sq(myX - w.originX) + sq(myY - w.originY));
            if (traveled > myDist) continue; // already passed

            // Simulate movement until wave catches us (max 30 turns)
            double simX = myX, simY = myY;
            int    ticks = Math.min(30, (int)((myDist - traveled) / w.bulletSpeed) + 1);
            for (int t = 0; t < ticks; t++) {
                simX = clamp(simX + sinH * 8.0, 18, getArenaWidth()  - 18);
                simY = clamp(simY + cosH * 8.0, 18, getArenaHeight() - 18);
            }

            double absAngle  = Math.toDegrees(
                    Math.atan2(simX - w.originX, simY - w.originY));
            double latAngle  = normalizeAngle(absAngle - w.directAngle);
            double maxAngle  = Math.toDegrees(
                    Math.asin(Math.min(1.0, 8.0 / w.bulletSpeed)));
            double gf        = clamp(latAngle / maxAngle, -1.0, 1.0);
            danger          += w.dangerBins[gfBin(gf)] + 1.0; // +1 base so empty bins aren't free
        }
        return danger;
    }

    /**
     * Iteratively adjusts {@code heading} by 5° in direction {@code rotDir}
     * until the projected position at {@code WALL_MARGIN} distance is safely
     * inside the arena.  Returns the adjusted heading (0–360).
     */
    private double wallSmooth(double x, double y, double heading, int rotDir) {
        double h = heading;
        for (int i = 0; i < 36; i++) {
            double testX = x + Math.sin(Math.toRadians(h)) * WALL_MARGIN;
            double testY = y + Math.cos(Math.toRadians(h)) * WALL_MARGIN;
            if (testX > WALL_MARGIN && testX < getArenaWidth()  - WALL_MARGIN &&
                testY > WALL_MARGIN && testY < getArenaHeight() - WALL_MARGIN) break;
            h = ((h + rotDir * 5.0) % 360 + 360) % 360;
        }
        return h;
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Movement strategies
    // ═════════════════════════════════════════════════════════════════════════

    private void executeMovement(int action) {
        switch (action) {
            case A_ANTIGRAVITY -> doAntiGravity(1.0);
            case A_CIRCLE_CW   -> doCircle(true);
            case A_CIRCLE_CCW  -> doCircle(false);
            case A_FLEE        -> doAntiGravity(2.5);
            case A_WAVE_SURF   -> doWaveSurf();
            case A_HUNT        -> doHunt();
            default            -> doAntiGravity(1.0);
        }
    }

    private void doAntiGravity(double repulseScale) {
        double[] force     = antiGravForce(repulseScale);
        double   targetDir = Math.toDegrees(Math.atan2(force[0], force[1]));
        double   smoothed  = wallSmooth(getX(), getY(), ((targetDir % 360) + 360) % 360, 1);
        double   turn      = normalizeAngle(smoothed - getDirection());
        setTurnRate(clamp(turn * 2.0, -maxBodyTurn(), maxBodyTurn()));
        setTargetSpeed(8.0);
    }

    private double[] antiGravForce(double repulseScale) {
        double x  = getX(), y = getY();
        double cx = getArenaWidth()  / 2.0;
        double cy = getArenaHeight() / 2.0;
        double fx = 0, fy = 0;

        fx += WALL_REPULSE / sq(x + 1);
        fx -= WALL_REPULSE / sq(getArenaWidth()  - x + 1);
        fy += WALL_REPULSE / sq(y + 1);
        fy -= WALL_REPULSE / sq(getArenaHeight() - y + 1);

        for (EnemyInfo e : enemies.values()) {
            double dx   = x - e.x, dy = y - e.y;
            double dist = Math.sqrt(dx * dx + dy * dy) + 1.0;
            double mag  = ENEMY_REPULSE * repulseScale / (dist * dist);
            fx += mag * (dx / dist);
            fy += mag * (dy / dist);
        }

        double cdx = cx - x, cdy = cy - y;
        double cd  = Math.sqrt(cdx * cdx + cdy * cdy) + 1.0;
        double cMag = CENTER_ATTRACT / (cd + 100.0);
        fx += cMag * (cdx / cd);
        fy += cMag * (cdy / cd);
        return new double[]{fx, fy};
    }

    private void doCircle(boolean cw) {
        EnemyInfo t = nearestEnemy();
        if (t == null) { doAntiGravity(1.0); return; }
        double absOrbit  = directionTo(t.x, t.y) + (cw ? 90.0 : -90.0);
        double smoothed  = wallSmooth(getX(), getY(), ((absOrbit % 360) + 360) % 360, cw ? 1 : -1);
        double turn      = normalizeAngle(smoothed - getDirection());
        setTurnRate(clamp(turn * 2.0, -maxBodyTurn(), maxBodyTurn()));
        setTargetSpeed(8.0);
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
        EnemyInfo target = nearestEnemy();
        if (target == null) {
            setRadarTurnRate(45.0);
        } else {
            double radarTurn = normalizeAngle(
                    directionTo(target.x, target.y) - getRadarDirection());
            setRadarTurnRate(clamp(radarTurn + Math.signum(radarTurn) * 10.0, -45.0, 45.0));
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Q-Learning
    // ═════════════════════════════════════════════════════════════════════════

    private int computeState() {
        double en = getEnergy();
        int eBucket = en < 20 ? 0 : en < 40 ? 1 : en < 70 ? 2 : 3;

        EnemyInfo nearest = nearestEnemy();
        int distBucket, bearBucket;
        if (nearest == null) {
            distBucket = 4; bearBucket = 0;
        } else {
            double d = distanceTo(nearest.x, nearest.y);
            distBucket = d < 100 ? 0 : d < 220 ? 1 : d < 380 ? 2 : d < 580 ? 3 : 4;
            double rel = normalizeAngle(directionTo(nearest.x, nearest.y) - getDirection());
            if (rel < 0) rel += 360.0;
            bearBucket = ((int)(rel / 45.0)) % 8;
        }

        int cnt     = getEnemyCount();
        int cBucket = cnt <= 1 ? 0 : cnt == 2 ? 1 : cnt == 3 ? 2 : 3;
        double wall = minWallDist();
        int wBucket = wall < 70 ? 2 : wall < 160 ? 1 : 0;

        return eBucket
             + S_ENERGY  * (distBucket
             + S_DIST    * (bearBucket
             + S_BEARING * (cBucket
             + S_COUNT   * wBucket)));
    }

    private int selectAction(int state) {
        if (Math.random() < epsilon) return (int)(Math.random() * NUM_ACTIONS);
        int best = 0;
        for (int a = 1; a < NUM_ACTIONS; a++) {
            if (qTable[state][a] > qTable[state][best]) best = a;
        }
        return best;
    }

    private void updateQ(int s, int a, double reward, int ns) {
        double maxNext = qTable[ns][0];
        for (int i = 1; i < NUM_ACTIONS; i++) if (qTable[ns][i] > maxNext) maxNext = qTable[ns][i];
        qTable[s][a] += LEARNING_RATE * (reward + DISCOUNT * maxNext - qTable[s][a]);
    }

    @SuppressWarnings("unchecked")
    private void loadQTable() {
        epsilon = EPSILON_INIT; qTable = new double[NUM_STATES][NUM_ACTIONS]; totalRounds = 0;
        File f = new File(QTABLE_FILE);
        if (!f.exists()) { System.out.println("[AISurvivor] Starting fresh."); return; }
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(f))) {
            double[][] loaded = (double[][]) in.readObject();
            epsilon = in.readDouble(); totalRounds = in.readInt();
            if (loaded.length == NUM_STATES && loaded[0].length == NUM_ACTIONS) {
                qTable = loaded;
                System.out.printf("[AISurvivor] Q-table loaded. Rounds: %d  e=%.3f%n",
                        totalRounds, epsilon);
            } else {
                System.out.println("[AISurvivor] Q-table shape mismatch — starting fresh.");
            }
        } catch (Exception ex) {
            System.out.println("[AISurvivor] Q-table load failed: " + ex.getMessage());
        }
    }

    private void saveQTable() {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(QTABLE_FILE))) {
            out.writeObject(qTable); out.writeDouble(epsilon); out.writeInt(totalRounds);
            System.out.printf("[AISurvivor] Q-table saved. Rounds: %d  e=%.3f%n",
                    totalRounds, epsilon);
        } catch (Exception ex) {
            System.out.println("[AISurvivor] Q-table save failed: " + ex.getMessage());
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
        for (EnemyInfo e : enemies.values()) { if (e.energy < minEn) { minEn = e.energy; best = e; } }
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

    /**
     * Maps (distance, lateralSpeed) to a segment index 0..NUM_SEGS-1.
     * Segments separate enemies that behave differently at different ranges
     * and movement speeds.
     */
    private static int segment(double dist, double latSpeed) {
        int dBucket = dist < 200 ? 0 : dist < 500 ? 1 : 2;
        int vBucket = latSpeed < 2.0 ? 0 : latSpeed < 5.0 ? 1 : 2;
        return dBucket * LAT_SEGS + vBucket;
    }

    /** Maps GF in [−1, +1] to a bin index in [0, GF_BINS−1]. */
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

    private static double bulletDamage(double p) {
        return 4.0 * p + (p > 1.0 ? 2.0 * (p - 1.0) : 0.0);
    }
}
