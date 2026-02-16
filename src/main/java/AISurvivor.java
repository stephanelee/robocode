import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;

import java.io.*;
import java.util.*;

/**
 * AISurvivor — Robocode Tank Royale melee bot (v3 — DQN overhaul).
 *
 *  DQN with experience replay + target network
 *    14 continuous features → 32-neuron hidden layer → 6 action Q-values.
 *    Boltzmann exploration, mini-batch replay (5000 buffer), target net sync.
 *    Persists MLP weights + opponent model across battles.
 *
 *  Action blending
 *    Each action produces a force vector; all 6 are blended via softmax
 *    of Q-values for smooth, less exploitable movement.
 *
 *  Minimum-risk movement
 *    Evaluates 48 candidate destination points; scores by wall proximity,
 *    enemy crosshair alignment, bullet intersection, and distance.
 *
 *  Wave surfing
 *    Tracks enemy bullet waves via energy-drop detection.  Orbits
 *    perpendicular to the wave in the direction with a less-targeted GF bin.
 *
 *  Opponent modelling
 *    Classifies enemies (random / oscillator / circular / linear) from
 *    scan history.  Per-class GF profiles bootstrap targeting for new enemies.
 *
 *  GuessFactor targeting (arc prediction, 47 bins × 9 segments)
 *    Blends per-enemy and per-class GF stats.  Stats decay 0.92 per round.
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
    private static final int A_MINRISK     = 0;
    private static final int A_CIRCLE_CW   = 1;
    private static final int A_CIRCLE_CCW  = 2;
    private static final int A_FLEE        = 3;
    private static final int A_WAVE_SURF   = 4;
    private static final int A_HUNT        = 5;

    // ─── Features ─────────────────────────────────────────────────────────────
    private static final int NUM_FEATURES = 14;
    private static final int HIDDEN_SIZE  = 32;

    // ─── Opponent classes ─────────────────────────────────────────────────────
    private static final int CLASS_UNKNOWN    = 0;
    private static final int CLASS_RANDOM     = 1;
    private static final int CLASS_OSCILLATOR = 2;
    private static final int CLASS_CIRCULAR   = 3;
    private static final int CLASS_LINEAR     = 4;
    private static final int NUM_CLASSES      = 5;

    // ─── Misc constants ─────────────────────────────────────────────────────
    private static final double SURVIVAL_BONUS = 0.1;
    private static final int    STALE_TURNS    = 40;
    private static final String AGENT_FILE     = "AISurvivor-qtable.dat";

    // ═════════════════════════════════════════════════════════════════════════
    //  Inner classes
    // ═════════════════════════════════════════════════════════════════════════

    private static final class Enemy {
        final int id;
        double x, y, direction, speed, energy;
        double lateralVelocity, prevEnergy, turnRate;
        int    lastScanTurn;
        boolean justFired;
        double  firedPower;

        // Opponent-modelling accumulators (carried forward across scans)
        int    scanCount;
        int    dirChanges;
        int    speedReversals;
        double sumAbsTurnRate;
        int    lastFireTurn;

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
                lastFireTurn = turn;
            }
        }

        void carryForward(Enemy prev) {
            scanCount      = prev.scanCount + 1;
            dirChanges     = prev.dirChanges + (Math.abs(turnRate) > 15 ? 1 : 0);
            speedReversals = prev.speedReversals +
                    (Math.signum(speed) != Math.signum(prev.speed)
                            && Math.abs(speed) > 0.5 ? 1 : 0);
            sumAbsTurnRate = prev.sumAbsTurnRate + Math.abs(turnRate);
            if (!justFired) lastFireTurn = prev.lastFireTurn;
        }
    }

    private static final class TargetingWave {
        double originX, originY, baseAngle, bulletSpeed;
        int fireTurn, targetId, segment;
        double lateralDir;
        double[] bins;
    }

    private static final class EnemyWave {
        double originX, originY, bulletSpeed, bearingToUs;
        int fireTurn, shooterId;
    }

    // ─── MLP (2-layer neural network) ────────────────────────────────────────

    static final class MLP implements Serializable {
        private static final long serialVersionUID = 1L;

        final int inputSize, hiddenSize, outputSize;
        double[][] W1, W2;
        double[] b1, b2;

        transient double[] lastInput, lastHidden, lastOutput;

        MLP(int input, int hidden, int output) {
            inputSize = input; hiddenSize = hidden; outputSize = output;
            W1 = new double[input][hidden];
            W2 = new double[hidden][output];
            b1 = new double[hidden];
            b2 = new double[output];
            xavierInit();
        }

        private void xavierInit() {
            Random rng = new Random();
            double s1 = Math.sqrt(2.0 / inputSize);
            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < hiddenSize; j++)
                    W1[i][j] = rng.nextGaussian() * s1;
            double s2 = Math.sqrt(2.0 / hiddenSize);
            for (int i = 0; i < hiddenSize; i++)
                for (int j = 0; j < outputSize; j++)
                    W2[i][j] = rng.nextGaussian() * s2;
        }

        double[] forward(double[] input) {
            lastInput = input;
            lastHidden = new double[hiddenSize];
            for (int j = 0; j < hiddenSize; j++) {
                double sum = b1[j];
                for (int i = 0; i < inputSize; i++) sum += input[i] * W1[i][j];
                lastHidden[j] = sum > 0 ? sum : 0; // ReLU
            }
            lastOutput = new double[outputSize];
            for (int j = 0; j < outputSize; j++) {
                double sum = b2[j];
                for (int i = 0; i < hiddenSize; i++) sum += lastHidden[i] * W2[i][j];
                lastOutput[j] = sum;
            }
            return lastOutput.clone();
        }

        void trainSingle(double[] input, int action, double targetQ, double lr) {
            double[] output = forward(input);
            double delta = targetQ - output[action];
            delta = clampS(delta, -1.0, 1.0); // gradient clipping

            // Hidden gradients (before updating W2)
            double[] dH = new double[hiddenSize];
            for (int j = 0; j < hiddenSize; j++)
                if (lastHidden[j] > 0) dH[j] = delta * W2[j][action];

            // Update output layer
            for (int j = 0; j < hiddenSize; j++)
                W2[j][action] += lr * delta * lastHidden[j];
            b2[action] += lr * delta;

            // Update hidden layer
            for (int j = 0; j < hiddenSize; j++) {
                if (dH[j] == 0) continue;
                for (int i = 0; i < inputSize; i++)
                    W1[i][j] += lr * dH[j] * lastInput[i];
                b1[j] += lr * dH[j];
            }
        }

        void copyFrom(MLP src) {
            for (int i = 0; i < inputSize; i++)
                System.arraycopy(src.W1[i], 0, W1[i], 0, hiddenSize);
            System.arraycopy(src.b1, 0, b1, 0, hiddenSize);
            for (int i = 0; i < hiddenSize; i++)
                System.arraycopy(src.W2[i], 0, W2[i], 0, outputSize);
            System.arraycopy(src.b2, 0, b2, 0, outputSize);
        }

        private static double clampS(double v, double lo, double hi) {
            return Math.max(lo, Math.min(hi, v));
        }
    }

    // ─── DQN Agent ───────────────────────────────────────────────────────────

    static final class DQNAgent implements Serializable {
        private static final long serialVersionUID = 4L;

        private static final double LR           = 0.002;
        private static final double GAMMA        = 0.95;
        private static final double TAU_INIT     = 1.0;
        private static final double TAU_MIN      = 0.15;
        private static final double TAU_DECAY    = 0.995;
        private static final int    REPLAY_CAP   = 5000;
        private static final int    BATCH        = 32;
        private static final int    MIN_REPLAY   = 64;
        private static final int    TARGET_SYNC  = 100;

        MLP policy, target;
        double tau;
        int totalRounds, totalSteps;

        // Opponent-model GF stats (persisted)
        double[][][] classGF; // [NUM_CLASSES][NUM_SEGS][GF_BINS]

        // Transient replay state
        transient ArrayList<double[]> replay;
        transient int replayPos;
        transient double[] lastFeatures;
        transient double[] lastQValues;
        transient int lastAction;

        DQNAgent(int features, int actions) {
            policy = new MLP(features, HIDDEN_SIZE, actions);
            target = new MLP(features, HIDDEN_SIZE, actions);
            target.copyFrom(policy);
            tau = TAU_INIT;
            classGF = new double[NUM_CLASSES][NUM_SEGS][GF_BINS];
        }

        void startEpisode() {
            if (replay == null) replay = new ArrayList<>();
            lastFeatures = null;
            lastAction = -1;
        }

        int observe(double[] features, double reward, boolean done) {
            if (lastFeatures != null) {
                store(lastFeatures, lastAction, reward, features, done);
                if (replay.size() >= MIN_REPLAY) replayLearn();
            }

            totalSteps++;
            if (totalSteps % TARGET_SYNC == 0) target.copyFrom(policy);

            lastQValues = policy.forward(features);
            int action = boltzmann(lastQValues);
            lastFeatures = features.clone();
            lastAction = action;
            return action;
        }

        void onDeath(double penalty) {
            if (lastFeatures != null) {
                store(lastFeatures, lastAction, penalty, lastFeatures, true);
                if (replay.size() >= MIN_REPLAY)
                    for (int i = 0; i < BATCH; i++) replayLearnOne();
            }
        }

        void endEpisode() {
            if (replay != null && replay.size() >= MIN_REPLAY)
                for (int i = 0; i < BATCH * 3; i++) replayLearnOne();
            totalRounds++;
            tau = Math.max(TAU_MIN, tau * TAU_DECAY);
        }

        private void store(double[] f, int a, double r, double[] nf, boolean done) {
            int fLen = f.length;
            double[] entry = new double[fLen * 2 + 3];
            System.arraycopy(f, 0, entry, 0, fLen);
            entry[fLen] = a;
            entry[fLen + 1] = r;
            System.arraycopy(nf, 0, entry, fLen + 2, fLen);
            entry[fLen * 2 + 2] = done ? 1.0 : 0.0;

            if (replay.size() < REPLAY_CAP) {
                replay.add(entry);
            } else {
                if (replayPos >= REPLAY_CAP) replayPos = 0;
                replay.set(replayPos++, entry);
            }
        }

        private void replayLearn() {
            for (int b = 0; b < BATCH; b++) replayLearnOne();
        }

        private void replayLearnOne() {
            double[] entry = replay.get((int)(Math.random() * replay.size()));
            int fLen = policy.inputSize;
            double[] f = new double[fLen];
            System.arraycopy(entry, 0, f, 0, fLen);
            int    a    = (int) entry[fLen];
            double r    = entry[fLen + 1];
            double[] nf = new double[fLen];
            System.arraycopy(entry, fLen + 2, nf, 0, fLen);
            boolean done = entry[fLen * 2 + 2] > 0.5;

            double tgt;
            if (done) {
                tgt = r;
            } else {
                double[] nq = target.forward(nf);
                double mx = nq[0];
                for (int i = 1; i < nq.length; i++) if (nq[i] > mx) mx = nq[i];
                tgt = r + GAMMA * mx;
            }
            policy.trainSingle(f, a, tgt, LR);
        }

        private int boltzmann(double[] q) {
            double mx = q[0];
            for (int i = 1; i < q.length; i++) if (q[i] > mx) mx = q[i];
            double[] p = new double[q.length];
            double sum = 0;
            for (int i = 0; i < q.length; i++) {
                p[i] = Math.exp((q[i] - mx) / tau);
                sum += p[i];
            }
            double r = Math.random() * sum, cum = 0;
            for (int i = 0; i < p.length; i++) {
                cum += p[i];
                if (r <= cum) return i;
            }
            return p.length - 1;
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Fields
    // ═════════════════════════════════════════════════════════════════════════

    private final Map<Integer, Enemy>        enemies        = new HashMap<>();
    private final Map<Integer, double[][]>   gfStats        = new HashMap<>();
    private final List<TargetingWave>        targetingWaves = new ArrayList<>();
    private final List<BulletState>          incomingBullets = new ArrayList<>();
    private final List<EnemyWave>            enemyWaves     = new ArrayList<>();
    private final Map<Integer, double[]>     surfStats      = new HashMap<>();

    private int    radarTargetId = -1;
    private int    radarSign     = 1;
    private int    dodgeSign     = 1;
    private double pendingReward = 0.0;

    private DQNAgent agent;

    // ═════════════════════════════════════════════════════════════════════════
    //  Entry point
    // ═════════════════════════════════════════════════════════════════════════

    public static void main(String[] args) { new AISurvivor().start(); }

    public AISurvivor() {
        super(BotInfo.fromFile("AISurvivor.json"));
        loadAgent();
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Main run loop
    // ═════════════════════════════════════════════════════════════════════════

    @Override
    public void run() {
        setAdjustGunForBodyTurn(true);
        setAdjustRadarForGunTurn(true);

        while (isRunning()) {
            pruneStaleEnemies();
            advanceTargetingWaves();
            advanceEnemyWaves();

            double[] features = encodeFeatures();
            int action = agent.observe(features, pendingReward + SURVIVAL_BONUS, false);
            pendingReward = 0.0;

            executeBlendedMovement(agent.lastQValues);
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
        enemyWaves.clear();
        pendingReward = 0.0;
        dodgeSign     = 1;
        radarTargetId = -1;
        radarSign     = 1;
        agent.startEpisode();

        // Decay GF and surf stats
        for (double[][] segs : gfStats.values())
            for (double[] bins : segs)
                for (int i = 0; i < bins.length; i++) bins[i] *= GF_DECAY;
        for (double[] bins : surfStats.values())
            for (int i = 0; i < bins.length; i++) bins[i] *= GF_DECAY;
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
        double prevEn   = (prev != null) ? prev.energy : e.getEnergy();
        double turnRate = (prev != null)
                ? normalizeAngle(e.getDirection() - prev.direction) : 0.0;

        double bearingToEnemy = directionTo(e.getX(), e.getY());
        double latVel = e.getSpeed() *
                Math.sin(Math.toRadians(e.getDirection() - bearingToEnemy));

        Enemy info = new Enemy(
                e.getScannedBotId(), e.getX(), e.getY(),
                e.getDirection(), e.getSpeed(), e.getEnergy(),
                latVel, prevEn, turnRate, getTurnNumber());
        if (prev != null) info.carryForward(prev);
        enemies.put(info.id, info);
        gfStats.computeIfAbsent(info.id, k -> new double[NUM_SEGS][GF_BINS]);

        // Emit enemy wave on detected fire
        if (info.justFired) {
            EnemyWave w = new EnemyWave();
            w.originX    = info.x;
            w.originY    = info.y;
            w.bulletSpeed = 20.0 - 3.0 * info.firedPower;
            w.fireTurn   = getTurnNumber();
            w.shooterId  = info.id;
            w.bearingToUs = Math.toDegrees(
                    Math.atan2(getX() - info.x, getY() - info.y));
            enemyWaves.add(w);
            surfStats.computeIfAbsent(info.id, k -> new double[GF_BINS]);
        }
    }

    @Override
    public void onHitByBullet(HitByBulletEvent e) {
        pendingReward -= e.getDamage() * (0.6 + 0.1 * getEnemyCount());
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
    public void onDeath(DeathEvent e) {
        agent.onDeath(-(60.0 + getEnemyCount() * 12.0));
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
    //  Feature encoding (14 continuous features)
    // ═════════════════════════════════════════════════════════════════════════

    private double[] encodeFeatures() {
        double[] f = new double[NUM_FEATURES];
        f[0] = getEnergy() / 100.0;

        Enemy t = bestTarget();
        if (t != null) {
            double dist = distanceTo(t.x, t.y);
            f[1]  = dist / 1000.0;
            f[2]  = clamp((getEnergy() - t.energy) / 100.0, -1, 1);
            double bearing = directionTo(t.x, t.y);

            // Closing speed: target's radial velocity component toward us
            double bearingFromTarget = Math.toDegrees(
                    Math.atan2(getX() - t.x, getY() - t.y));
            f[7]  = clamp(-t.speed * Math.cos(
                    Math.toRadians(t.direction - bearingFromTarget)) / 16.0, -1, 1);
            f[8]  = clamp((getTurnNumber() - t.lastFireTurn) / 20.0, 0, 1);
            double bearRad = Math.toRadians(bearing);
            f[10] = Math.sin(bearRad);
            f[11] = Math.cos(bearRad);
            f[12] = t.speed / 8.0;
            f[13] = clamp(t.turnRate / 10.0, -1, 1);
        } else {
            f[1] = 0.5; f[2] = 0; f[7] = 0; f[8] = 1; f[10] = 0; f[11] = 1;
            f[12] = 0; f[13] = 0;
        }

        f[3] = getEnemyCount() / 5.0;
        f[4] = clamp(1.0 - minWallDist() / 200.0, 0, 1);
        f[5] = getGunHeat() / 3.0;

        // Count incoming bullets aimed at us
        int aimed = 0;
        double myX = getX(), myY = getY();
        for (BulletState b : incomingBullets) {
            double dx = myX - b.getX(), dy = myY - b.getY();
            double bear = Math.toDegrees(Math.atan2(dx, dy));
            if (Math.abs(normalizeAngle(b.getDirection() - bear)) < 45) aimed++;
        }
        f[6] = aimed / 5.0;
        f[9] = getSpeed() / 8.0;

        return f;
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  GuessFactor Targeting + Opponent Modelling
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
            double gf = clamp(latAngle / maxAngle, -1.0, 1.0);
            int bin = gfBin(gf);

            // Update per-enemy stats
            w.bins[bin] += 1.0;

            // Update per-class stats
            int cls = classifyEnemy(t);
            agent.classGF[cls][w.segment][bin] += 1.0;

            it.remove();
        }
    }

    private int classifyEnemy(Enemy e) {
        if (e.scanCount < 8) return CLASS_UNKNOWN;
        double changeRate   = (double) e.dirChanges / e.scanCount;
        double reversalRate = (double) e.speedReversals / e.scanCount;
        double avgTurnRate  = e.sumAbsTurnRate / e.scanCount;

        if (changeRate > 0.35)   return CLASS_RANDOM;
        if (reversalRate > 0.25) return CLASS_OSCILLATOR;
        if (avgTurnRate > 4.0)   return CLASS_CIRCULAR;
        return CLASS_LINEAR;
    }

    private void aimGF() {
        Enemy t = bestTarget();
        if (t == null) return;

        double dist = distanceTo(t.x, t.y);
        double fp   = firePower(dist);
        double bspd = calcBulletSpeed(fp > 0 ? fp : 1.0);

        int        seg     = segment(dist, Math.abs(t.lateralVelocity));
        double[][] st      = gfStats.get(t.id);

        // Blend per-enemy + per-class GF stats
        int bestBin = GF_MID;
        double bestVal = -1;
        int cls = classifyEnemy(t);
        double w = Math.min(1.0, t.scanCount / 30.0); // trust per-enemy more as data grows

        for (int i = 0; i < GF_BINS; i++) {
            double perEnemy = (st != null) ? st[seg][i] : 0;
            double perClass = agent.classGF[cls][seg][i];
            double blended  = w * perEnemy + (1.0 - w) * perClass;
            if (blended > bestVal) { bestVal = blended; bestBin = i; }
        }

        double gf       = (bestBin / (double)(GF_BINS - 1)) * 2.0 - 1.0;
        double maxAngle = Math.toDegrees(Math.asin(Math.min(1.0, 8.0 / bspd)));
        double latDir   = Math.signum(t.lateralVelocity);
        if (latDir == 0) latDir = 1;

        // Arc prediction
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
    //  Movement — action blending + force-vector interface
    // ═════════════════════════════════════════════════════════════════════════

    private void executeBlendedMovement(double[] qValues) {
        if (qValues == null) { steerTo(getDirection(), 8.0); return; }

        double[] weights = softmax(qValues, agent.tau);
        double fx = 0, fy = 0, speed = 0;

        for (int a = 0; a < NUM_ACTIONS; a++) {
            if (weights[a] < 0.02) continue;
            double[] f = actionForce(a);
            fx    += weights[a] * f[0];
            fy    += weights[a] * f[1];
            speed += weights[a] * f[2];
        }

        double heading = normDir(Math.toDegrees(Math.atan2(fx, fy)));
        double smoothed = wallSmooth(getX(), getY(), heading, 1);
        steerTo(smoothed, speed);
    }

    private double[] actionForce(int action) {
        return switch (action) {
            case A_MINRISK    -> minRiskForce(1.0);
            case A_CIRCLE_CW  -> circleForce(true);
            case A_CIRCLE_CCW -> circleForce(false);
            case A_FLEE       -> minRiskForce(3.0);
            case A_WAVE_SURF  -> waveSurfForce();
            case A_HUNT       -> huntForce();
            default           -> minRiskForce(1.0);
        };
    }

    // ─── Minimum-risk movement ───────────────────────────────────────────────

    private double[] minRiskForce(double enemyScale) {
        double myX = getX(), myY = getY();
        double bestRisk = Double.MAX_VALUE, bestX = myX, bestY = myY;

        // 24 points at 120px + 12 at 60px + 12 at 200px = 48 candidates
        for (int ring = 0; ring < 3; ring++) {
            double radius = ring == 0 ? 120 : ring == 1 ? 60 : 200;
            int count = ring == 0 ? 24 : 12;
            for (int i = 0; i < count; i++) {
                double angle = i * (2 * Math.PI / count);
                double px = clamp(myX + Math.sin(angle) * radius,
                        WALL_MARGIN, getArenaWidth()  - WALL_MARGIN);
                double py = clamp(myY + Math.cos(angle) * radius,
                        WALL_MARGIN, getArenaHeight() - WALL_MARGIN);
                double risk = evaluateRisk(px, py, enemyScale);
                if (risk < bestRisk) { bestRisk = risk; bestX = px; bestY = py; }
            }
        }

        double dx = bestX - myX, dy = bestY - myY;
        double d = Math.sqrt(dx * dx + dy * dy) + 1;
        return new double[]{ dx / d, dy / d, 8.0 };
    }

    private double evaluateRisk(double px, double py, double enemyScale) {
        double risk = 0;

        // Wall proximity
        double wd = Math.min(Math.min(px, getArenaWidth() - px),
                Math.min(py, getArenaHeight() - py));
        if (wd < WALL_MARGIN) risk += 3.0 * (1.0 - wd / WALL_MARGIN);

        // Enemy danger: proximity + crosshair alignment + energy
        for (Enemy e : enemies.values()) {
            double dist = Math.sqrt(sq(px - e.x) + sq(py - e.y)) + 1;
            risk += enemyScale * e.energy / dist * 0.5;

            double bearFromEnemy = Math.toDegrees(Math.atan2(px - e.x, py - e.y));
            double angleOff = Math.abs(normalizeAngle(e.direction - bearFromEnemy));
            if (angleOff < 30) risk += enemyScale * 2.0 * (1.0 - angleOff / 30.0);
        }

        // Bullet intersection (project bullet 5 ticks forward)
        for (BulletState b : incomingBullets) {
            double bdir = Math.toRadians(b.getDirection());
            double bSpeed = 20.0 - 3.0 * b.getPower();
            double bx = b.getX() + Math.sin(bdir) * bSpeed * 5;
            double by = b.getY() + Math.cos(bdir) * bSpeed * 5;
            double dist = Math.sqrt(sq(px - bx) + sq(py - by));
            if (dist < 50) risk += 4.0 * (1.0 - dist / 50.0);
        }

        // Slight center preference
        double cx = getArenaWidth() / 2, cy = getArenaHeight() / 2;
        risk += Math.sqrt(sq(px - cx) + sq(py - cy)) * 0.002;

        return risk;
    }

    // ─── Circle movement ─────────────────────────────────────────────────────

    private double[] circleForce(boolean cw) {
        Enemy t = nearestEnemy();
        if (t == null) return minRiskForce(1.0);
        double orbitAngle = Math.toRadians(directionTo(t.x, t.y) + (cw ? 90 : -90));
        double speed = 8.0 * ((getTurnNumber() % 11 < 7) ? 1.0 : 0.5);
        return new double[]{ Math.sin(orbitAngle), Math.cos(orbitAngle), speed };
    }

    // ─── Wave surfing ────────────────────────────────────────────────────────

    private double[] waveSurfForce() {
        EnemyWave closest = closestWave();
        if (closest == null) return minRiskForce(1.0);

        double[] danger = surfStats.get(closest.shooterId);
        if (danger == null) return minRiskForce(1.0);

        double bearFromWave = Math.toDegrees(
                Math.atan2(getX() - closest.originX, getY() - closest.originY));
        double maxEscape = Math.toDegrees(
                Math.asin(Math.min(1.0, 8.0 / closest.bulletSpeed)));

        // Compare GF danger for CW vs CCW orbit
        double cwGF  = clamp(normalizeAngle(bearFromWave + 5 - closest.bearingToUs)
                / maxEscape, -1, 1);
        double ccwGF = clamp(normalizeAngle(bearFromWave - 5 - closest.bearingToUs)
                / maxEscape, -1, 1);

        double cwDanger  = danger[gfBin(cwGF)];
        double ccwDanger = danger[gfBin(ccwGF)];

        double orbitDir = (cwDanger <= ccwDanger)
                ? bearFromWave + 90 : bearFromWave - 90;
        double rad = Math.toRadians(orbitDir);
        return new double[]{ Math.sin(rad), Math.cos(rad), 8.0 };
    }

    private EnemyWave closestWave() {
        EnemyWave best = null;
        double minDist = Double.MAX_VALUE;
        double myX = getX(), myY = getY();
        for (EnemyWave w : enemyWaves) {
            double traveled = (getTurnNumber() - w.fireTurn) * w.bulletSpeed;
            double dist = Math.sqrt(sq(myX - w.originX) + sq(myY - w.originY));
            double remaining = dist - traveled;
            if (remaining > 0 && remaining < minDist) {
                minDist = remaining;
                best = w;
            }
        }
        return best;
    }

    private void advanceEnemyWaves() {
        double myX = getX(), myY = getY();
        Iterator<EnemyWave> it = enemyWaves.iterator();
        while (it.hasNext()) {
            EnemyWave w = it.next();
            double traveled = (getTurnNumber() - w.fireTurn) * w.bulletSpeed;
            double dist = Math.sqrt(sq(myX - w.originX) + sq(myY - w.originY));
            if (traveled >= dist - w.bulletSpeed) {
                // Wave reached us — record our GF in surf stats
                double bear = Math.toDegrees(Math.atan2(myX - w.originX, myY - w.originY));
                double maxEsc = Math.toDegrees(
                        Math.asin(Math.min(1.0, 8.0 / w.bulletSpeed)));
                double gf = clamp(normalizeAngle(bear - w.bearingToUs) / maxEsc, -1, 1);
                double[] bins = surfStats.get(w.shooterId);
                if (bins != null) bins[gfBin(gf)] += 1.0;
                it.remove();
            }
        }
    }

    // ─── Hunt movement ───────────────────────────────────────────────────────

    private double[] huntForce() {
        Enemy t = weakestEnemy();
        if (t == null) return minRiskForce(1.0);
        double angle = Math.toRadians(directionTo(t.x, t.y));
        return new double[]{ Math.sin(angle), Math.cos(angle), 8.0 };
    }

    // ─── Steering helper ─────────────────────────────────────────────────────

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
            Enemy target = bestTarget();
            if (target != null) {
                double radarTurn = normalizeAngle(
                        directionTo(target.x, target.y) - getRadarDirection());
                int staleness = getTurnNumber() - target.lastScanTurn;
                if (staleness <= 2 && Math.abs(radarTurn) < 20) {
                    setRadarTurnRate(45.0 * radarSign);
                } else {
                    setRadarTurnRate(clamp(
                            radarTurn + Math.signum(radarTurn) * 15.0, -45, 45));
                }
            } else {
                setRadarTurnRate(45.0);
            }
        } else {
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
                setRadarTurnRate(clamp(radarTurn + radarSign * 8.0, -45, 45));
            } else {
                double overshoot = Math.min(10.0 + staleness * 7.0, 45.0);
                setRadarTurnRate(clamp(
                        radarTurn + Math.signum(radarTurn) * overshoot, -45, 45));
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
                if (obj instanceof DQNAgent loaded) {
                    agent = loaded;
                    System.out.printf("[AISurvivor] DQN loaded. Rounds: %d  τ=%.3f%n",
                            agent.totalRounds, agent.tau);
                    return;
                }
                System.out.println("[AISurvivor] Incompatible agent — starting fresh DQN.");
            } catch (Exception ex) {
                System.out.println("[AISurvivor] Load failed: " + ex.getMessage());
            }
        } else {
            System.out.println("[AISurvivor] No agent file — starting fresh DQN.");
        }
        agent = new DQNAgent(NUM_FEATURES, NUM_ACTIONS);
    }

    private void saveAgent() {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(AGENT_FILE))) {
            out.writeObject(agent);
            System.out.printf("[AISurvivor] DQN saved. Rounds: %d  τ=%.3f%n",
                    agent.totalRounds, agent.tau);
        } catch (Exception ex) {
            System.out.println("[AISurvivor] Save failed: " + ex.getMessage());
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
        for (Enemy e : enemies.values())
            if (e.energy < minEn) { minEn = e.energy; best = e; }
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
        return Math.min(Math.min(x, getArenaWidth() - x),
                Math.min(y, getArenaHeight() - y));
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

    private static double[] softmax(double[] q, double tau) {
        double mx = q[0];
        for (int i = 1; i < q.length; i++) if (q[i] > mx) mx = q[i];
        double[] w = new double[q.length];
        double sum = 0;
        for (int i = 0; i < q.length; i++) {
            w[i] = Math.exp((q[i] - mx) / tau);
            sum += w[i];
        }
        for (int i = 0; i < w.length; i++) w[i] /= sum;
        return w;
    }

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
