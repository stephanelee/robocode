import java.util.*;

final class DQNAgent {
    private static final int HIDDEN_SIZE = 32;
    private static final int NUM_CLASSES = 5, NUM_SEGS = 9, GF_BINS = 47;
    final Random rng;
    final BotConfig config;

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
    ArrayList<double[]> replay;
    int replayPos;
    double[] lastFeatures;
    double[] lastQValues;
    int lastAction;

    DQNAgent(int features, int actions, BotConfig config) {
        this.config = config;
        rng = new Random(config.seed());
        policy = new MLP(features, HIDDEN_SIZE, actions, rng);
        target = new MLP(features, HIDDEN_SIZE, actions, rng);
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
        return observe(features, reward, done, false);
    }

    int observe(double[] features, double reward, boolean done, boolean duelEligible) {
        // Decide the hierarchy before crediting the prior transition; never sample then execute another action.
        boolean forced = config.forceDuel() && duelEligible;
        int action = forced ? 0 : -1;
        if (config.training() && lastFeatures != null) {
            store(lastFeatures, lastAction, reward, features, done, forced);
            if (replay.size() >= MIN_REPLAY) replayLearn();
        }

        if (config.training()) {
            totalSteps++;
            if (totalSteps % TARGET_SYNC == 0) target.copyFrom(policy);
        }

        lastQValues = policy.forward(features);
        if (!forced) action = config.mode() == BotConfig.Mode.FIXED ? config.fixedAction()
                : config.training() ? boltzmann(lastQValues) : greedy(lastQValues);
        lastFeatures = features.clone();
        lastAction = action;
        return action;
    }

    void finishEpisode(double reward) {
        if (lastFeatures == null) return;
        if (config.training()) {
            store(lastFeatures, lastAction, reward, lastFeatures, true, false);
            if (replay.size() >= MIN_REPLAY)
                for (int i = 0; i < BATCH * 3; i++) replayLearnOne();
            totalRounds++;
            tau = Math.max(TAU_MIN, tau * TAU_DECAY);
        }
        lastFeatures = null;
        lastAction = -1;
    }

    static int greedy(double[] q) {
        int best = 0;
        for (int i = 1; i < q.length; i++) if (q[i] > q[best]) best = i;
        return best;
    }

    private void store(double[] f, int a, double r, double[] nf, boolean done, boolean nextForced) {
        int fLen = f.length;
        double[] entry = new double[fLen * 2 + 4];
        System.arraycopy(f, 0, entry, 0, fLen);
        entry[fLen] = a;
        entry[fLen + 1] = r;
        System.arraycopy(nf, 0, entry, fLen + 2, fLen);
        entry[fLen * 2 + 2] = done ? 1.0 : 0.0;
        // Replay is transient and never serialized. Preserve the constrained next-state TD action set.
        entry[fLen * 2 + 3] = nextForced ? 1.0 : 0.0;

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
        double[] entry = replay.get(rng.nextInt(replay.size()));
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
            if (entry[fLen * 2 + 3] == 0)
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
        double r = rng.nextDouble() * sum, cum = 0;
        for (int i = 0; i < p.length; i++) {
            cum += p[i];
            if (r <= cum) return i;
        }
        return p.length - 1;
    }
}

