import java.io.*;
import java.nio.file.*;
import java.util.UUID;

/** Bounded, versioned numeric checkpoint. No Java object deserialization. */
final class ModelStore {
    private static final int MAGIC = 0x41495356, VERSION = 1;

    static boolean load(Path path, DQNAgent agent) throws IOException {
        if (!Files.exists(path)) return false;
        DQNAgent candidate = new DQNAgent(agent.policy.inputSize, agent.policy.outputSize, agent.config);
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            if (in.readInt() != MAGIC || in.readInt() != VERSION) throw new IOException("Unsupported checkpoint format/version");
            if (in.readInt() != candidate.policy.inputSize || in.readInt() != candidate.policy.hiddenSize
                    || in.readInt() != candidate.policy.outputSize) throw new IOException("Model dimensions differ");
            candidate.totalRounds = in.readInt();
            candidate.totalSteps = in.readInt();
            candidate.tau = number(in);
            if (candidate.totalRounds < 0 || candidate.totalSteps < 0 || candidate.tau < 0.15 || candidate.tau > 1)
                throw new IOException("Invalid model counters/temperature");
            readNetwork(in, candidate.policy);
            readNetwork(in, candidate.target);
            for (double[][] segments : candidate.classGF)
                for (double[] bins : segments)
                    for (int i = 0; i < bins.length; i++) {
                        bins[i] = number(in);
                        if (bins[i] < 0) throw new IOException("Negative targeting statistic");
                    }
            if (in.read() != -1) throw new IOException("Trailing checkpoint data");
        }
        agent.policy.copyFrom(candidate.policy);
        agent.target.copyFrom(candidate.target);
        agent.classGF = candidate.classGF;
        agent.totalRounds = candidate.totalRounds;
        agent.totalSteps = candidate.totalSteps;
        agent.tau = candidate.tau;
        return true;
    }

    static void save(Path path, DQNAgent agent) throws IOException {
        Path absolute = path.toAbsolutePath();
        Path parent = absolute.getParent();
        Files.createDirectories(parent);
        Path temp = Files.createTempFile(parent, absolute.getFileName().toString(), ".tmp");
        try {
            try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(temp)))) {
                out.writeInt(MAGIC); out.writeInt(VERSION);
                out.writeInt(agent.policy.inputSize); out.writeInt(agent.policy.hiddenSize); out.writeInt(agent.policy.outputSize);
                out.writeInt(agent.totalRounds); out.writeInt(agent.totalSteps); out.writeDouble(agent.tau);
                writeNetwork(out, agent.policy); writeNetwork(out, agent.target);
                for (double[][] segments : agent.classGF)
                    for (double[] bins : segments) for (double value : bins) out.writeDouble(value);
            }
            // Validate the complete new file before replacing the last usable checkpoint.
            load(temp, new DQNAgent(agent.policy.inputSize, agent.policy.outputSize, agent.config));
            if (Files.exists(absolute)) {
                try { load(absolute, new DQNAgent(agent.policy.inputSize, agent.policy.outputSize, agent.config)); }
                catch (IOException ex) {
                    Files.copy(absolute, absolute.resolveSibling(absolute.getFileName() + ".rejected-" + UUID.randomUUID()));
                }
            }
            // Fail safely if the filesystem cannot atomically replace a sibling file.
            Files.move(temp, absolute, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } finally { Files.deleteIfExists(temp); }
    }

    private static double number(DataInputStream in) throws IOException {
        double n = in.readDouble();
        if (!Double.isFinite(n)) throw new IOException("Non-finite model value");
        return n;
    }

    private static void readNetwork(DataInputStream in, MLP net) throws IOException {
        for (double[] row : net.W1) for (int i = 0; i < row.length; i++) row[i] = number(in);
        for (int i = 0; i < net.b1.length; i++) net.b1[i] = number(in);
        for (double[] row : net.W2) for (int i = 0; i < row.length; i++) row[i] = number(in);
        for (int i = 0; i < net.b2.length; i++) net.b2[i] = number(in);
    }

    private static void writeNetwork(DataOutputStream out, MLP net) throws IOException {
        for (double[] row : net.W1) for (double value : row) out.writeDouble(value);
        for (double value : net.b1) out.writeDouble(value);
        for (double[] row : net.W2) for (double value : row) out.writeDouble(value);
        for (double value : net.b2) out.writeDouble(value);
    }
}
