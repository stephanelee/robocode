import java.util.Random;

final class MLP {
    final int inputSize, hiddenSize, outputSize;
    double[][] W1, W2;
    double[] b1, b2;

    double[] lastInput, lastHidden, lastOutput;

    MLP(int input, int hidden, int output, Random rng) {
        inputSize = input; hiddenSize = hidden; outputSize = output;
        W1 = new double[input][hidden];
        W2 = new double[hidden][output];
        b1 = new double[hidden];
        b2 = new double[output];
        xavierInit(rng);
    }

    private void xavierInit(Random rng) {
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

