import java.nio.file.*;
import java.time.Duration;
import java.util.concurrent.*;

/** Synthetic output timing only; never starts a server or bot. */
public class ChallengerCompletionTest {
    public static void main(String[] args) throws Exception {
        Path root = Path.of(args[0]).toAbsolutePath();
        Files.createDirectories(root);
        Path model = root.resolve("model.json"), metrics = root.resolve("metrics.jsonl");
        Files.writeString(model, "{\"roundsCompleted\":1}");
        Files.writeString(metrics, "{\"roundsCompleted\":1}\n");
        BenchmarkBattle.awaitChallengerCompletion(model, metrics, 1, Duration.ofSeconds(1));

        // Results can arrive before GameEnded seals the final row. A final model alone is insufficient.
        Files.writeString(model, "{\"roundsCompleted\":2}");
        var writer = CompletableFuture.runAsync(() -> {
            try {
                Thread.sleep(60);
                Files.writeString(metrics, "{\"roundsCompleted\":1}\n{\"roundsCompleted\":2}\n");
            } catch (Exception error) { throw new RuntimeException(error); }
        });
        BenchmarkBattle.awaitChallengerCompletion(model, metrics, 2, Duration.ofSeconds(2));
        writer.get();

        // Partial append, extra rows, stale model and missing output all fail closed.
        Files.writeString(metrics, "{\"roundsCompleted\":1}\n{\"roundsCompleted\":2}");
        rejects(() -> BenchmarkBattle.awaitChallengerCompletion(model, metrics, 2, Duration.ofMillis(50)));
        Files.writeString(metrics, "{\"roundsCompleted\":1}\n{\"roundsCompleted\":2}\n{}\n");
        rejects(() -> BenchmarkBattle.awaitChallengerCompletion(model, metrics, 2, Duration.ofMillis(50)));
        Files.writeString(metrics, "{\"roundsCompleted\":1}\n{\"roundsCompleted\":2}\n");
        Files.writeString(model, "{\"roundsCompleted\":1}");
        rejects(() -> BenchmarkBattle.awaitChallengerCompletion(model, metrics, 2, Duration.ofMillis(50)));
        Files.delete(model);
        rejects(() -> BenchmarkBattle.awaitChallengerCompletion(model, metrics, 2, Duration.ofMillis(50)));
        rejects(() -> BenchmarkBattle.awaitChallengerCompletion(metrics, metrics, 2, Duration.ofMillis(50)));
        System.out.println("Seven synthetic completion checks passed; zero games");
    }
    interface Checked { void run() throws Exception; }
    static void rejects(Checked action) throws Exception {
        try { action.run(); } catch (IllegalStateException | IllegalArgumentException expected) { return; }
        throw new AssertionError("Incomplete or invalid output was accepted");
    }
}
