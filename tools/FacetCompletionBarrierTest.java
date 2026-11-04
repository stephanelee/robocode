import java.nio.file.*;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicReference;

/** Offline file/ordering contract tests, no engine games or opponent inputs. */
public class FacetCompletionBarrierTest {
    static int assertions;
    static void check(boolean value, String reason) {
        assertions++;
        if (!value) throw new AssertionError(reason);
    }
    static String rows(int rounds) {
        var text = new StringBuilder();
        for (int i=1;i<=rounds;i++) text.append("{\"schema\":3,\"round\":").append(i)
                .append(",\"snapshotStatus\":\"success\",\"terminalObservation\":\"success\",\"terminalHash\":\"success\"}\n");
        return text.toString();
    }
    static void reject(Path model, Path metrics, int rounds, String reason) throws Exception {
        try { BenchmarkBattle.awaitFacetCompletion(model, metrics, rounds, Duration.ofMillis(45)); }
        catch (IllegalStateException | IllegalArgumentException expected) { assertions++; return; }
        throw new AssertionError(reason);
    }
    public static void main(String[] args) throws Exception {
        Path root=Files.createTempDirectory("facet-runner-barrier-").toAbsolutePath();
        Path model=root.resolve("FACET_model.dat"), metrics=root.resolve("FACET_metrics.jsonl");
        try {
            // Only a final renamed export acknowledges readiness; delayed producer must stay alive.
            Path partial=root.resolve("FACET_model.dat.partial");
            Files.writeString(partial,"independent fixture bytes"); Files.writeString(metrics,rows(20));
            reject(model,metrics,20,"partial export must not acknowledge completion");
            AtomicReference<Throwable> failure=new AtomicReference<>();
            Thread writer=new Thread(()->{try {Thread.sleep(100);Files.move(partial,model);}catch(Throwable e){failure.set(e);}});
            writer.start();
            BenchmarkBattle.awaitFacetCompletion(model,metrics,20,Duration.ofSeconds(2));
            writer.join(2000);check(!writer.isAlive() && failure.get()==null,"delayed writer completed before runner cleanup");
            Files.writeString(metrics,rows(19));reject(model,metrics,20,"missing round");
            Files.writeString(metrics,rows(20).replace("\"round\":20,","\"round\":19,"));reject(model,metrics,20,"duplicate round");
            Files.writeString(metrics,rows(20).replace("\"snapshotStatus\":\"success\"","\"snapshotStatus\":\"failed\""));reject(model,metrics,20,"failed snapshot");
            Files.writeString(metrics,rows(20).stripTrailing());reject(model,metrics,20,"unterminated row");
            Files.writeString(metrics,rows(20));Files.delete(model);Files.createSymbolicLink(model,partial.getFileName());
            reject(model,metrics,20,"symlink output");Files.delete(model);
            Files.writeString(model,"");reject(model,metrics,20,"empty model");
            try(var file=new java.io.RandomAccessFile(model.toFile(),"rw")){file.setLength(134217729);}
            reject(model,metrics,20,"model cap");Files.writeString(model,"own fixture");
            try(var file=new java.io.RandomAccessFile(metrics.toFile(),"rw")){file.setLength(33554433);}
            reject(model,metrics,20,"metrics cap");
            reject(Path.of("FACET_relative"),metrics,20,"absolute path");
            reject(model,model,20,"distinct path");
            reject(root.resolve("foreign.dat"),metrics,20,"private basename");
            System.out.println("PASS Facet runner-lifetime completion barrier assertions="+assertions+" battles=0");
        } finally {
            try(var files=Files.list(root)){for(Path path:files.toList())Files.deleteIfExists(path);}
            Files.delete(root);
        }
    }
}
