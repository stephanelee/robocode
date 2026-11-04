import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.function.Supplier;

/** Own explicit DTOs only. This output is never loaded into a policy. */
final class PeakTelemetry {
    record Models(HitAngles.Snapshot hitAngles,CrossingGun.Snapshot crossingGun,double[] gunSkill,java.util.List<TrialSnapshot> pendingTrials) {}
    record TrialSnapshot(int due,double[] angles,double x,double y) {}
    record LabelEntry(int round,String type,Object label) {}
    record TrialLabel(int observedTurn,String reason,TrialSnapshot trial,Flight.Pose enemy) {}
    static final class Stats {
        long confirmedShots,hits,hitsTaken,skippedTurns,crossingLabels,acceptedHitLabels;
        long crossingMaskedMissingBracket,crossingMaskedInsideFront,crossingMaskedOutsideOffset,trialScored,trialMissingScan;
        double shotEnergy,damageDealt,damageTaken;
        long plannerDecisions,kernelEvaluations,decisionTotalNanos,maximumDecisionNanos;
        int maximumActiveWaves,maximumModeledRays,maximumRaysPerWave;
        double maximumWaveRisk;
    }
    record Snapshot(int schemaVersion,String name,String version,long seed,boolean checkpointLoaded,
                    int round,int completedRounds,long revision,boolean sealed,String coldStateSHA256,
                    long cumulativeCrossings,long cumulativeAcceptedHits,Stats stats,Models models) {}
    private static final Gson JSON=new GsonBuilder().serializeNulls().create();
    final Path directory;
    final long seed;
    private final Supplier<Models> models;
    Stats stats=new Stats();
    int round,completedRounds;
    long revision,cumulativeCrossings,cumulativeAcceptedHits;
    boolean sealed;
    String coldHash;
    PeakTelemetry(String output,long seed,Supplier<Models> models) {
        this.directory=output==null?null:Path.of(output);this.seed=seed;this.models=models;
        if(directory!=null)try {
            Files.createDirectory(directory);
            Files.createFile(directory.resolve("labels.jsonl"));
            byte[] data=bytes(snapshot());Files.write(directory.resolve("cold.json"),data,StandardOpenOption.CREATE_NEW);
            coldHash=HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(data));
        } catch(java.io.IOException e){throw new UncheckedIOException(e);}
          catch(java.security.NoSuchAlgorithmException e){throw new IllegalStateException(e);}
    }
    Snapshot snapshot() {return new Snapshot(1,"MeridianPeak","0.1.0",seed,false,round,completedRounds,revision,sealed,
            coldHash,cumulativeCrossings,cumulativeAcceptedHits,stats,models.get());}
    private static byte[] bytes(Object object) {return (JSON.toJson(object)+"\n").getBytes(StandardCharsets.UTF_8);}
    void start(int number) {
        if(number!=round+1 || (round>0&&!sealed))throw new IllegalStateException("Round start sequence");
        round=number;sealed=false;revision=0;stats=new Stats();
    }
    void end(int number) {
        if(number!=round || sealed || completedRounds+1!=number)throw new IllegalStateException("Duplicate/missing round end");
        completedRounds++;sealed=true;changed();
    }
    /** Late model mutations rewrite a seal, but never advance the completed-round counter. */
    void changed() {
        if(!sealed)return;
        revision++;
        if(directory!=null)try {
            byte[] data=bytes(snapshot());
            atomic(directory.resolve(String.format(java.util.Locale.ROOT,"round-%03d.json",round)),data);
            atomic(directory.resolve("endpoint.json"),data);
        }catch(java.io.IOException e){throw new UncheckedIOException(e);}
    }
    private static void atomic(Path destination,byte[] bytes) throws java.io.IOException {
        Path temporary=destination.resolveSibling(destination.getFileName()+".tmp");
        Files.write(temporary,bytes,StandardOpenOption.CREATE,StandardOpenOption.TRUNCATE_EXISTING);
        Files.move(temporary,destination,StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING);
    }
    private void label(String type,Object value) {
        if(directory!=null)try {
            Files.write(directory.resolve("labels.jsonl"),bytes(new LabelEntry(round,type,value)),
                    StandardOpenOption.CREATE,StandardOpenOption.APPEND);
        } catch(java.io.IOException e){throw new UncheckedIOException(e);}
    }
    void crossingLabels(java.util.List<CrossingGun.CrossingLabel> labels) {
        for(var item:labels) {
            switch(item.reason()) {
                case MISSING_BRACKET -> stats.crossingMaskedMissingBracket++;
                case INSIDE_PREVIOUS_FRONT -> stats.crossingMaskedInsideFront++;
                case OUTSIDE_OFFSET -> stats.crossingMaskedOutsideOffset++;
                case RESOLVED -> {} // Counted by the original learner's observe return value.
            }
            label("crossing",item);
        }
    }
    void trial(int observedTurn,TrialSnapshot trial,Flight.Pose enemy) {
        boolean scored=trial.due()==observedTurn;
        if(scored)stats.trialScored++;else stats.trialMissingScan++;
        label("trial",new TrialLabel(observedTurn,scored?"SCORED":"MISSING_SCAN",trial,enemy));
    }
    void crossings(int n){stats.crossingLabels+=n;cumulativeCrossings+=n;}
    void hitLabel(HitAngles.Label label){if(label==HitAngles.Label.ACCEPTED){stats.acceptedHitLabels++;cumulativeAcceptedHits++;}}
    void decision(long nanos,PeakRisk.Evaluation a,PeakRisk.Evaluation b) {
        stats.plannerDecisions++;stats.kernelEvaluations+=a.kernelEvaluations()+b.kernelEvaluations();
        stats.decisionTotalNanos+=nanos;stats.maximumDecisionNanos=Math.max(stats.maximumDecisionNanos,nanos);
        stats.maximumActiveWaves=Math.max(stats.maximumActiveWaves,Math.max(a.waves(),b.waves()));
        stats.maximumModeledRays=Math.max(stats.maximumModeledRays,Math.max(a.rays(),b.rays()));
        stats.maximumRaysPerWave=Math.max(stats.maximumRaysPerWave,Math.max(a.maximumRaysPerWave(),b.maximumRaysPerWave()));
        stats.maximumWaveRisk=Math.max(stats.maximumWaveRisk,Math.max(a.maximumWaveRisk(),b.maximumWaveRisk()));
    }
}
