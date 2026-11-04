import dev.robocode.tankroyale.runner.*;
import java.nio.file.*;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.*;

/** Observer only. Separate fresh bot JVMs/processes; no observer data reaches policy code. */
public class MeridianPeakBattle {
    static boolean aisComplete(Path path,int rounds) throws Exception {
        if(!Files.isRegularFile(path))return false;
        try(var in=new java.io.DataInputStream(Files.newInputStream(path))) {
            if(in.readInt()!=0x41495356||in.readInt()!=1)return false;
            in.skipNBytes(12);return in.readInt()==rounds;
        }
    }
    public static void main(String[] args) throws Exception {
        int port=Integer.parseInt(args[0]),rounds=Integer.parseInt(args[1]);
        if(!((port>=32900&&port<=32904&&rounds==5)||(port>=32910&&port<=32918&&rounds==20)))
            throw new IllegalArgumentException("Unregistered port/round population");
        Path output=Path.of(args[2]),markers=output.resolve("markers.txt");
        var setup=BattleSetup.oneVsOne(s->{s.setNumberOfRounds(rounds);s.setTurnTimeoutMicros(100000);});
        try(var runner=BattleRunner.create(b->b.embeddedServer(port).enableRecording(output).suppressServerOutput().botConnectTimeout(Duration.ofSeconds(80)));
            var battle=runner.startBattleAsync(setup,List.of(BotEntry.of(args[3]),BotEntry.of(args[4])))) {
            var pool=Executors.newSingleThreadExecutor();
            try {
                var pending=pool.submit(battle::awaitResults);
                long until=System.nanoTime()+TimeUnit.SECONDS.toNanos(1500);
                while(!pending.isDone()) {
                    if(Files.isRegularFile(markers)&&Files.readAllLines(markers).stream().anyMatch(m->m.contains(" skip ")))
                        throw new IllegalStateException("Own skipped turn; stop this attempt");
                    if(System.nanoTime()>until)throw new TimeoutException("Battle deadline exceeded");
                    Thread.sleep(25);
                }
                var result=pending.get();
                if(result.getNumberOfRounds()!=rounds)throw new IllegalStateException("Incomplete game");
                for(var row:result.getResults())System.out.printf("%s %s %d %d%n",row.getName(),row.getVersion(),row.getTotalScore(),row.getFirstPlaces());
                // Let every final callback revise its seal, including queued events after onRoundEnded.
                Thread.sleep(10000);
                if(!Files.isRegularFile(markers)||!Files.readAllLines(markers).contains(rounds+" end"))
                    throw new IllegalStateException("Final private callback marker missing");
                if(System.getenv("MERIDIAN_PEAK_STATE_DIR")!=null) {
                    Path endpoint=Path.of(System.getenv("MERIDIAN_PEAK_STATE_DIR")).resolve("endpoint.json");
                    if(!Files.isRegularFile(endpoint)||!java.util.regex.Pattern.compile("\"completedRounds\"\\s*:\\s*"+rounds+"\\b").matcher(Files.readString(endpoint)).find())
                        throw new IllegalStateException("Final Peak state did not seal");
                }
                if(System.getenv("AISURVIVOR_MODEL_PATH")!=null&&(!aisComplete(output.resolve("ais-model.dat"),rounds)
                    ||!Files.isRegularFile(output.resolve("ais.jsonl"))||Files.readAllLines(output.resolve("ais.jsonl")).size()!=rounds))
                    throw new IllegalStateException("AIS final callbacks/model incomplete");
            } finally {pool.shutdownNow();}
        }
    }
}
