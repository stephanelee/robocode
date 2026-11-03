import dev.robocode.tankroyale.runner.*;
import java.nio.file.*;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.*;

/** Observer only. A separate JVM runs each bot. */
public class MeridianBattle {
    static boolean modelComplete(Path file,int rounds) throws java.io.IOException {
        if(!Files.isRegularFile(file))return false;
        try(var in=new java.io.DataInputStream(Files.newInputStream(file))){
            if(in.readInt()!=0x41495356 || in.readInt()!=1)return false;
            in.skipNBytes(12);return in.readInt()==rounds;
        }
    }
    public static void main(String[] args) throws Exception {
        int port=Integer.parseInt(args[0]), rounds=Integer.parseInt(args[1]);
        Path output=Path.of(args[2]), markers=output.resolve("markers.txt");
        if(!((31410<=port && port<=31421) || (31740<=port && port<=31752) || (31820<=port && port<=31832)))throw new IllegalArgumentException("Unregistered port");
        var setup=BattleSetup.oneVsOne(s->{s.setNumberOfRounds(rounds);s.setTurnTimeoutMicros(100000);});
        try(var runner=BattleRunner.create(b->b.embeddedServer(port).enableRecording(output).suppressServerOutput().botConnectTimeout(Duration.ofSeconds(80)));
            var battle=runner.startBattleAsync(setup,List.of(BotEntry.of(args[3]),BotEntry.of(args[4])))){
            var pool=Executors.newSingleThreadExecutor();
            try {
                var result=pool.submit(battle::awaitResults).get(1500,TimeUnit.SECONDS);
                if(result.getNumberOfRounds()!=rounds)throw new IllegalStateException("Incomplete game");
                for(var row:result.getResults())System.out.printf("%s %s %d %d%n",row.getName(),row.getVersion(),row.getTotalScore(),row.getFirstPlaces());
                long until=System.nanoTime()+TimeUnit.SECONDS.toNanos(10);
                Path ais=output.resolve("ais.jsonl"), model=output.resolve("ais-model.dat");
                boolean requireAis=System.getenv("AISURVIVOR_MODEL_PATH")!=null;
                while(!Files.isRegularFile(markers)||!Files.readAllLines(markers).contains(rounds+" end")
                        || (requireAis && (!Files.isRegularFile(ais)||Files.readAllLines(ais).size()!=rounds||!modelComplete(model,rounds)))){
                    if(System.nanoTime()>until)throw new IllegalStateException("Final callback did not flush");
                    Thread.sleep(25);
                }
            } finally {pool.shutdownNow();}
        }
    }
}
