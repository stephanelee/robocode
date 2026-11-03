import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.*;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class DiagnosticTraceTest {
    @TempDir Path temp;
    @Test void onAndOffProduceIdenticalGunCommandsLearningAndPlannerCommands() throws Exception {
        var on=new PressureGun();var off=new PressureGun();var log=new DiagnosticTrace(temp.resolve("trace.jsonl"));
        on.diagnostic(log);double gun=0;
        for(int round=1;round<=2;round++) {
            on.startRound();off.startRound();log.startRound(round);
            PressureGun.Plan previous=null;
            for(int turn=1;turn<=180;turn++) {
                var scan=new DuelGun.Scan(2,turn,500,300+25*Math.sin(turn*.04),90,4,100);
                on.onScan(scan);off.onScan(scan);
                if(previous!=null&&previous.fire()) {
                    on.onFired(turn,turn,300,300,previous.power(),gun);
                    off.onFired(turn,turn,300,300,previous.power(),gun);
                }
                var a=on.decide(turn,300,300,gun,100,0,800,600);
                var b=off.decide(turn,300,300,gun,100,0,800,600);
                assertEquals(b,a);previous=a;gun=a.direction();
                var threat=new WaveSurfMove.Threat(500,300,14,turn-5,180,1,0,new double[47]);
                assertEquals(WaveSurfMove.choose(400,300,0,8,turn,500,300,List.of(threat),1,800,600),
                        WaveSurfMove.choose(400,300,0,8,turn,500,300,List.of(threat),1,800,600,log));
            }
            assertArrayEquals(off.attempts(),on.attempts());assertArrayEquals(off.hits(),on.hits());
            assertArrayEquals(off.pairedAttempts(),on.pairedAttempts());assertArrayEquals(off.pairedHits(),on.pairedHits());
            assertEquals(off.context().snapshot(),on.context().snapshot());
            log.finishRound(180,0,0,0);
        }
        String text=Files.readString(temp.resolve("trace.jsonl"));
        assertTrue(text.contains("\"type\":\"shot\""));assertTrue(text.contains("\"type\":\"passage\""));
        assertTrue(text.contains("\"type\":\"planner\""));assertTrue(text.contains("\"type\":\"virtual\""));
    }
    @Test void plannerObserverHasBitExactCommandsAcrossSeededWallsProfilesAndUnresolvedWaves() {
        var log=new DiagnosticTrace(temp.resolve("wide.jsonl"));log.startRound(1);var random=new java.util.SplittableRandom(20263500);
        for(int i=0;i<300;i++) {
            double x=random.nextDouble(18,782),y=random.nextDouble(18,582),heading=random.nextDouble(360),speed=random.nextDouble(-8,8);
            double[] bins=new double[47];bins[random.nextInt(47)]=random.nextDouble(40);
            var threat=new WaveSurfMove.Threat(random.nextDouble(800),random.nextDouble(600),random.nextDouble(11,19.7),
                    100-random.nextInt(30),random.nextDouble(360),random.nextBoolean()?1:-1,random.nextDouble(8),bins);
            int sign=random.nextBoolean()?1:-1;var waves=i%7==0?List.<WaveSurfMove.Threat>of():List.of(threat);
            assertEquals(WaveSurfMove.choose(x,y,heading,speed,100,500,300,waves,sign,800,600),
                    WaveSurfMove.choose(x,y,heading,speed,100,500,300,waves,sign,800,600,log));
        }
    }
    @Test void boundedBufferedRecorderNeverWritesBeforeBoundaryAndIgnoresClosedRound() throws Exception {
        Path path=temp.resolve("bounded.jsonl");var log=new DiagnosticTrace(path);log.startRound(1);
        for(int i=0;i<DiagnosticTrace.MAX_RECORDS+10;i++)log.emit("tick",i,"own",new double[]{1,2,3,4});
        assertFalse(Files.exists(path));log.finishRound(7,0,0,0);
        assertEquals(DiagnosticTrace.MAX_RECORDS,Files.readAllLines(path).size());
        assertTrue(Files.size(path)<=DiagnosticTrace.MAX_BYTES);
        String status=Files.readString(Path.of(path+".status.json"));assertTrue(status.contains("\"droppedRecords\":11"));
        long size=Files.size(path);log.emit("late",8);log.finishRound(8,0,0,0);assertEquals(size,Files.size(path));
        assertThrows(IllegalArgumentException.class,()->new DiagnosticTrace(temp.resolve("model.json"),temp.resolve("model.json")));
        assertThrows(IllegalArgumentException.class,()->new DiagnosticTrace(temp.resolve("trace"),temp.resolve("trace.status.json")));
    }
}
