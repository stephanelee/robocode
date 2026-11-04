import com.google.gson.*;
import dev.robocode.tankroyale.botapi.BulletState;
import dev.robocode.tankroyale.botapi.events.*;
import org.junit.jupiter.api.Test;
import java.nio.file.*;
import java.io.*;
import java.security.MessageDigest;
import java.util.*;
import java.util.zip.GZIPInputStream;
import static org.junit.jupiter.api.Assertions.*;

class RecordedGunLivenessTest {
    static final String HASH="f0c38e81e0965b29e5c71c79ec2f2eb7a1c18232dd275172d37b1dbf6251c2c1";
    record Tick(int round,int turn,JsonObject own,List<JsonObject> events) {}
    static double n(JsonObject o,String k){return o.get(k).getAsDouble();}
    static int i(JsonObject o,String k){return o.get(k).getAsInt();}
    static Flight.Pose pose(JsonObject o){return new Flight.Pose(n(o,"x"),n(o,"y"),n(o,"direction"),n(o,"speed"));}
    List<Tick> load() throws Exception {
        Path p=Path.of(System.getProperty("meridian.fixture"));
        assertEquals(HASH,HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(p))));
        var ticks=new ArrayList<Tick>();
        try(var reader=new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(p)),java.nio.charset.StandardCharsets.UTF_8))) {
            for(String line;(line=reader.readLine())!=null;) {
                var o=JsonParser.parseString(line).getAsJsonObject();
                if(!o.get("type").getAsString().equals("TickEventForObserver"))continue;
                JsonObject own=null;
                for(var b:o.getAsJsonArray("botStates"))if(i(b.getAsJsonObject(),"id")==1)own=b.getAsJsonObject();
                var legal=new ArrayList<JsonObject>();
                for(var e:o.getAsJsonArray("events")) {
                    var event=e.getAsJsonObject();String type=event.get("type").getAsString();
                    if(type.equals("ScannedBotEvent")&&i(event,"scannedByBotId")==1)legal.add(event);
                    if(type.equals("BulletFiredEvent")&&i(event.getAsJsonObject("bullet"),"ownerId")==1)legal.add(event);
                }
                if(own!=null)ticks.add(new Tick(i(o,"roundNumber"),i(o,"turnNumber"),own,legal));
            }
        }
        return ticks;
    }
    void input(GunHarness h,Tick t) {
        if(t.round!=h.round){h.bot.onRoundStarted(new RoundStartedEvent(t.round));h.round=t.round;}
        h.own=pose(t.own);h.energy=n(t.own,"energy");h.heat=n(t.own,"gunHeat");h.heading=n(t.own,"gunDirection");
        for(var e:t.events) {
            if(e.get("type").getAsString().equals("ScannedBotEvent"))h.scan(t.turn,pose(e),n(e,"energy"));
            else {
                var b=e.getAsJsonObject("bullet");
                h.bot.onBulletFired(new BulletFiredEvent(t.turn,new BulletState(i(b,"bulletId"),1,n(b,"power"),n(b,"x"),n(b,"y"),n(b,"direction"),null)));
            }
        }
    }
    boolean usable(GunHarness h,int t) throws Exception {
        var scan=(BaselineMeridian.Scan)h.field("scan");return scan!=null&&t-scan.t()<=3;
    }
    @Test void unchangedV3ReproducesEveryShotAndTheFullStarvationInterval() throws Exception {
        var h=new GunHarness(true);var actual=new TreeMap<String,Double>();var predicted=new TreeMap<String,Double>();
        int stalled=0,flips=0;double lastSign=0,maxLat=0,minError=Double.POSITIVE_INFINITY,maxError=0;
        for(var t:load()) {
            input(h,t);
            for(var e:t.events)if(e.has("bullet"))actual.put(t.round+":"+t.turn,n(e.getAsJsonObject("bullet"),"power"));
            if(!usable(h,t.turn)||h.energy<=0)continue;
            h.decide(t.turn);
            if(h.requests>0)predicted.put(t.round+":"+(t.turn+1),h.power);
            if(t.round==5&&t.turn>=144&&t.turn<=3037) {
                var scan=(BaselineMeridian.Scan)h.field("scan");assertEquals(t.turn,scan.t());assertEquals(0,h.heat);
                assertTrue(h.energy>1.65);assertEquals(0,h.requests);
                double bearing=Flight.angle(h.target.x()-h.own.x(),h.target.y()-h.own.y());
                double lat=h.target.v()*Flight.sin(h.target.h()-bearing),sign=lat<0?-1:1;
                if(lastSign!=0&&sign!=lastSign)flips++;lastSign=sign;maxLat=Math.max(maxLat,Math.abs(lat));
                minError=Math.min(minError,Math.abs(h.rate));maxError=Math.max(maxError,Math.abs(h.rate));stalled++;
            }
        }
        assertEquals(actual.keySet(),predicted.keySet(),"No false/missing fire decisions");
        for(String k:actual.keySet())assertEquals(actual.get(k),predicted.get(k),1e-10,k);
        assertEquals(244,actual.size());assertEquals(2894,stalled);assertEquals(1446,flips);assertTrue(maxLat<=1.475);
        System.out.printf(Locale.ROOT,"BASELINE shots=%d false=0 missing=0 stall=%d flips=%d maxLat=%.12f error=[%.12f,%.12f]%n",actual.size(),stalled,flips,maxLat,minError,maxError);
    }
    @Test void committedPlanSettlesWithinProspectiveGeometryBoundWithoutBattleSimulation() throws Exception {
        var ticks=load();var interval=ticks.stream().filter(t->t.round==5&&t.turn>=144&&t.turn<=3037).toList();
        double maxDelta=0,minTolerance=Double.POSITIVE_INFINITY,previous=Double.NaN;
        for(var t:interval) {
            var scan=t.events.stream().filter(e->e.get("type").getAsString().equals("ScannedBotEvent")).findFirst().orElseThrow();
            double dx=n(scan,"x")-n(t.own,"x"),dy=n(scan,"y")-n(t.own,"y"),bearing=Flight.angle(dx,dy);
            if(Double.isFinite(previous))maxDelta=Math.max(maxDelta,Math.abs(Flight.delta(bearing-previous)));
            previous=bearing;minTolerance=Math.min(minTolerance,Math.toDegrees(Math.atan2(12,Math.hypot(dx,dy))));
        }
        assertTrue(maxDelta<20&&maxDelta<=minTolerance,"Predeclared bound must be applicable");
        var base=new GunHarness(true);
        for(var t:ticks) {
            input(base,t);
            if(t.round==5&&t.turn==144)break;
            if(usable(base,t.turn)&&base.energy>0)base.decide(t.turn);
        }
        var h=new GunHarness(false);
        h.round=base.round;h.set("round",base.round);
        h.set("crossingGun",base.field("crossingGun"));
        System.arraycopy((double[])base.field("skill"),0,(double[])h.field("skill"),0,3);
        var s=(BaselineMeridian.Scan)base.field("scan");
        h.set("scan",new Meridian.Scan(s.t(),s.p(),s.energy(),s.omega()));
        h.own=base.own;h.target=base.target;h.energy=base.energy;h.heat=base.heat;h.heading=base.heading;
        base.decide(144);h.decide(144);
        assertEquals(base.rate,h.rate,1e-10,"Same initial v3 aim");
        assertEquals(25.17607197893551,h.rate,1e-9);assertEquals(3.2111017339626926,h.plan().offset(),1e-9);
        int bound=(int)Math.ceil(Math.max(0,Math.abs(h.rate)-20)/(20-maxDelta))+1;
        assertEquals(2,bound);
        var plan=h.plan();int steps=0;
        System.out.printf(Locale.ROOT,"SERVO bound=%d D=%.12f A=%.12f initialError=%.12f%n",bound,maxDelta,minTolerance,h.rate);
        while(h.requests==0&&steps<bound) {
            double nextHeading=h.heading+Flight.clip(h.rate,-20,20);
            var t=interval.get(++steps);
            // Own state and own visible scan only. Stop on the first available request.
            input(h,t);h.heading=nextHeading;h.decide(t.turn);
            assertSame(plan,h.plan());
            System.out.printf(Locale.ROOT,"SERVO turn=%d step=%d error=%.12f request=%d%n",t.turn,steps,h.rate,h.requests);
        }
        assertEquals(1,h.requests);assertTrue(steps<=bound);assertEquals(146,h.turn);
        assertEquals(-.8622972508655948,h.rate,1e-9);
    }
}
