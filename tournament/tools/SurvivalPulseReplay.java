import com.google.gson.*;
import dev.robocode.tankroyale.server.model.*;
import java.nio.file.Path;
import java.util.*;

/** Legal trigger and own-control replay. Only the independent scoring side sees the physical field. */
public class SurvivalPulseReplay extends CrossfireReplay {
    record Scan(int tick,double x,double y) {}
    static final class Trigger {
        final Map<Integer,Scan> scans=new HashMap<>();
        boolean eligible(int t, Phys.Pose self, double turnRate, int width, int height) {
            if (t%16!=0 || Math.abs(self.v())!=8 || Math.abs(turnRate)>=2) return false;
            double clearance=Math.min(Math.min(self.x()-18,width-18-self.x()),Math.min(self.y()-18,height-18-self.y()));
            if (clearance<80) return false;
            double nearest=Double.POSITIVE_INFINITY;
            for (var scan:scans.values()) if (scan.tick<=t && t-scan.tick<=3)
                nearest=Math.min(nearest,Phys.distance(self.x(),self.y(),scan.x,scan.y));
            return Double.isFinite(nearest) && nearest>=500;
        }
    }
    record Ray(int id,int owner,int fire,double power,double x,double y,double direction) {
        Point point(int t) {
            double distance=(t-fire+1)*Phys.bulletSpeed(power);
            return new Point(x+distance*Phys.cos(direction),y+distance*Phys.sin(direction));
        }
        Line segment(int t) {return new Line(point(t),point(t+1));}
        double damage() {return Phys.damage(power);}
    }
    static final class Window {
        final int start,sign;
        Phys.Pose path;
        final Map<Integer,Ray> active=new HashMap<>(), originalHits=new HashMap<>(), proposedHits=new HashMap<>();
        Window(int start,Phys.Pose p) {this.start=start;path=p;sign=p.v()>0?1:-1;}
    }
    static final class Score {
        int originalHits, proposedHits, gainedHits, avoidedHits;
        double original, proposed, avoidedUnresolved, latent, truncation;
    }
    static final class Stats {
        int windows,pairedTicks,complete,truncated,possibleRams,wallContacts;
        double ramPenalty;
        final Map<String,Score> shooters=new TreeMap<>();
    }
    final boolean score;
    final Trigger trigger=new Trigger();
    final Stats stats=new Stats();
    final Map<Integer,String> names=new HashMap<>();
    final Map<Integer,Ray> physical=new HashMap<>();
    Phys.Pose expected;
    Window window;
    SurvivalPulseReplay(boolean score) {super(false);this.score=score;}

    @Override void observe(JsonObject e) {
        super.observe(e);
        trigger.scans.put(integer(e,"scannedBotId"),new Scan(integer(e,"turnNumber"),number(e,"x"),number(e,"y")));
    }
    @Override void onDecision(int t,JsonObject state,Phys.Pose p,MeleePolicy.Opponent o) {
        // Reconstruct the accepted movement state on every original decision, including outside pulses.
        var command=policy.move(t,p,number(state,"energy"),width,height);
        expected=Phys.step(p,command.turn(),command.speed(),width,height);
        if (trigger.eligible(t,p,number(state,"turnRate"),width,height)) {
            if (window!=null) throw new IllegalStateException("Overlapping windows");
            window=new Window(t,p);stats.windows++;
            for (Ray b:physical.values()) if (b.owner!=own) window.active.put(b.id,b);
        }
    }
    static Point point(Phys.Pose p) {return new Point(p.x(),p.y());}
    static boolean hit(Line segment,Point pos,double radius) {
        return MathKt.isLineIntersectingCircle(segment,pos,radius);
    }
    boolean outside(Point p) {
        return p.getX()<=0 || p.getY()<=0 || p.getX()>=width || p.getY()>=height;
    }
    Score tally(Ray ray) {return stats.shooters.computeIfAbsent(names.get(ray.owner),k->new Score());}

    boolean reachable(Ray b,int t,Point center) {
        for (int k=1;k<=4;k++) {
            Line future=b.segment(t+k);
            if (outside(future.getStart())) break;
            if (hit(future,center,18+8*k)) return true;
        }
        return false;
    }
    void finish(boolean complete,int t,Point original,Set<Integer> physicalAlive) {
        var w=window;
        if (complete) stats.complete++; else stats.truncated++;
        Map<String,Double> base=new HashMap<>(),candidate=new HashMap<>();
        for (Ray b:w.originalHits.values()) {
            var s=tally(b);s.original+=b.damage();s.originalHits++;
            base.merge(names.get(b.owner),b.damage(),Double::sum);
            if (score && !w.proposedHits.containsKey(b.id)) s.avoidedHits++;
        }
        if (score) {
            for (Ray b:w.proposedHits.values()) {
                var s=tally(b);s.proposed+=b.damage();s.proposedHits++;
                candidate.merge(names.get(b.owner),b.damage(),Double::sum);
                if (!w.originalHits.containsKey(b.id)) s.gainedHits++;
            }
            for (Ray b:w.active.values()) {
                Point p=b.point(t);
                double passed=(p.getX()-w.path.x())*Phys.cos(b.direction)+(p.getY()-w.path.y())*Phys.sin(b.direction);
                if (w.originalHits.containsKey(b.id) && passed<=18) {
                    tally(b).avoidedUnresolved+=b.damage();
                } else if (original!=null && reachable(b,t,point(w.path))
                        && !(physicalAlive.contains(b.id) && reachable(b,t,original))) {
                    tally(b).latent+=b.damage();
                } else continue;
                candidate.merge(names.get(b.owner),b.damage(),Double::sum);
            }
            if (!complete) for (var entry:base.entrySet())
                stats.shooters.get(entry.getKey()).truncation+=Math.max(0,entry.getValue()-candidate.getOrDefault(entry.getKey(),0.));
        }
        window=null;
    }
    void evaluate(JsonObject m,int t,Map<Integer,Set<Integer>> victims,Map<Integer,Set<Integer>> collisions,
                  Map<Integer,JsonObject> bots,Set<Integer> alive, boolean ownDied) {
        var w=window;
        var s=bots.get(own);
        if (s==null) {finish(false,t-1,null,Set.of());return;}
        Point actual=new Point(number(s,"x"),number(s,"y"));
        if (expected==null || Math.hypot(expected.x()-actual.getX(),expected.y()-actual.getY())>1e-6
                || Math.abs(expected.v()-number(s,"speed"))>1e-6 || Math.abs(Phys.norm(expected.h()-number(s,"direction")))>1e-6)
            throw new IllegalStateException("Original movement mismatch in round "+round+" turn "+t);
        stats.pairedTicks++;
        Set<Integer> reproduced=new HashSet<>(),observed=new HashSet<>();
        for (Ray b:physical.values()) if (b.owner!=own) {
            Line line=b.segment(t);
            if (!outside(line.getStart()) && hit(line,actual,18)) {
                reproduced.add(b.id);w.originalHits.put(b.id,b);
            }
        }
        for (var e:victims.entrySet()) if (e.getValue().contains(own)) observed.add(e.getKey());
        if (!reproduced.equals(observed)) throw new IllegalStateException("Original incoming geometry mismatch");
        boolean ram=false;
        if (score) {
            double speed=w.sign*(t-w.start<=4?4:8);
            w.path=Phys.step(w.path,0,speed,width,height);
            Point pos=point(w.path);
            if (w.path.v()!=speed && (pos.getX()==18 || pos.getY()==18 || pos.getX()==width-18 || pos.getY()==height-18))
                throw new IllegalStateException("Unreachable wall contact");
            Map<Integer,Point> others=new HashMap<>();
            for (var e:bots.entrySet()) if (e.getKey()!=own) {
                Point other=new Point(number(e.getValue(),"x"),number(e.getValue(),"y"));others.put(e.getKey(),other);
                if (Math.hypot(pos.getX()-other.getX(),pos.getY()-other.getY())<=36) {
                    ram=true;stats.possibleRams++;stats.ramPenalty+=.6;
                }
            }
            var before=Set.copyOf(w.active.keySet());
            for (var it=w.active.values().iterator();it.hasNext();) {
                Ray b=it.next();Line segment=b.segment(t);
                if (outside(segment.getStart())) {it.remove();continue;}
                boolean touched=hit(segment,pos,18);
                if (touched) w.proposedHits.put(b.id,b);
                // All bot and bullet contacts on this tick coexist. Ignore protection from own fire.
                boolean end=touched || others.entrySet().stream().anyMatch(e->e.getKey()!=b.owner && hit(segment,e.getValue(),18))
                        || collisions.getOrDefault(b.id,Set.of()).stream().anyMatch(before::contains);
                if (end) it.remove();
            }
        }
        boolean full=t-w.start==8 && !ram;
        if (full || ram || ownDied || integer(s,"enemyCount")<=1) finish(full,t,actual,alive);
    }

    void tick(JsonObject m) {
        int t=integer(m,"turnNumber");
        Map<Integer,Set<Integer>> victims=new HashMap<>(),collisions=new HashMap<>();
        boolean ownDied=false;
        for (var item:m.getAsJsonArray("events")) {
            var e=item.getAsJsonObject();String type=string(e,"type");
            if (type.equals("BotDeathEvent") && integer(e,"victimId")==own) ownDied=true;
            if (!e.has("bullet")) continue;
            var b=e.getAsJsonObject("bullet");int id=integer(b,"bulletId");
            switch(type) {
                case "BulletFiredEvent" -> {
                    var ray=new Ray(id,integer(b,"ownerId"),t,number(b,"power"),number(b,"x"),number(b,"y"),number(b,"direction"));
                    physical.put(id,ray);
                    if (window!=null && ray.owner!=own) window.active.put(id,ray);
                }
                case "BulletHitBotEvent" -> victims.computeIfAbsent(id,k->new HashSet<>()).add(integer(e,"victimId"));
                case "BulletHitBulletEvent" -> {
                    int other=integer(e.getAsJsonObject("hitBullet"),"bulletId");
                    collisions.computeIfAbsent(id,k->new HashSet<>()).add(other);
                    collisions.computeIfAbsent(other,k->new HashSet<>()).add(id);
                }
            }
        }
        Set<Integer> alive=new HashSet<>();
        for (var item:m.getAsJsonArray("bulletStates")) alive.add(integer(item.getAsJsonObject(),"bulletId"));
        if (window!=null) {
            Map<Integer,JsonObject> bots=new HashMap<>();
            for (var item:m.getAsJsonArray("botStates")) {
                var b=item.getAsJsonObject();bots.put(integer(b,"id"),b);
            }
            evaluate(m,t,victims,collisions,bots,alive,ownDied);
        }
        physical.keySet().retainAll(alive);
    }
    @Override void endRound() {
        if (window!=null) finish(false,previousTurn,null,physical.keySet());
        physical.clear();trigger.scans.clear();expected=null;super.endRound();
    }
    @Override void event(JsonObject m) {
        String type=string(m,"type");
        if (type.equals("GameStartedEventForObserver")) for (var item:m.getAsJsonArray("participants")) {
            var p=item.getAsJsonObject();names.put(integer(p,"id"),string(p,"name"));
        }
        if (type.equals("TickEventForObserver") && integer(m,"roundNumber")==round) tick(m);
        super.event(m);
    }
    public static void main(String[] args) throws Exception {
        if (args.length!=2 || !Set.of("baseline","pulse").contains(args[1])) throw new IllegalArgumentException();
        if (!sha(Path.of(MathKt.class.getProtectionDomain().getCodeSource().getLocation().toURI()))
                .equals("16c277795bd823c9eda0b650f1d09a418ff35cfe43d59b820c52d5e6816f905c")) throw new IllegalStateException("Server hash");
        var readers=readGames(Path.of(args[0]),()->new SurvivalPulseReplay(args[1].equals("pulse")));
        Map<String,Stats> output=new TreeMap<>();readers.forEach((g,r)->output.put(g,r.stats));
        System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(output));
    }
}
