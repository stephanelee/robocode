import com.google.gson.*;
import dev.robocode.tankroyale.server.model.Line;
import dev.robocode.tankroyale.server.model.Point;
import dev.robocode.tankroyale.server.model.MathKt;
import dev.robocode.tankroyale.server.rules.RulesKt;
import java.nio.file.Path;
import java.util.*;

/** Offline single-shot counterfactuals with an independently propagated legal gun direction. */
public final class MidpointReplay extends CrossfireReplay {
    final boolean shadow;
    MidpointReplay(boolean shadow) { super(false); this.shadow = shadow; }
    record Slot(int target, double ray, boolean allowed, boolean far, double actualRay, double power) {}
    record Bullet(int id, int owner, int fired, double power, double x, double y, double ray) {
        Point position(int age) {
            double distance = Phys.bulletSpeed(power)*age;
            return new Point(x + Phys.cos(ray)*distance, y + Phys.sin(ray)*distance);
        }
        Line segment(int tick) {
            int age = tick - fired + 1;
            return new Line(position(age),position(age+1));
        }
        Bullet atRay(double newRay) { return new Bullet(id,owner,fired,power,x,y,newRay); }
    }
    record Contact(boolean wall, Set<Integer> bots, Set<Integer> bullets) {
        boolean terminal() { return wall || !bots.isEmpty() || !bullets.isEmpty(); }
    }
    static final Contact EMPTY = new Contact(false,Set.of(),Set.of());
    static final class Trial {
        final Bullet actual;
        final Slot slot;
        Contact observed, predicted;
        boolean calibrated, censored;
        Trial(Bullet actual, Slot slot) { this.actual=actual; this.slot=slot; }
    }
    final Map<Integer, Bullet> physical = new HashMap<>();
    final List<Trial> trials = new ArrayList<>();
    final Map<Integer,String> names = new HashMap<>();
    final Map<String,Map<String,Integer>> results = new TreeMap<>();
    Slot pending;
    double baseRay=Double.NaN, newRay, firingRay;
    MeleeFire.Aim updatedAim;
    boolean far;

    static double rotate(double ray,double error) {
        return MathKt.normalizeAbsoluteDegrees(ray + Phys.clip(error,-20,20));
    }

    static MeleeFire.Aim midpoint(MeleePolicy.Opponent o,int t,Phys.Pose self,double distance,
                                  double power,double ray,int width,int height) {
        Phys.Pose p = new Phys.Pose(o.x,o.y,o.heading,o.speed);
        double rate=Phys.clip(o.turnRate,-Phys.maxTurn(o.speed),Phys.maxTurn(o.speed));
        for (int i=0; i<Math.min(32,Math.max(0,t-o.turn)); i++) p=Phys.step(p,rate,o.speed,width,height);
        for (int f=1; f<=120; f++) {
            p=Phys.step(p,rate,o.speed,width,height);
            if ((f+0.5)*Phys.bulletSpeed(power)>=Phys.distance(self.x(),self.y(),p.x(),p.y())) break;
        }
        double err=Phys.norm(Phys.bearing(self.x(),self.y(),p.x(),p.y())-ray);
        return new MeleeFire.Aim(o,distance,power,err);
    }

    @Override void onDecision(int t, JsonObject s, Phys.Pose self, MeleePolicy.Opponent o) {
        double observed=number(s,"gunDirection"), energy=number(s,"energy");
        if (Double.isNaN(baseRay)) { baseRay=observed; newRay=observed; }
        if (Math.abs(Phys.norm(baseRay-observed))>1e-6) throw new IllegalStateException("Gun calibration");
        updatedAim=null;
        if (o==null || energy<=0) return;
        double d=Phys.distance(self.x(),self.y(),o.x,o.y);
        double power=MeleePolicy.power(d,energy,o.energy);
        var old=MeleeFire.aim(o,t,self,power,d,baseRay,width,height);
        baseRay=rotate(baseRay,old.error());
        firingRay=newRay;
        far=d>=500;
        if (shadow) {
            updatedAim=far ? midpoint(o,t,self,d,power,newRay,width,height)
                    : MeleeFire.aim(o,t,self,power,d,newRay,width,height);
            newRay=rotate(newRay,updatedAim.error());
        }
    }

    @Override MeleeFire.Aim decide(int t, JsonObject state, Counts c) {
        var shot=super.decide(t,state,c);
        if (shot!=null) {
            if (shadow && (updatedAim==null || updatedAim.target()!=shot.target() || updatedAim.power()!=shot.power()))
                throw new IllegalStateException("Changed opportunity identity");
            boolean allowed=!shadow || updatedAim.canFire(t,12,number(state,"energy"));
            pending=new Slot(shot.target().id,firingRay,allowed,far,number(state,"gunDirection"),shot.power());
        }
        return shot;
    }

    boolean outside(Point p) {
        return p.getX()<=0 || p.getX()>=width || p.getY()<=0 || p.getY()>=height;
    }

    Contact contact(Bullet b,int tick,Map<Integer,Line> field,Map<Integer,Point> bots) {
        Line line=b.segment(tick);
        if (outside(line.getStart())) return new Contact(true,Set.of(),Set.of());
        var victims=new TreeSet<Integer>(); var collisions=new TreeSet<Integer>();
        for (var e:bots.entrySet())
            if (e.getKey()!=b.owner && MathKt.isLineIntersectingCircle(line,e.getValue(),18)) victims.add(e.getKey());
        double diameter=2*RulesKt.getMAX_BULLET_SPEED();
        Point end=line.getEnd();
        for (var e:field.entrySet()) {
            if (e.getKey()==b.id) continue;
            Point other=e.getValue().getEnd();
            double dx=end.getX()-other.getX(),dy=end.getY()-other.getY();
            if (dx*dx+dy*dy<=diameter*diameter && MathKt.isLineIntersectingLine(line,e.getValue()))
                collisions.add(e.getKey());
        }
        return new Contact(false,victims,collisions);
    }

    void tick(JsonObject m) {
        int t=integer(m,"turnNumber");
        Map<Integer,Set<Integer>> botHits=new HashMap<>(), bulletHits=new HashMap<>();
        Set<Integer> walls=new HashSet<>();
        for (var item:m.getAsJsonArray("events")) {
            var e=item.getAsJsonObject();
            if (!e.has("bullet")) continue;
            var b=e.getAsJsonObject("bullet");
            int id=integer(b,"bulletId");
            switch (string(e,"type")) {
                case "BulletFiredEvent" -> {
                    var bullet=new Bullet(id,integer(b,"ownerId"),t,number(b,"power"),number(b,"x"),number(b,"y"),number(b,"direction"));
                    if (physical.put(id,bullet)!=null) throw new IllegalStateException("Duplicate bullet");
                    if (bullet.owner==own && pending!=null) {
                        if (pending.power!=bullet.power || Math.abs(Phys.norm(pending.actualRay-bullet.ray))>1e-6)
                            throw new IllegalStateException("Wrong shot");
                        trials.add(new Trial(bullet,pending)); pending=null;
                    }
                }
                case "BulletHitWallEvent" -> walls.add(id);
                case "BulletHitBotEvent" -> botHits.computeIfAbsent(id,k->new TreeSet<>()).add(integer(e,"victimId"));
                case "BulletHitBulletEvent" -> {
                    int other=integer(e.getAsJsonObject("hitBullet"),"bulletId");
                    bulletHits.computeIfAbsent(id,k->new TreeSet<>()).add(other);
                    bulletHits.computeIfAbsent(other,k->new TreeSet<>()).add(id);
                }
            }
        }
        if (pending!=null) throw new IllegalStateException("Missing own physical shot");
        var bots=new HashMap<Integer,Point>();
        for (var item:m.getAsJsonArray("botStates")) {
            var s=item.getAsJsonObject();
            bots.put(integer(s,"id"),new Point(number(s,"x"),number(s,"y")));
        }
        var field=new HashMap<Integer,Line>();
        for (var b:physical.values()) {
            Line segment=b.segment(t);
            if (!outside(segment.getStart())) field.put(b.id,segment);
        }
        trials.removeIf(v->{
            int id=v.actual.id;
            var actual=new Contact(walls.contains(id),botHits.getOrDefault(id,Set.of()),bulletHits.getOrDefault(id,Set.of()));
            if (!v.calibrated) {
                var check=contact(v.actual,t,field,bots);
                if (!check.equals(actual)) throw new IllegalStateException("Geometry mismatch round "+round+
                        " turn "+t+" bullet "+id+" predicted "+check+" observed "+actual);
                if (actual.terminal()) { v.calibrated=true; v.observed=actual; }
            }
            if (shadow && v.slot.allowed && v.predicted==null && !v.censored) {
                var prediction=contact(v.actual.atRay(v.slot.ray),t,field,bots);
                if (prediction.terminal()) v.predicted=prediction;
                else if (!actual.bullets.isEmpty()) v.censored=true;
            }
            if (v.calibrated && (!shadow || v.predicted!=null || !v.slot.allowed || v.censored)) {
                summarize(v); return true;
            }
            return false;
        });
        var alive=new HashSet<Integer>();
        for (var item:m.getAsJsonArray("bulletStates")) alive.add(integer(item.getAsJsonObject(),"bulletId"));
        if (!physical.keySet().containsAll(alive)) throw new IllegalStateException("Unknown physical bullet");
        physical.keySet().retainAll(alive);
    }

    void summarize(Trial v) {
        var c=results.computeIfAbsent(names.get(v.slot.target),k->new TreeMap<>());
        Contact observed=v.observed==null?EMPTY:v.observed;
        c.merge("shots",1,Integer::sum);
        c.merge("observedIntended",observed.bots.contains(v.slot.target)?1:0,Integer::sum);
        c.merge("observedAny",observed.bots.isEmpty()?0:1,Integer::sum);
        c.merge("observedCollateralEvents",(int)observed.bots.stream().filter(i->i!=v.slot.target).count(),Integer::sum);
        if (!shadow) return;
        Contact predicted=v.predicted==null?EMPTY:v.predicted;
        boolean before=observed.bots.contains(v.slot.target), after=predicted.bots.contains(v.slot.target);
        c.merge("gained",!before&&after?1:0,Integer::sum); c.merge("lost",before&&!after?1:0,Integer::sum);
        c.merge("predictedIntended",after?1:0,Integer::sum); c.merge("predictedAny",predicted.bots.isEmpty()?0:1,Integer::sum);
        c.merge("predictedCollateralEvents",(int)predicted.bots.stream().filter(i->i!=v.slot.target).count(),Integer::sum);
        c.merge("suppressed",v.slot.allowed?0:1,Integer::sum); c.merge("far",v.slot.far?1:0,Integer::sum);
        c.merge("changedRay",v.slot.allowed && Math.abs(Phys.norm(v.slot.actualRay-v.slot.ray))>1e-6?1:0,Integer::sum);
        c.merge("censoredCollision",v.censored?1:0,Integer::sum);
        c.merge("censoredRoundEnd",v.slot.allowed && v.predicted==null && !v.censored?1:0,Integer::sum);
    }

    @Override void event(JsonObject m) {
        String type=string(m,"type");
        if (type.equals("GameStartedEventForObserver")) for (var p:m.getAsJsonArray("participants")) {
            var o=p.getAsJsonObject(); names.put(integer(o,"id"),string(o,"name"));
        }
        if (type.equals("TickEventForObserver")) {
            if (integer(m,"roundNumber")==round) tick(m);
            else if (integer(m,"turnNumber")!=1 || !m.getAsJsonArray("bulletStates").isEmpty()
                    || m.getAsJsonArray("events").asList().stream().anyMatch(e->e.getAsJsonObject().has("bullet")))
                throw new IllegalStateException("Unexpected round-opening bullet");
        }
        super.event(m);
    }
    @Override void endRound() {
        if (pending!=null) throw new IllegalStateException("Request missing next tick");
        trials.forEach(this::summarize); trials.clear(); physical.clear(); baseRay=Double.NaN;
        super.endRound();
    }

    public static void main(String[] args) throws Exception {
        if (args.length!=2 || !Set.of("baseline","shadow").contains(args[1]))
            throw new IllegalArgumentException("Evidence directory and baseline|shadow required");
        Path server=Path.of(MathKt.class.getProtectionDomain().getCodeSource().getLocation().toURI());
        if (!sha(server).equals("16c277795bd823c9eda0b650f1d09a418ff35cfe43d59b820c52d5e6816f905c"))
            throw new IllegalArgumentException("Engine hash mismatch");
        boolean shadow=args[1].equals("shadow");
        var games=readGames(Path.of(args[0]),()->new MidpointReplay(shadow));
        Map<String,Map<String,Map<String,Integer>>> out=new TreeMap<>();
        games.forEach((k,v)->out.put(k,v.results));
        var totals=new HashMap<String,Integer>();
        out.values().forEach(g->g.values().forEach(c->c.forEach((k,v)->totals.merge(k,v,Integer::sum))));
        if (totals.get("shots")!=4042 || totals.get("observedIntended")!=620 || totals.get("observedCollateralEvents")!=43)
            throw new IllegalStateException("Incomplete observed attribution");
        System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(out));
    }
}
