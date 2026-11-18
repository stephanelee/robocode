import com.google.gson.*;
import dev.robocode.tankroyale.server.model.*;
import java.nio.file.Path;
import java.util.*;

/** Two separate file-only passes: the accepted gun is a calibrated control for the higher-power gun. */
public final class LongRangePowerReplay extends CrossfireReplay {
    record Pending(double power,double ray,double x,double y,boolean boosted) {}
    static class Stats {
        int shots, hits, boosts, openAtRoundEnd, matchedSlots, newSlots, displacedSlots;
        double powerSpent, rawDamage, creditedDamage;
        final Map<String,Double> byVictim = new TreeMap<>();
    }
    final boolean higher;
    final MidpointReplay geometry = new MidpointReplay(false);
    final Stats stats = new Stats();
    final Map<Integer,String> names = new HashMap<>();
    final Map<Integer,MidpointReplay.Bullet> field = new HashMap<>(), simulated = new HashMap<>();
    final Set<Integer> replaced = new HashSet<>();
    double heat, ray=Double.NaN, energyDelta, cooling;
    boolean shadowDead;
    int nextId=-1;
    Pending pending;

    LongRangePowerReplay(boolean higher) { super(false); this.higher=higher; }

    static double power(boolean higher,double d,double e,double victimEnergy) {
        if (!higher || d<500 || e<35) return MeleePolicy.power(d,e,victimEnergy);
        double cap = victimEnergy<=4 ? victimEnergy/4+.05 : (victimEnergy+2)/6+.05;
        return Phys.clip(Math.min(1.4,cap),.1,3);
    }
    @Override void onDecision(int t,JsonObject s,Phys.Pose self,MeleePolicy.Opponent target) {
        double physicalRay=number(s,"gunDirection"), physicalHeat=number(s,"gunHeat");
        if (Double.isNaN(ray)) { ray=physicalRay; heat=physicalHeat; }
        if (!higher && (Math.abs(Phys.norm(ray-physicalRay))>1e-6 || Math.abs(heat-physicalHeat)>1e-9
                || Math.abs(energyDelta)>1e-8)) throw new IllegalStateException("Control gun/energy drift");
        double energy=number(s,"energy")+energyDelta;
        if (energy < 0) shadowDead=true;
        if (target!=null && !shadowDead && energy>0) {
            double d=Phys.distance(self.x(),self.y(),target.x,target.y);
            double p=power(higher,d,energy,target.energy);
            var aim=MeleeFire.aim(target,t,self,p,d,ray,width,height);
            if (heat==0 && aim.canFire(t,12,energy))
                pending=new Pending(p,ray,self.x(),self.y(),p>MeleePolicy.power(d,energy,target.energy));
            ray=MidpointReplay.rotate(ray,aim.error());
        }
        heat=pending==null ? Math.max(0,heat-cooling) : Phys.gunHeat(pending.power);
    }
    record OriginalHit(int victim,double damage,double credit) {}
    void resolve(JsonObject msg) {
        int t=integer(msg,"turnNumber"), actualSlot=-1;
        Map<Integer,List<OriginalHit>> hits=new HashMap<>();
        for (var item:msg.getAsJsonArray("events")) {
            var e=item.getAsJsonObject();
            if (!e.has("bullet")) continue;
            var b=e.getAsJsonObject("bullet");
            int id=integer(b,"bulletId"), owner=integer(b,"ownerId");
            switch (string(e,"type")) {
                case "BulletFiredEvent" -> {
                    if (owner==own && previousMelee) {
                        replaced.add(id); actualSlot=id; energyDelta+=number(b,"power");
                        if (!higher && (pending==null || Math.abs(pending.power-number(b,"power"))>1e-9
                                || Math.abs(Phys.norm(pending.ray-number(b,"direction")))>1e-6
                                || Math.hypot(pending.x-number(b,"x"),pending.y-number(b,"y"))>1e-6))
                            throw new IllegalStateException("Control fire mismatch");
                    } else field.put(id,new MidpointReplay.Bullet(id,owner,t,number(b,"power"),number(b,"x"),
                            number(b,"y"),number(b,"direction")));
                }
                case "BulletHitBotEvent" -> {
                    if (replaced.contains(id)) {
                        double damage=number(e,"damage");
                        hits.computeIfAbsent(id,k->new ArrayList<>()).add(new OriginalHit(integer(e,"victimId"),
                                damage,Math.max(0,Math.min(damage,number(e,"energy")+damage))));
                        energyDelta-=3*number(b,"power");
                    }
                }
            }
        }
        if (pending!=null) {
            if (!higher && actualSlot<0) throw new IllegalStateException("Extra control shot");
            int id=higher ? nextId-- : actualSlot;
            simulated.put(id,new MidpointReplay.Bullet(id,own,t,pending.power,pending.x,pending.y,pending.ray));
            stats.shots++; stats.powerSpent+=pending.power;
            if (pending.boosted) stats.boosts++;
            if (actualSlot<0) stats.newSlots++; else stats.matchedSlots++;
            energyDelta-=pending.power;
        } else if (actualSlot>=0) stats.displacedSlots++;
        pending=null;
        Map<Integer,Point> centers=new HashMap<>();
        Map<Integer,Double> energies=new HashMap<>();
        for (var item:msg.getAsJsonArray("botStates")) {
            var s=item.getAsJsonObject(); int id=integer(s,"id");
            centers.put(id,new Point(number(s,"x"),number(s,"y"))); energies.put(id,number(s,"energy"));
        }
        Map<Integer,Line> lines=new HashMap<>();
        for (var group:List.of(field,simulated)) for (var b:group.values()) {
            var segment=b.segment(t);
            if (!geometry.outside(segment.getStart())) lines.put(b.id(),segment);
        }
        Map<Integer,Double> nominal=new HashMap<>(), originalRaw=new HashMap<>(), originalCredit=new HashMap<>();
        for (var group:hits.values()) for (var hit:group) {
            originalRaw.merge(hit.victim,hit.damage,Double::sum);
            originalCredit.merge(hit.victim,hit.credit,Double::sum);
        }
        Set<Integer> removeExternal=new HashSet<>();
        for (var it=simulated.values().iterator();it.hasNext();) {
            var b=it.next(); var contact=geometry.contact(b,t,lines,centers);
            if (!higher) {
                var expected=new TreeSet<Integer>();
                for (var h:hits.getOrDefault(b.id(),List.of())) expected.add(h.victim);
                if (!contact.bots().equals(expected)) throw new IllegalStateException("Control contact mismatch");
            }
            if (!contact.terminal()) continue;
            for (int victim:contact.bots()) {
                nominal.merge(victim,Phys.damage(b.power()),Double::sum);
                stats.hits++; energyDelta+=3*b.power();
            }
            removeExternal.addAll(contact.bullets());
            it.remove();
        }
        for (var e:nominal.entrySet()) {
            int v=e.getKey();
            double raw=originalRaw.getOrDefault(v,0.0),old=originalCredit.getOrDefault(v,0.0);
            double pool=old+(old+1e-9<raw ? 0 : Math.max(0,energies.get(v)));
            double credit=Math.min(e.getValue(),pool);
            stats.rawDamage+=e.getValue();stats.creditedDamage+=credit;
            stats.byVictim.merge(names.get(v),credit,Double::sum);
            if (!higher && Math.abs(credit-old)>1e-7) throw new IllegalStateException("Control damage mismatch");
        }
        field.keySet().removeAll(removeExternal);
        Set<Integer> alive=new HashSet<>();
        for (var item:msg.getAsJsonArray("bulletStates")) alive.add(integer(item.getAsJsonObject(),"bulletId"));
        field.keySet().retainAll(alive);
    }
    @Override void event(JsonObject msg) {
        String kind=string(msg,"type");
        if (kind.equals("GameStartedEventForObserver")) {
            var setup=msg.getAsJsonObject("gameSetup");
            cooling=number(setup,"gunCoolingRate");
            geometry.width=integer(setup,"arenaWidth"); geometry.height=integer(setup,"arenaHeight");
            for (var item:msg.getAsJsonArray("participants")) {
                var p=item.getAsJsonObject();names.put(integer(p,"id"),string(p,"name"));
            }
        }
        if (kind.equals("TickEventForObserver") && integer(msg,"roundNumber")==round) resolve(msg);
        super.event(msg);
    }
    @Override void endRound() {
        if (pending!=null) throw new IllegalStateException("Missing confirmation tick");
        stats.openAtRoundEnd+=simulated.size();
        field.clear();simulated.clear();replaced.clear();
        ray=Double.NaN;energyDelta=0;shadowDead=false;nextId=-1;
        super.endRound();
    }
    public static void main(String[] args) throws Exception {
        if (args.length!=2 || !Set.of("control","higher").contains(args[1])) throw new IllegalArgumentException();
        if (!sha(Path.of(MathKt.class.getProtectionDomain().getCodeSource().getLocation().toURI()))
                .equals("16c277795bd823c9eda0b650f1d09a418ff35cfe43d59b820c52d5e6816f905c")) throw new IllegalStateException("Server hash");
        boolean higher=args[1].equals("higher");
        var games=readGames(Path.of(args[0]),()->new LongRangePowerReplay(higher));
        Map<String,Stats> results=new TreeMap<>();
        games.forEach((k,v)->results.put(k,v.stats));
        if (!higher && (results.values().stream().mapToInt(s->s.shots).sum()!=4042
                || results.values().stream().mapToInt(s->s.hits).sum()!=663
                || Math.abs(results.values().stream().mapToDouble(s->s.creditedDamage).sum()-3071.906787331359)>1e-7))
            throw new IllegalStateException("Control endpoint");
        System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(results));
    }
}
