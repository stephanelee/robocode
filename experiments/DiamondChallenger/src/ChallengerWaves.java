import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;

/** Legal scan-energy inference and hit-only danger memory; no enemy bullet state is read before an actual hit. */
final class ChallengerWaves {
    record Own(int turn,double x,double y,double direction,double speed) {}
    record Pair(DuelGun.Scan previous,DuelGun.Scan current) {}
    record Hit(int turn,int owner,double power,double x,double y,double direction) {}
    record Wave(int owner,int creation,double x,double y,double speed,double reference,int sign,double absLateral,int context) {}
    private final EnemyFireTracker energy=new EnemyFireTracker();
    private final TreeMap<Integer,Own> history=new TreeMap<>();
    private final List<Pair> scans=new ArrayList<>();
    private final List<Hit> hits=new ArrayList<>();
    private final List<Wave> active=new ArrayList<>();
    private final Set<Integer> collisionTurns=new HashSet<>();
    private final double[][] danger=new double[4][47];
    private DuelGun.Scan previous;
    private int target=-1,inferred,matched,unmatched,rejected,totalInferred,totalMatched;
    private DiagnosticTrace diagnostic;
    private boolean closed;
    void diagnostic(DiagnosticTrace trace){diagnostic=trace;}

    synchronized void closeRound(int now,double width,double height){if(!closed){flush(now,width,height);closed=true;}}
    synchronized void startRound() {
        closed=false;
        energy.clear();history.clear();scans.clear();hits.clear();active.clear();collisionTurns.clear();previous=null;
        inferred=matched=unmatched=rejected=0;
        for(double[] bins:danger)for(int i=0;i<bins.length;i++)bins[i]*=.95;
    }
    synchronized void own(Own state) { if(closed)return;history.put(state.turn(),state);while(history.size()>128)history.pollFirstEntry(); }
    synchronized void scan(DuelGun.Scan state) {
        if(closed)return;
        if(state==null || state.targetId()<0 || !Double.isFinite(state.x()) || !Double.isFinite(state.y())
                || !Double.isFinite(state.energy()) || !Double.isFinite(state.direction()) || !Double.isFinite(state.speed()))return;
        if(state.targetId()!=target) {
            target=state.targetId();previous=null;active.clear();scans.clear();
            for(double[] bins:danger)Arrays.fill(bins,0);
        }
        if(previous!=null && state.turn()<=previous.turn())return;
        if(previous!=null)scans.add(new Pair(previous,state));
        previous=state;
    }
    synchronized void damage(int owner,int turn,double value) { if(!closed)energy.recordBulletDamage(owner,turn,value); }
    synchronized void bonus(int owner,int turn,double power) { if(!closed)energy.recordBulletHitBonus(owner,turn,power); }
    synchronized void collision(int turn) { if(!closed)collisionTurns.add(turn); }
    synchronized void hit(Hit hit) {
        if(closed)return;
        if(hit==null || hit.owner()<0 || !Double.isFinite(hit.power()) || hit.power()<.1 || hit.power()>3
                || !Double.isFinite(hit.x()) || !Double.isFinite(hit.y()) || !Double.isFinite(hit.direction())) {
            unmatched++;return;
        }
        hits.add(hit);
    }

    /** All scan pairs are inferred first, then original-turn hit events matched, then old waves pruned. */
    synchronized void flush(int now,double width,double height) {
        if(closed)return;
        scans.sort(java.util.Comparator.comparingInt(pair->pair.current().turn()));
        for(Pair pair:scans) {
            var before=pair.previous();var after=pair.current();
            double power=energy.inferPower(after.targetId(),before.turn(),after.turn(),before.energy(),after.energy());
            Own own=history.get(after.turn()-1);
            if(after.turn()-before.turn()!=1 || own==null || collisionTurns.contains(after.turn())
                    || wallAmbiguous(before,width,height) || wallAmbiguous(after,width,height)) { rejected++;continue; }
            if(power<=0)continue;
            double bearing=CombatMath.bearing(own.x()-before.x(),own.y()-before.y());
            double angle=Math.toRadians(bearing),direction=Math.toRadians(own.direction());
            double lateral=own.speed()*Math.sin(direction-angle);
            double range=Math.hypot(own.x()-before.x(),own.y()-before.y());
            int context=(range<450?0:2)+(Math.abs(lateral)<4?0:1);
            active.add(new Wave(after.targetId(),after.turn(),before.x(),before.y(),20-3*power,
                    bearing,lateral<0?-1:1,Math.abs(lateral),context));inferred++;totalInferred++;
            if(diagnostic!=null)diagnostic.emit("wave",after.turn(),"owner",after.targetId(),"creationTurn",after.turn(),
                    "originX",before.x(),"originY",before.y(),"power",power,"referenceBearing",bearing,
                    "sign",lateral<0?-1:1,"context",context,"aimOwn",new double[]{own.x(),own.y(),own.direction(),own.speed()});
        }
        scans.clear();
        for(Hit hit:hits) {
            Wave best=null;double bestError=Double.POSITIVE_INFINITY;
            for(Wave wave:active) {
                if(wave.owner()!=hit.owner() || hit.turn()<wave.creation() || Math.abs(wave.speed()-(20-3*hit.power()))>.01)continue;
                double error=Math.abs(Math.hypot(hit.x()-wave.x(),hit.y()-wave.y())-radius(wave,hit.turn()));
                if(error<=wave.speed()+18 && error<bestError) { best=wave;bestError=error; }
            }
            if(diagnostic!=null)diagnostic.emit("enemyHit",hit.turn(),"owner",hit.owner(),"power",hit.power(),
                    "x",hit.x(),"y",hit.y(),"direction",hit.direction(),"matched",best!=null,
                    "creationTurn",best==null?null:best.creation(),"radialResidual",best==null?null:bestError);
            if(best==null) {unmatched++;continue;}
            double escape=Math.toDegrees(Math.asin(8/best.speed()));
            double gf=Math.max(-1,Math.min(1,CombatMath.relativeAngle(hit.direction()-best.reference())*best.sign()/escape));
            if(!Double.isFinite(gf)) {unmatched++;continue;}
            double[] bins=danger[best.context()];bins[CombatMath.gfBin(gf,47)]++;
            double mass=Arrays.stream(bins).sum();if(mass>32)for(int i=0;i<47;i++)bins[i]*=32/mass;
            active.remove(best);matched++;totalMatched++;
        }
        hits.clear();
        Own current=history.get(now);
        active.removeIf(wave->now-wave.creation()>110 || current!=null
                && radius(wave,now)>Math.hypot(current.x()-wave.x(),current.y()-wave.y())+36);
        while(active.size()>16)active.remove(0);
        collisionTurns.removeIf(turn->turn<now-128);energy.prune(now);
    }
    static double radius(Wave wave,int turn) {return wave.speed()*(turn-wave.creation()+1);}
    private static boolean wallAmbiguous(DuelGun.Scan scan,double width,double height) {
        return Math.min(Math.min(scan.x()-18,width-18-scan.x()),Math.min(scan.y()-18,height-18-scan.y()))<=8;
    }
    List<WaveSurfMove.Threat> threats() {
        var result=new ArrayList<WaveSurfMove.Threat>();
        for(Wave wave:active)result.add(new WaveSurfMove.Threat(wave.x(),wave.y(),wave.speed(),wave.creation(),wave.reference(),
                wave.sign(),wave.absLateral(),danger[wave.context()].clone()));
        return result;
    }
    double[][] danger() {return Arrays.stream(danger).map(double[]::clone).toArray(double[][]::new);}
    int inferred() {return inferred;} int matched() {return matched;} int unmatched() {return unmatched;}
    int rejected() {return rejected;} int totalInferred() {return totalInferred;} int totalMatched() {return totalMatched;}
}
