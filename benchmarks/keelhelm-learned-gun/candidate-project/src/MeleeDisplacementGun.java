import java.util.*;

/** Legal scan-endpoint learning. Diagnostics are deliberately independent of aiming. */
final class MeleeDisplacementGun {
    static final int CAP = 1024;
    private static final int[] HORIZONS = {8,16,24,32,48,64,80,96};
    private static final double[] WEIGHTS = {2,1,1,.5,3,.75,.75,.75,.75,1,1};
    record Observation(int round, int turn, double x, double y, double heading, double speed,
                       double sx, double sy, double w, double h, double rate, double accel, boolean known) {}
    record Sample(long origin, double horizon, double[] features, double forward, double lateral, long serial) {}
    record Estimate(double forward, double lateral, double variance) {}
    record Aim(double x, double y, boolean learned) {}
    record Pending(int turn, int target, double power, double desired, double circular, boolean learned) {}
    record Shot(int target, double power) {}
    private static final class Model {
        final ArrayDeque<Observation> history = new ArrayDeque<>();
        final ArrayDeque<Sample> samples = new ArrayDeque<>();
        int cachedTurn = -1;
        double cachedPower, cachedHorizon;
        Estimate cached;
    }
    private final Map<Integer,Model> models = new TreeMap<>();
    private final Set<Integer> dead = new HashSet<>();
    private final LinkedHashMap<Integer,Shot> shots = new LinkedHashMap<>();
    private final Map<Integer,Integer> changedTargets = new TreeMap<>();
    private final StringBuilder trace = new StringBuilder();
    private int round, confirmations, changed, hits, walls, collisions, evictions, traceOverflow;
    long queries, fallbacks, samplesAdded, serial;
    double damage;
    private Pending pending;

    void newRound() {
        round++; clearTransient(); dead.clear(); shots.clear(); changedTargets.clear(); trace.setLength(0);
        confirmations = changed = hits = walls = collisions = evictions = traceOverflow = 0;
        queries = fallbacks = samplesAdded = 0; damage = 0;
    }
    void clearTransient() {
        pending = null;
        for (Model m : models.values()) { m.history.clear(); m.cachedTurn = -1; m.cached = null; }
    }
    void died(int id) {
        dead.add(id);
        Model m = models.get(id);
        if (m != null) { m.history.clear(); m.cachedTurn = -1; m.cached = null; }
    }
    int sampleCount(int id) { Model m = models.get(id); return m == null ? 0 : m.samples.size(); }
    static boolean finite(double... values) { for (double v : values) if (!Double.isFinite(v)) return false; return true; }
    void observe(int id, int turn, double x, double y, double heading, double speed,
                 double sx, double sy, double w, double h) {
        if (dead.contains(id) || !finite(x,y,heading,speed,sx,sy,w,h) || w < 36 || h < 36
                || x < 18 || x > w-18 || y < 18 || y > h-18 || Math.abs(speed) > 8) return;
        Model m = models.computeIfAbsent(id, k -> new Model());
        Observation previous = m.history.peekLast();
        if (previous != null && turn <= previous.turn) return;
        int gap = previous == null ? Integer.MAX_VALUE : turn - previous.turn;
        boolean known = gap >= 1 && gap <= 12;
        Observation now = new Observation(round,turn,x,y,heading,speed,sx,sy,w,h,
                known ? Phys.clip(Phys.norm(heading-previous.heading)/gap,-10,10) : 0,
                known ? Phys.clip((speed-previous.speed)/gap,-2,2) : 0,known);
        while (!m.history.isEmpty() && turn-m.history.peekFirst().turn > 99) m.history.removeFirst();
        for (int desired : HORIZONS) {
            Observation best = null; int error = 4;
            for (Observation anchor : m.history) {
                int dt = turn-anchor.turn, e = Math.abs(dt-desired);
                if (anchor.known && dt >= 8 && dt <= 96 && e <= 3 && e <= error) { best = anchor; error = e; }
            }
            if (best == null) continue;
            int dt = turn-best.turn;
            double dx=x-best.x, dy=y-best.y;
            if (Math.hypot(dx,dy) > 8*dt+.001) continue;
            double f=(dx*Phys.cos(best.heading)+dy*Phys.sin(best.heading))/(8*dt);
            double l=(-dx*Phys.sin(best.heading)+dy*Phys.cos(best.heading))/(8*dt);
            m.samples.addLast(new Sample(((long)best.round<<32)|(best.turn&0xffffffffL),dt,features(best,dt),f,l,++serial));
            if (m.samples.size()>CAP) m.samples.removeFirst();
            samplesAdded++;
        }
        m.history.addLast(now);
        if (m.history.size()>16) m.history.removeFirst();
        m.cachedTurn=-1; m.cached=null;
    }
    static double ray(double x, double y, double angle, double w, double h) {
        double dx=Phys.cos(angle), dy=Phys.sin(angle), d=Double.POSITIVE_INFINITY;
        if (dx>1e-12) d=Math.min(d,(w-18-x)/dx);
        if (dx< -1e-12) d=Math.min(d,(18-x)/dx);
        if (dy>1e-12) d=Math.min(d,(h-18-y)/dy);
        if (dy< -1e-12) d=Math.min(d,(18-y)/dy);
        return Phys.clip(d/400,0,1);
    }
    static double[] features(Observation o, double horizon) {
        double relative=o.heading-Phys.bearing(o.x,o.y,o.sx,o.sy);
        return new double[]{(o.speed+8)/16,(o.rate+10)/20,(o.accel+2)/4,
                Math.min(1,Phys.distance(o.x,o.y,o.sx,o.sy)/1200),horizon/96,
                ray(o.x,o.y,o.heading,o.w,o.h),ray(o.x,o.y,o.heading+180,o.w,o.h),
                ray(o.x,o.y,o.heading+90,o.w,o.h),ray(o.x,o.y,o.heading-90,o.w,o.h),
                (Phys.cos(relative)+1)/2,(Phys.sin(relative)+1)/2};
    }
    private Estimate nearest(Model m, Observation o, double horizon) {
        queries++;
        double[] query=features(o,horizon), distances=new double[24];
        Sample[] nearest=new Sample[24]; int count=0;
        for (Sample s : m.samples) {
            if (Math.abs(s.horizon-horizon)>12) continue;
            double distance=0;
            for (int i=0;i<WEIGHTS.length;i++) { double d=query[i]-s.features[i]; distance+=WEIGHTS[i]*d*d; }
            distance/=12.5;
            if (distance>.25) continue;
            int duplicate=-1;
            for (int i=0;i<count;i++) if (nearest[i].origin==s.origin) { duplicate=i; break; }
            if (duplicate>=0) {
                if (distances[duplicate]<distance || (distances[duplicate]==distance && nearest[duplicate].serial>s.serial)) continue;
                for (int i=duplicate;i<count-1;i++) { nearest[i]=nearest[i+1]; distances[i]=distances[i+1]; }
                count--;
            }
            int pos=0;
            while (pos<count && (distances[pos]<distance || (distances[pos]==distance && nearest[pos].serial>s.serial))) pos++;
            if (pos>=24) continue;
            int last=Math.min(count,23);
            for (int i=last;i>pos;i--) { nearest[i]=nearest[i-1]; distances[i]=distances[i-1]; }
            nearest[pos]=s; distances[pos]=distance; count=Math.min(24,count+1);
        }
        if (count<16) return null;
        double sum=0,f=0,l=0;
        for (int i=0;i<count;i++) { double weight=1/(.02+distances[i]); sum+=weight; f+=weight*nearest[i].forward; l+=weight*nearest[i].lateral; }
        f/=sum; l/=sum; double variance=0;
        for (int i=0;i<count;i++) { double df=nearest[i].forward-f,dl=nearest[i].lateral-l; variance+=(df*df+dl*dl)/(.02+distances[i]); }
        return new Estimate(f,l,Math.max(0,variance/sum));
    }
    Aim predict(int id, int turn, Phys.Pose self, double power, Phys.Pose circular, double w, double h) {
        Model m=models.get(id); Observation o=m==null?null:m.history.peekLast();
        double speed=Phys.bulletSpeed(power), horizon=o==null?0:turn-o.turn+Phys.distance(self.x(),self.y(),circular.x(),circular.y())/speed;
        if (o==null || !o.known || m.samples.size()<128 || turn-o.turn<0 || turn-o.turn>12 || horizon<8 || horizon>96) return fallback(circular);
        if (m.cachedTurn!=o.turn || Math.abs(m.cachedPower-power)>.1 || Math.abs(m.cachedHorizon-horizon)>=4) {
            m.cached=nearest(m,o,horizon); m.cachedTurn=o.turn; m.cachedPower=power; m.cachedHorizon=horizon;
        }
        Estimate estimate=m.cached;
        if (estimate==null) return fallback(circular);
        double vx=8*(estimate.forward*Phys.cos(o.heading)-estimate.lateral*Phys.sin(o.heading));
        double vy=8*(estimate.forward*Phys.sin(o.heading)+estimate.lateral*Phys.cos(o.heading));
        double x=0,y=0;
        for (int i=0;i<3;i++) {
            x=Phys.clip(o.x+vx*horizon,18,w-18); y=Phys.clip(o.y+vy*horizon,18,h-18);
            horizon=turn-o.turn+Phys.distance(self.x(),self.y(),x,y)/speed;
        }
        if (horizon<8 || horizon>96 || Math.abs(horizon-m.cachedHorizon)>12
                || 8*horizon*Math.sqrt(estimate.variance)>Math.max(36,.25*Phys.distance(self.x(),self.y(),o.x,o.y))) return fallback(circular);
        return new Aim(Phys.clip(o.x+vx*horizon,18,w-18),Phys.clip(o.y+vy*horizon,18,h-18),true);
    }
    private Aim fallback(Phys.Pose p) { fallbacks++; return new Aim(p.x(),p.y(),false); }
    void pending(int turn, int target, double power, double desired, double circular, boolean learned) {
        pending=new Pending(turn+1,target,power,desired,circular,learned);
    }
    void confirm(int turn, int bulletId, double power, double direction) {
        Pending p=pending; pending=null;
        if (p==null || p.turn!=turn || Math.abs(p.power-power)>1e-6) return;
        if (shots.size()>=256) { shots.remove(shots.keySet().iterator().next()); evictions++; }
        shots.put(bulletId,new Shot(p.target,power));
        confirmations++;
        if (p.learned && Math.abs(Phys.norm(p.desired-p.circular))>.1 && Math.abs(Phys.norm(direction-p.circular))>.1) {
            changed++; changedTargets.merge(p.target,1,Integer::sum);
        }
        if (confirmations<=1024) {
            if (trace.length()>0) trace.append(';');
            trace.append(turn).append(',').append(bulletId).append(',').append(p.target).append(',')
                    .append(power).append(',').append(direction).append(',').append(p.desired).append(',')
                    .append(p.circular).append(',').append(p.learned?1:0);
        } else traceOverflow++;
    }
    void hit(int bulletId, int victim, double value) { if (shots.remove(bulletId)!=null) { hits++; damage+=value; } }
    void wall(int bulletId) { if (shots.remove(bulletId)!=null) walls++; }
    void collision(int bulletId) { if (shots.remove(bulletId)!=null) collisions++; }
    String trace() { return trace.length()==0?"none":trace.toString(); }
    String stats() {
        StringJoiner targets=new StringJoiner(","); changedTargets.forEach((id,n)->targets.add(id+":"+n));
        return " learnedConfirmed="+changed+" learnedTargets="+(changedTargets.isEmpty()?"none":targets)
                +" modelQueries="+queries+" modelFallbacks="+fallbacks+" modelSamples="+samplesAdded
                +" confirmedMelee="+confirmations+" confirmedHits="+hits+" confirmedWalls="+walls
                +" confirmedCollisions="+collisions+" confirmedCensored="+(evictions+shots.size())
                +" confirmedDamage="+damage+" traceOverflow="+traceOverflow;
    }
}
