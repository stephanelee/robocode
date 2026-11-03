import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Bounded, past-only conditional wave-passage model. Passage labels are proxies, not collision truth. */
final class ContextGun {
    static final int CAPACITY=512, MIN_SUPPORT=32, K=16, GRID=47;
    private static final double[] WEIGHTS={2,2,1,1,1,1,1};
    record Query(double[] features,double reference,int sign,double escape,double linearGF,double bandwidth,
                 int target,int aimTurn,long createdSequence,double diagonal,double referenceX,double referenceY,int generation) {
        Query { features=features.clone(); }
    }
    record Sample(double[] features,double guessFactor,int aimTurn,int shotTurn,int observedTurn,
                  long createdSequence,long observedSequence,long insertionSequence,int generation,
                  double originX,double originY,double referenceX,double referenceY,double observedX,double observedY,
                  int sign,double escape) {
        Sample { features=features.clone(); }
    }
    private static final class Wave {
        final Query query;
        final int turn,maxAge;
        final double x,y,speed;
        int observed;
        Wave(Query query,int turn,double x,double y,double power) {
            this.query=query;this.turn=turn;this.x=x;this.y=y;speed=20-3*power;
            observed=turn-1;maxAge=(int)Math.ceil(query.diagonal()/speed)+3;
        }
    }
    private final List<Sample> samples=new ArrayList<>();
    private final List<Wave> waves=new ArrayList<>();
    private DuelGun.Scan latest;
    private int target=-1,lastShot=-1,acceptedRound,acceptedTotal,generation;
    private long sequence;
    private DiagnosticTrace diagnostic;
    void diagnostic(DiagnosticTrace trace){diagnostic=trace;}

    void startRound() { waves.clear();latest=null;lastShot=-1;acceptedRound=0;generation++; }
    int size() { return samples.size(); }
    int acceptedRound() { return acceptedRound; }
    int acceptedTotal() { return acceptedTotal; }
    List<Sample> samples() { return List.copyOf(samples); }

    void onScan(DuelGun.Scan scan) {
        if(latest!=null && scan.targetId()==target && scan.turn()<=latest.turn())return;
        if(scan.targetId()!=target) {target=scan.targetId();samples.clear();waves.clear();acceptedTotal=0;lastShot=-1;}
        latest=scan;sequence++;observe(scan);
    }
    private void observe(DuelGun.Scan scan) {
        waves.removeIf(w -> {
            if(scan.turn()==w.observed)return false;
            if(scan.targetId()!=w.query.target() || scan.turn()!=w.observed+1 || scan.turn()-w.turn>w.maxAge)return true;
            w.observed=scan.turn();
            double range=Math.hypot(scan.x()-w.x,scan.y()-w.y);
            // Fixed creation+1 convention: first fresh scan where the wave center reaches the target center.
            if(w.speed*(scan.turn()-w.turn+1)<range)return false;
            double angle=CombatMath.bearing(scan.x()-w.x,scan.y()-w.y);
            // Re-anchor the SAME frozen reference target point to the confirmed actual shot origin.
            double reference=CombatMath.bearing(w.query.referenceX()-w.x,w.query.referenceY()-w.y);
            double gf=clamp(CombatMath.relativeAngle(angle-reference)*w.query.sign()/w.query.escape(),-1,1);
            if(sequence<=w.query.createdSequence())throw new IllegalStateException("Future sample available at query");
            if(samples.size()==CAPACITY)samples.remove(0);
            samples.add(new Sample(w.query.features(),gf,w.query.aimTurn(),w.turn,scan.turn(),
                    w.query.createdSequence(),sequence,++acceptedTotal,w.query.generation(),w.x,w.y,
                    w.query.referenceX(),w.query.referenceY(),scan.x(),scan.y(),w.query.sign(),w.query.escape()));acceptedRound++;
            if(diagnostic!=null)diagnostic.passage(samples.get(samples.size()-1));
            return true;
        });
    }
    void confirmed(Query query,int turn,double x,double y,double power) {
        lastShot=turn;
        if(query==null || query.target()!=target || query.generation()!=generation || query.aimTurn()+1!=turn)return;
        if(waves.size()==48)waves.remove(0);
        waves.add(new Wave(query,turn,x,y,power));
        if(latest!=null && latest.turn()==turn)observe(latest);
    }

    Query query(int now,DuelGun.Scan current,DuelGun.Scan previous,double x,double y,double power,
                double linearAngle,double linearRange,double width,double height) {
        if(current==null || previous==null || current.targetId()!=target || previous.targetId()!=target
                || current.turn()!=now || previous.turn()!=now-1)return null;
        double range=Math.hypot(current.x()-x,current.y()-y),bearing=CombatMath.bearing(current.x()-x,current.y()-y);
        double radians=Math.toRadians(bearing),heading=Math.toRadians(current.direction());
        double vx=Math.cos(heading)*current.speed(),vy=Math.sin(heading)*current.speed();
        double lateral=-Math.sin(radians)*vx+Math.cos(radians)*vy;
        int sign=lateral<0?-1:1; // Includes zero and negative zero: deterministic+1.
        double escape=Math.toDegrees(Math.asin(8/(20-3*power)));
        double margin=Math.min(Math.min(current.x()-18,width-18-current.x()),Math.min(current.y()-18,height-18-current.y()));
        double[] features={clamp(range/(20-3*power)/60,0,1),clamp(Math.abs(lateral)/8,0,1),
                clamp((Math.cos(radians)*vx+Math.sin(radians)*vy)/8,-1,1),
                clamp(sign*CombatMath.relativeAngle(current.direction()-previous.direction())/10,-1,1),
                clamp((Math.abs(current.speed())-Math.abs(previous.speed()))/8,-1,1),clamp(margin/120,0,1),
                lastShot<0?1:clamp((now-lastShot)/40.0,0,1)};
        double prior=clamp(CombatMath.relativeAngle(linearAngle-bearing)*sign/escape,-1,1);
        double bandwidth=Math.max(2.0/(GRID-1),Math.toDegrees(Math.asin(Math.min(1,18/Math.max(18,linearRange))))/escape);
        return new Query(features,bearing,sign,escape,prior,bandwidth,target,now,sequence,Math.hypot(width,height),current.x(),current.y(),generation);
    }

    double predict(Query query) {
        if(query==null || query.target()!=target || query.generation()!=generation || samples.size()<MIN_SUPPORT)return Double.NaN;
        Sample[] near=new Sample[K];double[] distances=new double[K];Arrays.fill(distances,Double.POSITIVE_INFINITY);
        int past=0;
        for(Sample sample:samples) {
            if(sample.observedSequence()>query.createdSequence())continue;
            past++;
            double distance=0;
            for(int j=0;j<7;j++)distance+=WEIGHTS[j]*Math.pow(query.features()[j]-sample.features()[j],2);
            // FIFO iteration plus strict comparisons keeps insertion-order ties stable.
            if(distance>=distances[K-1])continue;
            int at=K-1;
            while(at>0 && distance<distances[at-1]){near[at]=near[at-1];distances[at]=distances[at-1];at--;}
            near[at]=sample;distances[at]=distance;
        }
        if(past<MIN_SUPPORT || near[K-1]==null || distances[K-1]>1)return Double.NaN;
        double[] weights=new double[K];double sum=0;
        for(int i=0;i<K;i++){weights[i]=1/(0.05+distances[i]);sum+=weights[i];}
        for(int i=0;i<K;i++)weights[i]*=K/sum; // Actual neighbor mass16; two linear pseudo-neighbors add mass2.
        double bestGF=query.linearGF(),best=Double.NEGATIVE_INFINITY;
        for(int bin=0;bin<GRID;bin++) {
            double gf=-1+2.0*bin/(GRID-1);
            double density=2*kernel(gf-query.linearGF(),query.bandwidth());
            for(int i=0;i<K;i++)density+=weights[i]*kernel(gf-near[i].guessFactor(),query.bandwidth());
            int comparison=Double.compare(density,best);
            if(comparison>0 || comparison==0 && (Math.abs(gf-query.linearGF())<Math.abs(bestGF-query.linearGF())
                    || Math.abs(gf-query.linearGF())==Math.abs(bestGF-query.linearGF()) && gf<bestGF)) {
                best=density;bestGF=gf;
            }
        }
        return query.reference()+query.sign()*bestGF*query.escape();
    }
    /** Observation-only duplicate distance summary; never used by predict or eligibility. */
    double[] diagnosticSupport(Query query) {
        if(query==null)return new double[]{0,Double.NaN};
        var distances=new ArrayList<Double>();
        for(Sample sample:samples)if(sample.observedSequence()<=query.createdSequence()) {
            double d=0;for(int j=0;j<7;j++)d+=WEIGHTS[j]*Math.pow(query.features()[j]-sample.features()[j],2);
            distances.add(d);
        }
        distances.sort(Double::compare);return new double[]{distances.size(),distances.size()<K?Double.NaN:distances.get(K-1)};
    }
    String snapshot() {
        StringBuilder json=new StringBuilder("[");
        for(Sample s:samples) {
            if(json.length()>1)json.append(',');
            json.append("{\"features\":").append(Arrays.toString(s.features())).append(",\"guessFactor\":").append(s.guessFactor())
                    .append(",\"aimTurn\":").append(s.aimTurn()).append(",\"shotTurn\":").append(s.shotTurn())
                    .append(",\"observedTurn\":").append(s.observedTurn()).append(",\"createdSequence\":").append(s.createdSequence())
                    .append(",\"observedSequence\":").append(s.observedSequence()).append(",\"insertionSequence\":").append(s.insertionSequence())
                    .append(",\"generation\":").append(s.generation()).append(",\"originX\":").append(s.originX()).append(",\"originY\":").append(s.originY())
                    .append(",\"referenceX\":").append(s.referenceX()).append(",\"referenceY\":").append(s.referenceY())
                    .append(",\"observedX\":").append(s.observedX()).append(",\"observedY\":").append(s.observedY())
                    .append(",\"sign\":").append(s.sign()).append(",\"escape\":").append(s.escape()).append('}');
        }
        return json.append(']').toString();
    }
    private static double kernel(double difference,double bandwidth){return Math.max(0,1-Math.abs(difference)/bandwidth);}
    private static double clamp(double value,double low,double high){return Math.max(low,Math.min(high,value));}
}
