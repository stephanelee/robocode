import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Conditional passage targeting with the former three virtual models as insufficient-data fallback. */
final class PressureGun {
    record Plan(double direction, double power, boolean fire, int model) {}
    record AimTrace(int turn,double x,double y,double gun,double energy,double heat,double chosen,double power,
                    int model,ContextGun.Query query,int support,double distance) {}
    private DiagnosticTrace diagnostic;
    void diagnostic(DiagnosticTrace trace){diagnostic=trace;context.diagnostic(trace);}
    private record Pending(int turn, int target, double power, double[] angles, int maxAge,
                           ContextGun.Query query,boolean learned) {}
    private static final class Flight {
        final int turn, target, maxAge;
        final double x, y, speed;
        final double[] angles;
        final boolean[] hit = new boolean[3], done = new boolean[3];
        int observed;
        Flight(int turn, int target, double x, double y, double power, double[] angles, int maxAge) {
            this.turn=turn; this.target=target; this.x=x; this.y=y; this.speed=20-3*power;
            this.angles=angles.clone(); this.maxAge=maxAge; observed=turn-1;
        }
    }
    private final int[] attempts = new int[3], hits = new int[3];
    private final List<Flight> flights = new ArrayList<>();
    private final ContextGun context = new ContextGun();
    private DuelGun.Scan latest, previous;
    private int target = -1;
    private Pending pending;
    private int learnedAimDecisions,fallbackAimDecisions,confirmedLearnedShots;

    void startRound() {
        latest=previous=null;pending=null;flights.clear();context.startRound();
        learnedAimDecisions=fallbackAimDecisions=confirmedLearnedShots=0;
    }
    int[] attempts() { return attempts.clone(); }
    int[] hits() { return hits.clone(); }
    DuelGun.Scan latest() { return latest; }
    int learnedAimDecisions() { return learnedAimDecisions; }
    int fallbackAimDecisions() { return fallbackAimDecisions; }
    int confirmedLearnedShots() { return confirmedLearnedShots; }
    ContextGun context() { return context; }
    double turnRate() {
        return previous==null || latest==null || latest.turn()-previous.turn()>3 ? 0
                : CombatMath.relativeAngle(latest.direction()-previous.direction())/(latest.turn()-previous.turn());
    }

    void onScan(DuelGun.Scan scan) {
        if (scan==null || scan.targetId()<1 || !finite(scan.x(),scan.y(),scan.direction(),scan.speed(),scan.energy())) return;
        if (scan.targetId()!=target) {
            target=scan.targetId(); Arrays.fill(attempts,0); Arrays.fill(hits,0);
            latest=previous=null;pending=null;flights.clear();
        }
        if (latest!=null && scan.turn()<=latest.turn()) return;
        previous=latest; latest=scan; context.onScan(scan);score(scan);
    }

    private void score(DuelGun.Scan scan) {
        flights.removeIf(f -> {
            if (scan.turn()==f.observed) return false;
            if (scan.targetId()!=f.target || scan.turn()!=f.observed+1 || scan.turn()-f.turn>f.maxAge) return true;
            f.observed=scan.turn();
            double lo=f.speed*(scan.turn()-f.turn+1), hi=lo+f.speed;
            double range=Math.hypot(scan.x()-f.x,scan.y()-f.y);
            boolean complete=true;
            for (int model=0;model<3;model++) {
                if (!f.done[model]) {
                    f.hit[model]=DuelGun.segmentDistance(scan.x(),scan.y(),f.x,f.y,f.angles[model],lo,hi)<=18+1e-9;
                    f.done[model]=f.hit[model] || lo>range+18;
                }
                complete &= f.done[model];
            }
            if (complete) for (int model=0;model<3;model++) {
                attempts[model]++; if(f.hit[model]) hits[model]++;
                if(diagnostic!=null)diagnostic.virtual(scan.turn(),f.turn,model,f.hit[model]);
            }
            return complete;
        });
    }

    void onFired(int turn,double x,double y,double power) {
        onFired(turn,-1,x,y,power,Double.NaN);
    }
    void onFired(int turn,long bulletId,double x,double y,double power,double direction) {
        Pending p=pending; pending=null;
        boolean matched=p!=null && p.turn()==turn && p.target()==target && Math.abs(p.power()-power)<=1e-9;
        if(diagnostic!=null)diagnostic.shot(turn,bulletId,x,y,power,direction,matched);
        context.confirmed(matched?p.query():null,turn,x,y,power);
        if(!matched)return;
        if(p.learned())confirmedLearnedShots++;
        if (flights.size()>=48) flights.remove(0); // An unobserved/discarded cohort is not a miss.
        flights.add(new Flight(turn,target,x,y,power,p.angles(),p.maxAge()));
        if (latest!=null && latest.turn()==turn) score(latest);
    }

    Plan decide(int now,double x,double y,double gun,double energy,double heat,double width,double height) {
        if (latest==null || now<latest.turn() || now-latest.turn()>1) {
            pending=null; return new Plan(gun,0,false,1);
        }
        double power=power(energy,latest.energy(),Math.hypot(latest.x()-x,latest.y()-y));
        if (power<0.1) { pending=null; return new Plan(gun,0,false,1); }
        int selected=1;
        double best=(hits[1]+1.0)/(attempts[1]+2.0)+0.02;
        if (attempts[1]>=8) for(int model=0;model<3;model++) {
            double value=(hits[model]+1.0)/(attempts[model]+2.0);
            if (attempts[model]>=8 && value>best) { selected=model; best=value; }
        }
        double[] angles=new double[3], ranges=new double[3];
        int horizon=(int)Math.ceil(Math.hypot(width,height)/(20-3*power))+3;
        for(int model=0;model<3;model++) {
            double[] point={latest.x(),latest.y()};
            for(int tick=1;tick<=horizon;tick++) {
                point=DuelGun.forecast(latest,now-latest.turn()+tick,model+1,turnRate(),width,height);
                if (Math.hypot(point[0]-x,point[1]-y)<=(20-3*power)*(tick+1)+18) break;
            }
            angles[model]=CombatMath.bearing(point[0]-x,point[1]-y);
            ranges[model]=Math.hypot(point[0]-x,point[1]-y);
        }
        var query=context.query(now,latest,previous,x,y,power,angles[1],ranges[1],width,height);
        double learned=context.predict(query);
        boolean useLearned=Double.isFinite(learned);
        double chosen=useLearned?learned:angles[selected],range=useLearned?ranges[1]:ranges[selected];
        if(useLearned)learnedAimDecisions++;else fallbackAimDecisions++;
        // Same estimated radial range as linear; only the conditional angle changes the current-ray gate.
        double tolerance=Math.toDegrees(Math.asin(Math.min(1,17/Math.max(17,range))));
        boolean fire=heat<=1e-9 && Math.abs(CombatMath.relativeAngle(chosen-gun))<=tolerance;
        if(diagnostic!=null) {
            double[] support=context.diagnosticSupport(query);
            diagnostic.aim(new AimTrace(now,x,y,gun,energy,heat,chosen,power,useLearned?3:selected,query,(int)support[0],support[1]));
        }
        pending=fire?new Pending(now+1,target,power,angles,horizon,query,useLearned):null;
        return new Plan(chosen,power,fire,useLearned?3:selected);
    }

    static double power(double ownEnergy,double enemyEnergy,double range) {
        if (!finite(ownEnergy,enemyEnergy,range) || enemyEnergy<0) return 0;
        double available=ownEnergy-(ownEnergy<8?0.25:1.0);
        if (available<0.1) return 0;
        double preferred=range<180?3.0:range<350?1.4:0.6;
        if (ownEnergy<8) preferred=Math.min(preferred,0.6);
        double lethal=enemyEnergy+1e-6;
        double finish=lethal<=4?lethal/4:(lethal+2)/6;
        return Math.min(available,Math.max(0.1,Math.min(preferred,finish)));
    }
    private static boolean finite(double... values) {
        for (double value:values) if(!Double.isFinite(value)) return false;
        return true;
    }
}
