import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/** Bounded, legal inferred-wave surfing. Density is a regularized hit heuristic, not hit probability. */
final class WaveSurfMove {
    static final int BINS=47, HORIZON=90;
    static final double BODY=18, EPS=1e-6;
    record Threat(double originX,double originY,double bulletSpeed,int creationTurn,
                  double referenceBearing,int latSign,double absOwnLatSpeed,double[] matchedBins47) {
        Threat { matchedBins47=matchedBins47==null?null:matchedBins47.clone(); }
        @Override public double[] matchedBins47(){return matchedBins47==null?null:matchedBins47.clone();}
    }
    record Decision(DuelSurf.Command command,boolean plannerUsed,boolean profileUsed,int option) {}
    record State(double x,double y,double heading,double speed) {}
    private record Selected(Threat threat,double time,double[] density,boolean profile) {}
    private WaveSurfMove() {}

    static Decision choose(double x,double y,double heading,double speed,int now,double enemyX,double enemyY,
                           List<Threat> threats,int lastSign,double width,double height) {
        return choose(x,y,heading,speed,now,enemyX,enemyY,threats,lastSign,width,height,null);
    }
    static Decision choose(double x,double y,double heading,double speed,int now,double enemyX,double enemyY,
                           List<Threat> threats,int lastSign,double width,double height,DiagnosticTrace diagnostic) {
        List<Selected> selected=select(x,y,now,threats);
        if(selected.isEmpty()) {
            var c=DuelSurf.choose(x,y,heading,speed,now,enemyX,enemyY,List.of(),lastSign,width,height);
            return new Decision(c,false,false,c.travelSign());
        }
        int first=lastSign<0?-1:1;
        double[] scores=diagnostic==null?null:new double[]{Double.NaN,Double.NaN,Double.NaN};
        Object[] timings=diagnostic==null?null:new Object[3];
        DuelSurf.Command best=null; double risk=Double.POSITIVE_INFINITY;
        for(int option:new int[]{first,-first,0}) {
            double[] timing=diagnostic==null?null:new double[2*selected.size()+2];
            if(timing!=null)java.util.Arrays.fill(timing,-1);
            double candidate=score(new State(x,y,heading,speed),now,enemyX,enemyY,selected,option,width,height,timing);
            if(scores!=null)scores[option+1]=candidate;
            if(timings!=null)timings[option+1]=timing;
            if(best==null || candidate<risk-EPS) {
                best=command(new State(x,y,heading,speed),enemyX,enemyY,option);
                risk=candidate;
            }
        }
        if(!Double.isFinite(risk)) {
            best=DuelSurf.choose(x,y,heading,speed,now,enemyX,enemyY,List.of(),lastSign,width,height);
        }
        if(diagnostic!=null) {
            Object[] summaries=new Object[selected.size()];
            for(int i=0;i<selected.size();i++) {
                var s=selected.get(i);var t=s.threat();
                summaries[i]=new Object[]{t.creationTurn(),t.originX(),t.originY(),t.bulletSpeed(),s.time(),
                        t.referenceBearing(),t.latSign(),s.profile(),t.matchedBins47()==null?0:java.util.Arrays.stream(t.matchedBins47()).sum()};
            }
            var next=step(new State(x,y,heading,speed),best);
            diagnostic.emit("planner",now,"scoresCCW_STOP_CW",scores,"unsafeCCW",!Double.isFinite(scores[0]),
                    "unsafeSTOP",!Double.isFinite(scores[1]),"unsafeCW",!Double.isFinite(scores[2]),"chosen",best.travelSign(),
                    "waves",summaries,"rolloutTimesCCW_STOP_CW",timings,
                    "predictedNext",new double[]{next.x(),next.y(),next.heading(),next.speed()});
        }
        return new Decision(best,true,selected.stream().anyMatch(Selected::profile),best.travelSign());
    }

    private static DuelSurf.Command command(State s,double ex,double ey,int option) {
        return DuelSurf.command(s.heading(),option==0?s.heading():
                DuelSurf.travelDirection(s.x(),s.y(),ex,ey,option),option);
    }

    /** Engine order: request clamped at old speed, new speed, old-heading move, new-speed turn clamp. */
    static State step(State s,DuelSurf.Command c) {
        double request=clamp(CombatMath.relativeAngle(c.bodyHeading()-s.heading()),
                -DuelSurf.maxTurn(s.speed()),DuelSurf.maxTurn(s.speed()));
        double v=DuelSurf.nextSpeed(s.speed(),c.targetSpeed());
        double angle=Math.toRadians(s.heading());
        return new State(s.x()+v*Math.cos(angle),s.y()+v*Math.sin(angle),
                s.heading()+clamp(request,-DuelSurf.maxTurn(v),DuelSurf.maxTurn(v)),v);
    }

    private static List<Selected> select(double x,double y,int now,List<Threat> threats) {
        var out=new ArrayList<Selected>();
        for(Threat t:threats) {
            if(t==null || !Double.isFinite(t.originX()) || !Double.isFinite(t.originY())
                    || !Double.isFinite(t.referenceBearing()) || !Double.isFinite(t.absOwnLatSpeed())
                    || !Double.isFinite(t.bulletSpeed()) || t.bulletSpeed()<11 || t.bulletSpeed()>19.7
                    || t.creationTurn()>now) continue;
            double distance=Math.hypot(x-t.originX(),y-t.originY());
            double innerRadius=t.bulletSpeed()*(now-t.creationTurn()+1);
            if(innerRadius>distance+BODY) continue;
            double time=Math.max(0,(distance-BODY-innerRadius-t.bulletSpeed())/t.bulletSpeed());
            out.add(new Selected(t,time,density(t),usable(t.matchedBins47)));
        }
        // List.sort is stable, so equal time/creation retains original wave insertion order.
        out.sort(Comparator.comparingDouble(Selected::time).thenComparingInt(s->s.threat().creationTurn()));
        return out.subList(0,Math.min(2,out.size()));
    }

    private static double score(State initial,int now,double ex,double ey,List<Selected> waves,
                                int option,double width,double height,double[] timing) {
        State state=initial;
        double[] lower=new double[waves.size()],upper=new double[waves.size()];
        boolean[] contact=new boolean[waves.size()],done=new boolean[waves.size()];
        int remaining=waves.size();
        java.util.Arrays.fill(lower,Double.POSITIVE_INFINITY);
        java.util.Arrays.fill(upper,Double.NEGATIVE_INFINITY);
        for(int k=1;k<=HORIZON;k++) {
            State next=step(state,command(state,ex,ey,option));
            if(next.x()<BODY || next.y()<BODY || next.x()>width-BODY || next.y()>height-BODY
                    || (DuelSurf.wallDistance(next.x(),next.y(),width,height)<30
                    && DuelSurf.wallDistance(next.x(),next.y(),width,height)
                    <DuelSurf.wallDistance(state.x(),state.y(),width,height)-1e-9)) {
                if(timing!=null){timing[timing.length-2]=k;timing[timing.length-1]=k;}
                return Double.POSITIVE_INFINITY;
            }
            if(timing!=null)timing[timing.length-1]=k;
            state=next;
            for(int i=0;i<waves.size();i++) {
                if(done[i]) continue;
                Threat t=waves.get(i).threat();
                double distance=Math.hypot(state.x()-t.originX(),state.y()-t.originY());
                double lo=segmentStart(t,now+k),hi=lo+t.bulletSpeed();
                double half=angularFootprint(distance,lo,hi);
                if(!Double.isNaN(half)) {
                    double escape=Math.toDegrees(Math.asin(8/t.bulletSpeed()));
                    double center=CombatMath.relativeAngle(CombatMath.bearing(state.x()-t.originX(),
                            state.y()-t.originY())-t.referenceBearing())*(t.latSign()<0?-1:1)/escape;
                    double span=Math.toDegrees(half)/escape;
                    lower[i]=Math.min(lower[i],center-span);upper[i]=Math.max(upper[i],center+span);
                    contact[i]=true;
                    if(timing!=null&&timing[2*i]<0)timing[2*i]=k;
                }
                if(lo>distance+BODY) {done[i]=true;remaining--;if(timing!=null)timing[2*i+1]=k;}
            }
            // Future decisions replan. Walls after every selected wave is gone and the existing 20-turn
            // wall horizon has elapsed must not retroactively invalidate an otherwise safe avoidance path.
            if(remaining==0 && k>=DuelSurf.WALL_HORIZON) break;
        }
        double risk=0;
        for(int i=0;i<waves.size();i++) {
            Selected s=waves.get(i);
            // A shell already behind every future position has no avoidable future contact. Unresolved is worst risk.
            double mass=!done[i]?1:contact[i]?integrate(s.density(),lower[i],upper[i]):0;
            double power=(20-s.threat().bulletSpeed())/3;
            risk+=(4*power+2*Math.max(0,power-1))*mass/Math.max(1,s.time());
        }
        return risk;
    }

    /**
     * Pinned 1.4 TurnProcessor increments Bullet.tick before collision. BulletLine spans current to next
     * position, hence creation k checks [v,2v], NOT [0,v]. This is the inner collision-segment radius.
     */
    static double segmentStart(Threat t,int turn){return t.bulletSpeed()*(turn-t.creationTurn()+1);}

    /** Half-angle intersecting a radius-18 circle with the entire radial bullet segment, NaN if disjoint. */
    static double angularFootprint(double distance,double lo,double hi) {
        if(hi<distance-BODY || lo>distance+BODY || hi<0) return Double.NaN;
        if(distance<=BODY) return Math.PI;
        double tangent=Math.sqrt(Math.max(0,distance*distance-BODY*BODY));
        double r=clamp(tangent,lo,hi);
        return Math.acos(clamp((distance*distance+r*r-BODY*BODY)/(2*distance*r),-1,1));
    }

    static boolean usable(double[] bins) {
        if(bins==null || bins.length!=BINS) return false;
        double sum=0;
        for(double v:bins){if(!Double.isFinite(v)||v<0)return false;sum+=v;}
        return Double.isFinite(sum)&&sum>=3;
    }

    static double[] density(Threat t) {
        double[] out=new double[BINS];java.util.Arrays.fill(out,8.0/BINS);
        add(out,23,4);
        double linear=Math.asin(clamp(Math.abs(t.absOwnLatSpeed())/t.bulletSpeed(),0,1))
                /Math.asin(8/t.bulletSpeed());
        add(out,bin(linear),4);
        double total=16;
        if(usable(t.matchedBins47)) {
            double sum=java.util.Arrays.stream(t.matchedBins47).sum(),scale=Math.min(1,32/sum);
            for(int i=0;i<BINS;i++)add(out,i,t.matchedBins47[i]*scale);
            total+=Math.min(32,sum);
        }
        for(int i=0;i<BINS;i++)out[i]/=total;
        return out;
    }
    private static void add(double[] out,int bin,double mass) {
        double norm=bin==0||bin==BINS-1?.75:1;
        for(int j=-1;j<=1;j++)if(bin+j>=0&&bin+j<BINS)out[bin+j]+=mass*(j==0?.5:.25)/norm;
    }
    private static int bin(double gf){return (int)Math.round((clamp(gf,-1,1)+1)*23);}
    static double integrate(double[] density,double lo,double hi) {
        if(hi< -1 || lo>1)return 0;
        // Bin cells overlap the swept body interval; conservative at cell boundaries.
        double mass=0;
        for(int i=0;i<BINS;i++)if(-1+i/23.0+1/46.0>=lo && -1+i/23.0-1/46.0<=hi)mass+=density[i];
        return Math.min(1,mass);
    }
    private static double clamp(double v,double lo,double hi){return Math.max(lo,Math.min(hi,v));}
}
