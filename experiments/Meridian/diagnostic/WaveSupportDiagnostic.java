import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.*;
import java.lang.reflect.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.zip.GZIPOutputStream;
import static org.mockito.Mockito.*;

/**
 * Frozen component acquisition only: never calls bot start/run/go or a server.
 * Pinned engine geometry below is a Java port with distance instrumentation of
 * Tank Royale model/math.kt, Copyright 2022 Flemming Nørnberg Larsen, Apache-2.0.
 * Original sources/license and exact commit/blob hashes are retained under
 * benchmarks/meridian-wave-support/engine-source and engine-lock.json.
 */
public final class WaveSupportDiagnostic {
    static final int DECISION=13, HORIZON=50;
    static final double WIDTH=800, HEIGHT=600, RADIUS=18;
    static final Gson JSON=new GsonBuilder().serializeNulls().create();
    record State(int id,String placement,int errorIndex,double headingOffset,double power,
                 double sx,double sy,double bearing,Flight.Pose own,boolean anchor) {}
    record Segment(double x1,double y1,double x2,double y2,double distanceSquared,boolean removed,boolean contact,
                   boolean pinnedEngineContact,boolean geometryAgreement) {}
    record Eligibility(double heading,List<String> reasons,Integer historyContact,Integer historyWall) {
        boolean eligible(){return reasons.isEmpty();}
    }
    record RayResult(double heading,boolean eligible,List<String> reasons,Integer firstContact,Integer wallRemoval,
                     Boolean oldAdmitted,Boolean newAdmitted,Double firstContactDistanceSquared) {}
    record Route(int sign,double[] oldScore,double[] newScore,int wallContacts,int oldMultiplicity,
                 int newMultiplicity,Flight.Pose firstPose,List<RayResult> rays,
                 Integer firstMotionDivergence,int motionMismatchRows) {}
    static final class Counts {
        long states,contexts,trajectories,positionRows,historyContactChecks,futureContactChecks,parityComparisons;
        long eligibleStateRays,excludedStateRays,eligibleContexts,actionChanges,eligibleActionChanges;
        long oldOmittedFirstContacts,newOmittedFirstContacts,retiredFutureContacts;
        long geometryClassificationDisagreements;
        long motionReferenceComparisons,motionMismatchRows,motionMismatchContexts,motionInputOutsideRows,motionDivergentTrajectories;
        double maximumMotionErrorX,maximumMotionErrorY,maximumMotionErrorSpeed,maximumMotionErrorHeading;
        long oldSelectedExposures,newSelectedExposures,commonEligibleRayContexts;
        long oldSelectedWallsEligibleContexts,newSelectedWallsEligibleContexts;
        long oldSelectedWallsRayWeighted,newSelectedWallsRayWeighted;
        long oldSelectedWallsAllContexts,newSelectedWallsAllContexts;
        boolean anchorContinuingContact,anchorReverseFirstMiss,anchorOldOmittedAll50,anchorNewAdmittedFirst,anchorFirstPoses;
    }

    static Map<String,Object> row(Object... values) {
        var map=new LinkedHashMap<String,Object>();
        for(int i=0;i<values.length;i+=2)map.put((String)values[i],values[i+1]);
        return map;
    }
    static final class Sink implements AutoCloseable {
        final BufferedWriter output;
        Sink(Path path) throws IOException {
            output=new BufferedWriter(new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(path,
                StandardOpenOption.CREATE_NEW)),StandardCharsets.UTF_8));
        }
        void add(Object object) throws IOException {output.write(JSON.toJson(object));output.newLine();}
        public void close() throws IOException {output.close();}
    }
    static void require(boolean condition,String text) {if(!condition)throw new IllegalStateException(text);}
    static boolean legal(Flight.Pose p) {return p.x()>=18&&p.x()<=782&&p.y()>=18&&p.y()<=582;}
    static Flight.Pose history(State s,int t) {
        int delta=DECISION-t;var p=s.own();
        return new Flight.Pose(p.x()-delta*p.v()*Flight.cos(p.h()),p.y()-delta*p.v()*Flight.sin(p.h()),p.h(),p.v());
    }
    static List<State> states() {
        var list=new ArrayList<State>();
        for(double power:new double[]{.1,.4,1,1.65,1.95,2.8,3}) {
            double speed=20-3*power,currentFar=15*speed,eps=1e-6;
            double[] errors={-(speed+18)-eps,-(speed+18)+eps,-22-eps,-22+eps,18-eps,18+eps,22+eps};
            for(int error=0;error<errors.length;error++)for(double heading:new double[]{0,90,180,270})
                for(double velocity:new double[]{-8,-3,0,3,8})for(String placement:new String[]{"open","wall","corner"}) {
                    double r=currentFar+errors[error],sx,sy,x,y,bearing;
                    if(placement.equals("open")){sx=400;sy=300;x=400+r;y=300;bearing=0;}
                    else if(placement.equals("wall")){x=780;y=300;sx=780-r;sy=300;bearing=0;}
                    else {x=780;y=580;sx=780-r/Math.sqrt(2);sy=580-r/Math.sqrt(2);bearing=45;}
                    list.add(new State(list.size(),placement,error,heading,power,sx,sy,bearing,
                        new Flight.Pose(x,y,bearing+heading,velocity),false));
                }
        }
        require(list.size()==2940,"Cartesian population differs from registration");
        list.add(new State(list.size(),"anchor",-1,270,1.95,400,300,0,new Flight.Pose(600.4,318.1,270,8),true));
        return list;
    }

    /** Independent oracle: pinned Bullet.kt advances near=(turn+1)*v; wall removes near before hit sweep. */
    static Segment segment(State s,int turn,double heading,Flight.Pose own) {
        double v=20-3*s.power(),near=(turn+1)*v,far=(turn+2)*v;
        double cos=Math.cos(Math.toRadians(heading)),sin=Math.sin(Math.toRadians(heading));
        double x1=s.sx()+cos*near,y1=s.sy()+sin*near,x2=s.sx()+cos*far,y2=s.sy()+sin*far;
        boolean removed=x1<=0||y1<=0||x1>=WIDTH||y1>=HEIGHT;
        double dx=x2-x1,dy=y2-y1,length=dx*dx+dy*dy;
        double alpha=((own.x()-x1)*dx+(own.y()-y1)*dy)/length;
        double clipped=Math.max(0,Math.min(1,alpha));
        double ex=x1+clipped*dx-own.x(),ey=y1+clipped*dy-own.y();
        double distance=ex*ex+ey*ey;
        boolean distanceContact=distance<=RADIUS*RADIUS;
        boolean pinnedContact=pinnedIntersection(x1,y1,x2,y2,own.x(),own.y());
        return new Segment(x1,y1,x2,y2,distance,removed,!removed&&distanceContact,!removed&&pinnedContact,distanceContact==pinnedContact);
    }

    /** Literal independent port of pinned model/math.kt; retain its epsilon/projection operation order. */
    static boolean pinnedIntersection(double x1,double y1,double x2,double y2,double cx,double cy) {
        double ex=x1-cx,ey=y1-cy;
        if(ex*ex+ey*ey<=324)return true;
        ex=x2-cx;ey=y2-cy;if(ex*ex+ey*ey<=324)return true;
        double dx=x2-x1,dy=y2-y1,len2=dx*dx+dy*dy;
        double dot=((cx-x1)*dx+(cy-y1)*dy)/len2;
        double px=x1+dot*dx,py=y1+dot*dy;
        double cross=(px-x1)*dy-(py-y1)*dx;
        if(!(Math.abs(cross)<1e-6))return false;
        boolean onSegment=Math.abs(dx)>=Math.abs(dy)
            ?(dx>0?px>=x1&&px<=x2:px>=x2&&px<=x1)
            :(dy>0?py>=y1&&py<=y2:py>=y2&&py<=y1);
        ex=px-cx;ey=py-cy;return onSegment&&ex*ex+ey*ey<=324;
    }

    /** Independent literal rules/math.kt + MutableBot motion + TWO CollisionDetector passes. */
    static double engineSpeed(double current,double target) {
        if(current<0)return -engineSpeed(-current,-target);
        if(current>0) {
            if(Math.signum(current)==Math.signum(target)) {
                double difference=target-current;
                return difference>=0?Math.min(current+Math.min(difference,1),8):Math.max(current+Math.max(difference,-2),-8);
            }
            return target==0?0:(1-current/2)*-1;
        }
        double acceleration=Math.min(Math.abs(target-current),1);
        return target-current>=0?Math.min(current+acceleration,8):Math.max(current-acceleration,-8);
    }
    static double[] constrain(double oldX,double oldY,double x,double y) {
        double nx=x,ny=y;
        if(x-18<0) {nx=18;double dx=x-oldX;if(dx!=0)ny=oldY+(y-oldY)*(nx-oldX)/dx;}
        else if(x+18>WIDTH) {nx=WIDTH-18;double dx=x-oldX;if(dx!=0)ny=oldY+(y-oldY)*(nx-oldX)/dx;}
        if(y-18<0) {ny=18;double dy=y-oldY;if(dy!=0)nx=oldX+(x-oldX)*(ny-oldY)/dy;}
        else if(y+18>HEIGHT) {ny=HEIGHT-18;double dy=y-oldY;if(dy!=0)nx=oldX+(x-oldX)*(ny-oldY)/dy;}
        return new double[]{nx,ny};
    }
    static Flight.Pose engineStep(Flight.Pose incoming,Flight.Command command) {
        double speed=engineSpeed(incoming.v(),command.speed());
        double radians=Math.toRadians(incoming.h());
        double x=incoming.x()+Math.cos(radians)*speed,y=incoming.y()+Math.sin(radians)*speed;
        double cap=10-.75*Math.abs(Math.max(-8,Math.min(8,speed)));
        double heading=(incoming.h()+Math.max(-cap,Math.min(cap,command.turn())))%360;
        if(heading<0)heading+=360;
        double[] first=constrain(incoming.x(),incoming.y(),x,y);
        if(first[0]!=x||first[1]!=y)speed=0;
        double[] second=constrain(incoming.x(),incoming.y(),first[0],first[1]);
        return new Flight.Pose(second[0],second[1],heading,speed);
    }

    static List<Eligibility> eligibility(State s,Sink historyRows,Counts totals) throws IOException {
        var reasons=new ArrayList<String>();var history=new ArrayList<Flight.Pose>();
        for(int t=-1;t<=DECISION;t++) {
            var p=history(s,t);history.add(p);if(!legal(p))reasons.add("historyOutsideArena@"+t);
        }
        var result=new ArrayList<Eligibility>();
        double[] headings=s.anchor()?new double[]{0}:new double[]{s.bearing()-30,s.bearing()-15,s.bearing(),s.bearing()+15,s.bearing()+30};
        for(double heading:headings) {
            var excluded=new ArrayList<>(reasons);Integer hit=null,wall=null;var checks=new ArrayList<Object>();
            for(int t=0;t<=DECISION;t++) {
                Segment q=segment(s,t,heading,history.get(t+1));totals.historyContactChecks++;
                if(!q.geometryAgreement())totals.geometryClassificationDisagreements++;
                checks.add(row("turn",t,"pose",history.get(t+1),"oracle",q));
                if(q.removed()){wall=t;excluded.add("historyBulletWall@"+t);break;}
                if(q.contact()){hit=t;excluded.add("historyContact@"+t);break;}
            }
            var e=new Eligibility(heading,List.copyOf(excluded),hit,wall);result.add(e);
            if(e.eligible())totals.eligibleStateRays++;else totals.excludedStateRays++;
            historyRows.add(row("stateID",s.id(),"heading",heading,"historyPosesTurnsMinus1To13",history,
                "eligible",e.eligible(),"reasons",excluded,"historyOracle",checks));
        }
        return result;
    }

    static HitAngles.Wave wave(State s) {
        var atLaunch=history(s,-1);
        double center=Flight.angle(atLaunch.x()-s.sx(),atLaunch.y()-s.sy());
        double lateral=atLaunch.v()*Flight.sin(atLaunch.h()-center);
        double lead=center+Math.toDegrees(Math.asin(lateral/(20-3*s.power())));
        return new HitAngles.Wave(0,s.sx(),s.sy(),center,lead,s.power());
    }
    static boolean retired(State s) {
        return Flight.farEdge(s.power(),DECISION+1)>Math.hypot(s.own().x()-s.sx(),s.own().y()-s.sy())+36;
    }
    static final class ActualPolicy {
        final Meridian bot;
        final Method route;
        final Field scan,side,waves;
        ActualPolicy() throws Exception {
            bot=spy(new Meridian());
            doReturn(800).when(bot).getArenaWidth();doReturn(600).when(bot).getArenaHeight();
            route=Meridian.class.getDeclaredMethod("route",Flight.Pose.class,int.class,int.class);route.setAccessible(true);
            scan=field("scan");side=field("side");waves=field("waves");
        }
        static Field field(String name) throws Exception {var f=Meridian.class.getDeclaredField(name);f.setAccessible(true);return f;}
        @SuppressWarnings("unchecked")
        void state(State s,HitAngles.Wave wave) throws Exception {
            scan.set(bot,new Meridian.Scan(DECISION,new Flight.Pose(s.sx(),s.sy(),0,0),100-s.power(),0));
            var list=(List<HitAngles.Wave>)waves.get(bot);list.clear();if(!retired(s))list.add(wave);
        }
        double score(State s,int committed,int sign) throws Exception {
            side.setInt(bot,committed);
            double result=(double)route.invoke(bot,s.own(),DECISION,sign);
            clearInvocations(bot); // Bound Mockito bookkeeping; no recorded calls enter a score.
            return result;
        }
    }

    static Route trajectory(State s,HitAngles.Wave wave,int sign,List<Eligibility> eligible,Sink positions,Counts totals) throws IOException {
        var p=s.own();boolean retired=retired(s);var cold=new HitAngles();
        double[] oldScore={sign==-1?0:2,sign==1?0:2},newScore=oldScore.clone();
        int oldCount=0,newCount=0,walls=0;Flight.Pose first=null;
        Integer firstDivergence=null;int motionMismatches=0;
        Integer[] hits=new Integer[eligible.size()],removed=new Integer[eligible.size()];
        Boolean[] oldAtHit=new Boolean[eligible.size()],newAtHit=new Boolean[eligible.size()];
        Double[] distanceAtHit=new Double[eligible.size()];
        for(int step=1;step<=HORIZON;step++) {
            var before=p;var command=Flight.orbit(before,s.sx(),s.sy(),sign,WIDTH,HEIGHT);
            double velocity=Flight.speed(before.v(),command.speed());
            double unrestrictedX=before.x()+Flight.cos(before.h())*velocity;
            double unrestrictedY=before.y()+Flight.sin(before.h())*velocity;
            p=Flight.step(before,command,WIDTH,HEIGHT);if(step==1)first=p;
            var reference=engineStep(before,command);
            totals.motionReferenceComparisons++;
            double errorX=Math.abs(p.x()-reference.x()),errorY=Math.abs(p.y()-reference.y());
            double errorSpeed=Math.abs(p.v()-reference.v()),errorHeading=Math.abs(Math.IEEEremainder(p.h()-reference.h(),360));
            boolean inputOutside=!legal(before),motionMismatch=errorX>1e-9||errorY>1e-9||errorSpeed>1e-9||errorHeading>1e-9;
            if(inputOutside)totals.motionInputOutsideRows++;
            if(motionMismatch) {totals.motionMismatchRows++;motionMismatches++;if(firstDivergence==null)firstDivergence=step;}
            totals.maximumMotionErrorX=Math.max(totals.maximumMotionErrorX,errorX);totals.maximumMotionErrorY=Math.max(totals.maximumMotionErrorY,errorY);
            totals.maximumMotionErrorSpeed=Math.max(totals.maximumMotionErrorSpeed,errorSpeed);totals.maximumMotionErrorHeading=Math.max(totals.maximumMotionErrorHeading,errorHeading);
            boolean wall=p.x()!=unrestrictedX||p.y()!=unrestrictedY;if(wall)walls++;
            double edge=Math.min(Math.min(p.x(),WIDTH-p.x()),Math.min(p.y(),HEIGHT-p.y()));
            double wallCost=Math.max(0,40-edge)*.10;
            double dx=p.x()-wave.x(),dy=p.y()-wave.y(),r=Math.hypot(dx,dy);
            double far=Flight.farEdge(wave.power(),DECISION+step-wave.t()+1);
            boolean oldMask=Math.abs(r-far)<=22,newMask=far-wave.speed()-18<=r&&r<=far+18;
            double risk=cold.risk(wave,dx,dy);
            if(oldMask)oldCount++;if(newMask)newCount++;
            for(int committed=0;committed<2;committed++) {
                oldScore[committed]+=wallCost;newScore[committed]+=wallCost;
                if(!retired&&oldMask)oldScore[committed]+=risk;
                if(!retired&&newMask)newScore[committed]+=risk;
            }
            var rayRows=new ArrayList<Object>();
            for(int ray=0;ray<eligible.size();ray++) {
                var e=eligible.get(ray);
                if(!e.eligible()||hits[ray]!=null||removed[ray]!=null) {
                    rayRows.add(row("heading",e.heading(),"status",!e.eligible()?"ineligible":hits[ray]!=null?"stoppedAfterContact":"stoppedAfterWall"));continue;
                }
                var oracle=segment(s,DECISION+step,e.heading(),p);totals.futureContactChecks++;
                if(!oracle.geometryAgreement())totals.geometryClassificationDisagreements++;
                rayRows.add(row("heading",e.heading(),"status","checked","oracle",oracle));
                if(oracle.removed())removed[ray]=step;
                else if(oracle.contact()) {
                    hits[ray]=step;oldAtHit[ray]=oldMask;newAtHit[ray]=newMask;distanceAtHit[ray]=oracle.distanceSquared();
                    if(!oldMask)totals.oldOmittedFirstContacts++;
                    if(!newMask)totals.newOmittedFirstContacts++;
                    if(retired)totals.retiredFutureContacts++;
                }
            }
            positions.add(row("stateID",s.id(),"sign",sign,"step",step,"turn",DECISION+step,"before",before,
                "command",command,"pose",p,"unrestrictedX",unrestrictedX,"unrestrictedY",unrestrictedY,"wallContact",wall,
                "motionReferencePose",reference,"motionErrorX",errorX,"motionErrorY",errorY,"motionErrorSpeed",errorSpeed,
                "motionErrorWrappedHeading",errorHeading,"motionInputOutside",inputOutside,"motionMismatch",motionMismatch,
                "firstMotionDivergenceStep",firstDivergence,
                "wallCost",wallCost,"radius",r,"far",far,"speed",wave.speed(),"radialError",r-far,
                "oldAdmitted",oldMask,"newAdmitted",newMask,"rawRisk",risk,"retired",retired,
                "oldRawContribution",oldMask?risk:0,"newRawContribution",newMask?risk:0,
                "oldPolicyContribution",!retired&&oldMask?risk:0,"newPolicyContribution",!retired&&newMask?risk:0,
                "oldMultiplicity",oldCount,"newMultiplicity",newCount,"oldAccumByCommittedMinusPlus",oldScore.clone(),
                "oldPolicyMultiplicity",retired?0:oldCount,"newPolicyMultiplicity",retired?0:newCount,
                "newAccumByCommittedMinusPlus",newScore.clone(),"rayChecks",rayRows));
            totals.positionRows++;
        }
        var rays=new ArrayList<RayResult>();
        for(int i=0;i<eligible.size();i++) {
            var e=eligible.get(i);rays.add(new RayResult(e.heading(),e.eligible(),e.reasons(),hits[i],removed[i],oldAtHit[i],newAtHit[i],distanceAtHit[i]));
        }
        totals.trajectories++;
        if(firstDivergence!=null)totals.motionDivergentTrajectories++;
        return new Route(sign,oldScore,newScore,walls,oldCount,newCount,first,rays,firstDivergence,motionMismatches);
    }
    static void compareBits(double expected,double actual,String label) {
        require(Double.doubleToRawLongBits(expected)==Double.doubleToRawLongBits(actual),"Source parity mismatch: "+label+" expected="+expected+" actual="+actual);
    }
    static int argmin(double minus,double plus) {return plus<minus?1:-1;}
    static Route chosen(Route minus,Route plus,int sign) {return sign==1?plus:minus;}
    static Map<String,Object> process(State s,ActualPolicy actual,Sink states,Sink history,Sink positions,Sink contexts,Counts count) throws Exception {
        require(legal(s.own()),"Illegal current state "+s.id());
        var wave=wave(s);var eligibility=eligibility(s,history,count);actual.state(s,wave);
        states.add(row("state",s,"wavePrior",wave,"currentFar",Flight.farEdge(s.power(),DECISION+1),"retired",retired(s),
                       "eligibleRays",eligibility));
        var minus=trajectory(s,wave,-1,eligibility,positions,count);
        var plus=trajectory(s,wave,1,eligibility,positions,count);
        for(int sideIndex=0;sideIndex<2;sideIndex++) {
            int side=sideIndex==0?-1:1;
            double actualMinus=actual.score(s,side,-1),actualPlus=actual.score(s,side,1);
            compareBits(actualMinus,minus.oldScore()[sideIndex],s.id()+" side="+side+" sign=-1");
            compareBits(actualPlus,plus.oldScore()[sideIndex],s.id()+" side="+side+" sign=1");count.parityComparisons+=2;
            int oldChoice=argmin(minus.oldScore()[sideIndex],plus.oldScore()[sideIndex]);
            int newChoice=argmin(minus.newScore()[sideIndex],plus.newScore()[sideIndex]);
            var oldRoute=chosen(minus,plus,oldChoice);
            var newRoute=chosen(minus,plus,newChoice);
            int eligibleCount=(int)eligibility.stream().filter(Eligibility::eligible).count();
            int oldExposure=0,newExposure=0;
            for(int ray=0;ray<eligibility.size();ray++)if(eligibility.get(ray).eligible()) {
                if(oldRoute.rays().get(ray).firstContact()!=null)oldExposure++;
                if(newRoute.rays().get(ray).firstContact()!=null)newExposure++;
            }
            count.contexts++;if(oldChoice!=newChoice)count.actionChanges++;
            if(minus.motionMismatchRows()>0||plus.motionMismatchRows()>0)count.motionMismatchContexts++;
            count.oldSelectedWallsAllContexts+=oldRoute.wallContacts();count.newSelectedWallsAllContexts+=newRoute.wallContacts();
            if(eligibleCount>0) {
                count.eligibleContexts++;if(oldChoice!=newChoice)count.eligibleActionChanges++;
                count.oldSelectedWallsEligibleContexts+=oldRoute.wallContacts();count.newSelectedWallsEligibleContexts+=newRoute.wallContacts();
            }
            count.commonEligibleRayContexts+=eligibleCount;count.oldSelectedExposures+=oldExposure;count.newSelectedExposures+=newExposure;
            count.oldSelectedWallsRayWeighted+=(long)oldRoute.wallContacts()*eligibleCount;
            count.newSelectedWallsRayWeighted+=(long)newRoute.wallContacts()*eligibleCount;
            contexts.add(row("stateID",s.id(),"committedSide",side,"switchCostMinus",side==-1?0:2,"switchCostPlus",side==1?0:2,
                "actualSourceScoresMinusPlus",new double[]{actualMinus,actualPlus},"helperOldScoresMinusPlus",new double[]{minus.oldScore()[sideIndex],plus.oldScore()[sideIndex]},
                "newScoresMinusPlus",new double[]{minus.newScore()[sideIndex],plus.newScore()[sideIndex]},"oldScoreBitsMinusPlus",new long[]{Double.doubleToRawLongBits(actualMinus),Double.doubleToRawLongBits(actualPlus)},
                "oldChoice",oldChoice,"newChoice",newChoice,"eligibleRayCount",eligibleCount,"oldSelectedExposures",oldExposure,"newSelectedExposures",newExposure,
                "oldSelectedWallContacts",oldRoute.wallContacts(),"newSelectedWallContacts",newRoute.wallContacts(),
                "minusFirstMotionDivergenceStep",minus.firstMotionDivergence(),"plusFirstMotionDivergenceStep",plus.firstMotionDivergence(),
                "minusMotionMismatchRows",minus.motionMismatchRows(),"plusMotionMismatchRows",plus.motionMismatchRows(),
                "minusRouteRays",minus.rays(),"plusRouteRays",plus.rays()));
        }
        count.states++;
        if(s.anchor()) {
            count.anchorContinuingContact=Integer.valueOf(1).equals(plus.rays().get(0).firstContact());
            count.anchorReverseFirstMiss=!Integer.valueOf(1).equals(minus.rays().get(0).firstContact());
            count.anchorOldOmittedAll50=minus.oldMultiplicity()==0&&plus.oldMultiplicity()==0;
            count.anchorNewAdmittedFirst=Boolean.TRUE.equals(plus.rays().get(0).newAdmitted());
            count.anchorFirstPoses=Math.abs(plus.firstPose().x()-600.4)<1e-10&&Math.abs(plus.firstPose().y()-310.1)<1e-10
                &&Math.abs(minus.firstPose().x()-600.4)<1e-10&&Math.abs(minus.firstPose().y()-315.1)<1e-10;
            return row("minusRoute",minus,"plusRoute",plus);
        }
        return null;
    }
    public static void main(String[] args) throws Exception {
        require(args.length==1,"Output directory argument required");
        Path output=Path.of(args[0]);require(Files.isDirectory(output),"Wrapper must create one empty attempt directory");
        var totals=new Counts();Map<String,Object> anchor=null;
        try(var states=new Sink(output.resolve("states.jsonl.gz"));var history=new Sink(output.resolve("history-rays.jsonl.gz"));
            var positions=new Sink(output.resolve("positions.jsonl.gz"));var contexts=new Sink(output.resolve("contexts.jsonl.gz"))) {
            var actual=new ActualPolicy();
            for(var state:states()) {var found=process(state,actual,states,history,positions,contexts,totals);if(found!=null)anchor=found;}
        } catch(Throwable failure) {
            Files.writeString(output.resolve("diagnostic-failure.json"),JSON.toJson(row("error",failure.toString(),"partialCounts",totals)),StandardOpenOption.CREATE_NEW);
            throw failure;
        }
        boolean counts=totals.states==2941&&totals.contexts==5882&&totals.trajectories==5882&&totals.positionRows==294100
            &&totals.parityComparisons==11764&&totals.motionReferenceComparisons==294100
            &&totals.eligibleStateRays+totals.excludedStateRays==14701&&totals.futureContactChecks<=1470100;
        boolean anchorPass=totals.anchorContinuingContact&&totals.anchorReverseFirstMiss&&totals.anchorOldOmittedAll50&&totals.anchorNewAdmittedFirst&&totals.anchorFirstPoses;
        boolean valid=counts&&totals.geometryClassificationDisagreements==0;
        boolean pass=valid&&anchorPass&&totals.newOmittedFirstContacts==0&&totals.retiredFutureContacts==0;
        boolean advance=pass&&totals.eligibleActionChanges>0&&totals.newSelectedExposures<totals.oldSelectedExposures
            &&totals.newSelectedWallsEligibleContexts<=totals.oldSelectedWallsEligibleContexts&&totals.motionMismatchRows==0;
        Files.writeString(output.resolve("summary.json"),JSON.toJson(row("counts",totals,"countsValid",counts,"anchor",anchor,
            "diagnosticValid",valid,"status",valid?"valid":"invalid",
            "diagnosticPass",pass,"advanceToProposal",advance,"decision",advance?"geometry supports separately registered live proposal":"retain geometry only; reject advancement",
            "limitations","Hypothetical finite in-matrix rays, cold prior, unchanged forecast physics; no games or strength evidence")),StandardOpenOption.CREATE_NEW);
        require(counts,"Frozen row count mismatch");
        require(valid,"Pinned/distance oracle classification disagreement invalidates diagnostic");
    }
}
