import java.util.ArrayDeque;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;

/** Pure movement transfer from Breakwater0.3; legal inferred waves and actual matched ray observations only. */
final class BreakwaterSurf {
    record Own(int turn,double x,double y,double heading,double speed,double width,double height) {}
    /** creationTurn is actual bullet creation, not the preceding shot decision/aim turn. */
    record Threat(int creationTurn,double originX,double originY,double bulletSpeed,double reference,double lateral) {
        double escape(){return Math.toDegrees(Math.asin(8/bulletSpeed));}
    }
    record Command(double turnRate,double targetSpeed,int travelSign) {}
    record Pose(double x,double y,double heading,double speed) {}
    private record Path(int side,int switchAfter) {int sign(int elapsed){return elapsed<switchAfter?side:-side;}}
    private static final int SEARCH=90,MEMORY=32;
    private static final double BODY=18;
    private final SplittableRandom random;
    private final ArrayDeque<Double> residuals=new ArrayDeque<>();
    private int side,since,primary,secondary;
    private Path path;

    BreakwaterSurf(long seed){random=new SplittableRandom(seed);startRound();}
    /** Retain same-opponent matched-ray memory; discard the prior round's trajectory commitment. */
    void startRound(){path=null;side=random.nextBoolean()?1:-1;}
    double[] corrections(){return residuals.stream().mapToDouble(Double::doubleValue).toArray();}
    void matched(Threat wave,double actualBulletHeading) {
        if(Math.abs(wave.lateral())<1)return;
        double gf=CombatMath.relativeAngle(actualBulletHeading-wave.reference())*Math.signum(wave.lateral())/wave.escape();
        if(!Double.isFinite(gf))return;
        if(residuals.size()==MEMORY)residuals.removeFirst();
        residuals.addLast(clamp(gf,-1,1));
    }
    Command choose(Own own,double enemyX,double enemyY,List<Threat> waves) {
        var pose=new Pose(own.x(),own.y(),own.heading(),own.speed());
        var threats=waves.stream().filter(w->w.creationTurn()<=own.turn()&&eta(own,w)>=-2)
                .sorted(Comparator.comparingDouble((Threat w)->eta(own,w)).thenComparingInt(Threat::creationTurn)).limit(2).toList();
        int first=threats.isEmpty()?-1:threats.get(0).creationTurn();
        int second=threats.size()<2?-1:threats.get(1).creationTurn();
        if(path==null||primary!=first||secondary!=second||own.turn()-since>=16) {
            path=select(own,pose,enemyX,enemyY,threats);since=own.turn();primary=first;secondary=second;
        }
        side=path.sign(own.turn()-since);
        return travel(pose,enemyX,enemyY,side,own.width(),own.height());
    }
    private static double eta(Own own,Threat w) {
        return (Math.hypot(own.x()-w.originX(),own.y()-w.originY())-BODY)/w.bulletSpeed()
                -(own.turn()-w.creationTurn()+2);
    }
    private Path select(Own own,Pose pose,double ex,double ey,List<Threat> waves) {
        Path[] options={new Path(side,SEARCH+1),new Path(-side,SEARCH+1),new Path(side,5),new Path(side,10)};
        int count=waves.isEmpty()?2:4;double[] scores=new double[count];double best=Double.POSITIVE_INFINITY;
        for(int i=0;i<count;i++){scores[i]=risk(own,pose,ex,ey,waves,options[i]);best=Math.min(best,scores[i]);}
        int seen=0,pick=0;
        for(int i=0;i<count;i++)if(scores[i]<=best+.08&&random.nextInt(++seen)==0)pick=i;
        return options[pick];
    }
    private double risk(Own own,Pose pose,double ex,double ey,List<Threat> waves,Path option) {
        double[] peakDanger=new double[waves.size()];double wallPenalty=0;
        for(int step=1;step<=SEARCH;step++) {
            pose=step(pose,travel(pose,ex,ey,option.sign(step-1),own.width(),own.height()));
            double gap=DuelSurf.wallDistance(pose.x(),pose.y(),own.width(),own.height())-BODY;
            if(gap<0)return 1000-step;
            wallPenalty=Math.max(wallPenalty,Math.max(0,72-gap)/72);
            boolean allPassed=true;
            for(int i=0;i<waves.size();i++) {
                Threat w=waves.get(i);double distance=Math.hypot(pose.x()-w.originX(),pose.y()-w.originY());
                double inner=(own.turn()+step-w.creationTurn()+1)*w.bulletSpeed();
                if(inner>distance+BODY)continue;
                allPassed=false;
                if(inner+w.bulletSpeed()>=distance-BODY)peakDanger[i]=Math.max(peakDanger[i],density(w,pose,distance));
            }
            if(allPassed&&step>=12)break;
        }
        double total=.18*wallPenalty;
        for(int i=0;i<peakDanger.length;i++)total+=peakDanger[i]/(1+i);
        return total;
    }
    private double density(Threat w,Pose pose,double distance) {
        double gf=CombatMath.relativeAngle(CombatMath.bearing(pose.x()-w.originX(),pose.y()-w.originY())-w.reference())
                *(w.lateral()<0?-1:1)/w.escape();
        double width=.10+Math.toDegrees(Math.asin(Math.min(1,BODY/Math.max(BODY,distance))))/w.escape();
        double linear=Math.toDegrees(Math.asin(Math.min(1,Math.abs(w.lateral())/w.bulletSpeed())))/w.escape();
        double sum=kernel(gf,width)+2*kernel(gf-linear,width);int count=3;
        if(Math.abs(w.lateral())>=1){for(double sample:residuals)sum+=kernel(gf-sample,width);count+=residuals.size();}
        return sum/count;
    }
    private static double kernel(double offset,double width){double z=offset/width;return 1/(1+z*z);}
    static Command travel(Pose pose,double ex,double ey,int sign,double width,double height) {
        double range=Math.hypot(ex-pose.x(),ey-pose.y());
        double angle=CombatMath.bearing(ex-pose.x(),ey-pose.y())+sign*(90-clamp((range-430)/7,-32,32));
        double centre=CombatMath.bearing(width/2-pose.x(),height/2-pose.y());
        double toward=Math.signum(CombatMath.relativeAngle(centre-angle));
        for(int i=0;i<18;i++) {
            double radians=Math.toRadians(angle),px=pose.x()+100*Math.cos(radians),py=pose.y()+100*Math.sin(radians);
            if(DuelSurf.wallDistance(px,py,width,height)>=30)break;
            angle+=10*toward;
        }
        double difference=CombatMath.relativeAngle(angle-pose.heading());boolean reverse=Math.abs(difference)>90;
        if(reverse)difference=CombatMath.relativeAngle(difference+180);
        double limit=DuelSurf.maxTurn(pose.speed());
        var motion=new Command(clamp(difference,-limit,limit),reverse?-8:8,sign);
        Pose next=step(pose,motion);
        return DuelSurf.wallDistance(next.x(),next.y(),width,height)<BODY?new Command(motion.turnRate(),0,sign):motion;
    }
    static Pose step(Pose pose,Command command) {
        double speed=DuelSurf.nextSpeed(pose.speed(),command.targetSpeed());
        double radians=Math.toRadians(pose.heading()),limit=DuelSurf.maxTurn(speed);
        return new Pose(pose.x()+speed*Math.cos(radians),pose.y()+speed*Math.sin(radians),
                pose.heading()+clamp(command.turnRate(),-limit,limit),speed);
    }
    private static double clamp(double v,double lo,double hi){return Math.max(lo,Math.min(hi,v));}
}
