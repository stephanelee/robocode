/** Source69's three reachable-path footprints, tested only against the actual current AIS gun ray. */
final class BreakwaterFireGuard {
    record Input(int turn,double ownX,double ownY,double width,double height,double gunDirection,double power,DuelGun.Scan target) {}
    record Result(int coveredPaths,int totalPaths,boolean allow) {}
    record Pose(double x,double y,double heading,double speed) {}
    private static final double BODY=18;
    private static final int HORIZON=90;
    private BreakwaterFireGuard() {}

    static Result evaluate(Input in) {
        DuelGun.Scan target=in.target();
        if(target==null||in.turn()<target.turn()||in.turn()-target.turn()>1
                ||!finite(in.ownX(),in.ownY(),in.width(),in.height(),in.gunDirection(),in.power(),
                target.x(),target.y(),target.direction(),target.speed())||in.power()<.1||in.power()>3
                ||in.width()<36||in.height()<36)return new Result(0,3,false);
        double v=20-3*in.power();
        double radial=CombatMath.bearing(target.x()-in.ownX(),target.y()-in.ownY());
        double lateral=target.speed()*Math.sin(Math.toRadians(target.direction()-radial));
        int sign=lateral<0?-1:1,launch=in.turn()+1,mask=0;
        for(int path=0;path<3;path++) {
            Pose pose=new Pose(target.x(),target.y(),target.direction(),target.speed());
            int orbit=path==0?sign:path==1?-sign:0;
            for(int tick=target.turn()+1;tick<launch+HORIZON;tick++) {
                pose=advance(pose,in,orbit);
                if(tick<launch)continue;
                double distance=Math.hypot(pose.x()-in.ownX(),pose.y()-in.ownY());
                double inner=v*(tick-launch+1);
                if(inner>distance+BODY)break;
                double half=halfFootprint(distance,inner,inner+v);
                if(Double.isFinite(half)) {
                    double angle=CombatMath.bearing(pose.x()-in.ownX(),pose.y()-in.ownY());
                    if(Math.abs(CombatMath.relativeAngle(in.gunDirection()-angle))<=half+1e-9)mask|=1<<path;
                }
            }
        }
        int covered=Integer.bitCount(mask);
        return new Result(covered,3,covered>=2);
    }

    /** Exact source69 orbit/stop hypotheses, not known opponent intentions or private motion. */
    static Pose advance(Pose pose,Input in,int orbit) {
        if(orbit==0)return step(pose,0,0);
        double dx=pose.x()-in.ownX(),dy=pose.y()-in.ownY();
        double radial=CombatMath.bearing(dx,dy);
        double tilt=clamp(.6*(Math.hypot(dx,dy)-650)/650,-.45*Math.PI,.45*Math.PI);
        double travel=radial+orbit*(90+Math.toDegrees(tilt));
        for(int i=0;i<72;i++) {
            double angle=Math.toRadians(travel),px=pose.x()+160*Math.cos(angle),py=pose.y()+160*Math.sin(angle);
            if(px>=BODY&&py>=BODY&&px<=in.width()-BODY&&py<=in.height()-BODY)break;
            travel+=orbit*5;
        }
        double turn=CombatMath.relativeAngle(travel-pose.heading());
        boolean back=Math.abs(turn)>90;if(back)turn=CombatMath.relativeAngle(turn+180);
        double limit=DuelSurf.maxTurn(pose.speed());
        return wall(pose,step(pose,clamp(turn,-limit,limit),back?-8:8),in.width(),in.height());
    }
    static Pose step(Pose pose,double turn,double targetSpeed) {
        double speed=DuelSurf.nextSpeed(pose.speed(),targetSpeed),radians=Math.toRadians(pose.heading());
        double limit=DuelSurf.maxTurn(speed);
        return new Pose(pose.x()+speed*Math.cos(radians),pose.y()+speed*Math.sin(radians),
                pose.heading()+clamp(turn,-limit,limit),speed);
    }
    /** Both wall passes use the preceding pose as the line origin, as in source69. */
    static Pose wall(Pose before,Pose after,double width,double height) {
        double x=after.x(),y=after.y();
        if(x>=BODY&&y>=BODY&&x<=width-BODY&&y<=height-BODY)return after;
        for(int pass=0;pass<2;pass++) {
            double oldX=x,oldY=y;
            if(oldX<BODY||oldX>width-BODY) {
                x=clamp(oldX,BODY,width-BODY);
                if(oldX!=before.x())y=before.y()+(oldY-before.y())*(x-before.x())/(oldX-before.x());
            }
            if(oldY<BODY||oldY>height-BODY) {
                y=clamp(oldY,BODY,height-BODY);
                if(oldY!=before.y())x=before.x()+(oldX-before.x())*(y-before.y())/(oldY-before.y());
            }
        }
        return new Pose(x,y,after.heading(),0);
    }
    /** Circle/annulus footprint in degrees, including exact tangencies. */
    static double halfFootprint(double distance,double inner,double outer) {
        if(outer<distance-BODY||inner>distance+BODY)return Double.NaN;
        if(distance==0)return 180;
        double tangent=Math.sqrt(Math.max(0,distance*distance-BODY*BODY));
        double radius=clamp(tangent,inner,outer);
        double cosine=(distance*distance+radius*radius-BODY*BODY)/(2*distance*radius);
        return Math.toDegrees(Math.acos(clamp(cosine,-1,1)));
    }
    private static double clamp(double v,double lo,double hi){return Math.max(lo,Math.min(hi,v));}
    private static boolean finite(double... values){for(double v:values)if(!Double.isFinite(v))return false;return true;}
}
