/** All angles in degrees counterclockwise from east. No bot API or opponent dependency. */
final class Flight {
    record Pose(double x, double y, double h, double v) {}
    record Command(double turn, double speed) {}
    static double clip(double x,double lo,double hi){ return Math.max(lo, Math.min(x,hi)); }
    static double delta(double a){ return Math.IEEEremainder(a,360); }
    static double angle(double x,double y){ return Math.toDegrees(Math.atan2(y,x)); }
    static double cos(double h){ return Math.cos(Math.toRadians(h)); }
    static double sin(double h){ return Math.sin(Math.toRadians(h)); }
    static double speed(double v,double target){
        if(v<0) return -speed(-v,-target);
        if(v==0) return clip(target,-1,1);
        if(target==0) return 0;
        if(target<0) return v/2-1;
        return v+clip(target-v,-2,1);
    }
    static Pose step(Pose s, Command c,double width,double height){
        double v=speed(s.v,c.speed), cap=10-.75*Math.abs(v);
        double x=s.x+cos(s.h)*v, y=s.y+sin(s.h)*v;
        double nx=clip(x,18,width-18),ny=clip(y,18,height-18);
        if(nx!=x && x!=s.x) ny=s.y+(y-s.y)*(nx-s.x)/(x-s.x);
        if(y<18 || y>height-18){
            ny=clip(y,18,height-18);
            if(y!=s.y) nx=s.x+(x-s.x)*(ny-s.y)/(y-s.y);
        }
        return new Pose(nx,ny,s.h+clip(c.turn,-cap,cap),nx!=x||ny!=y ? 0:v);
    }
    static Command orbit(Pose s,double ex,double ey,int side,double w,double h){
        double r=Math.hypot(s.x-ex,s.y-ey);
        double a=angle(ex-s.x,ey-s.y)+side*(90+clip((380-r)*.10,-32,32));
        // Rotate the look-ahead stick toward the arena until its endpoint fits.
        for(int i=0;i<72;i++,a+=side*5){
            double x=s.x+cos(a)*110,y=s.y+sin(a)*110;
            if(x>=24&&x<=w-24&&y>=24&&y<=h-24) break;
        }
        double d=delta(a-s.h);
        return Math.abs(d)>90 ? new Command(delta(d+180),-8):new Command(d,8);
    }
    static boolean canShoot(double gun,double aim,double range,double energy,double heat,double p){
        return heat==0 && energy>p && Math.abs(delta(aim-gun))<=Math.toDegrees(Math.atan2(12,range));
    }
    // First collision sweep starts one bullet step out and ends two steps out.
    static double farEdge(double power,int steps){return (steps+1)*(20-3*power);}
}
