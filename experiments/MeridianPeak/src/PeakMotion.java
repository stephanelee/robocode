/** Own-route physics only. Gun forecasts deliberately retain the original Flight.step. */
final class PeakMotion {
    private static double[] constrain(double ox, double oy, double x, double y, double w, double h) {
        double nx=x, ny=y;
        if (x<18 || x>w-18) {
            nx=Flight.clip(x,18,w-18);
            if (x!=ox) ny=oy+(y-oy)*(nx-ox)/(x-ox);
        }
        if (y<18 || y>h-18) {
            ny=Flight.clip(y,18,h-18);
            if (y!=oy) nx=ox+(x-ox)*(ny-oy)/(y-oy);
        }
        return new double[]{nx,ny};
    }
    static Flight.Pose step(Flight.Pose p, Flight.Command c, double w, double h) {
        double speed=Flight.speed(p.v(),c.speed());
        double x=p.x()+Flight.cos(p.h())*speed, y=p.y()+Flight.sin(p.h())*speed;
        double heading=p.h()+Flight.clip(c.turn(),-10+.75*Math.abs(speed),10-.75*Math.abs(speed));
        var first=constrain(p.x(),p.y(),x,y,w,h);
        if (first[0]!=x || first[1]!=y) speed=0;
        var second=constrain(p.x(),p.y(),first[0],first[1],w,h);
        return new Flight.Pose(second[0],second[1],heading,speed);
    }
}
