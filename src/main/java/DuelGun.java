import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Legal-scan virtual aiming. Scores are geometric proxies, not actual-hit attribution or calibrated probabilities. */
final class DuelGun {
    record Scan(int targetId, int turn, double x, double y, double direction, double speed, double energy) {}
    record Input(int turn, double ownX, double ownY, double gunDirection, double energy, double gunHeat,
                 double arenaWidth, double arenaHeight, Scan target, double fallbackDirection) {}
    record Plan(double direction, double power, boolean shouldFire, boolean fallback,
                String model, double confidence, int attempts, int hits) {}
    private static final String[] NAMES = {"current", "headon", "linear", "circular"};
    private static final int MIN_SHOTS = 12;
    private final int[] attempts = new int[4], hits = new int[4];
    private final List<Flight> flights = new ArrayList<>();
    private Scan latest, previous;
    private int targetId = -1;
    private Pending pending;

    private record Pending(int turn, int targetId, double power, double[] angles, double width, double height) {}
    private static final class Flight {
        final int turn, targetId, maxAge;
        final double x, y, speed;
        final double[] angles;
        final boolean[] hit = new boolean[4], done = new boolean[4];
        int observed;
        Flight(int turn, int targetId, double x, double y, double power, double[] angles, double width, double height) {
            this.turn=turn; this.targetId=targetId; this.x=x; this.y=y; this.speed=20-3*power;
            this.angles=angles.clone(); this.observed=turn-1;
            maxAge=(int)Math.ceil(Math.hypot(width,height)/speed)+3;
        }
    }

    /** Preserve evidence across rounds of the same opponent; discard trajectories and round-local observations. */
    void startRound() { latest=previous=null; pending=null; flights.clear(); }

    void onScan(Scan scan) {
        if (scan == null || scan.targetId < 1 || scan.turn < 0 || !finite(scan.x,scan.y,scan.direction,scan.speed,scan.energy)) return;
        if (scan.targetId != targetId) {
            targetId=scan.targetId; Arrays.fill(attempts,0); Arrays.fill(hits,0);
            latest=previous=null; pending=null; flights.clear();
        }
        if (latest != null && scan.turn <= latest.turn) return;
        previous=latest; latest=scan;
        score(scan);
    }

    private void score(Scan scan) {
        flights.removeIf(f -> {
            if (scan.targetId != f.targetId || scan.turn < f.turn) return false;
            if (scan.turn == f.observed) return false; // A new fire callback can revisit an already scored scan.
            if (scan.turn != f.observed+1 || scan.turn-f.turn > f.maxAge) return true; // Unknown gaps never become misses.
            f.observed=scan.turn;
            // v1.4 advances tick before testing BulletLine(position,nextPosition): creation tick uses [v,2v].
            double lo=f.speed*(scan.turn-f.turn+1), hi=lo+f.speed;
            double distance=Math.hypot(scan.x-f.x,scan.y-f.y);
            boolean complete=true;
            for (int m=0;m<4;m++) {
                if (!f.done[m]) {
                    f.hit[m]=segmentDistance(scan.x,scan.y,f.x,f.y,f.angles[m],lo,hi)<=18+1e-9;
                    f.done[m]=f.hit[m] || lo>distance+18;
                }
                complete &= f.done[m];
            }
            if (complete) for (int m=0;m<4;m++) { attempts[m]++; if(f.hit[m]) hits[m]++; }
            return complete;
        });
    }

    /** Only a confirmed own bullet creates a virtual cohort. No requested-shot or actual-hit labels are used. */
    void onFired(int turn, double x, double y, double power) {
        Pending p=pending; pending=null;
        if (p==null || turn!=p.turn || p.targetId!=targetId || Math.abs(power-p.power)>1e-9 || !finite(x,y,power)) return;
        if (flights.size()>=48) flights.remove(0); // Unresolved discarded cohorts are unknown, not misses.
        flights.add(new Flight(turn,p.targetId,x,y,power,p.angles,p.width,p.height));
        if (latest!=null && latest.turn==turn) score(latest); // Scans can be delivered before BulletFired in the same tick.
    }

    Plan decide(Input in) {
        onScan(in.target);
        Scan s=in.target;
        double current=Double.isFinite(in.gunDirection)?normalize(in.gunDirection):0;
        if (s==null || s.targetId!=targetId || in.turn<s.turn || in.turn-s.turn>2
                || in.arenaWidth<=36 || in.arenaHeight<=36
                || !finite(in.ownX,in.ownY,in.gunDirection,in.energy,in.gunHeat,in.arenaWidth,in.arenaHeight)
                || !finite(s.x,s.y,s.direction,s.speed,s.energy) || Math.abs(s.speed)>8+1e-9) {
            pending=null; return new Plan(current,0,false,true,"current",0,attempts[0],hits[0]);
        }
        double power=firepower(in.energy,s.energy,Math.hypot(s.x-in.ownX,s.y-in.ownY));
        if (power<.1) { pending=null; return new Plan(current,0,false,true,"current",0,attempts[0],hits[0]); }
        double rate=turnRate();
        double[] angles=new double[4];
        for(int m=1;m<4;m++) {
            double[] point=intercept(in,power,m,rate);
            angles[m]=bearing(point[0]-in.ownX,point[1]-in.ownY);
        }
        angles[0]=Double.isFinite(in.fallbackDirection)?normalize(in.fallbackDirection):angles[2];
        int selected=0;
        if(attempts[0]>=MIN_SHOTS) {
            double baseline=accuracy(0), best=baseline+.03;
            for(int m=1;m<4;m++) if(attempts[m]>=MIN_SHOTS && accuracy(m)>best) { selected=m; best=accuracy(m); }
        }
        // Coordinated ray/trajectory test at this power, using the CURRENT heading: firing precedes gun rotation.
        boolean aligned=in.gunHeat<=1e-9 && canHit(in,power,selected==0?2:selected,rate,current);
        boolean fire=in.gunHeat<=1e-9 && aligned;
        pending=fire?new Pending(in.turn+1,targetId,power,angles,in.arenaWidth,in.arenaHeight):null;
        return new Plan(angles[selected],power,fire,selected==0,NAMES[selected],
                attempts[selected]/(attempts[selected]+12.0),attempts[selected],hits[selected]);
    }

    private double accuracy(int m) { return (hits[m]+1.0)/(attempts[m]+2.0); }
    int attempts(String model) { return attempts[index(model)]; }
    int hits(String model) { return hits[index(model)]; }
    private static int index(String model) {
        for(int i=0;i<NAMES.length;i++) if(NAMES[i].equals(model)) return i;
        throw new IllegalArgumentException("Unknown model");
    }
    private double turnRate() {
        if(previous==null || latest.turn-previous.turn>3) return 0;
        return signed(latest.direction-previous.direction)/(latest.turn-previous.turn);
    }

    static double firepower(double ownEnergy, double enemyEnergy, double range) {
        if(!finite(ownEnergy,enemyEnergy,range) || enemyEnergy<=0) return 0;
        double reserve=ownEnergy<8?.35:2;
        double available=ownEnergy-reserve;
        if(available<.1) return 0;
        double preferred=range<180?2.4:range<400?1.8:1.2;
        if(ownEnergy<12) preferred=Math.min(preferred,.8);
        if(ownEnergy<3) preferred=Math.min(preferred,.3);
        double finish=enemyEnergy<=4?enemyEnergy/4:(enemyEnergy+2)/6;
        return Math.min(available,Math.min(3,Math.max(.1,Math.min(preferred,finish))));
    }

    /** Constant observed velocity/turn hypotheses; move on old heading, then rotate, stop at first wall contact. */
    static double[] forecast(Scan s, int turns, int model, double rate, double width, double height) {
        double x=s.x,y=s.y,heading=s.direction,speed=model==1?0:s.speed;
        for(int i=0;i<turns;i++) {
            double dx=Math.cos(Math.toRadians(heading))*speed,dy=Math.sin(Math.toRadians(heading))*speed;
            double fraction=1;
            if(dx>0 && x+dx>width-18) fraction=Math.min(fraction,(width-18-x)/dx);
            if(dx<0 && x+dx<18) fraction=Math.min(fraction,(18-x)/dx);
            if(dy>0 && y+dy>height-18) fraction=Math.min(fraction,(height-18-y)/dy);
            if(dy<0 && y+dy<18) fraction=Math.min(fraction,(18-y)/dy);
            fraction=Math.max(0,fraction); x+=dx*fraction; y+=dy*fraction;
            if(fraction<1) speed=0;
            if(model==3) heading+=Math.max(-10+.75*Math.abs(speed),Math.min(10-.75*Math.abs(speed),rate));
        }
        return new double[]{x,y};
    }

    private static double[] intercept(Input in,double power,int model,double rate) {
        double v=20-3*power; double[] point={in.target.x,in.target.y};
        int age=in.turn-in.target.turn;
        for(int t=1;t<=(int)Math.ceil(Math.hypot(in.arenaWidth,in.arenaHeight)/v)+3;t++) {
            point=forecast(in.target,age+t,model,rate,in.arenaWidth,in.arenaHeight);
            if(Math.hypot(point[0]-in.ownX,point[1]-in.ownY)<=v*(t+1)+18) break;
        }
        return point;
    }
    private static boolean canHit(Input in,double power,int model,double rate,double angle) {
        double v=20-3*power;
        for(int t=1;t<=(int)Math.ceil(Math.hypot(in.arenaWidth,in.arenaHeight)/v)+3;t++) {
            double[] p=forecast(in.target,in.turn-in.target.turn+t,model,rate,in.arenaWidth,in.arenaHeight);
            if(segmentDistance(p[0],p[1],in.ownX,in.ownY,angle,v*t,v*(t+1))<=18+1e-9) return true;
        }
        return false;
    }
    static double segmentDistance(double px,double py,double x,double y,double angle,double lo,double hi) {
        double c=Math.cos(Math.toRadians(angle)),s=Math.sin(Math.toRadians(angle));
        double along=Math.max(lo,Math.min(hi,(px-x)*c+(py-y)*s));
        return Math.hypot(px-x-along*c,py-y-along*s);
    }
    private static double bearing(double x,double y) { return normalize(Math.toDegrees(Math.atan2(y,x))); }
    private static double normalize(double a) { return (a%360+360)%360; }
    private static double signed(double a) { return (a%360+540)%360-180; }
    private static boolean finite(double... values) { for(double v:values) if(!Double.isFinite(v)) return false; return true; }
}
