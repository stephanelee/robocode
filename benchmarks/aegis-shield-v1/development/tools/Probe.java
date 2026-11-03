import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;

/** Dev-only probe: stationary target, logs inferred enemy fire and hit geometry. */
public class Probe extends Bot {
    PrintWriter log;
    double eEnergy=-1; int eScanTurn=-1; double ex,ey,pex,pey; int peTurn=-1;
    final Map<Integer,Double> corr=new HashMap<>(); int dirSign=1;
    final Map<Integer,double[]> own=new HashMap<>();
    public static void main(String[] a){new Probe().start();}
    Probe(){super(BotInfo.fromFile("Probe.json"));
        try{log=new PrintWriter(new FileWriter(System.getenv().getOrDefault("CHALLENGER_METRICS_PATH","probe.jsonl")+".trace",true));}catch(IOException e){throw new UncheckedIOException(e);} }
    @Override public void run(){
        setAdjustGunForBodyTurn(true);setAdjustRadarForGunTurn(true);setAdjustRadarForBodyTurn(true);
        while(isRunning()){
            own.put(getTurnNumber(),new double[]{getX(),getY()});
            if(eScanTurn<0) setRadarTurnRate(45);
            else{ double b=Math.toDegrees(Math.atan2(ey-getY(),ex-getX()));
                double r=norm(b-getRadarDirection()); r+=Math.signum(r)*20; setRadarTurnRate(Math.max(-45,Math.min(45,r)));}
            if(getTurnNumber()%37==0){setForward(12*dirSign);dirSign=-dirSign;}
            log.printf(Locale.ROOT,"{\"ev\":\"own\",\"turn\":%d,\"x\":%.4f,\"y\":%.4f}%n",getTurnNumber(),getX(),getY());
            go();
        }
    }
    static double norm(double a){a%=360;if(a>180)a-=360;if(a<-180)a+=360;return a;}
    @Override public void onScannedBot(ScannedBotEvent e){
        int t=e.getTurnNumber();
        if(eScanTurn==t-1&&eEnergy>=0){
            double drop=eEnergy-e.getEnergy()+corr.getOrDefault(t-1,0.0);
            if(drop>=0.0999&&drop<=3.0001){
                double[] me2=own.get(t-2);
                log.printf(Locale.ROOT,"{\"ev\":\"fire\",\"turn\":%d,\"power\":%.4f,\"ox\":%.3f,\"oy\":%.3f,\"aimx\":%.3f,\"aimy\":%.3f}%n",t,drop,ex,ey,me2==null?-1:me2[0],me2==null?-1:me2[1]);
            }
        }
        log.printf(Locale.ROOT,"{\"ev\":\"scan\",\"turn\":%d,\"x\":%.4f,\"y\":%.4f,\"e\":%.4f}%n",t,e.getX(),e.getY(),e.getEnergy());pex=ex;pey=ey;ex=e.getX();ey=e.getY();eEnergy=e.getEnergy();eScanTurn=t;
    }
    @Override public void onHitByBullet(HitByBulletEvent e){
        var b=e.getBullet(); corr.merge(e.getTurnNumber(),-3*b.getPower(),Double::sum);
        double d=Math.toRadians(b.getDirection()); double dx=getX()-b.getX(),dy=getY()-b.getY();
        double perp=-Math.sin(d)*dx+Math.cos(d)*dy;
        log.printf(Locale.ROOT,"{\"ev\":\"hit\",\"turn\":%d,\"power\":%.4f,\"bx\":%.3f,\"by\":%.3f,\"dir\":%.4f,\"mx\":%.3f,\"my\":%.3f,\"perp\":%.3f}%n",e.getTurnNumber(),b.getPower(),b.getX(),b.getY(),b.getDirection(),getX(),getY(),perp);
    }
    @Override public void onBulletHit(BulletHitBotEvent e){corr.merge(e.getTurnNumber(),e.getDamage(),Double::sum);}
    @Override public void onRoundStarted(RoundStartedEvent e){eEnergy=-1;eScanTurn=-1;own.clear();corr.clear();log.printf("{\"ev\":\"round\",\"n\":%d}%n",e.getRoundNumber());}
    @Override public void onRoundEnded(RoundEndedEvent e){log.flush();}
}
