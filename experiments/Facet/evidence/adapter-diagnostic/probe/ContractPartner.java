import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
/** Independent legal-event integration probe, deliberately stationary. */
public final class ContractPartner extends Bot {
  final PrintWriter out;
  long sequence;
  double targetX, targetY;
  boolean targetKnown;
  int lastRequest=-100000;
  ContractPartner() throws Exception {
    super(BotInfo.fromFile("ContractPartner.json"));
    out=new PrintWriter(Files.newBufferedWriter(Path.of(System.getenv("FACET_PROBE_PATH")), StandardOpenOption.CREATE_NEW));
  }
  public static void main(String[] args)throws Exception {new ContractPartner().start();}
  double relative(double a) {return ((a+540)%360)-180;}
  void row(String type,int eventTurn,String detail) {
    out.printf(Locale.ROOT,"{\"type\":\"%s\",\"sequence\":%d,\"round\":%d,\"turn\":%d,\"eventTurn\":%d,\"id\":%d,\"energy\":%.15g%s}%n",type,++sequence,getRoundNumber(),getTurnNumber(),eventTurn,getMyId(),getEnergy(),detail);out.flush();
  }
  String bullet(BulletState b) {return String.format(Locale.ROOT,",\"owner\":%d,\"bullet\":%d,\"power\":%.15g",b.getOwnerId(),b.getBulletId(),b.getPower());}
  @Override public void onRoundStarted(RoundStartedEvent e){targetKnown=false;lastRequest=-100000;row("roundStart",0,"");}
  @Override public void run(){
    setAdjustGunForBodyTurn(true);setAdjustRadarForBodyTurn(true);setAdjustRadarForGunTurn(true);setFireAssist(false);
    while(isRunning()){
      setTargetSpeed(0);setTurnRate(0);setRadarTurnRate(45);setFire(0);
      if(targetKnown){
        double desired=Math.toDegrees(Math.atan2(targetY-getY(),targetX-getX()));
        double error=relative(desired-getGunDirection());setGunTurnRate(error);
        if(getTurnNumber()-lastRequest>=60&&getGunHeat()==0&&getEnergy()>.1&&Math.abs(error)<1){
          boolean accepted=setFire(.1);lastRequest=getTurnNumber();row("request",getTurnNumber(),",\"accepted\":"+accepted);
        }
      } else setGunTurnRate(0);
      go();
    }
  }
  @Override public void onScannedBot(ScannedBotEvent e){targetX=e.getX();targetY=e.getY();targetKnown=true;}
  @Override public void onBulletFired(BulletFiredEvent e){row("fired",e.getTurnNumber(),bullet(e.getBullet()));}
  @Override public void onHitByBullet(HitByBulletEvent e){row("hitBy",e.getTurnNumber(),bullet(e.getBullet())+",\"damage\":"+e.getDamage());}
  @Override public void onBulletHit(BulletHitBotEvent e){row("hit",e.getTurnNumber(),bullet(e.getBullet())+",\"damage\":"+e.getDamage());}
  @Override public void onDeath(DeathEvent e){row("death",e.getTurnNumber(),"");}
  @Override public void onRoundEnded(RoundEndedEvent e){row("roundEnd",getTurnNumber(),"");}
  @Override public void onGameEnded(GameEndedEvent e){row("gameEnd",getTurnNumber(),"");out.close();}
  @Override public void onSkippedTurn(SkippedTurnEvent e){row("skip",e.getTurnNumber(),"");}
}
