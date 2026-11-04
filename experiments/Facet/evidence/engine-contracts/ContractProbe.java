import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;
import java.nio.file.*;
import java.io.*;
import java.util.*;
public class ContractProbe extends Bot {
 final String label; final PrintWriter out; long sequence=0; boolean cornerReady=false; int cornerStart=0;
 ContractProbe(String label) throws Exception {super(BotInfo.fromFile(label+".json"));this.label=label;out=new PrintWriter(Files.newBufferedWriter(Path.of(System.getenv("FACET_CONTRACT_DIR"),label+".jsonl")));}
 public static void main(String[] a)throws Exception{new ContractProbe(a[0]).start();}
 double diff(double a,double b){return ((a-b+540)%360)-180;}
 void row(String type,String extra){out.printf(Locale.ROOT,"{\"type\":\"%s\",\"sequence\":%d,\"round\":%d,\"turn\":%d,\"x\":%.15g,\"y\":%.15g,\"body\":%.15g,\"gun\":%.15g,\"radar\":%.15g,\"speed\":%.15g,\"heat\":%.15g,\"energy\":%.15g%s}%n",type,++sequence,getRoundNumber(),getTurnNumber(),getX(),getY(),getDirection(),getGunDirection(),getRadarDirection(),getSpeed(),getGunHeat(),getEnergy(),extra);out.flush();}
 @Override public void run(){setAdjustGunForBodyTurn(true);setAdjustRadarForBodyTurn(true);setAdjustRadarForGunTurn(true);setFireAssist(false);
 while(isRunning()){
 int t=getTurnNumber(); double speed=0,body=0,gun=0,radar=0; String phase="idle";
 if(label.equals("Alpha")){
 if(t<12){speed=8;phase="acceleration-positive";}
 else if(t==12){speed=0;phase="zero-stop-positive";}
 else if(t<24){speed=-8;phase="acceleration-negative";}
 else if(t<30){speed=8;phase="reverse-negative";}
 else if(t<40){speed=8;phase="accelerate";}
 else if(t<46){speed=-8;phase="reverse-positive";}
 else if(t==46){speed=0;phase="zero-stop";}
 else if(t<60){speed=8;body=30;gun=30;radar=-60;phase="turn-decoupled";}
 else if(t<68){speed=0;body=-30;gun=-30;radar=60;phase="stopped-turn";}
 else if(t==68){speed=0;phase="before-finite";}
 else if(t>=69&&t<80){phase="finite-distance";}
 else if(t<90){speed=0;body=diff(0,getDirection());phase="east-align";}
 else if(t<220){speed=8;phase="wall-east";}
 else if(t<270){speed=0;body=diff(225,getDirection());phase="inward-align";}
 else if(t<282){speed=8;phase="inward-drive";}
 else if(t<330){speed=0;body=diff(Math.toDegrees(Math.atan2(getArenaHeight()-18-getY(),getArenaWidth()-18-getX())),getDirection());phase="corner-align";}
 else {speed=8;phase="corner-ray";}
 }
 setTurnRate(body);setGunTurnRate(gun);setRadarTurnRate(radar);
 if(!(label.equals("Alpha")&&t>=70&&t<80))setTargetSpeed(speed);
 if(label.equals("Alpha")&&t==69)setForward(5);
 boolean accepted=false;if(label.equals("Alpha")&&t==55)accepted=setFire(.1);
 row("decision",String.format(Locale.ROOT,",\"phase\":\"%s\",\"targetSpeed\":%.15g,\"bodyRate\":%.15g,\"gunRate\":%.15g,\"radarRate\":%.15g,\"fireRequested\":%s,\"fireAccepted\":%s,\"distanceRemaining\":%.15g,\"width\":%d,\"height\":%d",phase,getTargetSpeed(),body,gun,radar,label.equals("Alpha")&&t==55,accepted,getDistanceRemaining(),getArenaWidth(),getArenaHeight()));
 for(BulletState b:getBulletStates()) row("bulletState",String.format(Locale.ROOT,",\"bullet\":%d,\"bulletX\":%.15g,\"bulletY\":%.15g,\"bulletDirection\":%.15g,\"power\":%.15g",b.getBulletId(),b.getX(),b.getY(),b.getDirection(),b.getPower()));
 if(label.equals("Alpha")&&t>=400){try{Files.writeString(Path.of(System.getenv("FACET_CONTRACT_DIR"),"ready"),"script complete\n");}catch(Exception e){throw new RuntimeException(e);}}
 go();
 }
 }
 @Override public void onBulletFired(BulletFiredEvent e){BulletState b=e.getBullet();row("fired",String.format(Locale.ROOT,",\"eventTurn\":%d,\"bullet\":%d,\"bulletX\":%.15g,\"bulletY\":%.15g,\"bulletDirection\":%.15g,\"power\":%.15g",e.getTurnNumber(),b.getBulletId(),b.getX(),b.getY(),b.getDirection(),b.getPower()));}
 @Override public void onHitWall(HitWallEvent e){row("wall",",\"eventTurn\":"+e.getTurnNumber());}
 @Override public void onSkippedTurn(SkippedTurnEvent e){row("skip",",\"eventTurn\":"+e.getTurnNumber());}
}
