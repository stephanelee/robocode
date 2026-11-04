import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;
import java.nio.file.*;
import java.io.*;
import java.util.*;

public final class MeridianPeak extends Bot {
    record Scan(int t,Flight.Pose p,double energy,double omega){}
    private final HitAngles hitAngles = new HitAngles();
    record Trial(int due,double[] angles,double x,double y){}
    private final CrossingGun crossingGun = new CrossingGun();
    private Scan scan;
    private final LinkedList<HitAngles.Wave> waves=new LinkedList<>();
    private final ArrayList<Trial> trials=new ArrayList<>();
    private final double[] skill={.2,.3,.25}; // stop, coast, turn
    private final Map<Integer,Double> corrections=new HashMap<>();
    private final Flight.Pose[] history=new Flight.Pose[4];
    private final int[] stamp=new int[4];
    private int side=1,round,radarSign=1;
    private final long seed=Long.parseLong(System.getenv().getOrDefault("MERIDIAN_PEAK_SEED","0"));
    private final String telemetry=System.getenv("MERIDIAN_PEAK_METRICS");
    private final PeakTelemetry state;
    private final Object roundState=new Object();
    private final ThreadLocal<Integer> runEpoch=new ThreadLocal<>();
    private int epoch,pendingRoundEnd;
    private boolean roundClosed=true,roundClosing;
    MeridianPeak(){this(System.getenv("MERIDIAN_PEAK_STATE_DIR"));}
    MeridianPeak(String output){
        super(BotInfo.fromFile("MeridianPeak.json"));
        state=new PeakTelemetry(output,seed,
            ()->new PeakTelemetry.Models(hitAngles.snapshot(),crossingGun.snapshot(),skill.clone(),trials.stream().map(q->new PeakTelemetry.TrialSnapshot(q.due,q.angles.clone(),q.x,q.y)).toList()));
    }
    public static void main(String[] args){new MeridianPeak().start();}
    private void mark(String value){
        if(telemetry==null)return;
        try {Files.writeString(Path.of(telemetry),round+" "+value+"\n",StandardOpenOption.CREATE,StandardOpenOption.APPEND);}
        catch(IOException e){throw new UncheckedIOException(e);}
    }
    public void onRoundStarted(RoundStartedEvent e){
        synchronized(roundState) {
            if(e.getRoundNumber()==round&&!roundClosed)return;
            sealPendingRound();
            if(e.getRoundNumber()!=state.completedRounds+1)throw new IllegalStateException("Round start sequence");
            crossingGun.newRound();
            round=e.getRoundNumber(); scan=null;waves.clear();trials.clear();corrections.clear();Arrays.fill(stamp,-1);
            side=new SplittableRandom(seed+round).nextBoolean()?1:-1;radarSign=side;state.start(round);mark("start");
            epoch++;roundClosed=false;roundClosing=false;
        }
    }
    public void onRoundEnded(RoundEndedEvent e){
        synchronized(roundState) {
            if(roundClosed||e.getRoundNumber()<=state.completedRounds||pendingRoundEnd!=0)return;
            if(e.getRoundNumber()!=round)throw new IllegalStateException("Round end sequence");
            pendingRoundEnd=e.getRoundNumber();roundClosing=true;mark("end");
        }
    }
    public void onGameEnded(GameEndedEvent e){synchronized(roundState){sealPendingRound();}}
    public void onSkippedTurn(SkippedTurnEvent e){
        synchronized(roundState) {
            if(!acceptsCallback())return;
            state.stats.skippedTurns++;mark("skip "+e.getTurnNumber());state.changed();
        }
    }
    public void onBulletHit(BulletHitBotEvent e){
        synchronized(roundState) {
            if(!acceptsCallback())return;
            corrections.merge(e.getTurnNumber(),-e.getDamage(),Double::sum);state.stats.hits++;state.stats.damageDealt+=e.getDamage();state.changed();
        }
    }
    public void onHitByBullet(HitByBulletEvent e){
        synchronized(roundState) {
            if(!acceptsCallback())return;
            var bullet = e.getBullet();
            corrections.merge(e.getTurnNumber(), 3 * bullet.getPower(), Double::sum);
            var label = hitAngles.observe(waves, e.getTurnNumber(), bullet.getX(), bullet.getY(),
                bullet.getDirection(), bullet.getPower());
            state.stats.hitsTaken++;state.stats.damageTaken+=e.getDamage();state.hitLabel(label);
            mark("hit " + label);state.changed();
        }
    }
    public void onBulletFired(BulletFiredEvent event) {
        synchronized(roundState) {
            if(!acceptsCallback())return;
            var bullet = event.getBullet();
            crossingGun.fired(event.getTurnNumber(), bullet.getX(), bullet.getY(), bullet.getPower());
            state.stats.confirmedShots++;state.stats.shotEnergy+=bullet.getPower();
            mark("fire "+event.getTurnNumber()+" "+bullet.getPower());state.changed();
        }
    }
    public void onScannedBot(ScannedBotEvent e){
        synchronized(roundState) {
            if(!acceptsCallback())return;
            int t=e.getTurnNumber();
            if(scan!=null && t<=scan.t)return;
            Flight.Pose p=new Flight.Pose(e.getX(),e.getY(),e.getDirection(),e.getSpeed());
            var crossingLabels=crossingGun.labelsFor(t,p,scan==null?-1:scan.t,scan==null?null:scan.p);
            int learned = crossingGun.observe(t, p, scan == null ? -1 : scan.t, scan == null ? null : scan.p);
            state.crossings(learned);state.crossingLabels(crossingLabels);
            if (learned > 0) mark("crossings " + learned);
            double omega=0;
            if(scan!=null && t==scan.t+1){
            omega=Flight.delta(p.h()-scan.p.h());
            double power=scan.energy-e.getEnergy()+corrections.getOrDefault(t-1,0.);
            int k=(t-1)%4;
            if(power>=.09999 && power<=3.00001 && stamp[k]==t-1){
                var own=history[k]; double a=Flight.angle(own.x()-scan.p.x(),own.y()-scan.p.y());
                double lat=own.v()*Flight.sin(own.h()-a);
                waves.add(new HitAngles.Wave(t,scan.p.x(),scan.p.y(),a,a+Math.toDegrees(Math.asin(lat/(20-3*power))),power));
            }
        }
            corrections.keySet().removeIf(k->k<t);
            for(var it=trials.iterator();it.hasNext();){
            Trial q=it.next(); if(q.due>t)continue;
            if(q.due==t){
                double a=Flight.angle(p.x()-q.x,p.y()-q.y),r=Math.hypot(p.x()-q.x,p.y()-q.y);
                for(int i=0;i<3;i++){
                    double miss=2*r*Math.abs(Flight.sin(Flight.delta(a-q.angles[i])/2));
                    skill[i]=.94*skill[i]+.06/(1+miss*miss/324);
                }
            }
            state.trial(t,new PeakTelemetry.TrialSnapshot(q.due,q.angles.clone(),q.x,q.y),p);
            it.remove();
        }
            scan=new Scan(t,p,e.getEnergy(),omega);state.changed();
        }
    }
    private void gun(int t,Flight.Pose own,double range){
        var p=scan.p; double energy=getEnergy();
        double power=Flight.clip(Math.min(Math.min(range<300?2.8:1.65,energy/12),Math.max(.1,(scan.energy+2)/6)),.1,3);
        double[] aims=new double[3]; int[] arrival=new int[3];
        for(int i=0;i<3;i++){
            var pred=p;
            for(int n=1;n<=120;n++){
                pred=Flight.step(pred,new Flight.Command(i==2?scan.omega:0,i==0?0:pred.v()),getArenaWidth(),getArenaHeight());
                if(Flight.farEdge(power,n)>=Math.hypot(pred.x()-own.x(),pred.y()-own.y())){
                    arrival[i]=t+n;break;
                }
            }
            aims[i]=Flight.angle(pred.x()-own.x(),pred.y()-own.y());
        }
        int best=0;for(int i=1;i<3;i++)if(skill[i]>skill[best])best=i;
        // Grade all rays on the chosen forecast's impact turn, only when a real scan arrives.
        if(scan.t==t && t%4==0)trials.add(new Trial(arrival[best],aims,own.x(),own.y()));
        var launch = CrossingGun.launch(t + 1, own, p, power, getArenaWidth(), getArenaHeight());
        double estimate = crossingGun.aim(launch);
        boolean learned = Double.isFinite(estimate);
        double aim = learned ? estimate : aims[best];
        setGunTurnRate(Flight.delta(aim - getGunDirection()));
        if (scan.t == t && Flight.canShoot(getGunDirection(), aim, range, energy, getGunHeat(), power)
                && (energy > 1.2 || scan.energy <= .1) && setFire(power)) {
            crossingGun.requested(launch);
            mark(learned ? "aim learned" : "aim fallback");
        }
    }
    private PeakRisk.Evaluation route(Flight.Pose p,int t,int sign){
        return PeakRisk.evaluate(p,t,sign,side,scan.p.x(),scan.p.y(),waves,hitAngles,getArenaWidth(),getArenaHeight());
    }
    private boolean acceptsCallback(){
        Integer owner=runEpoch.get();return !roundClosed&&(owner==null||owner==epoch);
    }
    private void sealPendingRound(){
        if(pendingRoundEnd==0)return;
        state.end(pendingRoundEnd);pendingRoundEnd=0;roundClosed=true;roundClosing=false;
    }
    boolean registerRunThread(){
        synchronized(roundState){
            if(Thread.currentThread().isInterrupted()||roundClosed||roundClosing||!acceptsCallback())return false;
            runEpoch.set(epoch);return true;
        }
    }
    public void run(){
        synchronized(roundState){
            if(!registerRunThread())return;
            setAdjustGunForBodyTurn(true);setAdjustRadarForGunTurn(true);setAdjustRadarForBodyTurn(true);setFireAssist(false);
        }
        // Waiting and SDK event dispatch must never hold the lifecycle monitor.
        while(isRunning()&&performTurn())go();
    }
    boolean performTurn(){
        synchronized(roundState){
            if(roundClosing||!acceptsCallback())return false;
            int t=getTurnNumber();
            var own=new Flight.Pose(getX(),getY(),getDirection(),getSpeed());
            history[t%4]=own;stamp[t%4]=t;
            waves.removeIf(w->Flight.farEdge(w.power(),t-w.t()+1)>Math.hypot(getX()-w.x(),getY()-w.y())+36);
            if(scan==null||t-scan.t>3){
                setRadarTurnRate(45*radarSign);setTargetSpeed(4);
                setTurnRate(Flight.delta(Flight.angle(getArenaWidth()/2.-getX(),getArenaHeight()/2.-getY())-getDirection()));
            }else{
                var p=scan.p;double a=Flight.angle(p.x()-getX(),p.y()-getY());
                double d=Flight.delta(a-getRadarDirection());if(Math.abs(d)>.01)radarSign=d>0?1:-1;
                setRadarTurnRate(d+radarSign*18);
                long decisionStart=System.nanoTime();
                var plus=route(own,t,1);var minus=route(own,t,-1);
                side=plus.score()<minus.score()?1:-1;
                var cmd=Flight.orbit(own,p.x(),p.y(),side,getArenaWidth(),getArenaHeight());
                setTurnRate(cmd.turn());setTargetSpeed(cmd.speed());
                gun(t,own,Math.hypot(p.x()-getX(),p.y()-getY()));
                state.decision(System.nanoTime()-decisionStart,plus,minus);state.changed();
            }
            return true;
        }
    }
}
