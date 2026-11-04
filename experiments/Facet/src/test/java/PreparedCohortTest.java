import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;
import java.util.*;

/** Actual Core handoff regressions; synthetic fixtures, no engine or opponent data. */
public final class PreparedCohortTest {
  static int assertions;
  static final Physics.Rules RULES = new Physics.Rules(800, 600, .1);
  static void check(boolean ok, String why) { assertions++; if (!ok) throw new AssertionError(why); }
  static void same(double a, double b, String why) { check(Math.abs(a-b)<1e-12, why); }
  static Physics.Body body(double x,double y,double heading,double speed,double gun) {
    return new Physics.Body(new Physics.V(x,y),heading,speed,gun,0,0);
  }
  static void add(Learner.Enemy e, int sequence) {
    e.longGun.add(new Memory.Observation(new double[Learner.LONG.length],Memory.Label.PRECISE_GF,
        .5,0,sequence,1,sequence,1,sequence,false));
    for (var s:e.recent) s.add(new Memory.Observation(new double[Learner.RECENT.length],
        Memory.Label.PRECISE_GF,.5,0,sequence,1,sequence,1,sequence,false));
  }
  record Scenario(Core core,Learner.Enemy enemy,Physics.Body requestOwn,FireControl.Plan plan) {}
  static Scenario prepare(boolean learned) {
    return prepare(learned,false);
  }
  static Scenario prepare(boolean learned,boolean disabled) {
    Core c=new Core(RULES,new Diagnostics(),1);c.roundReset(1);
    if(disabled)c.log.disabled.add("gun");
    var e=c.enemy(2);var own=body(400,300,0,0,.65);
    e.observe(new Memory.Sample(1,39,body(600,292,Math.PI/2,8,0),100,Memory.Quality.OBSERVED,0),own.p());
    if(learned){for(int i=1;i<=16;i++)add(e,i);e.crossings=16;}
    var intent=c.decide(own,100,40,1,List.of(new ScannedBotEvent(40,1,2,100,600,300,90,8)),List.of(),1);
    check(!intent.fire(),"first preparation is not physical submission");
    check(c.fire.plan!=null && c.fire.plan.cohort()!=null,"real preparation retains cohort");
    var moved=Physics.step(own,new Physics.Control(intent.move().speed(),intent.move().turn(),
        intent.gunTurn(),intent.radarTurn()),RULES).body();
    check(Math.abs(Physics.wrap(moved.gun()-c.fire.plan.angle()))<1e-12,"legal next gun rotation aligns plan");
    Physics.endWork();return new Scenario(c,e,moved,c.fire.plan);
  }
  static Core.Intent request(Scenario s,double headingOffset) {
    var b=s.requestOwn();var own=new Physics.Body(b.p(),b.heading(),b.speed(),
        Physics.norm(b.gun()+headingOffset),b.radar(),b.heat());
    var intent=s.core().decide(own,100,41,1,
        List.of(new ScannedBotEvent(41,1,2,100,600,307,90,7)),List.of(),1);
    check(intent.fire(),"minor lawful scan change leaves prior plan valid");
    Physics.endWork();return intent;
  }
  static Waves.Wave confirm(Scenario s, double offset, int id) {
    return confirm(s,offset,id,41);
  }
  static Waves.Wave confirm(Scenario s, double offset, int id, int requestTurn) {
    var c=s.core();var b=s.requestOwn();var own=new Physics.Body(b.p(),b.heading(),b.speed(),
        Physics.norm(b.gun()+offset),b.radar(),b.heat());
    var frozen=c.shotContext;var cohort=c.fire.plan.cohort();
    c.fire.submitted(true,own,requestTurn,1000);
    check(c.fire.plan==null && c.fire.pending.cohort()==cohort,"accepted Pending owns snapshot after Plan clears");
    var created=new BulletState(id,1,2,own.p().x(),own.p().y(),Math.toDegrees(own.gun()),null);
    var end=own.p().add(Physics.vector(own.gun()).mul(14));
    var moving=new BulletState(id,1,2,end.x(),end.y(),Math.toDegrees(own.gun()),null);
    int creation=requestTurn+1;
    c.observe(own,98,creation,1,List.of(new BulletFiredEvent(creation,created),
        new ScannedBotEvent(creation,1,2,100,600,315+8*(requestTurn-41),90,8)),List.of(moving),1,false);
    var wave=c.waves.all().stream().filter(w->w.role==Waves.Role.OUTGOING && Objects.equals(w.bullet,id)).findFirst().orElseThrow();
    check(wave.preparedCohort==cohort && wave.ratingEligible,"physical wave carries accepted immutable snapshot");
    check(wave.start==requestTurn && wave.creationTurn==creation && wave.firstMotionTurn==creation,"native physical clocks unchanged");
    check(wave.origin.equals(own.p()),"physical origin remains actual request origin");
    check(wave.predictions.equals(frozen.predictions),"confirmation preserves request binding");
    for(var p:cohort.predictions().values()) {
      var q=wave.predictions.get(p.gun());
      same(q.angle(),p.angle(),"prepared angle unchanged through confirmation");
      check(q.turn()==40 && q.version()==p.version() && q.wave()==wave.id,"time/version preserved; physical identity explicitly rebound");
    }
    check(c.fire.pending==null,"confirmation consumes pending once");
    long count=c.log.counters.getOrDefault("gun.cohort.confirmed",0L);
    c.observe(own,98,creation,1,List.of(new BulletFiredEvent(creation,created)),List.of(moving),1,false);
    check(c.log.counters.getOrDefault("gun.cohort.confirmed",0L)==count,"duplicate event adds no cohort publication");
    return wave;
  }
  static void mainRegression() {
    var s=prepare(true);var c=s.core();var cohort=s.plan().cohort();
    check(cohort.predictions().size()==3,"eligible ensemble frozen at preparation");
    add(s.enemy(),200);s.enemy().selected="long"; // New model data and a changed selector cannot rewrite old opportunity.
    request(s,0);
    check(c.fire.plan==s.plan() && cohort.selected().equals("analytic"),"request retains prepared selection despite current selector change");
    double fresh=Physics.intercept(c.shotContext.origin,c.shotContext.targetOrigin,2,RULES);
    check(Math.abs(Physics.wrap(fresh-cohort.predictions().get("analytic").angle()))>Math.toRadians(.5),"fixture retains historical fresh-vs-prepared discrepancy");
    same(c.shotContext.predictions.get("analytic").angle(),s.plan().angle(),"regression: scored analytic uses prepared angle");
    same(cohort.features()[4],0,"forecast missing acceleration preserved as provenance");
    same(c.shotContext.features[4],1.0/3,"physical training context is fresh, not forecast features");
    check(cohort.preparedTurn()==40 && cohort.expectedOriginTurn()==41,"forecast times retained separately");
    var physical=confirm(s,0,401);
    check(physical.preparedCohort.sourceWave()!=physical.id,"forecast identity retained separately from physical wave");
    int before=s.enemy().longGun.size();
    var point=physical.origin.add(Physics.vector(s.plan().angle()).mul(250));
    var outcome=new Memory.Sample(1,60,body(point.x(),point.y(),0,0,0),100,Memory.Quality.OBSERVED,0);
    Learner.trainGun(s.enemy(),physical,outcome,2000);
    for(var rating:s.enemy().ratings.values())check(rating.count==1,"common denominator advances once for prepared cohort");
    same(s.enemy().ratings.get("analytic").raw,1,"rating evaluates prepared angle against outcome");
    check(Arrays.equals(s.enemy().longGun.data.peekLast().features(),Learner.select(physical.features,Learner.LONG)),"new label uses actual-origin physical features");
    Learner.trainGun(s.enemy(),physical,outcome,2001);
    check(s.enemy().longGun.size()==before+1 && s.enemy().ratings.get("analytic").count==1,"duplicate outcome cannot retrain or rerate");
    c.roundReset(2);check(c.fire.plan==null && c.fire.pending==null && c.shotContext==null && c.waves.all().isEmpty(),"round reset clears ephemeral cohorts");
    c.log.close();
  }
  static void eligibilityAndTolerance() {
    var cold=prepare(false);var old=cold.plan().cohort();
    for(int i=1;i<=16;i++)add(cold.enemy(),i);cold.enemy().crossings=16;
    request(cold,Math.toRadians(.25));
    var wave=confirm(cold,Math.toRadians(.25),402);
    check(wave.predictions.keySet().equals(Set.of("analytic")),"newly eligible models not added to old cohort");
    same(Math.abs(Physics.wrap(wave.actualAngle-old.predictions().get("analytic").angle())),Math.toRadians(.25),"actual heading stays separate within unchanged tolerance");
    cold.core().log.close();
    var ready=prepare(true);ready.core().log.disabled.add("gun");request(ready,0);
    check(ready.core().shotContext.predictions.size()==3,"later disablement does not drop prepared eligible guns");
    ready.core().log.close();
    var fallback=prepare(true,true);request(fallback,0);
    check(fallback.plan().cohort().sourceWave()==-1 && fallback.plan().cohort().features().length==0,
        "cheap analytic fallback has explicit absent NN-context provenance");
    check(confirm(fallback,0,404).predictions.keySet().equals(Set.of("analytic")),"fallback preserves only its prepared analytic prediction");
    fallback.core().log.close();
  }
  static void lifetimeAndCancellation() {
    var delayed=prepare(true);var c=delayed.core();var b=delayed.requestOwn();
    var hot=new Physics.Body(b.p(),b.heading(),b.speed(),b.gun(),b.radar(),.1);
    c.decide(hot,100,41,1,List.of(new ScannedBotEvent(41,1,2,100,600,307,90,7)),List.of(),1);
    c.fire.plan=delayed.plan(); // Exercise the unchanged public two-turn lifetime explicitly.
    check(c.decide(b,100,42,1,List.of(new ScannedBotEvent(42,1,2,100,600,315,90,8)),List.of(),1).fire(),"valid two-turn-old plan remains eligible");
    Physics.endWork();
    var wave=confirm(delayed,0,405,42);
    check(wave.preparedCohort.expectedOriginTurn()==41 && wave.start==42,"forecast origin time does not overwrite delayed actual request time");
    same(c.log.observations.get("gun.cohort.requestAge")[3],2,"delayed preparation age is explicit");c.log.close();
    var expired=prepare(true);expired.enemy().observe(new Memory.Sample(1,43,body(600,321,Math.PI/2,7,0),100,Memory.Quality.OBSERVED,0),expired.requestOwn().p());
    check(!expired.core().fire.valid(expired.requestOwn(),100,expired.enemy(),43,RULES),"expiry gate is unchanged");
    expired.core().roundReset(2);check(expired.core().fire.plan==null,"expired snapshot cleared on reset");expired.core().log.close();
    var dead=prepare(true);var dc=dead.core();
    check(!dc.decide(dead.requestOwn(),100,41,1,List.of(new BotDeathEvent(41,2)),List.of(),1).fire() && dc.fire.plan==null,"target death cancels prepared plan");
    Physics.endWork();check(dead.enemy().ratings.values().stream().allMatch(r->r.count==0),"cancelled plan rates no gun");dc.log.close();
  }
  static void immutableAndDiscarded() {
    var s=prepare(true);var c=s.plan().cohort();var features=c.features();features[0]=99;
    check(c.features()[0]!=99,"feature accessor cannot mutate snapshot");
    try {c.predictions().clear();throw new AssertionError("mutable map");}catch(UnsupportedOperationException expected){assertions++;}
    var source=c.features();var map=new LinkedHashMap<>(c.predictions());
    var copy=new FireControl.Cohort(c.selected(),c.target(),c.meleeOrigin(),c.preparedTurn(),c.expectedOriginTurn(),c.sourceWave(),c.origin(),c.targetOrigin(),source,c.positive(),c.negative(),c.quality(),c.missingFeatures(),map);
    source[0]=99;map.clear();check(copy.features()[0]!=99 && copy.predictions().size()==3,"constructor deep-copies mutable containers");
    request(s,0);s.core().fire.submitted(false,s.requestOwn(),41,1000);
    check(s.core().fire.pending==null,"rejection creates no pending cohort");
    var unexpected=new BulletState(403,1,2,s.requestOwn().p().x(),s.requestOwn().p().y(),Math.toDegrees(s.requestOwn().gun()),null);
    s.core().observe(s.requestOwn(),98,42,1,List.of(new BulletFiredEvent(42,unexpected)),List.of(unexpected),1,false);
    check(s.core().waves.all().stream().noneMatch(w->w.role==Waves.Role.OUTGOING),"unmatched creation publishes no discarded cohort");
    check(s.core().shotContext==null,"reconciliation discards rejected shot context");
    s.core().log.close();
  }
  public static void main(String[] args) {
    mainRegression();eligibilityAndTolerance();immutableAndDiscarded();lifetimeAndCancellation();
    Physics.endWork();System.out.println("PASS prepared cohort actual-Core regression assertions="+assertions+" battles=0");
  }
}
