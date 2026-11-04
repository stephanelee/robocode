import java.util.*;
public class PreparedPredictionMismatch {
  static Physics.Body body(double x,double y,double h,double speed,double gun){return new Physics.Body(new Physics.V(x,y),h,speed,gun,0,0);}
  public static void main(String[] args){
    var rules=new Physics.Rules(800,600,.1);var diagnostics=new Diagnostics();var core=new Core(rules,diagnostics,1);core.roundReset(1);
    var enemy=core.enemy(2);var own=body(400,300,0,0,0);
    enemy.observe(new Memory.Sample(1,40,body(600,300,Math.PI/2,8,0),100,Memory.Quality.OBSERVED,1),own.p());
    Physics.beginWork(2400);
    var predicted=core.makeOutgoing(own,enemy,41,2,Waves.Role.VIRTUAL,1,true);
    double prepared= Learner.gun("analytic",enemy,predicted,0,rules);
    core.fire.plan=FireControl.prepare(prepared,2,enemy,own,40,1).withEvidence("gun=analytic mode=duel fixture");
    Physics.endWork();
    enemy.observe(new Memory.Sample(1,41,body(600,307,Math.PI/2,7,0),100,Memory.Quality.OBSERVED,1),own.p());enemy.crossings=16;
    own=body(400,300,0,0,prepared);
    if(!core.fire.valid(own,100,enemy,41,rules)) throw new AssertionError("prior plan must remain valid");
    var intent=core.decide(own,100,41,1,List.of(),List.of(),1);
    if(!intent.fire()||core.shotContext==null)throw new AssertionError("real Core failed to prepare physical request");
    double cohort=core.shotContext.predictions.get("analytic").angle();
    double difference=Math.abs(Physics.wrap(cohort-prepared));
    if(difference<Math.toRadians(.5))throw new AssertionError("fixture did not exceed maximum heading tolerance");
    System.out.println("synthetic-only fire="+intent.fire()+" preparedActualHeading="+prepared+" ratedAnalyticPrediction="+cohort+" differenceDegrees="+Math.toDegrees(difference)+" commonCohort="+core.shotContext.predictions.keySet());
    System.out.println("contexts plannedAccel="+predicted.features[4]+" plannedMissingMask="+predicted.missingFeatures+" physicalAccel="+core.shotContext.features[4]+" physicalMissingMask="+core.shotContext.missingFeatures);
    diagnostics.close();
  }
}
