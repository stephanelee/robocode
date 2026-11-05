import java.util.*;
public class MeleeBudgetCheck {
 public static void main(String[] args){var r=new Physics.Rules(800,600,.1);var own=new Physics.Body(new Physics.V(100,300),0,0,0,0,0);var e=new Learner.Enemy(1,new Memory.Budget());e.observe(new Memory.Sample(1,40,new Physics.Body(new Physics.V(500,300),0,0,0,0,0),100,Memory.Quality.OBSERVED,0),own.p());for(int i=0;i<80;i++)e.displacement.add(new Memory.Observation(new double[16],Memory.Label.DISPLACEMENT,4,0,i,1,i,1,i,true));var m=new Melee(49667);var a=m.select(own,List.of(e),100,40,2,r);System.out.println("powerStages="+m.projectionPasses+" fullNeighborQueriesPerStage=80 projectedSamplesQueried="+(m.projectionPasses*80)+" selectedPower="+a.power()+" range="+a.policy().range());}
}
