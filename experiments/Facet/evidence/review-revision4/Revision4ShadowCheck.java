import java.util.*;
public class Revision4ShadowCheck {
 public static void main(String[] args)throws Exception{
  ShadowOracle.setup();List<ShadowOracle.Case> cases=List.of(
   ShadowOracle.base("partial-annulus",new ShadowOracle.V(90,10),new ShadowOracle.V(108,10),6,true,999,0,true,true,true),
   ShadowOracle.base("radially-disjoint",new ShadowOracle.V(200,30),new ShadowOracle.V(218,30),6,true,999,0,true,true,true),
   ShadowOracle.base("angular-wrap",new ShadowOracle.V(99,-9),new ShadowOracle.V(99,9),6,true,999,0,true,true,true),
   ShadowOracle.base("wrong-tick",new ShadowOracle.V(99,-9),new ShadowOracle.V(99,9),7,true,999,0,true,true,true),
   ShadowOracle.base("broad-phase-overlong-robustness",new ShadowOracle.V(100,-100),new ShadowOracle.V(100,100),6,true,999,0,true,true,false));
  long all=0,fail=0;for(var c:cases){var arcs=Waves.shadowSegment(new Physics.V(c.origin().x(),c.origin().y()),new Physics.V(c.ownStart().x(),c.ownStart().y()),new Physics.V(c.ownEnd().x(),c.ownEnd().y()),c.speed()*(c.tick()-1-c.originTurn()),c.speed()*(c.tick()-c.originTurn()));long credited=0,wrong=0;List<Double> angles=new ArrayList<>();for(int i=0;i<65536;i++)angles.add(2*Math.PI*i/65536);for(var a:arcs)for(double f:new double[]{0,.001,.25,.5,.75,.999,1})angles.add(a.lo()+(a.hi()-a.lo())*f);for(double angle:angles){boolean credit=false;for(var a:arcs)if(angle>=a.lo()&&angle<=a.hi())credit=true;if(credit){credited++;if(!ShadowOracle.engine(c,angle))wrong++;}}all+=angles.size();fail+=wrong;System.out.println(c.name()+": sampled="+angles.size()+" credited="+credited+" falseCredit="+wrong+" arcs="+arcs);}
  System.out.println("samples="+all+" falseCredit="+fail+" result="+(fail==0?"PASS":"FAIL"));if(fail>0)throw new AssertionError("unsupported shadow safety credit");
 }
}
