import java.util.*;
import java.lang.reflect.*;
import dev.robocode.tankroyale.server.model.Point;
import dev.robocode.tankroyale.server.model.Line;
import dev.robocode.tankroyale.server.core.CollisionDetector;

/** Independent mathematical fixtures against actual pinned official predicates; no bot code. */
public final class ShadowOracle {
 static final double TAU=2*Math.PI,BROAD=39.4,ERODE=1e-9;
 static final int ANGLES=65536;
 record V(double x,double y){V add(V q){return new V(x+q.x,y+q.y);}V sub(V q){return new V(x-q.x,y-q.y);}V mul(double k){return new V(x*k,y*k);}double dot(V q){return x*q.x+y*q.y;}double length(){return Math.hypot(x,y);}Point point(){return new Point(x,y);}}
 record Arc(double lo,double hi){}
 record Case(String name,V origin,V ownStart,V ownEnd,double speed,int originTurn,int tick,boolean confirmed,int deathTurn,double originError,boolean exactTime,boolean exactPower,boolean legalSegment){}
 static Object detector;static Method broad;
 static double norm(double a){return (a%TAU+TAU)%TAU;}
 static double wrap(double a){return norm(a+Math.PI)-Math.PI;}
 static List<Arc> around(double center,double half){if(half>=Math.PI)return List.of(new Arc(0,TAU));double lo=norm(center-half),hi=norm(center+half);return lo<=hi?List.of(new Arc(lo,hi)):List.of(new Arc(0,hi),new Arc(lo,TAU));}
 static List<Arc> intersect(List<Arc>a,List<Arc>b){List<Arc> out=new ArrayList<>();for(Arc x:a)for(Arc y:b){double lo=Math.max(x.lo,y.lo),hi=Math.min(x.hi,y.hi);if(hi>lo)out.add(new Arc(lo,hi));}return out;}
 static boolean supported(Case c){return c.confirmed&&c.deathTurn>c.tick&&c.originError==0&&c.exactTime&&c.exactPower&&c.tick>c.originTurn;}
 static void roots(List<Double> ts,V a,V d,double radius){double A=d.dot(d),B=2*a.dot(d),C=a.dot(a)-radius*radius;if(A==0)return;double disc=B*B-4*A*C;if(disc<0)return;double sq=Math.sqrt(disc);for(double t:new double[]{(-B-sq)/(2*A),(-B+sq)/(2*A)})if(t>0&&t<1)ts.add(t);}
 static List<Arc> analytic(Case c){
  if(!supported(c))return List.of();double inner=c.speed*(c.tick-1-c.originTurn),outer=c.speed*(c.tick-c.originTurn);V a=c.ownStart.sub(c.origin),d=c.ownEnd.sub(c.ownStart);List<Double> ts=new ArrayList<>(List.of(0.,1.));roots(ts,a,d,inner);roots(ts,a,d,outer);ts.sort(Double::compare);List<Arc> arcs=new ArrayList<>();
  for(int i=1;i<ts.size();i++){double lo=ts.get(i-1),hi=ts.get(i);if(hi-lo<=1e-12)continue;V mid=a.add(d.mul((lo+hi)/2));double radius=mid.length();if(radius<inner||radius>outer)continue;V p=a.add(d.mul(lo)),q=a.add(d.mul(hi));if(p.length()<1e-10||q.length()<1e-10)continue;double start=Math.atan2(p.y,p.x),end=Math.atan2(q.y,q.x),delta=wrap(end-start);double half=Math.abs(delta)/2-ERODE;if(half>0)arcs.addAll(around(start+delta/2,half));}
  V end=c.ownEnd.sub(c.origin);double dist=end.length();if(dist==0)return outer<=BROAD?arcs:List.of();double cosine=(dist*dist+outer*outer-BROAD*BROAD)/(2*dist*outer);if(cosine>1)return List.of();if(cosine<=-1)return arcs;double half=Math.acos(cosine)-ERODE;if(half<=0)return List.of();return intersect(arcs,around(Math.atan2(end.y,end.x),half));
 }
 static boolean credited(List<Arc> arcs,double angle){angle=norm(angle);for(Arc a:arcs)if(angle>=a.lo&&angle<=a.hi)return true;return false;}
 static boolean engine(Case c,double angle)throws Exception{
  if(!supported(c))return false;double before=c.speed*(c.tick-1-c.originTurn),after=c.speed*(c.tick-c.originTurn);V u=new V(Math.cos(angle),Math.sin(angle));V start=c.origin.add(u.mul(before)),end=c.origin.add(u.mul(after));boolean endpoints=(boolean)broad.invoke(detector,c.ownEnd.point(),end.point());return endpoints&&dev.robocode.tankroyale.server.model.MathKt.isLineIntersectingLine(new Line(c.ownStart.point(),c.ownEnd.point()),new Line(start.point(),end.point()));
 }
 static void setup()throws Exception{
  Class<?> setupClass=Class.forName("dev.robocode.tankroyale.server.model.GameSetup");Constructor<?> ctor=Arrays.stream(setupClass.getDeclaredConstructors()).filter(c->c.getParameterCount()==20).findFirst().orElseThrow();ctor.setAccessible(true);Object[] args=new Object[20];Class<?>[] types=ctor.getParameterTypes();for(int i=0;i<20;i++){Class<?> t=types[i];args[i]=t==String.class?"1v1":t==int.class?800:t==long.class?100000L:t==double.class?.1:t==boolean.class?false:t==Integer.class?2:null;}Object setup=ctor.newInstance(args);detector=CollisionDetector.class.getConstructor(setupClass,Set.class,Random.class).newInstance(setup,Set.of(),new Random(0));broad=CollisionDetector.class.getDeclaredMethod("isBulletsMaxBoundingCirclesColliding",Point.class,Point.class);broad.setAccessible(true);
  if(!(boolean)broad.invoke(detector,new Point(0,0),new Point(39.4,0))||(boolean)broad.invoke(detector,new Point(0,0),new Point(39.400001,0)))throw new AssertionError("actual engine broad phase differs");
 }
 static Case base(String name,V a,V b,int tick,boolean confirmed,int death,double error,boolean time,boolean power,boolean legal){V o=new V(400,300);return new Case(name,o,o.add(a),o.add(b),18,0,tick,confirmed,death,error,time,power,legal);}
 public static void main(String[] args)throws Exception{
  setup();List<Case> cases=List.of(
   base("partial-annulus",new V(90,10),new V(108,10),6,true,999,0,true,true,true),
   base("radially-disjoint",new V(200,30),new V(218,30),6,true,999,0,true,true,true),
   base("angular-wrap",new V(99,-9),new V(99,9),6,true,999,0,true,true,true),
   base("wrong-tick",new V(99,-9),new V(99,9),7,true,999,0,true,true,true),
   base("broad-phase-reject-overlong-robustness",new V(100,-100),new V(100,100),6,true,999,0,true,true,false),
   base("death-before",new V(99,-9),new V(99,9),6,true,5,0,true,true,true),
   base("death-same-tick",new V(99,-9),new V(99,9),6,true,6,0,true,true,true),
   base("pending-own-request",new V(99,-9),new V(99,9),6,false,999,0,true,true,true),
   base("uncertain-origin",new V(99,-9),new V(99,9),6,true,999,8,true,true,true),
   base("uncertain-time",new V(99,-9),new V(99,9),6,true,999,0,false,true,true),
   base("uncertain-power",new V(99,-9),new V(99,9),6,true,999,0,true,false,true)
  );
  long all=0,falseCredit=0;System.out.print("{\"schema\":1,\"oracle\":\"compiled official server segment and endpoint predicates\",\"anglesPerCase\":"+ANGLES+",\"cases\":[");boolean comma=false;
  for(Case c:cases){List<Arc> arcs=analytic(c);long samples=0,credits=0,hits=0,falsePos=0,missed=0;List<Double> angles=new ArrayList<>();for(int i=0;i<ANGLES;i++)angles.add(TAU*i/ANGLES);for(Arc a:arcs)for(double f:new double[]{0,.0001,.25,.5,.75,.9999,1})angles.add(a.lo+(a.hi-a.lo)*f);for(double a:angles){boolean credit=credited(arcs,a),hit=engine(c,a);samples++;if(credit)credits++;if(hit)hits++;if(credit&&!hit)falsePos++;if(hit&&!credit)missed++;}all+=samples;falseCredit+=falsePos;if(comma)System.out.print(",");comma=true;System.out.print("\n{\"name\":\""+c.name+"\",\"legalSingleTickOwnSegment\":"+c.legalSegment+",\"samples\":"+samples+",\"creditedSamples\":"+credits+",\"engineHitSamples\":"+hits+",\"falseCredit\":"+falsePos+",\"uncreditedEngineHits\":"+missed+",\"arcs\":[");for(int i=0;i<arcs.size();i++){if(i>0)System.out.print(",");Arc a=arcs.get(i);System.out.print("["+a.lo+","+a.hi+"]");}System.out.print("]}");}
  System.out.println("\n],\"samples\":"+all+",\"falseCredit\":"+falseCredit+",\"status\":\""+(falseCredit==0?"passed":"failed")+"\"}");if(falseCredit!=0)throw new AssertionError("false shadow credit");
 }
}
