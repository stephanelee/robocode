import java.net.*;
import java.nio.file.*;
import java.lang.reflect.*;
/** Offline algorithm parity only: no engine, no competitive inference. */
public class MovementEquivalence {
    static class Policy implements AutoCloseable {
        final URLClassLoader loader;
        final Class<?> type,pose;
        final Object instance;
        Policy(String jar)throws Exception {
            loader=new URLClassLoader(new URL[]{Path.of(jar).toUri().toURL()},null);
            type=loader.loadClass("MeleePolicy");pose=loader.loadClass("Phys$Pose");
            var constructor=type.getDeclaredConstructor();constructor.setAccessible(true);instance=constructor.newInstance();
        }
        Object call(String name,Class<?>[] signature,Object...args)throws Exception {
            var method=type.getDeclaredMethod(name,signature);method.setAccessible(true);return method.invoke(instance,args);
        }
        void reset()throws Exception {call("newRound",new Class[]{int.class},3);}
        void observe(int id,int t,double x,double y,double h,double v,double energy)throws Exception {
            call("observe",new Class[]{int.class,int.class,double.class,double.class,double.class,double.class,double.class},id,t,x,y,h,v,energy);
        }
        String move(int t,double x,double y,double h,double v,Boolean predictive)throws Exception {
            var ctor=pose.getDeclaredConstructor(double.class,double.class,double.class,double.class);ctor.setAccessible(true);Object p=ctor.newInstance(x,y,h,v);
            return (predictive==null?call("move",new Class[]{int.class,pose,double.class,double.class,double.class},t,p,63.,1000.,1000.):call("move",new Class[]{int.class,pose,double.class,double.class,double.class,boolean.class},t,p,63.,1000.,1000.,predictive)).toString();
        }
        public void close()throws Exception {loader.close();}
    }
    public static void main(String[]args)throws Exception {
        int comparisons=0;
        try(var base=new Policy(args[0]);var candidate=new Policy(args[1]);var old=new Policy(args[2]);var candidateOff=new Policy(args[1])){
            for(int scenario=0;scenario<8;scenario++) {
                for(var p:new Policy[]{base,candidate,old,candidateOff})p.reset();
                for(int t=1;t<=80;t++) {
                    for(int id=1;id<=3;id++) if((t+id)%5==0)for(var p:new Policy[]{base,candidate,old,candidateOff})p.observe(id,t,100+(t*7+id*163)%800,100+(t*11+id*197)%800,(t*9+id*70)%360,(t+id)%3==0?0:8,100-t*.6);
                    double x=80+(t*13+scenario*107)%840,y=80+(t*17+scenario*71)%840,h=t*11%360,v=t%9==0?0:8;
                    String a=base.move(t,x,y,h,v,null),b=candidateOff.move(t,x,y,h,v,false);
                    if(!a.equals(b))throw new AssertionError("parent mismatch "+scenario+":"+t+" "+a+" "+b);
                    a=old.move(t,x,y,h,v,true);b=candidate.move(t,x,y,h,v,null);
                    if(!a.equals(b))throw new AssertionError("risk mismatch "+scenario+":"+t+" "+a+" "+b);
                    comparisons+=2;
                }
            }
        }
        System.out.println("PASS "+comparisons+" command comparisons; 640 reference-risk + 640 accepted-parent. Offline only.");
    }
}
