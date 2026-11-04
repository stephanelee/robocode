import java.util.Collection;
import java.util.List;

/** One future finite-sweep peak per modeled ray, then mixture and wave summation. */
final class PeakRisk {
    record Evaluation(double score, double maximumWaveRisk, int waves, int rays,
                      int maximumRaysPerWave, long kernelEvaluations) {}
    static double distanceSquared(double x, double y, double ax, double ay, double bx, double by) {
        double dx=bx-ax, dy=by-ay;
        double u=Flight.clip(((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy),0,1);
        double ex=ax+u*dx-x, ey=ay+u*dy-y;
        return ex*ex+ey*ey;
    }
    static double kernel(double distanceSquared) { return 40/(1+distanceSquared/500); }
    static boolean outside(double x,double y,double w,double h) { return x<=0||y<=0||x>=w||y>=h; }
    static double nearRadius(HitAngles.Wave wave,int turn,int step) {return (turn+step-wave.t()+1)*wave.speed();}
    static double weightedPeaks(double[] weights,double[][] kernelsByTick) {
        double[] peak=new double[weights.length];
        for(var tick:kernelsByTick)for(int r=0;r<peak.length;r++)peak[r]=Math.max(peak[r],tick[r]);
        double sum=0;for(int r=0;r<peak.length;r++)sum+=weights[r]*peak[r];return sum;
    }
    static Evaluation evaluate(Flight.Pose p,int turn,int sign,int committedSide,double ex,double ey,
                               Collection<HitAngles.Wave> incoming,HitAngles profile,double width,double height) {
        var waves=List.copyOf(incoming);
        var rays=new java.util.ArrayList<List<HitAngles.Ray>>();
        var peaks=new java.util.ArrayList<double[]>();
        var alive=new java.util.ArrayList<boolean[]>();
        int rayCount=0,maxRays=0;
        for(var wave:waves) {
            var mixture=profile.rays(wave);rays.add(mixture);peaks.add(new double[mixture.size()]);
            var active=new boolean[mixture.size()];java.util.Arrays.fill(active,true);alive.add(active);
            rayCount+=mixture.size();maxRays=Math.max(maxRays,mixture.size());
        }
        double score=sign==committedSide?0:2;long evaluations=0;
        for(int step=1;step<=50;step++) {
            p=PeakMotion.step(p,Flight.orbit(p,ex,ey,sign,width,height),width,height);
            double edge=Math.min(Math.min(p.x(),width-p.x()),Math.min(p.y(),height-p.y()));
            score+=.10*Math.max(0,40-edge);
            for(int w=0;w<waves.size();w++) {
                var wave=waves.get(w);double near=nearRadius(wave,turn,step),far=near+wave.speed();
                for(int r=0;r<rays.get(w).size();r++)if(alive.get(w)[r]) {
                    var ray=rays.get(w).get(r);double c=Flight.cos(ray.heading()),s=Flight.sin(ray.heading());
                    double ax=wave.x()+near*c,ay=wave.y()+near*s;
                    if(outside(ax,ay,width,height)){alive.get(w)[r]=false;continue;}
                    double d2=distanceSquared(p.x(),p.y(),ax,ay,wave.x()+far*c,wave.y()+far*s);
                    peaks.get(w)[r]=Math.max(peaks.get(w)[r],kernel(d2));evaluations++;
                }
            }
        }
        double maximumWave=0;
        for(int w=0;w<waves.size();w++) {
            double risk=0;for(int r=0;r<rays.get(w).size();r++)risk+=rays.get(w).get(r).weight()*peaks.get(w)[r];
            score+=risk;maximumWave=Math.max(maximumWave,risk);
        }
        return new Evaluation(score,maximumWave,waves.size(),rayCount,maxRays,evaluations);
    }
}
