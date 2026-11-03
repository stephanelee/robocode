import java.util.List;

/** Pure phase and actual-ray matching bridge; leaves legacy AIS wave learning/removal unchanged. */
final class BreakwaterWaveBridge {
    private BreakwaterWaveBridge() {}

    /** AIS stores the decision turn; the transferred movement expects creation on the following turn. */
    static BreakwaterSurf.Threat from(EnemyWave wave,AimCorridor.OwnState aimOwn) {
        if(wave==null || aimOwn==null || aimOwn.turn()!=wave.fireTurn
                || !finite(wave.originX,wave.originY,wave.bulletSpeed,aimOwn.x(),aimOwn.y(),aimOwn.direction(),aimOwn.speed())
                || wave.bulletSpeed<=8 || wave.bulletSpeed>=20)return null;
        // bearingToUs can use a later scan callback position: recompute from exactly the saved aim state.
        double reference=CombatMath.bearing(aimOwn.x()-wave.originX,aimOwn.y()-wave.originY);
        double lateral=aimOwn.speed()*Math.sin(Math.toRadians(aimOwn.direction()-reference));
        return new BreakwaterSurf.Threat(wave.fireTurn+1,wave.originX,wave.originY,wave.bulletSpeed,reference,lateral);
    }

    /**
     * Match only a uniquely identified actual ray, using source69's tolerances. No nearest/broad radial fallback.
     * Caller owns its independent candidate list and consumes the returned estimate once after learning.
     */
    static EnemyWave uniqueMatch(List<EnemyWave> waves,int owner,double power,int turn,
                                 double x,double y,double direction) {
        EnemyWave matched=null;
        for(EnemyWave wave:waves) {
            if(!matches(wave,owner,power,turn,x,y,direction))continue;
            if(matched!=null)return null;
            matched=wave;
        }
        return matched;
    }
    /** Caller must reject multiple matches rather than choosing a nearest candidate. */
    static boolean matches(EnemyWave wave,int owner,double power,int turn,double x,double y,double direction) {
        if(wave==null||!finite(power,x,y,direction,wave.originX,wave.originY,wave.bulletSpeed)||power<.1||power>3
                || wave.shooterId!=owner || turn<wave.fireTurn+1
                || Math.abs(power-(20-wave.bulletSpeed)/3)>1e-5)return false;
        double radians=Math.toRadians(direction),cos=Math.cos(radians),sin=Math.sin(radians);
        double dx=x-wave.originX,dy=y-wave.originY;
        double along=dx*cos+dy*sin,across=Math.abs(dx*sin-dy*cos);
        // tick increments before publishing an impact: v*(turn-creation+1)=v*(turn-AISdecision).
        double radius=(20-3*power)*(turn-wave.fireTurn);
        return across<=1e-3&&Math.abs(along-radius)<=1e-3;
    }
    private static boolean finite(double... values){for(double v:values)if(!Double.isFinite(v))return false;return true;}
}
