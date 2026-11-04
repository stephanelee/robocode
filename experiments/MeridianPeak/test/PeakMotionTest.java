import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class PeakMotionTest {
    @Test void routeUsesBothCornerPassesWhileGunForecastRemainsOriginal() {
        var p=new Flight.Pose(780,580,Math.toDegrees(Math.atan2(4,7)),8);
        var command=new Flight.Command(0,8);
        var route=PeakMotion.step(p,command,800,600);
        assertEquals(782,route.x(),1e-12);assertEquals(581.1428571428571,route.y(),1e-12);assertEquals(0,route.v());
        var gun=Flight.step(p,command,800,600);
        assertEquals(783.5,gun.x(),1e-12);assertEquals(582,gun.y(),1e-12);
    }
    @Test void newSpeedTranslationAndTurnCapRemainEngineRules() {
        var p=PeakMotion.step(new Flight.Pose(200,200,0,8),new Flight.Command(10,-8),800,600);
        assertEquals(203,p.x());assertEquals(200,p.y());assertEquals(3,p.v());assertEquals(7.75,p.h());
        assertEquals(0,PeakMotion.step(new Flight.Pose(200,200,0,8),new Flight.Command(10,0),800,600).v());
    }
}
