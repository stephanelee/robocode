import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class FlightTest {
    @Test void reverseStopAndBrakeDiffer(){
        assertEquals(0,Flight.speed(8,0));assertEquals(3,Flight.speed(8,-8));
        assertEquals(-.5,Flight.speed(1,-8));assertEquals(-3,Flight.speed(-8,8));
        assertEquals(6,Flight.speed(8,2));assertEquals(1,Flight.speed(0,8));
    }
    @Test void translationPrecedesRotation(){
        var s=Flight.step(new Flight.Pose(100,100,90,7),new Flight.Command(90,8),800,600);
        assertEquals(100,s.x(),1e-8);assertEquals(108,s.y());assertEquals(94,s.h());
    }
    @Test void wallStopsAndConstrains(){var p=Flight.step(new Flight.Pose(780,50,0,8),new Flight.Command(0,8),800,600);
        assertEquals(782,p.x());assertEquals(0,p.v());}
    @Test void obliqueWallIntercept(){var p=Flight.step(new Flight.Pose(780,100,45,8),new Flight.Command(0,8),800,600);
        assertEquals(782,p.x());assertEquals(102,p.y(),1e-8);}
    @Test void preRotationRayAndFirstSweep(){
        assertFalse(Flight.canShoot(0,8,500,20,0,1));assertTrue(Flight.canShoot(359,0,500,20,0,1));
        assertFalse(Flight.canShoot(0,0,100,1,0,1));assertFalse(Flight.canShoot(0,0,100,10,.05,1));
        assertEquals(34,Flight.farEdge(1,1));
    }
}