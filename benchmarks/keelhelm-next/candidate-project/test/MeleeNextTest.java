import static org.junit.jupiter.api.Assertions.*;
import java.util.List;
import org.junit.jupiter.api.Test;

class MeleeNextTest {
    @Test void radarRevisitIsBoundedAndDoesNotLoseStaleOrUnseenOpponents() {
        MeleePolicy p = new MeleePolicy(); Phys.Pose self = new Phys.Pose(100,100,0,0);
        p.observe(1,20,300,100,0,0,100); p.observe(2,20,100,300,0,0,100);
        var target=p.opponents().stream().filter(o->o.id==1).findFirst().orElseThrow();
        assertEquals(15,p.revisitRadar(22,self,0,1,2,target));
        assertEquals(45,p.revisitRadar(21,self,0,1,2,target),"outside revisit slot");
        assertEquals(45,p.revisitRadar(22,self,0,1,3,target),"unseen live opponent requires sweep");
        assertEquals(45,p.revisitRadar(22,self,90,1,2,target),"large detour prohibited");
        p.observe(1,30,300,100,0,0,100);
        assertEquals(45,p.revisitRadar(31,self,0,1,2,target),"other opponent stale");
        assertEquals(-45,p.revisitRadar(30,self,90,-1,2,target));
    }
    @Test void obstructionRequiresFreshNonTargetOnActualBulletPath() {
        MeleePolicy p=new MeleePolicy();Phys.Pose self=new Phys.Pose(100,100,0,0);
        p.observe(1,10,500,100,0,0,100);p.observe(2,10,250,100,0,0,100);
        var target=p.opponents().stream().filter(o->o.id==1).findFirst().orElseThrow();
        assertTrue(p.obstructed(10,self,target,0,1,1000,1000));
        assertFalse(p.obstructed(10,self,target,90,1,1000,1000));
        p.observe(1,19,500,100,0,0,100);
        assertFalse(p.obstructed(19,self,target,0,1,1000,1000),"stale blocker excluded");
        p.observe(2,20,600,100,0,0,100);
        assertFalse(p.obstructed(20,self,target,0,1,1000,1000),"behind target");
        p.died(2);assertFalse(p.obstructed(20,self,target,0,1,1000,1000));
    }
    @Test void obstructionUsesCurrentGunRayEvenWhenAimIsWithinFireTolerance() {
        MeleePolicy p=new MeleePolicy();Phys.Pose self=new Phys.Pose(100,100,0,0);
        p.observe(1,10,1000,100,0,0,100);p.observe(2,10,406,83,0,0,100);
        var target=p.opponents().stream().filter(o->o.id==1).findFirst().orElseThrow();
        double desiredAim=1, currentGun=0;
        assertTrue(Math.abs(desiredAim-currentGun)<Math.toDegrees(Math.atan(18.0/900))*.9);
        assertTrue(p.obstructed(10,self,target,currentGun,1,1000,1000));
        assertFalse(p.obstructed(10,self,target,desiredAim,1,1000,1000));
    }
    @Test void movingBlockerIsForecastRatherThanFrozenAtScan() {
        MeleePolicy p=new MeleePolicy();Phys.Pose self=new Phys.Pose(100,100,0,0);
        p.observe(1,10,500,100,0,0,100);p.observe(2,10,270,20,90,8,100);
        var target=p.opponents().stream().filter(o->o.id==1).findFirst().orElseThrow();
        assertTrue(p.obstructed(10,self,target,0,1,1000,1000));
    }
    @Test void transferIsPerIdFreshAndClearedAtDeathOrRoundBoundary() {
        var c=new MeleeTransfer();var a=new MeleePolicy.Opponent(1);var b=new MeleePolicy.Opponent(2);
        for(int t=10;t<=20;t++) {
            c.observe(1,t,new Phys.Pose(200+t,200,0,1),60,100,100);
            c.observe(2,t,new Phys.Pose(400,400+t,90,1),80,100,100);
            c.tick(t,new Phys.Pose(100+t,100,0,1));
        }
        assertNull(c.survivor(21,List.of(a,b)));assertNull(c.survivor(29,List.of(a)));
        c.died(2);var s=c.survivor(21,List.of(a));assertNotNull(s);
        assertEquals(220,s.enemy.get(20).x());assertEquals(120,s.own.get(20).x());assertEquals(60,s.energy);
        var copy=new Track();copy.copyFrom(s.enemy);c.clear();assertNull(c.survivor(21,List.of(a)));
        assertEquals(11,copy.observations());assertEquals(220,copy.get(20).x());
    }
    @Test void copiedHistoryKeepsFeaturesAndDoesNotAliasSubsequentWrites() {
        var source=new Track();for(int t=0;t<50;t++)source.add(t,new Phys.Pose(200+t,300,0,1),100,100);
        var copy=new Track();copy.copyFrom(source);
        assertArrayEquals(source.features(49,100,100,17,1000,1000),copy.features(49,100,100,17,1000,1000));
        source.clear();assertEquals(50,copy.observations());assertNotNull(copy.get(49));
        copy.clear();assertEquals(0,copy.observations());assertNull(copy.get(49));
    }
}
