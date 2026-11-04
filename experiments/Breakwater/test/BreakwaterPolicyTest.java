import java.util.List;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class BreakwaterPolicyTest {
    private static BreakwaterPolicy.Own own(int turn, double x, double y, double gun, double heat, double energy) {
        return new BreakwaterPolicy.Own(turn, x, y, 90, 8, gun, heat, energy, .1, 800, 600);
    }

    private static BreakwaterPolicy.Target target(int turn, double x, double y, double heading, double speed) {
        return new BreakwaterPolicy.Target(turn, x, y, heading, speed, 100);
    }

    @Test void directAndDelayedReversalObeyEngineMotionOrder() {
        var p = new BreakwaterPolicy.Pose(100, 100, 0, 8);
        var a = BreakwaterPolicy.step(p, new BreakwaterPolicy.Motion(4,-8));
        assertEquals(3, a.speed()); assertEquals(103, a.x()); assertEquals(100, a.y()); assertEquals(4, a.heading());
        var b = BreakwaterPolicy.step(p, new BreakwaterPolicy.Motion(4,0));
        assertEquals(0, b.speed()); assertEquals(100, b.x());
    }

    @Test void freshAlignedAttackFiresWhileTrackingTwoThreatsAndMoving() {
        var p = new BreakwaterPolicy(101);
        var waves = List.of(new BreakwaterPolicy.Wave(22,120,310,1.95,0,8),
                            new BreakwaterPolicy.Wave(35,120,310,1.95,0,8));
        var d = p.decide(own(35,640,310,180,0,100), target(35,120,310,90,0), waves);
        assertEquals(2,d.waves()); assertEquals(1.35,d.power()); assertEquals(8,Math.abs(d.targetSpeed()));
        assertEquals(0,p.decide(own(36,640,310,150,0,100), target(36,120,310,90,0), waves).power());
        assertEquals(0,p.decide(own(37,640,310,180,0.1,100), target(37,120,310,90,0), waves).power());
        assertEquals(-1,p.decide(own(40,640,310,180,0,100), target(37,120,310,90,0), waves).gun());
    }

    @Test void lowEnergyAndFinishersReduceOffensivePower() {
        var p = new BreakwaterPolicy(9);
        assertEquals(2,p.decide(own(20,500,300,180,0,100),target(20,240,300,0,0),List.of()).power());
        assertEquals(.5,p.decide(own(21,500,300,180,0,8),target(21,240,300,0,0),List.of()).power());
        var finishing = new BreakwaterPolicy.Target(22,240,300,0,0,1);
        assertEquals(1.001/4,p.decide(own(22,500,300,180,0,100),finishing,List.of()).power());
    }

    @Test void mirroredMatchedAnglesShareLearningButStationaryLaunchesDoNot() {
        var p = new BreakwaterPolicy(7);
        var positive = new BreakwaterPolicy.Wave(10,100,300,2,358,5);
        var negative = new BreakwaterPolicy.Wave(12,100,300,2,2,-5);
        p.matched(positive,2); p.matched(negative,358);
        p.matched(new BreakwaterPolicy.Wave(14,100,300,2,250,.5),260);
        assertArrayEquals(new double[]{4/positive.escape(),4/positive.escape()},p.corrections(),1e-9);
        p.startRound(); assertEquals(2,p.corrections().length);
        assertEquals(0,new BreakwaterPolicy(7).corrections().length);
    }

    @Test void oldHeadingWallEmergencyDoesNotDriveIntoTheWall() {
        var p=new BreakwaterPolicy(8);
        var own=new BreakwaterPolicy.Own(10,19,400,180,8,0,0,100,.1,800,600);
        var d=p.decide(own,target(10,510,300,10,7),List.of());
        var next=BreakwaterPolicy.step(new BreakwaterPolicy.Pose(19,400,180,8),
                new BreakwaterPolicy.Motion(d.turnRate(),d.targetSpeed()));
        assertTrue(next.x()>=18);
    }

    @Test void jointPassagesNeedConfirmationAndContiguousScansAndOnlyResolvedMemorySurvivesReset() {
        var p = new BreakwaterPolicy(11);
        var t = target(20,180,300,90,0);
        p.observe(t);
        var d=p.decide(own(20,100,300,0,0,100),t,List.of());
        p.observe(target(21,180,300,90,0)); // creation scan before fired notification
        p.attackLaunched(21,100,300,d.power(),t);
        for(int i=22;i<=27;i++) p.observe(target(i,180,300,90,0));
        assertArrayEquals(new int[]{7},p.attackHistory());
        assertArrayEquals(new int[]{1,1,1,1},p.attackScores());
        t=target(28,180,300,90,0);
        d=p.decide(own(28,100,300,0,0,100),t,List.of());
        p.attackLaunched(29,100,300,d.power(),t);
        p.observe(target(30,180,300,90,0)); // missing creation turn
        p.startRound(); p.observe(target(1,180,300,90,0));
        assertArrayEquals(new int[]{7},p.attackHistory());
        assertEquals(0,new BreakwaterPolicy(11).attackHistory().length);
    }

    @Test void coldCircularRayUsesOnlyPastConsecutiveMotionAndAStaleHistoryRestartsStraight() {
        var p=new BreakwaterPolicy(8);
        p.observe(target(19,500,300,0,8));
        var t=target(20,508,300,4,8);
        p.observe(t);
        var d=p.decide(own(20,150,300,0,1,100),t,List.of());
        assertEquals(2,d.gun()); assertTrue(d.aim()>6,d.toString());
        t=target(23,508,300,0,8);
        p.observe(t);
        assertEquals(0,p.decide(own(23,150,300,0,1,100),t,List.of()).aim(),1e-9);
    }

    @Test void seedControlsNearTieChoiceAndIdenticalSeedsReplay() {
        var plus=false; var minus=false;
        for(int seed=0;seed<16;seed++) {
            var own=own(30,400,300,0,0,100); var t=target(30,650,300,0,8);
            var a=new BreakwaterPolicy(seed).decide(own,t,List.of());
            var b=new BreakwaterPolicy(seed).decide(own,t,List.of());
            assertEquals(a,b);
            plus |= a.orbitSign()>0; minus |= a.orbitSign()<0;
        }
        assertTrue(plus && minus);
    }
}
