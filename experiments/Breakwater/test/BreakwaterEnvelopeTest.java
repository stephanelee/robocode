import static org.junit.jupiter.api.Assertions.*;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class BreakwaterEnvelopeTest {
    private static BreakwaterPolicy.Own own(int turn, double gun) {
        return new BreakwaterPolicy.Own(turn, 100, 300, 0, 0, gun, 0, 100, .1, 800, 600);
    }

    private static BreakwaterPolicy.Target target(int turn, double x, double y, double speed) {
        return new BreakwaterPolicy.Target(turn, x, y, 90, speed, 100);
    }

    private static BreakwaterPolicy.Decision baseline(double aim, double power) {
        return new BreakwaterPolicy.Decision(0,8,aim,power,0,0,1);
    }

    @Test void projectionBrakesImmediatelyAndMovesOnTheOldHeading() {
        var p = new BreakwaterPolicy.Pose(400,300,90,8);
        var stop = BreakwaterEnvelope.advance(p,own(10,0),0);
        assertEquals(300,stop.y()); assertEquals(0,stop.speed());
        var reverse = BreakwaterEnvelope.advance(p,own(10,0),-1);
        assertEquals(400,reverse.x(),1e-9); assertEquals(303,reverse.y(),1e-9);
        assertEquals(3,reverse.speed());
    }

    @Test void wallUsesOriginalLineForBothCornerConstraintPasses() {
        var before = new BreakwaterPolicy.Pose(20,21,225,8);
        var next = new BreakwaterPolicy.Pose(15,17,228,8);
        var corrected = BreakwaterEnvelope.wall(before,next,800,600);
        assertEquals(18,corrected.x(),1e-9); assertEquals(19.4,corrected.y(),1e-9);
        assertEquals(0,corrected.speed()); assertEquals(228,corrected.heading());
    }

    @Test void radiusAndCreationSegmentLimitAngularFootprints() {
        assertTrue(Double.isNaN(BreakwaterEnvelope.halfFootprint(60,17,34)));
        assertTrue(Double.isNaN(BreakwaterEnvelope.halfFootprint(10,34,51)));
        assertEquals(Math.toDegrees(Math.asin(18.0/100)), BreakwaterEnvelope.halfFootprint(100,90,110),1e-9);
        assertTrue(BreakwaterEnvelope.halfFootprint(100,82,83) < BreakwaterEnvelope.halfFootprint(100,90,110));
    }

    @Test void supportCountsDistinctPathsAndWorksAcrossAngleZero() {
        var arcs = List.of(new BreakwaterEnvelope.Arc(0,359,2),new BreakwaterEnvelope.Arc(0,359,1),
                new BreakwaterEnvelope.Arc(1,1,2), new BreakwaterEnvelope.Arc(2,180,10));
        var choice = BreakwaterEnvelope.choose(arcs,0,180);
        assertEquals(2,choice.support()); assertEquals(1,choice.actualSupport());
        assertTrue(Math.abs(CombatMath.relativeAngle(choice.angle())) <= 1);
        var separated = BreakwaterEnvelope.choose(List.of(new BreakwaterEnvelope.Arc(0,270,10),
                new BreakwaterEnvelope.Arc(1,0,5),new BreakwaterEnvelope.Arc(2,180,5)),0,20);
        assertEquals(1,separated.support()); assertEquals(270,separated.angle(),1e-9);
    }

    @Test void abstentionIsScoredOnTheSameFutureAsAnEligibleBaseline() {
        var rows = new ArrayList<BreakwaterEnvelope.Row>();
        var envelope = new BreakwaterEnvelope(rows::add);
        envelope.startRound(1);
        var plan = envelope.decide(own(10,0),target(10,650,300,8),baseline(0,1.35));
        assertTrue(plan.opportunity()); assertFalse(plan.supported()); assertEquals(0,plan.power());
        for (int i=11;i<=50;i++) envelope.observe(target(i,650,300,0));
        assertEquals(1,rows.size());
        var row=rows.get(0);
        assertEquals("resolved",row.status()); assertTrue(row.baselineEligible()); assertFalse(row.candidateEligible());
        assertEquals(5,row.hits() & 5); assertEquals(11,row.creationTurn());
        assertEquals(11 + (int)Math.floor((550+18)/(20-3*1.35)),row.endTurn());
    }

    @Test void eligibleNeedsPhysicalHeadingAndUnknownsAreNeverMisses() {
        var rows = new ArrayList<BreakwaterEnvelope.Row>();
        var envelope = new BreakwaterEnvelope(rows::add);
        envelope.startRound(1);
        var inside = envelope.decide(own(20,0),target(20,160,300,0),baseline(0,2));
        assertTrue(inside.supported()); assertEquals(2,inside.power());
        envelope.observe(target(22,160,300,0));
        assertEquals("gap",rows.get(0).status()); assertEquals(-1,rows.get(0).hits());
        var outside = envelope.decide(own(22,90),target(22,160,300,0),baseline(0,0));
        assertTrue(outside.supported()); assertEquals(0,outside.power());
        envelope.endRound(22); envelope.endRound(22); envelope.startRound(2);
        assertEquals(2,rows.size());
        assertEquals("terminal",rows.get(1).status()); assertEquals(-1,rows.get(1).hits());
        assertFalse(rows.get(1).candidateEligible()); assertFalse(rows.get(1).baselineEligible());
    }

    @Test void staleAndHotCommandsNeverOpenOpportunities() {
        var rows = new ArrayList<BreakwaterEnvelope.Row>();
        var envelope = new BreakwaterEnvelope(rows::add);
        envelope.startRound(1);
        var hot = new BreakwaterPolicy.Own(10,100,300,0,0,0,.1,100,.1,800,600);
        assertFalse(envelope.decide(hot,target(10,160,300,0),baseline(0,0)).opportunity());
        assertFalse(envelope.decide(own(10,0),target(8,160,300,0),baseline(0,0)).opportunity());
        envelope.endRound(10);
        assertTrue(rows.isEmpty());
    }
}
