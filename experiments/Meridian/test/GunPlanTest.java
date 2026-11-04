import dev.robocode.tankroyale.botapi.events.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class GunPlanTest {
    GunHarness planned() throws Exception {
        var h=new GunHarness(false);h.heading=180;h.scan(10,h.target,100);h.decide(10);
        assertNotNull(h.plan());assertEquals(0,h.requests);return h;
    }
    @Test void holdsSignedOffsetAndPowerButTracksLiveBearingAcrossReversalAndWrap() throws Exception {
        var h=planned();var plan=h.plan();
        h.own=new Flight.Pose(700,301,90,8);
        h.scan(11,new Flight.Pose(300,299,270,1),1);
        h.decide(11);
        double bearing=Flight.angle(h.target.x()-h.own.x(),h.target.y()-h.own.y());
        assertSame(plan,h.plan());
        assertEquals(Flight.delta(bearing+plan.offset()-h.heading),h.rate,1e-12);
        assertEquals(1.65,plan.power());
        h.heading=Flight.delta(bearing+plan.offset())+360;
        h.decide(11);assertEquals(1,h.requests);assertEquals(plan.power(),h.power);
    }
    @Test void launchUsesActualFiringDecisionNotPlanningOrCallbackScan() throws Exception {
        var h=planned();var plan=h.plan();
        h.own=new Flight.Pose(120,310,130,7);
        h.scan(11,new Flight.Pose(480,330,270,-4),30);
        double b=Flight.angle(h.target.x()-h.own.x(),h.target.y()-h.own.y());
        h.heading=b+plan.offset();h.decide(11);
        var expected=CrossingGun.launch(12,h.own,h.target,plan.power(),800,600);
        assertEquals(expected,h.requested());
        h.scan(12,new Flight.Pose(470,335,20,2),30);
        h.confirm(12);
        assertNull(h.plan());assertEquals(-1,h.field("pendingFireTurn"));
        var launches=(java.util.List<?>)GunHarness.field(h.field("crossingGun"),"inFlight");
        assertEquals(java.util.List.of(expected),launches);
    }
    @Test void acceptedRequestSuppressesDuplicatesUntilConfirmationOrExpiredTurn() throws Exception {
        var h=planned();h.heading=h.rate+h.heading;h.decide(10);
        assertEquals(1,h.requests);var request=h.requested();
        h.decide(10);assertEquals(0,h.requests);assertEquals(request,h.requested());
        h.scan(11,h.target,100);h.decide(11);assertEquals(0,h.requests);
        h.scan(12,h.target,100);h.decide(12);assertEquals(1,h.requests);
        assertEquals(13,h.requested().turn());
    }
    @Test void rejectedSetFireKeepsPlanAndDoesNotTeachOrEnterPending() throws Exception {
        var h=planned();h.heading+=h.rate;h.accepted=false;h.decide(10);
        assertEquals(1,h.requests);assertNotNull(h.plan());assertNull(h.requested());
        assertEquals(-1,h.field("pendingFireTurn"));h.accepted=true;h.decide(10);
        assertEquals(1,h.requests);assertNotNull(h.requested());
    }
    @Test void scanLossClearsPlanAndReacquisitionSelectsAgain() throws Exception {
        var h=planned();var old=h.plan();h.decide(11);assertNull(h.plan());assertEquals(0,h.requests);
        h.scan(12,new Flight.Pose(500,300,270,8),100);h.decide(12);
        assertNotNull(h.plan());assertNotSame(old,h.plan());assertTrue(h.plan().offset()<0);
    }
    @Test void hotGunDoesNotCreatePlanAndRoundChangeClearsPending() throws Exception {
        var h=new GunHarness(false);h.heat=.1;h.scan(10,h.target,100);h.decide(10);
        assertNull(h.plan());assertEquals(0,h.requests);
        h.heat=0;h.decide(10);h.heading+=h.rate;h.decide(10);assertEquals(11,h.field("pendingFireTurn"));
        h.bot.onRoundStarted(new RoundStartedEvent(2));
        assertNull(h.plan());assertNull(h.requested());assertNull(h.field("scan"));
        assertEquals(-1,h.field("pendingFireTurn"));
    }
    @Test void targetDisappearanceClearsScanAndAimButNotAcceptedShotContext() throws Exception {
        var h=planned();h.heading+=h.rate;h.decide(10);var requested=h.requested();
        h.bot.onBotDeath(new BotDeathEvent(11,99));assertNotNull(h.plan());
        h.bot.onBotDeath(new BotDeathEvent(11,2));assertNull(h.plan());assertNull(h.field("scan"));
        assertEquals(requested,h.requested());h.confirm(11);
        assertEquals(java.util.List.of(requested),GunHarness.field(h.field("crossingGun"),"inFlight"));
    }
    @Test void unaffordablePlanIsDroppedIncludingZeroAndEqualPowerEnergy() throws Exception {
        for(double energy:new double[]{0,.05,1.65}) {
            var h=planned();var old=h.plan();h.energy=energy;h.heading=180;h.decide(10);
            assertNotSame(old,h.plan());assertEquals(0,h.requests);
            if(energy<=.1)assertNull(h.plan());else assertTrue(h.plan().power()<energy);
        }
    }
    @Test void originalSafetyGuardsRemainBinding() throws Exception {
        var h=planned();h.heading+=h.rate;
        h.heat=.1;h.decide(10);assertEquals(0,h.requests);
        h.heat=0;h.decide(11);assertEquals(0,h.requests); // stale
        h.scan(12,h.target,100);h.energy=1.2;h.heading=0;h.decide(12);
        h.heading+=h.rate;h.decide(12);assertEquals(0,h.requests); // reserve
        h.scan(13,h.target,.1);h.decide(13);
        assertEquals(1,h.requests); // original finishing exception
        var aligned=planned();double aim=aligned.heading+aligned.rate;
        double tolerance=Math.toDegrees(Math.atan2(12,400));
        aligned.heading=aim+tolerance+1e-6;aligned.decide(10);assertEquals(0,aligned.requests);
        aligned.heading=aim+tolerance-1e-6;aligned.decide(10);assertEquals(1,aligned.requests);
    }
    @Test void affordabilityAndScanLossDoNotDiscardPendingLearningBeforeConfirmation() throws Exception {
        var h=planned();h.heading+=h.rate;h.decide(10);var requested=h.requested();
        h.energy=0;h.decide(11);assertNull(h.plan());assertEquals(0,h.requests);
        assertEquals(requested,h.requested());h.confirm(11);
        assertEquals(java.util.List.of(requested),GunHarness.field(h.field("crossingGun"),"inFlight"));
    }
    @Test void lateOrDuplicateConfirmationCannotConsumeANewerPendingRequest() throws Exception {
        var h=planned();h.heading+=h.rate;h.decide(10);
        h.scan(12,h.target,100);h.decide(12);var current=h.requested();
        assertEquals(13,current.turn());h.confirm(11);
        assertEquals(current,h.requested());assertEquals(13,h.field("pendingFireTurn"));
        h.confirm(13);h.confirm(13);
        assertEquals(java.util.List.of(current),GunHarness.field(h.field("crossingGun"),"inFlight"));
    }
    @Test void runLoopClearsPlanForAbsentOrLongStaleScanAndRoundEndClearsIt() throws Exception {
        for(boolean absent:new boolean[]{true,false}) {
            var h=planned();
            if(absent)h.set("scan",null);
            h.turn=14;
            org.mockito.Mockito.doReturn(true,false).when(h.bot).isRunning();
            org.mockito.Mockito.doNothing().when(h.bot).go();
            // Command setters only build an intent; no start/connect or actual go is allowed.
            h.bot.run();assertNull(h.plan());
        }
        var h=planned();h.heading+=h.rate;h.decide(10);
        h.bot.onRoundEnded(new RoundEndedEvent(1,10,null));
        assertNull(h.plan());assertEquals(-1,h.field("pendingFireTurn"));
    }
}
