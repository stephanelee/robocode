import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ChallengerWavesTest {
    static DuelGun.Scan scan(int turn,double energy) {return new DuelGun.Scan(2,turn,100,300,0,0,energy);}
    static ChallengerWaves prepared() {
        var w=new ChallengerWaves();w.startRound();
        w.own(new ChallengerWaves.Own(4,142,300,90,8));
        w.own(new ChallengerWaves.Own(5,500,400,0,0));
        w.scan(scan(4,100));return w;
    }
    @Test void creationHitMatchesAfterInferenceRegardlessOfDeliveryOrderAndUsesAimContext() {
        for(boolean hitFirst:new boolean[]{true,false}) {
            var w=prepared();var hit=new ChallengerWaves.Hit(5,2,1,117,300,0);
            if(hitFirst)w.hit(hit);w.scan(scan(5,99));if(!hitFirst)w.hit(hit);
            w.flush(5,800,600);
            assertEquals(1,w.inferred());assertEquals(1,w.matched());assertEquals(0,w.unmatched());
            // Own aim-time lateral speed8 selects near/fast context1; impact-time state would select another context.
            assertEquals(1,java.util.Arrays.stream(w.danger()[1]).sum());
            assertEquals(0,java.util.Arrays.stream(w.danger()[0]).sum());
        }
    }
    @Test void delayedDamageAndBonusDoNotManufactureShotsInEitherCallbackOrder() {
        for(boolean eventFirst:new boolean[]{true,false}) {
            var w=prepared();
            if(eventFirst){w.damage(2,5,2);w.bonus(2,5,.5);}
            w.scan(scan(5,100));
            if(!eventFirst){w.damage(2,5,2);w.bonus(2,5,.5);}
            w.own(new ChallengerWaves.Own(6,500,400,0,0));w.scan(scan(6,99.5));
            w.flush(6,800,600);assertEquals(0,w.inferred());
        }
    }
    @Test void gapsMissingAimHistoryRamAndNearWallAreRejected() {
        var gap=prepared();gap.scan(scan(6,99));gap.flush(6,800,600);assertEquals(0,gap.inferred());
        var absent=new ChallengerWaves();absent.scan(scan(4,100));absent.scan(scan(5,99));
        absent.flush(5,800,600);assertEquals(0,absent.inferred());
        var ram=prepared();ram.scan(scan(5,99));ram.collision(5);ram.flush(5,800,600);assertEquals(0,ram.inferred());
        var wall=prepared();wall.scan(new DuelGun.Scan(2,5,20,300,0,0,99));wall.flush(5,800,600);
        assertEquals(0,wall.inferred());
    }
    @Test void profileIsDefensiveRoundDecayedAndInvalidHitCannotPoisonIt() {
        var w=prepared();w.scan(scan(5,99));w.hit(new ChallengerWaves.Hit(5,2,1,117,300,0));w.flush(5,800,600);
        double[][] copy=w.danger();copy[1][23]=99;
        assertEquals(1,java.util.Arrays.stream(w.danger()[1]).sum());
        w.startRound();assertEquals(.95,java.util.Arrays.stream(w.danger()[1]).sum(),1e-15);
        assertTrue(w.threats().isEmpty());assertEquals(0,w.matched());assertEquals(1,w.totalMatched());
        w.hit(new ChallengerWaves.Hit(1,2,Double.NaN,0,0,0));assertEquals(1,w.unmatched());
    }
}
