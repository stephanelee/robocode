import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MeleeOutcomePowerTest {
    private int id;
    private void sample(MeleeOutcomePower p,int target,double range,boolean hit) {
        int t=++id*4;
        var c=p.choice(target,range,100,100,0,true);p.pending(t,c);p.confirm(t+1,id,c.power());
        p.terminal(t+2,id,hit?1:2,hit?4:0);p.advance(t+3);
    }
    @Test void calibrationRequiresThirtyTwoRealResolvedOutcomesAndTenSuccesses() {
        var p=new MeleeOutcomePower();p.newRound();
        for(int i=0;i<31;i++)sample(p,2,300,true);
        assertEquals(1.6,p.choice(2,300,100,100,0,true).power());
        sample(p,2,300,true);assertEquals(2,p.choice(2,300,100,100,0,true).power());
        var q=new MeleeOutcomePower();q.newRound();
        for(int i=0;i<32;i++)sample(q,2,300,i>=23);
        assertEquals(1.6,q.choice(2,300,100,100,0,true).power());
        sample(q,2,300,true);assertEquals(2,q.choice(2,300,100,100,0,true).power());
    }
    @Test void opponentAndRangeBinsStaySeparateAndWindowsPersistAcrossRounds() {
        var p=new MeleeOutcomePower();p.newRound();for(int i=0;i<32;i++)sample(p,2,200,true);
        assertEquals(2.6,p.choice(2,200,100,100,0,true).power());
        assertEquals(1.6,p.choice(2,300,100,100,0,true).power());
        assertEquals(2.2,p.choice(3,200,100,100,0,true).power());
        p.finishRound();p.newRound();assertEquals(2.6,p.choice(2,200,100,100,0,true).power());
        assertEquals(2.2,p.choice(2,200,100,100,0,false).power());
    }
    @Test void collisionAndHitInSameTickResolveOnceAsSuccessOnlyAfterTickCompletes() {
        var p=new MeleeOutcomePower();p.newRound();var c=p.choice(2,300,100,100,0,true);
        p.pending(1,c);p.confirm(2,5,c.power());p.terminal(10,5,4,0);p.terminal(10,5,1,3);
        p.advance(10);assertTrue(p.stats().contains("outcomeResolved=0"));
        p.advance(11);assertTrue(p.stats().contains("outcomeResolved=1 outcomeSuccesses=1"));
        assertTrue(p.trace().contains(",5,3.0,1,0"));
        assertEquals(1,p.choice(2,300,100,100,0,true).n());
    }
    @Test void roundEndDoesNotTeachCutoffNonHitsAndTransitionRetainsConfirmedLedger() {
        var p=new MeleeOutcomePower();p.newRound();var c=p.choice(2,300,100,100,0,true);
        p.pending(1,c);p.confirm(2,5,c.power());p.terminal(10,5,4,0);p.clearPending();p.finishRound();
        assertTrue(p.stats().contains("outcomeResolved=0 outcomeSuccesses=0 outcomeCensored=1"));
        p.newRound();assertEquals(0,p.choice(2,300,100,100,0,true).n());
        p.pending(1,c);p.confirm(2,6,c.power());p.clearPending();p.terminal(3,6,1,4);p.finishRound();
        assertTrue(p.stats().contains("outcomeResolved=1 outcomeSuccesses=1 outcomeCensored=0"));
        p.newRound();assertEquals(1,p.choice(2,300,100,100,0,true).n());
    }
    @Test void reservesStaleScansKillCapAndFarRangePreventBoost() {
        var p=new MeleeOutcomePower();p.newRound();for(int i=0;i<32;i++)sample(p,2,200,true);
        for(double own:new double[]{44.9,20,5})assertEquals(MeleePolicy.power(200,own,100),p.choice(2,200,own,100,0,true).power());
        assertEquals(2.2,p.choice(2,200,100,100,5,true).power());
        assertEquals(MeleePolicy.power(200,100,8),p.choice(2,200,100,8,0,true).power());
        assertEquals(1,p.choice(2,600,100,100,0,true).power());
        assertEquals(2.6,p.choice(2,200,45,100,4,true).power());
    }
    @Test void unmatchedConfirmationsNeverTeachOrExposeAndEvictionIsCensored() {
        var p=new MeleeOutcomePower();p.newRound();var c=p.choice(2,300,100,100,0,true);
        p.pending(1,c);p.confirm(3,1,c.power());p.terminal(4,1,1,4);p.advance(5);
        assertTrue(p.stats().contains("outcomeConfirmed=0 outcomeResolved=0"));
        for(int i=0;i<260;i++){p.pending(i*2,c);p.confirm(i*2+1,i,c.power());}
        assertTrue(p.stats().contains("outcomeCensored=260"));assertTrue(p.trace().contains(",E,"));
        assertEquals(0,p.choice(2,300,100,100,0,true).n());
    }
}
