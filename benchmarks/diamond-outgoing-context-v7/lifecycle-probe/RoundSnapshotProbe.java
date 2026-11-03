import java.util.Arrays;

/** Method-order reachability probe only: it does not establish actual engine callback ordering. */
final class RoundSnapshotProbe {
    private static DuelGun.Scan scan(int turn){return new DuelGun.Scan(2,turn,500,300,90,0,100);}
    public static void main(String[] args) {
        var gun=new PressureGun();var context=gun.context();
        // Populate legal confirmed/fresh passage history without direct state mutation or fake score injection.
        for(int i=0;i<32;i++) {
            int t=100+i*100;context.onScan(scan(t-1));context.onScan(scan(t));
            var query=context.query(t,scan(t),scan(t-1),300,300,1.4,0,200,800,600);
            context.confirmed(query,t+1,300,300,1.4);
            for(int age=1;age<=30;age++)context.onScan(scan(t+age));
        }
        gun.onScan(scan(3999));gun.onScan(scan(4000));
        var plan=gun.decide(4000,300,300,0,100,0,800,600);
        if(!plan.fire() || gun.supportedDecisions()!=1)throw new AssertionError("unsupported fixture");
        gun.onFired(4001,7,300,300,plan.power(),0);
        for(int turn=4001;turn<=4010;turn++)gun.onScan(scan(turn));
        int firstRowCompleted=gun.pairedCompletedRound();
        System.out.println("round1_snapshot completed="+firstRowCompleted+" persistent="
                +Arrays.toString(gun.pairedAttempts())+" pending="+gun.pairedPending());
        if(firstRowCompleted!=0 || gun.pairedPending()!=1)throw new AssertionError("cohort already resolved");
        // Hypothetical callback after the snapshot: source contains no closed-round guard in onScan/score.
        gun.onScan(scan(4011));
        System.out.println("late_scan4011 completed="+gun.pairedCompletedRound()+" persistent="
                +Arrays.toString(gun.pairedAttempts())+" pending="+gun.pairedPending());
        if(gun.pairedCompletedRound()!=1 || gun.pairedAttempts()[0]!=1)throw new AssertionError("not resolved");
        gun.startRound();
        int secondRowCompleted=gun.pairedCompletedRound(), endpoint=gun.pairedAttempts()[0];
        System.out.println("round2_snapshot completed="+secondRowCompleted+" persistent="
                +Arrays.toString(gun.pairedAttempts())+" pending="+gun.pairedPending());
        System.out.println("sum_snapshot_rows="+(firstRowCompleted+secondRowCompleted)
                +" endpoint="+endpoint+" difference="+(endpoint-firstRowCompleted-secondRowCompleted));
        if(endpoint!=1 || firstRowCompleted+secondRowCompleted!=0)throw new AssertionError("mismatch not reproduced");
        System.out.println("REACHABLE_METHOD_ORDER_ONLY; ACTUAL_PR65_CALLBACK_CAUSE_UNPROVEN");
    }
}
