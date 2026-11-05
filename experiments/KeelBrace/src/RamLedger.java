/**
 * Constant-space same-turn ram correction and protocol-fixed, diagnostic-only round counters.
 * Bullet damage/bonus corrections remain in KeelBrace's original next-scan map.
 */
final class RamLedger {
    static final long COUNTER_CAP = 1_000_000L;
    record Snapshot(double correction, long events) { }
    private static final Snapshot EMPTY = new Snapshot(0, 0);
    private final Runnable overflowMarker;
    private int ramTurn = -1;
    private long ramEvents;
    private double ramCorrection;
    private boolean overflowed;

    long rawRamEvents;
    long eligibleRamCorrectedScans;
    long eligibleRamCorrectionEvents;
    long ramChangedPowerScans;
    long ramChangedGateScans;

    RamLedger(Runnable overflowMarker) { this.overflowMarker = overflowMarker; }

    void newRound() {
        ramTurn = -1; ramEvents = 0; ramCorrection = 0; overflowed = false;
        rawRamEvents = 0; eligibleRamCorrectedScans = 0; eligibleRamCorrectionEvents = 0;
        ramChangedPowerScans = 0; ramChangedGateScans = 0;
    }

    void hit(int turn, int lastAcceptedScan) {
        rawRamEvents = count(rawRamEvents, 1);
        if (turn <= lastAcceptedScan) return;
        if (turn > ramTurn) { ramTurn = turn; ramEvents = 0; ramCorrection = 0; }
        if (turn == ramTurn) {
            // This physical correction does not depend on diagnostic counter values.
            ramEvents = Math.addExact(ramEvents, 1);
            ramCorrection += -0.6;
        }
    }

    Snapshot take(int scanTurn) {
        Snapshot snapshot = ramTurn == scanTurn ? new Snapshot(ramCorrection, ramEvents) : EMPTY;
        if (ramTurn <= scanTurn) { ramTurn = -1; ramEvents = 0; ramCorrection = 0; }
        return snapshot;
    }

    /** Called only inside the existing consecutive-enemy-scan power-computation branch. */
    void inferred(Snapshot snapshot, double withoutRamPower, double correctedPower) {
        if (snapshot.events() == 0) return;
        eligibleRamCorrectedScans = count(eligibleRamCorrectedScans, 1);
        eligibleRamCorrectionEvents = count(eligibleRamCorrectionEvents, snapshot.events());
        if (Double.compare(withoutRamPower, correctedPower) != 0)
            ramChangedPowerScans = count(ramChangedPowerScans, 1);
        if (legalPower(withoutRamPower) != legalPower(correctedPower))
            ramChangedGateScans = count(ramChangedGateScans, 1);
    }

    private static boolean legalPower(double power) { return power >= 0.0999 && power <= 3.0001; }

    private long count(long value, long amount) {
        if (amount < 0 || value < 0 || value > COUNTER_CAP || amount > COUNTER_CAP - value) {
            if (!overflowed) { overflowed = true; overflowMarker.run(); }
            return COUNTER_CAP;
        }
        return value + amount;
    }
}
