import static org.junit.jupiter.api.Assertions.*;

import dev.robocode.tankroyale.botapi.BaseBot;
import dev.robocode.tankroyale.botapi.BotState;
import dev.robocode.tankroyale.botapi.BulletState;
import dev.robocode.tankroyale.botapi.GameSetup;
import dev.robocode.tankroyale.botapi.events.*;
import java.io.PrintStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Source-derived legal observation/callback contracts, not an engine simulation or battle.
 * Both actual candidate callbacks and the frozen original Keel.class callbacks are executed.
 * SDK objects are inert: start(), run(), go(), networking and an engine are never invoked.
 */
class KeelBraceChronologyTest {
    private PrintStream originalOut, originalErr;
    private KeelBrace candidate;
    private Keel baseline;

    @BeforeEach void inertObjectsWithPinnedComparator() throws Exception {
        originalOut = System.out; originalErr = System.err;
        assertNull(System.getenv("KEEL_BRACE_METRICS"), "pure fixture must not write a bot marker path");
        assertNull(System.getenv("KEEL_METRICS"), "pure fixture must not write a baseline marker path");
        byte[] jar = Files.readAllBytes(Path.of(System.getProperty("keel.baseline.jar")));
        assertEquals("77b8a5c3114641c12a94f36c83083d27c419416ee691d725fd19f0284a69db6f",
                java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(jar)));
        // Constructors initialize SDK state only. There is no call to start/run/go or SDK event dispatch.
        candidate = new KeelBrace(); baseline = new Keel();
        Surfer.shadows = false;
        for (BaseBot bot : both()) {
            Object sdk = value(bot, "baseBotInternals");
            field(sdk.getClass(), "gameSetup").set(sdk, new GameSetup("1v1", 800, 600, 1, 0.1, 450, 100000, 100000));
            bot.onRoundStarted(new RoundStartedEvent(1));
        }
    }

    @AfterEach void restoreSdkConstructorOutputWrappers() {
        // The SDK constructor wraps System streams. Do not leak that fixture-only global effect.
        System.setOut(originalOut); System.setErr(originalErr); Surfer.shadows = false;
    }

    private List<BaseBot> both() { return List.of(baseline, candidate); }

    private static Field field(Class<?> type, String name) throws Exception {
        for (Class<?> c = type; c != null; c = c.getSuperclass()) {
            try { Field f = c.getDeclaredField(name); f.setAccessible(true); return f; }
            catch (NoSuchFieldException ignored) { }
        }
        throw new NoSuchFieldException(name);
    }

    private static Object value(Object object, String name) throws Exception {
        return field(object.getClass(), name).get(object);
    }

    private static BotState ownState() throws Exception {
        // Populate the pinned SDK value constructor without starting an SDK thread/server.
        Constructor<?> c = BotState.class.getConstructors()[0];
        assertEquals(22, c.getParameterCount(), "pinned 1.4.0 SDK constructor layout");
        Object[] args = new Object[c.getParameterCount()]; Class<?>[] types = c.getParameterTypes();
        for (int i = 0; i < args.length; i++) {
            if (types[i] == boolean.class) args[i] = Boolean.FALSE;
            else if (types[i] == int.class) args[i] = Integer.valueOf(1);
            else if (types[i] == double.class) args[i] = Double.valueOf(0);
            else args[i] = null;
        }
        args[1] = 100.0; args[2] = 400.0; args[3] = 300.0;
        return (BotState)c.newInstance(args);
    }

    private static void scan(BaseBot bot, int turn, double enemyEnergy) throws Exception {
        scan(bot, turn, enemyEnergy, 436, 300, 180, 0);
    }

    private static void scan(BaseBot bot, int turn, double energy, double x, double y, double heading, double speed) throws Exception {
        Object sdk = value(bot, "baseBotInternals");
        field(sdk.getClass(), "tickEvent").set(sdk, new TickEvent(turn, (Integer)value(bot, "round"), ownState(), List.of(), List.of()));
        // Own observations that the normal run loop would already have recorded; no future pose is inserted.
        Track own = (Track)value(bot, "me");
        if (turn > 0) own.add(turn - 1, new Phys.Pose(400, 300, 0, 0), x, y);
        bot.onScannedBot(new ScannedBotEvent(turn, 1, 2, energy, x, y, heading, speed));
    }

    private void scanBoth(int turn, double energy) throws Exception { for (BaseBot bot : both()) scan(bot, turn, energy); }
    private static void ram(BaseBot bot, int turn, double enemyEnergy, boolean isRammed) {
        bot.onHitBot(new HitBotEvent(turn, 2, enemyEnergy, 436, 300, isRammed));
    }
    private void ramBoth(int turn, double energy, boolean attribution) { for (BaseBot bot : both()) ram(bot, turn, energy, attribution); }
    private static List<Surfer.Wave> waves(BaseBot bot) throws Exception { return ((Surfer)value(bot, "surfer")).waves; }
    private RamLedger ledger() throws Exception { return (RamLedger)value(candidate, "ram"); }
    private static void assertWave(Surfer.Wave wave, int turn, double power) {
        assertEquals(turn, wave.t); assertEquals(power, wave.power, 1e-10);
        assertEquals(436, wave.ox, 1e-12); assertEquals(300, wave.oy, 1e-12);
        assertEquals(Track.DIMS, wave.features.length);
    }

    @Test void ramAloneSuppressesOriginalPhantomAndCountsEligibleBoundary() throws Exception {
        scanBoth(1, 100); ramBoth(2, 99.4, false); scanBoth(2, 99.4); scanBoth(3, 99.4);
        assertTrue(waves(candidate).isEmpty());
        assertEquals(1, waves(baseline).size()); assertWave(waves(baseline).get(0), 2, 0.6);
        assertEquals(1, ledger().rawRamEvents); assertEquals(1, ledger().eligibleRamCorrectedScans);
        assertEquals(1, ledger().eligibleRamCorrectionEvents); assertEquals(1, ledger().ramChangedPowerScans);
        assertEquals(1, ledger().ramChangedGateScans, "power0 is still eligible correction exposure");
    }

    @Test void ramAndMinimumShotPreserveActualPower() throws Exception { ramPlusShot(0.1); }
    @Test void ramAndMediumShotPreserveActualPower() throws Exception { ramPlusShot(1.2); }
    @Test void ramAndMaximumShotRecoverOriginalOutOfRangeWave() throws Exception { ramPlusShot(3.0); }
    private void ramPlusShot(double power) throws Exception {
        scanBoth(1, 100); ramBoth(2, 100 - power - 0.6, true); scanBoth(2, 100 - power - 0.6);
        assertEquals(1, waves(candidate).size()); assertWave(waves(candidate).get(0), 2, power);
        assertEquals(1, ledger().eligibleRamCorrectedScans);
        if (power == 3) { assertTrue(waves(baseline).isEmpty()); assertEquals(1, ledger().ramChangedGateScans); }
        else { assertEquals(1, waves(baseline).size()); assertWave(waves(baseline).get(0), 2, power + 0.6); assertEquals(0, ledger().ramChangedGateScans); }
    }

    @Test void followingRealPointSixShotIsNotCanceledByPreviousRam() throws Exception {
        scanBoth(1, 100); ramBoth(2, 99.4, false); scanBoth(2, 99.4); scanBoth(3, 98.8);
        assertEquals(1, waves(candidate).size()); assertWave(waves(candidate).get(0), 3, 0.6);
        assertEquals(1, waves(baseline).size()); assertWave(waves(baseline).get(0), 2, 0.6);
        assertEquals(1, ledger().eligibleRamCorrectedScans);
    }

    @Test void bothRammingAttributionsHaveSameCorrection() throws Exception {
        scanBoth(1, 100); ram(candidate, 2, 99.4, true); ram(baseline, 2, 99.4, false); scanBoth(2, 99.4);
        assertTrue(waves(candidate).isEmpty()); assertEquals(1, waves(baseline).size());
        // Candidate false-attribution is independently exercised by ramAloneSuppressesOriginalPhantom.
        assertEquals(1, ledger().eligibleRamCorrectionEvents);
    }

    private static void legalPostScanBulletEvents(BaseBot bot, int turn) {
        // Engine bullet effects occur after scanT, but SDK higher-priority callbacks can precede ScannedBotT.
        bot.onBulletHit(new BulletHitBotEvent(turn, 2, new BulletState(11, 1, 1, 436, 300, 0, null), 4, 96));
        bot.onHitByBullet(new HitByBulletEvent(turn, new BulletState(12, 2, 1, 400, 300, 180, null), 4, 96));
    }

    @Test void postScanBulletDamageAndBonusStillBelongToNextScan() throws Exception {
        scanBoth(1, 100);
        for (BaseBot bot : both()) legalPostScanBulletEvents(bot, 2);
        scanBoth(2, 100); // scan snapshot precedes those engine bullet effects
        scanBoth(3, 99); // −4 damage to enemy +3 enemy hit bonus, no enemy fire
        assertTrue(waves(candidate).isEmpty()); assertTrue(waves(baseline).isEmpty());
        assertEquals(0, ledger().rawRamEvents); assertEquals(0, ledger().eligibleRamCorrectedScans);
    }

    @Test void ramAndPostScanBulletEffectsStayOnDifferentEpochs() throws Exception {
        scanBoth(1, 100);
        for (BaseBot bot : both()) legalPostScanBulletEvents(bot, 2);
        ramBoth(2, 99.4, true); scanBoth(2, 99.4); scanBoth(3, 98.4);
        assertTrue(waves(candidate).isEmpty()); assertEquals(1, waves(baseline).size());
        assertWave(waves(baseline).get(0), 2, 0.6);
        assertEquals(1, ledger().eligibleRamCorrectedScans);
    }

    @Test void multipleLegitimateBulletCorrectionsAndOneRamSumOnce() throws Exception {
        scanBoth(1, 100);
        for (BaseBot bot : both()) {
            legalPostScanBulletEvents(bot, 2); // net enemy loss1
            bot.onBulletHit(new BulletHitBotEvent(2, 2, new BulletState(13, 1, 0.5, 436, 300, 0, null), 2, 94));
        }
        ramBoth(2, 99.4, false); scanBoth(2, 99.4); scanBoth(3, 96.4); // total post-scan enemy loss3
        assertTrue(waves(candidate).isEmpty()); assertEquals(1, waves(baseline).size());
        assertEquals(1, ledger().rawRamEvents, "no duplicated collision event invented");
        assertEquals(1, ledger().eligibleRamCorrectionEvents);
    }

    @Test void firstScanRamIsDiscardedWithoutDelayedCorrection() throws Exception {
        ramBoth(1, 99.4, false); scanBoth(1, 99.4); scanBoth(2, 98.8);
        assertEquals(1, waves(candidate).size()); assertWave(waves(candidate).get(0), 2, 0.6);
        assertTrue(waves(baseline).isEmpty());
        assertEquals(1, ledger().rawRamEvents); assertEquals(0, ledger().eligibleRamCorrectedScans);
    }

    @Test void missingScanDiscardsItsRamAndDoesNotLeakToLaterInference() throws Exception {
        scanBoth(1, 100); ramBoth(2, 99.4, false); scanBoth(3, 99.4); scanBoth(4, 98.8);
        assertEquals(1, waves(candidate).size()); assertWave(waves(candidate).get(0), 4, 0.6);
        assertEquals(0, ledger().eligibleRamCorrectedScans); assertEquals(1, ledger().rawRamEvents);
    }

    @Test void duplicateAndStaleScansCannotConsumeOrCountAgain() throws Exception {
        scanBoth(1, 100); ramBoth(2, 99.4, false); scanBoth(2, 99.4);
        // SDK stale/duplicate-delivery safety: callbacks see current tick, not a restarted engine turn.
        candidate.onScannedBot(new ScannedBotEvent(2, 1, 2, 99.4, 436, 300, 180, 0));
        candidate.onScannedBot(new ScannedBotEvent(1, 1, 2, 100, 436, 300, 180, 0));
        assertTrue(waves(candidate).isEmpty()); assertEquals(1, ledger().eligibleRamCorrectedScans);
        assertEquals(2, value(candidate, "scanT")); assertEquals(2, ((Track)value(candidate, "enemy")).latest());
    }

    @Test void lateRamCannotRetroactivelyChangeAcceptedScanOrCancelNextShot() throws Exception {
        scanBoth(1, 100); scanBoth(2, 99.4); // collision callback deliberately late: safety case, not default ordering
        ramBoth(2, 99.4, false); scanBoth(3, 98.8);
        assertEquals(2, waves(candidate).size()); assertWave(waves(candidate).get(1), 3, 0.6);
        assertEquals(1, waves(baseline).size());
        assertEquals(1, ledger().rawRamEvents); assertEquals(0, ledger().eligibleRamCorrectedScans);
    }

    @Test void staleRamIsCountedButNeverAppliedToFutureScan() throws Exception {
        scanBoth(1, 100); scanBoth(2, 100); ramBoth(1, 99.4, false); scanBoth(3, 99.4);
        assertEquals(1, waves(candidate).size()); assertWave(waves(candidate).get(0), 3, 0.6);
        assertEquals(1, ledger().rawRamEvents); assertEquals(0, ledger().eligibleRamCorrectedScans);
    }

    @Test void roundResetClearsBucketCountersAndOriginalBulletMap() throws Exception {
        scanBoth(1, 100); ramBoth(2, 99.4, false);
        for (BaseBot bot : both()) { legalPostScanBulletEvents(bot, 2); bot.onRoundStarted(new RoundStartedEvent(2)); }
        assertEquals(0, ledger().rawRamEvents); assertEquals(0, ledger().eligibleRamCorrectedScans);
        scanBoth(1, 100); scanBoth(2, 99.4);
        assertEquals(1, waves(candidate).size()); assertWave(waves(candidate).get(0), 2, 0.6);
        assertEquals(1, waves(baseline).size()); assertWave(waves(baseline).get(0), 2, 0.6);
    }

    @Test void wallHeuristicAndNoRamWaveFeaturesRemainOriginal() throws Exception {
        scanBoth(1, 100); scanBoth(2, 98.8);
        Surfer.Wave c = waves(candidate).get(0), b = waves(baseline).get(0);
        assertWave(c, 2, 1.2); assertWave(b, 2, 1.2);
        assertArrayEquals(b.features, c.features, 0); assertArrayEquals(b.density, c.density, 0);
        assertEquals(b.direct, c.direct, 0); assertEquals(b.latDir, c.latDir);
        for (BaseBot bot : both()) bot.onRoundStarted(new RoundStartedEvent(2));
        for (BaseBot bot : both()) { scan(bot, 1, 100, 781, 300, 0, 8); scan(bot, 2, 96.5, 782, 300, 0, 0); }
        assertTrue(waves(candidate).isEmpty()); assertTrue(waves(baseline).isEmpty());
    }

    @Test void eligibleCounterPrecedesMissingOwnHistoryWaveSubgate() throws Exception {
        scan(candidate, 1, 100); ((Track)value(candidate, "me")).clear();
        ram(candidate, 2, 98.2, false); scan(candidate, 2, 98.2);
        assertTrue(waves(candidate).isEmpty()); assertEquals(1, ledger().eligibleRamCorrectedScans);
        assertEquals(1, ledger().ramChangedPowerScans); assertEquals(0, ledger().ramChangedGateScans);
    }

    @Test void newerRamTurnReplacesAnUnconsumedOlderBucket() {
        RamLedger l = new RamLedger(() -> fail("unexpected counter overflow"));
        l.hit(2, 1); l.hit(3, 1);
        RamLedger.Snapshot snapshot = l.take(3);
        assertEquals(-0.6, snapshot.correction(), 0); assertEquals(1, snapshot.events());
        assertEquals(2, l.rawRamEvents); assertEquals(0, l.take(4).events());
        assertEquals(0, l.eligibleRamCorrectedScans, "taking a bucket is not an eligible inference");
    }

    @Test void checkedCounterOverflowMarksOnceAndNeverControlsPhysicalCorrection() {
        AtomicInteger faults = new AtomicInteger(); RamLedger l = new RamLedger(faults::incrementAndGet);
        l.rawRamEvents = RamLedger.COUNTER_CAP;
        l.hit(2, 1); RamLedger.Snapshot snapshot = l.take(2);
        assertEquals(-0.6, snapshot.correction(), 0); assertEquals(1, snapshot.events());
        assertEquals(RamLedger.COUNTER_CAP, l.rawRamEvents); assertEquals(1, faults.get());
        l.hit(3, 2); assertEquals(1, faults.get(), "one fault marker per round");
        l.newRound(); assertEquals(0, l.rawRamEvents); l.rawRamEvents = RamLedger.COUNTER_CAP;
        l.hit(4, 3); assertEquals(2, faults.get(), "reset permits the next round's one fault marker");
    }

    @Test void everyAggregateIsCheckedAndBounded() {
        AtomicInteger faults = new AtomicInteger(); RamLedger l = new RamLedger(faults::incrementAndGet);
        l.rawRamEvents = l.eligibleRamCorrectedScans = l.eligibleRamCorrectionEvents =
                l.ramChangedPowerScans = l.ramChangedGateScans = RamLedger.COUNTER_CAP;
        l.inferred(new RamLedger.Snapshot(-0.6, 1), 0.6, 0);
        assertEquals(1, faults.get());
        assertEquals(RamLedger.COUNTER_CAP, l.eligibleRamCorrectedScans);
        assertEquals(RamLedger.COUNTER_CAP, l.eligibleRamCorrectionEvents);
        assertEquals(RamLedger.COUNTER_CAP, l.ramChangedPowerScans);
        assertEquals(RamLedger.COUNTER_CAP, l.ramChangedGateScans);
    }
}
