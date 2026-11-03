import java.util.Collection;

/** Game-local distribution of aim directions on bullets that actually hit us. */
final class HitAngles {
    record Wave(int t, double x, double y, double center, double lead, double power) {
        double speed() { return 20 - 3 * power; }
        double escape() { return Math.toDegrees(Math.asin(8 / speed())); }
        double orientation() { return Math.signum(lead - center); }

        boolean fits(int turn, double bulletX, double bulletY, double heading, double observedPower) {
            if (turn < t || Math.abs(power - observedPower) > 1e-5) return false;
            // The hit event carries the advanced bullet position, at the start of its collision sweep.
            double distance = (turn - t + 1) * speed();
            return Math.hypot(bulletX - x - distance * Flight.cos(heading),
                              bulletY - y - distance * Flight.sin(heading)) <= 1;
        }
    }

    enum Label { ACCEPTED, UNMATCHED, AMBIGUOUS, UNORIENTED, OUTSIDE }
    private static final int MID = 10;
    private static final double LIMIT = 1.25;
    private final double[] weights = new double[2 * MID + 1];
    private double total;

    Label observe(Collection<Wave> waves, int turn, double x, double y, double heading, double power) {
        Wave match = null;
        for (Wave wave : waves) {
            if (!wave.fits(turn, x, y, heading, power)) continue;
            if (match != null) return Label.AMBIGUOUS;
            match = wave;
        }
        if (match == null) return Label.UNMATCHED;
        if (Math.abs(match.lead - match.center) < 1) return Label.UNORIENTED;
        double offset = Flight.delta(heading - match.center) * match.orientation() / match.escape();
        if (Math.abs(offset) > LIMIT) return Label.OUTSIDE;
        int bin = MID + (int) Math.round(offset * MID / LIMIT);
        for (int i = 0; i < weights.length; i++) weights[i] *= .95;
        weights[bin]++;
        total = .95 * total + 1;
        return Label.ACCEPTED;
    }

    private static double rayRisk(double dx, double dy, double heading) {
        double miss = dx * Flight.sin(heading) - dy * Flight.cos(heading);
        return 40 / (1 + miss * miss / 500);
    }

    double risk(Wave wave, double dx, double dy) {
        double prior = rayRisk(dx, dy, wave.center) + rayRisk(dx, dy, wave.lead);
        if (total == 0 || Math.abs(wave.lead - wave.center) < 1) return prior;
        double measured = 0;
        for (int i = 0; i < weights.length; i++) {
            if (weights[i] == 0) continue;
            double offset = (i - MID) * LIMIT / MID;
            double heading = wave.center + offset * wave.orientation() * wave.escape();
            measured += weights[i] * rayRisk(dx, dy, heading);
        }
        // Keep v1's two-ray total scale so switching and wall costs retain their meaning.
        return (4 * prior + 2 * measured) / (4 + total);
    }
}
