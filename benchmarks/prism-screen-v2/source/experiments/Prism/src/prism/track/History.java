package prism.track;

import prism.core.Angles;

import java.util.ArrayDeque;
import java.util.Iterator;

/**
 * Timestamped observation history for one bot within one round. Only observed samples are
 * stored; interpolation is computed on demand between same-round observations no more than
 * {@code maxGap} turns apart and is always marked as such.
 */
public final class History {
    public enum Provenance { OBSERVED, INTERPOLATED, PREDICTED }

    public record Sample(int turn, double x, double y, double heading, double speed, double energy, Provenance provenance) {}

    private final ArrayDeque<Sample> samples = new ArrayDeque<>();
    private final int capacity;
    private final int maxGap;

    public History(int capacity, int maxGap) { this.capacity = capacity; this.maxGap = maxGap; }

    public void add(Sample s) {
        if (s.provenance() != Provenance.OBSERVED) throw new IllegalArgumentException("only observations are stored");
        Sample last = samples.peekLast();
        if (last != null && s.turn() <= last.turn()) return; // duplicate or out of order: keep first
        samples.addLast(s);
        while (samples.size() > capacity) samples.removeFirst();
    }

    public void clear() { samples.clear(); }
    public int size() { return samples.size(); }
    public Sample latest() { return samples.peekLast(); }
    public Sample oldest() { return samples.peekFirst(); }

    /** Previous observation strictly before the latest one, or null. */
    public Sample previous() {
        if (samples.size() < 2) return null;
        Iterator<Sample> it = samples.descendingIterator();
        it.next();
        return it.next();
    }

    /**
     * Sample at a turn: the observation itself, a marked interpolation if both neighbours are
     * within the gap limit, or null. Never extrapolates beyond the latest observation.
     */
    public Sample at(int turn) {
        Sample before = null, after = null;
        Iterator<Sample> it = samples.descendingIterator();
        while (it.hasNext()) {
            Sample s = it.next();
            if (s.turn() == turn) return s;
            if (s.turn() > turn) after = s;
            else { before = s; break; }
        }
        if (before == null || after == null) return null;
        if (after.turn() - before.turn() > maxGap) return null;
        double f = (turn - before.turn()) / (double) (after.turn() - before.turn());
        double h = Angles.heading(before.heading() + f * Angles.wrap(after.heading() - before.heading()));
        return new Sample(turn, before.x() + f * (after.x() - before.x()), before.y() + f * (after.y() - before.y()), h,
                before.speed() + f * (after.speed() - before.speed()), Double.NaN, Provenance.INTERPOLATED);
    }

    /** Sample at least {@code back} turns before {@code turn}; falls back to the oldest available. */
    public Sample atOrOldest(int turn) {
        Sample s = at(turn);
        if (s != null) return s;
        Sample o = oldest();
        return o != null && o.turn() <= turn ? o : (o != null ? o : null);
    }
}
