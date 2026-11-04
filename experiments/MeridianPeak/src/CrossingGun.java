import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Learns target bearings at bullet arrival using Meridian's confirmed launches and later scans. */
final class CrossingGun {
    private static final int CAPACITY = 192;
    private static final int BINS = 51;
    private static final double LIMIT = 1.25;
    private static final double STEP = 2 * LIMIT / (BINS - 1);

    record Context(double lateral, double radial, double wall, double flight) {
        double weight(Context other) {
            double a = (lateral - other.lateral) / 3;
            double b = (radial - other.radial) / 4;
            double c = (wall - other.wall) / .35;
            double d = (flight - other.flight) / 15;
            return 1 / Math.pow(1 + a * a + b * b + c * c + d * d, 2);
        }
    }
    record Launch(int turn, double x, double y, double power, double bearing,
                  double sign, double range, Context context) {
        double speed() { return 20 - 3 * power; }
        double escape() { return Math.toDegrees(Math.asin(8 / speed())); }
        double offset(double angle) { return Flight.delta(angle - bearing) * sign / escape(); }
        double direction(double offset) { return bearing + sign * escape() * offset; }
    }
    private record Example(Context context, double offset) {}
    private final Example[] examples = new Example[CAPACITY];
    private int next, count;
    private final double[] scores = new double[BINS];
    private final List<Launch> inFlight = new ArrayList<>();
    private Launch requested;

    static Launch launch(int turn, Flight.Pose own, Flight.Pose enemy, double power, int width, int height) {
        double dx = enemy.x() - own.x(), dy = enemy.y() - own.y();
        double bearing = Flight.angle(dx, dy), range = Math.hypot(dx, dy);
        double lat = enemy.v() * Flight.sin(enemy.h() - bearing);
        double radial = enemy.v() * Flight.cos(enemy.h() - bearing);
        double travel = enemy.h() + (enemy.v() < 0 ? 180 : 0);
        double wall = Math.min(boundary(enemy.x(), Flight.cos(travel), width),
                               boundary(enemy.y(), Flight.sin(travel), height));
        var context = new Context(Math.abs(lat), radial, Flight.clip(wall / 180, 0, 1), range / (20 - 3 * power));
        return new Launch(turn, own.x(), own.y(), power, bearing, lat < 0 ? -1 : 1, range, context);
    }

    private static double boundary(double coordinate, double component, int size) {
        if (Math.abs(component) < 1e-9) return Double.POSITIVE_INFINITY;
        return ((component > 0 ? size - 18 : 18) - coordinate) / component;
    }

    void newRound() {
        requested = null;
        inFlight.clear();
    }

    void requested(Launch launch) { requested = launch; }

    void fired(int turn, double x, double y, double power) {
        if (requested != null && requested.turn == turn && Math.abs(requested.power - power) < 1e-6
                && Math.hypot(requested.x - x, requested.y - y) <= 1e-4) {
            inFlight.add(requested);
        }
        requested = null;
    }

    int observe(int turn, Flight.Pose enemy, int previousTurn, Flight.Pose previous) {
        int added = 0;
        for (var it = inFlight.iterator(); it.hasNext();) {
            Launch shot = it.next();
            int age = turn - shot.turn + 1;
            if (age < 1) continue;
            double range = Math.hypot(enemy.x() - shot.x, enemy.y() - shot.y);
            if (range > shot.speed() * (age + 1)) continue;
            it.remove();
            // Require the two scans that bracket the first crossing of the front's far edge.
            if (previous == null || previousTurn != turn - 1) continue;
            double before = Math.hypot(previous.x() - shot.x, previous.y() - shot.y);
            if (before <= age * shot.speed()) continue;
            double offset = shot.offset(Flight.angle(enemy.x() - shot.x, enemy.y() - shot.y));
            if (Math.abs(offset) > LIMIT) continue;
            examples[next] = new Example(shot.context, offset);
            next = (next + 1) % CAPACITY;
            count = Math.min(CAPACITY, count + 1);
            added++;
        }
        return added;
    }

    /** Return NaN while comparable, already-resolved launches have insufficient support. */
    double aim(Launch query) {
        if (count < 8) return Double.NaN;
        Arrays.fill(scores, 0);
        double support = 0;
        double footprint = Math.max(STEP, Math.toDegrees(Math.atan2(18, query.range)) / query.escape());
        for (int i = 0; i < count; i++) {
            Example example = examples[i];
            double weight = query.context.weight(example.context);
            support += weight;
            for (int bin = 0; bin < BINS; bin++) {
                double distance = Math.abs(bin * STEP - LIMIT - example.offset);
                if (distance < footprint) scores[bin] += weight * (1 - distance / footprint);
            }
        }
        if (support < 4) return Double.NaN;
        int best = BINS / 2;
        for (int i = 0; i < BINS; i++) {
            if (scores[i] > scores[best]) best = i;
        }
        return query.direction(best * STEP - LIMIT);
    }
    record ExampleSnapshot(int index, Context context, double offset) {}
    record Snapshot(int capacity, int count, int next, java.util.List<ExampleSnapshot> examples,
                    java.util.List<Launch> inFlight, Launch requested) {}
    Snapshot snapshot() {
        var copy = new java.util.ArrayList<ExampleSnapshot>();
        for (int i = 0; i < count; i++) copy.add(new ExampleSnapshot(i, examples[i].context, examples[i].offset));
        return new Snapshot(CAPACITY, count, next, java.util.List.copyOf(copy),
                java.util.List.copyOf(inFlight), requested);
    }
    enum Reason { RESOLVED, MISSING_BRACKET, INSIDE_PREVIOUS_FRONT, OUTSIDE_OFFSET }
    record CrossingLabel(int turn, Launch launch, Flight.Pose enemy, int previousTurn,
                         Flight.Pose previous, Reason reason, Double offset) {}
    /** Passive pre-observation labels, with the original observe method left intact. */
    java.util.List<CrossingLabel> labelsFor(int turn, Flight.Pose enemy, int previousTurn, Flight.Pose previous) {
        var labels=new java.util.ArrayList<CrossingLabel>();
        for(var shot:inFlight) {
            int age=turn-shot.turn+1;
            if(age<1 || Math.hypot(enemy.x()-shot.x,enemy.y()-shot.y)>shot.speed()*(age+1))continue;
            Reason reason; Double offset=null;
            if(previous==null || previousTurn!=turn-1)reason=Reason.MISSING_BRACKET;
            else if(Math.hypot(previous.x()-shot.x,previous.y()-shot.y)<=age*shot.speed())reason=Reason.INSIDE_PREVIOUS_FRONT;
            else {
                offset=shot.offset(Flight.angle(enemy.x()-shot.x,enemy.y()-shot.y));
                reason=Math.abs(offset)>LIMIT?Reason.OUTSIDE_OFFSET:Reason.RESOLVED;
            }
            labels.add(new CrossingLabel(turn,shot,enemy,previousTurn,previous,reason,offset));
        }
        return java.util.List.copyOf(labels);
    }
}
