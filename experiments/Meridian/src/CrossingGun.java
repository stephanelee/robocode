import java.io.*;
import java.nio.charset.StandardCharsets;
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
    private final Example[] examples;
    private final boolean frozen;

    CrossingGun() {
        examples = new Example[CAPACITY];
        frozen = false;
    }

    private CrossingGun(Example[] prior) {
        examples = prior;
        count = prior.length;
        frozen = true;
    }

    static CrossingGun loadPrior() {
        var resource = CrossingGun.class.getResourceAsStream("/crossing-prior.csv");
        if (resource == null) throw new IllegalStateException("Missing crossing prior");
        List<Example> rows = new ArrayList<>();
        try (var reader = new BufferedReader(new InputStreamReader(resource, StandardCharsets.US_ASCII))) {
            for (String line; (line = reader.readLine()) != null;) {
                double[] v = Arrays.stream(line.split(",")).mapToDouble(Double::parseDouble).toArray();
                if (v.length != 5 || !Arrays.stream(v).allMatch(Double::isFinite) || Math.abs(v[4]) > LIMIT)
                    throw new IllegalStateException("Invalid prior example");
                rows.add(new Example(new Context(v[0], v[1], v[2], v[3]), v[4]));
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        if (rows.size() != 206) throw new IllegalStateException("Incomplete crossing prior");
        return new CrossingGun(rows.toArray(Example[]::new));
    }
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

    void requested(Launch launch) { if (!frozen) requested = launch; }

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
        if (!frozen && support < 4) return Double.NaN;
        int best = BINS / 2;
        for (int i = 0; i < BINS; i++) {
            if (scores[i] > scores[best]) best = i;
        }
        return query.direction(best * STEP - LIMIT);
    }
}
