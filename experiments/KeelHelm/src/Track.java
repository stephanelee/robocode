/**
 * Per-round motion history of one bot, indexed by turn, plus the shared feature vector used by
 * both the gun (enemy as target) and the danger model (Keel as target). Legal observations only.
 */
final class Track {
    static final int DIMS = 8;
    private static final int CAP = 4096;
    private final Phys.Pose[] poses = new Phys.Pose[CAP];
    private final int[] stamp = new int[CAP];
    private final int[] lastDirChange = new int[CAP];
    private final int[] lastDecel = new int[CAP];
    private final int[] latDirAt = new int[CAP];
    private int latestTurn = -1;
    private int dirChange = 0, decel = 0, lastSign = 1;

    void clear() {
        java.util.Arrays.fill(stamp, -1);
        latestTurn = -1; dirChange = 0; decel = 0; lastSign = 1;
    }

    Track() { clear(); }

    /** Records pose at turn t. Lateral direction is relative to the observer position (ox, oy). */
    void add(int t, Phys.Pose p, double ox, double oy) {
        Phys.Pose prev = get(t - 1);
        double absBearing = Phys.bearing(ox, oy, p.x(), p.y());
        double lat = p.v() * Phys.sin(p.h() - absBearing);
        int sign = Math.abs(lat) > 0.05 ? (lat > 0 ? 1 : -1) : lastSign;
        if (sign != lastSign) dirChange = t;
        if (prev != null && Math.abs(p.v()) < Math.abs(prev.v()) - 0.01) decel = t;
        lastSign = sign;
        int k = t & (CAP - 1);
        poses[k] = p; stamp[k] = t; lastDirChange[k] = dirChange; lastDecel[k] = decel; latDirAt[k] = sign;
        latestTurn = Math.max(latestTurn, t);
    }

    int observations() { int n = 0; for (int t : stamp) if (t >= 0) n++; return n; }

    void copyFrom(Track other) {
        System.arraycopy(other.poses, 0, poses, 0, CAP);
        System.arraycopy(other.stamp, 0, stamp, 0, CAP);
        System.arraycopy(other.lastDirChange, 0, lastDirChange, 0, CAP);
        System.arraycopy(other.lastDecel, 0, lastDecel, 0, CAP);
        System.arraycopy(other.latDirAt, 0, latDirAt, 0, CAP);
        latestTurn = other.latestTurn; dirChange = other.dirChange; decel = other.decel; lastSign = other.lastSign;
    }

    Phys.Pose get(int t) {
        if (t < 0) return null;
        int k = t & (CAP - 1);
        return stamp[k] == t ? poses[k] : null;
    }

    int latest() { return latestTurn; }

    /** Most recent pose at or before t, within maxBack turns. */
    Phys.Pose atOrBefore(int t, int maxBack) {
        for (int i = 0; i <= maxBack; i++) { Phys.Pose p = get(t - i); if (p != null) return p; }
        return null;
    }

    int latDir(int t) { int k = t & (CAP - 1); return stamp[k] == t ? latDirAt[k] : lastSign; }

    /**
     * Features of this track's bot at turn t as seen by a shooter at (sx, sy) firing at bullet speed s.
     * Returns null if the pose is missing.
     */
    double[] features(int t, double sx, double sy, double s, double w, double h) {
        Phys.Pose p = get(t);
        if (p == null) return null;
        int k = t & (CAP - 1);
        double d = Phys.distance(sx, sy, p.x(), p.y());
        double ab = Phys.bearing(sx, sy, p.x(), p.y());
        double lat = p.v() * Phys.sin(p.h() - ab);
        double adv = -p.v() * Phys.cos(p.h() - ab);
        Phys.Pose prev = get(t - 1);
        double accel = prev == null ? 0 : Math.signum(Math.abs(p.v()) - Math.abs(prev.v()));
        double flight = Math.max(1, d / s);
        int dir = latDirAt[k];
        double mea = Math.toDegrees(Math.asin(8 / s));
        double fwd = wallRoom(sx, sy, d, ab, dir, mea, w, h);
        double rev = wallRoom(sx, sy, d, ab, -dir, mea, w, h);
        Phys.Pose back = atOrBefore(t - 10, 3);
        double disp = back == null ? 0.5 : Math.min(1, Phys.distance(p.x(), p.y(), back.x(), back.y()) / 80);
        return new double[]{
                Math.abs(lat) / 8,
                (adv / 8 + 1) / 2,
                Math.min(1, flight / 90),
                (accel + 1) / 2,
                Math.min(1, (t - lastDirChange[k]) / flight),
                Math.min(1, fwd / 1.2),
                Math.min(1, rev / 1.2),
                disp
        };
    }

    /** Fraction of the max escape angle reachable in direction dir before leaving the field. */
    static double wallRoom(double sx, double sy, double d, double ab, int dir, double mea, double w, double h) {
        double room = 0;
        for (double f = 0.05; f <= 1.25; f += 0.05) {
            double a = ab + dir * f * mea;
            double x = sx + Phys.cos(a) * d, y = sy + Phys.sin(a) * d;
            if (x < 18 || y < 18 || x > w - 18 || y > h - 18) break;
            room = f;
        }
        return room;
    }
}
