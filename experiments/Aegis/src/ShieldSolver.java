/**
 * Chooses a 0.1-power interception bullet against an inferred head-on shot, robust to small lateral aim errors.
 * Engine contract (pinned 1.4): a bullet created in turn f starts at the shooter's end-of-(f-1) position, uses the
 * end-of-(f-1) gun direction, and in turn j is tested on the segment [tick j-f+1, tick j-f+2].
 */
final class ShieldSolver {
    static final double POWER = 0.1, SPEED = Physics.bulletSpeed(POWER);
    static final double[] OFFSETS = {-4, -3, -2, -1, 0, 1, 2, 3, 4};

    record Plan(int fireTurn, double direction, int covered, double score, int interceptTurn, int mask) {}

    interface Positions { double x(int turn); double y(int turn); }

    /** Per-offset unit directions and first unshielded hit turns against our planned positions. */
    record Lines(double[] ex, double[] ey, int[] hit) {}

    static Lines lines(Shot shot, Positions own) {
        int n = OFFSETS.length; double[] exs = new double[n], eys = new double[n]; int[] hits = new int[n];
        double nx = -Math.sin(Math.toRadians(shot.heading)), ny = Math.cos(Math.toRadians(shot.heading));
        for (int i = 0; i < n; i++) {
            double ax = shot.tx + nx * OFFSETS[i], ay = shot.ty + ny * OFFSETS[i];
            double h = Math.toRadians(Physics.bearing(shot.psx, shot.psy, ax, ay)), ex = Math.cos(h), ey = Math.sin(h);
            int hit = Integer.MAX_VALUE;
            for (int j = Math.max(shot.fireTurn, shot.contactTurn() - 4); j < shot.fireTurn + 400; j++) {
                double d1 = shot.travelled(j), d2 = d1 + shot.speed;
                if (Physics.segmentHitsCircle(shot.ox + ex * d1, shot.oy + ey * d1, shot.ox + ex * d2, shot.oy + ey * d2, own.x(j), own.y(j), Physics.BOT_RADIUS)) { hit = j; break; }
                if (d1 > shot.range + 60) break;
            }
            exs[i] = ex; eys[i] = ey; hits[i] = hit;
        }
        return new Lines(exs, eys, hits);
    }

    /** Coverage of one shield direction fired in turn f from the end-of-(f-1) planned position. */
    static double score(Shot shot, int f, double direction, Positions own, int[] coveredOut) { return score(shot, f, direction, own, lines(shot, own), coveredOut); }

    static double score(Shot shot, int f, double direction, Positions own, Lines lines, int[] coveredOut) {
        int[] last = new int[1]; double v = score(shot, f, direction, own, lines, coveredOut, last); return v;
    }

    static double score(Shot shot, int f, double direction, Positions own, Lines lines, int[] coveredOut, int[] latestIntercept) {
        double px = own.x(f - 1), py = own.y(f - 1);
        double cw = Math.cos(Math.toRadians(direction)), sw = Math.sin(Math.toRadians(direction));
        double score = 0; int covered = 0;
        for (int i = 0; i < OFFSETS.length; i++) {
            double offset = OFFSETS[i], ex = lines.ex()[i], ey = lines.ey()[i]; int hit = lines.hit()[i];
            boolean ok = false;
            for (int j = f; j < Math.min(hit, f + 14); j++) {
                double d1 = shot.travelled(j), d2 = d1 + shot.speed, s1 = SPEED * (j - f + 1), s2 = s1 + SPEED;
                if (Physics.segmentsIntersect(shot.ox + ex * d1, shot.oy + ey * d1, shot.ox + ex * d2, shot.oy + ey * d2,
                        px + cw * s1, py + sw * s1, px + cw * s2, py + sw * s2)) { ok = true; latestIntercept[0] = Math.max(latestIntercept[0], j); break; }
            }
            if (ok) { covered++; if (coveredOut != null && coveredOut.length > 1) coveredOut[1] |= 1 << i; score += Math.exp(-offset * offset / 8.0) + (offset == 0 ? 1.0 : 0.0); }
        }
        if (coveredOut != null) coveredOut[0] = covered;
        return score;
    }

    /** Best direction for a fixed fire turn: coarse 0.5 degree sweep, then a 0.02 degree refinement. */
    static Plan best(Shot shot, int f, Positions own) { return best(shot, f, own, lines(shot, own)); }

    static Plan best(Shot shot, int f, Positions own, Lines lines) {
        double px = own.x(f - 1), py = own.y(f - 1);
        double base = Physics.bearing(px, py, shot.ox, shot.oy);
        double bestDir = base, bestScore = -1;
        for (double d = -80; d <= 80; d += 1.0) {
            double s = score(shot, f, base + d, own, lines, null);
            if (s > bestScore + 1e-12) { bestScore = s; bestDir = base + d; }
        }
        if (bestScore <= 0) return new Plan(f, bestDir, 0, 0, -1, 0);
        double lo = bestDir - 1.0, hi = bestDir + 1.0, sum = 0, n = 0, top = bestScore;
        for (double d = lo; d <= hi; d += 0.04) { double s = score(shot, f, d, own, lines, null); if (s > top + 1e-12) { top = s; sum = 0; n = 0; } if (Math.abs(s - top) < 1e-12) { sum += d; n++; } }
        double dir = n > 0 ? sum / n : bestDir; // centre of the best plateau
        int[] c = new int[2], j = new int[1]; double s = score(shot, f, dir, own, lines, c, j);
        return new Plan(f, Physics.normalize(dir), c[0], s, j[0], c[1]);
    }
}
