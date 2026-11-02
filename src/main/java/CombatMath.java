/** Pure calculations using Tank Royale's east=0, north=90 coordinates. */
final class CombatMath {
    private CombatMath() {}

    enum WaveObservation { BEFORE, CROSSING, PASSED }

    static WaveObservation observeTargetWave(double traveled, double distance, double bulletSpeed) {
        // One turn of travel plus bot collision radius bounds scan/tick uncertainty.
        double margin = bulletSpeed + 18;
        if (traveled > distance + margin) return WaveObservation.PASSED;
        if (traveled < distance - margin) return WaveObservation.BEFORE;
        return WaveObservation.CROSSING;
    }

    static double orbitHeading(double bearingToEnemy, boolean clockwise) {
        // The bearing points inward: +90 is clockwise around the enemy.
        return bearingToEnemy + (clockwise ? 90 : -90);
    }

    static double relativeAngle(double angle) {
        return ((angle + 180) % 360 + 360) % 360 - 180;
    }

    static double bearing(double dx, double dy) {
        return Math.toDegrees(Math.atan2(dy, dx));
    }

    static double turnRate(double previous, double current, int elapsed) {
        return elapsed > 0 ? relativeAngle(current - previous) / elapsed : 0;
    }

    static double[] project(double x, double y, double direction, double speed,
                            double turnRate, double turns) {
        double angle = Math.toRadians(direction);
        double omega = Math.toRadians(turnRate);
        if (Math.abs(omega) < 1e-8) {
            return new double[]{x + Math.cos(angle) * speed * turns,
                    y + Math.sin(angle) * speed * turns};
        }
        double radius = speed / omega;
        return new double[]{x + radius * (Math.sin(angle + omega * turns) - Math.sin(angle)),
                y + radius * (Math.cos(angle) - Math.cos(angle + omega * turns))};
    }

    static int bestBin(double[] enemy, double[] opponentClass, double enemyWeight) {
        int best = enemy.length / 2;
        double bestValue = enemyWeight * enemy[best] + (1 - enemyWeight) * opponentClass[best];
        for (int i = 0; i < enemy.length; i++) {
            double value = enemyWeight * enemy[i] + (1 - enemyWeight) * opponentClass[i];
            if (value > bestValue) {
                best = i;
                bestValue = value;
            }
        }
        return best;
    }

    static int gfBin(double gf, int bins) {
        return (int) Math.max(0, Math.min(bins - 1, Math.round((gf + 1) * (bins - 1) / 2)));
    }

    /**
     * Smallest-deviation heading, searched in 5-degree steps in both rotation directions over the full circle, whose
     * point {@code stick} units ahead lies strictly inside the {@code margin} envelope. If none exists (e.g. deep in a
     * corner), returns the heading toward the arena centre. The result is normalized to [0, 360).
     */
    static double nearestSafeHeading(double x, double y, double heading, double width, double height, double margin) {
        for (int step = 0; step <= 36; step++) {
            for (int sign : step == 0 ? new int[]{1} : new int[]{1, -1}) {
                double h = heading + sign * step * 5.0;
                double rad = Math.toRadians(h);
                double tx = x + Math.cos(rad) * margin, ty = y + Math.sin(rad) * margin;
                if (tx > margin && tx < width - margin && ty > margin && ty < height - margin)
                    return ((h % 360) + 360) % 360;
            }
        }
        double centre = bearing(width / 2 - x, height / 2 - y);
        return ((centre % 360) + 360) % 360;
    }
}
