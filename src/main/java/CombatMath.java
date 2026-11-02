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
}
