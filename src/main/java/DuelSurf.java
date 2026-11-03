import java.util.List;

/**
 * Pure 1v1 movement controller against an opponent that fires head-on or linear aim. The body stays roughly
 * perpendicular to the opponent (tilted to hold range) and the controller chooses a travel option each turn:
 * full speed in either direction along that line, or stopping. Each option is simulated with Tank Royale
 * acceleration (1) and deceleration (2) until every known incoming wave arrives, and the option that keeps the
 * largest minimum lateral clearance from both reconstructed aim points wins. Paths that leave the wall envelope
 * are rejected. Inputs are only our own state, the opponent's scanned position and legally inferred waves.
 */
final class DuelSurf {
    static final double PREFERRED_RANGE = 450;
    static final double MAX_TILT = 30;
    static final double WALL_MARGIN = 30;
    static final double CLEARANCE_CAP = 60;
    static final int HORIZON = 90;
    static final int WALL_HORIZON = 20;

    private DuelSurf() {}

    /** An inferred wave whose shooter aimed at {@code aimTurn} from the origin with the given bearings. */
    record Threat(double originX, double originY, double bulletSpeed, int aimTurn, double headOn, double linear) {}

    /** Body heading to turn toward, signed target speed, and the chosen travel sign (+1, -1, or 0 to stop). */
    record Command(double bodyHeading, double targetSpeed, int travelSign) {}

    static double nextSpeed(double speed, double target) {
        if (speed == target) return speed;
        if (speed == 0 || Math.signum(speed) == Math.signum(target)) {
            if (Math.abs(target) > Math.abs(speed)) return speed + Math.signum(target) * Math.min(1, Math.abs(target - speed));
            return speed - Math.signum(speed) * Math.min(2, Math.abs(speed - target));
        }
        // Opposite sign or stopping: decelerate by 2; crossing zero keeps only what deceleration leaves.
        double magnitude = Math.abs(speed) - 2;
        if (magnitude >= 0) return Math.signum(speed) * magnitude;
        return target == 0 ? 0 : Math.signum(target) * Math.min(1, -magnitude / 2);
    }

    /** Travel direction for a sign: perpendicular to the opponent, tilted inward when far and outward when close. */
    static double travelDirection(double x, double y, double enemyX, double enemyY, int sign) {
        double bearing = CombatMath.bearing(enemyX - x, enemyY - y);
        double distance = Math.hypot(enemyX - x, enemyY - y);
        double tilt = Math.max(-MAX_TILT, Math.min(MAX_TILT, (distance - PREFERRED_RANGE) / 10));
        return bearing + sign * (90 - tilt);
    }

    static Command choose(double x, double y, double bodyDirection, double speed, int now,
                          double enemyX, double enemyY, List<Threat> threats, int currentSign,
                          double width, double height) {
        int first = currentSign == 0 ? 1 : currentSign;
        int bestSign = first;
        double bestScore = Double.NEGATIVE_INFINITY, bestDirection = bodyDirection;
        // Rotate toward the centre only when neither perpendicular direction can be held (e.g. in a corner).
        boolean cornered = true;
        for (int sign : new int[]{1, -1})
            if (score(x, y, bodyDirection, speed, travelDirection(x, y, enemyX, enemyY, sign), sign, now,
                    List.of(), width, height) > -500) cornered = false;
        for (int sign : new int[]{first, -first, 0}) {
            double direction = sign == 0 ? (speed >= 0 ? bodyDirection : bodyDirection + 180)
                    : cornered ? safeTravelDirection(x, y, bodyDirection, speed, now, enemyX, enemyY, sign, width, height)
                    : travelDirection(x, y, enemyX, enemyY, sign);
            double score = score(x, y, bodyDirection, speed, direction, sign, now, threats, width, height);
            if (sign == currentSign) score += 3;  // hysteresis
            if (sign != 0) score += 1;            // prefer moving when equally safe
            if (score > bestScore) { bestScore = score; bestSign = sign; bestDirection = direction; }
        }
        if (bestSign == 0) return new Command(bodyDirection, 0, 0);
        // Drive forward or backward, whichever body orientation is closer to the current one.
        boolean forward = Math.abs(CombatMath.relativeAngle(bestDirection - bodyDirection)) <= 90;
        return new Command(forward ? normalize(bestDirection) : normalize(bestDirection + 180), forward ? 8 : -8, bestSign);
    }

    /**
     * The perpendicular travel direction for a sign, rotated toward the arena centre in 10-degree steps (up to
     * 90 degrees) when holding it would leave the wall envelope within the wall horizon.
     */
    static double safeTravelDirection(double x, double y, double bodyDirection, double speed, int now,
                                      double enemyX, double enemyY, int sign, double width, double height) {
        double direction = travelDirection(x, y, enemyX, enemyY, sign);
        double centre = CombatMath.bearing(width / 2 - x, height / 2 - y);
        double turn = Math.signum(CombatMath.relativeAngle(centre - direction));
        for (int step = 0; step <= 9; step++) {
            double candidate = direction + turn * step * 10;
            if (score(x, y, bodyDirection, speed, candidate, sign, now, List.of(), width, height) > -500)
                return candidate;
        }
        return direction;
    }

    /**
     * Minimum capped lateral clearance over threats when holding one option, or a very negative score if the
     * path leaves the wall envelope. The initial speed along {@code direction} is the projection of the current
     * velocity (body heading times signed speed); afterwards the bot accelerates toward 8 (or brakes to 0).
     */
    static double score(double x, double y, double bodyDirection, double speed, double direction, int sign, int now,
                        List<Threat> threats, double width, double height) {
        double rad = Math.toRadians(direction);
        double cos = Math.cos(rad), sin = Math.sin(rad);
        double along = speed * Math.cos(Math.toRadians(bodyDirection - direction));
        double px = x, py = y;
        double clearance = CLEARANCE_CAP;
        boolean[] done = new boolean[threats.size()];
        int remaining = threats.size();
        for (int i = 0; i < threats.size(); i++) {
            Threat t = threats.get(i);
            // A wave whose front is already more than a bot radius past us cannot hit any option.
            if (t.bulletSpeed() * (now - t.aimTurn() + 1) > Math.hypot(x - t.originX(), y - t.originY()) + 18) {
                done[i] = true;
                remaining--;
            }
        }
        int horizon = remaining == 0 ? WALL_HORIZON : HORIZON;
        for (int step = 1; step <= horizon; step++) {
            along = nextSpeed(along, sign == 0 ? 0 : 8);
            double before = wallDistance(px, py, width, height);
            px += cos * along;
            py += sin * along;
            double after = wallDistance(px, py, width, height);
            // Only moving further into the envelope counts: a bot already touching a wall may slide along or leave it.
            if (after < WALL_MARGIN && after < before - 1e-9) return -1000 + step;
            for (int i = 0; i < threats.size(); i++) {
                if (done[i]) continue;
                Threat t = threats.get(i);
                double distance = Math.hypot(px - t.originX(), py - t.originY());
                if (t.bulletSpeed() * (now + step - t.aimTurn() + 1) < distance) continue;
                done[i] = true;
                remaining--;
                double bearing = CombatMath.bearing(px - t.originX(), py - t.originY());
                double lateral = Math.min(lateral(bearing, t.headOn()), lateral(bearing, t.linear())) * distance;
                clearance = Math.min(clearance, lateral);
            }
            if (remaining == 0 && step >= WALL_HORIZON) break;
        }
        return clearance;
    }

    static double wallDistance(double x, double y, double width, double height) {
        return Math.min(Math.min(x, width - x), Math.min(y, height - y));
    }

    private static double lateral(double bearing, double aim) {
        return Math.abs(Math.sin(Math.toRadians(CombatMath.relativeAngle(bearing - aim))));
    }

    private static double normalize(double angle) {
        return ((angle % 360) + 360) % 360;
    }
}
