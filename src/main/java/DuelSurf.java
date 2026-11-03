import java.util.List;

/**
 * Pure 1v1 movement controller against an opponent that fires head-on or linear aim. The body stays roughly
 * perpendicular to the opponent (tilted to hold range) and each turn the controller chooses a travel option:
 * full speed in either direction along that line (driving forward or backward, whichever keeps the body
 * orientation), or stopping. Each option is simulated in the pinned Tank Royale 1.4.0 order: new speed from
 * {@code calcNewBotSpeed}, movement along the current body heading, then a turn limited by the new speed. The option
 * that keeps the largest minimum lateral clearance from both reconstructed aim points of every known incoming wave
 * wins; paths that touch a wall or move further into the wall envelope are rejected. Inputs are only our own state,
 * the opponent's scanned position and legally inferred waves.
 */
final class DuelSurf {
    static final double PREFERRED_RANGE = 450;
    static final double MAX_TILT = 30;
    static final double WALL_MARGIN = 30;
    static final double BOT_RADIUS = 18;
    static final double CLEARANCE_CAP = 60;
    static final int HORIZON = 90;
    static final int WALL_HORIZON = 20;

    private static final double ACCELERATION = 1, DECELERATION = -2, MAX_SPEED = 8, MAX_TURN = 10;

    private DuelSurf() {}

    /** An inferred wave whose shooter aimed at {@code aimTurn} from the origin with the given bearings. */
    record Threat(double originX, double originY, double bulletSpeed, int aimTurn, double headOn, double linear) {}

    /** Body heading to turn toward, signed target speed, and the chosen travel sign (+1, -1, or 0 to stop). */
    record Command(double bodyHeading, double targetSpeed, int travelSign) {}

    /** Port of Tank Royale 1.4.0 {@code calcNewBotSpeed} (rules/math.kt), including its zero-crossing rule. */
    static double nextSpeed(double current, double target) {
        if (current < 0) return -nextSpeed(-current, -target);
        if (current > 0) {
            if (Math.signum(current) == Math.signum(target)) {
                double diff = target - current;
                if (diff >= 0) return Math.min(current + Math.min(diff, ACCELERATION), MAX_SPEED);
                return Math.max(current + Math.max(diff, DECELERATION), -MAX_SPEED);
            }
            if (target == 0) return 0;
            double decelerationTime = current / -DECELERATION;
            return (1 - decelerationTime) * -ACCELERATION;
        }
        double diff = target - current;
        double acceleration = Math.min(Math.abs(diff), ACCELERATION);
        return diff >= 0 ? Math.min(current + acceleration, MAX_SPEED) : Math.max(current - acceleration, -MAX_SPEED);
    }

    /** Engine body turn limit for a speed. */
    static double maxTurn(double speed) {
        return MAX_TURN - 0.75 * Math.min(MAX_SPEED, Math.abs(speed));
    }

    /** Travel direction for a sign: perpendicular to the opponent, tilted inward when far and outward when close. */
    static double travelDirection(double x, double y, double enemyX, double enemyY, int sign) {
        double bearing = CombatMath.bearing(enemyX - x, enemyY - y);
        double distance = Math.hypot(enemyX - x, enemyY - y);
        double tilt = Math.max(-MAX_TILT, Math.min(MAX_TILT, (distance - PREFERRED_RANGE) / 10));
        return bearing + sign * (90 - tilt);
    }

    /** The command that realises a travel option: forward or backward along the travel line, or stop. */
    static Command command(double bodyDirection, double travel, int sign) {
        if (sign == 0) return new Command(normalize(bodyDirection), 0, 0);
        boolean forward = Math.abs(CombatMath.relativeAngle(travel - bodyDirection)) <= 90;
        return new Command(normalize(forward ? travel : travel + 180), forward ? MAX_SPEED : -MAX_SPEED, sign);
    }

    static Command choose(double x, double y, double bodyDirection, double speed, int now,
                          double enemyX, double enemyY, List<Threat> threats, int currentSign,
                          double width, double height) {
        int first = currentSign == 0 ? 1 : currentSign;
        // Rotate toward the centre only when neither perpendicular direction can be held (e.g. in a corner).
        boolean cornered = true;
        for (int sign : new int[]{1, -1}) {
            Command plain = command(bodyDirection, travelDirection(x, y, enemyX, enemyY, sign), sign);
            if (score(x, y, bodyDirection, speed, plain, now, List.of(), width, height) > -500) cornered = false;
        }
        Command best = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int sign : new int[]{first, -first, 0}) {
            double travel = sign == 0 ? bodyDirection : cornered
                    ? safeTravelDirection(x, y, bodyDirection, speed, now, enemyX, enemyY, sign, width, height)
                    : travelDirection(x, y, enemyX, enemyY, sign);
            Command candidate = command(bodyDirection, travel, sign);
            double score = score(x, y, bodyDirection, speed, candidate, now, threats, width, height);
            if (sign == currentSign) score += 3;  // hysteresis
            if (sign != 0) score += 1;            // prefer moving when equally safe
            if (score > bestScore) { bestScore = score; best = candidate; }
        }
        return best;
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
            if (score(x, y, bodyDirection, speed, command(bodyDirection, candidate, sign), now, List.of(),
                    width, height) > -500) return candidate;
        }
        return direction;
    }

    /**
     * Minimum capped lateral clearance over threats when executing a command every turn, or a very negative score
     * if the simulated path touches a wall or moves further into the wall envelope. Each simulated turn follows the
     * engine: speed update, movement along the current heading, then a turn limited both by the bot's request (current
     * speed) and by the engine (new speed).
     */
    static double score(double x, double y, double bodyDirection, double speed, Command command, int now,
                        List<Threat> threats, double width, double height) {
        double px = x, py = y, heading = bodyDirection, v = speed;
        double clearance = CLEARANCE_CAP;
        boolean[] done = new boolean[threats.size()];
        int remaining = threats.size();
        for (int i = 0; i < threats.size(); i++) {
            Threat t = threats.get(i);
            // A wave whose front is already more than a bot radius past us cannot hit any option.
            if (t.bulletSpeed() * (now - t.aimTurn() + 1) > Math.hypot(x - t.originX(), y - t.originY()) + BOT_RADIUS) {
                done[i] = true;
                remaining--;
            }
        }
        int horizon = remaining == 0 ? WALL_HORIZON : HORIZON;
        for (int step = 1; step <= horizon; step++) {
            double before = wallDistance(px, py, width, height);
            double requested = Math.max(-maxTurn(v), Math.min(maxTurn(v),
                    CombatMath.relativeAngle(command.bodyHeading() - heading)));
            v = nextSpeed(v, command.targetSpeed());
            double rad = Math.toRadians(heading);
            px += Math.cos(rad) * v;
            py += Math.sin(rad) * v;
            if (px < BOT_RADIUS || py < BOT_RADIUS || px > width - BOT_RADIUS || py > height - BOT_RADIUS)
                return -1000 + step;  // wall contact
            double after = wallDistance(px, py, width, height);
            // Moving further into the envelope is rejected; a bot already near a wall may slide along or leave it.
            if (after < WALL_MARGIN && after < before - 1e-9) return -1000 + step;
            heading += Math.max(-maxTurn(v), Math.min(maxTurn(v), requested));
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
