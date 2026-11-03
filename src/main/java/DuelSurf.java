import java.util.List;

/**
 * Pure 1v1 movement controller retaining an analytic prior with bounded matched-hit influence. The body stays roughly
 * perpendicular to the opponent (tilted to hold range) and each turn the controller chooses a travel option:
 * full speed in either direction along that line (driving forward or backward, whichever keeps the body
 * orientation), or stopping. Each option is simulated in the pinned Tank Royale 1.4.0 order: new speed from
 * {@code calcNewBotSpeed}, movement along the current body heading, then a turn limited by the new speed. The option
 * with the best worst-wave safety wins: analytic clearance blended with at most 25% learned footprint safety when
 * sufficient history exists, or exact analytic clearance otherwise. Paths that touch a wall or move further
 * into the wall envelope are rejected. Inputs are only our own state, scans, legally inferred waves and matched hits.
 */
final class DuelSurf {
    static final double PREFERRED_RANGE = 450;
    static final double MAX_TILT = 30;
    static final double WALL_MARGIN = 30;
    static final double BOT_RADIUS = 18;
    static final double CLEARANCE_CAP = 60;
    static final int HORIZON = 90;
    static final int WALL_HORIZON = 20;
    static final double MIN_PROFILE_HITS = 3;
    static final double MAX_PROFILE_WEIGHT = 0.25;
    static final double PROFILE_SUPPORT_SCALE = 24;

    private static final double ACCELERATION = 1, DECELERATION = -2, MAX_SPEED = 8, MAX_TURN = 10;

    private DuelSurf() {}

    /** An inferred wave whose shooter aimed at {@code aimTurn} from the origin with the given bearings. */
    record Threat(double originX, double originY, double bulletSpeed, int aimTurn, double headOn, double linear,
                  double bearingToUs, double[] danger) {
        Threat {
            // Each decision sees only the legal matched-hit history available when this threat was constructed.
            danger = danger == null ? null : danger.clone();
        }

        Threat(double originX, double originY, double bulletSpeed, int aimTurn, double headOn, double linear) {
            this(originX, originY, bulletSpeed, aimTurn, headOn, linear, Double.NaN, null);
        }
    }

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
     * Minimum safety score over threats when executing a command every turn, or a very negative score
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
                clearance = Math.min(clearance, blendedSafety(t, bearing, distance, lateral));
            }
            if (remaining == 0 && step >= WALL_HORIZON) break;
        }
        return clearance;
    }

    /**
     * Prospectively fixed support heuristic, not calibrated statistical confidence. The 25% cap is reached at
     * eight effective matched hits, not 24; at least 75% of analytic clearance is always retained. Invalid or
     * insufficient profiles preserve the original score exactly.
     */
    static double profileWeight(Threat threat) {
        double total = effectiveHits(threat);
        return Double.isFinite(total) && total >= MIN_PROFILE_HITS
                ? Math.min(MAX_PROFILE_WEIGHT, total / (total + PROFILE_SUPPORT_SCALE)) : 0;
    }

    static double blendedSafety(Threat threat, double bearing, double distance, double analyticClearance) {
        double analytic = Math.min(CLEARANCE_CAP, analyticClearance);
        double probability = profileDanger(threat, bearing, distance);
        if (!Double.isFinite(probability)) return analytic;
        double weight = profileWeight(threat);
        if (weight == 0) return analytic;
        double learned = CLEARANCE_CAP * (1 - probability);
        return (1 - weight) * analytic + weight * learned;
    }

    private static double effectiveHits(Threat threat) {
        if (threat.danger == null || threat.danger.length < 3) return Double.NaN;
        double total = 0;
        for (double value : threat.danger) {
            if (!Double.isFinite(value) || value < 0) return Double.NaN;
            total += value;
        }
        return Double.isFinite(total) ? total : Double.NaN;
    }

    /**
     * Fraction of the legally learned hit distribution covered by the predicted bot footprint. A one-bin
     * triangular kernel (1/4, 1/2, 1/4, renormalized at the edges) prevents a single bin boundary from appearing
     * perfectly safe. Returns NaN until at least three finite, nonnegative effective matched hits exist, so the
     * original analytic score remains exact without adequate evidence. This uses EnemyWave.hitBin's unsigned
     * geometric GF convention; no lateral-direction mirroring or unobserved opponent bullet is introduced.
     */
    static double profileDanger(Threat threat, double bearing, double distance) {
        double[] bins = threat.danger;
        if (bins == null || bins.length < 3 || !Double.isFinite(threat.bearingToUs())
                || !Double.isFinite(threat.bulletSpeed()) || threat.bulletSpeed() <= 0
                || !Double.isFinite(bearing) || !Double.isFinite(distance) || distance <= 0) return Double.NaN;
        double total = effectiveHits(threat);
        if (!Double.isFinite(total) || total < MIN_PROFILE_HITS) return Double.NaN;
        double escape = Math.toDegrees(Math.asin(Math.min(1, 8 / threat.bulletSpeed())));
        if (!Double.isFinite(escape) || escape <= 0) return Double.NaN;
        double offset = CombatMath.relativeAngle(bearing - threat.bearingToUs());
        double halfWidth = Math.toDegrees(Math.asin(Math.min(1, BOT_RADIUS / distance)));
        int low = CombatMath.gfBin((offset - halfWidth) / escape, bins.length);
        int high = CombatMath.gfBin((offset + halfWidth) / escape, bins.length);
        double probability = 0;
        for (int i = 0; i < bins.length; i++) {
            double available = 0.5 + (i > 0 ? 0.25 : 0) + (i + 1 < bins.length ? 0.25 : 0);
            double covered = i >= low && i <= high ? 0.5 : 0;
            if (i > 0 && i - 1 >= low && i - 1 <= high) covered += 0.25;
            if (i + 1 < bins.length && i + 1 >= low && i + 1 <= high) covered += 0.25;
            probability += (bins[i] / total) * covered / available;
        }
        return Math.max(0, Math.min(1, probability));
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
