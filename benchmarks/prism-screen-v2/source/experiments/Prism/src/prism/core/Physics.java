package prism.core;

/**
 * Pure engine-native one-turn predictor. Order follows the official Tank Royale 1.4.0 turn
 * processor: (1) new speed from the current speed and target speed, (2) translation on the
 * heading held at the start of the turn, (3) body rotation limited by the new speed, then
 * (4) wall constraint along the travelled segment, with speed set to zero and first-contact damage.
 * No learning or opponent state lives here.
 */
public final class Physics {
    private Physics() {}

    /** Engine bot motion state; heading in native radians. */
    public record Motion(double x, double y, double heading, double speed) {
        public Motion {
            if (!(Double.isFinite(x) && Double.isFinite(y) && Double.isFinite(heading) && Double.isFinite(speed)))
                throw new IllegalArgumentException("non-finite motion");
        }
    }

    public record Step(Motion next, boolean hitWall, double wallDamage) {}

    /** Engine calcNewBotSpeed, including instant stop for target zero and the crossing law. */
    public static double newSpeed(double current, double target) {
        if (current < 0) return -newSpeed(-current, -target);
        if (current > 0) {
            if (Math.signum(current) == Math.signum(target)) {
                double diff = target - current;
                if (diff >= 0) return Math.min(current + Math.min(diff, Rules.ACCELERATION), Rules.MAX_SPEED);
                return Math.max(current + Math.max(diff, Rules.DECELERATION), -Rules.MAX_SPEED);
            }
            if (target == 0.0) return 0.0;
            double decelTime = current / -Rules.DECELERATION;
            return (1 - decelTime) * -Rules.ACCELERATION;
        }
        double diff = target - current;
        double acc = Math.min(Math.abs(diff), Rules.ACCELERATION);
        return diff >= 0 ? Math.min(current + acc, Rules.MAX_SPEED) : Math.max(current - acc, -Rules.MAX_SPEED);
    }

    /**
     * One turn. {@code turnRate} is the requested body turn in radians (the engine clamps it by the
     * new speed). {@code wallContactLastTurn} suppresses repeated wall damage like the engine does.
     */
    public static Step step(Motion m, double targetSpeed, double turnRate, Rules rules, boolean wallContactLastTurn) {
        double t = Math.max(-Rules.MAX_SPEED, Math.min(Rules.MAX_SPEED, targetSpeed));
        double v = newSpeed(m.speed(), t);
        double nx = m.x() + Math.cos(m.heading()) * v;
        double ny = m.y() + Math.sin(m.heading()) * v;
        double limit = Rules.maxTurnRad(v);
        double tr = Math.max(-limit, Math.min(limit, turnRate));
        double h = Angles.heading(m.heading() + tr);
        // the engine constrains once while handling wall collisions and again in constrainBotPositions;
        // near corners the second pass changes the result (verified against a recorded 1.4.0 trace)
        double[] c = constrain(m.x(), m.y(), nx, ny, rules);
        c = constrain(m.x(), m.y(), c[0], c[1], rules);
        boolean hit = c[0] != nx || c[1] != ny;
        double dmg = hit && !wallContactLastTurn ? Rules.wallDamage(v) : 0;
        return new Step(new Motion(c[0], c[1], h, hit ? 0.0 : v), hit, dmg);
    }

    public static Step step(Motion m, double targetSpeed, double turnRate, Rules rules) {
        return step(m, targetSpeed, turnRate, rules, false);
    }

    /** Engine constrainBotPosition: clamp along the travelled line, x first then y. */
    static double[] constrain(double ox, double oy, double x, double y, Rules r) {
        double nx = x, ny = y, R = Rules.BOT_RADIUS;
        if (x - R < 0) {
            nx = R;
            double dx = x - ox;
            if (dx != 0) ny = oy + (y - oy) * (nx - ox) / dx;
        } else if (x + R > r.width()) {
            nx = r.width() - R;
            double dx = x - ox;
            if (dx != 0) ny = oy + (y - oy) * (nx - ox) / dx;
        }
        if (y - R < 0) {
            ny = R;
            double dy = y - oy;
            if (dy != 0) nx = ox + (x - ox) * (ny - oy) / dy;
        } else if (y + R > r.height()) {
            ny = r.height() - R;
            double dy = y - oy;
            if (dy != 0) nx = ox + (x - ox) * (ny - oy) / dy;
        }
        return new double[] {nx, ny};
    }

    /** Distance travelled until stopped when commanding target zero: the engine stops instantly. */
    public static double stoppingDistance(double speed) { return 0.0; }
}
