package prism.core;

/**
 * Back-as-front driving using the actual engine speed law. A desired travel heading becomes a
 * body turn of at most a quarter turn plus a signed target speed. When the travel direction must
 * reverse, the target is chosen from {+-8, 0} by predicting the engine's next speed: from high
 * speed the engine's instant stop (target 0) beats the crossing law.
 */
public final class Drive {
    private Drive() {}

    public record Control(double targetSpeed, double turnRate) {}

    public static Control toward(Physics.Motion m, double travelHeading, double speedMagnitude) {
        double fwd = Angles.wrap(travelHeading - m.heading());
        int sign;
        double turn;
        if (Math.abs(fwd) <= Math.PI / 2) { sign = 1; turn = fwd; }
        else { sign = -1; turn = Angles.wrap(travelHeading + Math.PI - m.heading()); }
        double desired = sign * Math.abs(speedMagnitude);
        double target = desired;
        if (desired != 0 && m.speed() != 0 && Math.signum(m.speed()) != sign) {
            double cross = Physics.newSpeed(m.speed(), desired);
            double stop = Physics.newSpeed(m.speed(), 0.0);
            if (sign * stop > sign * cross) target = 0.0;
        }
        return new Control(target, turn);
    }

    /** Stop command: engine target zero, preserving body orientation. */
    public static Control stop() { return new Control(0.0, 0.0); }
}
