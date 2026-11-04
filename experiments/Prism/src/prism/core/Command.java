package prism.core;

/**
 * One complete intent. Rates are native radians per turn. {@code firepower == 0} explicitly
 * clears any pending firepower in the API intent so a stale plan can never fire later.
 */
public record Command(long sequence, double targetSpeed, double bodyTurn, double gunTurn, double radarTurn,
                      double firepower, String moveReason) {
    public Command {
        if (!(Double.isFinite(targetSpeed) && Double.isFinite(bodyTurn) && Double.isFinite(gunTurn)
                && Double.isFinite(radarTurn) && Double.isFinite(firepower)))
            throw new IllegalArgumentException("non-finite command");
    }

    public static Command idle(long seq) { return new Command(seq, 0, 0, 0, Math.toRadians(45), 0, "idle"); }
}
