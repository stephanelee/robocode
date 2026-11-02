/** Convert a travel heading into bounded forward or reverse body commands. */
final class Steering {
    record Command(double turnRate, double targetSpeed) {}

    private Steering() {}

    static Command toward(double bodyHeading, double travelHeading, double speed, double currentSpeed) {
        double turn = CombatMath.relativeAngle(travelHeading - bodyHeading);
        double targetSpeed = speed;
        if (Math.abs(turn) > 90) {
            turn = CombatMath.relativeAngle(turn + 180);
            targetSpeed = -targetSpeed;
        }
        double maxTurn = 10 - 0.75 * Math.abs(currentSpeed);
        double turnRate = Math.max(-maxTurn, Math.min(maxTurn, turn * 2));
        return new Command(turnRate, targetSpeed);
    }
}
