/**
 * The game surface {@link SurvivorBrain} uses: the current turn's own state and arena, and the intent setters.
 * Implemented by the live {@link AISurvivor} adapter (delegating to the Tank Royale Bot API) and by
 * {@link TraceReplay.ReplayPort} for exact offline replay.
 */
interface BotPort {
    int turnNumber();
    int roundNumber();
    int enemyCount();
    int arenaWidth();
    int arenaHeight();
    double x();
    double y();
    double direction();
    double speed();
    double energy();
    double gunHeat();
    double gunDirection();
    double radarDirection();

    double turnRate();
    double targetSpeed();
    double gunTurnRate();
    double radarTurnRate();

    void commandTurnRate(double rate);
    void commandTargetSpeed(double speed);
    void commandGunTurnRate(double rate);
    void commandRadarTurnRate(double rate);
    /** Same contract as BaseBot.setFire: false when energy is below the power or the gun is hot. */
    boolean commandFire(double power);
    void adjustGunForBodyTurn(boolean adjust);
    void adjustRadarForGunTurn(boolean adjust);
    void adjustRadarForBodyTurn(boolean adjust);
}
