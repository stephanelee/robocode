/**
 * One enemy bullet inferred from a legal scan energy drop.
 * Created in turn {@code fireTurn} from the enemy's scanned position one turn earlier, aimed (head-on model)
 * at our position predicted one turn ahead from the scan two turns before creation (position + velocity along heading). Only legal scans and own state are used.
 */
final class Shot {
    final int fireTurn;
    final double power, speed, ox, oy, tx, ty;
    double heading;
    /** Our actual position one turn before creation: the reference the enemy's wave label is measured from. */
    double rx, ry;
    /** Where the shooter's classic-order predictor placed itself when aiming; the bullet keeps that angle from the true origin. */
    double psx, psy;
    final double range;
    boolean resolved, shielded, dodge;
    /** Side (+1/-1 along the line normal) one candidate angle behind our motion at aim time; 0 if unknown. */
    double behind;
    int shieldTurn = -1, shieldBulletId = -1;
    String outcome = "pending";

    Shot(int fireTurn, double power, double ox, double oy, double tx, double ty) {
        this.fireTurn = fireTurn; this.power = power; this.speed = Physics.bulletSpeed(power);
        this.ox = ox; this.oy = oy; this.tx = tx; this.ty = ty;
        this.heading = Physics.bearing(ox, oy, tx, ty);
        this.range = Math.hypot(tx - ox, ty - oy);
        this.rx = tx; this.ry = ty; this.psx = ox; this.psy = oy;
    }

    void predictedSource(double x, double y) { psx = x; psy = y; heading = Physics.bearing(x, y, tx, ty); }

    /** Distance travelled from the origin at the end of {@code turn}. */
    double travelled(int turn) { return speed * (turn - fireTurn + 1); }

    /** First turn whose checked segment reaches within a bot radius of the aim point. */
    int contactTurn() {
        for (int turn = fireTurn; ; turn++) if (travelled(turn) + speed >= range - Physics.BOT_RADIUS) return turn;
    }

    /** First state turn whose wave ring [travelled(t), travelled(t+1)] reaches the bot footprint (22 px proxy). */
    int breakFirst() { for (int turn = fireTurn; ; turn++) if (travelled(turn + 1) >= range - 22) return turn; }
    /** Last state turn whose ring still overlaps the footprint. */
    int breakLast() { for (int turn = fireTurn; ; turn++) if (travelled(turn + 1) > range + 22) return turn; }

    /** Last turn the wave front can still overlap a bot centred at the aim point. */
    int passTurn() {
        for (int turn = fireTurn; ; turn++) if (travelled(turn) > range + Physics.BOT_RADIUS) return turn;
    }
}
