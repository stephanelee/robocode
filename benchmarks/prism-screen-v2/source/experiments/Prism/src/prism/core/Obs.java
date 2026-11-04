package prism.core;

import java.util.List;

/**
 * Immutable observations produced by the engine boundary. Angles are native radians. Every event
 * carries the engine turn on which it occurred; nothing here is inferred.
 */
public final class Obs {
    private Obs() {}

    public sealed interface Event permits Scan, BulletFired, BulletHit, HitByBullet, BulletHitBullet,
            BulletHitWall, HitWall, HitBot, BotDeath, Death, Won, Skipped {
        int turn();
        /** Identity used for duplicate-callback suppression. */
        String key();
    }

    public record Scan(int turn, int botId, double energy, double x, double y, double heading, double speed) implements Event {
        public String key() { return "scan:" + turn + ":" + botId; }
    }

    /** Our own bullet confirmed created by the engine. */
    public record BulletFired(int turn, int bulletId, double power, double x, double y, double heading) implements Event {
        public String key() { return "fired:" + bulletId; }
    }

    /** Our bullet hit a bot. */
    public record BulletHit(int turn, int bulletId, int victimId, double damage, double victimEnergy, double power) implements Event {
        public String key() { return "hit:" + bulletId; }
    }

    /** An enemy bullet hit us; bullet data is delivered by the engine on impact. */
    public record HitByBullet(int turn, int bulletId, int ownerId, double power, double x, double y, double heading, double damage, double energy) implements Event {
        public String key() { return "hitby:" + ownerId + ":" + bulletId; }
    }

    public record BulletHitBullet(int turn, int ownBulletId, int otherBulletId, int otherOwnerId, double otherPower,
                                  double otherX, double otherY, double otherHeading) implements Event {
        public String key() { return "hbb:" + ownBulletId + ":" + otherOwnerId + ":" + otherBulletId; }
    }

    public record BulletHitWall(int turn, int bulletId) implements Event {
        public String key() { return "hbw:" + bulletId; }
    }

    public record HitWall(int turn) implements Event {
        public String key() { return "wall:" + turn; }
    }

    public record HitBot(int turn, int victimId, double victimEnergy, double x, double y, boolean rammed) implements Event {
        public String key() { return "hitbot:" + turn + ":" + victimId; }
    }

    public record BotDeath(int turn, int botId) implements Event {
        public String key() { return "death:" + botId; }
    }

    public record Death(int turn) implements Event {
        public String key() { return "selfdeath"; }
    }

    public record Won(int turn) implements Event {
        public String key() { return "won"; }
    }

    public record Skipped(int turn) implements Event {
        public String key() { return "skip:" + turn; }
    }

    /** Own state as reported for a tick; headings in native radians. */
    public record Own(int turn, double x, double y, double heading, double gunHeading, double radarHeading,
                      double speed, double energy, double gunHeat, int enemyCount) {}

    /** One turn's input batch, in callback order as delivered. */
    public record Turn(int round, Own own, List<Event> events) {}
}
