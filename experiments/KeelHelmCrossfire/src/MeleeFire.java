final class MeleeFire {
    record Aim(MeleePolicy.Opponent target, double distance, double power, double error) {
        boolean canFire(int turn, int maxAge, double energy) {
            double halfWidth = Math.toDegrees(Math.atan(Phys.BOT_RADIUS / Math.max(36, distance)));
            return turn - target.turn <= maxAge && Math.abs(error) <= halfWidth * 0.9
                    && energy > power + 0.5;
        }
    }

    record Request(int fireTurn, int targetId, double power, double direction) {
        boolean matches(int observedTurn, double observedPower, double observedDirection) {
            return observedTurn == fireTurn && observedPower == power
                    && Math.abs(Phys.norm(observedDirection - direction)) <= 1e-6;
        }
    }

    private MeleeFire() {}

    static Aim aim(MeleePolicy.Opponent target, int turn, Phys.Pose self, double power, double distance,
                   double direction, double width, double height) {
        Phys.Pose predicted = MeleePolicy.intercept(target, turn, self, power, width, height);
        double bearing = Phys.bearing(self.x(), self.y(), predicted.x(), predicted.y());
        return new Aim(target, distance, power, Phys.norm(bearing - direction));
    }

    static Aim choose(MeleePolicy policy, Aim incumbent, int turn, Phys.Pose self, double energy,
                      double gunHeat, double direction, double width, double height) {
        if (gunHeat != 0) return null;
        if (incumbent.canFire(turn, 12, energy)) return incumbent;
        Aim best = null;
        for (MeleePolicy.Opponent target : policy.opponents()) {
            int age = turn - target.turn;
            if (target == incumbent.target() || age < 0 || age > 3) continue;
            double distance = Phys.distance(self.x(), self.y(), target.x, target.y);
            if (distance >= 500 || best != null && (distance > best.distance()
                    || distance == best.distance() && target.id > best.target().id)) continue;
            double power = MeleePolicy.power(distance, energy, target.energy);
            if (energy <= power + 0.5) continue;
            Aim aim = aim(target, turn, self, power, distance, direction, width, height);
            if (aim.canFire(turn, 3, energy)) best = aim;
        }
        return best;
    }
}
