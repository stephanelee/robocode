import com.google.gson.*;
import java.nio.file.Path;
import java.util.*;

/** Offline trajectories only. Legal scans and incumbent choice come from CrossfireReplay. */
public final class CooldownShadow extends CrossfireReplay {
    record Decision(double turn, MeleeFire.Aim shot) {
        double cost() { return shot == null ? 0 : shot.power(); }
    }
    static final class Gun {
        double heat, ray;
        int commitment = -1;
        Gun(double heat, double ray) { this.heat = heat; this.ray = ray; }
        void advance(Decision decision, double cooling) {
            ray = (ray + Phys.clip(decision.turn, -20, 20) + 360) % 360;
            heat = decision.shot == null ? Math.max(0, heat - cooling) : Phys.gunHeat(decision.cost());
        }
    }
    static final class Stats {
        int matchedTurns, same, redirected, created, displaced, neither, missingNext, commitments, cancelled;
        final Map<String,Integer> rejects = new TreeMap<>();
        final Map<String,Integer> intendedPair = new TreeMap<>();
        void reject(String reason) { rejects.merge(reason, 1, Integer::sum); }
    }

    final Stats stats = new Stats();
    final Map<Integer,String> names = new HashMap<>();
    Gun base, proposal;
    double cooling, baseMinusShadowCost;
    Decision pendingBase, pendingShadow;

    CooldownShadow() { super(false); }

    static MeleeFire.Aim aim(MeleePolicy.Opponent target, int t, Phys.Pose self, double energy,
                             double ray, int w, int h) {
        double d = Phys.distance(self.x(), self.y(), target.x, target.y);
        return MeleeFire.aim(target, t, self, MeleePolicy.power(d, energy, target.energy), d, ray, w, h);
    }

    static boolean reachable(MeleeFire.Aim a, int ticks) {
        double tolerance = 0.9 * Math.toDegrees(Math.atan(Phys.BOT_RADIUS / Math.max(36, a.distance())));
        return Math.abs(a.error()) <= ticks * 20.0 + tolerance;
    }

    Decision decideGun(Gun gun, boolean retarget, MeleePolicy.Opponent target, int t, Phys.Pose self, double energy) {
        if (target == null || energy <= 0) {
            if (gun.commitment >= 0) stats.cancelled++;
            gun.commitment = -1;
            return new Decision(0, null);
        }
        var incumbent = aim(target, t, self, energy, gun.ray, width, height);
        if (gun.heat == 0) {
            gun.commitment = -1;
            return new Decision(incumbent.error(), MeleeFire.choose(policy, incumbent, t, self, energy,
                    0, gun.ray, width, height));
        }
        if (!retarget) return new Decision(incumbent.error(), null);
        if (gun.commitment >= 0) {
            for (var o : policy.opponents()) {
                if (o.id != gun.commitment || o == target || t - o.turn > 12) continue;
                var a = aim(o, t, self, energy, gun.ray, width, height);
                if (a.distance() < 500 && energy > a.power() + 0.5) return new Decision(a.error(), null);
            }
            stats.cancelled++;
            gun.commitment = -1;
        }
        int ticks = 0;
        for (double heat = gun.heat; heat > 0; heat = Math.max(0, heat - cooling)) ticks++;
        if (t - target.turn + ticks <= 12 && energy > incumbent.power() + 0.5 && reachable(incumbent, ticks)) {
            stats.reject("incumbent protected (turns)");
            return new Decision(incumbent.error(), null);
        }
        MeleeFire.Aim best = null;
        for (var o : policy.opponents()) {
            if (o == target) continue;
            if (t - o.turn < 0 || t - o.turn > 3) { stats.reject("scan age (opponents)"); continue; }
            var a = aim(o, t, self, energy, gun.ray, width, height);
            if (a.distance() >= 500) { stats.reject("range (opponents)"); continue; }
            if (energy <= a.power() + 0.5) { stats.reject("energy (opponents)"); continue; }
            if (!reachable(a, ticks)) { stats.reject("turn budget (opponents)"); continue; }
            if (best == null || a.distance() < best.distance()
                    || a.distance() == best.distance() && o.id < best.target().id) best = a;
        }
        if (best == null) return new Decision(incumbent.error(), null);
        gun.commitment = best.target().id;
        stats.commitments++;
        return new Decision(best.error(), null);
    }

    @Override void onDecision(int turn, JsonObject state, Phys.Pose self, MeleePolicy.Opponent target) {
        double heat = number(state, "gunHeat"), ray = number(state, "gunDirection"), energy = number(state, "energy");
        if (base == null) {
            base = new Gun(heat, ray);
            proposal = new Gun(heat, ray);
        }
        if (Math.abs(Phys.norm(base.ray - ray)) > 1e-6 || Math.abs(base.heat - heat) > 1e-9)
            throw new IllegalStateException("Baseline gun-state calibration failed round " + round + " turn " + turn
                    + " expected heat/ray " + base.heat + "/" + base.ray + " observed " + heat + "/" + ray);
        pendingBase = decideGun(base, false, target, turn, self, energy);
        double shadowEnergy = energy + Math.min(0, baseMinusShadowCost);
        pendingShadow = decideGun(proposal, true, target, turn, self, shadowEnergy);
        base.advance(pendingBase, cooling);
        proposal.advance(pendingShadow, cooling);
        baseMinusShadowCost += pendingBase.cost() - pendingShadow.cost();
    }

    @Override MeleeFire.Aim decide(int turn, JsonObject state, Counts c) {
        var actual = super.decide(turn, state, c);
        var b = pendingBase.shot;
        if ((actual == null) != (b == null) || b != null && (b.target() != actual.target() || b.power() != actual.power()))
            throw new IllegalStateException("Baseline shot disagreement");
        return actual;
    }

    String intended(Decision d) { return d.shot == null ? "none" : names.get(d.shot.target().id); }

    void match() {
        String b = intended(pendingBase), s = intended(pendingShadow);
        stats.matchedTurns++;
        if (pendingBase.shot == null && pendingShadow.shot == null) stats.neither++;
        else {
            stats.intendedPair.merge(b + " -> " + s, 1, Integer::sum);
            if (pendingBase.shot == null) stats.created++;
            else if (pendingShadow.shot == null) stats.displaced++;
            else if (b.equals(s)) stats.same++;
            else stats.redirected++;
        }
        pendingBase = pendingShadow = null;
    }

    @Override void event(JsonObject msg) {
        String type = string(msg, "type");
        if (type.equals("GameStartedEventForObserver")) {
            cooling = number(msg.getAsJsonObject("gameSetup"), "gunCoolingRate");
            if (!(cooling > 0)) throw new IllegalStateException("Missing cooling rate");
            for (var item : msg.getAsJsonArray("participants")) {
                var p = item.getAsJsonObject();
                names.put(integer(p, "id"), string(p, "name"));
            }
        }
        if (type.equals("TickEventForObserver") && integer(msg, "roundNumber") == round && pendingBase != null) {
            if (integer(msg, "turnNumber") != previousTurn + 1) throw new IllegalStateException("Missing tick");
            match();
        }
        super.event(msg);
    }

    @Override void endRound() {
        if (pendingBase != null) stats.missingNext++;
        pendingBase = pendingShadow = null;
        base = proposal = null;
        baseMinusShadowCost = 0;
        super.endRound();
    }

    public static void main(String[] args) throws Exception {
        var games = readGames(Path.of(args[0]), CooldownShadow::new);
        var results = new TreeMap<String,Stats>();
        int matched = 0;
        for (var e : games.entrySet()) {
            results.put(e.getKey(), e.getValue().stats);
            for (var counts : e.getValue().rounds.values()) matched += counts.matchedPhysical;
        }
        if (matched != 4042) throw new IllegalStateException("Incomplete physical calibration");
        System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(results));
    }
}
