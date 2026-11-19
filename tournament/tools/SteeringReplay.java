import com.google.gson.*;
import dev.robocode.tankroyale.server.model.*;
import java.nio.file.Path;
import java.util.*;

/** Shadow a single full-speed steering pulse. Physical rays and their labels are for scoring only. */
public class SteeringReplay extends SurvivalPulseReplay {
    static final String PRIMARY = "Diamond Port / 500+";
    record Label(String group, Double range) {}
    record Touch(Ray ray, int tick) {}
    record Trace(int id, int fire, double range, double damage, Integer originalHit, Integer candidateHit,
                 String disposition) {}
    static class Damage {
        int originalHits, candidateHits, delayedBeyondPulse, introduced;
        double original, candidate, unresolved, latent, truncation;
        double charged() { return candidate + unresolved + latent + truncation; }
    }
    static class Detail {
        int round, start, end, turnSign, pulseTicks, followTicks, rams, walls;
        double ramPenalty, wallPenalty;
        String stop;
        final Map<String,Damage> damage = new TreeMap<>();
        final List<Trace> diamondLong = new ArrayList<>();
    }
    final List<Detail> details = new ArrayList<>();
    final Map<Integer,Label> labels = new HashMap<>();
    final Map<Integer,Touch> baselineHits = new TreeMap<>(), steeringHits = new TreeMap<>();
    final Map<Integer,String> terminals = new HashMap<>();
    Phys.Pose previousPose;
    MeleePolicy.Command previousCommand;
    Detail detail;
    SteeringReplay(boolean score) { super(score); }

    @Override void onLaunch(Ray ray) {
        String shooter = names.get(ray.owner());
        Double range = previousPose == null ? null :
                Phys.distance(previousPose.x(), previousPose.y(), ray.x(), ray.y());
        String group = shooter;
        if ("Diamond Port".equals(shooter))
            group += range == null ? " / unknown" : range >= 500 ? " / 500+" : " / <500";
        labels.put(ray.id(), new Label(group, range));
    }

    @Override void onDecision(int t, JsonObject state, Phys.Pose p, MeleePolicy.Opponent target) {
        previousCommand = policy.move(t, p, number(state,"energy"), width, height);
        expected = Phys.step(p, previousCommand.turn(), previousCommand.speed(), width, height);
        if (t % 32 != 0 || !trigger.eligible(t,p,number(state,"turnRate"),width,height)) return;
        var nearest = trigger.scans.entrySet().stream()
                .filter(e -> t >= e.getValue().tick() && t - e.getValue().tick() <= 3)
                .min(Comparator.<Map.Entry<Integer,Scan>>comparingDouble(e ->
                        Phys.distance(p.x(),p.y(),e.getValue().x(),e.getValue().y()))
                        .thenComparingInt(Map.Entry::getKey)).orElseThrow().getValue();
        double cross = Math.signum(p.v()) * (Phys.cos(p.h()) * (p.y()-nearest.y())
                - Phys.sin(p.h()) * (p.x()-nearest.x()));
        if (window != null) throw new IllegalStateException("Overlapping windows");
        window = new Window(t,p);
        for (var ray : physical.values()) if (ray.owner() != own) window.active.put(ray.id(),ray);
        detail = new Detail();
        detail.round = round;
        detail.start = t;
        detail.turnSign = cross >= 0 ? 1 : -1;
        details.add(detail);
    }

    @Override void evaluate(JsonObject m,int t,Map<Integer,Set<Integer>> victims,Map<Integer,Set<Integer>> collisions,
                            Map<Integer,JsonObject> bots,Set<Integer> alive,boolean died) {
        var state = bots.get(own);
        if (state == null) { closeWindow(t-1,"absent",previousPose == null ? null : point(previousPose),physical.keySet()); return; }
        Point actual = new Point(number(state,"x"),number(state,"y"));
        if (expected == null || Phys.distance(expected.x(),expected.y(),actual.getX(),actual.getY()) > 1e-6
                || Math.abs(expected.v()-number(state,"speed")) > 1e-6
                || Math.abs(Phys.norm(expected.h()-number(state,"direction"))) > 1e-6)
            throw new IllegalStateException("Original movement mismatch: round " + round + " turn " + t);
        Set<Integer> predicted = new HashSet<>(), observed = new HashSet<>();
        for (var ray : physical.values()) {
            if (ray.owner() == own) continue;
            Line line = ray.segment(t);
            if (!outside(line.getStart()) && hit(line,actual,18)) {
                predicted.add(ray.id());
                baselineHits.putIfAbsent(ray.id(),new Touch(ray,t));
            }
        }
        victims.forEach((id,ids) -> { if (ids.contains(own)) observed.add(id); });
        if (!observed.equals(predicted)) throw new IllegalStateException("Original incoming IDs mismatch");

        int age = t - window.start;
        if (age <= 8) detail.pulseTicks++; else detail.followTicks++;
        if (score) {
            double turn = age <= 8 ? (age <= 4 ? 4 : -4) * detail.turnSign : previousCommand.turn();
            double speed = age <= 8 ? 8 * window.sign : previousCommand.speed();
            double attempted = Phys.speed(window.path.v(),speed);
            window.path = Phys.step(window.path,turn,speed,width,height);
            if (window.path.v() != attempted) {
                detail.walls++;
                detail.wallPenalty += Math.max(0, Math.abs(attempted)/2-1);
            }
            Point pos = point(window.path);
            Map<Integer,Point> others = new HashMap<>();
            bots.forEach((id,b) -> {
                if (id == own) return;
                Point other = new Point(number(b,"x"), number(b,"y"));
                others.put(id,other);
                if (Phys.distance(pos.getX(),pos.getY(),other.getX(),other.getY()) <= 36) {
                    detail.rams++;
                    detail.ramPenalty += .6;
                }
            });
            var before = Set.copyOf(window.active.keySet());
            for (var it = window.active.values().iterator(); it.hasNext();) {
                var ray = it.next();
                var segment = ray.segment(t);
                String terminal;
                if (outside(segment.getStart())) terminal = "arena exit";
                else if (hit(segment,pos,18)) {
                    steeringHits.putIfAbsent(ray.id(),new Touch(ray,t));
                    terminal = "contact";
                } else if (others.entrySet().stream().anyMatch(e -> e.getKey()!=ray.owner() && hit(segment,e.getValue(),18)))
                    terminal = "other body";
                else if (collisions.getOrDefault(ray.id(),Set.of()).stream().anyMatch(before::contains))
                    terminal = "enemy bullet collision";
                else continue;
                terminals.put(ray.id(),terminal);
                it.remove();
            }
        }
        String stop = detail.walls > 0 ? "wall" : detail.rams > 0 ? "ram" : died ? "death"
                : integer(state,"enemyCount") <= 1 ? "melee exit" : age == 24 ? "complete" : null;
        if (stop != null) closeWindow(t,stop,actual,alive);
    }

    void closeWindow(int t,String stop,Point original,Set<Integer> alive) {
        detail.end = t;
        detail.stop = stop;
        SortedSet<Integer> ids = new TreeSet<>(baselineHits.keySet());
        ids.addAll(steeringHits.keySet());
        if (score) ids.addAll(window.active.keySet());
        for (int id : ids) {
            Touch base = baselineHits.get(id), candidate = steeringHits.get(id);
            Ray ray = base != null ? base.ray() : candidate != null ? candidate.ray() : window.active.get(id);
            var label = labels.get(id);
            var d = detail.damage.computeIfAbsent(label.group(), k -> new Damage());
            double damage = ray.damage();
            if (base != null) { d.originalHits++; d.original += damage; }
            String disposition;
            if (!score) disposition = "control";
            else if (candidate != null) {
                d.candidateHits++;
                d.candidate += damage;
                if (base == null) d.introduced++;
                if (base != null && base.tick()-detail.start <= 8 && candidate.tick()-detail.start > 8)
                    d.delayedBeyondPulse++;
                disposition = base == null ? "introduced" : "both hit";
            } else if (terminals.containsKey(id)) disposition = terminals.get(id);
            else {
                var center = ray.point(t);
                double passed = (center.getX()-window.path.x())*Phys.cos(ray.direction())
                        + (center.getY()-window.path.y())*Phys.sin(ray.direction());
                if (base != null && passed <= 18) { d.unresolved += damage; disposition = "unresolved"; }
                else if (original != null && reachable(ray,t,point(window.path))
                        && !(alive.contains(id) && reachable(ray,t,original))) {
                    d.latent += damage; disposition = "latent exposure";
                } else if (base != null) disposition = "passed";
                else continue;
            }
            if (label.group().equals(PRIMARY)) detail.diamondLong.add(new Trace(id,ray.fire(),label.range(),damage,
                    base == null ? null : base.tick(),candidate == null ? null : candidate.tick(),disposition));
        }
        if (score && !stop.equals("complete"))
            for (var d : detail.damage.values()) d.truncation = Math.max(0,d.original-d.charged());
        baselineHits.clear(); steeringHits.clear(); terminals.clear();
        detail = null; window = null;
    }

    @Override void event(JsonObject msg) {
        super.event(msg);
        if (!string(msg,"type").equals("TickEventForObserver")) return;
        previousPose = null;
        for (var item : msg.getAsJsonArray("botStates")) {
            var b = item.getAsJsonObject();
            if (integer(b,"id") == own) previousPose = new Phys.Pose(number(b,"x"),number(b,"y"),
                    number(b,"direction"),number(b,"speed"));
        }
    }
    @Override void endRound() {
        if (window != null) closeWindow(previousTurn,"round end",previousPose == null ? null : point(previousPose),physical.keySet());
        labels.clear(); previousPose = null; previousCommand = null;
        super.endRound();
    }
    public static void main(String[] args) throws Exception {
        if (args.length != 2 || !Set.of("baseline","steer").contains(args[1])) throw new IllegalArgumentException();
        if (!sha(Path.of(MathKt.class.getProtectionDomain().getCodeSource().getLocation().toURI()))
                .equals("16c277795bd823c9eda0b650f1d09a418ff35cfe43d59b820c52d5e6816f905c"))
            throw new IllegalStateException("Server hash");
        var readers = readGames(Path.of(args[0]),() -> new SteeringReplay(args[1].equals("steer")));
        var output = new TreeMap<String,List<Detail>>();
        readers.forEach((game,r) -> output.put(game,r.details));
        System.out.println(new GsonBuilder().serializeNulls().create().toJson(output));
    }
}
