import com.google.gson.*;
import java.io.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import java.util.zip.GZIPInputStream;
import java.util.regex.Pattern;

/** Replay the measured JAR's selector over recorded own state and own radar scans. */
public class CrossfireReplay {
    static int integer(JsonObject o, String key) { return o.get(key).getAsInt(); }
    static double number(JsonObject o, String key) { return o.get(key).getAsDouble(); }
    static String string(JsonObject o, String key) { return o.get(key).getAsString(); }

    static class Counts {
        int melee, cool, hasTarget, incumbentReady, alternateWindow;
        int otherKnown, otherFresh, otherInRange, otherAffordable, alternateReady;
        int matchedPhysical, predictedOnly, physicalOnly, unobservedNextTick;
        final List<Double> closestAlternateErrorDegrees = new ArrayList<>();
    }

    final boolean delayScans;
    CrossfireReplay(boolean delayScans) { this.delayScans = delayScans; }

    void observe(JsonObject e) {
        policy.observe(integer(e, "scannedBotId"), integer(e, "turnNumber"), number(e,"x"), number(e,"y"),
                number(e,"direction"), number(e,"speed"), number(e,"energy"));
    }

    final MeleePolicy policy = new MeleePolicy();
    final Map<Integer, Counts> rounds = new TreeMap<>();
    int round, own, width, height, previousTurn;
    boolean previousMelee, dead;
    MeleeFire.Aim previousShot;
    double previousRay;

    void event(JsonObject msg) {
        String type = string(msg, "type");
        if (type.equals("GameStartedEventForObserver")) {
            var setup = msg.getAsJsonObject("gameSetup");
            width = integer(setup, "arenaWidth");
            height = integer(setup, "arenaHeight");
            own = msg.getAsJsonArray("participants").asList().stream()
                    .map(JsonElement::getAsJsonObject).filter(p -> string(p, "name").equals("KeelHelm Crossfire"))
                    .mapToInt(p -> integer(p, "id")).findFirst().orElseThrow();
        }
        if (!type.equals("TickEventForObserver")) return;
        int r = integer(msg, "roundNumber"), t = integer(msg, "turnNumber");
        if (r != round) {
            endRound();
            round = r;
            rounds.put(r, new Counts());
            policy.newRound(0);
            dead = false;
            previousTurn = 0;
        }
        if (t != previousTurn + 1) throw new IllegalStateException("Nonconsecutive ticks");
        previousTurn = t;
        Counts counts = rounds.get(r);
        var events = msg.getAsJsonArray("events");
        List<JsonObject> deferred = new ArrayList<>();
        JsonObject fired = null;
        for (var item : events) {
            var e = item.getAsJsonObject();
            switch (string(e, "type")) {
                case "ScannedBotEvent" -> {
                    if (integer(e, "scannedByBotId") == own) {
                        if (delayScans) deferred.add(e);
                        else observe(e);
                    }
                }
                case "BotDeathEvent" -> {
                    int victim = integer(e, "victimId");
                    policy.died(victim);
                    if (victim == own) dead = true;
                }
                case "BulletFiredEvent" -> {
                    var b = e.getAsJsonObject("bullet");
                    if (integer(b, "ownerId") == own) fired = b;
                }
            }
        }
        if (previousMelee) {
            boolean matched = previousShot != null && fired != null
                    && Math.abs(previousShot.power() - number(fired, "power")) < 1e-9
                    && Math.abs(Phys.norm(previousRay - number(fired, "direction"))) <= 1e-6;
            if (matched) counts.matchedPhysical++;
            else {
                if (previousShot != null) counts.predictedOnly++;
                if (fired != null) counts.physicalOnly++;
            }
        }
        previousMelee = false;
        previousShot = null;
        for (var item : msg.getAsJsonArray("botStates")) {
            var state = item.getAsJsonObject();
            if (!dead && integer(state,"id") == own && integer(state,"enemyCount") > 1) {
                previousMelee = true;
                previousRay = number(state, "gunDirection");
                previousShot = decide(t, state, counts);
            }
        }
        // These scans first affect the next decision and retain their original scan timestamps.
        deferred.forEach(this::observe);
    }

    void endRound() {
        if (previousMelee && previousShot != null) rounds.get(round).unobservedNextTick++;
        previousMelee = false;
        previousShot = null;
    }

    MeleeFire.Aim decide(int turn, JsonObject state, Counts c) {
        c.melee++;
        var self = new Phys.Pose(number(state,"x"), number(state,"y"), number(state,"direction"), number(state,"speed"));
        var target = policy.target(turn, self); // Updates incumbent stickiness even while the gun cools.
        if (number(state,"gunHeat") != 0) return null;
        c.cool++;
        if (target == null) return null;
        c.hasTarget++;
        double energy = number(state,"energy"), ray = number(state,"gunDirection");
        double range = Phys.distance(self.x(), self.y(), target.x, target.y);
        double power = MeleePolicy.power(range, energy, target.energy);
        var incumbent = MeleeFire.aim(target, turn, self, power, range, ray, width, height);
        var selected = MeleeFire.choose(policy, incumbent, turn, self, energy, 0, ray, width, height);
        if (selected == incumbent) {
            c.incumbentReady++;
            return selected;
        }
        c.alternateWindow++;
        // Count each turn at most once per nested gate, even if two alternates qualify.
        int stage = 0;
        double closestError = Double.POSITIVE_INFINITY;
        for (var other : policy.opponents()) {
            if (other == target) continue;
            stage = Math.max(stage, 1);
            int age = turn - other.turn;
            if (age < 0 || age > 3) continue;
            stage = Math.max(stage, 2);
            double distance = Phys.distance(self.x(), self.y(), other.x, other.y);
            if (distance >= 500) continue;
            stage = Math.max(stage, 3);
            double otherPower = MeleePolicy.power(distance, energy, other.energy);
            if (energy <= otherPower + 0.5) continue;
            stage = 4;
            var otherAim = MeleeFire.aim(other, turn, self, otherPower, distance, ray, width, height);
            closestError = Math.min(closestError, Math.abs(otherAim.error()));
        }
        if (stage >= 1) c.otherKnown++;
        if (stage >= 2) c.otherFresh++;
        if (stage >= 3) c.otherInRange++;
        if (stage >= 4) {
            c.otherAffordable++;
            c.closestAlternateErrorDegrees.add(closestError);
        }
        if (selected != null) c.alternateReady++;
        return selected;
    }

    static String sha(Path file) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file)));
    }

    void verifyTelemetry(Path metrics, String expectedHash) throws Exception {
        if (!sha(metrics).equals(expectedHash)) throw new IOException("Telemetry hash mismatch " + metrics);
        var stats = Pattern.compile("(\\d+) stats .* meleeTurns=(\\d+) meleeShots=(\\d+) .*");
        var crossfire = Pattern.compile("(\\d+) crossfire requests=(\\d+) .*");
        Set<Integer> seenStats = new HashSet<>(), seenRequests = new HashSet<>();
        for (var line : Files.readAllLines(metrics)) {
            var m = stats.matcher(line);
            if (m.matches()) {
                int r = Integer.parseInt(m.group(1));
                Counts c = rounds.get(r);
                if (!seenStats.add(r) || c.melee != Integer.parseInt(m.group(2))
                        || c.matchedPhysical + c.physicalOnly != Integer.parseInt(m.group(3))
                        || (!delayScans && c.predictedOnly + c.physicalOnly + c.unobservedNextTick != 0))
                    throw new IOException("Melee decision or physical shot mismatch " + metrics + " round " + r);
            }
            m = crossfire.matcher(line);
            if (m.matches()) {
                int r = Integer.parseInt(m.group(1));
                if (!seenRequests.add(r) || (!delayScans && rounds.get(r).alternateReady != Integer.parseInt(m.group(2))))
                    throw new IOException("Alternate request mismatch " + metrics + " round " + r);
            }
        }
        if (!rounds.keySet().equals(seenStats) || !rounds.keySet().equals(seenRequests))
            throw new IOException("Missing telemetry " + metrics);
    }

    public static void main(String[] args) throws Exception {
        boolean delayScans = args.length == 2 && args[1].equals("--previous-tick-scans");
        if (args.length != 1 && !delayScans)
            throw new IllegalArgumentException("Pass the evidence directory and optionally --previous-tick-scans");
        Path evidence = Path.of(args[0]);
        var audit = JsonParser.parseString(Files.readString(Path.of(
                "benchmarks/keelhelm-crossfire/stopped-attempt/independent-outcome-review.json"))).getAsJsonObject();
        Path loadedJar = Path.of(MeleeFire.class.getProtectionDomain().getCodeSource().getLocation().toURI());
        if (!sha(loadedJar).equals(string(audit, "candidateJarSHA256"))) throw new IOException("Candidate JAR mismatch");
        var output = new TreeMap<String, Map<Integer, Counts>>();
        for (var item : audit.getAsJsonArray("games")) {
            var entry = item.getAsJsonObject();
            String id = string(entry, "game");
            if (!id.endsWith("-candidate")) continue;
            Path dir = evidence.resolve("games").resolve(id).resolve("recordings");
            Path recording;
            try (var files = Files.list(dir)) {
                var matches = files.filter(p -> p.toString().endsWith(".battle.gz")).toList();
                if (matches.size() != 1) throw new IOException("Expected one recording for " + id);
                recording = matches.get(0);
            }
            String digest = sha(recording);
            if (!digest.equals(string(entry, "recordingSHA256"))) throw new IOException("Recording mismatch " + id);
            var replay = new CrossfireReplay(delayScans);
            try (var reader = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(recording)),
                    java.nio.charset.StandardCharsets.UTF_8))) {
                for (String line; (line = reader.readLine()) != null;)
                    replay.event(JsonParser.parseString(line).getAsJsonObject());
            }
            replay.endRound();
            if (replay.rounds.size() != integer(entry, "roundsVerified")) throw new IOException("Round count " + id);
            replay.verifyTelemetry(dir.resolveSibling("helm-metrics.txt"), string(entry, "telemetrySHA256"));
            output.put(id, replay.rounds);
        }
        if (output.size() != 12) throw new IOException("Incomplete candidate prefix");
        System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(output));
    }
}
