package prism;

import prism.core.Angles;
import prism.core.Command;
import prism.core.Drive;
import prism.core.Escape;
import prism.core.Geometry;
import prism.core.Obs;
import prism.core.Physics;
import prism.core.Rules;
import prism.gun.Firing;
import prism.gun.Guns;
import prism.learn.Features;
import prism.learn.NeighborStore;
import prism.move.Danger;
import prism.move.DuelPlanner;
import prism.move.MeleePlanner;
import prism.move.Shadows;
import prism.radar.Radar;
import prism.track.Enemy;
import prism.track.FireInference;
import prism.track.History;
import prism.wave.Wave;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * All decision logic behind a minimal port. One call to {@link #turn} is one transaction
 * (specification section 3.3): events → lifecycle/own bullets → scans and energy ledger →
 * wave crossings and labels → movement → validate prior firing plan → next aim and radar.
 * The class performs no API calls, so the same event stream, seed and rules yield the same commands.
 */
public final class Brain {
    private final Rules rules;
    private final Telemetry tel;
    private final long seed;
    private final Guns guns;
    private final DuelPlanner duel;
    private final MeleePlanner melee;
    private final Radar radar = new Radar();
    private final Map<Integer, Enemy> enemies = new LinkedHashMap<>();
    private final Map<Integer, Danger> danger = new LinkedHashMap<>();
    private final Enemy self;
    private final List<Wave> waves = new ArrayList<>();
    private final Map<Integer, Shadows.Bullet> bullets = new LinkedHashMap<>();
    private final Set<String> seen = new HashSet<>();
    private final Map<String, Integer> failures = new LinkedHashMap<>();
    private final Set<String> disabled = new HashSet<>();
    private final SplittableRandom rng;

    private int round = -1;
    private long waveIds = 1;
    private long sequence = 0;
    private long stateVersion = 0;
    private Firing.Plan plan;
    private Firing.Pending pending;
    private DuelPlanner.Action lastAction = DuelPlanner.Action.POS;
    private boolean meleeOrigin;
    private boolean dead;
    private int lastOwnWallTurn = Integer.MIN_VALUE;
    private int lastTurn = -1;
    private Command lastCommand;

    public Brain(Rules rules, long seed, Telemetry tel) {
        this.rules = rules;
        this.seed = seed;
        this.tel = tel;
        this.guns = new Guns(rules);
        this.duel = new DuelPlanner(rules);
        SplittableRandom root = new SplittableRandom(seed);
        this.melee = new MeleePlanner(rules, root.split().nextLong());
        this.rng = root.split();
        this.self = new Enemy(-1, rules);
        this.meleeOrigin = rules.initialEnemies() > 1;
    }

    // ------------------------------------------------------------------ lifecycle

    /** Round reset: clears waves, shots, predictions, trajectories, locks, timers; keeps learning. */
    public void startRound(int roundNumber) {
        round = roundNumber;
        waves.clear();
        bullets.clear();
        seen.clear();
        plan = null;
        pending = null;
        disabled.clear();
        failures.clear();
        radar.resetRound();
        melee.resetRound();
        self.resetRound();
        for (Enemy e : enemies.values()) e.resetRound();
        dead = false;
        lastOwnWallTurn = Integer.MIN_VALUE;
        lastTurn = -1;
        lastAction = DuelPlanner.Action.POS;
        tel.inc("rounds");
    }

    public Rules rules() { return rules; }
    public Guns guns() { return guns; }
    public Map<Integer, Enemy> enemies() { return enemies; }
    public Map<Integer, Danger> dangers() { return danger; }
    public List<Wave> waves() { return waves; }
    public Firing.Plan plan() { return plan; }
    public Firing.Pending pending() { return pending; }
    public Map<Integer, Shadows.Bullet> bullets() { return bullets; }
    public Telemetry telemetry() { return tel; }

    private Enemy enemy(int id) { return enemies.computeIfAbsent(id, k -> new Enemy(k, rules)); }
    private Danger dangerOf(int id) { return danger.computeIfAbsent(id, k -> new Danger()); }

    /** The engine refused a fire request at submission (API-side check). */
    public void fireRejected() { pending = null; tel.inc("fire.rejectedAtSubmit"); }

    // ------------------------------------------------------------------ the transaction

    public Command turn(Obs.Turn in) {
        long t0 = System.nanoTime();
        Obs.Own own = in.own();
        int turn = own.turn();
        if (in.round() != round) startRound(in.round());
        if (lastTurn >= 0 && turn > lastTurn + 1) tel.add("turns.gap", turn - lastTurn - 1);
        lastTurn = turn;
        stateVersion++;

        // 1. collect and deduplicate
        List<Obs.Event> events = new ArrayList<>();
        for (Obs.Event e : in.events()) {
            if (seen.add(e.key())) events.add(e); else tel.inc("events.duplicate");
        }
        if (seen.size() > 100_000) seen.clear();

        // own observation
        self.observe(turn, own.x(), own.y(), own.heading(), own.speed(), own.energy());
        melee.record(turn, own.x(), own.y());

        // 2. lifecycle and own bullets
        boolean confirmedThisTurn = false;
        for (Obs.Event e : events) {
            if (e instanceof Obs.Death) { dead = true; tel.inc("round.death"); }
            else if (e instanceof Obs.Won) tel.inc("round.won");
            else if (e instanceof Obs.Skipped) tel.inc("turns.skipped");
            else if (e instanceof Obs.BotDeath bd) onBotDeath(bd);
            else if (e instanceof Obs.BulletFired bf) confirmedThisTurn |= onFired(bf);
            else if (e instanceof Obs.BulletHit bh) onBulletHit(bh);
            else if (e instanceof Obs.BulletHitWall bw) killBullet(bw.bulletId(), bw.turn());
            else if (e instanceof Obs.HitWall hw) { lastOwnWallTurn = hw.turn(); tel.inc("collision.wall"); }
            else if (e instanceof Obs.HitBot hb) onHitBot(hb);
        }
        if (pending != null && turn >= pending.expectedConfirmTurn() && !confirmedThisTurn) {
            tel.inc("fire.unconfirmedCancelled");
            pending = null;
        }
        for (Obs.Event e : events) {
            if (e instanceof Obs.HitByBullet hb) onHitByBullet(hb, turn);
            else if (e instanceof Obs.BulletHitBullet bb) onBulletHitBullet(bb);
        }

        // 3. scans and energy ledger (after all known effects are registered)
        for (Obs.Event e : events) if (e instanceof Obs.Scan s) guard("scan", () -> onScan(s, own));
        for (Enemy en : enemies.values()) if (en.alive && en.scanAge(turn) == 0) en.turnsObservedTogether++;

        // 4. wave crossings and labels
        guard("waves", () -> resolveWaves(turn, own));
        if (turn % 100 == 0) enforceMemoryCap();

        if (dead) { lastCommand = Command.idle(++sequence); return lastCommand; }

        // 5. movement
        Physics.Motion me = new Physics.Motion(own.x(), own.y(), own.heading(), own.speed());
        Drive.Control move = Drive.stop();
        String reason = "none";
        Physics.Motion next = me;
        long tm = System.nanoTime();
        try {
            if (own.enemyCount() > 1) {
                MeleePlanner.Result r = melee.plan(turn, me, own.energy(), opponents(turn));
                move = r.control(); reason = r.reason();
            } else {
                DuelPlanner.Result r = planDuel(turn, me, own);
                move = r.control(); reason = r.reason();
                if (r.truncated()) tel.inc("move.budgetTruncated");
                if (r.action() != DuelPlanner.Action.STOP || reason.equals("orbit")) lastAction = r.action();
                tel.inc("move.action." + r.action());
            }
            succeed("move");
        } catch (RuntimeException ex) {
            fail("move", ex);
            DuelPlanner.Result r = duel.plan(me, turn, rules.width() / 2, rules.height() / 2, 0, 0, List.of(), lastAction, lastOwnWallTurn == turn);
            move = r.control(); reason = "fallback";
        }
        tel.time("move", System.nanoTime() - tm);
        Physics.Step ns = Physics.step(me, move.targetSpeed(), move.turnRate(), rules, lastOwnWallTurn == turn);
        next = ns.next();
        if (ns.hitWall()) tel.inc("move.predictedWallStep");
        tel.inc("move.reason." + reason);

        // 6. validate prior plan and fire
        double fire = 0;
        long tg = System.nanoTime();
        Firing.Verdict v = Firing.Verdict.NO_PLAN;
        if (plan != null) {
            Enemy target = enemies.get(plan.targetId());
            int scanTurn = target == null || target.latest() == null ? Integer.MIN_VALUE : target.latest().turn();
            v = Firing.validate(plan, turn, own.gunHeading(), own.gunHeat(), own.energy(), scanTurn,
                    target != null && target.alive, own.x(), own.y());
            tel.inc("fire.verdict." + v);
            if (v == Firing.Verdict.FIRE && pending == null) {
                fire = plan.power();
                pending = new Firing.Pending(turn, turn + 1, fire, own.gunHeading(), own.x(), own.y(), plan.targetId(), -1);
                tel.inc("fire.requested");
            }
            if (v != Firing.Verdict.HEADING && v != Firing.Verdict.HOT) plan = null;
        }

        // 7. virtual wave for this tick, next aim, radar
        double gunTurn = 0;
        try {
            gunTurn = aim(turn, own, next, fire > 0);
            succeed("gun");
        } catch (RuntimeException ex) {
            fail("gun", ex);
            plan = null;
        }
        tel.time("gun", System.nanoTime() - tg);
        double radarTurn;
        try {
            radarTurn = radar.command(turn, own.radarHeading(), next.x(), next.y(), own.enemyCount(), known());
        } catch (RuntimeException ex) {
            fail("radar", ex);
            radarTurn = Math.toRadians(45);
        }
        long elapsed = System.nanoTime() - t0;
        tel.time("turn", elapsed);
        Command c = new Command(++sequence, move.targetSpeed(), move.turnRate(), gunTurn, radarTurn, fire, reason);
        lastCommand = c;
        return c;
    }

    // ------------------------------------------------------------------ events

    private boolean onFired(Obs.BulletFired bf) {
        tel.inc("fire.confirmed");
        bullets.put(bf.bulletId(), new Shadows.Bullet(bf.bulletId(), bf.turn(), bf.x(), bf.y(), bf.heading(), bf.power(), Integer.MAX_VALUE));
        // bind to the wave frozen for this exact pending request: same id, fire turn, origin and power
        Wave match = null;
        if (pending != null && pending.waveId() >= 0) {
            for (Wave w : waves) {
                if (w.id != pending.waveId()) continue;
                if (w.role == Wave.Role.VIRTUAL && w.fireTurnMin == bf.turn() && Math.hypot(w.ox - bf.x(), w.oy - bf.y()) < 1e-6
                        && Math.abs(w.power - bf.power()) < 1e-9) match = w;
                break;
            }
        }
        if (match != null) { match.role = Wave.Role.OUTGOING; match.physical = true; match.bulletId = bf.bulletId(); }
        else tel.inc("fire.confirmedWithoutWave"); // the bullet still casts shadows; no physical wave is invented
        if (pending == null) tel.inc("fire.confirmedWithoutRequest");
        pending = null;
        return true;
    }

    private void onBulletHit(Obs.BulletHit bh) {
        killBullet(bh.bulletId(), bh.turn());
        Enemy e = enemy(bh.victimId());
        e.damageFromUs += bh.damage();
        e.fire.addEffect(new FireInference.Effect(bh.turn() + 1, -bh.damage(), 1e-9, "ourHit"));
        tel.inc("fire.hit");
        tel.add("damage.dealtMilli", Math.round(bh.damage() * 1000));
    }

    private void killBullet(int id, int turn) {
        Shadows.Bullet b = bullets.get(id);
        if (b != null) bullets.put(id, new Shadows.Bullet(b.id(), b.fireTurn(), b.x(), b.y(), b.heading(), b.power(), turn));
    }

    private void onHitBot(Obs.HitBot hb) {
        tel.inc("collision.bot");
        Enemy e = enemy(hb.victimId());
        e.fire.addEffect(new FireInference.Effect(hb.turn(), -Rules.RAM_DAMAGE, 1e-9, "ram"));
    }

    private void onBotDeath(Obs.BotDeath bd) {
        Enemy e = enemy(bd.botId());
        e.alive = false;
        e.deathTurn = Math.max(e.deathTurn, bd.turn()); // round-local tombstone
        for (Wave w : waves) if (w.targetId == bd.botId() && !w.isThreat() && !w.labelEmitted) { w.cancelled = true; w.cancelReason = "target death"; }
        if (plan != null && plan.targetId() == bd.botId()) plan = null;
    }

    private void onHitByBullet(Obs.HitByBullet hb, int turn) {
        Enemy e = enemy(hb.ownerId());
        e.damageToUs += hb.damage();
        e.fire.addEffect(new FireInference.Effect(hb.turn() + 1, Rules.hitBonus(hb.power()), 1e-9, "enemyHit"));
        tel.add("damage.takenMilli", Math.round(hb.damage() * 1000));
        tel.inc("enemy.hitsOnUs");
        Wave w = associate(hb.ownerId(), hb.power(), hb.x(), hb.y(), hb.turn());
        Danger d = dangerOf(hb.ownerId());
        if (w != null) {
            double gf = w.theoreticalGf(hb.heading());
            d.addHit(w, gf, false, round, turn);
            d.rawHits++;
            d.resolvedRealWaves++;
            w.hitAssociated = true;
            w.labelEmitted = true;
            waves.remove(w);
        } else tel.inc("enemy.hitUnmatched");
    }

    private void onBulletHitBullet(Obs.BulletHitBullet bb) {
        killBullet(bb.ownBulletId(), bb.turn());
        tel.inc("fire.bulletContact");
        Wave w = associate(bb.otherOwnerId(), bb.otherPower(), bb.otherX(), bb.otherY(), bb.turn());
        if (w != null) {
            dangerOf(bb.otherOwnerId()).addHit(w, w.theoreticalGf(bb.otherHeading()), true, round, bb.turn());
            w.labelEmitted = true;
            waves.remove(w);
        } else tel.inc("enemy.contactUnmatched");
    }

    /** Associates one physical enemy bullet event with at most one incoming wave. */
    Wave associate(int owner, double power, double x, double y, int turn) {
        Wave best = null;
        double bestRes = Double.POSITIVE_INFINITY;
        for (Wave w : waves) {
            if (w.role != Wave.Role.INCOMING || w.ownerId != owner || w.hitAssociated) continue;
            double res = Math.abs(w.power - power);
            if (res > 0.1 + w.powerUncertainty) continue;
            double d = Math.hypot(x - w.ox, y - w.oy);
            double tol = 2 * w.speed + Rules.BOT_RADIUS;
            if (d < w.innerRadius(turn) - tol || d > w.outerRadius(turn) + tol) continue;
            if (res < bestRes - 1e-12 || (Math.abs(res - bestRes) <= 1e-12 && best != null && w.id < best.id)) { best = w; bestRes = res; }
        }
        return best;
    }

    // ------------------------------------------------------------------ scans and inference

    private void onScan(Obs.Scan s, Obs.Own own) {
        Enemy e = enemy(s.botId());
        if (e.deathTurn != Integer.MIN_VALUE && s.turn() <= e.deathTurn) {
            tel.inc("scan.ignoredAtOrBeforeDeath"); // a same-batch scan never resurrects a confirmed death
            return;
        }
        e.alive = true;
        History.Sample prev = e.latest();
        double wMin = 0, wMax = 0;
        if (prev != null && s.turn() == prev.turn() + 1 && s.speed() == 0 && prev.speed() != 0 && atWall(s.x(), s.y())) {
            double vp = Math.abs(prev.speed());
            double lo = Math.max(0, vp - 2), hi = Math.min(8, vp + 1);
            wMax = Rules.wallDamage(hi);
            wMin = e.lastWallContactTurn == s.turn() - 1 ? 0 : Rules.wallDamage(lo);
            e.lastWallContactTurn = s.turn();
            tel.inc("enemy.wallInferred");
        }
        FireInference.Shot shot = e.fire.onScan(s.turn(), s.energy(), wMin, wMax);
        e.observe(s.turn(), s.x(), s.y(), s.heading(), s.speed(), s.energy());
        if (shot != null) createIncoming(e, shot, s.turn(), own);
    }

    private boolean atWall(double x, double y) {
        double R = Rules.BOT_RADIUS;
        return Math.abs(x - R) < 1e-6 || Math.abs(x - (rules.width() - R)) < 1e-6
                || Math.abs(y - R) < 1e-6 || Math.abs(y - (rules.height() - R)) < 1e-6;
    }

    private void createIncoming(Enemy e, FireInference.Shot shot, int turn, Obs.Own own) {
        History.Sample origin = e.history.at(shot.fireTurnMin() - 1);
        if (origin == null) origin = e.history.previous();
        if (origin == null) { tel.inc("enemy.shotNoOrigin"); return; }
        History.Sample me = self.history.at(shot.fireTurnMin() - 1);
        if (me == null) me = self.history.latest();
        // replace a forecast for this enemy
        waves.removeIf(w -> w.role == Wave.Role.FORECAST && w.ownerId == e.id);
        double conf = shot.certain() ? 1.0 : shot.hypotheses() <= Profile.MAX_SHOT_HYPOTHESES ? 0.7 : 0.5;
        Wave w = threatWave(Wave.Role.INCOMING, e, origin.x(), origin.y(), shot.fireTurnMin(), shot.fireTurnMax(),
                shot.power(), shot.powerUncertainty(), me, conf, turn, own.enemyCount());
        if (shot.certain()) e.power.add(Math.hypot(me.x() - origin.x(), me.y() - origin.y()), origin.energy(), me.energy(), shot.power());
        addWave(w);
        tel.inc(shot.certain() ? "enemy.shotCertain" : "enemy.shotAmbiguous");
    }

    private Wave threatWave(Wave.Role role, Enemy e, double ox, double oy, int fMin, int fMax, double power, double pu,
                            History.Sample me, double conf, int turn, int enemyCount) {
        int s = Angles.orbitSign(ox, oy, me.x(), me.y(), me.heading(), me.speed());
        if (s == 0) s = self.lastMoveSign == 0 ? 1 : lastNonzeroOrbit;
        else lastNonzeroOrbit = s;
        Features.Context ctx = Features.compute(ox, oy, mover(self, me.turn()), power, 0,
                enemyCount / (double) Math.max(1, rules.initialEnemies()), meleeOrigin, s, rules);
        double ref = Angles.bearing(ox, oy, me.x(), me.y());
        Wave w = new Wave(waveIds++, role, e.id, -1, fMin, fMax, ox, oy, power, pu, ref, s, Wave.Escape.THEORETICAL,
                0, 0, ctx, conf, turn);
        // legal constant-velocity aim at us from the origin, for the prior
        Guns.Aim a = new Guns.Aim(ox, oy, fMin, new Physics.Motion(me.x(), me.y(), me.heading(), me.speed()), me.turn(), 0,
                power, ctx, s, w.maxEscape, w.maxEscape, 0, false);
        List<Double> lin = guns.analytic(a);
        w.linearGf = lin.isEmpty() ? 0 : Math.max(-1, Math.min(1, w.theoreticalGf(lin.get(0))));
        return w;
    }

    private int lastNonzeroOrbit = 1;

    private Features.Mover mover(Enemy track, int turn) {
        History.Sample s = track.history.at(turn);
        if (s == null) s = track.history.latest();
        Physics.Motion m = new Physics.Motion(s.x(), s.y(), s.heading(), s.speed());
        return new Features.Mover(m, track.acceleration, Math.max(0, turn - track.lastVelocityChangeTurn),
                Math.max(0, turn - track.lastReversalTurn), disp(track, s, 8), disp(track, s, 20), disp(track, s, 40));
    }

    private static double disp(Enemy track, History.Sample now, int back) {
        History.Sample b = track.history.at(now.turn() - back);
        if (b == null) b = track.history.oldest();
        if (b == null) return Double.NaN;
        return Math.hypot(now.x() - b.x(), now.y() - b.y());
    }

    private void addWave(Wave w) {
        waves.add(w);
        // bound per enemy: retain unresolved real threats before virtual/forecast waves
        int owner = w.isThreat() ? w.ownerId : w.targetId;
        List<Wave> mine = new ArrayList<>();
        for (Wave x : waves) if ((x.isThreat() ? x.ownerId : x.targetId) == owner && x.isThreat() == w.isThreat()) mine.add(x);
        if (mine.size() > Profile.MAX_WAVES_PER_ENEMY) {
            mine.sort(Comparator.comparing((Wave x) -> x.role == Wave.Role.INCOMING || x.role == Wave.Role.OUTGOING ? 1 : 0).thenComparingLong(x -> x.id));
            waves.remove(mine.get(0));
            tel.inc("waves.evicted");
        }
    }

    // ------------------------------------------------------------------ wave resolution

    private void resolveWaves(int turn, Obs.Own own) {
        List<Wave> done = new ArrayList<>();
        for (Wave w : waves) {
            if (w.cancelled) { done.add(w); continue; }
            if (w.role == Wave.Role.FORECAST) {
                if (w.fireTurnMax < turn) { done.add(w); tel.inc("forecast.expired"); }
                continue;
            }
            if (w.role == Wave.Role.INCOMING) {
                Wave.Phase p = w.phaseFor(turn, own.x(), own.y());
                if (!w.labelEmitted && w.outerRadius(turn) >= Math.hypot(own.x() - w.ox, own.y() - w.oy)) {
                    // reliable real-wave self crossing: visit label (never a hit label)
                    double gf = w.theoreticalGf(Angles.bearing(w.ox, w.oy, own.x(), own.y()));
                    w.labelQuality = w.confidence >= 1 ? 1 : 0.5;
                    dangerOf(w.ownerId).addVisit(w, gf, round, turn);
                    w.labelEmitted = true;
                }
                if (p == Wave.Phase.PASSED) {
                    dangerOf(w.ownerId).resolvedRealWaves++;
                    done.add(w);
                }
                continue;
            }
            // VIRTUAL or OUTGOING: target crossing from legal observations only
            Enemy target = enemies.get(w.targetId);
            if (target == null || target.latest() == null) continue;
            int upTo = target.latest().turn();
            for (int t = w.lastProcessedTurn + 1; t <= upTo; t++) {
                History.Sample s = target.history.at(t);
                if (s == null) {
                    History.Sample l = target.latest();
                    if (w.outerRadius(t) >= Math.hypot(l.x() - w.ox, l.y() - w.oy) - 100) {
                        w.cancelled = true; w.cancelReason = "unobserved crossing";
                        tel.inc("waves.cancelledUnobserved");
                        break;
                    }
                    w.lastProcessedTurn = t;
                    continue;
                }
                w.lastProcessedTurn = t;
                double d = Math.hypot(s.x() - w.ox, s.y() - w.oy);
                if (w.fired(t) && w.outerRadius(t) >= d) {
                    w.labelBearing = Angles.bearing(w.ox, w.oy, s.x(), s.y());
                    w.labelHalfWidth = Geometry.halfWidth(d);
                    w.labelQuality = s.provenance() == History.Provenance.OBSERVED ? 1.0 : 0.75;
                    guns.rate(w);
                    if (guns.train(w, round, turn)) tel.inc("gun.labels");
                    if (meleeOrigin && w.targetAtOrigin != null) {
                        double[] lab = Guns.displacementLabel(w.targetAtOrigin, w.targetOriginTurn, s.x(), s.y(), t, w.orbitSign);
                        guns.data(w.targetId).displacement.add(w.context.meleeGun(), lab, w.labelQuality * w.context.quality(), round, t, w.id);
                    }
                    if (w.physical) tel.inc("gun.physicalResolved");
                    done.add(w);
                    break;
                }
            }
            if (w.cancelled) done.add(w);
        }
        waves.removeAll(done);
    }

    // ------------------------------------------------------------------ duel movement

    private DuelPlanner.Result planDuel(int turn, Physics.Motion me, Obs.Own own) {
        Enemy e = duelTarget();
        boolean wallNow = lastOwnWallTurn == turn;
        if (e == null || e.latest() == null) {
            return duel.plan(me, turn, rules.width() / 2, rules.height() / 2, 0, 0, List.of(), lastAction, wallNow);
        }
        History.Sample es = e.latest();
        int age = turn - es.turn();
        double evx = Math.cos(es.heading()) * es.speed(), evy = Math.sin(es.heading()) * es.speed();
        double ex = es.x() + evx * age, ey = es.y() + evy * age;
        maybeForecast(e, turn, own);
        List<DuelPlanner.Threat> threats = new ArrayList<>();
        List<Wave> ws = new ArrayList<>();
        for (Wave w : waves) if (w.isThreat() && w.phaseFor(turn, me.x(), me.y()) != Wave.Phase.PASSED) ws.add(w);
        ws.sort(Comparator.comparingDouble((Wave w) -> contactTime(w, turn, me)).thenComparingLong(w -> w.id));
        List<Shadows.Bullet> live = new ArrayList<>(bullets.values());
        for (Wave w : ws) {
            Danger d = dangerOf(w.ownerId);
            double range = Math.hypot(me.x() - w.ox, me.y() - w.oy);
            Danger.Density den = disabled.contains("danger") ? new Danger().density(w, range, w.linearGf) : d.density(w, range, w.linearGf);
            List<Geometry.Interval> sh = w.role == Wave.Role.INCOMING ? Shadows.compute(w, live, turn, me.x(), me.y(), rules) : List.of();
            double dmg = Rules.bulletDamage(Math.min(Rules.MAX_POWER, w.power + w.powerUncertainty));
            threats.add(new DuelPlanner.Threat(w, den, sh, dmg, Profile.SHADOW_CAP_INFERRED));
            if (threats.size() >= 2) break;
        }
        if (disabled.contains("move")) threats.clear();
        return duel.plan(me, turn, ex, ey, evx, evy, threats, lastAction, wallNow);
    }

    private static double contactTime(Wave w, int turn, Physics.Motion me) {
        double d = Math.hypot(me.x() - w.ox, me.y() - w.oy) - Rules.BOT_RADIUS;
        return w.fireTurnMin + d / w.speed - 2 - turn;
    }

    /** At most one forecast wave per enemy, only when heat is nearly cool and fewer than two threats exist. */
    private void maybeForecast(Enemy e, int turn, Obs.Own own) {
        waves.removeIf(w -> w.role == Wave.Role.FORECAST && w.ownerId == e.id);
        if (!e.alive || e.scanAge(turn) > 1) return;
        int credible = 0;
        for (Wave w : waves) if (w.role == Wave.Role.INCOMING && w.phaseFor(turn, own.x(), own.y()) != Wave.Phase.PASSED) credible++;
        if (credible >= 2) return;
        int fireTurn = Math.max(turn + 1, e.fire.earliestFireTurn());
        if (fireTurn > turn + 2) return;
        History.Sample es = e.latest();
        History.Sample me = self.history.latest();
        double range = Math.hypot(me.x() - es.x(), me.y() - es.y());
        double p = e.power.predict(range, es.energy(), own.energy());
        if (!(es.energy() > Rules.MIN_POWER)) return;
        Wave w = threatWave(Wave.Role.FORECAST, e, es.x(), es.y(), fireTurn, fireTurn, p, 0, me, Profile.FORECAST_CONFIDENCE, turn, own.enemyCount());
        waves.add(w);
        tel.inc("forecast.created");
    }

    private Enemy duelTarget() {
        Enemy best = null;
        for (Enemy e : enemies.values()) if (e.alive && e.latest() != null && (best == null || e.latest().turn() > best.latest().turn())) best = e;
        return best;
    }

    private List<MeleePlanner.Opponent> opponents(int turn) {
        List<MeleePlanner.Opponent> out = new ArrayList<>();
        for (Enemy e : enemies.values()) {
            if (!e.alive || e.latest() == null) continue;
            History.Sample s = e.latest();
            double threat = (e.damageToUs + 5) / Math.max(1, e.turnsObservedTogether);
            out.add(new MeleePlanner.Opponent(e.id, s.x(), s.y(), s.energy(), threat, turn - s.turn(),
                    Math.cos(s.heading()) * s.speed(), Math.sin(s.heading()) * s.speed()));
        }
        return out;
    }

    private List<Radar.Known> known() {
        List<Radar.Known> k = new ArrayList<>();
        for (Enemy e : enemies.values()) if (e.alive && e.latest() != null) k.add(new Radar.Known(e.id, e.latest().x(), e.latest().y(), e.latest().turn()));
        return k;
    }

    // ------------------------------------------------------------------ gun

    /**
     * Creates this tick's virtual wave(s) (origin: our current position, fire turn t+1) with
     * frozen predictions on firing opportunities, then prepares the plan for the next tick from the
     * predicted next own position. Returns the gun turn.
     */
    private double aim(int turn, Obs.Own own, Physics.Motion next, boolean firingNow) {
        int liveCount = own.enemyCount();
        double population = liveCount / (double) Math.max(1, rules.initialEnemies());
        int coolTurns = rules.turnsToCool(own.gunHeat());
        // virtual waves for every fresh target
        for (Enemy e : enemies.values()) {
            // the wave of an actual request is always created and frozen with the request's own power
            boolean request = firingNow && pending != null && pending.waveId() < 0 && pending.targetId() == e.id && e.latest() != null;
            if (!e.alive || (e.scanAge(turn) != 0 && !request)) continue;
            History.Sample ts = e.latest();
            double range = Math.hypot(ts.x() - own.x(), ts.y() - own.y());
            double power = liveCount > 1 ? Firing.meleePower(range, own.energy(), ts.energy()) : Firing.duelPower(range, own.energy(), ts.energy());
            if (power <= 0) power = Rules.MIN_POWER;
            if (request) power = pending.power();
            Guns.Aim a = aimFor(e, own.x(), own.y(), turn + 1, power, firingNow ? 0 : coolTurns, population, own.gunHeading());
            Wave w = new Wave(waveIds++, Wave.Role.VIRTUAL, -1, e.id, turn + 1, turn + 1, own.x(), own.y(), power, 0,
                    a.reference(), a.orbitSign(), Wave.Escape.PRECISE, a.escPos(), a.escNeg(), a.context(), 1.0, turn);
            w.targetAtOrigin = a.target();
            w.targetOriginTurn = ts.turn();
            w.lastProcessedTurn = turn;
            if (own.gunHeat() == 0.0 && liveCount == 1) {
                w.fireOpportunity = true;
                Guns.Data d = guns.data(e.id);
                for (String g : guns.eligible(d)) {
                    Double p = guns.predict(g, a, d);
                    if (p != null) w.predictions.put(g, p);
                }
            }
            addWave(w);
            if (request) {
                Firing.Pending p = pending;
                pending = new Firing.Pending(p.requestTurn(), p.expectedConfirmTurn(), p.power(), p.angle(), p.originX(), p.originY(), p.targetId(), w.id);
            }
        }
        // the plan for next tick
        if (liveCount > 1) return aimMelee(turn, own, next, coolTurns);
        Enemy e = duelTarget();
        if (e == null) return 0;
        History.Sample ts = e.latest();
        double range = Math.hypot(ts.x() - next.x(), ts.y() - next.y());
        double direct = Angles.bearing(next.x(), next.y(), ts.x(), ts.y());
        int coolAfter = firingNow ? 99 : Math.max(0, coolTurns - 1);
        if (coolAfter > Profile.EXPENSIVE_AIM_COOL_TURNS || turn - ts.turn() > 2) {
            plan = null;
            return clampGun(Angles.wrap(direct - own.gunHeading()));
        }
        double power = Firing.duelPower(range, own.energy() - (firingNow ? pending.power() : 0), ts.energy());
        Guns.Aim a = aimFor(e, next.x(), next.y(), turn + 2, Math.max(power, Rules.MIN_POWER), 0, population, own.gunHeading());
        Guns.Data d = guns.data(e.id);
        Guns.Choice ch;
        if (disabled.contains("gun")) ch = new Guns.Choice(Guns.ANALYTIC, guns.predict(Guns.ANALYTIC, a, d) == null ? direct : guns.predict(Guns.ANALYTIC, a, d), "disabled");
        else ch = guns.choose(a, d);
        if (ch.fallback() != null) tel.inc("gun.fallback");
        tel.inc("gun.aim." + ch.gun());
        double angle = ch.angle();
        if (power > 0 && coolAfter == 0) {
            plan = new Firing.Plan(e.id, angle, power, turn, turn + Profile.PLAN_EXPIRY_TURNS, Guns.fireTolerance(range),
                    next.x(), next.y(), stateVersion, ch.gun(), ts.turn());
        } else plan = null;
        return clampGun(Angles.wrap(angle - own.gunHeading()));
    }

    private Guns.Aim aimFor(Enemy e, double ox, double oy, int fireTurn, double power, int fireProx, double population, double gunHeading) {
        History.Sample ts = e.latest();
        Physics.Motion tm = new Physics.Motion(ts.x(), ts.y(), ts.heading(), ts.speed());
        int s = Angles.orbitSign(ox, oy, ts.x(), ts.y(), ts.heading(), ts.speed());
        Integer last = lastTargetOrbit.get(e.id);
        if (s == 0) s = last == null ? 1 : last;
        else lastTargetOrbit.put(e.id, s);
        Features.Context ctx = Features.compute(ox, oy, mover(e, ts.turn()), power, fireProx, population, meleeOrigin, s, rules);
        Escape.Reach r = Escape.reach(ox, oy, tm, ts.turn(), fireTurn, Rules.bulletSpeed(power), s, rules);
        double turnRate = 0;
        History.Sample p = e.history.previous();
        if (p != null && ts.turn() - p.turn() <= 2) turnRate = Angles.wrap(ts.heading() - p.heading()) / (ts.turn() - p.turn());
        return new Guns.Aim(ox, oy, fireTurn, tm, ts.turn(), turnRate, power, ctx, s, r.positive(), r.negative(), gunHeading, meleeOrigin);
    }

    private final Map<Integer, Integer> lastTargetOrbit = new LinkedHashMap<>();

    private double aimMelee(int turn, Obs.Own own, Physics.Motion next, int coolTurns) {
        List<Guns.Target> targets = new ArrayList<>();
        Enemy nearest = null;
        double nd = Double.POSITIVE_INFINITY;
        double population = own.enemyCount() / (double) Math.max(1, rules.initialEnemies());
        for (Enemy e : enemies.values()) {
            if (!e.alive || e.latest() == null || e.scanAge(turn) > Profile.MAX_INTERP_GAP) continue;
            History.Sample ts = e.latest();
            double range = Math.hypot(ts.x() - next.x(), ts.y() - next.y());
            if (range < nd) { nd = range; nearest = e; }
            double power = Firing.meleePower(range, own.energy(), ts.energy());
            Guns.Aim a = aimFor(e, next.x(), next.y(), turn + 2, Math.max(power, Rules.MIN_POWER), 0, population, own.gunHeading());
            targets.add(new Guns.Target(e.id, next.x(), next.y(), turn + 2, a.target(), ts.turn(), a.context(), a.orbitSign(), Math.max(power, Rules.MIN_POWER)));
        }
        if (nearest == null) { plan = null; return 0; }
        // expensive projections every other turn until four turns from readiness, then every turn
        boolean expensive = coolTurns <= 4 || (turn % 2 == 0);
        Guns.MeleeChoice mc = expensive && !disabled.contains("gun") ? guns.aimMelee(targets, own.gunHeading()) : null;
        double angle;
        int primary;
        String gun;
        if (mc != null) { angle = mc.angle(); primary = mc.primaryTarget(); gun = Guns.DISPLACEMENT; }
        else {
            Guns.Target t = targets.stream().filter(x -> x.id() == nearestId(targets, next)).findFirst().orElse(targets.get(0));
            Enemy ne = enemies.get(t.id());
            Guns.Aim a = aimFor(ne, next.x(), next.y(), turn + 2, t.power(), 0, population, own.gunHeading());
            List<Double> an = guns.analytic(a);
            angle = an.isEmpty() ? a.reference() : an.get(0);
            primary = t.id();
            gun = Guns.ANALYTIC;
            tel.inc("gun.meleeFallback");
        }
        Enemy pe = enemies.get(primary);
        History.Sample ps = pe.latest();
        double range = Math.hypot(ps.x() - next.x(), ps.y() - next.y());
        double power = Firing.meleePower(range, own.energy(), ps.energy());
        int coolAfter = Math.max(0, coolTurns - 1);
        if (power > 0 && coolAfter == 0) {
            plan = new Firing.Plan(primary, angle, power, turn, turn + Profile.PLAN_EXPIRY_TURNS, Guns.fireTolerance(range),
                    next.x(), next.y(), stateVersion, gun, ps.turn());
        } else plan = null;
        return clampGun(Angles.wrap(angle - own.gunHeading()));
    }

    private static int nearestId(List<Guns.Target> ts, Physics.Motion me) {
        int id = ts.get(0).id();
        double best = Double.POSITIVE_INFINITY;
        for (Guns.Target t : ts) {
            double d = Math.hypot(t.motion().x() - me.x(), t.motion().y() - me.y());
            if (d < best) { best = d; id = t.id(); }
        }
        return id;
    }

    private static double clampGun(double v) {
        double lim = Math.toRadians(Rules.MAX_GUN_TURN_DEG);
        return Math.max(-lim, Math.min(lim, v));
    }

    // ------------------------------------------------------------------ memory bound

    /** Estimated neighbour payload across all stores. */
    public long storePayloadBytes() {
        long total = 0;
        for (Guns.Data d : guns.all().values()) {
            total += d.longStore.payloadBytes() + d.displacement.payloadBytes();
            for (var s : d.recent) total += s.payloadBytes();
            for (var s : d.surfer) total += s.payloadBytes();
        }
        for (Danger d : danger.values()) total += d.simple.payloadBytes() + d.context.payloadBytes() + d.recent.payloadBytes() + d.visits.payloadBytes();
        return total;
    }

    /** Global cap: evict the oldest low-priority virtual-wave samples (gun stores) first. */
    void enforceMemoryCap() {
        enforceMemoryCap(Profile.GLOBAL_STORE_BYTES);
    }

    /** Package-private explicit budget permits bounded cap regression fixtures. */
    void enforceMemoryCap(long cap) {
        if (cap < 0) throw new IllegalArgumentException("negative payload cap");
        int guard = 0;
        while (storePayloadBytes() > cap && guard++ < 10_000) {
            Guns.Data victim = null;
            for (Guns.Data d : guns.all().values()) if (victim == null || d.longStore.size() + d.displacement.size() > victim.longStore.size() + victim.displacement.size()) victim = d;
            if (victim != null && victim.longStore.size() + victim.displacement.size() > 0) {
                victim.longStore.evictOldest(64);
                victim.displacement.evictOldest(64);
            } else {
                // Preserve virtual-wave eviction priority, then include confirmed-wave stores.
                NeighborStore physical = null;
                for (Guns.Data d : guns.all().values()) for (NeighborStore store : d.surfer)
                    if (physical == null || store.payloadBytes() > physical.payloadBytes()) physical = store;
                if (physical == null || physical.size() == 0) break;
                physical.evictOldest(64);
            }
            tel.inc("memory.evictions");
        }
    }

    // ------------------------------------------------------------------ failure handling

    private void guard(String subsystem, Runnable r) {
        try { r.run(); succeed(subsystem); } catch (RuntimeException ex) { fail(subsystem, ex); }
    }

    private void succeed(String s) { failures.put(s, 0); }

    private void fail(String s, RuntimeException ex) {
        tel.inc("failure." + s);
        int n = failures.merge(s, 1, Integer::sum);
        if (n >= Profile.FAILURES_TO_DISABLE && disabled.add(s)) tel.inc("failure.disabled." + s);
        if (tel.get("failure." + s) <= 3) {
            System.err.println("Prism " + s + " failure: " + ex);
            ex.printStackTrace();
        }
    }

    public Set<String> disabledSubsystems() { return disabled; }
    public long seed() { return seed; }
}
