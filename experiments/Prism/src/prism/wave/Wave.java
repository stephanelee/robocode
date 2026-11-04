package prism.wave;

import prism.core.Geometry;
import prism.core.Rules;
import prism.learn.Features;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * A wave of possible bullet bearings. Roles are kept distinct:
 * OUTGOING = our engine-confirmed bullet; VIRTUAL = our hypothetical shot used only for learning
 * and gun rating; INCOMING = an enemy shot inferred from scan energy; FORECAST = a hypothesised
 * future enemy shot used only for movement.
 *
 * Radial shell on turn t (engine contract): [(t - fireTurnMax + 1) b, (t - fireTurnMin + 2) b].
 */
public final class Wave {
    public enum Role { OUTGOING, VIRTUAL, INCOMING, FORECAST }
    public enum Escape { PRECISE, THEORETICAL }
    public enum Phase { PENDING, APPROACHING, OVERLAPPING, CENTER_PASSED, PASSED }

    public final long id;
    public Role role;
    public final int ownerId, targetId;
    public final int fireTurnMin, fireTurnMax;
    public final double ox, oy;
    public final double power, powerUncertainty, speed;
    public final double reference;      // source-to-target bearing at origin
    public final int orbitSign;         // target orbit sign at origin, never from outcome
    public final Escape escape;
    public final double escapePositive, escapeNegative, maxEscape;
    public final Features.Context context;
    public double confidence;
    public final int createdTurn;

    public int bulletId = -1;           // engine bullet id when associated
    public boolean physical;            // OUTGOING after confirmation
    public boolean fireOpportunity;     // our gun was ready at origin (rated cohort)
    public final Map<String, Double> predictions = new java.util.LinkedHashMap<>(); // gun -> frozen angle
    public final List<Geometry.Interval> crossing = new ArrayList<>();
    public List<Geometry.Interval> shadows = List.of();
    public int lastProcessedTurn;
    public Phase phase = Phase.PENDING;
    public boolean labelEmitted, cancelled, hitAssociated;
    public String cancelReason;
    public double labelBearing = Double.NaN, labelHalfWidth = Double.NaN;
    public double labelQuality = 1;
    /** Theoretical GF of a legal constant-velocity aim at the target from the origin (danger prior). */
    public double linearGf = 0;
    /** Target motion and turn at origin (for displacement labels). */
    public prism.core.Physics.Motion targetAtOrigin;
    public int targetOriginTurn;

    public Wave(long id, Role role, int ownerId, int targetId, int fireTurnMin, int fireTurnMax, double ox, double oy,
                double power, double powerUncertainty, double reference, int orbitSign, Escape escape,
                double escapePositive, double escapeNegative, Features.Context context, double confidence, int createdTurn) {
        if (fireTurnMax < fireTurnMin) throw new IllegalArgumentException("fire interval");
        this.id = id; this.role = role; this.ownerId = ownerId; this.targetId = targetId;
        this.fireTurnMin = fireTurnMin; this.fireTurnMax = fireTurnMax; this.ox = ox; this.oy = oy;
        this.power = power; this.powerUncertainty = powerUncertainty; this.speed = Rules.bulletSpeed(power);
        this.reference = reference; this.orbitSign = orbitSign == 0 ? 1 : orbitSign; this.escape = escape;
        this.escapePositive = Math.max(1e-6, escapePositive); this.escapeNegative = Math.max(1e-6, escapeNegative);
        this.maxEscape = Rules.maxEscape(speed);
        this.context = context; this.confidence = confidence; this.createdTurn = createdTurn;
        this.lastProcessedTurn = createdTurn;
    }

    /** Inner radius of the shell on turn t (for the latest possible fire turn); negative before firing. */
    public double innerRadius(int t) { return (t - fireTurnMax + 1) * speed; }
    /** Outer radius of the shell on turn t (for the earliest possible fire turn). */
    public double outerRadius(int t) { return (t - fireTurnMin + 2) * speed; }
    public boolean fired(int t) { return t >= fireTurnMin; }

    /** Engine footprint interval of a bot centred at (x,y) on turn t, or null. */
    public Geometry.Interval footprint(int t, double x, double y) {
        if (!fired(t)) return null;
        return Geometry.footprint(ox, oy, x, y, Rules.BOT_RADIUS, Math.max(0, innerRadius(t)), outerRadius(t), reference);
    }

    public Phase phaseFor(int t, double x, double y) {
        if (!fired(t)) return Phase.PENDING;
        double d = Math.hypot(x - ox, y - oy);
        double r0 = innerRadius(t), r1 = outerRadius(t);
        if (r1 < d - Rules.BOT_RADIUS) return Phase.APPROACHING;
        if (r0 > d + Rules.BOT_RADIUS) return Phase.PASSED;
        if (r0 > d) return Phase.CENTER_PASSED;
        return Phase.OVERLAPPING;
    }

    /** Precise guess factor of an absolute bearing using this wave's frozen convention. */
    public double preciseGf(double bearing) {
        double u = orbitSign * Geometry.rel(bearing, reference);
        return u >= 0 ? u / escapePositive : u / escapeNegative;
    }

    public double decodePrecise(double gf) {
        double u = gf >= 0 ? gf * escapePositive : gf * escapeNegative;
        return reference + orbitSign * u;
    }

    public double theoreticalGf(double bearing) { return orbitSign * Geometry.rel(bearing, reference) / maxEscape; }
    public double decodeTheoretical(double gf) { return reference + orbitSign * gf * maxEscape; }

    /** Relative offset (to reference) of a theoretical GF. */
    public double offsetOfTheoretical(double gf) { return orbitSign * gf * maxEscape; }

    public boolean isThreat() { return role == Role.INCOMING || role == Role.FORECAST; }
}
