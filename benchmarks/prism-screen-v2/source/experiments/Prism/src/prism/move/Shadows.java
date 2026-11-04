package prism.move;

import prism.core.Geometry;
import prism.core.Rules;
import prism.wave.Wave;

import java.util.ArrayList;
import java.util.List;

/**
 * Bullet shadows (specification section 9.4). For each engine-confirmed own bullet, the per-turn
 * bullet segment is intersected with an incoming wave's per-turn radial shell. Only turns while the
 * bullet is alive (strictly before its confirmed death turn), inside the field, and early enough
 * that the enemy bullet cannot yet have reached us are used.
 */
public final class Shadows {
    private Shadows() {}

    /** A confirmed own bullet; deathTurn is Integer.MAX_VALUE while alive. */
    public record Bullet(int id, int fireTurn, double x, double y, double heading, double power, int deathTurn) {
        double speed() { return Rules.bulletSpeed(power); }
        /** Engine position after t - fireTurn + 1 ticks: the start of the segment checked on turn t. */
        double[] start(int t) { double r = (t - fireTurn + 1) * speed(); return new double[] {x + Math.cos(heading) * r, y + Math.sin(heading) * r}; }
        double[] end(int t) { double r = (t - fireTurn + 2) * speed(); return new double[] {x + Math.cos(heading) * r, y + Math.sin(heading) * r}; }
    }

    /**
     * Merged shadow intervals (relative to the wave reference) for wave w given our bullets, from
     * turn {@code now} on. {@code selfX,selfY} bound the time the enemy bullet could reach us.
     */
    public static List<Geometry.Interval> compute(Wave w, List<Bullet> bullets, int now, double selfX, double selfY, Rules rules) {
        List<Geometry.Interval> parts = new ArrayList<>();
        double d0 = Math.hypot(selfX - w.ox, selfY - w.oy);
        for (Bullet b : bullets) {
            int from = Math.max(now, Math.max(b.fireTurn(), w.fireTurnMin));
            for (int t = from; t < from + 200; t++) {
                if (t >= b.deathTurn()) break;
                double[] s = b.start(t);
                if (s[0] <= 0 || s[1] <= 0 || s[0] >= rules.width() || s[1] >= rules.height()) break;
                double r0 = Math.max(0, w.innerRadius(t)), r1 = w.outerRadius(t);
                // the enemy bullet may already have reached our footprint: stop counting
                double reachable = d0 - Rules.BOT_RADIUS - Rules.MAX_SPEED * Math.max(0, t - now);
                if (r1 >= reachable) break;
                if (r1 <= 0) continue;
                double[] e = b.end(t);
                parts.addAll(Geometry.segmentShell(w.ox, w.oy, s[0], s[1], e[0], e[1], r0, r1, w.reference));
            }
        }
        return Geometry.union(parts);
    }
}
