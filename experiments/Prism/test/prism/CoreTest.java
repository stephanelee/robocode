package prism;

import org.junit.jupiter.api.Test;
import prism.core.Angles;
import prism.core.Drive;
import prism.core.Geometry;
import prism.core.Physics;
import prism.core.Rules;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/** Native angles, engine motion law, and wave geometry against brute force. */
class CoreTest {
    static final Rules R = new Rules(800, 600, 0.1, 1);

    @Test
    void cardinalVectorsWrapAndOrbitSign() {
        assertEquals(0, Angles.bearing(0, 0, 1, 0), 1e-12);
        assertEquals(Math.PI / 2, Angles.bearing(0, 0, 0, 1), 1e-12);
        assertEquals(Math.PI, Angles.bearing(0, 0, -1, 0), 1e-12);
        assertEquals(3 * Math.PI / 2, Angles.bearing(0, 0, 0, -1), 1e-12);
        assertEquals(-Math.PI, Angles.wrap(Math.PI), 1e-12);           // [-pi, pi)
        assertEquals(-Math.PI, Angles.wrap(-Math.PI), 1e-12);
        assertEquals(0.1, Angles.wrap(0.1 + 4 * Math.PI), 1e-12);
        assertEquals(0, Angles.heading(2 * Math.PI), 1e-12);
        assertTrue(Angles.heading(-1e-18) < Angles.TWO_PI);
        // effective heading reverses when backing; crossing zero keeps the body heading
        assertEquals(Math.PI, Angles.effectiveHeading(0, -3), 1e-12);
        assertEquals(0, Angles.effectiveHeading(0, 3), 1e-12);
        assertEquals(0, Angles.effectiveHeading(0, 0), 1e-12);
        // target east of source moving north orbits counterclockwise (+1); backing south is the same
        assertEquals(1, Angles.orbitSign(0, 0, 100, 0, Math.PI / 2, 8));
        assertEquals(1, Angles.orbitSign(0, 0, 100, 0, 3 * Math.PI / 2, -8));
        assertEquals(-1, Angles.orbitSign(0, 0, 100, 0, Math.PI / 2, -8));
        assertEquals(0, Angles.orbitSign(0, 0, 100, 0, 0, 8));
        // classic angle (analyst comparison only)
        assertEquals(0, Angles.classicFromNative(Math.PI / 2), 1e-12);
        assertEquals(Math.PI / 2, Angles.classicFromNative(0), 1e-12);
    }

    @Test
    void engineSpeedLaw() {
        assertEquals(1, Physics.newSpeed(0, 8));
        assertEquals(8, Physics.newSpeed(7.5, 8));
        assertEquals(6, Physics.newSpeed(8, 3));      // braking limited by 2
        assertEquals(0, Physics.newSpeed(8, 0), 0.0);      // target zero stops instantly (engine 1.4.0)
        assertEquals(0, Physics.newSpeed(-5, 0), 0.0);
        assertEquals(3, Physics.newSpeed(8, -8), 1e-12);    // crossing law from high speed stays positive
        assertEquals(0.5, Physics.newSpeed(3, -8), 1e-12);
        assertEquals(-0.5, Physics.newSpeed(1, -8), 1e-12); // reverses within the turn from low speed
        assertEquals(0.5, Physics.newSpeed(-1, 8), 1e-12);
        assertEquals(-1, Physics.newSpeed(0, -2));
    }

    @Test
    void translationUsesStartHeadingAndTurnLimitUsesNewSpeed() {
        Physics.Step s = Physics.step(new Physics.Motion(400, 300, 0, 8), 8, Math.toRadians(10), R);
        assertEquals(408, s.next().x(), 1e-12);
        assertEquals(300, s.next().y(), 1e-12);
        assertEquals(Math.toRadians(4), s.next().heading(), 1e-12);  // 10 - 0.75*8
        Physics.Step stop = Physics.step(new Physics.Motion(400, 300, 0, 8), 0, Math.toRadians(10), R);
        assertEquals(400, stop.next().x(), 1e-12);
        assertEquals(Math.toRadians(10), stop.next().heading(), 1e-12);
    }

    @Test
    void wallConstraintAlongTravelLineAndRepeatedContactDamage() {
        // diagonal push into the east wall pins the bot: the engine clamps along the travelled line
        Physics.Step s = Physics.step(new Physics.Motion(782, 300, Math.toRadians(30), 0), 1, 0, R, false);
        assertTrue(s.hitWall());
        assertEquals(782, s.next().x(), 1e-12);
        assertEquals(300, s.next().y(), 1e-12);
        assertEquals(0, s.next().speed());
        Physics.Step hard = Physics.step(new Physics.Motion(778, 300, 0, 8), 8, 0, R, false);
        assertEquals(3, hard.wallDamage(), 1e-12);
        assertEquals(0, Physics.step(new Physics.Motion(778, 300, 0, 8), 8, 0, R, true).wallDamage());
        // corner: engine formula clamps x then y along the original line
        Physics.Step c = Physics.step(new Physics.Motion(779, 579, Math.toRadians(45), 8), 8, 0, R, false);
        assertTrue(c.hitWall());
        assertEquals(782, c.next().x(), 1e-9);
        assertEquals(582, c.next().y(), 1e-9);
    }

    @Test
    void backAsFrontChoosesInstantStopOverSlowCrossing() {
        Physics.Motion m = new Physics.Motion(400, 300, 0, 8);
        Drive.Control c = Drive.toward(m, Math.PI, 8);   // reverse travel: back up, no body turn
        assertEquals(0.0, c.targetSpeed());              // 8 -> 0 beats 8 -> 3 forward
        assertEquals(0, c.turnRate(), 1e-12);
        Drive.Control low = Drive.toward(new Physics.Motion(400, 300, 0, 1), Math.PI, 8);
        assertEquals(-8, low.targetSpeed());              // 1 -> -0.5 beats 1 -> 0
        Drive.Control quarter = Drive.toward(new Physics.Motion(400, 300, 0, 0), Math.toRadians(100), 8);
        assertTrue(Math.abs(quarter.turnRate()) <= Math.PI / 2 + 1e-12);
    }

    @Test
    void footprintMatchesBruteForceIncludingTangencyAndWrap() {
        Random r = new Random(7);
        for (int k = 0; k < 400; k++) {
            double sx = r.nextDouble() * 800, sy = r.nextDouble() * 600;
            double cx = r.nextDouble() * 800, cy = r.nextDouble() * 600;
            double d = Math.hypot(cx - sx, cy - sy);
            double r0 = Math.max(0, d - 40 + r.nextDouble() * 60), r1 = r0 + 5 + r.nextDouble() * 20;
            double ref = r.nextDouble() * 7 - 3.5;
            Geometry.Interval iv = Geometry.footprint(sx, sy, cx, cy, 18, r0, r1, ref);
            // brute force: sample bearings and radii
            double lo = Double.POSITIVE_INFINITY, hi = Double.NEGATIVE_INFINITY;
            double c0 = Geometry.rel(Math.atan2(cy - sy, cx - sx), ref);
            boolean any = false;
            for (int a = -2000; a <= 2000; a++) {
                double off = a / 2000.0 * (d > 18 ? Math.asin(18 / d) * 1.05 : Math.PI);
                for (int q = 0; q <= 60; q++) {
                    double rr = r0 + (r1 - r0) * q / 60.0;
                    double px = sx + Math.cos(c0 + ref + off) * rr, py = sy + Math.sin(c0 + ref + off) * rr;
                    if (Math.hypot(px - cx, py - cy) <= 18) { any = true; lo = Math.min(lo, c0 + off); hi = Math.max(hi, c0 + off); break; }
                }
            }
            if (d <= 18) { if (iv != null) assertEquals(2 * Math.PI, iv.width(), 1e-9); continue; }
            if (!any) { assertTrue(iv == null || iv.width() < 2e-3, "unexpected interval at case " + k); continue; }
            assertNotNull(iv, "missing interval at case " + k);
            double tolA = 2 * Math.asin(18 / d) / 2000 * 2 + 1e-3;
            assertEquals(lo, iv.lo(), tolA, "lo case " + k);
            assertEquals(hi, iv.hi(), tolA, "hi case " + k);
        }
        // tangent ring exactly at sqrt(d^2-R^2)
        double rt = Math.sqrt(200 * 200 - 18 * 18);
        Geometry.Interval t = Geometry.footprint(0, 0, 200, 0, 18, rt, rt, 0);
        assertEquals(Math.asin(18 / 200.0), t.hi(), 1e-12);
        // wraparound: target due west with reference due east gives offsets near +-pi handled once
        Geometry.Interval w = Geometry.footprint(0, 0, -200, 1e-9, 18, 150, 250, Math.PI - 1e-6);
        assertTrue(w.width() < 0.2);
        assertNull(Geometry.footprint(0, 0, 200, 0, 18, 0, 100, 0));     // approaching: none yet
        assertNull(Geometry.footprint(0, 0, 200, 0, 18, 230, 250, 0));   // passed
    }

    @Test
    void unionKeepsDisconnectedPartsAndOverlapMeasures() {
        List<Geometry.Interval> u = Geometry.union(List.of(new Geometry.Interval(0.3, 0.4), new Geometry.Interval(0, 0.1), new Geometry.Interval(0.05, 0.2)));
        assertEquals(2, u.size());
        assertEquals(0.2, u.get(0).hi(), 1e-12);
        assertEquals(0.15, Geometry.overlap(new Geometry.Interval(0.1, 0.35), u), 1e-12);
    }

    @Test
    void segmentShellMatchesBruteForce() {
        Random r = new Random(11);
        for (int k = 0; k < 200; k++) {
            double ax = 100 + r.nextDouble() * 400, ay = 100 + r.nextDouble() * 300;
            double h = r.nextDouble() * 6.28, len = 5 + r.nextDouble() * 15;
            double bx = ax + Math.cos(h) * len, by = ay + Math.sin(h) * len;
            double d = Math.hypot(ax, ay);
            double r0 = d - 20 + r.nextDouble() * 30, r1 = r0 + 11;
            List<Geometry.Interval> iv = Geometry.union(Geometry.segmentShell(0, 0, ax, ay, bx, by, r0, r1, 0.5));
            double measure = 0;
            for (Geometry.Interval i : iv) measure += i.width();
            // brute force along the segment
            double lo = Double.POSITIVE_INFINITY, hi = Double.NEGATIVE_INFINITY;
            for (int q = 0; q <= 20000; q++) {
                double px = ax + (bx - ax) * q / 20000.0, py = ay + (by - ay) * q / 20000.0;
                double rr = Math.hypot(px, py);
                if (rr >= r0 && rr <= r1) { double a = Geometry.rel(Math.atan2(py, px), 0.5); lo = Math.min(lo, a); hi = Math.max(hi, a); }
            }
            if (lo > hi) { assertTrue(measure < 1e-6, "case " + k); continue; }
            assertEquals(lo, iv.get(0).lo(), 1e-4, "case " + k);
            assertEquals(hi, iv.get(iv.size() - 1).hi(), 1e-4, "case " + k);
        }
    }
}
