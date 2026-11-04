import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import org.junit.jupiter.api.Test;

/** Implementation contracts only; these establish no competitive strength. */
class KeelContractTest {
    @Test void speedLawMatchesPinnedServer() {
        assertEquals(0, Phys.speed(8, 0), 1e-12, "zero target stops at once");
        assertEquals(3, Phys.speed(8, -8), 1e-12, "reversal from +8");
        assertEquals(-0.5, Phys.speed(1, -8), 1e-12);
        assertEquals(6, Phys.speed(8, 6), 1e-12, "deceleration limit 2");
        assertEquals(6, Phys.speed(8, 1), 1e-12);
        assertEquals(1, Phys.speed(0, 8), 1e-12, "acceleration limit 1");
        assertEquals(-1, Phys.speed(0, -8), 1e-12);
        assertEquals(0, Phys.speed(-8, 0), 1e-12);
        assertEquals(-3, Phys.speed(-8, 8), 1e-12);
        assertEquals(8, Phys.speed(7.5, 8), 1e-12);
    }

    @Test void stepMovesOnOldHeadingThenTurnsWithNewSpeedLimit() {
        var p = Phys.step(new Phys.Pose(400, 300, 0, 8), 10, 8, 800, 600);
        assertEquals(408, p.x(), 1e-9);
        assertEquals(300, p.y(), 1e-9);
        assertEquals(4, p.h(), 1e-9, "turn capped at 10 - 0.75 * 8");
        var stop = Phys.step(new Phys.Pose(400, 300, 0, 8), 0, 0, 800, 600);
        assertEquals(400, stop.x(), 1e-9);
        assertEquals(0, stop.v(), 1e-12);
    }

    @Test void wallStopsBotAndZeroesSpeed() {
        var p = Phys.step(new Phys.Pose(785, 300, 0, 8), 0, 8, 800, 600);
        assertEquals(782, p.x(), 1e-9);
        assertEquals(0, p.v(), 1e-12);
    }

    @Test void halfWidthIsTangentAngleWhenAnnulusContainsBot() {
        double D = 400;
        double hw = Surfer.halfWidth(D, 300, 500);
        assertEquals(Math.toDegrees(Math.asin(18 / D)), hw, 1e-9);
        assertEquals(-1, Surfer.halfWidth(D, 0, 381.9), 0);
        assertTrue(Surfer.halfWidth(D, 381, 383) < hw);
    }

    @Test void densityIsNormalizedAndCoversAll() {
        var s = new Surfer();
        double[] f = new double[Track.DIMS];
        double[] d = s.density(f);
        double sum = 0;
        for (double x : d) { assertTrue(x > 0); sum += x; }
        assertEquals(1, sum, 1e-9);
        assertEquals(1, Surfer.covered(d, new double[Surfer.BINS], -1.1, 1.1), 1e-9);
        int head = Surfer.bin(0);
        for (int b = 0; b < Surfer.BINS; b++) assertTrue(d[head] >= d[b], "cold prior peaks at head-on");
    }

    @Test void hitsShiftDanger() {
        var s = new Surfer();
        double[] f = new double[Track.DIMS];
        for (int i = 0; i < 8; i++) s.hits.add(f, 0.6, 1);
        double[] d = s.density(f);
        assertTrue(d[Surfer.bin(0.6)] > d[Surfer.bin(-0.6)]);
    }

    @Test void stationaryDangerMatchesFootprintWidth() {
        double[] uniform = new double[Surfer.BINS];
        java.util.Arrays.fill(uniform, 1.0 / Surfer.BINS);
        var w = new Surfer.Wave(1, 0, 300, 2, 0, 1, new double[Track.DIMS], uniform);
        var path = new ArrayList<Phys.Pose>();
        for (int i = 0; i < 60; i++) path.add(new Phys.Pose(400, 300, 0, 0));
        double danger = Surfer.danger(w, path, 1);
        double gfWidth = 2 * Math.toDegrees(Math.asin(18 / 400.0)) / w.mea;
        assertEquals(gfWidth / (2.0 * Surfer.BINS / (Surfer.BINS - 1)), danger, 1e-3);
    }

    @Test void shadowBlocksOnlyWhereOurBulletCrossesTheWave() {
        double[] uniform = new double[Surfer.BINS];
        java.util.Arrays.fill(uniform, 1.0 / Surfer.BINS);
        var s = new Surfer();
        Surfer.shadows = true;
        var w = new Surfer.Wave(10, 100, 300, 2, 0, 1, new double[Track.DIMS], uniform);
        s.waves.add(w);
        // Off-axis crossing: our bullet travels south across the wave near GF 0 (radial bullets shade ~nothing).
        s.bullets.add(new Surfer.Own(1, 10, 400, 300 + 17 * 21, -90, Phys.bulletSpeed(1)));
        s.reshade(800, 800);
        double near0 = 0;
        for (int b = Surfer.bin(-0.05); b <= Surfer.bin(0.2); b++) near0 = Math.max(near0, w.shade[b]);
        assertTrue(near0 > 0, "crossing region is blocked");
        assertEquals(0, w.shade[Surfer.bin(0.8)], 1e-12);
        assertEquals(0, w.shade[Surfer.bin(-0.8)], 1e-12);
        Surfer.shadows = false;
    }

    @Test void wallRoomShrinksNearWalls() {
        double open = Track.wallRoom(400, 300, 200, 0, 1, 30, 800, 600);
        double tight = Track.wallRoom(400, 50, 200, 0, -1, 30, 800, 600);
        assertTrue(open >= 1.0);
        assertTrue(tight < open);
    }

    @Test void knnReturnsClosest() {
        var k = new Knn(new double[]{1}, 10);
        for (int i = 0; i < 10; i++) k.add(new double[]{i}, i, 1);
        var near = k.nearest(new double[]{3.1}, 2);
        double a = near.get(0).e().gf(), b = near.get(1).e().gf();
        assertEquals(java.util.Set.of(3.0, 4.0), java.util.Set.of(a, b));
    }
}
