import java.util.*;
/** Native Cartesian radians. Motion assumptions require real-engine acceptance. */
final class Physics {
  static final double TAU = 2 * Math.PI, RADIUS = 18, MARGIN = 1;
  static double clamp(double x, double a, double b) {
    return Math.max(a, Math.min(b, x));
  }
  static double wrap(double a) {
    return a - TAU * Math.floor((a + Math.PI) / TAU);
  }
  static double norm(double a) {
    return a - TAU * Math.floor(a / TAU);
  }
  static void finite(double... a) {
    for (double x : a)
      if (!Double.isFinite(x))
        throw new IllegalArgumentException("nonfinite");
  }
  record V(double x, double y) {
    V {
      finite(x, y);
    }
    V add(V b) {
      return new V(x + b.x, y + b.y);
    }
    V sub(V b) {
      return new V(x - b.x, y - b.y);
    }
    V mul(double k) {
      return new V(x * k, y * k);
    }
    double length() {
      return Math.hypot(x, y);
    }
    double distance(V b) {
      return sub(b).length();
    }
    double bearing(V b) {
      return Math.atan2(b.y - y, b.x - x);
    }
  }
  static V vector(double h) {
    return new V(Math.cos(h), Math.sin(h));
  }
  record Rules(double width, double height, double cooling) {
    Rules {
      finite(width, height, cooling);
      if (width < 38 || height < 38 || cooling <= 0)
        throw new IllegalArgumentException("rules");
    }
    boolean safe(V p) {
      return p.x >= 19 && p.y >= 19 && p.x <= width - 19 && p.y <= height - 19;
    }
    V bound(V p, double margin) {
      return new V(clamp(p.x, margin, width - margin), clamp(p.y, margin, height - margin));
    }
  }
  record Body(V p, double heading, double speed, double gun, double radar, double heat) {
    Body {
      finite(heading, speed, gun, radar, heat);
    }
  }
  record Control(double speed, double turn, double gunTurn, double radarTurn) {
    Control {
      finite(speed, turn, gunTurn, radarTurn);
    }
    static Control stop() {
      return new Control(0, 0, 0, 0);
    }
  }
  record
      Step(Body body, boolean wall, double damage, double remainingTurn, double remainingDistance) {
  }
  static double bulletSpeed(double p) {
    return 20 - 3 * p;
  }
  static double damage(double p) {
    return 4 * p + 2 * Math.max(0, p - 1);
  }
  static double effective(Body s) {
    return norm(s.heading + (s.speed < 0 ? Math.PI : 0));
  }
  static double maxTurn(double speed) {
    return Math.toRadians(10 - .75 * Math.abs(speed));
  }
  // Official 1.4.0 law: STOP instantaneous, fractional reversal.
  static double velocity(double old, double target) {
    target = clamp(target, -8, 8);
    if (old < 0)
      return -velocity(-old, -target);
    if (old > 0) {
      if (target > 0)
        return old + clamp(target - old, -2, 1);
      return target == 0 ? 0 : old / 2 - 1;
    }
    return clamp(target, -1, 1);
  }
  static V segmentBound(V old, V raw, Rules r) {
    double x = raw.x, y = raw.y, nx = x, ny = y;
    if (x < 18 || x > r.width - 18) {
      nx = clamp(x, 18, r.width - 18);
      if (x != old.x)
        ny = old.y + (y - old.y) * (nx - old.x) / (x - old.x);
    }
    if (y < 18 || y > r.height - 18) {
      ny = clamp(y, 18, r.height - 18);
      if (y != old.y)
        nx = old.x + (x - old.x) * (ny - old.y) / (y - old.y);
    }
    return new V(nx, ny);
  }
  static V constantPosition(Body target, double time, Rules r) {
    V raw = target.p.add(vector(target.heading).mul(target.speed * time));
    return segmentBound(target.p, segmentBound(target.p, raw, r), r);
  }
  static final class WorkLimit extends RuntimeException {}
  static final class Work {
    int used, limit;
    Work(int limit) {
      this.limit = limit;
    }
  }
  static final ThreadLocal<Work> WORK = new ThreadLocal<>();
  static void beginWork(int limit) {
    WORK.set(new Work(limit));
  }
  static int workRemaining() {
    return WORK.get() == null ? Integer.MAX_VALUE : WORK.get().limit - WORK.get().used;
  }
  static int workUsed() {
    return WORK.get() == null ? 0 : WORK.get().used;
  }
  static void endWork() {
    WORK.remove();
  }
  static Step step(Body s, Control c, Rules r) {
    Work work = WORK.get();
    if (work != null) {
      if (work.used >= work.limit)
        throw new WorkLimit();
      work.used++;
    }

    double v = velocity(s.speed, c.speed), t = clamp(c.turn, -maxTurn(v), maxTurn(v));
    V raw = s.p.add(vector(s.heading).mul(v)), p = segmentBound(s.p, segmentBound(s.p, raw, r), r);
    boolean wall = p.distance(raw) > 1e-10;
    return new Step(new Body(p, norm(s.heading + t), wall ? 0 : v,
                        norm(s.gun + clamp(c.gunTurn, -Math.toRadians(20), Math.toRadians(20))),
                        norm(s.radar + clamp(c.radarTurn, -Math.toRadians(45), Math.toRadians(45))),
                        Math.max(0, s.heat - r.cooling)),
        wall, wall ? Math.max(0, Math.abs(v) * .5 - 1) : 0, c.turn - t, 0);
  }
  static Step finiteStep(Body s, double distance, double turn, Rules r) {
    double target = dev.robocode.tankroyale.botapi.internal.IntentValidator.getNewTargetSpeed(
        s.speed, distance, 8);
    Step z = step(s, new Control(target, turn, 0, 0), r);
    return new Step(z.body, z.wall, z.damage, z.remainingTurn, distance - z.body.speed);
  }
  static Control toward(Body s, V dest, boolean stop) {
    double a = wrap(s.p.bearing(dest) - s.heading), sign = 1;
    if (Math.abs(a) > Math.PI / 2) {
      a = wrap(a + Math.PI);
      sign = -1;
    }
    return new Control(stop ? 0 : 8 * sign, a, 0, 0);
  }
  static double gf(double bearing, double base, int sign, double pos, double neg) {
    double u = sign * wrap(bearing - base);
    return u / Math.max(1e-6, u >= 0 ? pos : neg);
  }
  static double decode(double g, double base, int sign, double pos, double neg) {
    return norm(base + sign * g * Math.max(1e-6, g >= 0 ? pos : neg));
  }
  static double escape(double p) {
    return Math.asin(Math.min(1, 8 / bulletSpeed(p)));
  }
  static double intercept(V source, Body target, double power, Rules r) {
    double t = source.distance(target.p) / bulletSpeed(power), old = -1;
    V p = target.p;
    for (int i = 0; i < 12; i++) {
      p = constantPosition(target, t, r);
      double n = source.distance(p) / bulletSpeed(power);
      if (Math.abs(n - t) < 1e-6)
        break;
      if (Math.abs(n - old) < 1e-6) {
        t = Math.max(t, n);
        p = constantPosition(target, t, r);
        break;
      }
      old = t;
      t = n;
    }
    return source.bearing(p);
  }
}
