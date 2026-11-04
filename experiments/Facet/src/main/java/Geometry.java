import java.util.*;
final class Geometry {
  record Arc(double lo, double hi) {
    Arc {
      Physics.finite(lo, hi);
      if (lo < 0 || hi > Physics.TAU + 1e-9 || hi < lo)
        throw new IllegalArgumentException("arc");
    }
    double width() {
      return hi - lo;
    }
  }
  static List<Arc> around(double center, double half) {
    if (half >= Math.PI)
      return List.of(new Arc(0, Physics.TAU));
    double lo = Physics.norm(center - half), hi = Physics.norm(center + half);
    return lo <= hi ? List.of(new Arc(lo, hi)) : List.of(new Arc(0, hi), new Arc(lo, Physics.TAU));
  }
  static List<Arc> merge(Collection<Arc> arcs) {
    List<Arc> sorted = new ArrayList<>(arcs);
    sorted.sort(Comparator.comparingDouble(Arc::lo));
    List<Arc> out = new ArrayList<>();
    for (Arc a : sorted) {
      if (out.isEmpty() || a.lo > out.get(out.size() - 1).hi + 1e-12)
        out.add(a);
      else {
        Arc b = out.remove(out.size() - 1);
        out.add(new Arc(b.lo, Math.max(b.hi, a.hi)));
      }
    }
    return List.copyOf(out);
  }
  static double width(List<Arc> arcs) {
    return merge(arcs).stream().mapToDouble(Arc::width).sum();
  }
  static double coverage(List<Arc> footprint, List<Arc> shadows) {
    double total = width(footprint), over = 0;
    for (Arc a : merge(footprint))
      for (Arc b : merge(shadows)) over += Math.max(0, Math.min(a.hi, b.hi) - Math.max(a.lo, b.lo));
    return total <= 1e-12 ? 0 : Physics.clamp(over / total, 0, 1);
  }
  /**
   * Conservative swept-circle shell intersection. 16 segments, radius inflated by half-segment
   * length <= .25 at legal speed. This overcovers angularly, never claims an exact engine trace.
   */
  static List<Arc> swept(
      Physics.V source, Physics.V from, Physics.V to, double inner, double outer) {
    List<Arc> out = new ArrayList<>();
    double inflate = from.distance(to) / 32;
    for (int i = 0; i < 16; i++) {
      Physics.V p = from.add(to.sub(from).mul((i + .5) / 16));
      double d = source.distance(p), r = 18 + inflate;
      if (d + r + 1e-9 < inner || d - r - 1e-9 > outer)
        continue;
      if (d <= r)
        return List.of(new Arc(0, Physics.TAU));
      out.addAll(around(source.bearing(p), Math.asin(Math.min(1, r / d))));
    }
    return merge(out);
  }
  static double cdf(double x) {
    double z = Math.abs(x), t = 1 / (1 + .2316419 * z),
           poly = t
        * (.319381530
            + t * (-.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    double tail = .3989422804014327 * Math.exp(-.5 * z * z) * poly;
    return x >= 0 ? 1 - tail : tail;
  }
  static double truncatedGaussian(double x, double mean, double sigma, double limit) {
    if (Math.abs(x) > limit)
      return 0;
    return gaussian(x - mean, sigma)
        / Math.max(1e-9, cdf((limit - mean) / sigma) - cdf((-limit - mean) / sigma));
  }
  static double gaussian(double x, double sigma) {
    double z = x / Math.max(1e-4, sigma);
    return Math.exp(-.5 * z * z) / (Math.sqrt(2 * Math.PI) * Math.max(1e-4, sigma));
  }
  interface Density {
    double at(double angle);
    default double upperBound() {
      return Double.POSITIVE_INFINITY;
    }
  }
  static double integrate(List<Arc> arcs, Density density) {
    if (arcs.isEmpty())
      return 1;
    int count = 0;
    for (Arc a : arcs) count += 8 * Math.max(1, (int) Math.ceil(a.width() / .2)) + 1;
    if (count > 81) {
      double bound = density.upperBound();
      if (bound < 0 || Double.isNaN(bound))
        throw new IllegalArgumentException("invalid density bound");
      return bound * width(arcs);
    }
    double sum = 0;
    for (Arc a : arcs) {
      int steps = 8 * Math.max(1, (int) Math.ceil(a.width() / .2));
      double h = a.width() / steps;
      for (int i = 0; i <= steps; i++)
        sum += density.at(a.lo + i * h) * h * (i == 0 || i == steps ? .5 : 1);
    }
    return Math.max(1e-9, sum);
  }
  static double segmentDistance(Physics.V p, Physics.V a, Physics.V b) {
    Physics.V d = b.sub(a);
    double n = d.x() * d.x() + d.y() * d.y();
    double t = n < 1e-12
        ? 0
        : Physics.clamp(((p.x() - a.x()) * d.x() + (p.y() - a.y()) * d.y()) / n, 0, 1);
    return p.distance(a.add(d.mul(t)));
  }
}
