import java.util.*;
final class Memory {
  enum Quality { OBSERVED, INTERPOLATED, PREDICTED }
  enum Label { PRECISE_GF, THEORETICAL_GF, DISPLACEMENT, POWER }
  record Sample(
      int round, int turn, Physics.Body body, double energy, Quality quality, double uncertainty) {
    Sample {
      Physics.finite(energy, uncertainty);
    }
  }
  static final class History {
    final NavigableMap<Integer, Sample> samples = new TreeMap<>();
    int round = -1;
    void add(Sample s) {
      if (round != s.round) {
        samples.clear();
        round = s.round;
      }
      samples.put(s.turn, s);
      while (samples.size() > 512) samples.pollFirstEntry();
    }
    Optional<Sample> at(int t) {
      Sample exact = samples.get(t);
      if (exact != null)
        return Optional.of(exact);
      var a = samples.floorEntry(t);
      var b = samples.ceilingEntry(t);
      if (a == null || b == null || b.getKey() - a.getKey() > 8)
        return Optional.empty();
      Sample x = a.getValue(), y = b.getValue();
      double f = (t - x.turn) / (double) (y.turn - x.turn);
      Physics.Body p = x.body, q = y.body;
      return Optional.of(new Sample(round, t,
          new Physics.Body(p.p().add(q.p().sub(p.p()).mul(f)),
              Physics.norm(p.heading() + f * Physics.wrap(q.heading() - p.heading())),
              p.speed() + f * (q.speed() - p.speed()), p.gun(), p.radar(), p.heat()),
          x.energy + f * (y.energy - x.energy), Quality.INTERPOLATED, 1));
    }
    double displacement(int t, int horizon) {
      if (samples.isEmpty())
        return 0;
      Sample now = samples.lastEntry().getValue(),
             old = at(t - horizon).orElse(samples.firstEntry().getValue());
      return now.body.p().distance(old.body.p());
    }
    void clear() {
      samples.clear();
      round = -1;
    }
  }
  record Observation(double[] features, Label type, double a, double b, long wave, int round,
      int turn, double quality, long sequence, boolean melee) {
    Observation {
      features = features.clone();
      Physics.finite(features);
      Physics.finite(a, b, quality);
      if (quality <= 0)
        throw new IllegalArgumentException("quality");
    }
    @Override
    public double[] features() {
      return features.clone();
    }
  }
  record Neighbor(Observation observation, double distance) {
    double weight() {
      return observation.quality / Math.sqrt(Math.max(distance, 1e-6));
    }
  }
  static final class Budget {
    static final long LIMIT = 128L * 1024 * 1024;
    long bytes, evictions;
    final List<Store> stores = new ArrayList<>();
    void enforce() {
      while (bytes > LIMIT) {
        Store victim = null;
        long oldest = Long.MAX_VALUE;
        for (Store s : stores)
          if (!s.data.isEmpty() && s.priority == 0 && s.data.peekFirst().sequence < oldest) {
            victim = s;
            oldest = s.data.peekFirst().sequence;
          }
        if (victim == null)
          for (Store s : stores)
            if (!s.data.isEmpty() && s.data.peekFirst().sequence < oldest) {
              victim = s;
              oldest = s.data.peekFirst().sequence;
            }
        if (victim == null)
          break;
        victim.evict();
        evictions++;
      }
    }
  }
  static final class Store {
    final ArrayDeque<Observation> data = new ArrayDeque<>();
    final int capacity, dimension, priority;
    final Label label;
    final Budget budget;
    long version;
    Store(int cap, int dim, Label label, Budget budget, int priority) {
      if (cap <= 0 || dim <= 0)
        throw new IllegalArgumentException();
      capacity = cap;
      dimension = dim;
      this.label = label;
      this.budget = budget;
      this.priority = priority;
      budget.stores.add(this);
    }
    int size() {
      return data.size();
    }
    long cost() {
      return 112L + 8L * dimension;
    }
    void evict() {
      if (data.pollFirst() != null) {
        budget.bytes -= cost();
        version++;
      }
    }
    void add(Observation o) {
      if (o.type != label || o.features.length != dimension)
        throw new IllegalArgumentException("label/schema");
      if (data.size() == capacity)
        evict();
      data.addLast(o);
      budget.bytes += cost();
      version++;
      budget.enforce();
    }
    List<Neighbor> query(double[] q, double[] w, int k) {
      Physics.finite(q);
      Physics.finite(w);
      if (q.length != dimension || w.length != dimension || k < 0)
        throw new IllegalArgumentException("query schema");
      if (k == 0)
        return List.of();
      Comparator<Neighbor> order = Comparator.comparingDouble(Neighbor::distance)
                                       .thenComparingLong(n -> n.observation.sequence);
      PriorityQueue<Neighbor> heap = new PriorityQueue<>(Math.max(1, k), order.reversed());
      for (Observation o : data) {
        double d = 0;
        for (int i = 0; i < dimension; i++) {
          double z = w[i] * (o.features[i] - q[i]);
          d += z * z;
        }
        if (!Double.isFinite(d))
          throw new IllegalArgumentException("distance overflow");
        Neighbor n = new Neighbor(o, d);
        if (heap.size() < k)
          heap.add(n);
        else if (order.compare(n, heap.peek()) < 0) {
          heap.poll();
          heap.add(n);
        }
      }
      List<Neighbor> out = new ArrayList<>(heap);
      out.sort(order);
      return List.copyOf(out);
    }
  }
}
