import java.util.*;
final class Radar {
  int direction = 1, mismatch, acquiredAfter = -1;
  double turn(Physics.Body next, Collection<Learner.Enemy> enemies, int now, int participantCount) {
    List<Learner.Enemy> alive =
        enemies.stream()
            .filter(e -> e.alive && e.latest != null && e.latest.turn() > acquiredAfter)
            .toList();
    if (alive.size() != participantCount) {
      if (++mismatch >= 24) {
        mismatch = 0;
        acquiredAfter = now;
        return direction * Math.PI / 4;
      }
    } else
      mismatch = 0;
    if (alive.size() != participantCount || alive.isEmpty())
      return direction * Math.PI / 4;
    Learner.Enemy target =
        alive.stream()
            .max(Comparator.<Learner.Enemy>comparingInt(e -> now - e.latest.turn())
                    .thenComparingDouble(e
                        ->
                        - Math.abs(
                            Physics.wrap(next.p().bearing(e.latest.body().p()) - next.radar())))
                    .thenComparingInt(e -> - e.id))
            .orElseThrow();
    int age = now - target.latest.turn();
    if (age >= 4)
      return direction * Math.PI / 4;
    double delta = Physics.wrap(next.p().bearing(target.latest.body().p()) - next.radar());
    int sign = Math.abs(delta) < 1e-8 ? direction : delta > 0 ? 1 : -1;
    if (alive.size() > 1 && sign != direction && Math.abs(delta) < Math.toRadians(10))
      sign = direction;
    direction = sign;
    double room = Math.max(0, Math.PI / 4 - Math.abs(delta)),
           over = Math.min(Math.toRadians(20), room);
    if (room >= Math.toRadians(2))
      over = Math.max(Math.toRadians(2), over);
    return Physics.clamp(delta + sign * over, -Math.PI / 4, Math.PI / 4);
  }
  void reset() {
    direction = 1;
    mismatch = 0;
    acquiredAfter = -1;
  }
}
