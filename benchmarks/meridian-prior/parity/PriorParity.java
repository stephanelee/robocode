import java.nio.file.*;
import java.util.*;
class PriorParity {
    public static void main(String[] args) throws Exception {
        var gun = CrossingGun.loadPrior();
        double largest = 0;
        int count = 0;
        for (String line : Files.readAllLines(Path.of(args[0]))) {
            double[] v = Arrays.stream(line.split(" ")).mapToDouble(Double::parseDouble).toArray();
            var context = new CrossingGun.Context(v[4], v[5], v[6], v[7]);
            var launch = new CrossingGun.Launch(1, 0, 0, v[3], v[0], v[1], v[2], context);
            largest = Math.max(largest, Math.abs(Flight.delta(gun.aim(launch) - v[8])));
            count++;
        }
        if (largest > 1e-8) throw new AssertionError(largest);
        System.out.println(count + " predictions match; largest angular difference " + largest);
    }
}
