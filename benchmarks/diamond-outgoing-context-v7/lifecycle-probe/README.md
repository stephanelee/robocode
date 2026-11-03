# PR65 round-snapshot lifecycle reachability probe

This is a deterministic local method-order probe, not a battle or a reproduction of the actual PR65 callback trace. It establishes a source vulnerability that could produce the observed invariant mismatch; it does **not** establish that this ordering occurred in PR65. No controller, gameplay, model settings, or experiment evidence was changed.

Exact measured source: `e845711924382baf8b7e6661e70119f3352b069d`.

The probe executes the unchanged frozen JAR from cycle1:

- SHA256 `59de57bd2d2f57a9c54542bf7dca60d09bd21e7b0760e21bdcdeb0a1797e3ddc`
- Git blob `b35f65e3eff0af41c1c6149337f632a368a63aef`, equal to the JAR blob at the exact measured source.
- Java `17.0.18`.

`RoundSnapshotProbe.java` uses only ordinary package-visible controller methods, without reflection or state injection. Thirty-two confirmed, consecutively scanned passage labels make a learned candidate supported. A further actually confirmed simulated shot creates a pending paired cohort. The probe takes a conceptual round-metric snapshot at turn4010, before the next fresh scan resolves that cohort. It then invokes the final scan, followed by `startRound()` and a second conceptual round snapshot.

Observed unchanged-JAR output:

```text
round1_snapshot completed=0 persistent=[0, 0, 0, 0] pending=1
late_scan4011 completed=1 persistent=[1, 1, 1, 1] pending=0
round2_snapshot completed=0 persistent=[1, 1, 1, 1] pending=0
sum_snapshot_rows=0 endpoint=1 difference=1
REACHABLE_METHOD_ORDER_ONLY; ACTUAL_PR65_CALLBACK_CAUSE_UNPROVEN
```

The source permits this method order because `PressureGun.onScan` has no closed-round guard and increments both persistent paired attempts and the current round completion counter. `startRound` clears the round counter while preserving paired attempts. `DiamondChallenger.onRoundEnded` snapshots these counters but does not close gun mutation. If a scan arrives between that snapshot and the next start, the next endpoint can contain one extra paired completion absent from all captured rows. The observed PR65 difference531−530=1 is compatible with this vulnerability; compatibility is not proof of its actual cause. Actual callback order was not captured by PR65.

Reproduction from cycle3 (PowerShell):

```powershell
& 'C:/Program Files/Microsoft/jdk-17.0.18.8-hotspot/bin/javac.exe' -cp C:/Users/admin2/Downloads/wsl/robocode-improve-cycle1/experiments/DiamondChallenger/DiamondChallenger.jar -d review/pr65-lifecycle-probe/classes review/pr65-lifecycle-probe/RoundSnapshotProbe.java
& 'C:/Program Files/Microsoft/jdk-17.0.18.8-hotspot/bin/java.exe' -cp 'review/pr65-lifecycle-probe/classes;C:/Users/admin2/Downloads/wsl/robocode-improve-cycle1/experiments/DiamondChallenger/DiamondChallenger.jar' RoundSnapshotProbe
```

No fix or retry is proposed or executed by this probe. The invalid PR65 outcome remains invalid and its full evidence must remain intact.
