import dev.robocode.tankroyale.botapi.*;
import dev.robocode.tankroyale.botapi.events.*;
public class SkipProbe extends Bot {
    public static void main(String[] a) { new SkipProbe().start(); }
    SkipProbe() { super(BotInfo.fromFile("SkipProbe.json")); }
    @Override public void run() { while (isRunning()) go(); }
    @Override public void onSkippedTurn(SkippedTurnEvent e) {
        try (var w = new java.io.FileWriter("/tmp/claude-0/-home-user-robocode/7c4b824f-ae59-50fe-97af-f5f65cc4f3ae/scratchpad/probe/skips.txt", true)) {
            w.write("probe round " + getRoundNumber() + " turn " + e.getTurnNumber() + "\n");
        } catch (java.io.IOException ex) { }
    }
}
