/** Synthetic contract checks only; no recording or battle process. */
public class CooldownShadowFixtures {
    static final Phys.Pose SELF = new Phys.Pose(500,500,0,0);
    static void check(boolean ok, String message) { if (!ok) throw new AssertionError(message); }
    static CooldownShadow shadow() {
        var s = new CooldownShadow();
        s.width = s.height = 1000;
        s.cooling = 0.1;
        s.policy.newRound(3);
        return s;
    }
    static MeleePolicy.Opponent add(CooldownShadow s, int id, int t, double x, double y) {
        s.policy.observe(id,t,x,y,0,0,100);
        return s.policy.opponents().stream().filter(o -> o.id == id).findFirst().orElseThrow();
    }
    public static void main(String[] args) {
        var s = shadow();
        var primary = add(s,1,10,600,500);
        add(s,2,10,600,510);
        var hot = new CooldownShadow.Gun(.1,0);
        s.decideGun(hot,true,primary,10,SELF,60);
        check(hot.commitment == -1, "reachable incumbent protected");
        var a = shadow();
        var behind = add(a,1,10,400,500);
        add(a,3,10,600,510);
        add(a,2,10,600,490);
        var turn = a.decideGun(hot,true,behind,10,SELF,60);
        check(hot.commitment == 2 && turn.turn() < 0,"tie by ID and reachable alternate");
        var oldRay = hot.ray;
        hot.advance(turn,0.1);
        check(hot.heat == 0 && hot.ray != oldRay, "cool first, rotate for next decision");
        var shot = a.decideGun(hot,true,behind,11,SELF,60);
        check(shot.shot() != null && shot.shot().target().id == 2 && hot.commitment == -1,
                "cold fire uses pre-turn ray and releases commitment");
        hot.advance(shot,0.1);
        check(hot.heat == Phys.gunHeat(shot.cost()), "full heat on fire");
        var previousAim = add(a,4,12,400,500);
        var coolBase = new CooldownShadow.Gun(0,180);
        var displaced = a.decideGun(coolBase,false,previousAim,12,SELF,60);
        check(displaced.shot() != null && a.decideGun(hot,true,previousAim,12,SELF,60).shot() == null,
                "extra shot can displace next incumbent shot");
        hot.commitment = 2;
        a.policy.died(2);
        a.decideGun(hot,true,behind,12,SELF,60);
        check(hot.commitment != 2,"dead target discarded");
        var wall = add(a,5,10,981,510);
        wall.speed = 8;
        var p = MeleePolicy.intercept(wall,10,SELF,1.6,1000,1000);
        check(p.x() <= 982 && p.x() >= 18 && p.y() <= 982 && p.y() >= 18, "wall-constrained intercept");
        a.base = hot;
        a.baseMinusShadowCost = -2;
        a.endRound();
        check(a.base == null && a.baseMinusShadowCost == 0, "round reset");
        var h = new CooldownShadow.Gun(.2,350);
        h.advance(new CooldownShadow.Decision(100,null),.1);
        check(h.heat == .1 && h.ray == 10, "clamped physical turn and wrap");
        check(a.decideGun(h,true,behind,12,SELF,0).turn() == 0, "disabled at zero energy");
        System.out.println("Cooldown shadow contracts passed");
    }
}
