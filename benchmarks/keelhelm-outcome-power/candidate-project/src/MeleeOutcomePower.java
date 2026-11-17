import java.util.*;

/** Actual legal own shot feedback only; no virtual accuracy or observer data. */
final class MeleeOutcomePower {
    record Key(int target, int bin) {}
    static final class Window {
        final ArrayDeque<Boolean> values=new ArrayDeque<>(); int successes;
        void add(boolean hit) {
            values.addLast(hit);if(hit)successes++;
            if(values.size()>32 && values.removeFirst())successes--;
        }
    }
    record Choice(int target,int bin,double baseline,double power,double range,double ownEnergy,
                  double targetEnergy,int age,int n,int successes,long context) {}
    record Pending(int turn,Choice choice) {}
    static final class Shot {
        final int id; final Choice choice; int callbackTurn=-1,mask; double damage;
        Shot(int id,Choice choice){this.id=id;this.choice=choice;}
    }
    private final Map<Key,Window> windows=new HashMap<>();
    private final LinkedHashMap<Integer,Shot> shots=new LinkedHashMap<>();
    private final Map<Integer,Integer> changedTargets=new TreeMap<>();
    private final StringBuilder trace=new StringBuilder();
    private Pending pending;
    private long sequence;
    private int records,overflow,confirmed,resolved,successful,evicted,changed,lastTurn;
    private boolean finished;

    void newRound() {
        shots.clear();pending=null;changedTargets.clear();trace.setLength(0);
        records=overflow=confirmed=resolved=successful=evicted=changed=lastTurn=0;finished=false;
    }
    void clearPending(){pending=null;}
    static int bin(double range){return range<250?0:range<500?1:2;}
    Choice choice(int target,double range,double own,double enemy,int age,boolean enabled) {
        int bin=bin(range);Window w=windows.get(new Key(target,bin));int n=w==null?0:w.values.size(),h=w==null?0:w.successes;
        double base=MeleePolicy.power(range,own,enemy),power=base;
        if(enabled && range<500 && age>=0 && age<=4 && own>=45 && enemy>=8 && n==32 && 10*(h+2)>=3*(n+8)) {
            double kill=enemy<=4?enemy/4+.05:(enemy+2)/6+.05;
            double proposed=Math.min(2.6,Math.min(base+.4,kill));
            if(proposed-base>=.2)power=proposed;
        }
        return new Choice(target,bin,base,power,range,own,enemy,age,n,h,sequence);
    }
    void pending(int turn,Choice choice){pending=new Pending(turn+1,choice);}
    private void emit(String value) {
        sequence++;records++;
        if(records>4096){overflow++;return;}
        if(trace.length()>0)trace.append(';');trace.append(sequence).append(',').append(value);
    }
    void confirm(int turn,int bulletId,double power) {
        Pending p=pending;pending=null;
        if(p==null || p.turn!=turn || Math.abs(p.choice.power-power)>1e-6)return;
        if(shots.size()>=256) {
            Shot old=shots.remove(shots.keySet().iterator().next());evicted++;
            emit("E,"+turn+","+old.id+","+old.choice.target+","+old.choice.bin);
        }
        Choice c=p.choice;shots.put(bulletId,new Shot(bulletId,c));confirmed++;
        if(power-c.baseline>.1){changed++;changedTargets.merge(c.target,1,Integer::sum);}
        emit("F,"+c.context+","+(turn-1)+","+turn+","+bulletId+","+c.target+","+c.bin+","+power+","+c.baseline+","+c.power+","+c.range+","+c.ownEnergy+","+c.targetEnergy+","+c.age+","+c.n+","+c.successes);
    }
    void terminal(int turn,int bulletId,int mask,double damage) {
        Shot shot=shots.get(bulletId);if(shot==null)return;
        shot.callbackTurn=Math.max(shot.callbackTurn,turn);shot.mask|=mask;shot.damage+=damage;
    }
    /** Complete the previous tick's callback set before teaching unsuccessful outcomes. */
    void advance(int turn) {
        lastTurn=turn;
        var it=shots.values().iterator();
        while(it.hasNext()) {
            Shot shot=it.next();
            if(shot.mask!=0 && shot.callbackTurn<turn){resolve(shot,turn,false);it.remove();}
        }
    }
    private void resolve(Shot shot,int turn,boolean end) {
        boolean hit=shot.damage>0;Choice c=shot.choice;
        windows.computeIfAbsent(new Key(c.target,c.bin),k->new Window()).add(hit);
        resolved++;if(hit)successful++;
        emit("R,"+turn+","+shot.id+","+c.target+","+c.bin+","+shot.callbackTurn+","+shot.mask+","+shot.damage+","+(hit?1:0)+","+(end?1:0));
    }
    /** A round-end cutoff establishes a known positive, never a not-yet-complete negative. */
    void finishRound() {
        if(finished)return;finished=true;
        var it=shots.values().iterator();
        while(it.hasNext()) {
            Shot shot=it.next();if(shot.damage>0){resolve(shot,Math.max(lastTurn,shot.callbackTurn),true);it.remove();}
        }
        pending=null;
    }
    String trace(){return trace.length()==0?"none":trace.toString();}
    String stats() {
        StringJoiner names=new StringJoiner(",");changedTargets.forEach((k,v)->names.add(k+":"+v));
        return " powerConfirmed="+changed+" powerTargets="+(changedTargets.isEmpty()?"none":names)
                +" outcomeConfirmed="+confirmed+" outcomeResolved="+resolved+" outcomeSuccesses="+successful
                +" outcomeCensored="+(evicted+shots.size())+" powerTraceOverflow="+overflow;
    }
}
