import java.nio.file.*;
import java.io.IOException;
import java.util.*;

/** Optional bounded observation only: no logged value is read by a controller. I/O only at boundaries. */
final class DiagnosticTrace {
    static final int MAX_RECORDS=60000, MAX_BYTES=32*1024*1024;
    private final Path path;
    private final StringBuilder buffer=new StringBuilder();
    private final Map<Integer,Shot> shots=new HashMap<>();
    private PressureGun.AimTrace aim;
    private int ownTurn=-1;
    private double[] decisionOwn;
    private int round,records,dropped,bytes;
    private String failure;
    private boolean closed;
    private record Shot(long id,PressureGun.AimTrace aim,double x,double y) {}
    DiagnosticTrace(Path path,Path... forbidden) {
        this.path=path;
        if(path!=null && (!path.isAbsolute() || Files.exists(path) || Files.exists(Path.of(path+".status.json"))))
            throw new IllegalArgumentException("Diagnostic output must be separate, absolute and absent");
        if(path!=null)for(Path other:forbidden)if(path.equals(other)||Path.of(path+".status.json").equals(other))
            throw new IllegalArgumentException("Diagnostic output/sidecar must not overlap policy state");
    }
    boolean enabled(){return path!=null;}
    void startRound(int round){this.round=round;shots.clear();aim=null;decisionOwn=null;ownTurn=-1;closed=false;}
    void ownDecision(int turn,double[] state){if(enabled()&&!closed){ownTurn=turn;decisionOwn=state.clone();}}
    void aim(PressureGun.AimTrace value){if(enabled()&&!closed)aim=value;}
    void shot(int turn,long id,double x,double y,double power,double direction,boolean matched) {
        if(!enabled()||closed)return;
        var a=matched&&aim!=null&&aim.turn()+1==turn?aim:null;
        if(shots.size()<4096)shots.put(turn,new Shot(id,a,x,y));
        var q=a==null?null:a.query();
        emit("shot",turn,"shotId",id,"decisionTurn",a==null?null:a.turn(),"originX",x,"originY",y,
                "power",power,"direction",direction,"aimDirection",a==null?null:a.chosen(),
                "model",a==null?null:a.model(),"learned",a==null?null:a.model()==3,
                "decisionGunDirection",a==null?null:a.gun(),"ownX",a==null?null:a.x(),"ownY",a==null?null:a.y(),
                "ownState",a!=null&&ownTurn==a.turn()?decisionOwn:null,
                "energy",a==null?null:a.energy(),"heat",a==null?null:a.heat(),
                "features",q==null?null:q.features(),"linearGF",q==null?null:q.linearGF(),
                "predictedGF",q==null||a==null?null:CombatMath.relativeAngle(a.chosen()-q.reference())*q.sign()/q.escape(),
                "support",a==null?null:a.support(),"kthDistance",a==null?null:a.distance(),
                "escape",q==null?null:q.escape(),"referenceBearing",q==null?null:q.reference(),"sign",q==null?null:q.sign());
    }
    void passage(ContextGun.Sample sample) {
        if(!enabled() || closed)return;
        var shot=shots.get(sample.shotTurn());var a=shot==null?null:shot.aim();
        double reference=CombatMath.bearing(sample.referenceX()-sample.originX(),sample.referenceY()-sample.originY());
        double observed=CombatMath.bearing(sample.observedX()-sample.originX(),sample.observedY()-sample.originY());
        double raw=CombatMath.relativeAngle(observed-reference)*sample.sign()/sample.escape();
        emit("passage",sample.observedTurn(),"shotId",shot==null?null:shot.id(),"creationTurn",sample.shotTurn(),
                "decisionTurn",sample.aimTurn(),"x",sample.observedX(),"y",sample.observedY(),
                "predictedDirection",a==null?null:a.chosen(),"referenceBearing",reference,"escape",sample.escape(),
                "sign",sample.sign(),"predictedGF",a==null?null:CombatMath.relativeAngle(a.chosen()-reference)*sample.sign()/sample.escape(),
                "observedGF",sample.guessFactor(),"rawObservedGF",raw,"support",a==null?null:a.support(),
                "kthDistance",a==null?null:a.distance());
    }
    void virtual(int turn,int creation,int model,boolean hit) {
        if(!enabled()||closed)return;var shot=shots.get(creation);
        emit("virtual",turn,"shotId",shot==null?null:shot.id(),"creationTurn",creation,"model",model,"hit",hit);
    }
    void emit(String type,int turn,Object... values) {
        if(!enabled() || closed)return;
        var row=new StringBuilder("{\"type\":").append(json(type)).append(",\"round\":").append(round).append(",\"turn\":").append(turn);
        for(int i=0;i<values.length;i+=2)row.append(',').append(json(values[i])).append(':').append(json(values[i+1]));
        row.append("}\n");int size=row.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8).length;
        if(records>=MAX_RECORDS || bytes+size>MAX_BYTES){dropped++;return;}
        buffer.append(row);records++;bytes+=size;
    }
    void finishRound(int turn,int actualShots,int actualHits,double energy) {
        if(!enabled() || closed)return;
        emit("round",turn,"shots",actualShots,"hits",actualHits,"energySpent",energy,"truncated",dropped>0,
                "droppedRecords",dropped,"failed",failure!=null);
        try {
            Files.createDirectories(path.getParent());
            Files.writeString(path,buffer,StandardOpenOption.CREATE,StandardOpenOption.APPEND);buffer.setLength(0);
            Files.writeString(Path.of(path+".status.json"),"{\"schemaVersion\":1,\"records\":"+records+",\"bytes\":"+bytes+
                    ",\"droppedRecords\":"+dropped+",\"roundsCompleted\":"+round+",\"failed\":"+(failure!=null)+
                    ",\"failure\":"+json(failure)+"}\n");
        } catch(IOException e){failure=e.getClass().getSimpleName()+": "+e.getMessage();}
        closed=true;
    }
    static String json(Object value) {
        if(value==null)return "null";
        if(value instanceof Number n)return Double.isFinite(n.doubleValue())?value.toString():"null";
        if(value instanceof Boolean)return value.toString();
        if(value instanceof double[] a){var items=new ArrayList<String>();for(double v:a)items.add(json(v));return "["+String.join(",",items)+"]";}
        if(value instanceof Object[] a){var items=new ArrayList<String>();for(Object v:a)items.add(json(v));return "["+String.join(",",items)+"]";}
        return "\""+value.toString().replace("\\","\\\\").replace("\"","\\\"").replace("\n","\\n").replace("\r","\\r")+"\"";
    }
}
