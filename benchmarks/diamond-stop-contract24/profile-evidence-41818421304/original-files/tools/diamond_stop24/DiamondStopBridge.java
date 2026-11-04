import dev.robocode.tankroyale.server.rules.MathKt;
import java.io.*;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;

/** Acquisition-only main. Preparation compiles/inspects this class but never runs it. */
public final class DiamondStopBridge {
    private static final String SERVER_SHA = "16c277795bd823c9eda0b650f1d09a418ff35cfe43d59b820c52d5e6816f905c";
    private static final String MATH_RESOURCE = "dev/robocode/tankroyale/server/rules/MathKt.class";
    private static String hash(byte[] bytes) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    }
    private static String fileHash(Path path) throws Exception {
        var digest=MessageDigest.getInstance("SHA-256");
        try(var input=Files.newInputStream(path)) {byte[] buffer=new byte[65536];int n;
            while((n=input.read(buffer))!=-1)digest.update(buffer,0,n);}
        return HexFormat.of().formatHex(digest.digest());
    }
    private static String q(String value) {
        var out=new StringBuilder("\"");
        for(char c:value.toCharArray()) switch(c) {
            case '"' -> out.append("\\\""); case '\\' -> out.append("\\\\");
            case '\n' -> out.append("\\n");case '\r' -> out.append("\\r");case '\t' -> out.append("\\t");
            default -> {if(c<32)out.append(String.format(Locale.ROOT,"\\u%04x",(int)c));else out.append(c);}
        }
        return out.append('"').toString();
    }
    private static String number(double value) {
        if(!Double.isFinite(value))throw new IllegalStateException("Nonfinite bridge output");
        return "{\"value\":"+Double.toString(value)+",\"hex\":"+q(Double.toHexString(value))+
                ",\"bits\":"+q(String.format(Locale.ROOT,"%016x",Double.doubleToRawLongBits(value)))+"}";
    }
    private static String position(double x,double y) {
        return "{\"x\":"+number(x)+",\"y\":"+number(y)+"}";
    }
    private static Path source(Class<?> cls) throws Exception {
        if(cls.getProtectionDomain().getCodeSource()==null)throw new IllegalStateException("No CodeSource");
        return Path.of(cls.getProtectionDomain().getCodeSource().getLocation().toURI()).toRealPath();
    }
    private static String origins(Path server,Path bridge,String bridgeSha,String mathSha) throws Exception {
        String expectedClasspath=bridge+File.pathSeparator+server;
        if(!expectedClasspath.equals(System.getProperty("java.class.path")))throw new IllegalStateException("Unexpected classpath");
        for(String key:List.of("CLASSPATH","JAVA_TOOL_OPTIONS","_JAVA_OPTIONS","JDK_JAVA_OPTIONS"))
            if(System.getenv(key)!=null)throw new IllegalStateException("Unsanitized Java injection: "+key);
        if(!source(MathKt.class).equals(server)||!source(DiamondStopBridge.class).equals(bridge))
            throw new IllegalStateException("Loaded class CodeSource differs");
        var resources=Collections.list(ClassLoader.getSystemClassLoader().getResources(MATH_RESOURCE));
        if(resources.size()!=1)throw new IllegalStateException("Duplicate/missing MathKt resource");
        URL resource=resources.get(0);
        String classHash;
        try(var stream=resource.openStream()){classHash=hash(stream.readAllBytes());}
        String serverHash=fileHash(server),actualBridge=fileHash(bridge);
        if(!serverHash.equals(SERVER_SHA)||!actualBridge.equals(bridgeSha)||!classHash.equals(mathSha))
            throw new IllegalStateException("Loaded artifact/class hash differs");
        for(String name:List.of("calcNewBotSpeed","limitTurnRate")) {
            var method=MathKt.class.getDeclaredMethod(name,double.class,double.class);
            if(!java.lang.reflect.Modifier.isPublic(method.getModifiers())||
               !java.lang.reflect.Modifier.isStatic(method.getModifiers())||method.getReturnType()!=double.class)
                throw new IllegalStateException("Wrong actual MathKt signature");
        }
        return "{\"type\":\"java-provenance\",\"mathCodeSource\":"+q(source(MathKt.class).toString())+
                ",\"mathClassResource\":"+q(resource.toString())+",\"mathClassSHA256\":"+q(classHash)+
                ",\"serverJar\":"+q(server.toString())+",\"serverSHA256\":"+q(serverHash)+
                ",\"bridgeCodeSource\":"+q(source(DiamondStopBridge.class).toString())+
                ",\"bridgeSHA256\":"+q(actualBridge)+",\"classpath\":"+q(expectedClasspath)+
                ",\"javaHome\":"+q(System.getProperty("java.home"))+",\"javaVersion\":"+q(System.getProperty("java.version"))+"}";
    }

    /** Literal CollisionDetector.constrainBotPosition; each branch uses original candidate x/y. */
    private static double[] constrain(double oldX,double oldY,double x,double y) {
        double newX=x,newY=y;
        if(x-18<0) {
            newX=18;double dx=x-oldX;
            if(dx!=.0){double dy=y-oldY,dx2=newX-oldX,dy2=dy*dx2/dx;newY=oldY+dy2;}
        }else if(x+18>800) {
            newX=800-18;double dx=x-oldX;
            if(dx!=.0){double dy=y-oldY,dx2=newX-oldX,dy2=dy*dx2/dx;newY=oldY+dy2;}
        }
        if(y-18<0) {
            newY=18;double dy=y-oldY;
            if(dy!=.0){double dx=x-oldX,dy2=newY-oldY,dx2=dx*dy2/dy;newX=oldX+dx2;}
        }else if(y+18>600) {
            newY=600-18;double dy=y-oldY;
            if(dy!=.0){double dx=x-oldX,dy2=newY-oldY,dx2=dx*dy2/dy;newX=oldX+dx2;}
        }
        return new double[]{newX,newY};
    }
    private static String reference(String[] f) {
        double oldX=Double.parseDouble(f[1]),oldY=Double.parseDouble(f[2]);
        double heading=Double.parseDouble(f[3]),oldSpeed=Double.parseDouble(f[4]);
        double target=Double.parseDouble(f[5]),request=Double.parseDouble(f[6]);int time=Integer.parseInt(f[7]);
        // These are actual pinned public engine methods, exactly once per registered card each.
        double newSpeed=MathKt.calcNewBotSpeed(oldSpeed,target);
        double limitedTurn=MathKt.limitTurnRate(request,newSpeed);
        // Pinned Bot.moveToNewPosition uses old direction. Heading changes after translation, before walls.
        double radians=Math.toRadians(heading);
        double rawX=oldX+Math.cos(radians)*newSpeed,rawY=oldY+Math.sin(radians)*newSpeed;
        double normalAngle=(heading+limitedTurn)%360;
        double finalHeading=normalAngle>=0?normalAngle:normalAngle+360;
        double[] first=constrain(oldX,oldY,rawX,rawY);
        boolean hit=rawX!=first[0]||rawY!=first[1];
        double finalSpeed=hit?0.0:newSpeed;
        // Second positional safety pass uses original old position; no second speed/heading update.
        double[] second=constrain(oldX,oldY,first[0],first[1]);
        return "{\"type\":\"reference\",\"id\":"+q(f[0])+",\"newSpeedBeforeWall\":"+number(newSpeed)+
            ",\"sdkRequestedTurn\":"+number(request)+",\"limitedTurn\":"+number(limitedTurn)+
            ",\"unconstrained\":"+position(rawX,rawY)+",\"firstPass\":"+position(first[0],first[1])+
            ",\"secondPass\":"+position(second[0],second[1])+",\"firstPassConstrained\":"+hit+
            ",\"secondPassConstrained\":"+(first[0]!=second[0]||first[1]!=second[1])+
            ",\"final\":{\"x\":"+number(second[0])+",\"y\":"+number(second[1])+
            ",\"headingDegrees\":"+number(finalHeading)+",\"speed\":"+number(finalSpeed)+
            ",\"time\":"+(time+1)+"}}";
    }
    public static void main(String[] args) throws Exception {
        if(args.length!=4)throw new IllegalArgumentException("server.jar bridge.jar bridgeSHA256 MathKtSHA256");
        Path server=Path.of(args[0]).toRealPath(),bridge=Path.of(args[1]).toRealPath();
        var out=new PrintWriter(new OutputStreamWriter(System.out,StandardCharsets.UTF_8),true);
        out.println(origins(server,bridge,args[2],args[3]));
        int count=0;
        try(var reader=new BufferedReader(new InputStreamReader(System.in,StandardCharsets.UTF_8))){String line;
            while((line=reader.readLine())!=null){
                if(line.length()>4096)throw new IllegalArgumentException("Overlong input");
                String[] fields=line.split("\t",-1);
                if(fields.length!=8||count>=24||!fields[0].equals(String.format(Locale.ROOT,"%02d",count+1)))
                    throw new IllegalArgumentException("Expected exactly ordered registered IDs");
                out.println(reference(fields));count++;
            }
        }
        if(count!=24)throw new IllegalStateException("Incomplete24-case bridge process");
    }
}
