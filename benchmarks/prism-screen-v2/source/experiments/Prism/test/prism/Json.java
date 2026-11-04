package prism;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Minimal JSON reader for recorded engine-trace fixtures (objects, arrays, numbers, strings, booleans). */
final class Json {
    private final String s;
    private int i;

    private Json(String s) { this.s = s; }

    static Object parse(String text) { Json j = new Json(text); Object v = j.value(); return v; }

    @SuppressWarnings("unchecked")
    static Map<String, Object> obj(String text) { return (Map<String, Object>) parse(text); }

    private void ws() { while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++; }

    private Object value() {
        ws();
        char c = s.charAt(i);
        if (c == '{') {
            i++;
            Map<String, Object> m = new LinkedHashMap<>();
            ws();
            if (s.charAt(i) == '}') { i++; return m; }
            while (true) {
                ws();
                String k = (String) value();
                ws(); i++; // :
                m.put(k, value());
                ws();
                if (s.charAt(i++) == '}') return m;
            }
        }
        if (c == '[') {
            i++;
            List<Object> l = new ArrayList<>();
            ws();
            if (s.charAt(i) == ']') { i++; return l; }
            while (true) {
                l.add(value());
                ws();
                if (s.charAt(i++) == ']') return l;
            }
        }
        if (c == '"') {
            int j = s.indexOf('"', i + 1);
            String r = s.substring(i + 1, j);
            i = j + 1;
            return r;
        }
        if (s.startsWith("true", i)) { i += 4; return Boolean.TRUE; }
        if (s.startsWith("false", i)) { i += 5; return Boolean.FALSE; }
        int j = i;
        while (j < s.length() && "+-0123456789.eE".indexOf(s.charAt(j)) >= 0) j++;
        double d = Double.parseDouble(s.substring(i, j));
        i = j;
        return d;
    }

    static double num(Map<String, Object> m, String k) { return (Double) m.get(k); }

    @SuppressWarnings("unchecked")
    static Map<String, Object> sub(Map<String, Object> m, String k) { return (Map<String, Object>) m.get(k); }

    @SuppressWarnings("unchecked")
    static List<Map<String, Object>> list(Map<String, Object> m, String k) { return (List<Map<String, Object>>) m.get(k); }
}
