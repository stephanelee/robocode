import java.io.*;
import java.nio.file.*;
import java.security.*;
import java.util.*;
/** Export-only own derived state. Deliberately no loader; battle IDs are local to this game. */
final class ModelSnapshot {
  static final int SCHEMA = 2;
  static final long MAX_BYTES = 128L * 1024 * 1024;
  static byte[] configHash() {
    try (var in = ModelSnapshot.class.getResourceAsStream("/facet.properties")) {
      if (in == null)
        throw new IllegalStateException("config absent");
      return MessageDigest.getInstance("SHA-256").digest(in.readAllBytes());
    } catch (IOException | NoSuchAlgorithmException e) {
      throw new IllegalStateException(e);
    }
  }
  static void optional(Core core) {
    optional(core, 1);
  }
  static void optional(Core core, int game) {
    String value = System.getenv("FACET_STATE_PATH");
    if (value == null)
      value = System.getenv("FACET_MODEL_PATH");
    if (value == null || value.isBlank())
      return;
    Path path = Path.of(value).toAbsolutePath();
    if (!path.getFileName().toString().startsWith("FACET_"))
      throw new IllegalArgumentException("FACET_MODEL_PATH basename must start FACET_");
    if (game > 1)
      path = path.resolveSibling(path.getFileName() + ".game-" + game);
    Path partial = path.resolveSibling(path.getFileName() + ".partial");
    try {
      if (Files.exists(path))
        throw new IOException("refuse model overwrite");
      Files.createDirectories(path.getParent());
      try (OutputStream out = Files.newOutputStream(partial, StandardOpenOption.CREATE_NEW)) {
        write(core, out);
      }
      Files.move(partial, path);
      core.log.count("model.exported");
    } catch (IOException ex) {
      core.log.count("model.exportFailure");
      throw new IllegalStateException("model export failed; partial is invalid evidence", ex);
    }
  }
  static void roundHash(Core core) {
    for (var enemy : core.enemies.values())
      for (var rating : enemy.ratings.entrySet())
        core.log.event("gun.rating", core.round, core.lastTurn,
            "target=" + enemy.id + " gun=" + rating.getKey() + " cohort=" + rating.getValue().count
                + " raw=" + rating.getValue().raw + " adjusted=" + rating.getValue().adjusted);
    try {
      var digest = MessageDigest.getInstance("SHA-256");
      write(core, new java.security.DigestOutputStream(OutputStream.nullOutputStream(), digest));
      core.log.event("model.roundHash", core.round, core.lastTurn,
          java.util.HexFormat.of().formatHex(digest.digest()));
    } catch (IOException | NoSuchAlgorithmException ex) {
      throw new IllegalStateException(ex);
    }
  }
  static void write(Core core, OutputStream raw) throws IOException {
    class Limited extends FilterOutputStream {
      long count;
      Limited(OutputStream out) {
        super(out);
      }
      public void write(int b) throws IOException {
        if (++count > MAX_BYTES)
          throw new IOException("model cap");
        out.write(b);
      }
      public void write(byte[] b, int off, int len) throws IOException {
        if (count + len > MAX_BYTES)
          throw new IOException("model cap");
        count += len;
        out.write(b, off, len);
      }
    }
    DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new Limited(raw)));
    out.writeUTF("FACET-OWN-GAME-EXPORT");
    out.writeInt(SCHEMA);
    out.writeUTF("1.4.0");
    out.write(configHash());
    out.writeDouble(core.rules.width());
    out.writeDouble(core.rules.height());
    out.writeDouble(core.rules.cooling());
    out.writeInt(core.round);
    out.writeLong(core.sequence);
    out.writeLong(core.waves.nextId);
    out.writeInt(core.enemies.size());
    for (Learner.Enemy e : core.enemies.values()) {
      out.writeInt(e.id);
      out.writeUTF("LOCAL_GAME_ID_NOT_PORTABLE_IDENTITY");
      out.writeInt(e.crossings);
      out.writeLong(e.shots);
      out.writeLong(e.hits);
      out.writeLong(e.contacts);
      out.writeLong(e.rawHits);
      out.writeLong(e.rawContacts);
      out.writeLong(e.observedTogether);
      out.writeDouble(e.damageTaken);
      out.writeDouble(e.damageGiven);
      out.writeBoolean(e.comparisonEligible);
      out.writeLong(e.startupRating.count);
      out.writeDouble(e.startupRating.raw);
      out.writeDouble(e.startupRating.adjusted);
      out.writeUTF(e.selected);
      for (var rating : e.ratings.entrySet()) {
        out.writeUTF(rating.getKey());
        out.writeLong(rating.getValue().count);
        out.writeDouble(rating.getValue().raw);
        out.writeDouble(rating.getValue().adjusted);
      }
      List<Memory.Store> stores = List.of(e.longGun, e.recent[0], e.recent[1], e.recent[2],
          e.displacement, e.recentDisplacement[0], e.recentDisplacement[1], e.recentDisplacement[2],
          e.simple, e.context, e.recentHit, e.visits, e.powers);
      out.writeInt(stores.size());
      for (Memory.Store s : stores) {
        out.writeUTF(s.label.name());
        out.writeInt(s.dimension);
        out.writeInt(s.capacity);
        out.writeLong(s.version);
        out.writeInt(s.size());
        for (Memory.Observation o : s.data) {
          out.writeInt(1);
          out.writeLong(o.wave());
          out.writeInt(o.round());
          out.writeInt(o.turn());
          out.writeDouble(o.quality());
          out.writeLong(o.sequence());
          out.writeBoolean(o.melee());
          for (double f : o.features()) out.writeDouble(f);
          out.writeDouble(o.a());
          out.writeDouble(o.b());
        }
      }
    }
    out.writeUTF("END-FACET");
    out.flush();
  }
}
