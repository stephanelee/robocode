# Dependency and license evidence

Facet adds no gameplay library. Runtime build dependency is the official `dev.robocode.tankroyale:robocode-tankroyale-bot-api:1.4.0` artifact. Development used the supplied shaded API JAR SHA256 `ab65c4d5cec1808adeb71375adae6d15341ae250def89a6c10fc9da879de0752`. Test JSON decoding uses Gson already included in that artifact. Java 17 uses the JDK standard library and official Bot/Bullet/Event APIs; the finite-distance diagnostic helper uses the pinned official public IntentValidator class.

Embedded Maven metadata in the supplied shaded JAR identifies:

| Included component | Version | License declared in embedded POM |
|---|---|---|
| com.google.code.gson:gson | 2.14.0 | Apache-2.0 |
| uk.co.foundationsedge:nv-i18n | 1.34.1 | Apache License 2.0 |
| com.google.errorprone:error_prone_annotations | 2.48.0 | Apache License 2.0 |

These are observed shaded development versions, not a claim that a separately cached Maven artifact has the identical transitive payload. The supplied official contract subset and shaded API JAR contain no top-level license/notice text for the API itself. No network acquisition was authorized. Root must verify the API distribution license and required notices against its approved Maven metadata and retain license texts before publishing; this session does not fabricate missing license evidence. Facet's manual fat JAR retains embedded Maven metadata. Gradle payload parity remains required.
