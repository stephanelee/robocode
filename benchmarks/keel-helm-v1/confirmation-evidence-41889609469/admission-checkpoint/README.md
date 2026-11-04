# Lossless confirmation evidence checkpoint

This is an outcome-agnostic original-byte archive, not a competitive result or
confirmation eligibility/acceptance claim. It uses existing owner-downloaded
originals only. No network, Git mutation, runtime preparation, bot/helper,
metric reducer or game is run by packaging or verification.

The exact ORIGINAL provider ZIP is retained in verbatim parts at most2MiB;
it is never recompressed. Every original logical member, including staged work
and any failure/partial metadata, remains inside that authoritative ZIP with
its indexed filename, byte length, SHA256 and original ZIP metadata. Key stage
metadata and ALL actual original .battle.gz files are also direct regular files.
No expected18-recording inventory is fabricated when a stage fails or stops.
The unchanged raw decisions, failures, budget and unplayed-ID artifacts are
retained without normalization; interpretation belongs to the independent audit.

Provider API/log/review/receipt/comment and independent audit extras retain
original bytes and hashes. Any additional original ZIP is split verbatim too.
A byte/hash-only portable check needs no Git, network or mutable source tree:

    python3 -I -B -S verify.py

Do not infer validity, consumed physical rounds, performance or stage acceptance
from this archive's successful byte verification. Preserve failed attempts and
obtain independent outcome/source/runtime review before any new authorized stage.
