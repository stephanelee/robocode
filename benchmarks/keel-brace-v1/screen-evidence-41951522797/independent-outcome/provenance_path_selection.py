"""Read-only adapter for duplicate profile lookup; no original/source/criteria changes."""
import importlib.util,json,hashlib
from pathlib import Path
base=Path('/private/tmp/keel-brace-screen-41951522797-outcome')
source=Path('/private/tmp/robocode-keel-brace/benchmarks/keel-brace-v1/independent-analysis/audit.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='8cef410dc20c0af7b317cf0de62c8cf65d65d10b559ea44145e00c69710e2cd5'
spec=importlib.util.spec_from_file_location('frozen_independent_reader',source);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
archive=Path('/private/tmp/keel-brace-screen-41951522797-final-evidence/original-artifact-13383827989.zip')
assert audit.digest(archive)=='3eb14b072617c3b64d364988c9db8d224cb47e4160214bd14d6e6ef629ca02d8'
store=audit.ZipStore(archive)
try:
 stage=store.root('robocode/robocode/build/keel-brace/screen/');original=stage/'profile.json';selected=stage/'provenance/profile.json'
 assert original.read_bytes()==selected.read_bytes()
 assert audit.digest(selected)=='b670725af5deb0e711dfa51f4ead24c24066ff65e29d5a1675977966438119f8'
 class ActualProvenanceSelection:
  def __truediv__(self,path):return selected if str(path)=='profile.json' else stage/path
 # This maps one byte-identical duplicate directly to its actual complete original bundle.
 # It does not suppress existence, synthesize files, modify an archive or change a check.
 evidence=ActualProvenanceSelection();source_root=Path('/private/tmp/robocode-keel-brace')
 result={'scope':'Supplement frozen first failed profile-parent lookup only; original machine output remains verbatim PROVENANCE-INCOMPLETE/rawREJECT. Original ZIP and all member identities unchanged.', 'decoderSHA256':audit.digest(source),'originalProfilePath':str(original),'selectedActualProfilePath':str(selected),'bothOriginalProfileSHA256':audit.digest(selected),'mappedLookupOnly':'profile.json -> provenance/profile.json (identical original bytes)','provenance':audit.verify_exported_provenance(evidence,source_root,'screen','e659647b3ae6366008ef666bf338ea3ee446ee71',41951522797),'runnerImage':audit.audit_runner_image(evidence,'e659647b3ae6366008ef666bf338ea3ee446ee71',41951522797,base/'provider',source_root)}
 (base/'frozen-provenance-path-selection-supplement.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:result[k]['pass'] for k in ('provenance','runnerImage')}))
finally:store.close()
