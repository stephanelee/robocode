#!/usr/bin/env python3
"""Generate the prospective inline workflow from the sole operational collector source."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'tools/keel_brace_image_probe.py'
TARGET = ROOT / '.github/workflows/keel-brace-image-probe.yml'
MARKER = '__KEEL_BRACE_IMAGE_PROBE_COLLECTOR__'


def workflow(source):
    if MARKER in source or not source.endswith('\n') or '\r' in source or '${{' in source:
        raise ValueError('collector must be LF-only with final newline, no heredoc delimiter or GitHub expression interpolation')
    prefix = '''name: KeelBrace zero-game image provenance probe
on:
  pull_request:
    types: [labeled]
permissions:
  contents: read
concurrency:
  group: keel-brace-v1-image-probe
  cancel-in-progress: false
jobs:
  image-observations:
    name: Two snapshots of image metadata (zero games)
    if: >-
      github.event.pull_request.head.repo.full_name == github.repository &&
      github.head_ref == 'codex/keel-brace-v1' && github.run_attempt == 1 &&
      github.event.label.name == 'run-keel-brace-image-probe' &&
      contains(github.event.pull_request.body, format('KEEL-BRACE-IMAGE-PROBE-CLEARED {0}', github.event.pull_request.head.sha))
    runs-on: ubuntu-26.04
    timeout-minutes: 10
    env:
      KEEL_BRACE_IMAGE_PROBE_HEAD_SHA: ${{ github.event.pull_request.head.sha }}
    steps:
      - name: Capture before checkout or tool setup
        id: before
        continue-on-error: true
        shell: bash
        run: |
          set -euo pipefail
          umask 077
          PROBE_ROOT="$RUNNER_TEMP/keel-brace-image-probe-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
          PROBE_COLLECTOR="$RUNNER_TEMP/keel-brace-image-probe-collector-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.py"
          test ! -e "$PROBE_ROOT" && test ! -L "$PROBE_ROOT"
          test ! -e "$PROBE_COLLECTOR" && test ! -L "$PROBE_COLLECTOR"
          mkdir -m 700 "$PROBE_ROOT"
          cat > "$PROBE_COLLECTOR" <<'__KEEL_BRACE_IMAGE_PROBE_COLLECTOR__'
'''
    suffix = '''          __KEEL_BRACE_IMAGE_PROBE_COLLECTOR__
          /usr/bin/python3 "$PROBE_COLLECTOR" --phase before-tools --output-dir "$PROBE_ROOT/before-tools"
      - name: Apply the same Java setup version used by bootstrap
        id: java
        if: always()
        uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: '17.0.21+9'
      - name: Apply the same Python setup version used by bootstrap
        id: python
        if: always()
        uses: actions/setup-python@v5
        with:
          python-version: '3.13.17'
      - name: Capture after Java and Python setup attempts
        id: after
        if: always()
        continue-on-error: true
        shell: bash
        env:
          KEEL_BRACE_IMAGE_PROBE_JAVA_SETUP_OUTCOME: ${{ steps.java.outcome }}
          KEEL_BRACE_IMAGE_PROBE_PYTHON_SETUP_OUTCOME: ${{ steps.python.outcome }}
        run: |
          set -euo pipefail
          PROBE_ROOT="$RUNNER_TEMP/keel-brace-image-probe-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
          PROBE_COLLECTOR="$RUNNER_TEMP/keel-brace-image-probe-collector-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.py"
          /usr/bin/python3 "$PROBE_COLLECTOR" --phase after-tools --output-dir "$PROBE_ROOT/after-tools"
      - name: Upload original observations even on diagnostic errors
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: keel-brace-image-probe-${{ github.run_id }}
          path: ${{ runner.temp }}/keel-brace-image-probe-${{ github.run_id }}-${{ github.run_attempt }}/
          if-no-files-found: error
          include-hidden-files: true
          retention-days: 90
      - name: Report collection errors without deciding image admission
        if: always()
        shell: bash
        env:
          PROBE_BEFORE_OUTCOME: ${{ steps.before.outcome }}
          PROBE_AFTER_OUTCOME: ${{ steps.after.outcome }}
          PROBE_JAVA_OUTCOME: ${{ steps.java.outcome }}
          PROBE_PYTHON_OUTCOME: ${{ steps.python.outcome }}
        run: |
          test "$PROBE_BEFORE_OUTCOME" = success && test "$PROBE_AFTER_OUTCOME" = success && test "$PROBE_JAVA_OUTCOME" = success && test "$PROBE_PYTHON_OUTCOME" = success
'''
    return prefix + ''.join(('          ' + line if line else '') + '\n' for line in source.splitlines()) + suffix


if __name__ == '__main__':
    TARGET.write_text(workflow(SOURCE.read_bytes().decode('utf-8')))
