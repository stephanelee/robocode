"""Consume exactly one durable atomic receipt for an independently cleared Peak stage."""
import argparse
import json
import os
import subprocess

import benchmark
from meridian_peak import check_confirmation, receipt_ref
from meridian_peak_common import ROOT, REG, save
from meridian_peak_profile import PROVENANCE, check_host, validate_payload


def api(endpoint, value):
    return json.loads(subprocess.check_output(['gh', 'api', '--method', 'POST', endpoint, '--input', '-'], input=json.dumps(value), text=True))


def activate(stage):
    if os.environ.get('GITHUB_RUN_ATTEMPT') != '1':
        raise ValueError('No reruns')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if head != os.environ.get('PAYLOAD_SHA'):
        raise ValueError('Checkout differs from independently cleared payload')
    validate_payload(json.loads((PROVENANCE / 'profile.json').read_text()))
    host = check_host(stage)
    if stage == 'confirmation':
        check_confirmation()
    receipt = dict(experiment='meridian-peak-v1', stage=stage, payloadSHA=head,
        freezeSHA256=benchmark.digest(REG / 'freeze.json'), runID=os.environ['GITHUB_RUN_ID'], runAttempt=1,
        receiptRef=receipt_ref(stage), hostIdentity=host,
        hostDocumentSHA256=benchmark.digest(PROVENANCE / (stage + '-host.json')))
    repo = os.environ['GITHUB_REPOSITORY']
    tag = api(f'repos/{repo}/git/tags', dict(tag=receipt['receiptRef'].removeprefix('refs/tags/'),
              message=json.dumps(receipt, sort_keys=True), object=head, type='commit'))
    # Atomic create, not update: any prior complete OR invalid stage permanently consumes its budget.
    api(f'repos/{repo}/git/refs', dict(ref=receipt['receiptRef'], sha=tag['sha']))
    receipt['receiptObject'] = tag['sha']
    save(PROVENANCE / (stage + '-activation.json'), receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('screen', 'confirmation'), required=True)
    activate(parser.parse_args().stage)
