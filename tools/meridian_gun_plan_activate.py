"""Atomically consume the sole hosted #90 attempt; never remove the receipt."""
import json
import os
from pathlib import Path
import subprocess
import meridian_gun_plan as comparison

REF = "refs/tags/experiment-receipts/meridian-gp60-issue90"


def api(endpoint, payload):
    return json.loads(subprocess.check_output(
        ["gh", "api", "--method", "POST", endpoint, "--input", "-"],
        input=json.dumps(payload), text=True))


def main():
    if os.environ["GITHUB_RUN_ATTEMPT"] != "1":
        raise ValueError("No reruns")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != os.environ["PAYLOAD_SHA"]:
        raise ValueError("Checkout is not the cleared payload")
    host = comparison.validate_host_identity()
    receipt = dict(experiment="meridian-gp60-issue90", payloadSHA=head,
                   runID=os.environ["GITHUB_RUN_ID"], runAttempt=1, receiptRef=REF,
                   hostIdentity=host,
                   hostDocumentSHA256=comparison.benchmark.digest(comparison.OUT / "gun-plan-host.json"))
    repo = os.environ["GITHUB_REPOSITORY"]
    tag = api(f"repos/{repo}/git/tags", dict(
        tag=REF.removeprefix("refs/tags/"), message=json.dumps(receipt, sort_keys=True),
        object=head, type="commit"))
    # GitHub atomically rejects an existing ref, including a prior failed attempt.
    api(f"repos/{repo}/git/refs", dict(ref=REF, sha=tag["sha"]))
    receipt["receiptObject"] = tag["sha"]
    Path("build/meridian/activation.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    main()
