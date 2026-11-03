"""Compile the actual runner adapter; synthetic file timing only, no games."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import benchmark


class ChallengerCompletionTests(unittest.TestCase):
    def test_final_output_barrier_before_runner_close(self):
        name = "robocode-tankroyale-runner-1.4.0.jar"
        runner = benchmark.download(benchmark.ROOT / "build/completion-test-cache", name, benchmark.ASSETS[name])
        java_home = os.environ.get("JAVA_HOME")
        java = str(Path(java_home) / "bin/java") if java_home else "java"
        javac = str(Path(java_home) / "bin/javac") if java_home else "javac"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run([javac, "--release", "17", "-cp", str(runner), "-d", str(root),
                            str(benchmark.ROOT / "tools/BenchmarkBattle.java"),
                            str(benchmark.ROOT / "tools/fixtures/ChallengerCompletionTest.java")],
                           check=True, capture_output=True, text=True, timeout=30)
            result = subprocess.run([java, "-cp", os.pathsep.join((str(root), str(runner))),
                                     "ChallengerCompletionTest", str(root / "outputs")],
                                    check=True, capture_output=True, text=True, timeout=10)
            self.assertIn("Seven synthetic completion checks passed; zero games", result.stdout)


if __name__ == "__main__":
    unittest.main()
