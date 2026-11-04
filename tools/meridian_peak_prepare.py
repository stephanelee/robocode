"""Java 17 builds/tests and parity only. Never starts a bot, server or benchmark."""
import json
import os
from pathlib import Path
import shutil
import subprocess

from meridian_peak_common import ROOT, OUT, save
from meridian_peak_profile import BASE, prepare_walls


def main():
    java = Path(os.environ['JAVA_HOME'])
    version = subprocess.check_output([str(java / 'bin/java'), '-version'], stderr=subprocess.STDOUT, text=True)
    if '17.0.21' not in version or 'Temurin-17.0.21+9' not in version:
        raise ValueError('Hosted preparation requires exact registered Java 17')
    temporary = Path(os.environ['RUNNER_TEMP']) / 'meridian-peak-build-preparation'
    temporary.mkdir()
    commands = []

    def run(command, name):
        commands.append(dict(command=command, log=name))
        with (temporary / name).open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)

    committed = temporary / 'production.jar'
    shutil.copy2(ROOT / 'AISurvivor/AISurvivor.jar', committed)
    try:
        run(['./gradlew', '--no-daemon', 'clean', 'build'], 'production-build.log')
        # The root clean above removes build/. Capture the runtime proof afterwards.
        old = ROOT / 'build/meridian'
        old.mkdir(parents=True, exist_ok=True)
        (old / 'build-java.txt').write_text(version)
        (old / 'build-java-home.txt').write_text(str(java.resolve()) + '\n')
        import benchmark
        (old / 'build-java.sha256').write_text(benchmark.digest(java / 'bin/javac') + '  javac\n' + benchmark.digest(java / 'lib/modules') + '  modules\n')
        run(['python', 'tools/verify_jar.py', str(committed), 'AISurvivor/AISurvivor.jar'], 'production-parity.log')
        shutil.copy2(committed, ROOT / 'AISurvivor/AISurvivor.jar')
        run(['python', 'tools/meridian_gun_plan_fixture.py'], 'fixed-fixture.log')
        for name in ('Meridian', 'MeridianPeak'):
            run(['./gradlew', '--no-daemon', '-p', 'experiments/' + name, 'clean', 'build'], name + '-build.log')
            run(['python', 'tools/verify_jar.py', f'experiments/{name}/{name}.jar',
                 f'experiments/{name}/build/bot/{name}/{name}.jar'], name + '-parity.log')
        # Materialize the exact control source without changing current bot files.
        control = ROOT / 'build/peak-control-source'
        control.mkdir()
        archive = temporary / 'control.tar'
        with archive.open('wb') as fd:
            subprocess.run(['git', 'archive', BASE, 'experiments/Meridian'], cwd=ROOT, stdout=fd, check=True)
        subprocess.run(['tar', '-xf', str(archive), '-C', str(control)], check=True)
        source = control / 'experiments/Meridian'
        run(['./gradlew', '--no-daemon', '-p', str(source), 'clean', 'build'], 'control-build.log')
        run(['python', 'tools/verify_jar.py', str(source / 'Meridian.jar'), str(source / 'build/bot/Meridian/Meridian.jar')], 'control-parity.log')
        prepare_walls()
        run(['python', '-m', 'unittest', 'discover', '-s', 'tools', '-p', 'test_meridian_peak*.py', '-v'], 'harness-tests.log')
    finally:
        destination = OUT / 'preparation'
        destination.mkdir(parents=True, exist_ok=True)
        for path in temporary.iterdir():
            if path.suffix != '.tar' and path.name != 'production.jar':
                shutil.copy2(path, destination / path.name)
        save(destination / 'commands.json', commands)


if __name__ == '__main__':
    main()
