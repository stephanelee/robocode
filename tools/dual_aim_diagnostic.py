"""Offline conditional command feasibility; never connects to a game or reads private bot state.

Angles are degrees (0 east, positive counterclockwise). Engine movement mirrors
Tank Royale v1.4.0, including its surprising target-speed zero/reversal behavior.
Bullet-sector distance is analytic over the complete angular interval, not sampled rays.
"""
import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path

ENGINE = "v1.4.0"
RETCON = "28981c7a469ac064194b41001cd7183ec5ecf189"
RADIUS = 18.0
WIDTH = HEIGHT = 1000.0


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def relative(a):
    return (a + 180) % 360 - 180


def bearing(x, y):
    return math.degrees(math.atan2(y, x))


def next_speed(current, target):
    """Literal server calcNewBotSpeed, NOT textbook +/-2 braking."""
    if current < 0:
        return -next_speed(-current, -target)
    if current > 0:
        if target > 0:
            difference = target - current
            return (min(current + min(difference, 1), 8) if difference >= 0
                    else max(current + max(difference, -2), -8))
        return 0.0 if target == 0 else -(1 - current / 2)
    return clamp(target, -1, 1)


@dataclass(frozen=True)
class State:
    x: float
    y: float
    heading: float
    speed: float


def constrain(old_x, old_y, x, y, width=WIDTH, height=HEIGHT):
    """Server first-wall intersection (y case recomputes from original move)."""
    nx, ny = x, y
    if x < RADIUS or x > width - RADIUS:
        nx = clamp(x, RADIUS, width - RADIUS)
        if x != old_x:
            ny = old_y + (y - old_y) * (nx - old_x) / (x - old_x)
    if y < RADIUS or y > height - RADIUS:
        ny = clamp(y, RADIUS, height - RADIUS)
        if y != old_y:
            nx = old_x + (x - old_x) * (ny - old_y) / (y - old_y)
    return nx, ny


def step(state, command, width=WIDTH, height=HEIGHT):
    speed = next_speed(state.speed, command[0])
    angle = math.radians(state.heading)
    x, y = state.x + speed * math.cos(angle), state.y + speed * math.sin(angle)
    turn = clamp(command[1], -(10 - .75 * abs(speed)), 10 - .75 * abs(speed))
    heading = (state.heading + turn) % 360
    nx, ny = constrain(state.x, state.y, x, y, width, height)
    wall = nx != x or ny != y
    # TurnProcessor first applies the wall outcome, THEN constrainBotPositions.
    # The Y branch of constrain can overwrite the X clamp at a corner, so this
    # second pass is observable and necessary (same pre-turn origin).
    nx, ny = constrain(state.x, state.y, nx, ny, width, height)
    return State(nx, ny, heading, 0.0 if wall else speed), wall


def linear_aim(origin, target, bullet_speed, width=WIDTH, height=HEIGHT):
    """Retcon's iterative model; coordinate-clamped forecast is NOT engine movement."""
    x, y = target.x, target.y
    dx = math.cos(math.radians(target.heading)) * target.speed
    dy = math.sin(math.radians(target.heading)) * target.speed
    for turns in range(1, 151):
        x = clamp(x + dx, RADIUS, width - RADIUS)
        y = clamp(y + dy, RADIUS, height - RADIUS)
        if bullet_speed * turns >= math.hypot(x - origin[0], y - origin[1]):
            break
    return bearing(x - origin[0], y - origin[1]), turns, (x, y)


def sector_distance(x, y, heading, half_angle, near, far):
    """Minimum point distance to union of radial segments over a closed angle interval.

    Includes every admissible current gun heading, including interval interiors.
    Continuing bullets through arena walls is deliberately conservative: removed
    bullets can only remove threats, never invalidate positive clearance here.
    """
    radius = math.hypot(x, y)
    delta = math.radians(max(0.0, abs(relative(bearing(x, y) - heading)) - half_angle))
    along = clamp(radius * math.cos(delta), near, far)
    return math.hypot(radius * math.sin(delta), radius * math.cos(delta) - along)


def trace(initial, command, delay, horizon):
    # delay is reaction delay AFTER the adversarial shot decision at t=0.
    baseline = (8 if initial.speed >= 0 else -8, 0)
    state = initial
    result = []
    for turn in range(1, horizon + 1):
        state, wall = step(state, baseline if turn <= delay else command)
        result.append((state, wall))
    return result


def threats(origin, initial, extra_angle, scan_age, powers=(1.2, 2.0, 3.0)):
    # Constant-heading back-extrapolation is a sensitivity scenario, NOT a true
    # reconstruction of Retcon's unknown private scan or movement history.
    angle = math.radians(initial.heading)
    stale = State(clamp(initial.x - scan_age * initial.speed * math.cos(angle), RADIUS, WIDTH - RADIUS),
                  clamp(initial.y - scan_age * initial.speed * math.sin(angle), RADIUS, HEIGHT - RADIUS),
                  initial.heading, initial.speed)
    distance = math.hypot(stale.x - origin[0], stale.y - origin[1])
    tolerance = math.degrees(math.atan(RADIUS / distance)) + extra_angle
    result = []
    for power in powers:
        speed = 20 - 3 * power
        linear, _, _ = linear_aim(origin, stale, speed)
        for model, aim in (("head-on", bearing(stale.x - origin[0], stale.y - origin[1])),
                           ("linear", linear)):
            result.append((model, speed, aim, tolerance))
    return result


def clearance(path, origin, shot_models):
    minimum = math.inf
    limiting = None
    for turn, (state, _) in enumerate(path, 1):
        x, y = state.x - origin[0], state.y - origin[1]
        radius = math.hypot(x, y)
        for model, speed, aim, tolerance in shot_models:
            near, far = speed * turn, speed * (turn + 1)
            if max(near - radius, radius - far, 0) - RADIUS >= minimum:
                continue
            # v1.4 increments bullet.tick before BulletLine uses [tick,tick+1].
            margin = sector_distance(x, y, aim, tolerance, near, far) - RADIUS
            if margin < minimum:
                minimum, limiting = margin, (turn, model, speed)
    return minimum, limiting


def cases(suite):
    if suite == "development":
        ranges, speeds, headings, layouts = ((150, 250, 350, 450, 550, 650), range(-8, 9),
                                             (0, 90), ("open", "wall0", "wall32", "corner"))
    elif suite == "withheld":
        ranges, speeds, headings, layouts = ((200, 400, 600), (-7, -3, 0, 3, 7),
                                             (45, 135), ("open", "wall16", "corner"))
    else:
        raise ValueError("Unknown suite")
    for distance, speed, heading, layout in itertools.product(ranges, speeds, headings, layouts):
        y = 500 if layout == "open" else RADIUS + (int(layout[4:]) if layout.startswith("wall") else 0)
        x = WIDTH - RADIUS if layout == "corner" else 80 + distance
        yield distance, speed, heading, layout, State(x, y, heading, speed), (x - distance, y)


def diagnose(suite="development"):
    commands = list(itertools.product((-8, 0, 8), (-10, 0, 10)))
    profiles = (("fresh-alignment", 0, 0), ("age2-extra5deg", 2, 5),
                ("age5-extra15deg", 5, 15), ("unknown-heading", 0, 180))
    horizon = math.ceil(math.hypot(WIDTH, HEIGHT) / 11) + 1
    rows = []
    for distance, speed, heading, layout, initial, origin in cases(suite):
        baseline = (8 if speed >= 0 else -8, 0)
        for delay in (0, 1, 3):
            paths = {command: trace(initial, command, delay, horizon) for command in commands}
            for profile, age, extra in profiles:
                models = threats(origin, initial, extra, age)
                scores = {command: clearance(path, origin, models) for command, path in paths.items()}
                # Baseline wins ties; optimization is same-state in-sample feasibility,
                # not a predictive/generalization result or strength comparison.
                chosen = max(commands, key=lambda c: (scores[c][0], c == baseline,
                             -sum(w for _, w in paths[c]), -abs(c[1]), -abs(c[0])))
                base_margin = scores[baseline][0]
                best_margin = scores[chosen][0]
                def forecast_error(command):
                    # Opponent constant-velocity forecast error, not simulator error.
                    dx = initial.speed * math.cos(math.radians(initial.heading))
                    dy = initial.speed * math.sin(math.radians(initial.heading))
                    x, y = initial.x, initial.y
                    errors = []
                    for state, _ in paths[command][:20]:
                        x, y = clamp(x + dx, RADIUS, WIDTH - RADIUS), clamp(y + dy, RADIUS, HEIGHT - RADIUS)
                        errors.append((state.x - x) ** 2 + (state.y - y) ** 2)
                    return round(math.sqrt(sum(errors) / len(errors)), 6)
                rows.append(dict(range=distance, speed=speed, layout=layout, heading=heading,
                                 delay=delay, profile=profile, baselineCommand=baseline,
                                 proposedCommand=chosen, baselineClearance=round(base_margin, 6),
                                 proposedClearance=round(best_margin, 6),
                                 baselineSafe=base_margin > 1e-7, proposedSafe=best_margin > 1e-7,
                                 baselineWallTurns=sum(w for _, w in paths[baseline]),
                                 proposedWallTurns=sum(w for _, w in paths[chosen]),
                                 baselineLinearForecastRmse20=forecast_error(baseline),
                                 proposedLinearForecastRmse20=forecast_error(chosen),
                                 limitingThreat=scores[chosen][1]))
    summary = []
    for profile, _, _ in profiles:
        for delay in (0, 1, 3):
            group = [r for r in rows if r["profile"] == profile and r["delay"] == delay]
            summary.append(dict(profile=profile, delay=delay, states=len(group),
                                baselineSafe=sum(r["baselineSafe"] for r in group),
                                proposedSafe=sum(r["proposedSafe"] for r in group),
                                noCertifiedCommand=sum(not r["proposedSafe"] for r in group),
                                baselineWallContact=sum(r["baselineWallTurns"] > 0 for r in group),
                                baselineForecastRmse20=round(sum(r["baselineLinearForecastRmse20"] for r in group) / len(group), 6),
                                proposedForecastRmse20=round(sum(r["proposedLinearForecastRmse20"] for r in group) / len(group), 6),
                                wallContactChosen=sum(r["proposedWallTurns"] > 0 for r in group)))
    return dict(schemaVersion=1, suiteVersion="dual-aim-v2", suite=suite, engine=ENGINE, retconCommit=RETCON,
                arena=[WIDTH, HEIGHT], horizon=horizon, commands=commands, powers=[1.2, 2, 3],
                criterion="worst-case continuous angular corridor, swept bullet vs post-move circle; conservative after bullet wall removal",
                scope="single shot at t0; stationary shooter; hypothetical scan histories; no opponent private input; finite commands only; not scores",
                summary=summary, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("development", "withheld"), default="development")
    args = parser.parse_args()
    result = diagnose(args.suite)
    result["toolSha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
