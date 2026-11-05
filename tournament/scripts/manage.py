"""Create a draft event or resolve its website link; never launch games or publish."""
import argparse
import json
import os
from pathlib import Path
import re
from urllib.parse import urlencode, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def scaffold(event_id, source_commit, event_root=ROOT / 'events', project_id=None,
             repository_root=ROOT.parent):
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', event_id):
        raise ValueError('Event ID must contain lowercase letters, digits and single hyphens')
    if not re.fullmatch(r'[0-9a-fA-F]{40}', source_commit):
        raise ValueError('Use the full 40-character frozen source commit')
    config = json.loads((ROOT / 'templates/event.json').read_text())
    config.update(id=event_id, title=event_id, sourceCommit=source_commit.lower())
    config['site']['projectId'] = project_id
    config['paths'] = {k: v.format(event_id=event_id) for k, v in config['paths'].items()}
    for directory in {ROOT / 'events', Path(event_root)}:
        for manifest in directory.glob('*/event.json'):
            existing = json.loads(manifest.read_text())
            if existing.get('id') == event_id:
                raise FileExistsError(f'Event ID already registered: {event_id}')
    for value in config['paths'].values():
        reserved = Path(repository_root) / value
        if reserved.exists():
            raise FileExistsError(f'Event output path already exists: {value}')
    target = Path(event_root) / event_id
    target.mkdir(parents=True, exist_ok=False)
    (target / 'event.json').write_text(json.dumps(config, indent=2) + '\n')
    (target / 'README.md').write_text(
        f'# {event_id}\n\nDraft event; no games have started.\n\n'
        'Complete the new-event checklist before using a reviewed runner. '
        'This scaffold does not activate the event or modify the current tournament.\n')
    return target


def current_event():
    pointer = json.loads((ROOT / 'current.json').read_text())
    path = (ROOT / pointer['event']).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Current event must remain inside tournament/')
    return json.loads(path.read_text())


def site_url(config, stage=None, environment=None):
    environment = os.environ if environment is None else environment
    variable = config['site']['baseUrlEnvironment']
    base = environment.get(variable, '').strip()
    parsed = urlsplit(base)
    if (parsed.scheme not in ('http', 'https') or not parsed.netloc
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError(f'Set {variable} to the website base URL returned by deployment (no credentials, query or fragment)')
    if stage is None:
        return urljoin(base.rstrip('/') + '/', 'index.html')
    if not re.fullmatch(r'practice|swiss-[1-9][0-9]*|playoff-[1-9][0-9]*', stage):
        raise ValueError('Unknown tournament stage')
    return urljoin(base.rstrip('/') + '/', 'matches.html') + '?' + urlencode({'stage': stage})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    new = commands.add_parser('init', help='Create a separate draft; no games, activation or publication')
    new.add_argument('event_id')
    new.add_argument('--source-commit', required=True)
    new.add_argument('--site-project-id')
    new.add_argument('--event-root', type=Path, default=ROOT / 'events')
    link = commands.add_parser('site-url', help='Resolve a link using an environment-provided base URL')
    link.add_argument('--event', type=Path, help='Read another event.json; defaults to current.json')
    link.add_argument('--stage')
    args = parser.parse_args()
    try:
        if args.command == 'init':
            print(scaffold(args.event_id, args.source_commit, args.event_root, args.site_project_id))
        else:
            config = json.loads(args.event.read_text()) if args.event else current_event()
            print(site_url(config, args.stage))
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
