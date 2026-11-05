# Tournament workspace

This folder records the current invitational and provides a starting point for a
separate tournament. It does not authorize another stage or event.

- [Current event: 2026-11-04](events/2026-11-04/README.md): rules, entrants, results,
  pins, replay status and exceptional rulings.
- [Start a new event](new-event.md): prospective registration and execution guide.
- [Reusable configuration template](templates/event.json).
- [Current-event pointer](current.json).

The website address is supplied through `TOURNAMENT_SITE_URL`, using the URL
returned by the hosting deployment. No production hostname is embedded in this
folder. The event configuration identifies the hosting project separately.

```sh
# Set TOURNAMENT_SITE_URL in your environment from the deployment response.
python3 tournament/manage.py site-url --stage playoff-4

# Create a draft in a separate folder; this launches no games.
python3 tournament/manage.py init next-event --source-commit <full-commit-sha>
```

Event manifest paths are relative to the repository root.

`init` rejects registered IDs and existing output paths, leaves `current.json` unchanged and creates
fresh evidence/media paths. The current runner still contains the original
invitational's fixed entrants and seeds; the [new-event guide](new-event.md)
explains the required review before a new competition can run.

Run the documentation/configuration helper checks from the repository root:

```sh
python3 -m unittest discover -s tournament -p 'test_*.py'
PYTHONPATH=tools python3 -m unittest tools/test_tournament.py tools/test_tournament_playoffs.py
```
