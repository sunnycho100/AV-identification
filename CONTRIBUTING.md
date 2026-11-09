# Contributing

Use Python 3.10 and install the CPU development dependencies described in [Getting started](code/docs/GETTING_STARTED.md). Run `ruff check code/src code/tests code/tools code/manage.py` and `python code/manage.py test` before submitting maintained-code changes.

- Put reusable logic in `code/src/roadside_identification/`, not a dated experiment script.
- Keep configuration explicit and paths relative to `code/`. Do not add machine-specific usernames or absolute data locations to maintained code.
- Keep imports side-effect free: no dataset loading, network download, training, GPU initialization or checkpoint loading at module import.
- Add behavioral tests for numerical or data-boundary changes. Use synthetic inputs; private field data and checkpoints must not be needed by ordinary tests.
- Preserve SI units, actual time intervals, separate time/feature masks, and the common ego--leader geometry. Document any change in prediction semantics.
- Treat research protocols and published predictions as versioned evidence. Create a new experiment rather than overwriting old results; record seeds, splits and input hashes.
- Do not bulk-reformat upstream code or relabel its license. Keep modifications localized and preserve notices.
- Run `python code/manage.py inventory` after adding entry points, and inspect the source export before publishing.

For issues or pull requests, state the problem, observed behavior, proposed change, environment, and validation. Include a synthetic reproducer when possible. Never attach raw field footage, GPS or review records to a public issue by default.

For the pending stabilization experiment, use [the collaborator handoff](code/docs/STABILIZATION_HANDOFF.md). An unfilled result is preferable to a fabricated or incompatible comparison.
