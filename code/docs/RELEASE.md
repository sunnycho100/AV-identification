# Preparing the GitHub source repository

Run from the repository root with a Python interpreter containing the package's CPU dependencies:

```sh
python code/manage.py inventory
python code/manage.py test
python code/manage.py release
```

`code/outputs/github_release/latest.json` points to a timestamped source tree and ZIP. Exporting does not initialize Git, create a remote, publish, push, or alter the working datasets. Upload the exported `roadside-identification/` tree rather than dragging the entire research workspace into GitHub.

The exporter uses an explicit allowlist, records a SHA256 for every source file, rejects unexpectedly large files and obvious credential signatures, and excludes raw recordings, GPS, review decisions, checkpoints, environments, compiled binaries, generated experiment folders, local catalogs, manuscript files and generated package metadata. These checks supplement review; they are not a guarantee against every possible secret. The synthetic review demo is included.

Only the current setup, architecture, data contract, result summary, code index and collaborator handoff enter `code/docs/` in the export. Early integration notes and experiment diaries remain in the local workspace, where their linked artifacts are available. Recorded research scripts retain their original paths and can require those local artifacts; the portable commands are identified in `GETTING_STARTED.md`.

AB3DMOT is excluded from the source tree because its supplied license restricts redistribution. Keep the dependency in your local research installation, or obtain it from its upstream authors under their terms. The detector therefore requires this external dependency before full tracking can run. AnyCalib retains Apache-2.0; the BEVHeight upstream notice is included. [THIRD_PARTY.md](../../THIRD_PARTY.md) records scope and upstream links.

Original project code uses MIT. This does not license private datasets, third-party code, model weights or manuscript content. Distribute those separately only under their applicable terms. A source-only publication does not claim that unreleased assets can be downloaded from a nonexistent URL.

## Before pushing

1. Inspect `release_report.json` and `SOURCE_MANIFEST.json` in the exported bundle.
2. From the exported tree, run the synthetic demo, core/review tests and `doctor`. Missing field models/data are expected.
3. Confirm the tree contains code, tests, configs, documentation and synthetic demo assets; manuscript files and installation metadata must remain excluded.
4. Initialize your repository in the exported tree, add your chosen remote, review `git diff --cached`, and push when ready. No remote is configured by these tools.

The GitHub Actions workflow runs CPU tests on Linux and Windows. GPU inference, private-data reproduction, browser interaction and paper rendering are separate checks; CI does not pretend to cover them. The local release audit records what was actually executed.
