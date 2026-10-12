# Working on Media to Pose

Media to Pose is a Blender 5.2 extension (Python 3.13). This file describes
the layout, the rules every change follows and the commands that check them.
The 0.2.0 plan in `docs/plans/0.2.0.md` explains why the rules exist.

## Layout

```
media_to_pose/            extension source; this is what ships
  __init__.py             register()/unregister() only
  core/                   stdlib + numpy only: types, errors, requests,
                          coordinator, fs, media_io, cache/, camera/, motion/
  providers/              media → landmarks (mediapipe/, sam3d/)
  backends/               model install/load lifecycle (anycalib/, rohm/, sam3d/)
  rig/                    rig profiles, mapping, face schemes, planning
  retarget/               landmarks → bone rotations and keyframes
  postprocess/            cleanup modules, settings schema, pipeline
  session/                CaptureSession, transaction, results
  evaluate/               metrics and comparison renders
  api/                    synchronous Python API and CLI command
  blender/                RNA properties, operators, UI panels
tests/unit/               plain pytest tier, no Blender
tests/                    Blender suite (tests/run_blender_tests.py)
tests/fixtures/           rigs, media, golden snapshots
ci/                       release and integration scripts
docs/                     user docs, plans, agent recipes
```

Local runtimes, downloaded models and scratch files go in `.workspace/`
(ignored). Built packages go in `dist/` (ignored). Never commit either.

## Layers

Allowed import direction:

```
blender → api → session → retarget, postprocess, rig, providers, backends → core
```

- `core/` imports only the standard library and numpy. No `bpy`, no
  `mathutils`.
- `mathutils` is allowed in `rig/`, `retarget/` and `postprocess/`.
- Nothing imports `blender/` except the package `__init__.py`.
- `tests/unit/test_layering.py` parses every module and enforces this. Its
  `ALLOWLIST` holds the known violations; it may only shrink.

## Blender access

- Only `blender/operators/*` and `api/` read `bpy.context`. Everything else
  takes scene, armature and fps as arguments.
- Blender data is changed only on the main thread. Worker threads see only
  `core` types.
- PropertyGroups are converted to frozen dataclasses once, at the operator or
  API boundary. Lower code never does `getattr(settings, "x", default)` on
  RNA.

## Long-running work

- `iter_x(...)` yields `Progress`; a plain `x(...)` drains it.
- A `CaptureSession` exposes `step()` and `cancel()`. The modal operator, the
  synchronous API and the CLI are all drivers of `step()`.

## Errors and logging

- Raised errors subclass `core.errors.MtpError` (`MediaError`,
  `ModelMissing`, `MappingIncomplete`, `OperationBusy`, `Cancelled`,
  `BackendError`). Each has a stable `code` string.
- `except Exception` only at thread entry points, operators and the CLI
  entry, and it must log the traceback. Ruff `BLE` enforces this.
- Use `logging.getLogger(__name__)`; everything sits under the
  `media_to_pose` logger. No `print` outside `ci/`.

## Data and files

- No pickle. Intermediate and cached data use npz (`allow_pickle=False`) plus
  JSON manifests with `schema_version`.
- Atomic JSON/npz writes live in `core/fs.py` only.
- `pathlib.Path` for all paths.

## Python style

- Ruff rules: see `pyproject.toml` (adds `UP`, `RET`, `PIE`, `PTH`, `BLE`,
  `TID252`). Inside the package only relative imports work, because Blender
  loads it as `bl_ext.<repository>.media_to_pose`; `TID252` therefore
  applies to `tests/` and `ci/`.
- Built-in generics (`list[int]`), `X | None` instead of `Optional[X]`.
- Frozen dataclasses for requests, results and snapshots; `Enum` for modes
  and states instead of bare strings.
- Pyright basic mode on the modules listed in `pyproject.toml`; widen the
  list when a module passes.
- Soft limits: about 600 lines per module and 80 per function. Note
  deliberate exceptions in review.

## Blender naming

- `bl_idname` prefix `mtp.`; classes `MTP_OT_`, `MTP_PT_`, `MTP_UL_`,
  `MTP_MT_`; PropertyGroups `MTP<Thing>`.
- Each `blender/` module exports `CLASSES`; only those are registered.

## Commands

Set up a local Python 3.13 environment once (numpy must match Blender's):

```
uv venv --python 3.13 .workspace/venv
uv pip install --python .workspace/venv numpy==2.3.4 pytest==9.1.1 ruff==0.12.12 pyright==1.1.406
```

Checks, run before every commit:

```
ruff check .
ruff format --check .
pyright
python -m pytest                      # unit tier, no Blender, < 30 s
blender --background --factory-startup --python-exit-code 1 \
    --python tests/run_blender_tests.py
```

The Blender suite needs the bundled wheels installed in the Blender profile
it runs with. Build and install the package for your platform into an
isolated profile under `.workspace/` (set the `BLENDER_USER_*` variables as
in `.github/workflows/ci.yml`), then run the suite against the source
checkout.

Golden retarget snapshots live in `tests/fixtures/golden/`. After an
intended change to retargeting output, regenerate them with
`MTP_UPDATE_GOLDEN=1` and review the diff.

## Commits and pull requests

- Conventional prefixes: `refactor:`, `feat:`, `fix:`, `test:`, `ci:`,
  `docs:`, `build:`, `style:`.
- Moves and renames get their own commits with no behaviour change.
- No AI attribution lines in commit messages or pull request descriptions.
- Pushing and opening pull requests need the maintainer's approval.
