# Changelog

## 0.2.0

The panel works as before. Underneath, the code is reorganised into layers
with enforced rules, and everything the panel does can now be scripted and
run from the command line.

### Added

- Python API (`media_to_pose.api`): `capture`, `retarget_cached` (re-apply
  retained detections with new clean-up settings, in seconds),
  `describe_rig`, `auto_map`, `get_config` / `set_config`,
  `list_postprocessors`, `list_providers`, `model_status`,
  `export_detections` / `import_detections`, `evaluate` and
  `render_comparison`. It works in the foreground, with `blender -b` and from
  the Blender MCP server.
- Command line: `blender -b scene.blend -c media_to_pose <command>` with JSON
  output and one exit code per error class. Nothing is saved without
  `--save`.
- A JSON result for every capture: action and slot, frame ranges,
  per-frame and per-part confidence, rejected, filled and foot-contact
  frames, each clean-up step with its settings and statistics, mapping
  issues and warnings with stable codes.
- Animation metrics: foot sliding, ground penetration, jitter, twist and
  joint-limit violations, root speed spikes and elbow/knee fit against the
  detections.
- Comparison renders: source frames with the detected skeleton beside the
  posed rig, for review by people and vision-capable agents.
- Output modes for scripted captures: a new action, replace the current
  action, or a new NLA strip; and source frame ranges for partial re-runs.
- `docs/agents.md`: recipes, result fields, metric ranges and which setting
  to change for which symptom.

### Changed

- The extension source moved into `media_to_pose/` and is split into
  `core/`, `providers/`, `backends/`, `rig/`, `retarget/`, `postprocess/`,
  `session/`, `evaluate/`, `api/` and `blender/`. A test enforces the import
  direction between them.
- AnyCalib, RoHM and SAM 3D Body share one model backend for download,
  verification, install, load and removal. Pinned revisions, checksums and
  storage locations are unchanged.
- Errors carry stable codes. Messages go through the `media_to_pose` logger
  instead of `print`.
- Temporary capture data uses npz files instead of pickle. No pickle is read
  or written anywhere.
- Clean-up settings are described once, in `postprocess/settings_schema.py`;
  the panel, validation and the API read that description.

### Fixed

- Mapping validation after loading a file uses each scene's own capture
  settings instead of the active scene's.

### Compatibility

- Scene settings, rig mappings, mapping presets and installed models from
  0.1.0 are used as they are. Retained detections keep their format and are
  reused.
- Scripts that imported internal modules (for example
  `media_to_pose.mapping`) need the new module paths, or better, the API.
