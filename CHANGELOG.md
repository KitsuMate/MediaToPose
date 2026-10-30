# Changelog

## Unreleased (0.3.0)

### Added

- Generate Motion with Kimodo: text to motion, in-betweening from keyed
  poses, root waypoints and curve paths, several prompts on one timeline.
  Keyed poses are reached exactly. Available in the sidebar, as
  `api.generate` and as the `generate`, `guides` and `encode-prompt`
  commands.
- `models install` and `api.install_model` download optional models from
  scripts, with licence acceptance recorded per model revision.
- `render` turns a generate result into a rig contact sheet or an MP4.
- `capture --key` keys an image pose at a chosen frame, so an image can
  block a pose for generation.
- Generated motion plants the feet during the model's contacts (as Kimodo's
  own post-processing does), keeps knees and elbows as hinges, and uses the
  full-precision model by default; reduced-precision graphs shake.

### Fixed

- AnyCalib, RoHM and SAM 3D installs respect Blender's offline setting.
- `providers` reports MediaPipe correctly from the command line.
- Interrupted downloads never resume a part file of another model revision.

## 0.2.1

Face capture is stronger and measurable: more rigs are recognised, bone
faces move correctly, video expressions are cleaned up, and results can be
checked with numbers and face close-ups.

### Added

- VRM face output for VRoid and other VRM characters: per-eye blinks,
  A/I/U/E/O mouth shapes and the emotion expressions on the `Fcl_*` keys.
- Smooth Motion cleans video face expressions before retargeting: neutral
  calibration of each expression's resting level, lag-free smoothing that
  keeps blinks, and holding a lost face briefly before it fades to neutral
  (`neutral_calibration`, `preserve_blinks`).
- Bone face lids follow the gaze up and down.
- Input and output ranges for every face mapping, replacing gain, offset,
  clamp and invert: the detected values treated as rest and full, and the
  target's value at rest, full and (for brows, mouth corners, lids, jaw
  side, gaze) negative full, in the target's units (key value, metres,
  degrees). 0.2.0 mappings and presets are converted. The Auto-Rig Pro
  defaults are fitted to ARKit keys (`ci/fit_face_travel_blender.py`); bone
  lids close fully from a detected blink of 0.75.
- Face metrics in `evaluate`: `expression_fit`, `blink_recall`,
  `face_jitter` and `asymmetry`; `--face-output` on the command line.
- `render_comparison(face=True)` / `render --face`: the detected face next
  to the rig's face. Capture results record their `face_output`.
- Test fixtures: an Auto-Rig Pro face rig with prefixed ARKit keys and 13
  CC0 face photos with checks against the real detector; golden snapshots
  of ARKit, FACS, bone and VRM face output and of a face video sequence.

### Fixed

- Model-install lock checks no longer signal or terminate Windows processes.
- `render --out` keeps the PNG sheet instead of overwriting it with JSON.
- ARKit keys with prefixes (`arkit_`, `facs_bs_`, `ctrl_`, `blendShape.`)
  are recognised, correctives are left alone, and only one key per mesh is
  driven for each expression.
- Face bone controls move along the face's own axes by face-sized amounts,
  so lids close, brows and mouth corners move up and down, and the
  Auto-Rig Pro jaw opens.
- Eye gaze moves look-at targets by a natural angle and drives the eyes for
  every face output, not only bones.
- FACS jaw units: AU25 (lips part), AU26 (jaw drop) and AU27 (mouth stretch)
  are separated.
- Face mapping completeness is reported only when face capture is on;
  `describe_rig` no longer lists unclassified keys as a shape-key scheme.

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
