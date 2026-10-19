# Media to Pose for scripts and agents

Everything the sidebar does can run without UI: from Python inside Blender
(including `blender -b` and the Blender MCP `execute_blender_code` tool) and
from the command line. This page is written for coding agents and pipeline
scripts: what to call, what comes back, how to judge the result, and which
setting to change when something looks wrong.

- [Getting started](#getting-started)
- [Recipes](#recipes)
- [The capture result](#the-capture-result)
- [Metrics and what to change](#metrics-and-what-to-change)
- [Configuration](#configuration)
- [Command line reference](#command-line-reference)
- [Errors and exit codes](#errors-and-exit-codes)

## Getting started

The extension must be installed and enabled in the Blender profile you run.
Inside Blender it is imported under its extension package name:

```python
import importlib

api = importlib.import_module("bl_ext.user_default.media_to_pose.api")
```

(`user_default` is the repository the extension was installed into. Find
the right name with `[n for n in bpy.context.preferences.addons.keys() if
n.endswith("media_to_pose")]`.)

From a shell, the same functions run through Blender's command interface:

```
blender -b scene.blend -c media_to_pose capture --media clip.mp4 \
    --armature Rig --parts body,hands --out result.json --save
```

Nothing is written to the .blend file unless you pass `--save` (CLI) or
save it yourself (Python). The command line prints JSON with `--json`.

A first session usually looks like this:

```python
api.describe_rig("Rig")            # is the rig mapped well enough?
result = api.capture("clip.mp4", "Rig", parts={"body", "hands"})
metrics = api.evaluate("Rig", result=result)
api.render_comparison("check.png", "Rig", result=result)   # look at it
```

Options you leave out come from the scene's Media to Pose settings, so a
script and the panel produce the same animation for the same settings.

## Recipes

### Animate a rig from a video

```python
api.auto_map("Rig")                       # only needed once per rig
result = api.capture(
    "clip.mp4", "Rig",
    parts={"body", "hands"},
    root_motion="AUTOMATIC",              # or IN_PLACE / OFF
    start_frame=1,
    result_path="result.json",
)
assert result.succeeded
print(result.action, result.timeline_frame_range, result.warnings)
```

Videos are keyed (`mode="KEYFRAMES"`); images set one pose (`mode="POSE"`)
at the current frame. Choose a video frame for a pose with `frame=` (0-based).

Captures keep their detections (the detection cache) unless you pass
`keep_cache=False`. Keep them: every refinement below re-applies them in
seconds without running detection again.

### Fix a bad section

1. Find it: `metrics.worst_frames` lists the timeline frames with the worst
   jitter, foot sliding, twist and root speed; `result.rejected_frames` and
   `result.filled_frames` show where the detections were weak.
2. Look at it: `api.render_comparison("bad.png", "Rig", result=result,
   frames=[40, 41, 42])`.
3. Mask the frames the detector got wrong and re-apply only that section:

```python
fixed = api.retarget_cached(
    "clip.mp4", "Rig",
    parts={"body", "hands"},
    frame_range=(36, 48),                 # source frames, end excluded
    # 1-based panel numbers: source frames 39-42 in result numbering
    config={"DETECTION_FILTER": {"manual_bad_frames": "40-43", "manual_bad_scope": "ARM_L"}},
)
```

`frame_range` keys only those source frames, at the same timeline frames as
the full capture, and leaves the rest of the action untouched. Note the two
numbering schemes: result frames and `frame_range` are 0-based, while
`manual_bad_frames` uses the panel's 1-based source frame numbers (source
frame 40 is `"41"`).

To change detections by hand instead, export them, edit the JSON or npz, and
import them before `retarget_cached` (see `api.export_detections`).

### Tune foot contact

```python
for strength in (0.3, 0.6, 0.9):
    result = api.retarget_cached(
        "clip.mp4", "Rig", parts={"body"}, output="NLA_STRIP",
        config={"TEMPORAL_CORRECTION": {"enabled": True, "settings": {
            "ground_contacts": True, "smooth_motion": True,
            "smoothing_strength": strength}}},
    )
    metrics = api.evaluate("Rig", action=result.action, result=result)
    print(strength, metrics.foot_sliding_m_per_s, metrics.ground_penetration_m)
```

`output="NLA_STRIP"` puts each attempt on its own NLA track and keeps the
current action, so versions sit side by side. Keep the best one with
`api.set_config("Rig", {"post_processing": {...}})` and capture with
`output="REPLACE"`.

### Set up face capture

```python
rig = api.describe_rig("Face")
print(rig["completeness"]["arkit"], rig["sections"]["SHAPES"]["unmapped"])
result = api.capture("face.mp4", "Face", parts={"face"}, face_output="ARKIT")
print(result.face_output_warning)
metrics = api.evaluate("Face", result=result)
print(metrics.expression_fit, metrics.blink_recall, metrics.face_jitter)
api.render_comparison("face.png", "Face", result=result, face=True)
```

Pick the face output by rig type (`describe_rig` shows each mesh's
shape-key `schemes` and the mapped `face_bones`):

| Rig | `face_output` | Check first |
| --- | --- | --- |
| ARKit / Apple blendshapes (52 keys, any prefix such as `arkit_` or `facs_bs_`) | `"ARKIT"` | `completeness.arkit` close to 1.0 |
| FACS action-unit keys (`AU12_L`, ...) | `"FACS"` | `completeness.facs` |
| VRoid and other VRM characters (`Fcl_*` keys) | `"VRM"` | `completeness.vrm` |
| Bone-driven face (Auto-Rig Pro face, Rigify face controls) | `"BONES"` | `completeness.face`, `sections.FACE_RIG.unmapped` |

`"VRM"` drives the `Fcl_*` expression keys (per-eye blinks, the A/I/U/E/O
mouth shapes, smiling, sad and angry mouth and brows, wide and smiling eyes)
from the same ARKit coefficients. Eye bones are driven by gaze for every
face output; with look-at rigs (an eye bone aiming at a target) the target
moves. Bone output moves face controls in face space by a share of the eye
distance, and the upper lids follow the gaze up and down.

`render_comparison(..., face=True)` shows each frame as the detected face
next to the rig's face, with the camera following the head.

Every face mapping (shape key, face bone control, custom property, eye
gaze) has an input and an output range, editable in Rig Setup and through
`set_config` (`mappings[].input_zero`, `input_full`, `output_zero`,
`output_full`, `output_negative`):

- the detection between `input_zero` (rest) and `input_full` is the
  strength, clamped to 0..1 (-1..1 for signed controls: brows, mouth
  corners, jaw side, lids, gaze);
- the target goes from `output_zero` to `output_full` (`output_negative` on
  the negative side), times `influence`, in its own units: shape key value,
  metres along a bone control's face-space axis, degrees about it, the
  custom property's value, or degrees of gaze for the eyes.

Defaults: shape keys 0 to 1 in and out; gaze 18 degrees; face bones a share
of the eye distance for moving controls or 25 degrees for rotating ones.
The Auto-Rig Pro shares were fitted to the ARKit keys of the project's
FaceFX test rig (`ci/fit_face_travel_blender.py`). Bone lids treat a
detected blink of 0.75 as fully closed (`input_full` 0.75): with shape
keys, squint and look-down keys close the eye further than the blink
alone, and a bone lid has only the one control. `get_config` lists every
mapping's ranges. Mappings and presets from 0.2.0 (gain, offset, clamp,
invert) are converted.

For video, **Smooth Motion** (`TEMPORAL_CORRECTION`) cleans the face before
retargeting: it removes each expression's resting level
(`neutral_calibration`), smooths without lag at `smoothing_strength` while
keeping blinks (`preserve_blinks`), zeroes values under 0.05, and holds a
lost face for half a second before fading it to neutral.

VRM output targets VRoid-style `Fcl_*` shape keys; VRM preset metadata and
arbitrary preset-named keys are not resolved. Face confidence reports
detection presence, not a calibrated occlusion or head-turn score.

| Symptom | Look at | Change |
| --- | --- | --- |
| Eyes do not close | `blink_recall` | Keep `preserve_blinks` on, lower `smoothing_strength`; lower the blink mapping's `input_full` (e.g. 0.8: the detector rarely reports a full blink); for bones (`FACE_LID_*`, closed from 0.75 by default) lower `input_full` further or raise `output_full` |
| Mouth or brows too weak | `expression_fit`, the face sheet | Lower that mapping's `input_full` or raise its `output_full` (above 1 needs the shape key's `slider_max`) |
| Face twitches | `face_jitter` | Enable `TEMPORAL_CORRECTION` or raise `smoothing_strength` |
| Relaxed face looks surprised (raised brows) | the sheet's neutral frames | Video: keep `neutral_calibration` on; stills: raise the brow mappings' `input_zero` (e.g. 0.4) |
| Sides swapped | `asymmetry`, the sheet | Check `mirror`: mirroring flips the whole performance, so a left wink becomes a right one |
| An expression never moves | `sections.SHAPES.unmapped` in `describe_rig` | Map it in Rig Setup, or accept that the rig has no such key |

Known limits: MediaPipe reports a resting brow raise of about 0.5 on many
relaxed faces, which only video captures can calibrate away; a wink comes
through as a one-sided squint rather than a one-sided blink; extreme
close-ups where the face fills the frame can be misfitted. Sparse bone
controls can produce a weaker broad smile than ARKit keys even when
`expression_fit` is low; prefer ARKit on rigs with suitable keys when
likeness matters. VRoid's authored joy key can open the mouth even for a
closed-mouth smile; combining it with visemes can exaggerate that opening.
`expression_fit` measures mapped values, not visual likeness to the source.

### Run from the Blender MCP server

`execute_blender_code` runs in the open Blender session, so the same calls
work there. Captures block until they finish; give long videos a `timeout`
(seconds) and check `result.status`.

## The capture result

`capture` and `retarget_cached` return a `CaptureResult`; `result_path=`
(or `--out`) also writes it as JSON. `schema_version` changes when the layout
does. Functions that take a `result` accept the object, its dict or the JSON
path; `api.load_result(result)` returns the dict.

| Field | Meaning |
| --- | --- |
| `status`, `message` | `SUCCEEDED`, plus the panel's status line |
| `action`, `slot`, `output_mode`, `nla_track` | Where the animation went |
| `source_frame_range` | Applied source frames `[first, end)`, 0-based |
| `timeline_frame_range` | First and last keyed timeline frames |
| `frames_processed`, `cached` | Frames applied; whether detection was skipped |
| `cache_key` | Identity of the retained detections (empty when not kept) |
| `confidence.per_part` | Mean landmark confidence for `body`, `hands`, `face` |
| `confidence.per_frame` | Per applied frame: `source_frame`, `timeline_frame` and part confidence |
| `rejected_frames`, `filled_frames` | Source-frame ranges the detection filter masked or interpolated |
| `contact_frames` | Source-frame ranges with a planted foot |
| `cleanup_steps` | Each enabled clean-up module: `parameters` used and `statistics` (for example `rejected_observations`, `repaired_samples`, `contact_frames`, `root_drift_removed_mean`) |
| `mapping_issues` | Rig mapping entries with warnings or errors |
| `face_output_warning` | Why face output may be incomplete |
| `face_output` | The face output keyed (`ARKIT`, `FACS`, `VRM` or `BONES`), empty when the face was not captured; `evaluate` and `render_comparison(face=True)` use it |
| `warnings` | `{code, message}`; branch on `code` |

Warning codes: `DETECTIONS_REJECTED`, `GAPS_FILLED`, `MANUAL_MASK`,
`LOW_EVIDENCE`, `GROUND_CONTACTS`, `ROHM_APPLIED`, `ROHM`, `CAMERA_MOTION`,
`ROOT_MOTION`, `FACE_OUTPUT`, and `CAPTURE_WARNING` for anything else.

## Metrics and what to change

`api.evaluate(armature, result=..., action=..., frame_range=...)` samples the
evaluated pose at every frame and returns `Metrics`. Distances are metres,
speeds per second at the scene frame rate, angles degrees. Pass `result` to
measure the frames that capture keyed and to get `landmark_fit_error_deg`.

The ranges below are guidance for a person filmed at normal distance; check
the comparison render before acting on a borderline value.

| Metric | Good | Investigate | What to change |
| --- | --- | --- | --- |
| `foot_sliding_m_per_s`: horizontal foot speed while planted | < 0.05 | > 0.15 | Enable `TEMPORAL_CORRECTION.ground_contacts`; raise `smoothing_strength`; with a moving camera enable `CAMERA_MOTION_CORRECTION`; for in-place motion use `root_motion="IN_PLACE"` |
| `ground_penetration_m`: deepest foot point below `ground_height` | < 0.01 | > 0.03 | Pass the real floor as `ground_height`; enable `ground_contacts` |
| `jitter_mm`: RMS distance of joints from the midpoint of their neighbours in time | < 2 | > 5 | Enable `smooth_motion`, raise `smoothing_strength`; `stabilize_hands` for fingers; `DETECTION_FILTER` for isolated spikes |
| `twist_violations`: bone-frames twisted beyond 90° about the bone | 0 | > 0 | `ARM_AXIS_CORRECTION` (`upper_arm_twist_limit`, `forearm_twist_limit`), `LEG_AXIS_CORRECTION` (`thigh_twist_limit`) |
| `joint_limit_violations`: elbows or knees bent past 160°, or bones rotated over 170° from rest | 0 | > 0 | Mask the frames in `worst_frames.joint_limits` with `DETECTION_FILTER.manual_bad_frames`; check the mapping |
| `root_speed_spikes`: frames whose root speed is over four times the median and over half a body height per second | 0 | > 0 | `DETECTION_FILTER.reject_unusable_frames`; mask the frames; `CAMERA_MOTION_CORRECTION` for camera pans |
| `landmark_fit_error_deg`: mean difference between detected and rig elbow and knee bends | < 10 | > 20 | Lower `smoothing_strength` if over-smoothed; fix `mapping_issues`; check `mirror` |

Face metrics are measured when the capture used face output (or with
`face_output=`); they are in expression units (0 to 1) and `None` otherwise.
All but `face_jitter` need the result's retained detections.

| Metric | Good | Investigate | What to change |
| --- | --- | --- | --- |
| `expression_fit`: mean difference between each mapped expression's strength (detection through the input range) and the strength the rig shows (its value through the output range) | < 0.05 | > 0.15 | High for video with Smooth Motion is expected (it calibrates and smooths the face); otherwise check that `output_full` fits the shape key's slider range, and `sections.SHAPES.unmapped` in `describe_rig` |
| `blink_recall`: share of detected blinks that close the rig's eye to at least 0.8 | 1.0 | < 0.8 | Keep `TEMPORAL_CORRECTION.preserve_blinks` on; lower `smoothing_strength`; lower the blink mapping's `input_full` |
| `face_jitter`: RMS distance of face values from the midpoint of their neighbours in time | < 0.03 | > 0.08 | Enable `TEMPORAL_CORRECTION` (`smooth_motion`) or raise `smoothing_strength` |
| `asymmetry`: mean left/right difference on the rig where the detection was symmetric | < 0.05 | > 0.1 | Check the left/right mapping and `mirror`; a rig whose left and right keys differ in strength needs per-side `output_full` |

`foot_contact_ratio`, `penetration_frames` and `max_root_speed_m_per_s` add
context. `worst_frames` holds up to five timeline frames per problem.

## Configuration

- `api.list_postprocessors(armature=None)`: every clean-up module with its
  parameters (type, range, default, description); with an armature, the
  rig's current values.
- `api.get_config(armature)`: the rig's complete configuration (profile,
  mapping, FK/IK switches, clean-up modules), the same JSON as the panel's
  preset export.
- `api.set_config(armature, config)`: apply a complete configuration, or a
  partial one with only `post_processing`. Settings are validated against
  the schema before anything changes; unknown modules or settings, wrong
  types and out-of-range values raise `InvalidRequest`.
- `config=` on `capture` / `retarget_cached`: the same `post_processing`
  shapes, applied to one run only; the rig keeps its settings.

`post_processing` accepts the saved list form or
`{"MODULE_ID": {"enabled": true, "settings": {...}}}` or
`{"MODULE_ID": {"setting": value}}`.

Other introspection: `api.describe_rig(armature)` (mapping coverage per
section, unmapped targets, detected profile, shape keys per mesh and scheme,
and face bones),
`api.auto_map(armature)`, `api.list_providers()` and `api.model_status()`
(optional model install state and download sizes).

## Command line reference

`blender -b FILE -c media_to_pose COMMAND [options]`; add `--json` to print
the JSON document and `--save` to save the .blend afterwards.

| Command | Does |
| --- | --- |
| `capture --media M --armature A [--parts body,hands] [--mode] [--frame N] [--frame-range FIRST:END] [--start-frame F] [--root-motion] [--face-output] [--mirror] [--config FILE] [--output NEW_ACTION\|REPLACE\|NLA_STRIP] [--no-cache] [--timeout S] [--out FILE]` | `api.capture` |
| `retarget` (same options, without `--no-cache`) | `api.retarget_cached` |
| `describe-rig --armature A` | `api.describe_rig` |
| `auto-map --armature A [--force-profile]` | `api.auto_map` |
| `evaluate --armature A [--action] [--frame-range FIRST:LAST] [--result FILE] [--ground-height Z] [--face-output]` | `api.evaluate` |
| `config get --armature A [--out FILE]` / `config set FILE --armature A` | `api.get_config` / `api.set_config` |
| `postprocessors [--armature A]` | `api.list_postprocessors` |
| `providers`, `models` | `api.list_providers`, `api.model_status` |
| `export-detections PATH`, `import-detections PATH` | `api.export_detections`, `api.import_detections` |
| `render --armature A --result FILE --out PNG [--frames 10,20] [--count N] [--size PX] [--no-meshes] [--face]` | `api.render_comparison` |

## Errors and exit codes

Every failure raises a subclass of `MtpError` with a stable `code`; the
command line exits with the class's code and, with `--json`, prints
`{"error": {"code", "message", "details"}}`.

| Exit | Code | When |
| --- | --- | --- |
| 0 | | Success |
| 1 | `MTP_ERROR`, `INTERNAL_ERROR` | Unexpected failure (details in the log) |
| 2 | `INVALID_REQUEST` | Bad arguments, configuration or names; no matching retained detections |
| 3 | `MEDIA_ERROR` | Media missing or unreadable |
| 4 | `MODEL_MISSING`, `DEPENDENCY_MISSING` | Optional model or bundled dependency unavailable |
| 4 | `ONLINE_ACCESS_DISABLED` | A model must be downloaded but Blender's online access is off; nothing was downloaded |
| 5 | `MAPPING_INCOMPLETE` | The rig mapping does not cover the requested parts |
| 6 | `OPERATION_BUSY` | Another capture or model job is running |
| 7 | `CANCELLED` | Cancelled or timed out; Blender data was rolled back |
| 8 | `BACKEND_ERROR` | A model runtime failed |
| 9 | `CAPTURE_FAILED` | The capture started and failed; Blender data was rolled back |
| 10 | `LICENSE_REQUIRED` | A model's licence has not been accepted; `details` carries `license_id`, `summary`, `url` and the exact `accept_license` value |
| 11 | `INSUFFICIENT_RESOURCES` | Not enough disk or memory; `details` carries `resource`, `needed_bytes`, `available_bytes` and, when one exists, a `fitting_variant` |

On `LICENSE_REQUIRED`, show the licence (`details.url` and `summary`) to
the user and accept it only after the user agrees; an agent never accepts a
licence on its own. Acceptance is recorded per model revision, with when and
where (panel or command line) it was given.
