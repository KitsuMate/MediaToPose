# Learned correction: RoHM root only, Kimodo span repair (not scheduled)

Moved out of the 0.3.0 plan on 2026-10-19. 0.3.0 integrates Kimodo for
generation and in-betweening only (`docs/plans/0.3.0-kimodo.md`); using it as
a capture clean-up module is planned for a later version, on top of that
integration (the SOMA conversion, sampler, guide warp and apply path it
builds). Phase names below keep their original C0–C3 labels.

## Why

**RoHM pose correction cannot fix what it is used for.**
`iter_correct_window` (`core/motion/rohm_correction.py:482-591`) runs
TrajNet (100 steps), PoseNet (1,000), controlled TrajNet (100) and PoseNet
again (1,000): 2,200 ONNX calls per 144-frame window, on the main thread.
The pose result then barely matters:

- it replaces only joints below the visibility threshold;
- the detection filter has already interpolated those joints;
- it may move them at most 2° from that interpolation
  (`MAX_LEARNED_GLOBAL_POSE_DELTA_DEGREES`, line 90).

So RoHM can neither restore a missing gesture nor fix a wrong limb. The
root result is usable, but it comes from the final PoseNet pass, so "root
only" (`learned_pose_residual` off) still pays all 2,200 calls. Kimodo's
constraint branch was trained to fill motion between keyframes and along a
root path. Capture can supply exactly that: trusted frames as full-body
guides, trusted hands and feet as end-effector guides, and the corrected
root as a dense path. Without a prompt, it does not need the 5.8 GB text
encoder.

## Plan

Kimodo is a motion prior trained on clean mocap. It can repair a span that
the detector got wrong. It can also pull an unusual performance (floor work,
stylised dance) toward typical motion. The design therefore regenerates only
what capture marks as unreliable, keeps trusted frames exactly, and measures
both repair and fidelity before choosing a default.

### C0. Measure before choosing (`ci/correction_eval.py`)

- **Synthetic set, with ground truth.**
  - **Clean source clips:** about 10 clean clips on the VRoid and Rigify
    fixtures, covering walk, run, turn, sit, wave, jump, crouch, dance and
    floor work.
    - The ground truth must not come from Kimodo itself, or the comparison
      favours it. Use CC-licensed or maintainer-provided mocap (see
      Decisions).
  - **Corruptions, at three strengths each:**
    - per-joint jitter;
    - occlusion dropouts: one limb at low confidence for 5–60 frames;
    - single-frame flips (a 180° twist);
    - root noise and drift;
    - injected foot skating.

    Confidence is lowered where a corruption applies, as the detection
    filter would see it.
- **Real set.** Fixture videos through SAM 3D and MediaPipe. There is no
  ground truth, so this set uses the `evaluate` metrics,
  `landmark_fit_error_deg` and a visual review.
- **Configurations:**
  1. no learned correction;
  2. RoHM as today (full, with `learned_pose_residual`);
  3. RoHM root only as today (4 passes, pose not applied);
  4. RoHM TrajNet only (pass 1, 100 calls);
  5. Kimodo `SPANS`;
  6. Kimodo `REFINE`;
  7. RoHM root (4 or 3, whichever wins) followed by Kimodo `SPANS`.
- **Measures:**
  - Ground-truth error on the synthetic set: joint position (m), local
    rotation (°) and root (m), reported separately for corrupted and clean
    frames.
  - On clean frames: change against the uncorrected input
    (`fidelity_deg`).
  - Foot sliding, jitter, penetration, limit violations and root spikes.
  - Seconds per 100 frames on WebGPU and on CPU.
- **Decision rules, fixed before the run:**
  - **RoHM's pose path is removed** unless it lowers corrupted-frame
    rotation error by at least 10 % against configuration 1 without
    worsening any other measure by more than 10 %. Today's 2° cap makes
    removal the expected outcome.
  - **TrajNet only replaces the 4-pass root** if its root error is within
    10 % of configuration 3.
  - **Kimodo `SPANS` becomes the recommended pose repair** if it beats
    configurations 1 and 2 on corrupted-frame error, and changes clean
    frames by no more than 1° (`fidelity_deg`). It still ships as
    Experimental and off by default in 0.3.0; whether it becomes the
    default is decided in 0.3.x, after user feedback.
  - **`REFINE` ships** only if it lowers jitter and sliding with
    `fidelity_deg` under 5° on the real set. Otherwise it stays out.
- The results table goes into Outcome and the PR.

### C1. RoHM becomes root cleanup

- **Split the sampler.** Split `iter_correct_window` into
  `iter_correct_root`:
  - TrajNet pass 1 only, or passes 1–3 if C0 says the controlled pass is
    needed;
  - 100 or 1,200 calls instead of 2,200.

  The final PoseNet pass is no longer run for root work.
- **Remove the pose path:**
  - `learned_pose_residual`, the 2° bounded proposal and
    `_iter_apply_rohm_pose` are removed if C0 confirms.
  - The PoseNet session is not loaded when the pass is not needed. This
    saves memory and session time. The bundle stays as it is; a root-only
    bundle can follow later.
- **Settings stay compatible:**
  - The module keeps the id `ROHM_CORRECTION`, so saved configurations
    still load.
  - Its display name becomes "RoHM Root Cleanup".
  - A saved `learned_pose_residual: true` is ignored, with the warning
    `ROHM_POSE_REMOVED`.
  - `set_config` accepts the old key for one release.
- **Off the main thread.** TrajNet sampling moves to a worker thread
  through the polling pattern of C2. A RoHM run then no longer blocks the
  UI for one ONNX call per tick.

### C2. `KIMODO_CORRECTION` clean-up module

- **Placement.** It is a SEQUENCE module, after `DETECTION_FILTER` and
  `ROHM_CORRECTION`, before `TEMPORAL_CORRECTION`. It is
  Experimental and off by default.
- **Input.** The per-frame rig samples already captured for RoHM
  (`retarget/capture.py:_capture_rohm_frame_input`): rest-relative global
  rotations, joint offsets and per-joint confidence. These are converted to
  the SOMA skeleton as in K3 of the Kimodo plan, together with the detection filter's
  rejections.
- **Trust.** A joint is trusted on a frame when its confidence is at least
  `trust_threshold` and the detection filter neither rejected nor filled
  it.
  - Untrusted frames of any limb chain or the torso form spans.
  - Spans are dilated by 3 frames and merged across gaps shorter than 6.
- **Mode `SPANS` (default).** Regenerate each span plus 5 context frames on
  each side, the same transition length as upstream continuation. The
  model gets:
  - full-body guides on the context frames;
  - end-effector guides on frames inside the span where a hand or foot is
    still trusted;
  - a dense root path from the (RoHM-corrected) root over the whole
    window;
  - no text, so no encoder is needed.

  Spans longer than 300 frames use the Kimodo plan's windows (K1).
- **Mode `REFINE`** (only if C0 passes it).
  - Full-body guides every `guide_stride` trusted frames (default 6, about
    5 Hz), plus the dense root path.
  - The whole range is regenerated.
  - A spike has to show first that more than 20 full-body guides per window
    does not degrade the result.
- **Applying.** The same path as the Kimodo plan's K5:
  - SOMA rotations become `TargetPose` desired rotations (owner `KIMODO`)
    and go through IK.
  - The guide warp then makes trusted frames match the capture exactly,
    with smoothstep blending at span edges.
  - Kimodo's foot-contact channel replaces the heel-landmark heuristic for
    `TEMPORAL_CORRECTION.ground_contacts` inside regenerated spans.
- **Settings:** `mode`, `trust_threshold`, `max_span_frames` (spans
  longer than this are left to the detection filter, with a warning),
  `guide_stride`, `seed`, `steps` (25), `guide_guidance` (2.0), and an
  optional `prompt` that names the action to steer long spans (the prompt
  needs the encoder).
- **Threading.** Sampling runs in a worker future on `core` types. The
  sequence generator yields `Progress` while it polls, so the panel stays
  responsive and Cancel works between steps.
- **Report.** `cleanup_steps` statistics:
  - `spans` (ranges and lengths);
  - `frames_regenerated`;
  - `max_change_deg` and `fidelity_deg`;
  - `seconds`.

  Warnings: `KIMODO_SPAN_TOO_LONG`, `CORRECTION_LARGE_CHANGE` (any trusted
  frame moved more than 2° before the warp, a sign of unusual motion) and
  `KIMODO_SKIPPED` (model missing, mapping incomplete).
- **Requirements.** The 22-semantic mapping (as RoHM) and the Kimodo
  denoiser. Licence: NVIDIA OML only; Llama 3 applies only when a prompt is
  used.

### C3. Correcting animation that did not come from capture

- `api.correct(armature, frame_range, spans="auto" | [(first, last), ...],
  mode, ...)` runs the C2 engine on any action.
  - Spans are given explicitly, or chosen from `evaluate`'s
    `worst_frames`: jitter, limits, twists and root spikes.
  - Trust is 1.0 outside spans and 0.0 inside, so keys outside spans stay
    as they are.
- **"Fix a section"** is the same call: give the span, and the frames
  around it become the guides. The keys inside it are replaced in one
  transaction, which rolls back on cancel.
- It shares `GenerateResult` (Kimodo plan, K5), with `spans` added, so the metrics and
  `render` work the same way.

## Decisions for the maintainer

1. **Ground truth for C0.** Which clean mocap may be used? Candidates:
   - a CMU Graphics Lab subset, whose terms allow free use;
   - the maintainer's own captures;
   - clean Rigify/VRoid animations made for the purpose.

   Recommended: about 10 short CMU clips, retargeted once and kept in
   `.workspace/`. Commit only what the licence allows.
2. **RoHM's future.** If C0 shows TrajNet alone is enough, should a
   root-only bundle be published to `KitsuMate/rohm-onnx`? That would be
   a smaller download and needs upload approval. Or should the current
   bundle stay, with PoseNet simply never loaded? Recommended: keep the
   bundle in 0.3.0 and decide on publishing once the module has settled.
3. **Default correction.** Should Kimodo `SPANS` stay opt-in even
   if it wins C0, as planned, or be on by default for video captures when
   the model is installed? Recommended: opt-in, with the panel suggesting
   it when a capture reports `DETECTIONS_REJECTED` or `GAPS_FILLED`.
