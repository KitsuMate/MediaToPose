![Media to Pose: image to detected skeleton to posed character](docs/media/banner.png)

Media to Pose is a Blender 5.2 extension that creates a pose or animation
from an image or video. It can capture the body, hands, and face, then apply
the result to a selected character rig.

## Basic usage: image

1. Open **3D View > Sidebar > Media to Pose > Create from Media**.
2. Select the character armature or mesh and choose **Use Selected**.
3. Choose a **Source Image**.
4. Select MediaPipe or SAM 3D Body and enable the parts you need.
5. Choose **Apply Pose**.
6. If a part does not move, open **Rig Setup** and check its mapping.

MediaPipe works offline and supports body, hands, and face. SAM 3D Body
supports body and hands and needs a separate model download.

## Basic usage: video

[Watch the 12-second plugin preview](docs/media/preview.mp4).

1. Select the target character and choose a **Source Video**.
2. Select the capture provider and the body parts you need.
3. Set **Character Movement** to **From Video**, **In Place**, or
   **Keep Existing**.
4. Move the Blender timeline to the first output frame.
5. Choose **Apply Video Motion**.

The extension analyses the video, repairs short gaps, runs enabled cleanup
steps, and writes the result to the active animation. Use **Apply Current
Frame** when you only want the frame shown in the video viewer.

## Install

Download the package for your platform. In Blender, open
**Preferences > Get Extensions**, use **Install from Disk**, and select the
ZIP file. Do not unpack the ZIP file first.

Supported packages:

- Windows x64
- Linux x64
- macOS Apple silicon

## Post processors

- **Torso Curve (Experimental)** adjusts the curve of the spine. It is off
  by default.
- **Shoulders** adds clavicle movement when the arms are raised.
- **Elbows & Arm Twist** limits unstable arm twist and keeps the elbow bend
  direction.
- **Knees & Leg Twist** limits unstable leg twist and keeps the knee bend
  direction.
- **Reduce Camera-Induced Sliding (Experimental)** removes sideways root
  movement caused by camera motion in a video.
- **Reject Bad Detections & Fill Gaps (Experimental)** rejects weak or sudden
  detections and repairs short missing sections.
- **RoHM Cleanup (Experimental)** uses an optional model to repair masked root
  movement in a video. Its **AI Pose Cleanup** option is Alpha and is disabled
  by default. The complete RoHM processor is also disabled by default.
- **Smooth Motion & Foot Contact (Experimental)** smooths motion, stabilises
  hands, and keeps planted feet in place.

Cleanup settings are stored on the armature and run in a fixed order. You can
enable or disable each step under **Pose & Motion Cleanup**.
All video cleanup processors are experimental.

## Features that need an additional model

- **SAM 3D Body** provides detailed body and hand capture. The model download
  is about 1.76 GiB. Face capture is not available with this provider.
- **Perspective Correction with AnyCalib** estimates camera lens settings for
  better depth and root placement.
- **RoHM Cleanup (Experimental)** repairs selected video motion gaps. Its
  **AI Pose Cleanup** option is Alpha and disabled by default. The model
  download is about 222 MB.

- **Generate Motion with Kimodo** (NVIDIA Kimodo-SOMA-RP v1.1, converted to
  ONNX by KitsuMate) creates body motion from a text prompt, between keyed
  poses, through waypoints or along a curve. The motion model download is
  about 0.6 GiB (NVIDIA Open Model License). Prompts also need the LLM2Vec
  text encoder, about 5.5 GiB (Meta Llama 3 Community License); generating
  from keyed poses alone does not. Built with Meta Llama 3. Kimodo makes
  realistic single-person motion on flat ground, without fingers or face.

These models are not included in the plugin ZIP. The extension downloads a
pinned model only after you choose **Install**. It checks file size and SHA-256
before use and stores the model outside the extension folder.

## Face output

MediaPipe face capture drives one of four outputs:

- **ARKit** shape keys (the 52 Apple blendshapes, with or without a prefix
  such as `arkit_`),
- **FACS** action-unit shape keys,
- **VRM** expressions for VRoid and other VRM characters (`Fcl_*` keys),
- **face bones**, such as an Auto-Rig Pro face; controls move in face space
  and the lids follow the gaze.

Eye bones follow the gaze for every output. For video, **Smooth Motion**
also calibrates each expression's resting level, smooths the face without
losing blinks, and holds it briefly when it is lost. Rig Setup shows how
much of the chosen output the character has.

Face capture is still marked experimental: Rigify face controls are not yet
verified, a wink reads as a one-sided squint, and relaxed faces in single
images can show slightly raised brows (video calibrates this away).

## Scripting and command line

Everything in the sidebar also runs without UI, from Python inside Blender
(including `blender -b` and AI agents using the Blender MCP server) and from
the command line:

```
blender -b scene.blend -c media_to_pose capture --media clip.mp4 \
    --armature Rig --parts body,hands --out result.json --save
blender -b scene.blend -c media_to_pose evaluate --armature Rig \
    --result result.json --json
blender -b scene.blend -c media_to_pose generate --armature Rig \
    --prompt "A person walks forward." --frame-range 1:120 \
    --waypoint 1:0,0 --waypoint 120:0,-3 --out walk.json --save
blender -b scene.blend -c media_to_pose render --armature Rig \
    --result walk.json --out walk.mp4
```

Each capture returns a JSON result (frames, confidence, rejected and filled
frames, clean-up statistics, warnings with codes). `evaluate`
measures foot sliding, jitter, twist, root speed spikes and, for faces,
expression fit, blink recall and face jitter. Clean-up settings can be
changed and the capture re-applied from its retained detections in seconds.
See [docs/agents.md](docs/agents.md) for the Python API, recipes and what
each metric means.

## Development

Pull requests run Ruff and the complete Blender test suite against an
installed Linux release package. Weekly integration tests run real MediaPipe
inference. SAM 3D GPU testing uses a protected runner.

The extension source is in `media_to_pose/`; `AGENTS.md` describes its
layout, the layering rules and the checks to run before a commit. Release
scripts are in `ci/` and the release steps in `docs/RELEASING.md`. Local
files and downloaded models belong in `.workspace/`. Built release packages
belong in `dist/`.

## License

Media to Pose is licensed under GPL-3.0-or-later. See `LICENSE` and
`THIRD_PARTY_NOTICES.md`.
