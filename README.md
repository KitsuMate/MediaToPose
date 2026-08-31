<p align="center">
  <img src="docs/media/icon.png" alt="Media to Pose icon" width="96">
</p>

# Media to Pose

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

These models are not included in the plugin ZIP. The extension downloads a
pinned model only after you choose **Install**. It checks file size and SHA-256
before use and stores the model outside the extension folder.

## Face output

MediaPipe face capture is experimental. You can select ARKit morphs, FACS
morphs, or facial bones. The selected character must have matching shape keys,
bones, or custom properties in **Rig Setup**.

## Development

Pull requests run Ruff and the complete Blender test suite against an
installed Linux release package. Weekly integration tests run real MediaPipe
inference. SAM 3D GPU testing uses a protected runner.

Release scripts are in `ci/`. Local files and downloaded models belong in
`.workspace/`. Built release packages belong in `dist/`.

## License

Media to Pose is licensed under GPL-3.0-or-later. See `LICENSE` and
`THIRD_PARTY_NOTICES.md`.
