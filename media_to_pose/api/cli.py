"""The ``media_to_pose`` Blender command.

Run as ``blender -b scene.blend -c media_to_pose <command> [options]``.
Every command prints a short summary, or the full JSON document with
``--json``. The exit code is 0 on success and the error class's
``exit_code`` otherwise (see ``core/errors.py``). Nothing is saved unless
``--save`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import bpy

from ..core.errors import InvalidRequest, MtpError
from ..core.log import get_logger

# Functions, not submodules: the package re-exports ``capture`` under the
# name of its own submodule.
from .capture import capture, retarget_cached
from .config import get_config, list_postprocessors, set_config
from .detections import export_detections, import_detections
from .evaluation import evaluate, render_comparison
from .models import list_providers, model_status
from .rig import auto_map, describe_rig

COMMAND = "media_to_pose"
LOGGER = get_logger(__name__)


class _Parser(argparse.ArgumentParser):
    """Raise ``InvalidRequest`` instead of exiting the Blender process."""

    def error(self, message: str):
        raise InvalidRequest(f"{self.prog}: {message}")

    def exit(self, status: int = 0, message: str | None = None):
        if message:
            sys.stderr.write(message)
        raise _ParserExitError(status)


class _ParserExitError(Exception):
    def __init__(self, status: int):
        super().__init__(status)
        self.status = status


def _frame_range(value: str) -> tuple[int, int]:
    try:
        first, end = (int(part) for part in value.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use FIRST:END, for example 0:120") from exc
    return first, end


def _frames(value: str) -> list[int]:
    try:
        return [int(part) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated frames") from exc


def _read_json(path: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf8"))
    except (OSError, ValueError) as exc:
        raise InvalidRequest(f"Could not read {path}: {exc}") from exc


def _capture_options(parser: argparse.ArgumentParser, *, cached: bool) -> None:
    parser.add_argument("--media", required=True, help="image or video file")
    parser.add_argument("--armature", required=True, help="armature object name")
    parser.add_argument("--provider", help="MEDIAPIPE or SAM3D_BODY")
    parser.add_argument("--parts", help="comma-separated: body,hands,face")
    parser.add_argument("--mode", help="KEYFRAMES or POSE (default: by media)")
    parser.add_argument("--frame", type=int, help="0-based video frame for POSE")
    parser.add_argument(
        "--frame-range",
        type=_frame_range,
        help="source frames FIRST:END (end excluded)",
    )
    parser.add_argument("--start-frame", type=float, help="timeline frame of frame 0")
    parser.add_argument("--root-motion", help="AUTOMATIC, IN_PLACE or OFF")
    parser.add_argument("--face-output", help="ARKIT, FACS or BONES")
    parser.add_argument(
        "--mirror", action=argparse.BooleanOptionalAction, help="mirror the media"
    )
    parser.add_argument("--config", help="clean-up settings JSON for this run")
    parser.add_argument(
        "--output", default="NEW_ACTION", help="NEW_ACTION, REPLACE or NLA_STRIP"
    )
    if not cached:
        parser.add_argument(
            "--no-cache", action="store_true", help="do not keep the detections"
        )
    parser.add_argument("--timeout", type=float, help="seconds before cancelling")
    parser.add_argument("--out", help="write the result JSON here")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument(
        "--save", action="store_true", help="save the .blend file afterwards"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog=f"blender -b FILE -c {COMMAND}", description=__doc__)
    commands = parser.add_subparsers(
        dest="command", required=True, parser_class=_Parser
    )

    for name, help_text, cached in (
        ("capture", "detect and retarget media onto an armature", False),
        ("retarget", "re-run retargeting from retained detections", True),
    ):
        sub = commands.add_parser(name, help=help_text)
        _capture_options(sub, cached=cached)
        _common(sub)

    sub = commands.add_parser("auto-map", help="map the rig automatically")
    sub.add_argument("--armature", required=True)
    sub.add_argument(
        "--force-profile", action="store_true", help="re-detect the rig profile"
    )
    sub.add_argument("--out")
    _common(sub)

    sub = commands.add_parser("describe-rig", help="mapping coverage and profile")
    sub.add_argument("--armature", required=True)
    sub.add_argument("--out")
    _common(sub)

    sub = commands.add_parser("evaluate", help="measure the animation quality")
    sub.add_argument("--armature", required=True)
    sub.add_argument("--action")
    sub.add_argument("--frame-range", type=_frame_range, help="timeline FIRST:LAST")
    sub.add_argument("--result", help="capture result JSON for landmark fit")
    sub.add_argument("--ground-height", type=float, default=0.0)
    sub.add_argument("--out")
    _common(sub)

    sub = commands.add_parser("config", help="read or change a rig's configuration")
    sub.add_argument("action", choices=("get", "set"))
    sub.add_argument("file", nargs="?", help="JSON to apply (set)")
    sub.add_argument("--armature", required=True)
    sub.add_argument("--out")
    _common(sub)

    sub = commands.add_parser("postprocessors", help="clean-up modules and settings")
    sub.add_argument("--armature")
    _common(sub)

    for name, help_text in (
        ("providers", "capture providers and availability"),
        ("models", "optional model install state"),
    ):
        _common(commands.add_parser(name, help=help_text))

    for name, help_text in (
        ("export-detections", "write retained detections as JSON and npz"),
        ("import-detections", "store edited detections as retained"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("path")
        _common(sub)

    sub = commands.add_parser("render", help="contact sheet of source and rig")
    sub.add_argument("--armature", required=True)
    sub.add_argument("--result", required=True, help="capture result JSON")
    sub.add_argument("--out", required=True, help="PNG path")
    sub.add_argument("--frames", type=_frames, help="timeline frames, comma-separated")
    sub.add_argument("--count", type=int, default=6)
    sub.add_argument("--size", type=int, default=384)
    sub.add_argument("--no-meshes", action="store_true", help="skip mesh rendering")
    _common(sub)
    return parser


def _capture(args, *, cached: bool) -> dict[str, Any]:
    options = {
        "provider": args.provider,
        "parts": args.parts,
        "mode": args.mode,
        "frame": args.frame,
        "frame_range": args.frame_range,
        "start_frame": args.start_frame,
        "root_motion": args.root_motion,
        "face_output": args.face_output,
        "mirror": args.mirror,
        "config": _read_json(args.config) if args.config else None,
        "output": args.output,
        "timeout": args.timeout,
        "result_path": args.out,
    }
    if cached:
        result = retarget_cached(args.media, args.armature, **options)
    else:
        if args.no_cache:
            options["keep_cache"] = False
        result = capture(args.media, args.armature, **options)
    return result.to_dict()


def _write(path: str | None, payload: Any) -> None:
    if path:
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf8")


def run(args) -> Any:
    command = args.command
    if command in {"capture", "retarget"}:
        return _capture(args, cached=command == "retarget")
    if command == "describe-rig":
        payload = describe_rig(args.armature)
    elif command == "auto-map":
        payload = auto_map(args.armature, force_profile=args.force_profile)
    elif command == "evaluate":
        payload = evaluate(
            args.armature,
            action=args.action,
            frame_range=args.frame_range,
            result=args.result,
            ground_height=args.ground_height,
        ).to_dict()
    elif command == "config":
        if args.action == "set":
            if not args.file:
                raise InvalidRequest("config set needs a JSON file")
            payload = set_config(args.armature, _read_json(args.file))
        else:
            payload = get_config(args.armature)
    elif command == "postprocessors":
        payload = list_postprocessors(args.armature)
    elif command == "providers":
        payload = list_providers()
    elif command == "models":
        payload = model_status()
    elif command == "export-detections":
        payload = export_detections(args.path)
    elif command == "import-detections":
        payload = import_detections(args.path)
    elif command == "render":
        payload = render_comparison(
            args.out,
            args.armature,
            result=args.result,
            frames=args.frames,
            count=args.count,
            size=args.size,
            render_meshes=not args.no_meshes,
        )
    else:  # argparse rejects unknown commands first
        raise InvalidRequest(f"Unknown command {command}")
    _write(getattr(args, "out", None), payload)
    return payload


def _summary(command: str, payload: Any) -> str:
    if command in {"capture", "retarget"}:
        return (
            f"{payload['status']}: {payload['message']}\n"
            f"action {payload['action'] or '-'}, {payload['frames_processed']} frame(s)"
        )
    if command == "evaluate":
        return "\n".join(
            f"{key}: {value}"
            for key, value in payload.items()
            if key not in {"worst_frames", "schema_version"}
        )
    return json.dumps(payload, indent=2)


def execute(argv: Sequence[str]) -> int:
    """Entry point for ``bpy.utils.register_cli_command``."""
    json_output = "--json" in argv
    try:
        args = build_parser().parse_args(list(argv))
        payload = run(args)
        if args.save:
            if not bpy.data.filepath:
                raise InvalidRequest("--save needs a .blend file opened with Blender")
            bpy.ops.wm.save_mainfile()
    except _ParserExitError as exit_request:
        return int(exit_request.status)
    except MtpError as exc:
        if json_output:
            sys.stdout.write(json.dumps({"error": exc.to_dict()}) + "\n")
        else:
            sys.stderr.write(f"Error [{exc.code}]: {exc}\n")
        return int(exc.exit_code)
    except Exception as exc:  # The command boundary: report, never crash Blender.
        LOGGER.exception("media_to_pose command failed")
        if json_output:
            error = {"code": "INTERNAL_ERROR", "message": str(exc)}
            sys.stdout.write(json.dumps({"error": error}) + "\n")
        return 1
    if json_output:
        sys.stdout.write(json.dumps(payload) + "\n")
    else:
        sys.stdout.write(_summary(args.command, payload) + "\n")
    return 0


__all__ = ["COMMAND", "build_parser", "execute", "run"]
