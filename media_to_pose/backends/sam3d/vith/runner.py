"""One body-and-hands inference pass over the split graphs."""

from __future__ import annotations

import gc
from collections.abc import Mapping
from pathlib import PurePosixPath

import numpy as np

from ..mhr_runtime import MhrModel
from .common import VithRuntimeError
from .crops import (
    CropBatch,
    backbone_input,
    decoder_condition,
    full_to_crop,
    hand_boxes_from_tokens,
    prepare_crop_batch,
    project_camera,
    ray_condition,
)
from .graphs import GraphSessions
from .mhr_params import (
    _fused_local_wrist,
    _mhr_model_parameters,
    decode_raw_mhr,
    previous_estimate,
)
from .rotations import (
    euler_xzy_to_matrix,
    fix_wrist_euler,
    matrix_to_euler_xzy,
    rotation_angle_difference,
)


class VithSplitRunner:
    """Exact six-layer body/hand orchestration over persistent ORT sessions."""

    def __init__(self, bundle, sessions: Mapping[str, object]) -> None:
        self.bundle = bundle
        self.graphs = GraphSessions(bundle, sessions)
        artifacts = {
            PurePosixPath(item.path).name: bundle.root.joinpath(
                *PurePosixPath(item.path).parts
            )
            for item in bundle.artifacts
        }
        required = (
            "body_model.lbs",
            "correctives.bin",
            "keypoint_mapping.bin",
            "hand_runtime.bin",
        )
        missing = [name for name in required if name not in artifacts]
        if missing:
            raise VithRuntimeError(
                "Split bundle is missing MHR artifacts: " + ", ".join(missing)
            )
        self.mhr = MhrModel.load(*(artifacts[name] for name in required))

    def _branch(
        self,
        branch: str,
        features: np.ndarray,
        batch: CropBatch,
        prompt_points: np.ndarray,
        prompt_labels: np.ndarray,
        prev_estimate: np.ndarray,
        *,
        use_initial_estimate: bool,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        people = len(features)
        initialized = self.graphs.run(
            f"token_initializer.{branch}",
            {
                "features": np.ascontiguousarray(features, dtype=np.float32),
                "condition_info": decoder_condition(batch),
                "ray_cond": ray_condition(batch),
                "prompt_points": np.ascontiguousarray(prompt_points, dtype=np.float32),
                "prompt_labels": np.ascontiguousarray(prompt_labels, dtype=np.int64),
                "prev_estimate": np.ascontiguousarray(prev_estimate, dtype=np.float32),
                "use_initial_estimate": np.full(
                    people, int(use_initial_estimate), dtype=np.int64
                ),
            },
        )
        tokens = np.ascontiguousarray(initialized["tokens"], dtype=np.float32)
        token_pe = np.ascontiguousarray(initialized["token_pe"], dtype=np.float32)
        image_features = np.ascontiguousarray(
            initialized["image_features"], dtype=np.float32
        )
        image_pe_features = np.ascontiguousarray(
            initialized["image_pe"], dtype=np.float32
        )
        image_tokens = np.ascontiguousarray(
            image_features.transpose(0, 2, 3, 1).reshape(people, 32 * 24, 1280)
        )
        image_pe = np.ascontiguousarray(
            image_pe_features.transpose(0, 2, 3, 1).reshape(people, 32 * 24, 1280)
        )
        output = None
        normalized = None
        for index in range(6):
            tokens = self.graphs.run(
                f"decoder.{branch}.layer.{index}",
                {
                    "tokens": tokens,
                    "image_tokens": image_tokens,
                    "token_pe": token_pe,
                    "image_pe": image_pe,
                },
            )["tokens_out"]
            normalized = self.graphs.run(
                f"decoder.{branch}.final_norm", {"tokens": tokens}
            )["normalized_tokens"]
            head = self.graphs.run(
                f"raw_head.{branch}", {"pose_token": normalized[:, 0]}
            )
            output = decode_raw_mhr(
                self.mhr,
                head["raw_mhr_parameters"],
                head["raw_camera"],
                hand_model=branch == "hand",
            )
            project_camera(
                output,
                batch,
                default_scale_factor=10.0 if branch == "hand" else 1.0,
            )
            if index < 5:
                updated = self.graphs.run(
                    f"keypoint_updater.{branch}",
                    {
                        "tokens": tokens,
                        "token_pe": token_pe,
                        "image_features": image_features,
                        "keypoints_2d_crop": output["pred_keypoints_2d_cropped"][
                            :, :70
                        ],
                        "keypoint_depth": output["pred_keypoints_2d_depth"][:, :70],
                        "keypoints_3d": output["pred_keypoints_3d"][:, :70],
                    },
                )
                tokens = updated["updated_tokens"]
                token_pe = updated["updated_token_pe"]
        if output is None or normalized is None:
            raise AssertionError("six-layer decoder produced no output")
        return output, normalized

    def _backbone(self, batch: CropBatch) -> np.ndarray:
        return np.asarray(
            self.graphs.run("backbone", {"image": backbone_input(batch)})["features"],
            dtype=np.float32,
        )

    def _initial_branch(
        self, branch: str, batch: CropBatch
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        features = self._backbone(batch)
        people = len(features)
        points = np.zeros((people, 1, 2), dtype=np.float32)
        labels = np.full((people, 1), -2, dtype=np.int64)
        prev = np.zeros((people, 522), dtype=np.float32)
        output, normalized = self._branch(
            branch,
            features,
            batch,
            points,
            labels,
            prev,
            use_initial_estimate=True,
        )
        return output, normalized, features

    def _body_only(self, body_batch: CropBatch) -> dict[str, np.ndarray]:
        body, _, _ = self._initial_branch("body", body_batch)
        return {
            "pred_keypoints_3d": np.ascontiguousarray(body["pred_keypoints_3d"]),
            "pred_keypoints_2d": np.ascontiguousarray(body["pred_keypoints_2d"]),
            "pred_cam_t": np.ascontiguousarray(body["pred_cam_t"]),
            "focal_length": np.ascontiguousarray(body["focal_length"]),
            "global_rot": np.ascontiguousarray(body["global_rot"]),
            "scale_params": np.ascontiguousarray(body["scale"]),
            "pred_joint_coords": np.ascontiguousarray(body["pred_joint_coords"]),
            "pred_global_rots": np.ascontiguousarray(body["joint_global_rots"]),
        }

    def _full_hand(
        self, image_rgb: np.ndarray, body_batch: CropBatch
    ) -> dict[str, np.ndarray]:
        people = len(body_batch.images)
        height, width = image_rgb.shape[:2]
        body, body_tokens, body_features = self._initial_branch("body", body_batch)
        # The initial pass has one prompt.  Pose/previous/prompt occupy tokens
        # 0..2, followed by the two hand detector tokens.
        hand_box = self.graphs.run(
            "hand_box_head", {"hand_tokens": body_tokens[:, 3:5]}
        )["hand_boxes"]
        left_xyxy, right_xyxy = hand_boxes_from_tokens(hand_box, body_batch)
        wrist_pose_indices = np.asarray((41, 43, 42, 31, 33, 32), dtype=np.intp)
        original_local_wrist = euler_xzy_to_matrix(
            body["body_pose"][:, wrist_pose_indices].reshape(people, 2, 3)
        )

        flipped = np.ascontiguousarray(image_rgb[:, ::-1])
        flipped_left = left_xyxy.copy()
        flipped_left[:, 0] = width - left_xyxy[:, 2] - 1
        flipped_left[:, 2] = width - left_xyxy[:, 0] - 1
        right_intrinsic = body_batch.camera_intrinsics[0]
        left_intrinsic = right_intrinsic.copy()
        left_intrinsic[0, 2] = np.float32(width - 1) - left_intrinsic[0, 2]
        left_batch = prepare_crop_batch(
            flipped,
            flipped_left,
            padding=0.9,
            camera_intrinsics=left_intrinsic,
        )
        right_batch = prepare_crop_batch(
            image_rgb,
            right_xyxy,
            padding=0.9,
            camera_intrinsics=right_intrinsic,
        )
        left, _, _ = self._initial_branch("hand", left_batch)
        right, _, _ = self._initial_branch("hand", right_batch)

        mean_right = self.mhr.scale_mean[8]
        mean_left = self.mhr.scale_mean[9]
        component_right = self.mhr.scale_components[8, 8]
        component_left = self.mhr.scale_components[9, 9]
        left["scale"][:, 9] = (
            mean_right + component_right * left["scale"][:, 8] - mean_left
        ) / component_left
        left["joint_global_rots"][:, 78] = left["joint_global_rots"][:, 42]
        left["joint_global_rots"][:, 78, (1, 2), :] *= np.float32(-1.0)
        left["hand"][:, :54] = left["hand"][:, 54:]
        # Match the upstream batch mutation after hand decoding.  The already
        # computed crop projection remains in flipped-image coordinates.
        left_batch.bbox_center[:, 0] = width - left_batch.bbox_center[:, 0] - 1

        lowarm = body["joint_global_rots"][:, (76, 40)]
        wrist_zero = lowarm @ self.mhr.joint_rotation[[77, 41]]
        predicted_wrist = np.stack(
            (left["joint_global_rots"][:, 78], right["joint_global_rots"][:, 42]),
            axis=1,
        )
        fused_local = _fused_local_wrist(predicted_wrist, wrist_zero)
        valid_angle_initial = (
            rotation_angle_difference(original_local_wrist, fused_local) < 1.4
        )
        valid_box = np.stack(
            (
                np.all(left_batch.bbox_scale > 64.0, axis=1),
                np.all(right_batch.bbox_scale > 64.0, axis=1),
            ),
            axis=1,
        )
        valid_inside = np.stack(
            (
                np.max(np.abs(left["pred_keypoints_2d_cropped"]), axis=(1, 2)) < 0.5,
                np.max(np.abs(right["pred_keypoints_2d_cropped"]), axis=(1, 2)) < 0.5,
            ),
            axis=1,
        )
        right_wrist = right["pred_keypoints_2d"][:, [41]].copy()
        left_wrist = left["pred_keypoints_2d"][:, [41]].copy()
        left_wrist[:, :, 0] = width - left_wrist[:, :, 0] - 1
        body_right_wrist = body["pred_keypoints_2d"][:, [41]]
        body_left_wrist = body["pred_keypoints_2d"][:, [62]]
        # Preserve the pinned implementation's intentionally crossed scale
        # denominators and left/right stack order.
        right_distance = (
            np.linalg.norm((right_wrist - body_right_wrist).reshape(people, 2), axis=1)
            / left_batch.bbox_scale[:, 0]
        )
        left_distance = (
            np.linalg.norm((left_wrist - body_left_wrist).reshape(people, 2), axis=1)
            / right_batch.bbox_scale[:, 0]
        )
        valid_wrist_distance = np.stack(
            (left_distance < 0.25, right_distance < 0.25), axis=1
        )
        hand_valid = (
            valid_angle_initial & valid_box & valid_inside & valid_wrist_distance
        )

        right_wrist_crop = full_to_crop(body_batch, right_wrist)
        left_wrist_crop = full_to_crop(body_batch, left_wrist)
        right_elbow_crop = full_to_crop(body_batch, body["pred_keypoints_2d"][:, [8]])
        left_elbow_crop = full_to_crop(body_batch, body["pred_keypoints_2d"][:, [7]])
        prompt_points = np.concatenate(
            (right_wrist_crop, left_wrist_crop, right_elbow_crop, left_elbow_crop),
            axis=1,
        )
        prompt_labels = np.tile(np.asarray((41, 62, 8, 7), dtype=np.int64), (people, 1))
        prompt_hand_valid = hand_valid[:, (1, 0, 1, 0)]
        if people > 1:
            invalid = (
                (prompt_points[..., 0] < -0.5)
                | (prompt_points[..., 0] > 0.5)
                | (prompt_points[..., 1] < -0.5)
                | (prompt_points[..., 1] > 0.5)
                | ~prompt_hand_valid
            )
            prompt_points = np.clip(prompt_points + 0.5, 0.0, 1.0)
            prompt_points[invalid] = 0.0
            prompt_labels[invalid] = -2
        else:
            valid = (
                np.all((prompt_points[0] > -0.5) & (prompt_points[0] < 0.5), axis=1)
                & prompt_hand_valid[0]
            )
            prompt_points = np.clip(prompt_points[:, valid] + 0.5, 0.0, 1.0)
            prompt_labels = prompt_labels[:, valid]
        if prompt_points.shape[1]:
            body, _ = self._branch(
                "body",
                body_features,
                body_batch,
                prompt_points,
                prompt_labels,
                previous_estimate(body),
                use_initial_estimate=False,
            )

        updated_hand = np.concatenate(
            (left["hand"][:, :54], right["hand"][:, 54:]), axis=1
        )
        updated_scale = body["scale"].copy()
        updated_scale[:, 9] = left["scale"][:, 9]
        updated_scale[:, 8] = right["scale"][:, 8]
        updated_scale[:, 18:] = (left["scale"][:, 18:] + right["scale"][:, 18:]) * 0.5
        updated_shape = body["shape"].copy()
        updated_shape[:, 40:] = (left["shape"][:, 40:] + right["shape"][:, 40:]) * 0.5

        fk_parameters, _ = _mhr_model_parameters(
            self.mhr,
            global_rotation=body["global_rot"],
            body_pose=body["body_pose"],
            hand_pose=updated_hand,
            scale=updated_scale,
            hand_model=False,
        )
        fk_rotations = self.mhr.forward(
            updated_shape, fk_parameters, body["face"]
        ).joint_rotations
        lowarm = fk_rotations[:, (76, 40)]
        wrist_zero = lowarm @ self.mhr.joint_rotation[[77, 41]]
        fused_local = _fused_local_wrist(predicted_wrist, wrist_zero)
        wrist_xzy = fix_wrist_euler(matrix_to_euler_xzy(fused_local))
        valid_angle = (
            rotation_angle_difference(original_local_wrist, fused_local) < 1.4
        ) & hand_valid

        original_body_wrist = body["body_pose"][:, wrist_pose_indices].reshape(
            people, 2, 3
        )
        body["body_pose"][:, wrist_pose_indices] = np.where(
            valid_angle[..., None], wrist_xzy, original_body_wrist
        ).reshape(people, 6)
        body_hand = body["hand"].reshape(people, 2, 54)
        body["hand"] = np.where(
            valid_angle[..., None], updated_hand.reshape(people, 2, 54), body_hand
        ).reshape(people, 108)
        original_hand_scale = np.stack(
            (body["scale"][:, 9], body["scale"][:, 8]), axis=1
        )
        replacement_hand_scale = np.stack(
            (updated_scale[:, 9], updated_scale[:, 8]), axis=1
        )
        selected_hand_scale = np.where(
            valid_angle, replacement_hand_scale, original_hand_scale
        )
        body["scale"][:, 9] = selected_hand_scale[:, 0]
        body["scale"][:, 8] = selected_hand_scale[:, 1]
        valid_count = valid_angle.sum(axis=1, keepdims=True).astype(np.float32)
        has_valid = valid_count > 0
        shared_scale = (
            left["scale"][:, 18:] * valid_angle[:, [0]]
            + right["scale"][:, 18:] * valid_angle[:, [1]]
        ) / (valid_count + np.float32(1e-8))
        shared_shape = (
            left["shape"][:, 40:] * valid_angle[:, [0]]
            + right["shape"][:, 40:] * valid_angle[:, [1]]
        ) / (valid_count + np.float32(1e-8))
        body["scale"][:, 18:] = np.where(has_valid, shared_scale, body["scale"][:, 18:])
        body["shape"][:, 40:] = np.where(has_valid, shared_shape, body["shape"][:, 40:])

        final_parameters, _ = _mhr_model_parameters(
            self.mhr,
            global_rotation=body["global_rot"],
            body_pose=body["body_pose"],
            hand_pose=body["hand"],
            scale=body["scale"],
            hand_model=False,
        )
        final_geometry = self.mhr.forward(
            body["shape"], final_parameters, body["face"]
        ).camera_space()
        body["pred_keypoints_3d"] = final_geometry.keypoints
        body["pred_joint_coords"] = final_geometry.joint_coords
        # Preserve the pinned upstream result contract.  The hand-refinement
        # path recomputes geometry after wrist/hand fusion, but it never writes
        # the recomputed joint rotations back to ``pose_output``.  Therefore
        # ``pred_global_rots`` intentionally remains the decoder result from
        # the prompted body pass (or initial pass when there are no prompts).
        body["mhr_model_params"] = final_parameters
        camera_points = body["pred_keypoints_3d"] + body["pred_cam_t"][:, None, :]
        normalized = camera_points / camera_points[..., 2:3]
        body["pred_keypoints_2d"] = np.einsum(
            "bij,bkj->bki",
            body_batch.camera_intrinsics,
            normalized,
            optimize=True,
        )[..., :2]
        return {
            "pred_keypoints_3d": np.ascontiguousarray(body["pred_keypoints_3d"]),
            "pred_keypoints_2d": np.ascontiguousarray(body["pred_keypoints_2d"]),
            "pred_cam_t": np.ascontiguousarray(body["pred_cam_t"]),
            "focal_length": np.ascontiguousarray(body["focal_length"]),
            "global_rot": np.ascontiguousarray(body["global_rot"]),
            "scale_params": np.ascontiguousarray(body["scale"]),
            "pred_joint_coords": np.ascontiguousarray(body["pred_joint_coords"]),
            "pred_global_rots": np.ascontiguousarray(body["joint_global_rots"]),
        }

    def run(
        self,
        image_rgb: object,
        bboxes_xyxy: object,
        *,
        include_hands: bool,
        camera_intrinsics: object | None = None,
    ) -> dict[str, np.ndarray]:
        self.graphs.executed_roles.clear()
        image = np.asarray(image_rgb)
        boxes = np.asarray(bboxes_xyxy, dtype=np.float32).reshape(-1, 4)
        if not len(boxes):
            return {
                "pred_keypoints_3d": np.empty((0, 70, 3), dtype=np.float32),
                "pred_keypoints_2d": np.empty((0, 70, 2), dtype=np.float32),
                "pred_cam_t": np.empty((0, 3), dtype=np.float32),
                "focal_length": np.empty((0,), dtype=np.float32),
                "global_rot": np.empty((0, 3), dtype=np.float32),
                "scale_params": np.empty((0, 28), dtype=np.float32),
                "pred_joint_coords": np.empty((0, 127, 3), dtype=np.float32),
                "pred_global_rots": np.empty((0, 127, 3, 3), dtype=np.float32),
            }
        body_batch = prepare_crop_batch(
            image,
            boxes,
            padding=1.25,
            camera_intrinsics=camera_intrinsics,
        )
        return (
            self._full_hand(image, body_batch)
            if include_hands
            else self._body_only(body_batch)
        )

    def close(self) -> None:
        mhr = getattr(self, "mhr", None)
        if mhr is not None:
            mhr.close()
            self.mhr = None
        self.graphs.sessions.clear()
        gc.collect()
