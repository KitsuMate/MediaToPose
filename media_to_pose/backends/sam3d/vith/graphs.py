"""The bundle's ONNX graphs and their sessions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .common import GraphExecutionError, VithRuntimeError


def required_graph_roles() -> tuple[str, ...]:
    roles = ["backbone"]
    for branch in ("body", "hand"):
        roles.append(f"token_initializer.{branch}")
        roles.extend(f"decoder.{branch}.layer.{index}" for index in range(6))
        roles.append(f"decoder.{branch}.final_norm")
        roles.append(f"raw_head.{branch}")
        roles.append(f"keypoint_updater.{branch}")
    roles.append("hand_box_head")
    return tuple(roles)


class GraphSessions:
    """Logical-name adapter around the 22 manifest-declared ORT sessions."""

    def __init__(self, bundle, sessions: Mapping[str, object]) -> None:
        self.bundle = bundle
        self.sessions = dict(sessions)
        declared = {graph.role: graph for graph in bundle.graphs}
        missing = [role for role in required_graph_roles() if role not in declared]
        if missing:
            raise VithRuntimeError(
                "Split bundle is missing graph roles: " + ", ".join(missing)
            )
        extra_sessions = set(self.sessions) - set(declared)
        if extra_sessions:
            raise VithRuntimeError(
                "Sessions were supplied for undeclared roles: "
                + ", ".join(sorted(extra_sessions))
            )
        missing_sessions = set(required_graph_roles()) - set(self.sessions)
        if missing_sessions:
            raise VithRuntimeError(
                "Split graph sessions are missing roles: "
                + ", ".join(sorted(missing_sessions))
            )
        self.graphs = declared
        self.executed_roles: set[str] = set()

    def run(self, role: str, values: Mapping[str, object]) -> dict[str, np.ndarray]:
        graph = self.graphs[role]
        missing = set(graph.inputs) - set(values)
        if missing:
            raise VithRuntimeError(
                f"{role} is missing logical inputs: {', '.join(sorted(missing))}"
            )
        feed = {
            graph.inputs[name]: np.ascontiguousarray(values[name])
            for name in graph.inputs
        }
        logical_outputs = tuple(graph.outputs)
        output_names = [graph.outputs[name] for name in logical_outputs]
        try:
            result = self.sessions[role].run(output_names, feed)
        except Exception as exc:
            raise GraphExecutionError(f"Split graph {role!r} failed: {exc}") from exc
        self.executed_roles.add(role)
        if len(result) != len(logical_outputs):
            raise VithRuntimeError(
                f"Split graph {role!r} returned the wrong output count"
            )
        return {
            name: np.asarray(value)
            for name, value in zip(logical_outputs, result, strict=True)
        }
