"""Portable Franka hand action contract, shared with the installed DROID adapter.

This file deliberately uses only Python 3.8's standard library. Install an exact
copy beside droid/franka/robot.py; no simulation or robot connection is imported.
Widths are total finger gap in metres unless explicitly named per_finger.
"""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GripperCommand:
    # Preserve DROID's negative API grasp width for filtering/result semantics.
    width: float
    grasp: bool
    speed: float
    force: float
    epsilon_inner: float = -1.0
    epsilon_outer: float = -1.0
    stop: bool = False

    def __post_init__(self):
        if not all(
            math.isfinite(x)
            for x in (
                self.width,
                self.speed,
                self.force,
                self.epsilon_inner,
                self.epsilon_outer,
            )
        ):
            raise ValueError("Gripper command must be finite")
        if self.stop:
            if self.grasp:
                raise ValueError("A gripper command cannot both stop and grasp")
        elif (
            self.speed <= 0
            or self.force < 0
            or self.width > 0.08
            or (not self.grasp and self.width < 0)
        ):
            raise ValueError("Invalid gripper width, speed or force")


@dataclass(frozen=True)
class FrankaHandContract:
    version: str = "franka_continuous_aperture_v1"
    max_width: float = 0.08
    grasp_threshold: float = 0.95
    speed: float = 0.2
    grasp_force: float = 5.0

    def __post_init__(self):
        if (
            self.max_width != 0.08
            or not 0 <= self.grasp_threshold < 1
            or not math.isfinite(self.speed)
            or self.speed <= 0
            or not math.isfinite(self.grasp_force)
            or self.grasp_force < 0
        ):
            raise ValueError("Invalid Franka hand contract")

    @staticmethod
    def _finite_clip(value, low, high):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Gripper aperture must be finite")
        return max(low, min(high, value))

    def width_from_closure(self, closure):
        return self.max_width * (1 - self._finite_clip(closure, 0, 1))

    def closure_from_width(self, width):
        return 1 - self._finite_clip(width, 0, self.max_width) / self.max_width

    def closure_from_per_finger(self, per_finger):
        return self.closure_from_width(2 * float(per_finger))

    def decode(self, closure):
        closure = self._finite_clip(closure, 0, 1)
        if closure > self.grasp_threshold:
            return GripperCommand(-0.2, True, self.speed, self.grasp_force, 0.1, 0.9)
        return GripperCommand(self.width_from_closure(closure), False, self.speed, 1.0)

    def metadata(self):
        return dict(
            asdict(self),
            policy_units="normalized_closure",
            training_units="per_finger_metres",
            grasp_comparison=">",
        )


DEFAULT_HAND_CONTRACT = FrankaHandContract()


def send_aperture(
    gripper, closure, was_grasp=False, contract=DEFAULT_HAND_CONTRACT, blocking=False
):
    """Send the same stop/goto transition as DROID and return requested mode.

    API acknowledgment is not motion completion. At the raw driver boundary,
    a stop sent while busy can still be replaced by the following goto; this
    helper intentionally makes no stronger release/completion guarantee.
    """
    command = contract.decode(closure)
    if command.grasp:
        gripper.grasp(
            speed=command.speed,
            force=command.force,
            grasp_width=command.width,
            epsilon_inner=command.epsilon_inner,
            epsilon_outer=command.epsilon_outer,
            blocking=blocking,
        )
    else:
        if was_grasp:
            gripper.stop(blocking=blocking)
        gripper.goto(
            width=command.width,
            speed=command.speed,
            force=command.force,
            blocking=blocking,
        )
    return command.grasp
