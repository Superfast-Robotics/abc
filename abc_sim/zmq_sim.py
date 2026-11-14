"""Continuously running MuJoCo replacement for ABC follower and camera nodes."""

from __future__ import annotations

import argparse
import signal
import struct
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path

import mujoco
import numpy as np
import zmq

import abc_sim
import deploy.robot.communication as comms
from abc_sim.env import _GRIPPER_CTRL_MAX, MuJoCoYAMEnv
from abc_sim.rendering.replay.camera_providers import MujocoCameraProvider

# shortcut: nominal linear_4310 stroke; use the measured stroke for motor-torque comparisons.
_GRIPPER_MOTOR_STROKE = -6.57
_INTERP_SECONDS = 2.0


class ZMQSimulator:
    def __init__(
        self,
        env: MuJoCoYAMEnv,
        *,
        state_rate: float = 30.0,
        camera_rate: float = 30.0,
        cameras: bool = True,
        camera_height: int = 480,
        camera_width: int = 640,
    ):
        for name, value in (("state_rate", state_rate), ("camera_rate", camera_rate)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if camera_height < 1 or camera_width < 1:
            raise ValueError("camera dimensions must be positive")
        self.env = env
        self.state_rate = state_rate
        self.camera_rate = camera_rate
        self.cameras = cameras
        self.camera_height = camera_height
        self.camera_width = camera_width
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._error: Exception | None = None
        joint_ids = env.model.actuator_trnid[env._ctrl_indices, 0]
        self._dofs = env.model.jnt_dofadr[joint_ids]

    def stop(self) -> None:
        self._stop.set()

    def _joint_observations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = self.env.get_state().astype(np.float64)
        velocity = self.env.data.qvel[self._dofs].copy()
        effort = self.env.data.qfrc_actuator[self._dofs].copy()
        for index in self.env._gripper_indices:
            velocity[index] /= _GRIPPER_CTRL_MAX
            # i2rt reports normalized gripper velocity but raw motor torque.
            effort[index] *= _GRIPPER_CTRL_MAX / _GRIPPER_MOTOR_STROKE
        return state, velocity, effort

    def _physics_loop(self) -> None:
        try:
            with zmq.Context() as context, ExitStack() as sockets:
                subscribers, states, observations = [], [], []
                for name in self.env.robot_names:
                    subscribers.append(
                        sockets.enter_context(
                            comms.create_subscriber(
                                context, f"leader_{name}_actions", conflate=1
                            )
                        )
                    )
                    states.append(
                        sockets.enter_context(
                            comms.create_publisher(
                                context, f"follower_{name}_state", linger=0
                            )
                        )
                    )
                    observations.append(
                        sockets.enter_context(
                            comms.create_publisher(
                                context, f"follower_{name}_obs", linger=0
                            )
                        )
                    )

                action = self.env.get_state().astype(np.float64)
                command = action.copy()
                interpolations = [None] * len(subscribers)
                next_step = next_state = time.perf_counter()
                dt = self.env.model.opt.timestep
                while not self._stop.is_set():
                    if self._stop.wait(max(0.0, next_step - time.perf_counter())):
                        break
                    now = time.perf_counter()
                    for index, subscriber in enumerate(subscribers):
                        if not subscriber.poll(0):
                            continue
                        try:
                            target, extras = comms.subscribe(subscriber)
                            if (
                                target.shape != (7,)
                                or target.dtype.kind not in "fiu"
                                or not np.isfinite(target).all()
                                or (np.abs(target) > np.finfo(np.float32).max).any()
                            ):
                                raise ValueError("expected seven finite joint targets")
                            kind = extras.get("type", "servo")
                            if kind not in ("servo", "interp"):
                                raise ValueError(f"unknown command type: {kind!r}")
                        except (
                            ValueError,
                            TypeError,
                            KeyError,
                            struct.error,
                            UnicodeError,
                            AttributeError,
                            OverflowError,
                        ) as error:
                            print(
                                f"[{self.env.robot_names[index]}] invalid command: {error}",
                                file=sys.stderr,
                                flush=True,
                            )
                            continue
                        arm = slice(index * 7, (index + 1) * 7)
                        command[arm] = target
                        if kind == "interp":
                            interpolations[index] = (
                                self.env.get_state()[arm].copy(),
                                now,
                            )
                        else:
                            interpolations[index] = None
                            action[arm] = target

                    for index, interpolation in enumerate(interpolations):
                        if interpolation is None:
                            continue
                        start, started_at = interpolation
                        alpha = min((now - started_at) / _INTERP_SECONDS, 1.0)
                        arm = slice(index * 7, (index + 1) * 7)
                        action[arm] = (1.0 - alpha) * start + alpha * command[arm]
                        if alpha == 1.0:
                            interpolations[index] = None

                    with self._lock:
                        self.env.set_joint_targets(action)
                        self.env.advance_physics(steps=1)
                        if now >= next_state:
                            state, velocity, effort = self._joint_observations()
                            sim_time = float(self.env.data.time)
                    if now >= next_state:
                        for index, (state_pub, obs_pub) in enumerate(
                            zip(states, observations)
                        ):
                            arm = slice(index * 7, (index + 1) * 7)
                            extras = {"sim_time": sim_time}
                            comms.publish(state_pub, state[arm], extras)
                            comms.publish(
                                obs_pub,
                                np.concatenate(
                                    [
                                        state[arm],
                                        velocity[arm],
                                        effort[arm],
                                        command[arm],
                                    ]
                                ),
                                extras,
                            )
                        next_state += 1.0 / self.state_rate
                        if next_state < now:
                            next_state = now + 1.0 / self.state_rate
                    next_step += dt
                    # Like Unitree's loop, re-anchor after a long scheduler stall.
                    if time.perf_counter() - next_step > 0.1:
                        next_step = time.perf_counter()
        except Exception as error:  # noqa: BLE001 -- propagate worker failures to run().
            self._error = error
        finally:
            self.stop()

    def run(self) -> None:
        """Run until stop(), SIGINT, or an error; rendering stays on this thread."""
        worker = threading.Thread(target=self._physics_loop, name="abc-physics")
        try:
            with zmq.Context() as context, ExitStack() as resources:
                camera_publishers = {}
                if self.cameras:
                    render_data = mujoco.MjData(self.env.model)
                    provider = MujocoCameraProvider(
                        self.env.model,
                        render_data,
                        height=self.camera_height,
                        width=self.camera_width,
                        camera_names=tuple(self.env.camera_names),
                    )
                    resources.callback(provider.close)
                    if set(provider.camera_names) != set(self.env.camera_names):
                        raise ValueError("scene is missing configured cameras")
                    for name in self.env.camera_names:
                        camera_publishers[name] = resources.enter_context(
                            comms.create_publisher(
                                context, self.env.config.cameras[name].socket, linger=0
                            )
                        )
                worker.start()
                frame = 0
                while not self._stop.is_set():
                    started = time.perf_counter()
                    if self.cameras:
                        with self._lock:
                            mujoco.mj_copyData(
                                render_data, self.env.model, self.env.data
                            )
                        sim_time = float(render_data.time)
                        for name, image in provider.frames_for_step(
                            frame, sim_time
                        ).items():
                            comms.publish(
                                camera_publishers[name], image, {"sim_time": sim_time}
                            )
                        frame += 1
                    self._stop.wait(
                        max(
                            0.0,
                            1.0 / self.camera_rate - (time.perf_counter() - started),
                        )
                    )
                if self._error is not None:
                    raise RuntimeError("ABC physics/ZMQ loop failed") from self._error
        finally:
            self.stop()
            if worker.ident is not None:
                worker.join()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="empty")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scene-xml", type=Path)
    parser.add_argument("--state-rate", type=float, default=30.0)
    parser.add_argument("--camera-rate", type=float, default=30.0)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--no-cameras", action="store_true")
    args = parser.parse_args()
    env = abc_sim.make_env(
        task=args.task, scene_xml=args.scene_xml, render_cameras=False
    )
    previous_sigterm = None
    try:
        env.reset(seed=args.seed)
        simulator = ZMQSimulator(
            env,
            state_rate=args.state_rate,
            camera_rate=args.camera_rate,
            cameras=not args.no_cameras,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
        )
        previous_sigterm = signal.signal(signal.SIGTERM, lambda *_: simulator.stop())
        print(
            f"ABC ZMQ simulator: task={args.task}, arms={env.robot_names}, "
            f"physics_dt={env.model.opt.timestep:g}",
            flush=True,
        )
        simulator.run()
    except KeyboardInterrupt:
        pass
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        env.close()


if __name__ == "__main__":
    main()
