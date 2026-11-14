"""Exercise the simulator through the unmodified deployment ZMQ clients."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import zmq

import abc_sim
import deploy.robot.communication as comms
from abc_sim.config import get_i2rt_config
from abc_sim.zmq_sim import ZMQSimulator
from deploy.robot.gym.yam_env import YAMEnv


def write_scene(path: Path) -> None:
    """A mesh-free two-arm scene, so tests need no downloaded assets."""
    bodies, actuators = [], []
    for side, y in (("left", 0.2), ("right", -0.2)):
        for joint in range(1, 7):
            bodies.append(
                f'<body pos="0.05 {y if joint == 1 else 0} 0">'
                f'<joint name="{side}_joint{joint}" axis="0 0 1"/>'
                '<geom type="capsule" size="0.01" fromto="0 0 0 0.05 0 0"/>'
            )
            actuators.append(
                f'<position name="{side}_joint{joint}" joint="{side}_joint{joint}"/>'
            )
        bodies.append(
            '<body pos="0.05 0 0">'
            f'<joint name="{side}_left_finger" type="slide" axis="1 0 0"/>'
            '<geom type="sphere" size="0.01"/></body>' + "</body>" * 6
        )
        actuators.append(
            f'<position name="{side}_gripper" joint="{side}_left_finger"/>'
        )
    path.write_text(
        '<mujoco><option gravity="0 0 0" integrator="implicitfast"/>'
        '<default><joint damping="1" armature="0.02"/>'
        '<geom contype="0" conaffinity="0" rgba="0.2 0.8 0.3 1"/>'
        '<position kp="30" kv="2"/></default><worldbody>'
        '<light pos="0 0 2"/><body pos="0 0 0.5"><freejoint/>'
        '<geom type="sphere" size="0.02"/></body>'
        '<camera name="top" pos="0.2 0 1"/>'
        '<camera name="left" pos="0.2 0.1 1"/>'
        '<camera name="right" pos="0.2 -0.1 1"/>'
        + "".join(bodies)
        + "</worldbody><actuator>"
        + "".join(actuators)
        + "</actuator></mujoco>"
    )


class LockstepTests(unittest.TestCase):
    def test_target_updates_do_not_step_and_gym_paths_keep_their_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "scene.xml"
            write_scene(scene)
            env = abc_sim.make_env(
                task="empty", scene_xml=scene, render_cameras=False, chunk_dim=3
            )
            self.addCleanup(env.close)
            obs, _ = env.reset(seed=0)
            env.set_joint_targets(obs["state"])
            self.assertEqual(env.data.time, 0.0)
            period = env.model.opt.timestep * env._control_decimation
            result = env.step(obs["state"])
            self.assertEqual(len(result), 5)
            self.assertAlmostEqual(env.data.time, period)
            final, history, _, _, _, _ = env.step_chunk(np.tile(obs["state"], (3, 1)))
            self.assertEqual(final["state"].shape, (14,))
            self.assertEqual(history["state"].shape, (3, 14))
            self.assertAlmostEqual(env.data.time, period * 4)

    def test_observations_use_dof_addresses_and_gripper_units(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "scene.xml"
            write_scene(scene)
            env = abc_sim.make_env(task="empty", scene_xml=scene, render_cameras=False)
            self.addCleanup(env.close)
            env.reset(seed=0)
            sim = ZMQSimulator(env, cameras=False)
            env.data.qvel[sim._dofs] = np.arange(1, 15)
            env.data.qfrc_actuator[sim._dofs] = np.arange(15, 29)
            _, velocity, effort = sim._joint_observations()
            np.testing.assert_array_equal(velocity[:6], np.arange(1, 7))
            np.testing.assert_array_equal(effort[:6], np.arange(15, 21))
            self.assertAlmostEqual(velocity[6], 7 / 0.0475)
            self.assertAlmostEqual(effort[6], 21 * 0.0475 / -6.57)
            with self.assertRaises(ValueError):
                ZMQSimulator(env, state_rate=0)


class ZMQClientTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="abc-zmq-test-", dir="/tmp")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.scene = self.root / "scene.xml"
        write_scene(self.scene)
        ipc = str(self.root / "ipc")
        self.ipc_patch = patch.object(comms, "_IPC_DIR", ipc)
        self.ipc_patch.start()
        self.addCleanup(self.ipc_patch.stop)
        self.client = YAMEnv(get_i2rt_config(), action_resync_max_retries=0)
        self.addCleanup(self.client.close)
        self.context = zmq.Context()
        self.addCleanup(self.context.term)
        self.obs_sockets = {}
        for side in ("left", "right"):
            socket = comms.create_subscriber(
                self.context, f"follower_{side}_obs", conflate=1
            )
            self.obs_sockets[side] = socket
            self.addCleanup(socket.close, 0)
        self.log = tempfile.TemporaryFile(mode="w+t")  # noqa: SIM115 -- owned by addCleanup.
        self.addCleanup(self.log.close)

    def start_simulator(self, *, cameras=False):
        # Isolate IPC paths without changing the real client's topic names.
        bootstrap = (
            "import sys; import deploy.robot.communication as comms; "
            "comms._IPC_DIR = sys.argv.pop(1); "
            "from abc_sim.zmq_sim import main; main()"
        )
        args = [
            sys.executable,
            "-c",
            bootstrap,
            comms._IPC_DIR,
            "--scene-xml",
            str(self.scene),
            "--camera-width",
            "64",
            "--camera-height",
            "48",
        ]
        if not cameras:
            args.append("--no-cameras")
        self.process = subprocess.Popen(
            args,
            cwd=Path(__file__).resolve().parents[1],
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        self.addCleanup(self.stop_simulator)
        try:
            for socket in self.client.robot_state_sockets.values():
                comms.subscribe(socket, timeout_ms=10_000)
        except comms.SubscribeTimeout:
            self.log.seek(0)
            self.fail(self.log.read())

    def stop_simulator(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                self.fail("simulator did not stop cleanly")
        if self.process.returncode != 0:
            self.log.seek(0)
            self.fail(self.log.read())

    def wait_command(self, side, target, *, republish=None):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if republish is not None:
                self.client._publish_action(republish)
            obs, extras = comms.subscribe(self.obs_sockets[side], timeout_ms=1000)
            self.assertEqual(obs.shape, (28,))
            self.assertTrue(np.isfinite(obs).all())
            if np.allclose(obs[21:], target):
                return obs, extras
        self.fail(f"{side} did not receive the target")

    def test_existing_client_servo_interp_and_observations_without_commands(self):
        self.start_simulator()
        initial, before = comms.subscribe(self.obs_sockets["left"], timeout_ms=1000)
        _, after = comms.subscribe(self.obs_sockets["left"], timeout_ms=1000)
        self.assertGreater(after["sim_time"], before["sim_time"])
        self.assertGreater(after["timestamp"], before["timestamp"])
        action = self.client._get_state_obs()
        action[0] += 0.15
        action[7] -= 0.1
        action[6], action[13] = 0.35, 0.6
        state = self.client.step_state(action)
        self.assertEqual(state.shape, (14,))
        for index, side in enumerate(("left", "right")):
            target = action[index * 7 : (index + 1) * 7]
            self.wait_command(side, target, republish=action)
        time.sleep(0.4)
        state = self.client._get_state_obs()
        np.testing.assert_allclose(state, action, atol=0.03)

        for invalid in (np.zeros(6), np.full(7, np.nan), np.full(7, 1e300)):
            comms.publish(self.client.robot_action_sockets["left"], invalid)
            time.sleep(0.05)
        unchanged, _ = self.wait_command("left", action[:7])
        np.testing.assert_allclose(unchanged[21:], action[:7])

        target = action[:7].copy()
        target[0] += 0.4
        comms.publish(
            self.client.robot_action_sockets["left"], target, {"type": "interp"}
        )
        _, start = self.wait_command("left", target)
        time.sleep(0.6)
        midway, middle = comms.subscribe(self.obs_sockets["left"], timeout_ms=1000)
        self.assertGreater(middle["sim_time"], start["sim_time"] + 0.3)
        self.assertGreater(midway[0], action[0] + 0.03)
        self.assertLess(midway[0], target[0] - 0.15)
        time.sleep(1.7)
        final, _ = comms.subscribe(self.obs_sockets["left"], timeout_ms=1000)
        np.testing.assert_allclose(final[:7], target, atol=0.03)
        np.testing.assert_allclose(final[21:], target)
        self.assertFalse(np.allclose(initial[:7], final[:7]))
        self.stop_simulator()
        self.log.seek(0)
        self.assertIn("invalid command", self.log.read())

    def test_existing_client_receives_rgb_and_can_step(self):
        self.start_simulator(cameras=True)
        image, _ = comms.subscribe(self.client.camera_sockets["top"], timeout_ms=10_000)
        self.assertEqual(image.shape, (48, 64, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(image.max(), 0)
        reset_obs, _ = self.client.reset()
        self.assertEqual(reset_obs["state"].shape, (14,))
        obs, reward, terminated, truncated, _ = self.client.step(
            self.client._get_state_obs()
        )
        self.assertEqual(obs["state"].shape, (14,))
        self.assertEqual(set(obs["images"]), {"top", "left", "right"})
        for frame in obs["images"].values():
            self.assertEqual(frame.shape, (3, 48, 64))
            self.assertEqual(frame.dtype, np.uint8)
        self.assertTrue(all(obs["masks"].values()))
        self.assertTrue(
            all(timestamp > 0 for timestamp in obs["camera_timestamps"].values())
        )
        self.assertEqual((reward, terminated, truncated), (0.0, False, False))


if __name__ == "__main__":
    unittest.main()
