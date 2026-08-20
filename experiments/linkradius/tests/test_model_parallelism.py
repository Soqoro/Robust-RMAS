from dataclasses import asdict
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError:
    torch = None

from RecursiveMAS.inference_utils.linkradius_runtime import (
    LinkRadiusRuntime,
    RuntimeConfig,
)
from experiments.linkradius.grid import build_grid
from experiments.linkradius.run_linkradius import _build_grid_config, build_parser


REPO_ROOT = Path(__file__).resolve().parents[3]


class RoleDeviceConfigTests(unittest.TestCase):
    def test_role_devices_canonicalize_to_the_fallback_device(self) -> None:
        fallback = RuntimeConfig(device="cpu")
        explicit = RuntimeConfig(
            device="cpu",
            planner_device="cpu",
            critic_device="cpu",
            solver_device="cpu",
            terminal_solver_device="cpu",
        )
        self.assertEqual(fallback.resolved_role_devices(), {
            "planner": "cpu",
            "critic": "cpu",
            "solver": "cpu",
            "terminal_solver": "cpu",
        })
        self.assertEqual(asdict(fallback), asdict(explicit))

    def test_partial_role_override_uses_canonical_fallbacks(self) -> None:
        config = RuntimeConfig(
            device="cuda:9",
            planner_device="cuda:0",
            critic_device="cuda:1",
        )
        self.assertEqual(config.resolved_role_devices(), {
            "planner": "cuda:0",
            "critic": "cuda:1",
            "solver": "cuda:9",
            "terminal_solver": "cuda:9",
        })

    def test_relay_transfer_mode_is_canonical_and_validated(self) -> None:
        self.assertEqual(
            RuntimeConfig(device="cpu").relay_transfer_mode,
            "cpu_staged",
        )
        self.assertEqual(
            RuntimeConfig(
                device="cpu", relay_transfer_mode=" DIRECT "
            ).relay_transfer_mode,
            "direct",
        )
        with self.assertRaisesRegex(ValueError, "relay_transfer_mode"):
            RuntimeConfig(device="cpu", relay_transfer_mode="peer_magic")

    def test_autograd_memory_mode_is_canonical_and_validated(self) -> None:
        self.assertEqual(RuntimeConfig(device="cpu").autograd_memory_mode, "none")
        self.assertEqual(
            RuntimeConfig(
                device="cpu", autograd_memory_mode=" CHECKPOINT "
            ).autograd_memory_mode,
            "checkpoint",
        )
        with self.assertRaisesRegex(ValueError, "autograd_memory_mode"):
            RuntimeConfig(device="cpu", autograd_memory_mode="offload_magic")

    def test_edge_consumer_roles_are_explicit(self) -> None:
        self.assertEqual(LinkRadiusRuntime.edge_consumer_role("p2c@0"), "critic")
        self.assertEqual(LinkRadiusRuntime.edge_consumer_role("c2s@0"), "solver")
        self.assertEqual(LinkRadiusRuntime.edge_consumer_role("s2p@0"), "planner")

    @staticmethod
    def _task_key(*extra: str) -> str:
        args = build_parser().parse_args([
            "--workflow", "engineering", "--stage", "discover", *extra
        ])
        return build_grid(_build_grid_config(args))[0].config_key

    def test_resolved_topology_is_part_of_gpu_task_identity(self) -> None:
        default = self._task_key()
        explicit_same = self._task_key(
            "--planner-device", "cuda:0",
            "--critic-device", "cuda:0",
            "--solver-device", "cuda:0",
        )
        parallel = self._task_key(
            "--planner-device", "cuda:0",
            "--critic-device", "cuda:1",
            "--solver-device", "cuda:2",
        )
        split_terminal = self._task_key(
            "--planner-device", "cuda:0",
            "--critic-device", "cuda:1",
            "--solver-device", "cuda:2",
            "--terminal-solver-device", "cuda:3",
        )
        self.assertEqual(default, explicit_same)
        self.assertNotEqual(default, parallel)
        self.assertNotEqual(parallel, split_terminal)

    def test_relay_transfer_mode_is_part_of_gpu_task_identity(self) -> None:
        staged = self._task_key("--relay-transfer-mode", "cpu_staged")
        direct = self._task_key("--relay-transfer-mode", "direct")
        self.assertNotEqual(staged, direct)

    def test_autograd_memory_mode_is_part_of_gpu_task_identity(self) -> None:
        ordinary = self._task_key("--autograd-memory-mode", "none")
        checkpointed = self._task_key("--autograd-memory-mode", "checkpoint")
        self.assertNotEqual(ordinary, checkpointed)

    def test_shell_rejects_array_gpu_mask_with_role_placement(self) -> None:
        common = REPO_ROOT / "experiments" / "linkradius" / "linkradius_common.sh"
        env = dict(os.environ)
        env.update({
            "GPU_LIST": "3",
            "PLANNER_DEVICE": "cuda:0",
            "CRITIC_DEVICE": "cuda:1",
            "SOLVER_DEVICE": "cuda:2",
            "TERMINAL_SOLVER_DEVICE": "cuda:3",
        })
        completed = subprocess.run(
            ["bash", "-c", f'source "{common}"; lr_configure_gpu 0'],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("cannot be combined with per-role devices", completed.stderr)

    def test_shell_passes_authoritative_relay_transfer_mode(self) -> None:
        common = REPO_ROOT / "experiments" / "linkradius" / "linkradius_common.sh"
        env = dict(os.environ)
        env["RELAY_TRANSFER_MODE"] = "direct"
        completed = subprocess.run(
            [
                "bash",
                "-c",
                (
                    f'source "{common}"; '
                    "lr_validate_relay_transfer_mode; "
                    "lr_build_command engineering discover 0; "
                    "printf '%s\\n' \"${LR_COMMAND[@]}\""
                ),
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        arguments = completed.stdout.splitlines()
        flag_index = arguments.index("--relay-transfer-mode")
        self.assertEqual(arguments[flag_index + 1], "direct")

    def test_shell_passes_terminal_device_and_checkpoint_mode(self) -> None:
        common = REPO_ROOT / "experiments" / "linkradius" / "linkradius_common.sh"
        env = dict(os.environ)
        env.update({
            "TERMINAL_SOLVER_DEVICE": "cuda:3",
            "AUTOGRAD_MEMORY_MODE": "checkpoint",
        })
        completed = subprocess.run(
            [
                "bash",
                "-c",
                (
                    f'source "{common}"; '
                    "lr_validate_autograd_memory_mode; "
                    "lr_build_command engineering gradient 1; "
                    "printf '%s\\n' \"${LR_COMMAND[@]}\""
                ),
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        arguments = completed.stdout.splitlines()
        terminal_index = arguments.index("--terminal-solver-device")
        memory_index = arguments.index("--autograd-memory-mode")
        self.assertEqual(arguments[terminal_index + 1], "cuda:3")
        self.assertEqual(arguments[memory_index + 1], "checkpoint")


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SystemLoaderPlacementTests(unittest.TestCase):
    @staticmethod
    def _paths():
        roles = ("planner", "critic", "solver")
        return SimpleNamespace(
            style="sequential_scaled",
            family="sequential",
            dataset="gpqa",
            repo_ids={role: f"Org/{role}" for role in roles} | {"outer": "Org/outer"},
            repo_paths={role: Path(f"/{role}") for role in roles},
            inner_adapter_paths={role: Path(f"/{role}/inner") for role in roles},
            outer_adapter_paths={
                key: Path(f"/{key}") for key in ("outer_12", "outer_23", "outer_31")
            },
        )

    @staticmethod
    def _model():
        model = torch.nn.Module()
        model.embedding = torch.nn.Embedding(4, 8)
        model.get_input_embeddings = lambda: model.embedding
        return model

    def test_loader_places_agents_terminal_replica_and_outer_adapters(self) -> None:
        from RecursiveMAS import system_loader

        model_devices = []
        inner_devices = []
        outer_devices = []

        def load_model(**kwargs):
            model_devices.append(str(kwargs["device"]))
            return self._model(), object()

        def load_inner(**kwargs):
            inner_devices.append(str(kwargs["device"]))
            return torch.nn.Identity()

        def load_outer(**kwargs):
            outer_devices.append(str(kwargs["device"]))
            return torch.nn.Identity()

        topology = {
            "planner": "cuda:0",
            "critic": "cuda:1",
            "solver": "cuda:2",
            "terminal_solver": "cuda:3",
        }
        with (
            mock.patch.object(system_loader, "resolve_mas_paths", return_value=self._paths()),
            mock.patch.object(system_loader, "_validate_role_devices"),
            mock.patch.object(system_loader.base, "load_agent_model_and_tokenizer", side_effect=load_model),
            mock.patch.object(system_loader.base, "load_inner_adapter_module", side_effect=load_inner),
            mock.patch.object(system_loader.base, "infer_outer_adapter_out_dim_from_file", return_value=8),
            mock.patch.object(system_loader.base, "load_outer_adapter_module", side_effect=load_outer),
        ):
            system = system_loader.load_mas_system(
                "sequential_scaled",
                dataset="gpqa",
                device="cuda:0",
                role_devices=topology,
            )

        self.assertEqual(
            model_devices,
            ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        )
        self.assertEqual(inner_devices, ["cuda:0", "cuda:1", "cuda:2"])
        # outer_12 is planner->critic, outer_23 critic->solver, and outer_31
        # solver->planner, so each adapter belongs beside its source role.
        self.assertEqual(outer_devices, ["cuda:0", "cuda:1", "cuda:2"])
        self.assertEqual(
            {role: str(device) for role, device in system.role_devices.items()},
            topology,
        )
        self.assertIsNotNone(system.terminal_solver)
        self.assertEqual(system.terminal_solver.role, "terminal_solver")

    def test_unknown_role_is_rejected_before_path_resolution(self) -> None:
        from RecursiveMAS import system_loader

        with mock.patch.object(system_loader, "resolve_mas_paths") as resolve:
            with self.assertRaisesRegex(ValueError, "not part of"):
                system_loader.load_mas_system(
                    "sequential_scaled",
                    device="cpu",
                    role_devices={"summarizer": "cpu"},
                )
        resolve.assert_not_called()

    def test_hidden_cuda_role_fails_before_path_resolution(self) -> None:
        from RecursiveMAS import system_loader

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=2),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(system_loader, "resolve_mas_paths") as resolve,
        ):
            with self.assertRaisesRegex(RuntimeError, "scheduler-visible"):
                system_loader.load_mas_system(
                    "sequential_scaled",
                    device="cuda:0",
                    role_devices={
                        "planner": "cuda:0",
                        "critic": "cuda:1",
                        "solver": "cuda:2",
                    },
                )
        resolve.assert_not_called()

    def test_hidden_terminal_replica_fails_before_path_resolution(self) -> None:
        from RecursiveMAS import system_loader

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=3),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(system_loader, "resolve_mas_paths") as resolve,
        ):
            with self.assertRaisesRegex(RuntimeError, "terminal_solver"):
                system_loader.load_mas_system(
                    "sequential_scaled",
                    device="cuda:0",
                    role_devices={
                        "planner": "cuda:0",
                        "critic": "cuda:1",
                        "solver": "cuda:2",
                        "terminal_solver": "cuda:3",
                    },
                )
        resolve.assert_not_called()

    def test_same_device_relay_stays_direct_and_differentiable(self) -> None:
        runtime = LinkRadiusRuntime(RuntimeConfig(device="cpu"))
        source = torch.tensor([[-2.0, 0.5, 3.0]], requires_grad=True)

        receiver, realized = runtime._transfer_relay(
            source,
            torch.device("cpu"),
            torch.float32,
        )

        self.assertEqual(realized, "same_device_direct")
        self.assertTrue(torch.equal(receiver, source))
        receiver.square().sum().backward()
        self.assertTrue(torch.equal(source.grad, 2.0 * source.detach()))

    def test_cpu_staged_cross_gpu_relay_preserves_values_and_gradients(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("requires at least two scheduler-visible CUDA devices")

        runtime = LinkRadiusRuntime(
            RuntimeConfig(
                device="cuda:0",
                planner_device="cuda:0",
                critic_device="cuda:1",
                solver_device="cuda:1",
                relay_transfer_mode="cpu_staged",
            )
        )
        source = torch.linspace(
            -4.0,
            4.0,
            257,
            device="cuda:0",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        receiver, realized = runtime._transfer_relay(
            source,
            torch.device("cuda:1"),
            torch.bfloat16,
        )

        self.assertEqual(realized, "cpu_float32_staged_cross_device")
        self.assertEqual(receiver.device, torch.device("cuda:1"))
        self.assertEqual(receiver.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(receiver).all().item())
        self.assertTrue(torch.equal(receiver.cpu(), source.detach().cpu()))

        receiver.float().square().sum().backward()
        self.assertIsNotNone(source.grad)
        self.assertTrue(torch.isfinite(source.grad).all().item())
        self.assertGreater(torch.count_nonzero(source.grad).item(), 0)


if __name__ == "__main__":
    unittest.main()
