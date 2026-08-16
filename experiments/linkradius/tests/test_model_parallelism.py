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
        )
        self.assertEqual(fallback.resolved_role_devices(), {
            "planner": "cpu",
            "critic": "cpu",
            "solver": "cpu",
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
        })

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
        self.assertEqual(default, explicit_same)
        self.assertNotEqual(default, parallel)

    def test_shell_rejects_array_gpu_mask_with_role_placement(self) -> None:
        common = REPO_ROOT / "experiments" / "linkradius" / "linkradius_common.sh"
        env = dict(os.environ)
        env.update({
            "GPU_LIST": "3",
            "PLANNER_DEVICE": "cuda:0",
            "CRITIC_DEVICE": "cuda:1",
            "SOLVER_DEVICE": "cuda:2",
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

    def test_loader_places_agents_and_outer_adapters_by_source_role(self) -> None:
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

        self.assertEqual(model_devices, ["cuda:0", "cuda:1", "cuda:2"])
        self.assertEqual(inner_devices, ["cuda:0", "cuda:1", "cuda:2"])
        # outer_12 is planner->critic, outer_23 critic->solver, and outer_31
        # solver->planner, so each adapter belongs beside its source role.
        self.assertEqual(outer_devices, ["cuda:0", "cuda:1", "cuda:2"])
        self.assertEqual(
            {role: str(device) for role, device in system.role_devices.items()},
            topology,
        )

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


if __name__ == "__main__":
    unittest.main()
