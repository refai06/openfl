# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import shutil
from logging import getLogger
from pathlib import Path
from shutil import copytree
from typing import Any, Dict, Tuple

import yaml

from openfl.experimental.workflow.interface.cli.cli_helper import print_tree

logger = getLogger(__name__)


class WorkspaceManager:
    """Manages workspace operation and configuration generation.
    Attributes:
        notebook_path: Absolute path of jupyter notebook.
        output_workspace_path: Output directory for new generated workspace
        template_workspace_path: Path to template workspace provided with Openfl
        script_path: Path to the generated Python script

    """

    def __init__(self, output_workspace_path: Path) -> None:
        """Initialize workspace Manager object"""
        self.output_workspace_path = output_workspace_path
        # self._prepare_output_directory()

        self.template_workspace_path = self._get_template_workspace_path()
        self.created_workspace_path = self._setup_workspace()

    def _get_template_workspace_path(self) -> Path:
        """Returns the path to the template workspace

        Returns:
            Path to the template workspace
        """
        return (
            Path(f"{__file__}")
            .parent.parent.parent.parent.parent.joinpath(
                "openfl-workspace",
                "experimental",
                "workflow",
                "AggregatorBasedWorkflow",
                "template_workspace",
            )
            .resolve(strict=True)
        )

    def _setup_workspace(self) -> Path:
        """Sets up the workspace directory structure
        Returns:
            Path: Path to created workspace
        """
        # Regenerate the workspace if it already exists
        if self.output_workspace_path.exists():
            shutil.rmtree(self.output_workspace_path)
        self.output_workspace_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy template workspace to output directory
        created_workspace_path = Path(
            copytree(self.template_workspace_path, self.output_workspace_path)
        )
        logger.info(f"Copied template workspace to {self.created_workspace_path}")

        print_tree(created_workspace_path, level=2)

        return created_workspace_path

    def _read_yaml(self, path) -> None:
        """Read YAML file"""
        try:
            with open(path, "r") as y:
                return yaml.safe_load(y)
        except Exception as e:
            logger.warning(f"Error reading Yaml file {path}:{e}")
            return None

    def _write_yaml(self, path, data) -> None:
        """Write YAML file"""

        with open(path, "w") as y:
            yaml.safe_dump(data, y)

    def create_experiment_archive(self) -> Tuple[str, str]:
        """
        Create archive of the generated workspace

        Returns:
            Tuple[str, str]: A tuple containing:
                (generated_workspace_path, archive_path, flow_class_name).
        """
        parent_directory = self.output_workspace_path.parent
        archive_path = parent_directory / "experiment"

        # Create a ZIP archive of the generated_workspace directory
        arch_path = shutil.make_archive(str(archive_path), "zip", str(self.output_workspace_path))

        print(f"Archive created at {archive_path}.zip")

        return arch_path, self.flow_class_name

    def update_requirements(self, requirements) -> None:
        """Update requiements.txt with new requirements.
        Args:
            requirements: List of requirements strings to add
        """
        requirements_filepath = str(
            self.created_workspace_path.joinpath("requirements.txt").resolve()
        )

        with open(requirements_filepath, "a") as f:
            f.writelines(requirements)

    def generate_plan_yaml(self, flow_config: Dict[str, Any]) -> None:
        """Generate plan.yaml with flow configuration"""

        plan_path = self.created_workspace_path.joinpath("plan", "plan.yaml").resolve()

        data = self._read_yaml(plan_path)

        if data is None:
            data = {"federated_flow": {"settings": {}, "template": ""}}

        # update with new flow configuration
        data["federated_flow"].update(flow_config)

        # write updated plan
        self._write_yaml(plan_path, data)

    def _prepare_aggregator_data(
        self,
        data: Dict[str, Any],
        aggregator_info: Dict[str, Any],
        flow_name: str,
        script_name: str,
    ) -> Dict[str, Any]:
        """Prepare aggregator configuration for data.yaml."""
        runtime_name = "runtime_local"

        if aggregator_info["private_attrs_callable"] is not None:
            data["aggregator"] = {
                "callable_func": {
                    "settings": {},
                    "template": (
                        f"src.{script_name}."
                        f"{aggregator_info['private_attrs_callable'].__name__}"
                    ),
                }
            }

            # Process kwargs using initialization arguments
            for key, value in aggregator_info["kwargs"].items():
                if isinstance(value, (int, str, bool)):
                    data["aggregator"]["callable_func"]["settings"][key] = value
                else:
                    # Use initialization arguments to get the correct name
                    arg = aggregator_info["init_args"].get(key, key)
                    data["aggregator"]["callable_func"]["settings"][key] = (
                        f"src.{script_name}.{arg}"
                    )

        elif aggregator_info["private_attributes"]:
            with open(self.script_path, "a") as f:
                f.write(f"\n{runtime_name} = {flow_name}._runtime\n")
                f.write(
                    f"aggregator_private_attributes = "
                    f"{runtime_name}._aggregator.private_attributes\n"
                )

            data["aggregator"] = {
                "private_attributes": f"src.{script_name}.aggregator_private_attributes"
            }

        return data

    def _prepare_collaborator_data(
        self,
        data: Dict[str, Any],
        collaborator_info: Dict[str, Any],
        flow_name: str,
        script_name: str,
    ) -> Dict[str, Any]:
        """Prepare collaborator configuration for data.yaml."""
        runtime_name = "runtime_local"
        runtime_created = False
        runtime_collab_created = False

        for collab_name, collab_data in collaborator_info.items():
            if collab_data["callable_func"]:
                # Pass initialization arguments to handler
                data = self._handle_collaborator_callable(
                    collab_name, collab_data["kwargs"], collab_data["init_args"], data, script_name
                )
            elif collab_data["private_attributes"]:
                with open(self.script_path, "a") as f:
                    if not runtime_created:
                        f.write(f"\n{runtime_name} = {flow_name}._runtime\n")
                        runtime_created = True
                    if not runtime_collab_created:
                        f.write(
                            f"\nruntime_collaborators = "
                            f"{runtime_name}._LocalRuntime__collaborators\n"
                        )
                        runtime_collab_created = True
                    f.write(
                        f"\n{collab_name}_private_attributes = "
                        f"runtime_collaborators['{collab_name}'].private_attributes\n"
                    )

                data[collab_name] = {
                    "private_attributes": (f"src.{script_name}.{collab_name}_private_attributes")
                }

        return data

    def _handle_collaborator_callable(
        self,
        collab_name: str,
        kwargs: Dict[str, Any],
        init_args: Dict[str, Any],
        data: Dict[str, Any],
        script_name: str,
    ) -> Dict[str, Any]:
        """Handle collaborator with callable function."""
        if collab_name not in data:
            data[collab_name] = {"callable_func": {"settings": {}, "template": None}}

        for key, value in kwargs.items():
            if key == "private_attributes_callable":
                value = f"src.{script_name}.{value}"
                data[collab_name]["callable_func"]["template"] = value
            elif isinstance(value, (int, str, bool)):
                data[collab_name]["callable_func"]["settings"][key] = value
            else:
                # Use initialization arguments to get the correct name
                arg = init_args.get(key, key)
                data[collab_name]["callable_func"]["settings"][key] = f"src{script_name}.{arg}"

        return data

    def generate_data_yaml(self, runtime_info, script_name) -> None:
        """Generate data.yaml with runtime configuration"""

        data_yaml = self.created_workspace_path.joinpath("plan", "data.yaml").resolve()
        data = self._read_yaml(data_yaml) or {}

        data = self._prepare_aggregator_data(
            data, runtime_info["aggregator_info"], runtime_info["flow_name"], script_name
        )

        data = self._prepare_collaborator_data(
            data, runtime_info["collaborator_info"], runtime_info["flow_name"], script_name
        )

        self._write_yaml(data_yaml, data)
