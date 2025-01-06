import importlib
from logging import getLogger
from pathlib import Path
from typing import Tuple

from .analyser import CodeAnalyzer
from .manager import WorkspaceManager

logger = getLogger(__name__)


class NotebookTools:
    """Orchestrates the export process."""

    def __init__(self, notebook_path: str, output_workspace: str):
        """Initialize NotebookTools.

        Args:
            notebook_path: Path to the Jupyter notebook
            output_workspace: Path where the workspace will be created
            code_analyzer: Optional CodeAnalyzer instance
            workspace_manager: Optional WorkspaceManager instance
        """
        self.notebook_path = Path(notebook_path).resolve()
        if not self.notebook_path.exists() or not self.notebook_path.is_file():
            raise FileNotFoundError(f"The Jupyter notebook at {notebook_path} does not exist")

        self.output_workspace_path = Path(output_workspace).resolve()

        # Initialize workspace manager
        self.workspace_manager = WorkspaceManager(self.output_workspace_path)

        # Initialize code analyzer
        self.code_analyzer = CodeAnalyzer(
            self.notebook_path, self.workspace_manager.created_workspace_path
        )

    @classmethod
    def export_federated(cls, notebook_path: str, output_workspace: str) -> Tuple[str, str]:
        """Export workspace for FederatedRuntime.

        Args:
            notebook_path: Path to the Jupyter notebook
            output_workspace: Path where the workspace will be created

        Returns:
            Tuple[str, str]: (archive_path, flow_class_name)
        """
        instance = cls(notebook_path, output_workspace)
        instance.generate_requirements()
        instance.generate_plan_yaml()
        return instance.workspace_manager.create_experiment_archive()

    @classmethod
    def export(cls, notebook_path: str, output_workspace: str) -> None:
        """Export complete workspace with all configurations.

        Args:
            notebook_path: Path to the Jupyter notebook
            output_workspace: Path where the workspace will be created
        """
        instance = cls(notebook_path, output_workspace)
        instance.generate_requirements()
        instance.generate_plan_yaml()
        instance.generate_data_yaml()

    def generate_plan_yaml(self) -> None:
        """Generate plan.yaml with flow configuration."""
        # Get FLSpec class
        flspec = importlib.import_module("openfl.experimental.workflow.interface").FLSpec

        # Get flow class details
        flow_details = self.code_analyzer.get_flow_class_details(flspec)

        # Analyze and generate plan configuration
        flow_config = self.code_analyzer.analyze_flow_configuration(flow_details)

        # Generate plan.yaml
        self.workspace_manager.generate_plan_yaml(flow_config)

    def generate_data_yaml(self) -> None:
        """Generate data.yaml with runtime configurations."""

        # Ensure flow class
        if not hasattr(self, "flow_class_name"):
            flspec = importlib.import_module("openfl.experimental.workflow.interface").FLSpec
            flow_details = self.code_analyzer.get_flow_class_details(flspec)
            self.flow_class_name = flow_details["flow_class_name"]

        # Get runtime information
        runtime, flow_name = self.code_analyzer._get_runtime_info()

        # Get configurations
        runtime_info = {
            "aggregator_info": self.code_analyzer._get_aggregator_info(runtime),
            "collaborator_info": self.code_analyzer._get_collaborator_info(runtime),
            "flow_name": flow_name,
        }

        # Generate data.yaml
        self.workspace_manager.generate_data_yaml(runtime_info, self.code_analyzer.script_name)

    def generate_requirements(self) -> None:
        """Generate requirements.yaml with package dependencies."""
        # Get package requirements
        requirements, line_numbers, file_content = self.code_analyzer.analyze_requirements()

        self.workspace_manager.update_requirements(requirements)

        self.code_analyzer.remove_lines(line_numbers, file_content)
