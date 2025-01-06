import ast
import importlib
import inspect
import re
import sys
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nbformat
from nbdev.export import nb_export

logger = getLogger(__name__)


class CodeAnalyzer:
    """Code transforamtion and analysis functionality for workspace export
    Args:
        notebook_path: Path to the Jupyter notebook
        output_workspace: Path where the workspace will be created
    """

    def __init__(self, notebook_path: Path, output_path: Path) -> None:
        """Initialize CodeTransformer"""

        self.notebook_path = notebook_path
        self.output_path = output_path
        # self.script_path = None

        # These will be set when importing the script
        self.exported_script_module = None
        self.available_modules_in_exported_script = None

        self._initialize_script()
        self.script_name = self.script_path.name.split(".")[0].strip()

        # Comment flow.run() so when script is imported flow does not start
        # executing
        self._comment_flow_execution()
        # This is required as Ray created actors too many actors when
        # backend="ray"
        self._change_runtime()

    def _initialize_script(self) -> None:
        """Initialize and process the script from notebook"""
        export_filename = self._get_exp_name()
        if export_filename is None:
            raise NameError(
                "Please include `#| default_exp <experiment_name>` in "
                "the first cell of the notebook."
            )
        self.script_path = Path(
            self.__convert_to_python(
                self.notebook_path,
                self.output_path.joinpath("src"),
                f"{export_filename}.py",
            )
        ).resolve()

    def _get_exp_name(self) -> None:
        """Fetch the experiment name from the Jupyter notebook."""
        with open(str(self.notebook_path), "r") as f:
            notebook_content = nbformat.read(f, as_version=nbformat.NO_CONVERT)

        for cell in notebook_content.cells:
            if cell.cell_type == "code":
                code = cell.source
                match = re.search(r"#\s*\|\s*default_exp\s+(\w+)", code)
                if match:
                    logger.info(f"Retrieved {match.group(1)} from default_exp")
                    return match.group(1)
        return None

    def _convert_to_python(self, notebook_path: Path, output_path: Path, export_filename) -> Path:
        """Converts a Jupyter notebook to a Python script.

        Args:
            notebook_path (Path): The path to the Jupyter notebook file
                to be converted.
            output_path (Path): The directory where the exported Python
                script should be saved.
            export_filename: The name of the exported Python script file.
        """
        nb_export(notebook_path, output_path)

        return Path(output_path).joinpath(export_filename).resolve()

    def _comment_flow_execution(self) -> None:
        """In the python script search for ".run()" and comment it."""
        with open(self.script_path, "r") as f:
            data = f.readlines()
        for idx, line in enumerate(data):
            if ".run()" in line:
                data[idx] = f"# {line}"
        with open(self.script_path, "w") as f:
            f.writelines(data)

    def _change_runtime(self) -> None:
        """Change the LocalRuntime backend from ray to single_process."""
        with open(self.script_path, "r") as f:
            data = f.read()

        if "backend='ray'" in data or 'backend="ray"' in data:
            data = data.replace("backend='ray'", "backend='single_process'").replace(
                'backend="ray"', 'backend="single_process"'
            )

        with open(self.script_path, "w") as f:
            f.write(data)

    def _get_class_arguments(self, class_name) -> list:
        """Given the class name returns expected class arguments.

        Args:
            class_name (str): Name of the class
        """
        # Import python script if not already
        if not hasattr(self, "exported_script_module"):
            self.__import_exported_script()

        # Find class from imported python script module
        for idx, attr in enumerate(self.available_modules_in_exported_script):
            if attr == class_name:
                cls = getattr(
                    self.exported_script_module,
                    self.available_modules_in_exported_script[idx],
                )

        # If class not found
        if "cls" not in locals():
            raise NameError(f"{class_name} not found.")

        if inspect.isclass(cls):
            # Check if the class has an __init__ method
            if "__init__" in cls.__dict__:
                init_signature = inspect.signature(cls.__init__)
                # Extract the parameter names (excluding 'self', 'args', and
                # 'kwargs')
                arg_names = [
                    param
                    for param in init_signature.parameters
                    if param not in ("self", "args", "kwargs")
                ]
                return arg_names
            return []
        logger.error(f"{cls} is not a class")

    def _get_class_name_and_sourcecode_from_parent_class(
        self, parent_class
    ) -> Optional[Tuple[Optional[str], Optional[str]]]:
        """Provided the parent_class name returns derived class source code and
        name.

        Args:
            parent_class: FLSpec instance
        """
        # Import python script if not already
        if not hasattr(self, "exported_script_module"):
            self.__import_exported_script()

        # Going though all attributes in imported python script
        for attr in self.available_modules_in_exported_script:
            t = getattr(self.exported_script_module, attr)
            if inspect.isclass(t) and t != parent_class and issubclass(t, parent_class):
                return inspect.getsource(t), attr

        return None, None

    def _extract_class_initializing_args(self, class_name) -> Dict[str, Any]:  # noqa: C901
        """Provided name of the class returns expected arguments and it's
        values in form of dictionary.

        Args:
            class_name (str): Name of the class
        """
        instantiation_args = {"args": {}, "kwargs": {}}

        with open(self.script_path, "r") as s:
            tree = ast.parse(s.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == class_name:
                        # We found an instantiation of the class
                        for arg in node.args:
                            # Iterate through positional arguments
                            if isinstance(arg, ast.Name):
                                # Use the variable name as the argument value
                                instantiation_args["args"][arg.id] = arg.id
                            elif isinstance(arg, ast.Constant):
                                instantiation_args["args"][arg.s] = ast.unparse(arg)
                            else:
                                instantiation_args["args"][arg.arg] = ast.unparse(arg).strip()

                        for kwarg in node.keywords:
                            # Iterate through keyword arguments
                            value = ast.unparse(kwarg.value).strip()

                            # If paranthese or brackets around the value is
                            # found and it's not tuple or list remove
                            # paranthese or brackets
                            if value.startswith("(") and "," not in value:
                                value = value.lstrip("(").rstrip(")")
                            if value.startswith("[") and "," not in value:
                                value = value.lstrip("[").rstrip("]")
                            try:
                                # Evaluate the value to convert it from a
                                # string representation into its corresponding
                                # python object.
                                value = ast.literal_eval(value)
                            except ValueError:
                                # ValueError is ignored because we want the
                                # value as a string
                                pass
                            instantiation_args["kwargs"][kwarg.arg] = value

        return instantiation_args

    def analyze_requirements(self):
        data = None
        with open(self.script_path, "r") as f:
            requirements = []
            line_nos = []
            data = f.readlines()
            for i, line in enumerate(data):
                line = line.strip()
                if "pip install" in line:
                    line_nos.append(i)
                    # Avoid commented lines, libraries from *.txt file, or openfl.git
                    # installation
                    if not line.startswith("#") and "-r" not in line and "openfl.git" not in line:
                        requirements.append(f"{line.split(' ')[-1].strip()}\n")

            return requirements, line_nos, data

    def remove_lines(self, data: List[str], line_nos: List[int]) -> None:
        """Removes pip install lines from the script"""

        with open(self.script_path, "w") as f:
            for i, line in enumerate(data):
                if i not in line_nos:
                    f.write(line)

    def get_flow_class_details(self, parent_class) -> Dict[str, Any]:
        # Get flow class name
        _, flow_class_name = self._get_class_name_and_sourcecode_from_parent_class(parent_class)
        if not flow_class_name:
            raise ValueError("No flow class found that inherits from FLSpec")

        # Get expected arguments
        expected_arguments = self._get_class_arguments(flow_class_name)

        # get initialization arguments
        init_args = self._extract_class_initializing_args(flow_class_name)

        return {
            "flow_class_name": flow_class_name,
            "expected_args": expected_arguments,
            "init_args": init_args,
        }

    def analyze_flow_configuration(self, flow_details: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze flow configuration from flow details.

        Args:
            flow_details: Dictionary containing flow class details

        Returns:
            Dict containing plan configuration
        """
        flow_config = {
            "federated_flow": {
                "settings": {},
                "template": f"src.{self.script_name}.{flow_details['flow_class_name']}",
            }
        }

        def update_dictionary(args: dict, dtype: str = "args") -> None:
            """Update plan configuration with argument values.

            Args:
                args: Dictionary of arguments to process
                dtype: Type of arguments ('args' or 'kwargs')
            """
            for idx, (k, v) in enumerate(args.items()):
                if dtype == "args":
                    v = getattr(self.exported_script_module, str(k), None)
                    if v is not None and not isinstance(v, (int, str, bool)):
                        v = f"src.{self.script_name}.{k}"
                    k = flow_details["expected_args"][idx]
                elif dtype == "kwargs":
                    if v is not None and not isinstance(v, (int, str, bool)):
                        v = f"src.{self.script_name}.{v}"
                flow_config["federated_flow"]["settings"].update({k: v})

        # Process arguments
        pos_args = flow_details["init_args"].get("args", {})
        update_dictionary(pos_args, "args")
        kw_args = flow_details["init_args"].get("kwargs", {})
        update_dictionary(kw_args, "kwargs")

        return flow_config

    def _import_exported_script(self) -> None:
        """
        Imports generated python script with help of importlib
        """
        try:
            sys.path.append(str(self.script_path.parent))
            self.exported_script_module = importlib.import_module(self.script_name)
            self.available_modules_in_exported_script = dir(self.exported_script_module)

        except ImportError as e:
            logger.error(f"Failed to import script {self.script_name}: {e}")
            raise

    def _get_runtime_info(self) -> Tuple[Any, str]:
        """
        Find the federated flow class and runtime
        """
        if not hasattr(self, "exported_script_module"):
            self._import_exported_script()

        # if not hasattr(self, "flow_class_name"):
        #     raise ValueError("Flow class name not set")

        federated_flow_class = getattr(self.exported_script_module, self.flow_class_name)

        for t in self.available_modules_in_exported_script:
            tempstring = t
            t = getattr(self.exported_script_module, t)
            if isinstance(t, federated_flow_class):
                flow_name = tempstring
                if not hasattr(t, "_runtime"):
                    raise AttributeError("Unable to locate LocalRuntime instantiation")
                runtime = t._runtime
                if not hasattr(runtime, "collaborators"):
                    raise AttributeError("LocalRuntime instance does not have collaborators")

                return runtime, flow_name

    def _get_aggregator_info(self, runtime: Any) -> Dict[str, Any]:
        """Get aggregator configuration information."""
        aggregator = runtime._aggregator

        # Get initialization arguments for Aggregator
        init_args = self._extract_class_initializing_args("Aggregator")["kwargs"]

        return {
            "private_attrs_callable": aggregator.private_attributes_callable,
            "private_attributes": aggregator.private_attributes,
            "kwargs": getattr(aggregator, "kwargs", {}),
            "init_args": init_args,  # Include initialization arguments
        }

    def _get_collaborator_info(self, runtime: Any) -> Dict[str, Dict[str, Any]]:
        """Get collaborator configuration information."""
        collaborators = runtime._LocalRuntime__collaborators

        # Get initialization arguments for Collaborator
        init_args = self._extract_class_initializing_args("Collaborator")["kwargs"]

        collaborator_info = {}
        for collab in collaborators.values():
            collab_name = collab.get_name()
            collaborator_info[collab_name] = {
                "callable_func": collab.private_attributes_callable,
                "private_attributes": collab.private_attributes,
                "kwargs": runtime.get_collaborator_kwargs(collab_name),
                "init_args": init_args,  # Include initialization arguments
            }

        return collaborator_info
