# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""openfl.experimental.interface.flspec module."""

from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Callable, List, Type

from openfl.experimental.runtime import Runtime
from openfl.experimental.utilities import (
    MetaflowInterface,
    SerializationError,
    aggregator_to_collaborator,
    checkpoint,
    collaborator_to_aggregator,
    filter_attributes,
    generate_artifacts,
    should_transfer,
)


class FLSpec:
    _clones = []
    _initial_state = None

    def __init__(self, checkpoint: bool = False):
        self._foreach_methods = []
        self._checkpoint = checkpoint

    @classmethod
    def _create_clones(cls, instance: Type[FLSpec], names: List[str]) -> None:
        """Creates clones for instance for each collaborator in names"""
        cls._clones = {name: deepcopy(instance) for name in names}

    @classmethod
    def _reset_clones(cls):
        """Reset clones"""
        cls._clones = {}

    @classmethod
    def save_initial_state(cls, instance: Type[FLSpec]) -> None:
        """Save initial state of instance before executing the flow"""
        cls._initial_state = deepcopy(instance)

    def run(self) -> None:
        """Starts the execution of the flow"""
        # Submit flow to Runtime
        if str(self._runtime) == "LocalRuntime":
            self._metaflow_interface = MetaflowInterface(self.__class__, self.runtime.backend)
            self._run_id = self._metaflow_interface.create_run()
            # Initialize aggregator private attributes
            self.runtime.initialize_aggregator()
            self._foreach_methods = []
            FLSpec._reset_clones()
            FLSpec._create_clones(self, self.runtime.collaborators)
            # Initialize collaborator private attributes
            self.runtime.initialize_collaborators()
            if self._checkpoint:
                print(f"Created flow {self.__class__.__name__}")
            try:
                # Execute all Participant (Aggregator & Collaborator) tasks and
                # retrieve the final attributes
                # start step is the first task & invoked on aggregator through runtime.execute_task
                final_attributes = self.runtime.execute_task(
                    self,
                    self.start,
                )
            except Exception as e:
                if "cannot pickle" in str(e) or "Failed to unpickle" in str(e):
                    msg = (
                        "\nA serialization error was encountered that could not"
                        "\nbe handled by the ray backend."
                        "\nTry rerunning the flow without ray as follows:\n"
                        "\nLocalRuntime(...,backend='single_process')\n"
                        "\n or for more information about the original error,"
                        "\nPlease see the official Ray documentation"
                        "\nhttps://docs.ray.io/en/releases-2.2.0/ray-core/\
                        objects/serialization.html"
                    )
                    raise SerializationError(str(e) + msg)
                else:
                    raise e
            for name, attr in final_attributes:
                setattr(self, name, attr)
        elif str(self._runtime) == "FederatedRuntime":
            pass
        else:
            raise Exception("Runtime not implemented")

    @property
    def runtime(self) -> Type[Runtime]:
        """Returns flow runtime"""
        return self._runtime

    @runtime.setter
    def runtime(self, runtime: Type[Runtime]) -> None:
        """Sets flow runtime"""
        if isinstance(runtime, Runtime):
            self._runtime = runtime
        else:
            raise TypeError(f"{runtime} is not a valid OpenFL Runtime")

    def _capture_instance_snapshot(self, kwargs):
        """
        Takes backup of self before exclude or include filtering

        Args:
            kwargs: Key word arguments originally passed to the next function.
                    If include or exclude are in the kwargs, the state of the
                    aggregator needs to be retained
        """
        return_objs = []
        if "exclude" in kwargs or "include" in kwargs:
            backup = deepcopy(self)
            return_objs.append(backup)
        return return_objs

    def _is_at_transition_point(self, f: Callable, parent_func: Callable) -> bool:
        """
        Has the collaborator finished its current sequence?

        Args:
            f:           The next function to be executed
            parent_func: The previous function executed
        """
        if parent_func.__name__ in self._foreach_methods:
            self._foreach_methods.append(f.__name__)
            if should_transfer(f, parent_func):
                print(f"Should transfer from {parent_func.__name__} to {f.__name__}")
                self.execute_next = f.__name__
                return True
        return False

    def _display_transition_logs(self, f: Callable, parent_func: Callable) -> None:
        """
        Prints aggregator to collaborators or
        collaborators to aggregator state transition logs
        """
        if aggregator_to_collaborator(f, parent_func):
            print("Sending state from aggregator to collaborators")

        elif collaborator_to_aggregator(f, parent_func):
            print("Sending state from collaborator to aggregator")

    def filter_exclude_include(self, f, **kwargs):
        """
        This function filters exclude/include attributes

        Args:
            flspec_obj  :  Reference to the FLSpec (flow) object
            f           :  The task to be executed within the flow
        """
        selected_collaborators = getattr(self, kwargs["foreach"])

        for col in selected_collaborators:
            clone = FLSpec._clones[col]
            clone.input = col
            if ("exclude" in kwargs and hasattr(clone, kwargs["exclude"][0])) or (
                "include" in kwargs and hasattr(clone, kwargs["include"][0])
            ):
                filter_attributes(clone, f, **kwargs)
            artifacts_iter, _ = generate_artifacts(ctx=self)
            for name, attr in artifacts_iter():
                setattr(clone, name, deepcopy(attr))
            clone._foreach_methods = self._foreach_methods

    def restore_instance_snapshot(self, ctx: FLSpec, instance_snapshot: List[FLSpec]):
        """Restores attributes from backup (in instance snapshot) to ctx"""
        for backup in instance_snapshot:
            artifacts_iter, _ = generate_artifacts(ctx=backup)
            for name, attr in artifacts_iter():
                if not hasattr(ctx, name):
                    setattr(ctx, name, attr)

    def get_clones(self, kwargs):
        """
        Create, and prepare clones
        """
        FLSpec._reset_clones()
        FLSpec._create_clones(self, self.runtime.collaborators)
        selected_collaborators = self.__getattribute__(kwargs["foreach"])

        for col in selected_collaborators:
            clone = FLSpec._clones[col]
            clone.input = col
            artifacts_iter, _ = generate_artifacts(ctx=clone)
            attributes = artifacts_iter()
            for name, attr in attributes:
                setattr(clone, name, deepcopy(attr))
            clone._foreach_methods = self._foreach_methods
            clone._metaflow_interface = self._metaflow_interface

    def next(self, f, **kwargs):
        """
        Next task in the flow to execute
        """
        # Get the name and reference to the calling function
        parent = inspect.stack()[1][3]
        parent_func = getattr(self, parent)

        if str(self._runtime) == "LocalRuntime":
            # Checkpoint current attributes (if checkpoint==True)
            checkpoint(self, parent_func)

        # Take back-up of current state of self
        agg_to_collab_ss = None
        if aggregator_to_collaborator(f, parent_func):
            agg_to_collab_ss = self._capture_instance_snapshot(kwargs=kwargs)

            if str(self._runtime) == "FederatedRuntime":
                if len(FLSpec._clones) == 0:
                    self.get_clones(kwargs)

        # Remove included / excluded attributes from next task
        filter_attributes(self, f, **kwargs)

        if str(self._runtime) == "FederatedRuntime":
            if f.collaborator_step and not f.aggregator_step:
                self._foreach_methods.append(f.__name__)

            if "foreach" in kwargs:
                self.filter_exclude_include(f, **kwargs)
                # if "foreach" in kwargs:
                self.execute_task_args = (
                    self,
                    f,
                    parent_func,
                    FLSpec._clones,
                    agg_to_collab_ss,
                    kwargs,
                )
            else:
                self.execute_task_args = (self, f, parent_func, kwargs)

        elif str(self._runtime) == "LocalRuntime":
            # update parameters required to execute execute_task function
            self.execute_task_args = [f, parent_func, agg_to_collab_ss, kwargs]
