# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 VMware, Inc.
# SPDX-License-Identifier: Apache-2.0
import importlib

if importlib.util.find_spec('torch'):
    from openfl.pipelines.eden_pipeline import EdenPipeline  # NOQA

from openfl.pipelines.kc_pipeline import KCPipeline
from openfl.pipelines.no_compression_pipeline import NoCompressionPipeline
from openfl.pipelines.random_shift_pipeline import RandomShiftPipeline
from openfl.pipelines.skc_pipeline import SKCPipeline
from openfl.pipelines.stc_pipeline import STCPipeline
from openfl.pipelines.tensor_codec import TensorCodec
