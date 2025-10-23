# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from typing import Any, Callable, Dict

from nemo_skills.evaluation.evaluator.base import BaseEvaluator
from nemo_skills.evaluation.evaluator.bfcl import eval_bfcl
from nemo_skills.evaluation.evaluator.code import (
    eval_bigcodebench,
    eval_evalplus,
    eval_human_eval_infilling,
    eval_livebench_coding,
    eval_livecodebench_pro,
)
from nemo_skills.evaluation.evaluator.ifbench import eval_ifbench
from nemo_skills.evaluation.evaluator.ifeval import eval_if
from nemo_skills.evaluation.evaluator.ioi import eval_ioi
from nemo_skills.evaluation.evaluator.livecodebench import eval_livecodebench
from nemo_skills.evaluation.evaluator.math import (
    Lean4ProofEvaluator,
    Lean4StatementEvaluator,
    MathEvaluator,
)
from nemo_skills.evaluation.evaluator.mcq import eval_mcq
from nemo_skills.evaluation.evaluator.mrcr import eval_mrcr
from nemo_skills.evaluation.evaluator.ojbench import eval_ojbench
from nemo_skills.evaluation.evaluator.ruler import eval_ruler
from nemo_skills.evaluation.evaluator.scicode import eval_scicode


def dummy_eval(cfg):
    return


EVALUATOR_MAP = {
    # Function-based evaluators (batch-only)
    "evalplus": eval_evalplus,
    "if": eval_if,
    "ifbench": eval_ifbench,
    "bfcl": eval_bfcl,
    "no-op": dummy_eval,
    "multichoice": eval_mcq,
    "ruler": eval_ruler,
    "livecodebench": eval_livecodebench,
    "livebench_coding": eval_livebench_coding,
    "livecodebench_pro": eval_livecodebench_pro,
    "scicode": eval_scicode,
    "mrcr": eval_mrcr,
    "ioi": eval_ioi,
    "bigcodebench": eval_bigcodebench,
    "ojbench": eval_ojbench,
    "human_eval_infilling": eval_human_eval_infilling,
}

# Evaluator class mapping
EVALUATOR_CLASS_MAP = {
    "math": MathEvaluator,
    "lean4-proof": Lean4ProofEvaluator,
    "lean4-statement": Lean4StatementEvaluator,
    # Other evaluators can be added here as they're converted to classes
}

# Validation: Ensure no overlap between class and function maps
_class_types = set(EVALUATOR_CLASS_MAP.keys())
_function_types = set(EVALUATOR_MAP.keys())
_overlap = _class_types.intersection(_function_types)
if _overlap:
    raise ValueError(
        f"Evaluator types cannot be in both EVALUATOR_CLASS_MAP and EVALUATOR_MAP: {_overlap}. "
        f"Each eval_type must be in exactly one map."
    )


def is_evaluator_registered(eval_type: str):
    """Check if evaluator is registered in either class or function map."""
    return eval_type in EVALUATOR_CLASS_MAP or eval_type in EVALUATOR_MAP


def register_evaluator(eval_type: str, eval_fn: Callable[[Dict[str, Any]], None]):
    if is_evaluator_registered(eval_type):
        raise ValueError(f"Evaluator for {eval_type} already registered")

    EVALUATOR_MAP[eval_type] = eval_fn


def get_evaluator_class(eval_type: str, config: Dict[str, Any]) -> BaseEvaluator:
    """Get evaluator instance by type."""
    if eval_type not in EVALUATOR_CLASS_MAP:
        raise ValueError(
            f"Evaluator class not found for type: {eval_type}.\n"
            f"Available types with class support: {list(EVALUATOR_CLASS_MAP.keys())}\n"
            f"All supported types: {list(EVALUATOR_MAP.keys())}"
        )

    evaluator_class = EVALUATOR_CLASS_MAP[eval_type]
    return evaluator_class(config)


def supports_single_eval(eval_type: str, config: Dict[str, Any]) -> bool:
    """Check if evaluator supports single data point evaluation during generation."""
    if eval_type not in EVALUATOR_CLASS_MAP:
        return False  # Only class-based evaluators support single eval

    evaluator = get_evaluator_class(eval_type, config)
    return evaluator.supports_single_eval()


def evaluate(cfg):
    """Main evaluation function that handles both class-based and function-based evaluators."""
    eval_type = cfg.eval_type

    # Check if it's a class-based evaluator first
    if eval_type in EVALUATOR_CLASS_MAP:
        evaluator = get_evaluator_class(eval_type, cfg.eval_config)
        return asyncio.run(evaluator.eval_full(cfg.input_files))

    # Fall back to function-based evaluator
    if eval_type in EVALUATOR_MAP:
        return EVALUATOR_MAP[eval_type](cfg)

    # Not found in either map
    all_types = list(EVALUATOR_CLASS_MAP.keys()) + list(EVALUATOR_MAP.keys())
    raise ValueError(f"Evaluator not found for type: {eval_type}.\nSupported types: {sorted(all_types)}")
