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

import logging
from dataclasses import asdict, field

from nemo_skills.code_execution.proof_utils import (
    ProofBuildConfig,
    build_lean4_proof,
    determine_proof_status,
)
from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.evaluation.evaluator.base import BaseEvaluator
from nemo_skills.evaluation.math_grader import evaluate_result
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class MathEvaluatorConfig:
    numeric_precision: int = 15
    timeout: int = 10
    # if True will not attempt to re-extract based on \boxed or regex
    use_predicted_answer_key: bool = False

    extract_from_boxed: bool = True
    # only used if extract_from_boxed is False
    extract_regex: str = r"The final answer is (.+)$"
    take_modulo: int | None = None  # will take modulo of the gt and predicted answers if not None


@nested_dataclass(kw_only=True)
class LeanEvaluatorConfig:
    sandbox: dict = field(default_factory=lambda: {"sandbox_type": "local"})
    num_parallel_requests: int = 10
    timeout: float = 30.0
    final_answer_key: str = "**FINAL ANSWER**"
    restate_formal_statement: bool = True
    # Which code block to extract when multiple are present: "first" or "last"
    extract_code_mode: str = "last"


# Evaluator Classes


class MathEvaluator(BaseEvaluator):
    def __init__(self, config: dict, num_parallel_requests=10):
        super().__init__(config, num_parallel_requests)
        self.eval_config = MathEvaluatorConfig(**self.config)
        self.eval_config_dict = asdict(self.eval_config)

    async def eval_single(self, data_point: dict[str, any]) -> dict[str, any]:
        """Evaluate single problem for math"""
        return evaluate_result(data_point, **self.eval_config_dict)


class Lean4ProofEvaluator(BaseEvaluator):
    """Lean4 proof evaluator - supports both single and batch evaluation."""

    def __init__(self, config: dict, num_parallel_requests=10):
        """Initialize Lean4ProofEvaluator with sandbox."""
        super().__init__(config, num_parallel_requests)
        eval_config = LeanEvaluatorConfig(**self.config)
        self.sandbox = get_sandbox(**eval_config.sandbox)
        self.eval_config = eval_config

    async def eval_single(self, data_point: dict[str, any]) -> dict[str, any]:
        """Evaluate single Lean4 proof during generation."""

        # Prepare predicted_proof using shared utility
        generation = data_point["generation"]

        config = ProofBuildConfig(
            final_answer_key=self.eval_config.final_answer_key,
            extract_code_mode=self.eval_config.extract_code_mode,
            restate_formal_statement=self.eval_config.restate_formal_statement,
            strip_theorem_from_proof=True,  # Default behavior for proofs
        )

        predicted_proof = build_lean4_proof(
            generation=generation, data_point=data_point, config=config, answer_format="lean4-proof"
        )

        # Execute proof and get compiler output
        output, _ = await self.sandbox.execute_code(
            generated_code=predicted_proof,
            language="lean4",
            timeout=self.eval_config.timeout,
        )

        # Determine proof status using shared utility
        proof_status = determine_proof_status(output)

        return {
            "predicted_proof": predicted_proof,
            "proof_status": proof_status,
            "lean_evaluation": {**output, "timeout": self.eval_config.timeout},
        }


class Lean4StatementEvaluator(BaseEvaluator):
    """Lean4 statement evaluator - only supports batch evaluation."""

    def __init__(self, config: dict, num_parallel_requests=10):
        """Initialize Lean4StatementEvaluator with sandbox."""
        super().__init__(config, num_parallel_requests)
        eval_config = LeanEvaluatorConfig(**self.config)
        self.sandbox = get_sandbox(**eval_config.sandbox)
        self.eval_config = eval_config

    async def eval_full(self, input_files: list[str]) -> None:
        """Batch evaluate Lean4 statements."""
        eval_config_dict = asdict(self.eval_config)
        eval_config_dict.pop("sandbox")
        await self.sandbox.batch_evaluate_results(
            input_files=input_files,
            answer_format="lean4-statement",
            **eval_config_dict,
        )
