# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import json
import logging
import shlex
import shutil
import subprocess
import sys
import textwrap
from contextlib import asynccontextmanager
from dataclasses import field
from typing import Any, Dict, List, Tuple

from nemo_skills.code_execution.sandbox import Sandbox, get_sandbox
from nemo_skills.evaluation.evaluator.code import preprocess_code
from nemo_skills.utils import get_logger_name, nested_dataclass, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))

LIVECODEBENCH_PYTHON_GIT_URL = "git+https://github.com/wasiahmad/livecodebench.git@livecodebench"
LIVECODEBENCH_PYPY3_GIT_URL = "git+https://github.com/wasiahmad/livecodebench.git"


@nested_dataclass(kw_only=True)
class LiveCodeBenchEvaluatorConfig:
    sandbox: dict = field(default_factory=lambda: {"sandbox_type": "local"})
    language: str = "python"  # "cpp" is another option now
    test_file: str = None
    interpreter: str = "python"  # use either "python" or "pypy3"
    timeout: int = 6
    num_processes: int = 12


@asynccontextmanager
async def sandbox_context(config: dict):
    """Provides a managed sandbox instance."""
    sandbox = get_sandbox(**config)
    try:
        yield sandbox
    finally:
        LOG.info("Closing sandbox...")
        await sandbox.close()


def _preprocess_and_validate_file(jsonl_file: str, language: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Reads a JSONL file, preprocesses samples, and validates the release version.
    This function consolidates logic previously duplicated in both evaluation functions.
    """
    with open(jsonl_file, "r", encoding="utf-8") as f_in:
        samples = [preprocess_code(json.loads(line), language) for line in f_in]

    if not samples:
        raise ValueError(f"No samples found in {jsonl_file}")

    # Validate that all samples share the same release version
    versions = {s["release_version"] for s in samples}
    if len(versions) > 1:
        raise ValueError(f"All samples should have the same release version. Found: {versions}")
    release_version = versions.pop()

    # Prepare samples for evaluation
    for s in samples:
        s["question_id"] = s["task_id"]  # Required by the evaluator
        s["code_list"] = [s["completion"]]

    # Overwrite the file with the processed samples
    with open(jsonl_file, "w", encoding="utf-8") as f_out:
        f_out.writelines(json.dumps(sample) + "\n" for sample in samples)

    return samples, release_version


def _postprocess_results(jsonl_file: str, samples: List[Dict[str, Any]]):
    """
    Reads evaluation results, merges them into the samples, and saves the final output.
    This function consolidates logic previously duplicated in both evaluation functions.
    """
    results_file = jsonl_file[:-6] + "_eval_results.json"
    with open(results_file, "r", encoding="utf-8") as f_in:
        eval_grades = json.load(f_in)

    with open(jsonl_file, "w", encoding="utf-8") as f_out:
        for s in samples:
            s["graded_list"] = eval_grades["eval"][s["task_id"]]["graded_list"]
            f_out.write(json.dumps(s) + "\n")

    # Move eval file to ensure metrics are recomputed if run again
    shutil.move(results_file, jsonl_file[:-6] + "_eval_results-saved.json")
    LOG.info(f"Finished processing {jsonl_file}, results saved.")


async def _install_packages_in_sandbox(sandbox: Sandbox, interpreter: str) -> bool:
    """Installs required packages in the provided sandbox."""
    LOG.info(f"Installing livecodebench with {interpreter} in sandbox...")
    pip_cmd = "pip" if interpreter == "python" else "pypy3 -m pip"
    git_url = LIVECODEBENCH_PYTHON_GIT_URL if interpreter == "python" else LIVECODEBENCH_PYPY3_GIT_URL
    cmd = f"{pip_cmd} install {git_url}"

    result, _ = await sandbox.execute_code(cmd, language="shell", timeout=300)
    if result.get("process_status") != "completed":
        LOG.warning(f"Failed to install livecodebench: {result.get('stderr', 'Unknown error')}")
        return False

    LOG.info("Successfully installed livecodebench.")
    return True


def _install_packages_locally(interpreter: str):
    """Installs packages on the local machine."""
    git_url = LIVECODEBENCH_PYTHON_GIT_URL if interpreter == "python" else LIVECODEBENCH_PYPY3_GIT_URL
    try:
        from livecodebench.evaluate import evaluate

        return evaluate  # Return the function if already installed
    except ImportError:
        LOG.info("Package 'livecodebench' not found. Attempting to install...")
        try:
            # Use the specified python interpreter for installation
            pip_executable = sys.executable if interpreter == "python" else interpreter
            subprocess.check_call([pip_executable, "-m", "pip", "install", git_url])
            LOG.info("Package installed successfully!")
            from livecodebench.evaluate import evaluate

            return evaluate
        except (subprocess.CalledProcessError, ImportError) as e:
            LOG.error(f"Failed to install/import 'livecodebench'. Please install it manually. Error: {e}")
            raise


async def eval_livecodebench_async(cfg, eval_config: LiveCodeBenchEvaluatorConfig):
    """Evaluation running within a sandbox."""
    async with sandbox_context(eval_config.sandbox) as sandbox:
        if not await _install_packages_in_sandbox(sandbox, eval_config.interpreter):
            return

        for jsonl_file in unroll_files(cfg.input_files):
            LOG.info(f"Processing file: {jsonl_file} in sandbox")
            try:
                samples, release_version = _preprocess_and_validate_file(jsonl_file, eval_config.language)
            except ValueError as e:
                LOG.error(f"Skipping {jsonl_file} due to pre-processing error: {e}")
                continue

            test_file_arg = repr(eval_config.test_file) if eval_config.test_file else "None"
            eval_code = textwrap.dedent(f"""
                from livecodebench.evaluate import evaluate
                evaluate(
                    custom_output_file='{jsonl_file}',
                    release_version='release_{release_version}',
                    test_file={test_file_arg},
                    k_list=[1],
                    language='{eval_config.language}',
                    num_process_evaluate={eval_config.num_processes},
                    timeout={eval_config.timeout}
                )
            """)

            cmd = f"{eval_config.interpreter} -c {shlex.quote(eval_code)}"
            output, _ = await sandbox.execute_code(
                cmd,
                language="shell",
                timeout=eval_config.timeout * len(samples) + 60,
                max_output_characters=100_000,
            )

            if output.get("process_status") != "completed":
                LOG.error(f"Evaluation failed for {jsonl_file}. Stderr: {output.get('stderr')}")
                continue

            _postprocess_results(jsonl_file, samples)


def eval_livecodebench_without_sandbox(cfg, eval_config: LiveCodeBenchEvaluatorConfig):
    """Evaluation running on the local machine."""
    evaluate_fn = _install_packages_locally(eval_config.interpreter)
    if not evaluate_fn:
        return

    for jsonl_file in unroll_files(cfg.input_files):
        LOG.info(f"Processing file: {jsonl_file} locally")
        try:
            samples, release_version = _preprocess_and_validate_file(jsonl_file, eval_config.language)
        except ValueError as e:
            LOG.error(f"Skipping {jsonl_file} due to pre-processing error: {e}")
            continue

        evaluate_fn(
            custom_output_file=jsonl_file,
            release_version=f"release_{release_version}",
            k_list=[1],
            language=eval_config.language,
            test_file=eval_config.test_file,
            num_process_evaluate=eval_config.num_processes,
            timeout=eval_config.timeout,
        )

        _postprocess_results(jsonl_file, samples)


def eval_livecodebench(cfg):
    """Main entry point for LiveCodeBench evaluation."""
    eval_config = LiveCodeBenchEvaluatorConfig(_init_nested=True, **cfg.eval_config)

    if eval_config.language == "python" and eval_config.interpreter not in ["python", "pypy3"]:
        raise ValueError("Python interpreter must be 'python' or 'pypy3'.")
    if eval_config.language == "cpp" and eval_config.test_file is None:
        raise ValueError("C++ evaluation requires a test_file.")

    # Simplified condition: run locally only for the standard python interpreter in a local "sandbox".
    # All other cases (pypy3, C++, remote sandboxes) use the async sandboxed path.
    use_sandbox = not (
        eval_config.sandbox.get("sandbox_type") == "local"
        and eval_config.language == "python"
        and eval_config.interpreter == "python"
    )

    if use_sandbox:
        asyncio.run(eval_livecodebench_async(cfg, eval_config))
    else:
        eval_livecodebench_without_sandbox(cfg, eval_config)
