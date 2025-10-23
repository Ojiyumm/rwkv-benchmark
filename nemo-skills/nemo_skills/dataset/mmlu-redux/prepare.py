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

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from nemo_skills.dataset.utils import get_mcq_fields

# mmlu subcategories from https://github.com/hendrycks/test/blob/master/categories.py
subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}


# dataset preparing strategy adapted from ZeroEval (https://github.com/WildEval/ZeroEval/blob/main/data_prep/mmlu-redux.py)
def format_entry(entry, category):
    if entry["error_type"] == "ok":
        final_answer = chr(65 + entry["answer"])
    elif entry["error_type"] == "wrong_groundtruth" and entry["correct_answer"] in list("ABCD"):
        final_answer = "correct_answer"
    else:
        # bad_question_clarity, bad_options_clarity, no_correct_answer,
        # multiple_correct_answers, expert and wrong_groundtruth with no labels
        return None
    return {
        "expected_answer": final_answer,
        "subset_for_metrics": subcategories[category][0],
        "subcategory": category,
        "source": entry["source"],
        **get_mcq_fields(entry["question"], entry["choices"]),
    }


def write_data_to_file(output_file, data, category):
    with open(output_file, "at", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {category} ({subcategories[category][0]}) to {output_file.name}"):
            if final_entry := format_entry(entry, category):
                json.dump(final_entry, fout)
                fout.write("\n")


def main(args):
    # Create the output directory if it doesn't exist
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)

    print(f"Loading categories: {list(subcategories.keys())}")

    # create output_file or remove its contents if it exists
    output_file = data_dir / f"{args.split}.jsonl"
    open(output_file, "w")

    # Load the dataset and write it to the output
    for category in tqdm(subcategories):
        dataset = load_dataset("edinburgh-dawg/mmlu-redux-2.0", name=category, split="test")
        write_data_to_file(output_file, dataset, category)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=(["test"]), help="Dataset split to process.")
    args = parser.parse_args()
    main(args)
