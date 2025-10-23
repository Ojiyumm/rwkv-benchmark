# will run all tests starting from only a HF checkpoint. Only requires 1 GPU.
# also need to define HF_TOKEN for some of the tests
# model needs to be inside /mnt/datadrive/nemo-skills-test-data/Meta-Llama-3.1-8B-Instruct
# if you need to place it in a different location, modify test-local.yaml config
# example: HF_TOKEN=<> ./tests/gpu-tests/run.sh
set -e

export NEMO_SKILLS_TEST_HF_MODEL=/mnt/datadrive/nemo-skills-test-data/Meta-Llama-3.1-8B-Instruct
export NEMO_SKILLS_TEST_MODEL_TYPE=llama

# generation/evaluation tests
pytest tests/gpu-tests/test_eval.py -s -x
pytest tests/gpu-tests/test_generate.py -s -x
pytest tests/gpu-tests/test_context_retry.py -s -x
pytest tests/gpu-tests/test_judge.py -s -x
pytest tests/gpu-tests/test_run_cmd_llm_infer.py -s -x
pytest tests/gpu-tests/test_contamination.py -s -x

# for sft we are using the tiny random model to run much faster
ns run_cmd --cluster test-local --config_dir tests/gpu-tests --container nemo \
    python /nemo_run/code/tests/gpu-tests/make_tiny_llm.py --model_type $NEMO_SKILLS_TEST_MODEL_TYPE

# converting the model through test
export NEMO_SKILLS_TEST_HF_MODEL=/tmp/nemo-skills-tests/$NEMO_SKILLS_TEST_MODEL_TYPE/tiny-model-hf
# training tests
pytest tests/gpu-tests/test_train.py -s -x
