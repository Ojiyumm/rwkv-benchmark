#!/bin/bash

# schedule with chron on a machine with corresponding cluster config setup, e.g. to run each Sunday at 5am
# 0 5 * * 0 HF_TOKEN=<> WANDB_API_KEY=<> NEMO_SKILLS_CONFIG_DIR=<path to the configs dir> <path to a copy of this script> <cluster name> > /tmp/nemo-skills-slurm-cron.log 2>&1
# the metrics will be logged in w&b and you will get emails on failure as long as it's configured in your config

LOCAL_WORKSPACE=/tmp/nemo-skills-slurm-ci

rm -rf $LOCAL_WORKSPACE
mkdir -p $LOCAL_WORKSPACE
cd $LOCAL_WORKSPACE
git clone https://github.com/NVIDIA/NeMo-Skills.git
cd NeMo-Skills

curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=$LOCAL_WORKSPACE sh
$LOCAL_WORKSPACE/uv venv .venv --python 3.10
source .venv/bin/activate
$LOCAL_WORKSPACE/uv pip install -e .

./tests/slurm-tests/run_all.sh $1
