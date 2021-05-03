#!/bin/sh
data=${1}
out=${2}
ml_path=${3}
venv_path=${4}
action=${5}
model=${6}
cd ${ml_path}
PIPENV_VENV_IN_PROJECT=1 pipenv install
. ${venv_path}/bin/activate
python -m CNN -d ${data} -o ${out} -a ${action} -l ${model}
