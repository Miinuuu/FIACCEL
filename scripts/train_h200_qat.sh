#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
training_root="${FIACCEL_TRAIN_ROOT:-/data/jmw/fiaccel}"
python_bin="${FIACCEL_PYTHON:-${training_root}/.venv/bin/python}"
run_dir="${FIACCEL_RUN_DIR:-${training_root}/runs/qat_$(date -u +%Y%m%d_%H%M%SZ)}"
mkdir -p "$run_dir"
resume_args=()
source_dir="$run_dir/source"
if [[ "${FIACCEL_RESUME:-0}" == 1 ]]; then
    if [[ ! -f "$run_dir/checkpoints_int/last.ckpt" ]]; then
        echo "Resume checkpoint not found in $run_dir/checkpoints_int" >&2
        exit 1
    fi
    source_dir="$run_dir/source_resume_$(date -u +%Y%m%d_%H%M%SZ)"
    resume_args=(checkpoint.resume_training=true)
elif [[ -e "$source_dir" ]]; then
    echo "Run directory already has a source snapshot: $run_dir" >&2
    exit 1
fi
mkdir "$source_dir"
cp -a "$project_root/config" "$project_root/model" "$project_root/datamodules" \
    "$project_root/utils" "$project_root/model_train.py" \
    "$project_root/model_pipeline.py" "$project_root/model_lightning.py" "$source_dir/"
if [[ -d "$run_dir/.hydra" ]]; then
    cp -a "$run_dir/.hydra" "$source_dir/previous_hydra"
fi
rm -f "$run_dir/exit_code"
exec > >(tee -a "$run_dir/train.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit_code"' EXIT
printf '%s\n' "Run: $run_dir" "Started (UTC): $(date -u --iso-8601=seconds)"
printf '%s\n' "$$" > "$run_dir/launcher.pid"
"$python_bin" -m pip freeze > "$run_dir/requirements.freeze.txt"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=offline
cd "$source_dir"
"$python_bin" -u model_train.py --config-name train_config_h200_qat \
    "hydra.run.dir=$run_dir" "${resume_args[@]}" "$@"
