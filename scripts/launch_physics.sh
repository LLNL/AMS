#!/usr/bin/env bash
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

usage="Usage: $(basename "$0") [JSON file] -- Script that launch AMS based on settings defined in the JSON file."
function version { echo "$@" | awk -F. '{ printf("%d%03d%03d%03d\n", $1,$2,$3,$4); }'; }

AMS_JSON="$1"
if ! [[ -f "$AMS_JSON" ]]; then
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] Error: $AMS_JSON does not exists."
  exit 1
fi

# Flux-core Minimum version required by AMS
[[ -z ${MIN_VER_FLUX+z} ]] && MIN_VER_FLUX="0.45.0"

export LC_ALL="C"
export FLUX_F58_FORCE_ASCII=1
export FLUX_SSH="ssh"

# We check that Flux exist
if ! [[ -x "$(command -v flux)" ]]; then
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] Error: flux is not installed."
  exit 1
fi

flux_version=$(version $(flux version | awk '/^commands/ {print $2}'))
MIN_VER_FLUX_LONG=$(version ${MIN_VER_FLUX})
# We need to remove leading 0 because they are interpreted as octal numbers in bash
if [[ "${flux_version#00}" -lt "${MIN_VER_FLUX_LONG#00}" ]]; then
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] Error: Flux $(flux version | awk '/^commands/ {print $2}') is not supported.\
  AMS requires flux>=${MIN_VER_FLUX}"
  exit 1
fi

# The -r flag is important here to remove quotes around results from jq which confuse Flux
FLUX_URI=$(jq -r ".flux.global_uri" $AMS_JSON)
PHYSICS_URI=$(jq -r ".flux.physics_uri" $AMS_JSON)
NODES_PHYSICS=$(jq ".physics.nodes" $AMS_JSON)
EXEC=$(jq -r ".ams_app.executable" $AMS_JSON)
ML_PATH=$(jq -r ".ams_app.modelpath" $AMS_JSON)
USE_GPU=$(jq -r ".ams_app.use_gpu" $AMS_JSON)
USE_DB=$(jq -r ".ams_app.use_db" $AMS_JSON)
DBTYPE=$(jq -r ".ams_app.dbtype" $AMS_JSON)
MPI_RANKS=$(jq -r ".ams_app.mpi_ranks" $AMS_JSON)
# -1 for all debug messages, 0 for no debug messages
VERBOSE=$(jq -r ".ams_app.verbose" $AMS_JSON)

AMS_ARGS="-S ${ML_PATH}"

if $USE_GPU; then
  AMS_ARGS="${AMS_ARGS} -d cuda"
fi

if $USE_DB; then
  OUTPUTS="output"
  if [[ $DBTYPE = "csv" || $DBTYPE = "hdf5" ]]; then
    mkdir -p $OUTPUTS
  elif [[ $DBTYPE = "rmq" ]]; then
    RMQ_CONFIG=$(jq ".rabbitmq" $AMS_JSON)
    # We have to write that JSON for AMS app to work (AMS does not read from stdin)
    OUTPUTS="${OUTPUTS}.json"
    echo $RMQ_CONFIG > $OUTPUTS
  fi
  AMS_ARGS="${AMS_ARGS} -dt ${DBTYPE} -db ${OUTPUTS}"
fi

echo "[$(date +'%m%d%Y-%T')@$(hostname)] Launching AMS on ${NODES_PHYSICS} nodes"
echo "[$(date +'%m%d%Y-%T')@$(hostname)] AMS binary         = ${EXEC}"
echo "[$(date +'%m%d%Y-%T')@$(hostname)] AMS verbose level  = ${VERBOSE}"
echo "[$(date +'%m%d%Y-%T')@$(hostname)] AMS Arguments      = ${AMS_ARGS}"
echo "[$(date +'%m%d%Y-%T')@$(hostname)] MPI ranks          = ${MPI_RANKS}"
echo "[$(date +'%m%d%Y-%T')@$(hostname)]   > Cores/rank = 1"
echo "[$(date +'%m%d%Y-%T')@$(hostname)]   > GPUs/rank  = 1"

ams_jobid=$(
  LIBAMS_VERBOSITY_LEVEL=${VERBOSE} FLUX_URI=$PHYSICS_URI flux mini submit \
    --job-name="ams-app" \
    -N ${NODES_PHYSICS} -n $MPI_RANKS -c 1 -g 1 \
    -o mpi=spectrum -o cpu-affinity=per-task -o gpu-affinity=per-task \
    ${EXEC} ${AMS_ARGS}
)
echo "[$(date +'%m%d%Y-%T')@$(hostname)] Launched job $ams_jobid"
echo "[$(date +'%m%d%Y-%T')@$(hostname)] To debug: FLUX_URI=$PHYSICS_URI flux job attach $ams_jobid"
