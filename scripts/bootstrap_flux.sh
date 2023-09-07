#!/usr/bin/env bash
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

usage="Usage: $(basename "$0") [#NODES] [JSON file] -- Script that bootstrap Flux on NNODES and writes Flux URIs to the JSON file."

function version() {
  echo "$@" | awk -F. '{ printf("%d%03d%03d%03d\n", $1,$2,$3,$4); }';
}

# Check if the allocation has the right size
# args:
#   - $1 : number of nodes requested for Flux
function check_main_allocation() {
  if [[ "$(flux getattr size)" -eq "$1" ]]; then
    echo "[$(date +'%m%d%Y-%T')@$(hostname)] Flux launch successful with $(flux getattr size) nodes"
  else
    echo "[$(date +'%m%d%Y-%T')@$(hostname)] Error: Requested nodes=$1 but Flux allocation size=$(flux getattr size)"
    exit 1
  fi
}

# Check if the 3 inputs are integers
function check_input_integers() {
  re='^[0-9]+$'
  if ! [[ $1 =~ $re && $2 =~ $re && $3 =~ $re ]] ; then
    echo "[$(date +'%m%d%Y-%T')@$(hostname)] Error: number of nodes is not an integer ($1, $2, $3)"
    exit 1
  fi
}

# Check if an allocation is running and if yes set the second parameter to the URI
#   - $1 : Flux Job ID
#   - $2 : the resulting URI
function check_allocation_running() {
  local JOBID="$1"
  local _result=$2
  local temp_uri=''
  # NOTE: with more recent versions of Flux, instead of sed here we could use flux jobs --no-header
  if [[ "$(flux jobs -o '{status}' $JOBID | sed -n '1!p')" == "RUN" ]]; then
    echo "[$(date +'%m%d%Y-%T')@$(hostname)] Job $JOBID is running"
    temp_uri=$(flux uri --remote $JOBID)
  else
    echo "[$(date +'%m%d%Y-%T')@$(hostname)] Warning: failed to launch job ($JOBID)"
  fi
  eval $_result="'$temp_uri'"
}

# Wait for a file to be created
#   - $1 : the file
#   - $2 : Max number of retry (one retry every 5 seconds)
function wait_for_file() {
  local FLUX_SERVER="$1"
  local EXIT_COUNTER=0
  local MAX_COUNTER="$2"
  while [ ! -f $FLUX_SERVER ]; do
    sleep 5s
    echo "[$(date +'%m%d%Y-%T')@$(hostname)] $FLUX_SERVER does not exist yet."
    exit_counter=$((EXIT_COUNTER + 1))
    if [ "$EXIT_COUNTER" -eq "$MAX_COUNTER" ]; then
      echo "[$(date +'%m%d%Y-%T')@$(hostname)] Timeout: Failed to find file (${FLUX_SERVER})."
      exit 1
    fi
  done
}

# ------------------------------------------------------------------------------
# the script needs the number of nodes for flux
FLUX_NODES="$1"
# JSON configuration for AMS that will get updated by this script
AMS_JSON="$2"
FLUX_SERVER="ams-uri.log"
FLUX_LOG="ams-flux.log"
# Flux-core Minimum version required by AMS
[[ -z ${MIN_VER_FLUX+z} ]] && MIN_VER_FLUX="0.45.0"

re='^[0-9]+$'
if ! [[ $FLUX_NODES =~ $re ]] ; then
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] ERROR: '$FLUX_NODES' is not a number."
  echo $usage  
  exit 1
fi

if ! [[ -f "$AMS_JSON" ]]; then
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] Error: $AMS_JSON does not exists."
  exit 1
fi

echo "[$(date +'%m%d%Y-%T')@$(hostname)] Launching Flux with $FLUX_NODES nodes"
echo "[$(date +'%m%d%Y-%T')@$(hostname)] Writing Flux configuration/URIs into $AMS_JSON"

unset FLUX_URI
export LC_ALL="C"
export FLUX_F58_FORCE_ASCII=1
export FLUX_SSH="ssh"
# Cleanup from previous runs
rm -f $FLUX_SERVER $FLUX_LOG

if ! [[ -x "$(command -v flux)" ]]; then
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] Error: flux is not installed."
  exit 1
fi
echo "[$(date +'%m%d%Y-%T')@$(hostname)] flux = $(which flux)"

flux_version=$(version $(flux version | awk '/^commands/ {print $2}'))
MIN_VER_FLUX_LONG=$(version ${MIN_VER_FLUX})
# We need to remove leading 0 because they are interpreted as octal numbers in bash
if [[ "${flux_version#00}" -lt "${MIN_VER_FLUX_LONG#00}" ]]; then
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] Error Flux $(flux version | awk '/^commands/ {print $2}') is not supported.\
  AMS requires flux>=${MIN_VER_FLUX}"
  exit 1
fi

echo "[$(date +'%m%d%Y-%T')@$(hostname)] flux version"
flux version

# We create a Flux wrapper around sleep on the fly to get the main Flux URI
FLUX_SLEEP_WRAPPER="./$(mktemp flux-wrapper.XXXX.sh)"
cat << 'EOF' > $FLUX_SLEEP_WRAPPER
#!/usr/bin/env bash
echo "ssh://$(hostname)$(flux getattr local-uri | sed -e 's!local://!!')" > "$1"
sleep inf
EOF
chmod u+x $FLUX_SLEEP_WRAPPER

MACHINE=$(echo $HOSTNAME | sed -e 's/[0-9]*$//')
if [[ "$MACHINE" == "lassen" ]] ; then
  # To use module command we must source this file
  # Those options are needed on IBM machines (CORAL)
  # Documented: https://flux-framework.readthedocs.io/en/latest/tutorials/lab/coral.html
  source /etc/profile.d/z00_lmod.sh
  module use /usr/tce/modulefiles/Core
  module use /usr/global/tools/flux/blueos_3_ppc64le_ib/modulefiles
  module load pmi-shim

  PMIX_MCA_gds="^ds12,ds21" \
    jsrun -a 1 -c ALL_CPUS -g ALL_GPUS -n ${FLUX_NODES} \
      --bind=none --smpiargs="-disable_gpu_hooks" \
      flux start -o,-S,log-filename=$FLUX_LOG -v $FLUX_SLEEP_WRAPPER $FLUX_SERVER &
elif [[ "$MACHINE" == "pascal" || "$MACHINE" == "ruby" ]] ; then
    srun -n ${FLUX_NODES} -N ${FLUX_NODES} --pty --mpi=none --mpibind=off \
      flux start -o,-S,log-filename=$FLUX_LOG -v $FLUX_SLEEP_WRAPPER $FLUX_SERVER &
else
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] machine $MACHINE is not supported at the moment."
  exit 1
fi

echo ""
# now, wait for the flux info file
# we retry 20 times (one retry every 5 seconds)
wait_for_file $FLUX_SERVER 20
export FLUX_URI=$(cat $FLUX_SERVER)
echo "[$(date +'%m%d%Y-%T')@$(hostname)] Run: export FLUX_URI=$(cat $FLUX_SERVER)"
check_main_allocation ${FLUX_NODES}

# Read configuration file with number of nodes/cores for each sub allocations
NODES_PHYSICS=$(jq ".physics.nodes" $AMS_JSON)
NODES_ML=$(jq ".ml.nodes" $AMS_JSON)
NODES_CONTAINERS=$(jq ".containers.nodes" $AMS_JSON)
check_input_integers $NODES_PHYSICS $NODES_ML $NODES_CONTAINERS 

CORES_PHYSICS=$(jq ".physics.cores" $AMS_JSON)
CORES_ML=$(jq ".ml.cores" $AMS_JSON)
CORES_CONTAINERS=$(jq ".containers.cores" $AMS_JSON)
check_input_integers $CORES_PHYSICS $CORES_ML $CORES_CONTAINERS 

GPUS_PHYSICS=$(jq ".physics.gpus" $AMS_JSON)
GPUS_ML=$(jq ".ml.gpus" $AMS_JSON)
GPUS_CONTAINERS=$(jq ".containers.gpus" $AMS_JSON)
check_input_integers $GPUS_PHYSICS $GPUS_ML $GPUS_CONTAINERS 

# Partition resources for physics, ML and containers (RabbitMQ, filtering)
# NOTE: with more recent Flux (>=0.46), we could use flux alloc --bg instead
JOBID_PHYSICS=$(
  flux mini batch --job-name="ams-physics" \
    --output="ams-physics-{{id}}.log" \
    --exclusive \
    --nslots=1 --nodes=$NODES_PHYSICS \
    --cores-per-slot=$CORES_PHYSICS \
    --gpus-per-slot=$GPUS_PHYSICS \
    --wrap sleep inf
)
sleep 2s
check_allocation_running $JOBID_PHYSICS FLUX_PHYSICS_URI

JOBID_ML=$(
  flux mini batch --job-name="ams-ml" \
    --output="ams-ml-{{id}}.log" \
    --exclusive \
    --nslots=1 --nodes=$NODES_ML\
    --cores-per-slot=$CORES_ML \
    --gpus-per-slot=$GPUS_ML \
    --wrap sleep inf
)
sleep 2s
check_allocation_running $JOBID_ML FLUX_ML_URI

JOBID_CONTAINERS=$(
  flux mini batch --job-name="ams-containers" \
    --output="ams-containers-{{id}}.log" \
    --nslots=1 --nodes=$NODES_CONTAINERS \
    --cores-per-slot=$CORES_CONTAINERS \
    --gpus-per-slot=$GPUS_CONTAINERS \
    --wrap sleep inf
)
sleep 2s
check_allocation_running $JOBID_CONTAINERS FLUX_CONTAINERS_URI

# Add all URIs to existing AMS JSON file
AMS_JSON_BCK=${AMS_JSON}.bck
cp -f $AMS_JSON $AMS_JSON_BCK
jq '. += {flux:{}}' $AMS_JSON > $AMS_JSON_BCK && cp $AMS_JSON_BCK $AMS_JSON
jq --arg var "$(id -u)" '.flux += {"uid":$var}' $AMS_JSON > $AMS_JSON_BCK && cp $AMS_JSON_BCK $AMS_JSON
jq --arg flux_uri "$FLUX_URI" '.flux += {"global_uri":$flux_uri}' $AMS_JSON > $AMS_JSON_BCK && cp $AMS_JSON_BCK $AMS_JSON
jq --arg flux_uri "$FLUX_PHYSICS_URI" '.flux += {"physics_uri":$flux_uri}' $AMS_JSON > $AMS_JSON_BCK && cp $AMS_JSON_BCK $AMS_JSON
jq --arg flux_uri "$FLUX_ML_URI" '.flux += {"ml_uri":$flux_uri}' $AMS_JSON > $AMS_JSON_BCK && cp $AMS_JSON_BCK $AMS_JSON
jq --arg flux_uri "$FLUX_CONTAINERS_URI" '.flux += {"container_uri":$flux_uri}' $AMS_JSON > $AMS_JSON_BCK && cp $AMS_JSON_BCK $AMS_JSON

# We move the file only if jq is sucessful otherwise jq will likey erase the original file
if [[ "$?" -eq 0 ]]; then
  mv -f $AMS_JSON_BCK $AMS_JSON && rm -f $AMS_JSON_BCK
fi
