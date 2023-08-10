#!/usr/bin/env bash

usage="Usage: $(basename "$0") [NNODES] [JSON file] -- Script that bootstrap Flux on NNODES and writes Flux URIs to the JSON file."

# the script needs the number of nodes for flux
FLUX_NODES="$1"
# JSON configuration for AMS that will get updated by this script
AMS_JSON="$2"
FLUX_SERVER="ams-uri.log"
FLUX_LOG="ams-flux.log"

re='^[0-9]+$'
if ! [[ $FLUX_NODES =~ $re ]] ; then
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] ERROR: '$FLUX_NODES' is not a number."
  echo $usage  
  exit 1
fi


echo "[$(date +'%m%d%Y-%T')@$(hostname)] Launching Flux with $FLUX_NODES nodes"
echo "[$(date +'%m%d%Y-%T')@$(hostname)] Writing Flux configuration/URIs into $AMS_JSON"

unset FLUX_URI
export LC_ALL="C"
export FLUX_F58_FORCE_ASCII=1
export FLUX_SSH="ssh"

# CLeanup
rm -f $FLUX_SERVER $FLUX_LOG

if ! [[ -x "$(command -v flux)" ]]; then
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] Error: flux is not installed."
  exit 1
fi

echo "[$(date +'%m%d%Y-%T')@$(hostname)] flux = $(which flux)"
echo "[$(date +'%m%d%Y-%T')@$(hostname)] flux version"
flux version

# To use module command we must source this file
# Those options are needed on IBM machines (CORAL)
# Documented: https://flux-framework.readthedocs.io/en/latest/tutorials/lab/coral.html
source /etc/profile.d/z00_lmod.sh
module use /usr/tce/modulefiles/Core
module use /usr/global/tools/flux/blueos_3_ppc64le_ib/modulefiles
module load pmi-shim

# We create a Flux wrapper around sleep on the fly to get the main Flux URI
FLUX_SLEEP_WRAPPER="./temp-flux.sh"
cat << 'EOF' > $FLUX_SLEEP_WRAPPER
#!/usr/bin/env bash
echo "ssh://$(hostname)$(flux getattr local-uri | sed -e 's!local://!!')" > "$1"
sleep inf
EOF
chmod u+x $FLUX_SLEEP_WRAPPER

PMIX_MCA_gds="^ds12,ds21" \
    jsrun -a 1 -c ALL_CPUS -g ALL_GPUS -n ${FLUX_NODES} \
    --bind=none --smpiargs="-disable_gpu_hooks" \
    flux start -o,-S,log-filename=$FLUX_LOG -v $FLUX_SLEEP_WRAPPER $FLUX_SERVER &
echo ""

# ------------------------------------------------------------------------------
# now, wait for the flux info file
EXIT_COUNTER=0
MAX_COUNTER=20
while [ ! -f $FLUX_SERVER ]; do
  sleep 5s
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] $FLUX_SERVER does not exist yet."
  exit_counter=$((EXIT_COUNTER + 1))
  if [ "$EXIT_COUNTER" -eq "$MAX_COUNTER" ]; then
    echo "[$(date +'%m%d%Y-%T')@$(hostname)] Timeout: Failed to find file (${FLUX_SERVER})."
    exit 1
  fi
done

export FLUX_URI=$(cat $FLUX_SERVER)
echo "[$(date +'%m%d%Y-%T')@$(hostname)] Flux launch successful with $(flux getattr size) nodes"
echo "[$(date +'%m%d%Y-%T')@$(hostname)] You can run: export FLUX_URI=$(cat $FLUX_SERVER)"

rm $FLUX_SLEEP_WRAPPER

# Read configuration file with number of nodes/cores for each sub allocations
# TODO: add number of cores/GPUs for each
NNODES_PHYSICS=$(jq ".physics_nodes" $AMS_JSON)
NNODES_ML=$(jq ".ml_nodes" $AMS_JSON)
NNODES_CONTAINERS=$(jq ".containers_nodes" $AMS_JSON)
re='^[0-9]+$'
if ! [[ $NNODES_PHYSICS =~ $re && $NNODES_ML =~ $re && $NNODES_CONTAINERS =~ $re ]] ; then
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] Error: number of nodes is not an integer (${NNODES_PHYSICS}, ${NNODES_ML}, ${NNODES_CONTAINERS})"
  echo $usage
  exit 1
fi

# Partition resources for physics, ML and containers (RabbitMQ, filtering)
JOBID_PHYSICS=$(
    flux mini batch --job-name="ams-physics" \
    --output="ams-physics-{{id}}.log" \
    --nslots=1 --cores-per-slot=1 --nodes=$NNODES_PHYSICS \
    --wrap sleep inf
)
FLUX_PHYSICS_URI=$(flux uri --remote $JOBID_PHYSICS)

JOBID_ML=$(
    flux mini batch --job-name="ams-ml" \
    --output="ams-ml-{{id}}.log" \
    --nslots=1 --cores-per-slot=1 --nodes=$NNODES_ML \
    --wrap sleep inf
)
FLUX_ML_URI=$(flux uri --remote $JOBID_ML)

JOBID_CONTAINERS=$(
    flux mini batch --job-name="ams-containers" \
    --output="ams-containers-{{id}}.log" \
    --nslots=1 --cores-per-slot=1 --nodes=$NNODES_CONTAINERS \
    --wrap sleep inf
)
FLUX_CONTAINERS_URI=$(flux uri --remote $JOBID_CONTAINERS)

# Add all URIs to existing AMS JSON file
AMS_JSON_BCK=${AMS_JSON}.bck
cp -f $AMS_JSON $AMS_JSON_BCK
jq '. += {flux:{}}' $AMS_JSON > $AMS_JSON_BCK && cp $AMS_JSON_BCK $AMS_JSON
jq --arg flux_uri "$FLUX_URI" '.flux += {"FLUX_URI":$flux_uri}' $AMS_JSON > $AMS_JSON_BCK && cp $AMS_JSON_BCK $AMS_JSON
jq --arg flux_uri "$FLUX_PHYSICS_URI" '.flux += {"FLUX_PHYSICS_URI":$flux_uri}' $AMS_JSON > $AMS_JSON_BCK && cp $AMS_JSON_BCK $AMS_JSON
jq --arg flux_uri "$FLUX_ML_URI" '.flux += {"FLUX_ML_URI":$flux_uri}' $AMS_JSON > $AMS_JSON_BCK && cp $AMS_JSON_BCK $AMS_JSON
jq --arg flux_uri "$FLUX_CONTAINERS_URI" '.flux += {"FLUX_CONTAINERS_URI":$flux_uri}' $AMS_JSON_BCK > $AMS_JSON && cp $AMS_JSON_BCK $AMS_JSON

if [[ $? -ne 0 ]]; then
  mv -f $AMS_JSON_BCK $AMS_JSON
fi

cat $AMS_JSON
