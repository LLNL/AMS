#!/usr/bin/env bash
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

usage="Usage: $(basename "$0") [#NODES] [JSON file] -- Launch the AMS workflow on N nodes based on JSON configuration file."

# the script needs the number of nodes for flux
FLUX_NODES="$1"
# JSON configuration for AMS that will get updated by this script
AMS_JSON="$2"
UPDATE_SECRETS=rmq_add_secrets.sh
SSH_BRIDGE="ssh quartz"
BOOTSTRAP=bootstrap_flux.sh
START_PHYSICS=launch_physics.sh

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

# Flux-core Minimum version required by AMS
export MIN_VER_FLUX="0.45.0"
export LC_ALL="C"
export FLUX_F58_FORCE_ASCII=1
export FLUX_SSH="ssh"

USE_DB=$(jq -r ".ams_app.use_db" $AMS_JSON)
DBTYPE=$(jq -r ".ams_app.dbtype" $AMS_JSON)

if [[ $USE_DB && $DBTYPE = "rmq" ]]; then
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] Trying to update secrets on OpenShift"
  $SSH_BRIDGE $UPDATE_SECRETS $AMS_JSON
fi

$BOOTSTRAP $FLUX_NODES $AMS_JSON

#TODO: send data to RabbitMQ here

$START_PHYSICS $AMS_JSON

echo "[$(date +'%m%d%Y-%T')@$(hostname)] AMS workflow is ready!"
