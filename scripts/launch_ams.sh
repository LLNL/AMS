#!/usr/bin/env bash
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

usage="Usage: $(basename "$0") [#NODES] [JSON file] -- Launch the AMS workflow on N nodes based on JSON configuration file."

if [ -z ${AMS_ROOT+x} ]; then
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] Please export AMS_ROOT to where AMS repository is located."
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] with: export AMS_ROOT=<AMS repo path>"
  exit 1
fi

# the script needs the number of nodes for flux
FLUX_NODES="$1"
# JSON configuration for AMS that will get updated by this script
AMS_JSON="$2"
UPDATE_SECRETS="${AMS_ROOT}/scripts/rmq_add_secrets.sh"
# Ssh bridge could be needed if OpenShift is not reachable from every clusters.
SSH_BRIDGE="quartz"
BOOTSTRAP="${AMS_ROOT}/scripts/bootstrap_flux.sh"
START_PHYSICS="${AMS_ROOT}/scripts/launch_physics.sh"

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

# Get the absolute path
AMS_JSON=$(realpath "$2")

# Flux-core Minimum version required by AMS
export MIN_VER_FLUX="0.45.0"
export LC_ALL="C"
export FLUX_F58_FORCE_ASCII=1
export FLUX_SSH="ssh"

USE_DB=$(jq -r ".ams_app.use_db" $AMS_JSON)
DBTYPE=$(jq -r ".ams_app.dbtype" $AMS_JSON)
# 1. If we use RabbitMQ and asynchronous traning we add needed secrets OpenShift so AMS daemon can connect to RabbitMQ
# Note:
#   This step might fail if you have not logged in OpenShift already.
#   If that step fails, please try to log in OC with the following command
#   oc login --insecure-skip-tls-verify=true --server=https://api.czapps.llnl.gov:6443 -u $(whoami)
if [[ $USE_DB && $DBTYPE = "rmq" ]]; then
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] Trying to update secrets on OpenShift"
  ssh ${SSH_BRIDGE} bash <<-EOF
  $UPDATE_SECRETS $AMS_JSON
EOF
fi

echo "[$(date +'%m%d%Y-%T')@$(hostname)] Starting bootstraping Flux on $FLUX_NODES"
# 2. We bootstrap Flux on FLUX_NODES nodes
$BOOTSTRAP $FLUX_NODES $AMS_JSON

RMQ_TMP="rmq.json"
CERT_TLS="rmq.cert"
# This require to install the AMS python package
AMS_BROKER_EXE="AMSBroker"

# 3. We send the current UID and the Flux ML URI to the AMS daemon listening
if [[ $USE_DB && $DBTYPE = "rmq" ]]; then
  RMQ_CONFIG=$(jq ".rabbitmq" $AMS_JSON)
  echo $RMQ_CONFIG > $RMQ_TMP
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] Extracted RabbitMQ configuration to ${RMQ_TMP}"

  REMOTE_HOST=$(jq -r '.rabbitmq."service-host"' $AMS_JSON)
  REMOTE_PORT=$(jq -r '.rabbitmq."service-port"' $AMS_JSON)
  openssl s_client -connect ${REMOTE_HOST}:${REMOTE_PORT} -showcerts < /dev/null 2>/dev/null | sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p' > ${CERT_TLS}
  if [[ "$?" -eq 0 ]]; then
    echo "[$(date +'%m%d%Y-%T')@$(hostname)] Successfuly generated TLS certificate written in ${CERT_TLS}"
  else
    echo "[$(date +'%m%d%Y-%T')@$(hostname)] Error during TLS certificate generation"
    exit 1
  fi
  AMS_DAEMON_QUEUE=$(jq -r '.daemon."queue-training-init"' $AMS_JSON)
  AMS_UID=$(id -u)
  AMS_ML_URI=$(jq -r '.flux.ml_uri' $AMS_JSON)
  # Warning: there should be no whitespace in the message
  MSG="{\"uid\":${AMS_UID},\"ml_uri\":\"${AMS_ML_URI}\"}"
  ${AMS_BROKER_EXE} -c ${RMQ_TMP} -t ${CERT_TLS} -q ${AMS_DAEMON_QUEUE} -s $MSG
  if [[ "$?" -eq 0 ]]; then
    echo "[$(date +'%m%d%Y-%T')@$(hostname)] Successfuly sent message ${MSG} to queue ${AMS_DAEMON_QUEUE}"
  else
    echo "[$(date +'%m%d%Y-%T')@$(hostname)] Error: message did not get send to RabbitMQ"
    exit 1
  fi
fi

# 4. We start the physics code 
$START_PHYSICS $AMS_JSON
echo "[$(date +'%m%d%Y-%T')@$(hostname)] AMS workflow is ready to run!"
