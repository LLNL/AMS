#!/usr/bin/env bash
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

usage="Usage: $(basename "$0") [JSON file] [USER (optional)] -- Script that adds the JSON file as secrets in OpenShift"
#TODO: Only for LC (add it somewhere as config value)
export PATH="$PATH:/usr/global/openshift/bin/"

check_cmd() {
  err=$($@ 2>&1)
  if [ $? -ne 0 ]; then
    echo "[$(date +'%m%d%Y-%T')@$(hostname)] error: $err"
    # if [[ -x "$(command -v oc)" ]]; then
    #   oc logout
    # fi
    exit 1
  else
    echo $err
  fi
}

AMS_JSON="$1"
USER="$2"
[ -z "$2" ] && USER=$(whoami) # If no argument $2 we take the default user
URL="https://api.czapps.llnl.gov"
PORT=6443
PROJECT_NAME="cz-amsdata"
RMQ_CREDS="creds.json"
SECRET="rabbitmq-creds"

if ! [[ -f "$AMS_JSON" ]]; then
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] Error: config file \"$AMS_JSON\" does not exists."
  exit 1
fi

if ! [[ -x "$(command -v oc)" ]]; then
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] Error: OpenShift (oc) not found."
  exit 1
fi
echo "[$(date +'%m%d%Y-%T')@$(hostname)] oc = $(which oc)"

RMQ_CONFIG=$(jq ".rabbitmq" $AMS_JSON)
echo "$RMQ_CONFIG" > $RMQ_CREDS

echo "[$(date +'%m%d%Y-%T')@$(hostname)] Login in ${URL}:${PORT} as ${USER}"
oc login --insecure-skip-tls-verify=true --server=${URL}:${PORT} -u ${USER}
# Warning: Do not use function check_cmd to wrap oc login here (it will block oc login)
if [[ "$?" -ne 0 ]]; then
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] Error while connecting to OpenShift."
  exit 1
fi
echo "[$(date +'%m%d%Y-%T')@$(hostname)] Logged in as $(oc whoami), switching to project ${PROJECT_NAME}"
check_cmd oc project $PROJECT_NAME

err=$(oc create secret generic $SECRET --from-file=$RMQ_CREDS 2>&1)

if [[ "$?" -ne 0 ]]; then
  check_cmd echo $err | grep "already exists"
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] secret already exists, we are updating it."
  check_cmd oc delete secret $SECRET
  check_cmd oc create secret generic $SECRET --from-file=$RMQ_CREDS
else
  check_cmd oc get secrets $SECRET
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] Added secrets successfully."
fi

# check_cmd oc logout