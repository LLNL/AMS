#!/usr/bin/env bash

usage="Usage: $(basename "$0") [JSON file] [USER (optional)] -- Script that adds the JSON file as secrets in OpenShift"

export PATH="$PATH:/usr/global/openshift/bin/"

AMS_JSON="$1"
USER="$2"
[ -z "$2" ] && USER=$(whoami) # If no argument $2 we take the default user
URL="https://api.czapps.llnl.gov"
PORT=6443
PROJECT_NAME="cz-amsdata"
RMQ_CREDS="creds.json"
SECRET="rabbitmq-creds"

if ! [[ -f "$AMS_JSON" ]]; then
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] Error: $AMS_JSON does not exists."
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
echo "[$(date +'%m%d%Y-%T')@$(hostname)] Switching to project ${PROJECT_NAME}"
oc project $PROJECT_NAME

err=$(oc create secret generic $SECRET --from-file=$RMQ_CREDS 2>&1)

if [[ "$?" -ne 0 ]]; then
  test_err=$(echo $err | grep "already exists")
  # Unknown error so we exit
  if [[ "$?" -ne 0 ]]; then
    echo "[$(date +'%m%d%Y-%T')@$(hostname)] Error while adding secret: $err"
    exit 1
  fi
  echo "[$(date +'%m%d%Y-%T')@$(hostname)] secret already exists, we are updating it."
  err=$(oc delete secret $SECRET 2>&1)
  if [[ "$?" -ne 0 ]]; then
    echo "[$(date +'%m%d%Y-%T')@$(hostname)] Error while deleting secret: $err"
    exit 1
  fi
  err=$(oc create secret generic $SECRET --from-file=$RMQ_CREDS 2>&1)
  if [[ "$?" -ne 0 ]]; then
    echo "[$(date +'%m%d%Y-%T')@$(hostname)] Error while adding secret: $err"
    exit 1
  fi
fi
