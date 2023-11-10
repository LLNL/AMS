#!/bin/bash

#1) In the first run we do not have a model
if [ $# -ne 2 ]; then
  echo "Wrong command line. Please provide:"
  echo "$0 'TEMP DIR' 'DB DIR'"
  exit
fi

SRC=$1
db_path=$2

#We start clean
mkdir -p ${SRC}
rm -r ${db_path}/*

# Create an AMS database and the respective file-system support
python -m ams_wf.AMSStore create --path ${db_path} --name example

for i in {1..3}; do

  model=$(python -m ams_wf.AMSStore query --path ${db_path} -e models -f uri --version latest)

  echo "Current Model file is : ${model}"

  # We need to clean the temp files produced by the previous step
  rm -rf ${SRC}/*

  app_args=(
    -dt hdf5
    -db ${SRC}
    -d cpu
  )

  if [ ! -z ${model} ]; then
    app_args+=(-S)
    app_args+=(${model})
  fi

  echo "Arguments are: ${app_args[@]}"
  @CMAKE_CURRENT_BINARY_DIR@/ams_example ${app_args[@]}

  #2) We need to run the DBStage mechanism to pick up the intermediate data and move them into the kosh-store
  args=(
    -db ${db_path}
  #user options to use random pruner at the staging level
    --load @CMAKE_CURRENT_BINARY_DIR@/prune.py --class RandomPruneAction -f 0.001
  #Options to store the results in the candidates directory using a file format of h5
    --policy process -m fs  --dest ${db_path}/candidates --db-type dhdf5
    #Where to pick data from. AMS dumps data in sparse dataset thus we pass (shdf5)
    --src ${SRC} --src-type shdf5 --pattern "*.h5"
  #Make results public to store
  --store
  )

# Copy data from local storage, apply user defined trimming and move to candidates
  python -m ams_wf.AMSDBStage ${args[@]}

# Sub-select candidate data, store them into storage.
  python @CMAKE_CURRENT_BINARY_DIR@/sub_selection.py -db ${db_path}

#Train a new model and push it into the store
CUBLAS_WORKSPACE_CONFIG=:4096:8 python @CMAKE_CURRENT_BINARY_DIR@/train.py -db ${db_path} --device @TRAIN_DEVICE@

done
