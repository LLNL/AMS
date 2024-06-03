import sys
import json
from pathlib import Path
import pandas as pd
import h5py
import numpy as np
import os


def get_suffix(db_type):
    if db_type == "csv":
        return "csv"
    if db_type == "none":
        return "none"
    if db_type == "hdf5":
        return "h5"
    return "unknown"


def verify_data_collection(fs_path, db_type, num_inputs, num_outputs, name="test", debug_db=False):
    # Returns a tuple of the input/ouput data and 0/1 for correct incorrect file.
    # Checks whether the files also have the right number of columns
    if not Path(fs_path).exists():
        print("Expecting output directory to exist")
        return None, 1

    suffix = get_suffix(db_type)
    if suffix == "none":
        return None, 0

    fn = f"{name}_0.{suffix}"
    if debug_db and db_type != "hdf5":
        print("Debug DB is only supported on hdf5")
        return None, 1
    elif debug_db:
        fn = f"{name}_0.debug.{suffix}"

    fp = Path(f"{fs_path}/{fn}")

    if name == "" and fp.exists():
        print("I was expecting file to not exist")
        fp.unlink()
        return None, 1
    elif name == "":
        return (np.empty((0, 0)), np.empty((0, 0))), 0
    elif not fp.exists():
        print(f"File path {fn} does not exist")
        return None, 1

    if db_type == "csv":
        df = pd.read_csv(str(fp), sep=":")
        assert len(df.columns) == (num_inputs + num_outputs), "Expected equal number of inputs/outputs"

        inputs = sum(1 for s in df.columns if "input" in s)
        assert inputs == num_inputs, "Expected equal number of inputs"
        outputs = sum(1 for s in df.columns if "output" in s)
        assert outputs == num_outputs, "Expected equal number of outputs"
        input_data = df[[f"input_{i}" for i in range(inputs)]].to_numpy()
        output_data = df[[f"output_{i}" for i in range(outputs)]].to_numpy()
        fp.unlink()
        return (input_data, output_data), 0
    elif db_type == "hdf5":
        with h5py.File(fp, "r") as fd:
            dsets = fd.keys()
            assert len(dsets) == (num_inputs + num_outputs + int(debug_db)), "Expected equal number of inputs/outputs"
            inputs = sum(1 for s in dsets if "input" in s)
            assert inputs == num_inputs, "Expected equal number of inputs"
            outputs = sum(1 for s in dsets if "output" in s)
            assert outputs == num_outputs, "Expected equal number of outputs"
            input_data = [[] for _ in range(num_inputs)]
            output_data = [[] for _ in range(num_outputs)]
            for d in dsets:
                if d == "predicate":
                    continue
                loc = int(d.split("_")[1])
                if len(fd[d]):
                    if "input" in d:
                        input_data[loc] = fd[d][:]
                    elif "output" in d:
                        output_data[loc] = fd[d][:]
            predicate = None
            if debug_db:
                predicate = np.array(fd["predicate"][:])
            input_data = np.array(input_data)
            output_data = np.array(output_data)
            fp.unlink()
            if debug_db:
                return (input_data.T, output_data.T, predicate), 0
            return (input_data.T, output_data.T), 0

    else:
        return None, 1


def verify(
    use_device,
    num_inputs,
    num_outputs,
    model_path,
    data_type,
    uq_name,
    threshold,
    num_iterations,
    num_elements,
    db_type,
    fs_path,
    name="test",
    debug_db=False,
):
    print("debug db is", debug_db)
    # When AMS has no model path it always calls the domain solution.
    # As such it behaves identically with threshold 0
    if model_path == None or model_path == "":
        threshold = 0.0

    # Name maps to the db-name. When empty it means we did not want to collect any data
    if name == "":
        threshold = 1.0

    if db_type != "none":
        data, correct = verify_data_collection(fs_path, db_type, num_inputs, num_outputs, name, debug_db)
        if correct:
            return 1
        inputs = data[0]
        outputs = data[1]

        if (model_path == None or model_path == "") and name == "":
            return 0

        # Check data type.
        if db_type == "hdf5":
            if "data_type" == "double":
                assert inputs.dtype == np.float64, "Data types do not match"
            elif "data_type" == "float":
                assert inputs.dtype == np.float32, "Data types do not match"

        # When debug db is set, we store always all elements
        if debug_db:
            predicate = data[2]
            assert (
                len(predicate) == num_elements
            ), f"debug db should always contain all data but now it has {len(predicate)}"
            assert (
                len(inputs) == num_elements and len(outputs) == num_elements
            ), f"Num elements should be the same as experiment {len(inputs)} {num_elements}"
            # Predicate points to 'true' when we use the model. The sum should be "equal" to
            # the threshold multiplied by the number of elements
            arg = sum(predicate)
            actual_elems = int(threshold * num_elements)
            assert arg == actual_elems, "Predicate does not accumulate to the expected value"
            # Over here I pick the values from input/outputs that will be selected for "domain" evaluation
            # This will allow the code to verify that predicates pick the "right" values
            inputs = inputs[np.logical_not(predicate)]
            outputs = outputs[np.logical_not(predicate)]
        elif threshold == 0.0:
            # Threshold 0 means collect all data. Verify the sizes.
            assert (
                len(inputs) == num_elements and len(outputs) == num_elements
            ), f"Num elements should be the same as experiment {len(inputs)} {num_elements}"

        elif threshold == 1.0:
            # Threshold 1.0 means to not collect any data. Verify the sizes.
            assert len(inputs) == 0 and len(outputs) == 0, "Num elements should be zero"
            # There is nothing else we can check here
            return 0
        else:
            # Compute a theoritical range of possible values in the db.
            # The duq/faiss tests have specific settings. The random one can have a
            # bound. This checks for all these cases
            lb = num_elements * (1 - threshold) - num_elements * 0.05
            ub = num_elements * (1 - threshold) + num_elements * 0.05
            assert (
                len(inputs) > lb and len(inputs) < ub
            ), f"Not in the bounds of correct items {lb} {ub} {len(inputs)} {name}"
            assert (
                len(outputs) > lb and len(outputs) < ub
            ), f"Not in the bounds of correct items {lb} {ub} {len(inputs)} {name}"

        if "delta" in uq_name:
            assert "mean" in uq_name or "max" in uq_name, "unknown Delta UQ mechanism"
            d_type = np.float32
            if data_type == "double":
                d_type = np.float64

            if "mean" in uq_name:
                # Our DUQ-mean model skips odd evaluations.
                # Here we set on verify_inputs the inputs of those evaluations
                verify_inputs = np.zeros((len(inputs), num_inputs), dtype=d_type)
                if threshold == 0.0:
                    step = 1
                elif threshold == 0.5:
                    verify_inputs[0] = np.ones(num_inputs, dtype=d_type)
                    step = 2
                for i in range(1, len(inputs)):
                    verify_inputs[i] = verify_inputs[i - 1] + step
                # Compare whether the results match our base function.
                diff_sum = np.sum(np.abs(verify_inputs - inputs))
                assert np.isclose(diff_sum, 0.0), "Mean Input data do not match"
                verify_output = np.sum(inputs, axis=1).T * num_outputs
                outputs = np.sum(outputs, axis=1)
                diff_sum = np.sum(np.abs(outputs - verify_output))
                assert np.isclose(diff_sum, 0.0), "Mean Output data do not match"
            elif "max" in uq_name:
                # Our DUQ-max model skips even evaluations.
                # Here we set on verify_inputs the inputs of those evaluations
                verify_inputs = np.zeros((len(inputs), num_inputs), dtype=d_type)
                if threshold == 0.0:
                    step = 1
                elif threshold == 0.5:
                    step = 2
                for i in range(1, len(inputs)):
                    verify_inputs[i] = verify_inputs[i - 1] + step
                diff_sum = np.sum(np.abs(verify_inputs - inputs))
                assert np.isclose(diff_sum, 0.0), "Max Input data do not match"
                verify_output = np.sum(inputs, axis=1).T * num_outputs
                outputs = np.sum(outputs, axis=1)
                diff_sum = np.sum(np.abs(outputs - verify_output))
                assert np.isclose(diff_sum, 0.0), "Max Output data do not match"
    else:
        return 0

    return 0


def from_cli(argv):
    use_device = int(argv[0])
    num_inputs = int(argv[1])
    num_outputs = int(argv[2])
    model_path = argv[3]
    data_type = argv[4]
    uq_name = argv[5]
    threshold = float(argv[6])
    num_iterations = int(argv[7])
    num_elements = int(argv[8])
    db_type = argv[9]
    fs_path = argv[10]

    return verify(
        use_device,
        num_inputs,
        num_outputs,
        model_path,
        data_type,
        uq_name,
        threshold,
        num_iterations,
        num_elements,
        db_type,
        fs_path,
    )


def from_json(argv):
    print(argv)
    use_device = int(argv[0])
    num_inputs = int(argv[1])
    num_outputs = int(argv[2])
    data_type = argv[3]
    num_elements = int(argv[4])
    model_1 = argv[5]
    model_2 = argv[6]

    env_file = Path(os.environ["AMS_OBJECTS"])
    if not env_file.exists():
        print("Environment file does not exist")
        return -1

    with open(env_file, "r") as fd:
        data = json.load(fd)

    db_type = data["db"]["dbType"]
    fs_path = data["db"]["fs_path"]

    for m in [model_1, model_2]:
        print("Testing Model", m)
        ml_id = data["domain_models"][m]
        model = data["ml_models"][ml_id]

        uq_type = model["uq_type"]
        print(json.dumps(model, indent=6))
        if "uq_aggregate" in model:
            uq_type += " ({0})".format(model["uq_aggregate"])

        print(uq_type)

        threshold = model["threshold"]
        db_label = model["db_label"]
        model_path = model.get("model_path", None)
        is_debug = model.get("debug_db", False)
        res = verify(
            use_device,
            num_inputs,
            num_outputs,
            model_path,
            data_type,
            uq_type,
            threshold,
            -1,
            num_elements,
            db_type,
            fs_path,
            db_label,
            is_debug,
        )
        if res != 0:
            return res
        print("[Success] Model", m)
    return 0


if __name__ == "__main__":
    if "AMS_OBJECTS" in os.environ:
        sys.exit(from_json(sys.argv[1:]))
    sys.exit(from_cli(sys.argv[1:]))
