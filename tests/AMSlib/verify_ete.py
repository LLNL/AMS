import sys
from pathlib import Path
import pandas as pd
import h5py
import numpy as np


def get_suffix(db_type):
    if db_type == "csv":
        return "csv"
    if db_type == "none":
        return "none"
    if db_type == "hdf5":
        return "h5"
    return "unknown"


def verify_data_collection(fs_path, db_type, num_inputs, num_outputs, name="test"):
    if not Path(fs_path).exists():
        print("Expecting output directory to exist")
        return None, 1

    suffix = get_suffix(db_type)
    if suffix == "none":
        return None, 0

    fn = f"test_0.{suffix}"
    fp = Path(f"{fs_path}/{fn}")

    if not fp.exists():
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
            assert len(dsets) == (num_inputs + num_outputs), "Expected equal number of inputs/outputs"
            inputs = sum(1 for s in dsets if "input" in s)
            assert inputs == num_inputs, "Expected equal number of inputs"
            outputs = sum(1 for s in dsets if "output" in s)
            assert outputs == num_outputs, "Expected equal number of outputs"
            input_data = [[] for _ in range(num_inputs)]
            output_data = [[] for _ in range(num_outputs)]
            for d in dsets:
                loc = int(d.split("_")[1])
                if len(fd[d]):
                    if "input" in d:
                        input_data[loc] = fd[d][:]
                    elif "output" in d:
                        output_data[loc] = fd[d][:]
            input_data = np.array(input_data)
            output_data = np.array(output_data)
            fp.unlink()
            return (input_data.T, output_data.T), 0
    else:
        return None, 1


def main():
    use_device = int(sys.argv[1])
    num_inputs = int(sys.argv[2])
    num_outputs = int(sys.argv[3])

    model_path = sys.argv[4]
    data_type = sys.argv[5]
    uq_name = sys.argv[6]
    threshold = float(sys.argv[7])

    num_iterations = int(sys.argv[8])
    num_elements = int(sys.argv[9])
    db_type = sys.argv[10]
    fs_path = sys.argv[11]

    if db_type != "none":
        data, correct = verify_data_collection(fs_path, db_type, num_inputs, num_outputs)
        if correct:
            return 1
        inputs = data[0]
        outputs = data[1]

        if db_type == "hdf5":
            if "data_type" == "double":
                assert inputs.dtype == np.float64, "Data types do not match"
            elif "data_type" == "float":
                assert inputs.dtype == np.float32, "Data types do not match"
        if threshold == 0.0:
            assert (
                len(inputs) == num_elements and len(outputs) == num_elements
            ), "Num elements should be the same as experiment"

        elif threshold == 1.0:
            assert len(inputs) == 0 and len(outputs) == 0, "Num elements should be zero"
            # There is nothing else we can check here
            return 0
        else:
            lb = num_elements * threshold - num_elements * 0.05
            ub = num_elements * threshold + num_elements * 0.05
            print(lb, ub)
            assert len(inputs) > lb and len(inputs) < ub, "Not in the bounds of correct items"
            assert len(outputs) > lb and len(outputs) < ub, "Not in the bounds of correct items"

        if "delta" in uq_name:
            assert "mean" in uq_name or "max" in uq_name, "unknown Delta UQ mechanism"
            d_type = np.float32
            if data_type == "double":
                d_type = np.float64

            if "mean" in uq_name:
                verify_inputs = np.zeros((len(inputs), num_inputs), dtype=d_type)
                if threshold == 0.0:
                    step = 1
                elif threshold == 0.5:
                    verify_inputs[0] = np.ones(num_inputs, dtype=d_type)
                    step = 2
                for i in range(1, len(inputs)):
                    verify_inputs[i] = verify_inputs[i - 1] + step
                diff_sum = np.sum(np.abs(verify_inputs - inputs))
                assert np.isclose(diff_sum, 0.0), "Input data do not match"
                verify_output = np.sum(inputs, axis=1).T * num_outputs
                outputs = np.sum(outputs, axis=1)
                diff_sum = np.sum(np.abs(outputs - verify_output))
                assert np.isclose(diff_sum, 0.0), "Output data do not match"
            elif "max" in uq_name:
                verify_inputs = np.zeros((len(inputs), num_inputs), dtype=d_type)
                if threshold == 0.0:
                    step = 1
                elif threshold == 0.5:
                    step = 2
                for i in range(1, len(inputs)):
                    verify_inputs[i] = verify_inputs[i - 1] + step
                diff_sum = np.sum(np.abs(verify_inputs - inputs))
                assert np.isclose(diff_sum, 0.0), "Input data do not match"
                verify_output = np.sum(inputs, axis=1).T * num_outputs
                outputs = np.sum(outputs, axis=1)
                diff_sum = np.sum(np.abs(outputs - verify_output))
                assert np.isclose(diff_sum, 0.0), "Output data do not match"
    else:
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
