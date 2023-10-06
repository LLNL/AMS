# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import datetime

import torch
from ams.config import AMSInstance
from ams.store import AMSDataStore
from ams.views import AMSDataView
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

# These make the model training "deterministic"
# Not sure though this is sufficient. For later
# cuda versions an environment variable should
# also be set CUBLAS_WORKSPACE_CONFIG=:4096:8
# or CUBLAS_WORKSPACE_CONFIG=:16:8
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear1 = torch.nn.Linear(inputSize, outputSize, dtype=torch.float64)
        self.linear2 = torch.nn.Linear(outputSize, outputSize, dtype=torch.float64)

    def forward(self, x):
        y1 = self.linear1(x)
        y = self.linear2(y1)
        return y


class ExampleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = torch.tensor(self.X[idx])
        target = torch.tensor(self.y[idx])
        return features, target


def train(n_epochs, model, loss_fn, optimiser, loader, device):
    model.train()
    total_step = len(loader)
    for n in range(1, n_epochs + 1):
        for i, (inputs, targets) in enumerate(loader):
            if device is not None:
                inputs = inputs.to(device)
                targets = targets.to(device)
            output = model(Variable(inputs))
            loss = loss_fn(output, Variable(targets))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            print(
                "{} Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    datetime.datetime.now(), n, n_epochs, i + 1, total_step, loss.item()
                ),
                inputs.shape,
                targets.shape,
            )


def validate(model, loss_fn, val_loader, device):
    model.eval()  # set to eval mode to avoid batchnorm
    validation_loss = 0.0
    with torch.no_grad():  # avoid calculating gradients
        for inputs, targets in val_loader:
            if device is not None:
                inputs = inputs.to(device)
                targets = targets.to(device)
            test_output = model(inputs)
            loss = loss_fn(test_output, targets)
            validation_loss += loss.item()
    print("VALIDATION SET ACCURACY: %.2f" % validation_loss)
    return float(validation_loss) / float(len(val_loader))


class ExampleView(AMSDataView):
    """
    A class providing semantic information
    to the data stored in the kosh-data store
    """

    input_feature_names = ["density", "pressure"]
    input_feature_dims = [1] * 2
    input_feature_types = ["scalar"] * 2

    output_feature_names = ["pressure", "soundspeed", "bulkmod", "temperature"]
    output_feature_dims = [1] * 4
    output_feature_types = ["scalar"] * 4

    def __init__(self, ams_store, entry, versions=None, **options):
        super().__init__(ams_store, entry, versions=versions, **options)


def main():
    parser = argparse.ArgumentParser(description="Training of AMS model")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of training iterations")

    parser.add_argument("--device", "-d", choices=("cpu", "gpu"), default="gpu")

    args = parser.parse_args()
    ams_config = AMSInstance()

    # Open kosh-store wrapper
    with AMSDataStore(ams_config.db_path, ams_config.db_store, ams_config.name) as db:
        with ExampleView(db, "data") as dset:
            # Pull the data from the store
            X, targets = dset.get_data()

            ##########################################################################
            #               Train some model, this is just                           #
            #                 indicative to show the steps                           #
            ##########################################################################
            length = X.shape[0]

            split = 0.8
            learningRate = 1e-2
            model = linearRegression(X.shape[-1], targets.shape[-1])
            device = None

            # Currently I am selecting cuda:3 for conveniency
            if args.device == "gpu":
                model = model.to("cuda:3")
                device = "cuda:3"

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

            # We reduce the validation/training data-size to reduce the example time.
            train_dset = ExampleDataset(X[: int(split * length * 0.1), ...], targets[: int(split * length * 0.1), ...])
            valid_dset = ExampleDataset(
                X[int(split * length * 0.1) : int(split * length * 0.15), ...],
                targets[int(split * length * 0.1) : int(split * length * 0.15), ...],
            )
            # Batch size is randombly picked to be 4096.
            # For a realistic training process
            # we will need to pick this value better
            train_loader = DataLoader(train_dset, shuffle=False, batch_size=4096)
            valid_loader = DataLoader(valid_dset, shuffle=False, batch_size=4096)

            train(args.epochs, model, criterion, optimizer, train_loader, device)
            val_error = validate(model, criterion, valid_loader, device)

            ##########################################################################
            #               Store the model in JIT format and make                   #
            #                 it accessible to the kosh store                        #
            ##########################################################################
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()
                # Mentioned in the torch source code.
                # I am not sure we need this or whether we are going to have
                # portability issues because of constant folding and device pinning.
                with torch.jit.optimized_execution(True):
                    j_model = torch.jit.script(model)

                # Ask the AMS store for a convenient file location. This is not mandatory
                # but it is good practice. It guarantees the model will be stored in a directory
                # accessible by "everyone" in the store
                fn = db.suggest_model_file_name()

                # Torch saves the model file.
                j_model.save(fn)

                # Push model to the kosh store, associate
                # the validation error with the
                # model store.
                db.add_model(fn, metadata={"val_error": val_error})


if __name__ == "__main__":
    main()
