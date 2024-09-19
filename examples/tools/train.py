#!/usr/bin/env python

from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time

from ams.config import AMSInstance
from ams.store import AMSDataStore
from ams.store_types import AMSModelDescr
from ams.store_types import UQType
from ams.store_types import UQAggregate
from ams.views import AMSDataView
import argparse
from pathlib import Path
import torch.nn.init as init


class ExampleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        features = torch.tensor(self.x[idx]).to(torch.float32)
        target = torch.tensor(self.y[idx]).to(torch.float32)
        return features, target


class BinomialOptionsView(AMSDataView):
    """
    A class providing semantic information
    to the data stored in the kosh-data store
    """

    input_feature_names = ["S", "X", "R", "V", "T"]
    input_feature_dims = [1] * 5
    input_feature_types = ["scalar"] * 5

    output_feature_names = ["CallValue"]
    output_feature_dims = [1]
    output_feature_types = ["scalar"]

    def __init__(self, ams_store, domain_name, entry, versions=None, **options):
        super().__init__(ams_store, domain_name, entry, versions=versions, **options)


class BenchNeuralNetwork(nn.Module):
    def __init__(self, i_scalars, o_scalars, network_params):
        super(BenchNeuralNetwork, self).__init__()
        hidden1_features = network_params.get("hidden1_features")
        hidden2_features = network_params.get("hidden2_features")
        dropout = network_params.get("dropout")

        n_ipt_features = i_scalars
        n_opt_features = o_scalars

        if hidden2_features != 0:
            self.layers = nn.Sequential(
                nn.Linear(n_ipt_features, hidden1_features),
                nn.BatchNorm1d(hidden1_features),
                nn.Dropout(dropout),
                nn.LeakyReLU(),
                nn.Linear(hidden1_features, hidden2_features),
                nn.BatchNorm1d(hidden2_features),
                nn.Dropout(dropout),
                nn.LeakyReLU(),
                nn.Linear(hidden2_features, n_opt_features),
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(n_ipt_features, hidden1_features),
                nn.BatchNorm1d(hidden1_features),
                nn.Dropout(dropout),
                nn.LeakyReLU(),
                nn.Linear(hidden1_features, n_opt_features),
            )
        self.register_buffer("ipt_min", torch.full((1, i_scalars), torch.inf))
        self.register_buffer("ipt_max", torch.full((1, i_scalars), -torch.inf))

        self.register_buffer("opt_min", torch.full((1, o_scalars), torch.inf))
        self.register_buffer("opt_max", torch.full((1, o_scalars), -torch.inf))

    def get_model_device(self):
        return next(self.parameters()).device

    def forward(self, x):
        x = (x - self.ipt_min) / (self.ipt_max - self.ipt_min)
        x = self.layers(x)
        x = torch.clamp(x, min=0)
        x = x * (self.opt_max - self.opt_min) + self.opt_min
        return x

    def calculate_and_save_normalization_parameters(self, train_dl):
        for x, y in train_dl:
            x = x.to(self.get_model_device())
            y = y.to(self.get_model_device())
            batch_min = x.min(dim=0, keepdim=True).values
            batch_max = x.max(dim=0, keepdim=True).values
            self.ipt_min = torch.min(self.ipt_min, batch_min)
            self.ipt_max = torch.max(self.ipt_max, batch_max)

            batch_min = y.min(dim=0, keepdim=True).values
            batch_max = y.max(dim=0, keepdim=True).values
            self.opt_min = torch.min(self.opt_min, batch_min)
            self.opt_max = torch.max(self.opt_max, batch_max)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def train_loop(dataloader, model, loss_fn, optimizer, device, log_interval=5):
    model.train()
    running_loss = 0.0
    total_error = 0.0
    prev_X = None
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_error += loss.item()

        if batch_idx % log_interval == 0 and batch_idx > 0:
            avg_loss = running_loss / log_interval
            running_loss = 0.0

    return total_error / len(dataloader)


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    avg_test_loss = test_loss / len(dataloader)
    return avg_test_loss


def jit_and_save_model(model, filename, device, sample=None, trace=False, to_double=True):
    model = model.to(device)
    if to_double:
        model = model.to(torch.float64)

    if trace and sample is None:
        raise TypeError(f"Jitting was instructed to trace but no sample was given")

    if sample is not None:
        data = torch.tensor(sample).to(device)
        if to_double:
            data = data.to(torch.float64)

    model.eval()
    with torch.jit.optimized_execution(True):
        if trace:
            traced = torch.jit.trace(model, (torch.randn(inputDim, dtype=torch.double).to(device),))
            traced.save(path)
        else:
            scripted = torch.jit.script(model)
            scripted.save(filename)

    return filename


def main():
    train_args = {
        "hidden1_features": 5,
        "hidden2_features": 396,
        "weight_decay": 0.000658,
        "learning_rate": 1e-4,
        "dropout": 0.061841,
    }

    parser = argparse.ArgumentParser(description="Training of AMS model")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--split", "-s", type=float, default=0.8, help="Fraction of data to be used as training set")
    parser.add_argument("--batch-size", "-bs", default=1024, type=int, help="Batch size of training")

    parser.add_argument("--persistent-db-path", "-db", help="The path of the AMS database", required=True)

    parser.add_argument("--domain-name", "-dn", help="The domain name to train a model for", required=True)

    args = parser.parse_args()
    ams_config = AMSInstance.from_path(args.persistent_db_path)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    with AMSDataStore(ams_config.db_path, ams_config.db_store, ams_config.name) as db:
        with BinomialOptionsView(db, args.domain_name, "data") as dset:
            # Pull the data from the store
            X, targets = dset.get_data()
            """
            From this point on I am training a typical model. There is no reason to do this like this
            but it should be "clean" and not dependent on paths.
            """
            length = X.shape[0]
            train_size =  int(length * args.split)
            test_size = int(length - train_size)
            print("Length", length, train_size, test_size)
            all_data = ExampleDataset(X, targets)
            generator = torch.Generator().manual_seed(42)
            train_dset, test_dset = torch.utils.data.random_split(
                all_data, [train_size, test_size], generator=generator
            )
            print(len(train_dset), len(test_dset))

            model = BenchNeuralNetwork(5, 1, train_args)
            model = model.to(device)
            model.calculate_and_save_normalization_parameters(train_dset)
            model.initialize_weights()

            train_loader = DataLoader(
                train_dset,
                shuffle=True,
                batch_size=args.batch_size,
                num_workers=4,
                drop_last=True,
            )
            test_loader = DataLoader(
                test_dset,
                shuffle=False,
                batch_size=args.batch_size,
                num_workers=4,
                drop_last=True,
            )

            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=train_args["learning_rate"], weight_decay=train_args["weight_decay"]
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=train_args["learning_rate"])

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, verbose=True)
            train_loss = 0
            test_loss = 0
            for epoch in range(0, args.epochs):
                e_start = time.time()
                train_loss = train_loop(train_loader, model, criterion, optimizer, device)
                test_loss = test_loop(test_loader, model, criterion, device)
                scheduler.step(test_loss)
                e_end = time.time()
                print(f"{epoch+1} test time: {e_end-e_start} Train loss: {train_loss} Test-Loss: {test_loss}")

            """ 
            End of my training code. Now we need to register the model to kosh with metadata.
            """

            # We convert and jit the model. For D-UQ we need to think on how to get this done
            # in a generic way.
            # NOTE: Talk with ROB about this
            filename = jit_and_save_model(model, db.suggest_model_file_name(), device, to_double=True)

            # Register model to kosh store. This exposes the model to the rest of the workflow
            # NOTE: Talk with ROB, VIVEK and JAYRAM
            model_descr = AMSModelDescr(path=filename, threshold=0.5, uq_type=UQType.Random)
            md = {"train_parameters": train_args, "batch_size": args.batch_size, "view_files": dset.get_files()}
            db.add_model(args.domain_name, model_descr, test_loss, train_loss, metadata=md)
            print(model_descr)
            print(filename)


if __name__ == "__main__":
    main()
