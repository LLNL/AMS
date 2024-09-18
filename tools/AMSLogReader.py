#!/usr/bin/env python3
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from pathlib import Path
import glob
import re
import pandas as pd
import numpy as np


def plot_lines(ax, data, xtitle, ytitle):
    ax.plot(data, linewidth=1, label=data.columns.tolist())
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot(df, uq, memory, out):
    import matplotlib.pyplot as plt

    num_rows = len(df["FileName"].unique())
    if memory:
        # We need one row per rank/file and 2 cols for intro+outro and diff
        fig = plt.figure(constrained_layout=True)
        subfigs = fig.subfigures(nrows=num_rows, ncols=1)
        for i, (fn, gdf) in enumerate(df.groupby("FileName")):
            if num_rows > 1:
                subfig = subfigs[i]
            else:
                subfig=subfigs
            subfig.suptitle(f"Log File : {fn}")
            ax = subfig.subplots(nrows=1, ncols=1)
            plot_lines(
                ax,
                gdf[["AMS Memory At Intro", "AMS Memory At Outro"]],
                xtitle="invocation-id",
                ytitle="Memory (MB)",
            )
            ax.legend(frameon=False, shadow=False, fancybox=False)
#            tmp = pd.DataFrame(gdf["AMS Memory At Outro"] - gdf["AMS Memory At Intro"], columns=["diff"])
#            plot_lines(
#                ax[1],
#                tmp,
#                xtitle="invocation-id",
#                ytitle="Memory Diff (Outro-Intro)",
#            )

        fig.savefig(f"{out}.ams.mem.pdf")
        plt.close()

    if uq:
        fig = plt.figure(constrained_layout=True)
        subfigs = fig.subfigures(nrows=num_rows, ncols=1)
        for i, (fn, gdf) in enumerate(df.groupby("FileName")):
            subfig = subfigs[i]
            subfig.suptitle(f"Log File : {fn}")
            ax = subfig.subplots(nrows=1, ncols=2)
            plot_lines(
                ax[0],
                gdf[["Domain Model", "ML Model"]],
                xtitle="invocation-id",
                ytitle="# Every model",
            )
            ax[0].legend(frameon=False, shadow=False, fancybox=False)
            tmp = pd.DataFrame(gdf["Domain Model"] / gdf["total"], columns=["fraction"])
            plot_lines(
                ax[1],
                tmp,
                xtitle="invocation-id",
                ytitle="Fraction Of Domain",
            )
        fig.savefig(f"{out}.ams.uq.pdf")
        plt.close()


def digest_memory(memory_lines):
    pattern = r"RS:(.*?)\n"
    mem = {"Start": list(), "End": list()}
    for l in memory_lines:
        match = re.search(pattern, l)
        if match:
            value = float(match.group(1)) / (1024.0 * 1024.0)
            print(value)
            if "Start" in l:
                mem["Start"].append(value)
            elif "End" in l:
                mem["End"].append(value)
            else:
                raise RuntimeError(f"Memory Line : {l} does not contain End/Start")
    
    mem["End"] = mem["End"][0:len(mem["Start"])]
    return mem


def get_lines(lines, pattern):
    matching_lines = [line for line in lines if re.match(pattern, line)]
    return matching_lines


def get_uq(lines):
    pattern = r"\[AMS:INFO:Workflow\] Computed (\d+) using physics out of the (\d+) items \((.*?)\)"
    mem = {"physics": list(), "ml": list(), "total": list()}
    for line in lines:
        match = re.findall(pattern, line)
        if match:
            vals = match[0]
            assert len(vals) == 3, "Expecting 3 Values"
            mem["physics"].append(int(vals[0]))
            mem["ml"].append(int(vals[1]) - int(vals[0]))
            mem["total"].append(int(vals[1]))
    return mem


def parse(file, memory, uq):
    # Define the regex pattern to match lines that start with [AMS:DEBUG:MEM]
    mem_pattern = r"^\[AMS:DEBUG:MEM\].*"

    with open(file, "r") as fd:
        lines = fd.readlines()

    results = []
    columns = []
    if memory:
        memory_lines = get_lines(lines, mem_pattern)
        memory_consumption = digest_memory(memory_lines)
        for k, v in memory_consumption.items():
            results.append(v)
            columns.append(k)

    if uq:
        uq_results = get_uq(lines)
        for k, v in uq_results.items():
            results.append(v)
            columns.append(k)

    return results, columns


def main():
    parser = argparse.ArgumentParser(
        description="Simple Script digesting AMS logs and output/ploting results using matplotlib/pandas"
    )
    parser.add_argument(
        "-l",
        "--log-file",
        default=str,
        help="Log-file or files to glob and read",
        required=True,
    )

    parser.add_argument("--memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--uq", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--csv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument(
        "-o",
        "--out-file",
        dest="out_file",
        default=str,
        help="File with ams verbosity test",
        required=True,
    )

    args = parser.parse_args()
    files = {}

    for log in glob.glob(args.log_file):
        path = Path(log)
        files[str(path.stem)] = parse(path, args.memory, args.uq)

    dfs = []

    for k, v in files.items():
        if len(v[0]) > 0:
            for v1 in v[0]:
                print(len(v1))
            print(v[1])
            data = np.array(v[0])
            df = pd.DataFrame(data.T, columns=v[1])
            df["FileName"] = k
            dfs.append(df)

    df = pd.concat(dfs)
    new_names = {"FileName": "FileName"}
    if args.memory:
        new_names.update(
            {
                "Start": "AMS Memory At Intro",
                "End": "AMS Memory At Outro",
            }
        )
    if args.uq:
        new_names.update(
            {
                "physics": "Domain Model",
                "ml": "ML Model",
                "Total Elements": "total",
            }
        )

    df.rename(columns=new_names, inplace=True)

    if args.csv:
        df.to_csv(args.out_file)

    if args.plot:
        plot(df, args.uq, args.memory, args.out_file)


if __name__ == "__main__":
    main()
