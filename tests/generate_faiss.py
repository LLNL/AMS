#!/usr/bin/env python3
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import numpy as np
import faiss  # make faiss available


def create_elements(nb, d, distance=10, offset=0):
    centers = []
    for i in range(nb):
        c = ((i + 1) * distance) + offset
        centers.append(np.random.uniform(low=c - 0.5, high=c + 0.5, size=(100, d)))

    xb = np.vstack(centers).astype("float32")
    return xb


def main():
    parser = argparse.ArgumentParser(
        description="Faiss index generator for test purposes"
    )
    parser.add_argument(
        "-d", "--dimensions", type=int, help="Dimensions of our vectors", default=4
    )
    parser.add_argument(
        "-dst",
        "--distance",
        type=int,
        help="Jump between centers in Faiss index",
        default=10,
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="File to store FAISS index",
        default="faiss_debug.pt",
    )
    parser.add_argument(
        "-c",
        "--num-centers",
        type=int,
        help="Number of centers in FAISS index",
        default=10,
    )
    args = parser.parse_args()

    xb = create_elements(args.num_centers, args.dimensions)
    index = faiss.IndexFlatL2(args.dimensions)  # build the index
    index.add(xb)  # add vectors to the index
    faiss.write_index(index, args.file)
    # print(xb)

    # k=1
    # xq_false = create_elements(args.num_centers, args.dimensions, args.distance, 5)
    # xq_true = create_elements(args.num_centers,args.dimensions)
    # xq = np.vstack((xq_false, xq_true)).astype('float32')
    # print("Search for")
    # print(xq)
    # D, I = index.search(xq, k)
    # print("Distance is", D)
    # print("Indexes are", I)

    # predicate = np.any(D < 4, axis=1)
    # print(predicate)


if __name__ == "__main__":
    main()
