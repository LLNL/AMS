#!/usr/bin/env python3

# ------------------------------------------------------------------------------
import argparse
from ctypes import CDLL, POINTER, c_bool, c_double
import numpy as np
import os

# ------------------------------------------------------------------------------
def setup_args():

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-d", "--device", help="Device config string", default="cpu")
    p.add_argument("-c", "--stop-cycle", type=int, help="Stop cycle", default=10)
    p.add_argument("-m", "--num-mats", type=int,help="Number of materials", default=5)
    p.add_argument("-e", "--num-elems", type=int,help="Number of elements", default=10000)
    p.add_argument("-q", "--num-qpts", type=int, help="Number of quadrature points per element", default=64)
    p.add_argument("-r", "--empty-element-ratio", type=float, help="Fraction of elements that are empty " "for each material. If -1 use a random value for each. ", default=-1)
    p.add_argument("-p", "--pack-sparse", "--np", "--do-not-pack-sparse", type=bool, help="pack sparse material data before evals (cpu only)", default=True)

    args = p.parse_args()
    print(args)
    return args


# ------------------------------------------------------------------------------
def find_symbol(libpath, funcname):

    # version 1 function from Charles. not being used anymore
    from subprocess import Popen, PIPE
    from ctypes import cdll

    with Popen(("nm", libpath), stdout=PIPE, stderr=PIPE) as p:
        o, e = p.communicate()

        # Figures out the mangled name
        # I know it's a hack
        for l in o.decode().split("\n"):
            if funcname in l:

                eval_data_function_name = l.split()[2]
                print (f'found: ({l}) :: ({eval_data_function_name})')
                lib = cdll.LoadLibrary(libpath)
                return getattr(lib, eval_data_function_name)

    return None


# ------------------------------------------------------------------------------
def setup_data(num_mats, num_elems, num_qpts,
               empty_element_ratio=-1, min_ratio=0.2):

    print (f'Setting up indicators (empty_element_ratio = {empty_element_ratio}, min_ratio = {min_ratio})')
    #indicators = np.zeros(num_mats*num_elems, dtype=bool)
    indicators = np.zeros((num_mats, num_elems), dtype=bool)

    for mat_idx in range(num_mats):

        if empty_element_ratio == -1:
            ratio = np.random.random() * (1-min_ratio) + min_ratio
        else:
            ratio = 1 - empty_element_ratio

        num_nonzero_elems = int(ratio * num_elems)
        print (f'using {num_nonzero_elems}/{num_elems} for material {mat_idx}')

        nz = 0
        for elem_idx in range(num_elems):
            if nz < num_nonzero_elems:
                if (num_nonzero_elems - nz) == (num_elems - elem_idx) or np.random.random() <= ratio:
                    indicators[mat_idx, elem_idx] = True
                    print (f' setting (mat = {mat_idx}, elem = {elem_idx}) = 1')
                    nz += 1

    # now fill the input arrays
    print (f'Setting up input data')
    density = np.zeros((num_mats, num_elems, num_qpts), dtype=float)
    energy = np.zeros((num_mats, num_elems, num_qpts), dtype=float)

    #density = np.zeros(num_mats*num_elems*num_qpts, dtype=float)
    #energy = np.zeros(num_mats*num_elems*num_qpts, dtype=float)

    for mat_idx in range(num_mats):
        for elem_idx in range(num_elems):

            #me = mat_idx*num_elems + elem_idx
            if not indicators[mat_idx, elem_idx]:
                continue

            density[mat_idx, elem_idx, : ] = 0.1 + np.random.rand(num_qpts)
            energy[mat_idx, elem_idx, : ] = 0.1 + np.random.rand(num_qpts)

            #for qpt_idx in range(num_qpts):
            #    density[qpt_idx + me*num_qpts] = 0.1 + np.random.random()
            #    energy[qpt_idx + me*num_qpts]  = 0.1 + np.random.random()

    print (f'Density = {density.shape}, Energy = {energy.shape}, Indicators = {indicators.shape}')
    return indicators.reshape(-1), density.reshape(-1), energy.reshape(-1)

# ------------------------------------------------------------------------------
if __name__ == '__main__':

    try:
        # TODO: the lib name and path are still hardcoded!
        libname = 'libmmp'
        libpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../install/lib')

        if os.path.isfile(os.path.join(libpath, f'{libname}.dylib')):
            libpath = os.path.join(libpath, f'{libname}.dylib')
        elif os.path.isfile(os.path.join(libpath, f'{libname}.so')):
            libpath = os.path.join(libpath, f'{libname}.so')

        mmp_lib = CDLL(libpath)
        print (f'Found \"mmp_main\" = {mmp_lib.mmp_main} in ({libpath})')
    except:
        print (f'Failed to find \"mmp_main\" in ({libpath})')
        exit (1)


    args = setup_args()

    # create input data
    indicators, density, energy = \
        setup_data(args.num_mats, args.num_elems, args.num_qpts,
                   args.empty_element_ratio, 0.2)

    # Now let's call it
    mmp_lib.mmp_main(args.device.lower() == "cpu",
                     args.stop_cycle, args.pack_sparse,
                     args.num_mats, args.num_elems, args.num_qpts,
                     density.ctypes.data_as(POINTER(c_double)),
                     energy.ctypes.data_as(POINTER(c_double)),
                     indicators.ctypes.data_as(POINTER(c_bool)))

# ------------------------------------------------------------------------------
