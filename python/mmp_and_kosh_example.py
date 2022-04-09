from subprocess import Popen, PIPE
from ctypes import cdll, POINTER, c_bool, c_double
import argparse
import kosh
import sys
import os.path as pth
import numpy

# get the Koshloaders from AMS repo
ams_repo_dir = pth.abspath("../ams")    # this might fail depending on your local setup
# ams_repo_dir = "/path/to/your/repo"
sys.path.append(pth.join(ams_repo_dir, "workflow/kosh"))
from loaders.nptxt_loader import NumpyTxtLoaderWithFeaturesRegexed, NumpyTxtLoaderWithFeaturesNumbered

# This is is an example
# The data we are reading are density and energy
# But I don't know how to map that to num_qpt, num_elt, num_mat
# So far now we are just reading enough data from kosh to match what the suer asked for
# It is obviously wrong.
 
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("-d", "--device", help="Device config string", default="cpu")
p.add_argument("-c", "--stop-cycle", type=int, help="Stop cycle", default=10)
p.add_argument("-m", "--num-mats", type=int,help="Number of materials", default=5)
p.add_argument("-e", "--num-elems", type=int,help="Number of elements", default=10000)
p.add_argument("-q", "--num-qpts", type=int, help="Number of quadrature points per element", default=64)
p.add_argument("-r", "--empty-element-ratio", type=float, help="Fraction of elements that are empty " "for each material. If -1 use a random value for each. ", default=-1)
p.add_argument("-p", "--pack-sparse", "--np", "--do-not-pack-sparse", type=bool, help="pack sparse material data before evals (cpu only)", default=True)

args = p.parse_args()

num_values_to_read = args.num_elems * args.num_mats * args.num_qpts

print(args)

# Let's create an empty Kosh store
store = kosh.connect("kosh_stroe_example", delete_all_contents=True)
# Let's inform the store about the loader
store.add_loader(NumpyTxtLoaderWithFeaturesRegexed)
# Let's create a dataset
dataset = store.create("test_mmp")

# The file des not contain the features names
# so we need to put them manually as metadata
input_fnames= ['rho',
               'energy',
               '*tatb-solid',
               '*monofurazan-solid',
               '*difurazan-solid',
               '*c3n2polymer-solid']
# Let's associate the input file with the Kosh dataset
dataset.associate("/usr/workspace/AMS/datasets/cheetah-pore-dump/pore75/inputs*", "numpy/txt-with-features", metadata = {"feature_names":input_fnames})
# if we also want to associate the output features
# then uncomment these lines
"""
output_fnames=['pressure',
               'temperature',
               'beta derivative',
               'sound speed',
               'cv',
               'thermal conductivity',
               'shear viscosity',
               'o2']
dataset.associate("/usr/workspace/AMS/datasets/cheetah-pore-dump/pore75/outputs*", "numpy/txt-with-features", metadata = {"feature_names":output_fnames})
"""
# Now fill the input arrays
# They come back shaped [num_proc][N]
density = dataset["rho"][0,:num_values_to_read][0]
energy = dataset["energy"][0,:num_values_to_read][0]
# Assuming True everywhere ???
indicators = numpy.ones(args.num_elems*args.num_mats, dtype=bool)

print(density.shape, density.dtype)

is_cpu = args.device.lower() == "cpu"
libname = './mmp-toss_3_x86_64_ib.so'
lib = cdll.LoadLibrary(libname)

p = Popen(("nm", f"{libname}"), stdout=PIPE, stderr=PIPE)
o, e = p.communicate()

# Figures out the mangled name
# I know it's a hack
for l in o.decode().split("\n"):
    if "eval_data" in l:
        eval_data_function_name = l.split()[2]

eval_data_function = getattr(lib, eval_data_function_name)

# Now let's call it
eval_data_function(args.stop_cycle, is_cpu, args.pack_sparse,
                   args.num_qpts, args.num_elems, args.num_mats,
                   density.ctypes.data_as(POINTER(c_double)),
                   energy.ctypes.data_as(POINTER(c_double)),
                   indicators.ctypes.data_as(POINTER(c_bool)))
