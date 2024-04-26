import os
import os.path as osp
import numpy
import torch
from setuptools import setup
from pkg_resources import parse_version
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
from distutils.sysconfig import get_config_vars



(opt,) = get_config_vars("OPT")
os.environ["OPT"] = " ".join(
    flag for flag in opt.split() if flag != "-Wstrict-prototypes"
)

PROJECT_DIR = osp.realpath(osp.dirname(osp.dirname(__file__)))


# Find the Numpy headers
include_dirs = [numpy.get_include()]

# Cpp Compilation and linkage options
# MIN_OPS_PER_THREAD roughly controls parallelization, see doc in README.md
# COMP_T_ON_32_BITS for components identifiers on 32 bits rather than 16
if os.name == 'nt':  # windows
    cpp_extra_compile_args = ["/std:c++14", "/openmp",
        "-DMIN_OPS_PER_THREAD=10000", "-DCOMP_T_ON_32_BITS"]
    extra_link_args = ["/lgomp"]
elif os.name == 'posix':  # linux
    cpp_extra_compile_args = ["-std=c++11", "-fopenmp",
        "-DMIN_OPS_PER_THREAD=10000", "-DCOMP_T_ON_32_BITS"]
    extra_link_args = ["-lgomp"]
else:
    raise NotImplementedError('OS not supported yet.')

# CUDA Compilation options
extra_compile_args = {"cxx": ["-std=c++14"]}
if parse_version(torch.__version__) >= parse_version('2.0.0'):
    extra_compile_args['cxx'] = ['-std=c++17']
nvcc_args = [
    "-DCUDA_HAS_FP16=1", "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__",
    "--ftemplate-depth=2048"
    # "–pending_instantiations=2048",
]
nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
if nvcc_flags_env != "":
    nvcc_args.extend(nvcc_flags_env.split(" "))
CC = os.environ.get("CC", None)
if CC is not None:
    CC_arg = "-ccbin={}".format(CC)
    if CC_arg not in nvcc_args:
        if any(arg.startswith("-ccbin") for arg in nvcc_args):
            raise ValueError("Inconsistent ccbins")
        nvcc_args.append(CC_arg)
extra_compile_args["nvcc"] = nvcc_args



# Cpp / CUDA source files
def find_sources(src):
    sources =  [
        os.path.join(root, file)
        for root, dirs, files in os.walk(src)
        for file in files
        if file.endswith(".cpp") or file.endswith(".cu")
    ]
    return sources


grid_graph_ext_modules = CppExtension(
    name='grid_graph',
    sources=find_sources("src/grid_graph"),
    include_dirs=include_dirs,
    extra_compile_args=cpp_extra_compile_args,
    extra_link_args=extra_link_args
)

parallel_cut_pursuit_ext_modules = CppExtension(
    name='cp_d0_dist_cpy',
    sources=find_sources("src/parallel_cut_pursuit"),
    include_dirs=include_dirs,
    extra_compile_args=cpp_extra_compile_args,
    extra_link_args=extra_link_args
)

prefix_sum_ext_modules = CUDAExtension(
    name="superpoint_graph.prefix_sum_C",
    sources=find_sources("src/prefix_sum")
)

frnn_ext_modules = CUDAExtension(
    name="superpoint_graph.frnn_C",
    sources=find_sources("src/frnn"),
    # include_dirs=include_dirs,
    # define_macros=[],
    extra_compile_args=extra_compile_args
)



class BuildExtension_noninja(BuildExtension):
    def __init__(self, *args, **kwargs):
        super().__init__(use_ninja=False, *args, **kwargs)



setup(
    name="superpoint_graph",
    version="1.0",
    install_requires=["torch", "numpy", "pgeof", "omegaconf"],
    # 告诉setuptools哪些目录下的文件被映射到哪个源码包, 
    # packages=["superpoint_graph", 'superpoint_graph.data', 'superpoint_graph.utils', 'superpoint_graph.transforms'],
    packages=["superpoint_graph", 'superpoint_graph.utils'],
    package_dir={
        "superpoint_graph": "functions",
        # 'superpoint_graph.data': "functions/data",
        'superpoint_graph.utils': "functions/utils",
        # 'superpoint_graph.transforms': "functions/transforms"
    },
    ext_modules=[
        grid_graph_ext_modules,
        parallel_cut_pursuit_ext_modules,
        prefix_sum_ext_modules,
        frnn_ext_modules
    ],
    cmdclass={"build_ext": BuildExtension_noninja},
)
