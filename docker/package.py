# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

from spack import *


class Faiss(CMakePackage, CudaPackage):
    """Faiss is a library for efficient similarity search and clustering of
       dense vectors.

      Faiss contains algorithms that search in sets of vectors of any size, up
      to ones that possibly do not fit in RAM. It also contains supporting code
      for evaluation and parameter tuning. Faiss is written in C++ with
      complete wrappers for Python/numpy. Some of the most useful algorithms
      are implemented on the GPU. It is developed by Facebook AI Research.
    """

    homepage = "https://github.com/facebookresearch/faiss"
    url      = "https://github.com/facebookresearch/faiss/archive/v1.7.2.tar.gz"

    maintainers = ['bhatiaharsh']

    # 1.7.2 was moved to cmake
    version('1.7.2', sha256='d49b4afd6a7a5b64f260a236ee9b2efb760edb08c33d5ea5610c2f078a5995ec')

    # previous versions used autotools
    version('1.6.3', sha256='e1a41c159f0b896975fbb133e0240a233af5c9286c09a28fde6aefff5336e542', deprecated=True)
    version('1.5.3', sha256='b24d347b0285d01c2ed663ccc7596cd0ea95071f3dd5ebb573ccfc28f15f043b', deprecated=True)

    variant('python', default=False, description='Build Python bindings')
    variant('shlib',  default=False, description='Build shared library')
    variant('tests',  default=False, when='@:1.6.3', description='Build tests (not functional for v1.7.2)')

    conflicts('+tests', when='~python', msg='+tests must be accompanied by +python')

    depends_on('cmake@3.17:',                   type='build')
    depends_on('py-pip',        when='+python', type='build')
    depends_on('py-wheel',      when='+python', type='build')
    depends_on('py-setuptools', when='+python', type='build')
    depends_on('swig',          when='+python', type='build')

    depends_on('blas')
    depends_on('python@3.7:',   when='+python', type=('build', 'run'))
    depends_on('py-numpy',      when='+python', type=('build', 'run'))
    # depends_on('py-scipy',      when='+tests',  type=('build', 'run'))

    # --------------------------------------------------------------------------
    # patch for v1.5.3
    # faiss assumes that the "source directory" will always
    # be called "faiss" (not spack-src or faiss-1.5.3)
    # so, we will have to create a symlink to self (faiss did that in 1.6.3)
    # and add an include path
    patch('fixes-in-v1.5.3.patch', when='@1.5.3')

    # patch for v1.6.3
    # for v1.6.3, GPU build has a bug (two files need to be deleted)
    #   https://github.com/facebookresearch/faiss/issues/1159
    # also, some include paths in gpu/tests/Makefile are missing
    patch('fixes-in-v1.6.3.patch', when='@1.6.3')

    # patch for v1.7.2
    # a shared object is missing in the python/setup.py
    #   https://github.com/facebookresearch/faiss/issues/2063
    #   https://github.com/facebookresearch/faiss/pull/2062
    # a header is missing in a test file
    #   https://github.com/facebookresearch/faiss/issues/2300
    patch('fixes-in-v1.7.2.patch', when='@1.7.2')

    # --------------------------------------------------------------------------
    def setup_run_environment(self, env):
        if '+python' in self.spec:
            env.prepend_path('PYTHONPATH', python_platlib)
            env.prepend_path('LD_LIBRARY_PATH', os.path.join(python_platlib, "faiss"))

    def prefix_and_install(self, infile, prefix='faiss'):
        if len(prefix) > 0:
            outfile = prefix+'_'+infile
            os.rename(infile, outfile)
        else:
            outfile = infile
        install(outfile, self.prefix.bin)

    # --------------------------------------------------------------------------
    @when("@1.7.2:")
    def cmake_args(self):

        variants2cmake = {'cuda':   'FAISS_ENABLE_GPU',
                          'python': 'FAISS_ENABLE_PYTHON',
                          'tests':  'BUILD_TESTING',
                          'shlib':  'BUILD_SHARED_LIBS'}

        cmake_args = []
        for v, c in variants2cmake.items():
            if '+'+v in self.spec:
                cmake_args.append('-D'+c+'=ON')
            else:
                cmake_args.append('-D'+c+'=OFF')

        return cmake_args

    # --------------------------------------------------------------------------
    # faiss creates several individual targets for testing and demo!
    # let's pull out all from makefile
    def fetch_targets_v172(self):

        is_test = lambda x: 'Test' == x[:4]
        is_demo = lambda x: 'demo' == x[:4]
        is_gpu  = lambda x: 'gpu' in x.lower()

        targets = os.popen('make help').read()
        targets = [t.split()[1] for t in targets.split('\n') if t[:3] == '...']

        tests = [t for t in targets if is_test]
        demos = [t for t in targets if is_demo]

        gtests = [t for t in tests if is_gpu(t)]
        ctests = [t for t in tests if not is_gpu(t)]

        gdemos = [t for t in demos if is_gpu(t)]
        cdemos = [t for t in demos if not is_gpu(t)]

        return (ctests, gtests), (cdemos, gdemos)

    # --------------------------------------------------------------------------
    def build(self, spec, prefix):

        with working_dir(self.build_directory):

            # make the faiss build
            make('faiss')

            # build python bindings
            if '+python' in self.spec:
                make('swigfaiss')

            # build tests
            if '+tests' in self.spec:
                make('faiss_test')

                (ctests, gtests), (cdemos, gdemos) = self.fetch_targets_v172()
                for t in ctests:
                    make(t)
                for d in cdemos:
                    make(d)

                if '+cuda' in self.spec:
                    # the make target actually runs the tests as well!
                    # having some trouble with linking to cudart.so
                    for t in gtests:
                        make(t)
                    for d in gdemos:
                        make(d)

    # --------------------------------------------------------------------------
    def install(self, spec, prefix):

        with working_dir(self.build_directory):

            make('install')
            if '+python' in self.spec:
                with working_dir('faiss/python'):
                    args = std_pip_args + ['--prefix='+prefix, '.']
                    pip(*args)

            if '+tests' in self.spec:
                if not os.path.isdir(self.prefix.bin):
                    os.makedirs(self.prefix.bin)

                (ctests, gtests), (cdemos, gdemos) = self.fetch_targets_v172()

                # CPU tests and demos
                with working_dir('tests'):
                    self.prefix_and_install('faiss_test', prefix='')

                with working_dir('demos'):
                    for d in cdemos:
                        self.prefix_and_install(d)

                # GPU tests and demos
                if '+cuda' in self.spec:

                    with working_dir('faiss/gpu/test'):
                        for t in gtests:
                            self.prefix_and_install(t)

                    with working_dir('demos'):
                        for d in gdemos:
                            self.prefix_and_install(d)

    # --------------------------------------------------------------------------
    # deprecated versions (< 1.7.2) that used Autotools
    # --------------------------------------------------------------------------
    # Remove the following once versions < 1.7.2 are dropped
    @when("@:1.6.3")
    def configure_args(self):
        args = []
        if '+cuda' in self.spec:
            args.append('--with-cuda='+self.spec['cuda'].prefix)
        else:
            args.append('--without-cuda')

        return args

    # turn cmake into a configure step
    @when("@:1.6.3")
    def cmake(self, spec, prefix):

        configure('--prefix='+prefix, *self.configure_args())

        # spack injects its own optimization flags
        makefile = FileFilter('makefile.inc')
        makefile.filter('CPUFLAGS     = -mavx2 -mf16c',
                        '#CPUFLAGS     = -mavx2 -mf16c')

    @when('@:1.6.3')
    def build(self, spec, prefix):

        make()
        if '+python' in self.spec:
            make('-C', 'python')

        # CPU tests
        if '+tests' in self.spec:
            with working_dir('tests'):
                make('gtest')
                make('tests')

        # GPU tests
        if '+tests+cuda' in self.spec:
            with working_dir(os.path.join('gpu', 'test')):
                make('gtest')
                make('build')                       # target added by the patch
                make('demo_ivfpq_indexing_gpu')

    @when('@:1.6.3')
    def install(self, spec, prefix):

        make('install')
        if '+python' in self.spec:
            with working_dir('python'):
                args = std_pip_args + ['--prefix=' + prefix, '.']
                pip(*args)

        if '+tests' not in self.spec:
            return

        if not os.path.isdir(self.prefix.bin):
            os.makedirs(self.prefix.bin)

        # CPU tests
        with working_dir('tests'):
            # rename the exec to keep consistent with gpu tests
            os.rename('tests', 'TestCpu')
            _prefix_and_install('TestCpu')

        # GPU tests
        if '+cuda' in self.spec:
            with working_dir(os.path.join('gpu', 'test')):
                self.prefix_and_install('TestGpuIndexFlat')
                self.prefix_and_install('TestGpuIndexBinaryFlat')
                self.prefix_and_install('TestGpuIndexIVFFlat')
                self.prefix_and_install('TestGpuIndexIVFPQ')
                self.prefix_and_install('TestGpuMemoryException')
                self.prefix_and_install('TestGpuSelect')
                self.prefix_and_install('demo_ivfpq_indexing_gpu')

    # --------------------------------------------------------------------------

# ------------------------------------------------------------------------------

