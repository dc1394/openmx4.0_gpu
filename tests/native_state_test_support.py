"""Host compiler policy shared by the production solver-state extraction tests."""
import os
import resource
import shlex
import sys


def bounded_stack():
    # An unlimited Linux stack can overlap ASan's shadow address range.
    _, hard = resource.getrlimit(resource.RLIMIT_STACK)
    limit = 8 * 1024 * 1024
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_STACK, (limit, hard))


def compiler_command(language, compiler=None):
    """Use the configured compiler/flags, with a portable sanitizer policy.

    CMake supplies the same driver as the core to avoid mixing GCC and Clang
    sanitizer runtimes. C callers must use -x c for the source and reset -x
    before passing the shared library; their driver can therefore be C++.
    """
    variable, default, standard = ('CXX', 'c++', 'c++17') if language == 'c++' else ('CC', 'cc', 'c11')
    command = shlex.split(compiler or os.environ.get(variable, default))
    command += shlex.split(os.environ.get('CXXFLAGS' if language == 'c++' else 'CFLAGS', ''))
    command += ['-std=' + standard, '-O1', '-g', '-fsanitize=address,undefined',
                '-fno-sanitize-recover=all']
    # Fixed-address ELF executables avoid occasional Linux ASan shadow-map
    # collisions. Darwin uses its normal PIE executables and sanitizer layout.
    if sys.platform.startswith('linux'):
        command += ['-fno-pie', '-no-pie']
    return command
