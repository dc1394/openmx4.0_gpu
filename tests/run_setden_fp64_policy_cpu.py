#!/usr/bin/env python3
"""Check production native-density split selection and explicit overrides."""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'source/set_hamiltonian_gpu.cu').read_text()
start = source.index('    const char *split = std::getenv("OPENMX_SETDEN_FP64_SPLIT_SPIN");')
end = source.index('    const std::size_t shared_bytes', start)
code = '''
#include <cassert>
#include <cstdlib>
#include <cstdio>
static bool selected(int max_no, int spin_count) {
''' + source[start:end] + '''
    return split_spin;
}
int main() {
    unsetenv("OPENMX_SETDEN_FP64_SPLIT_SPIN");
    struct Case { int orbitals, spins; bool expected; };
    const Case cases[] = {
        {1, 1, false}, {13, 4, false}, {26, 2, false}, {26, 4, false},
        {31, 2, false}, {32, 2, false}, {32, 4, false},
        {33, 1, false}, {33, 2, true}, {33, 4, true},
        {41, 2, true}, {41, 4, true}, {48, 1, false}, {48, 2, true}, {48, 4, true}
    };
    for (const Case &item : cases)
        assert(selected(item.orbitals, item.spins) == item.expected);
    setenv("OPENMX_SETDEN_FP64_SPLIT_SPIN", "0", 1);
    assert(!selected(33, 2) && !selected(48, 4));
    setenv("OPENMX_SETDEN_FP64_SPLIT_SPIN", "1", 1);
    assert(selected(13, 2) && selected(26, 4) && selected(48, 4));
    assert(!selected(48, 1));
    setenv("OPENMX_SETDEN_FP64_SPLIT_SPIN", "", 1);
    assert(!selected(32, 4) && selected(33, 2));
    puts("PASS: 23 native FP64 split policy cases: tier boundary, single spin, explicit 0/1 and empty overrides");
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-setden-policy-') as tmp:
    cfile, binary = Path(tmp)/'check.cpp', Path(tmp)/'check'
    cfile.write_text(code)
    subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++11', '-O2', '-Wall', '-Wextra',
                    '-Werror', str(cfile), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
