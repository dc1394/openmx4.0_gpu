#!/usr/bin/env python3
"""Check HIP CSR staging budgets and overflow guards (the DF image is in place)."""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / 'source/Set_Density_Grid_GPU.c').read_text()
header = (root / 'source/openmx_common.h').read_text()


def function(name):
    match = re.search(r'^static [^\n]*\b' + name + r'\([^;]*?\)\n\{', source, re.M)
    pos, depth = match.end(), 1
    while depth:
        depth += (source[pos] == '{') - (source[pos] == '}')
        pos += 1
    return source[match.start():pos]


def typedef(text, name):
    end = text.index('} ' + name + ';') + len('} ' + name + ';')
    return text[text.rfind('typedef struct {', 0, end):end]


prefix = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
typedef float Type_Orbs_Grid;
'''
main = r'''
int main(void) {
  int cases = 0;
  SDGLocalContext c;
  for(int resident=0;resident<2;resident++)
  for(int flags=0;flags<16;flags++) {
    memset(&c,0,sizeof(c));
    c.d_out_ptr=resident ? (uint32_t*)&c : NULL; c.extra_bytes=4096; c.dm_count=271;
    c.t.total_orbs0=123; c.t.total_orbs1=457; c.t.total_nolg=31; c.t.pair_count=7;
    c.t.orbs0_resident=(flags&1)!=0; c.t.orbs1_resident=(flags&2)!=0;
    c.t.nolg_resident=(flags&4)!=0; c.t.meta_resident=(flags&8)!=0;
    /* Independent accounting of every allocation made by Local_Run. */
    size_t want=resident ? 0:4096;
    if(!(flags&1)) want += 123 * sizeof(float);
    if(!(flags&2)) want += 457 * sizeof(float);
    if(!(flags&4)) want += 31 * sizeof(int);
    if(!(flags&8)) want += 7 * (2*sizeof(int)+4*sizeof(size_t));
    assert(SDG_local_call_bytes(&c)==want);
    cases++;
  }
  memset(&c,0,sizeof(c));
  c.dm_count=0; c.extra_bytes=SIZE_MAX-3; c.t.total_orbs0=1;
  assert(SDG_local_call_bytes(&c)==SIZE_MAX);
  c.extra_bytes=0; c.t.total_orbs0=SIZE_MAX;
  assert(SDG_local_call_bytes(&c)==SIZE_MAX);
  c.t.total_orbs0=0; c.t.total_nolg=SIZE_MAX;
  assert(SDG_local_call_bytes(&c)==SIZE_MAX);
  printf("PASS: %d CSR staging budgets (resident/transient tables, in-place double-float image) and 3 overflow guards\n",cases);
  return 0;
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-density-memory-') as tmp:
    cfile, binary = Path(tmp)/'check.c', Path(tmp)/'check'
    cfile.write_text(prefix + typedef(header, 'SetHamiltonianMETables') +
                     typedef(source, 'SDGLocalContext') + function('SDG_add_bytes') +
                     function('SDG_local_call_bytes') + main)
    subprocess.run(shlex.split(os.environ.get('CC', 'cc')) +
                   ['-std=c11', '-O2', '-Wall', '-Wextra', '-Werror'] +
                   shlex.split(os.environ.get('CFLAGS', '')) +
                   [str(cfile), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
