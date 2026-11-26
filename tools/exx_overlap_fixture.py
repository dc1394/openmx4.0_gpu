#!/usr/bin/env python3
"""
exx_overlap_fixture.py -- write the overlap fixture of the exx unit tests from a run of
the main program.

The unit test exx-two-center-overlap compares the two-centre integrals of modules/exx
(overlap kernel, PAOs) with the overlap matrix OLP[0] of the main program.  This script
collects everything the test needs into one text file:

  * the 1DFFT settings of the run (number of radial and reciprocal grid points, cutoff),
  * the unit cell and the atoms,
  * the radial functions of the PAOs that the run used, copied from the .pao files,
  * blocks of the overlap matrix, read from the .scfout file of the run.

How to make the .scfout file: run the main program with "HS.fileout on" on a small input
(scf.maxIter 1 is enough).  Example with the CPU application:

  openmx sich.dat
  python3 tools/exx_overlap_fixture.py --input sich.dat --scfout sich.scfout \\
      --pao-dir DFT_DATA19/PAO --output modules/exx/tests/fixtures/overlap_reference.txt

Only the standard library is used.  The .scfout layout is the one of source/read_scfout.c
(format version 3, native byte order).

Fixture format (plain text, one record per line, values in Hartree atomic units)

  exx_overlap_fixture 1
  grid <1DFFT.NumGridR> <1DFFT.NumGridK> <sqrt(1DFFT.EnergyCutoff)> <kmin>
  lattice <a1x a1y a1z a2x ... a3z>
  nspecies <count>
  species <index> <name> <cutoff radius> <mesh points> <shells>
  radii <index>                        # next line: the radial mesh of the .pao file
  shell <index> <l> <multiplicity>     # next line: the radial function on that mesh
  natoms <count>
  atom <index> <species index> <x y z>
  npairs <count>
  pair <atom a> <atom b> <R1 R2 R3> <rows> <columns> <distance>
                                       # next line: OLP block, row-major, rows on atom a at
                                       # the origin cell, columns on atom b in cell R
"""
import argparse
import math
import re
import struct
import sys


def read_input(path):
    """Species, atoms and 1DFFT settings of an OpenMX input file."""
    text = open(path).read()

    def block(name):
        m = re.search(r'<%s\s*\n(.*?)\n\s*%s>' % (re.escape(name), re.escape(name)), text, re.S | re.I)
        if not m:
            raise SystemExit('the input file has no block ' + name)
        return [line.split() for line in m.group(1).splitlines() if line.split()]

    def scalar(name, default):
        m = re.search(r'^\s*%s\s+(\S+)' % re.escape(name), text, re.M | re.I)
        return m.group(1) if m else default

    species = []
    for row in block('Definition.of.Atomic.Species'):
        name, basis = row[0], row[1]
        m = re.fullmatch(r'(.+?)-((?:[spdfgh]\d+)+)', basis)
        if not m:
            raise SystemExit('cannot read the basis specification ' + basis)
        counts = [0] * 6
        for letter, number in re.findall(r'([spdfgh])(\d+)', m.group(2)):
            counts['spdfgh'.index(letter)] = int(number)
        while counts and counts[-1] == 0:
            counts.pop()
        species.append({'name': name, 'pao': m.group(1), 'counts': counts})
    names = [s['name'] for s in species]
    if scalar('Atoms.SpeciesAndCoordinates.Unit', 'Ang').lower() != 'au':
        raise SystemExit('this script expects Atoms.SpeciesAndCoordinates.Unit AU')
    atoms = [names.index(row[1]) for row in block('Atoms.SpeciesAndCoordinates')]
    grid = {'NumGridR': int(scalar('1DFFT.NumGridR', '900')), 'NumGridK': int(scalar('1DFFT.NumGridK', '900')),
            'EnergyCutoff': float(scalar('1DFFT.EnergyCutoff', '3600.0'))}
    return species, atoms, grid


def read_pao(path, counts):
    """Radial mesh and the first counts[l] radial functions of every l from a .pao file."""
    text = open(path).read()
    cutoff = float(re.search(r'^radial\.cutoff\.pao\s+(\S+)', text, re.M).group(1))
    lmax = int(re.search(r'^PAO\.Lmax\s+(\d+)', text, re.M).group(1))
    mul = int(re.search(r'^PAO\.Mul\s+(\d+)', text, re.M).group(1))
    if len(counts) - 1 > lmax or max(counts) > mul:
        raise SystemExit(path + ' does not hold the requested orbitals')
    radii, shells = None, []
    for l, count in enumerate(counts):
        tag = 'pseudo.atomic.orbitals.L=%d' % l
        m = re.search(r'<%s\s*\n(.*?)\n\s*%s>' % (re.escape(tag), re.escape(tag)), text, re.S)
        rows = [line.split() for line in m.group(1).splitlines() if line.split()]
        r = [row[1] for row in rows]
        if radii is None:
            radii = r
        elif radii != r:
            raise SystemExit(path + ': the radial meshes of the angular momenta differ')
        for k in range(count):
            shells.append((l, k, [row[2 + k] for row in rows]))
    return cutoff, radii, shells


class Reader:
    def __init__(self, path):
        self.data = open(path, 'rb').read()
        self.pos = 0

    def ints(self, n):
        out = struct.unpack_from('=%di' % n, self.data, self.pos)
        self.pos += 4 * n
        return list(out)

    def doubles(self, n):
        out = struct.unpack_from('=%dd' % n, self.data, self.pos)
        self.pos += 8 * n
        return list(out)


def read_scfout(path):
    """Geometry, neighbour lists and the overlap matrix of a .scfout file (format version 3)."""
    f = Reader(path)
    atomnum, spin_version, _, _, _, tcpycell = f.ints(6)
    spinp, version = spin_version % 4, spin_version // 4
    if version != 3:
        raise SystemExit('unsupported .scfout format version %d' % version)
    f.ints(1)                                                   # order_max
    atv = [f.doubles(4)[1:] for _ in range(tcpycell + 1)]
    atv_ijk = [f.ints(4)[1:] for _ in range(tcpycell + 1)]
    norbs = f.ints(atomnum)
    fnan = f.ints(atomnum)
    natn = [f.ints(fnan[a] + 1) for a in range(atomnum)]         # 1-based atom numbers
    ncn = [f.ints(fnan[a] + 1) for a in range(atomnum)]
    tv = [f.doubles(4)[1:] for _ in range(3)]
    [f.doubles(4) for _ in range(3)]                             # reciprocal vectors
    xyz = [f.doubles(4)[1:] for _ in range(atomnum)]

    def matrix():
        out = []
        for a in range(atomnum):
            blocks = []
            for h in range(fnan[a] + 1):
                b = natn[a][h] - 1
                blocks.append([f.doubles(norbs[b]) for _ in range(norbs[a])])
            out.append(blocks)
        return out

    for _ in range(spinp + 1):
        matrix()                                                 # Kohn-Sham Hamiltonian
    if spinp == 3:
        for _ in range(3):
            matrix()                                             # its imaginary parts
    olp = matrix()
    return {'atv': atv, 'atv_ijk': atv_ijk, 'norbs': norbs, 'natn': natn, 'ncn': ncn, 'tv': tv, 'xyz': xyz,
            'olp': olp}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='input file of the run')
    ap.add_argument('--scfout', required=True, help='.scfout file of the run')
    ap.add_argument('--pao-dir', required=True, help='directory of the .pao files')
    ap.add_argument('--output', required=True, help='fixture to write')
    ap.add_argument('--nearest', type=int, default=8, help='neighbours kept per atom, by distance (default 8)')
    ap.add_argument('--stride', type=int, default=12,
                    help='of the remaining neighbours every stride-th one is kept (default 12)')
    ap.add_argument('--kmin', type=float, default=10e-7, help='Radial_kmin of the main program')
    args = ap.parse_args()

    species, atoms, grid = read_input(args.input)
    scf = read_scfout(args.scfout)
    if len(atoms) != len(scf['norbs']):
        raise SystemExit('the input file and the .scfout file have different numbers of atoms')
    out = ['exx_overlap_fixture 1',
           '# written by tools/exx_overlap_fixture.py from a run of the main program (HS.fileout on)',
           'grid %d %d %r %r' % (grid['NumGridR'], grid['NumGridK'], math.sqrt(grid['EnergyCutoff']), args.kmin),
           'lattice ' + ' '.join(repr(x) for v in scf['tv'] for x in v),
           'nspecies %d' % len(species)]
    for i, s in enumerate(species):
        cutoff, radii, shells = read_pao('%s/%s.pao' % (args.pao_dir, s['pao']), s['counts'])
        s['norbs'] = sum(2 * l + 1 for l, _, _ in shells)
        out.append('species %d %s %r %d %d' % (i, s['pao'], cutoff, len(radii), len(shells)))
        out.append('radii %d' % i)
        out.append(' '.join(radii))
        for l, k, values in shells:
            out.append('shell %d %d %d' % (i, l, k))
            out.append(' '.join(values))
    out.append('natoms %d' % len(atoms))
    for a, sp in enumerate(atoms):
        if scf['norbs'][a] != species[sp]['norbs']:
            raise SystemExit('atom %d: the number of orbitals differs between the two files' % (a + 1))
        out.append('atom %d %d %s' % (a, sp, ' '.join(repr(x) for x in scf['xyz'][a])))

    pairs = []
    for a in range(len(atoms)):
        found = []
        for h, (b1, rn) in enumerate(zip(scf['natn'][a], scf['ncn'][a])):
            b = b1 - 1
            d = [scf['xyz'][b][i] + scf['atv'][rn][i] - scf['xyz'][a][i] for i in range(3)]
            found.append((math.sqrt(sum(x * x for x in d)), h, b, scf['atv_ijk'][rn]))
        found.sort(key=lambda t: (t[0], t[2], t[3]))
        keep = found[:args.nearest + 1] + found[args.nearest + 1::args.stride]
        for dist, h, b, cell in keep:
            pairs.append((a, b, cell, dist, scf['olp'][a][h]))
    out.append('npairs %d' % len(pairs))
    count = 0
    for a, b, cell, dist, blockvalues in pairs:
        rows, cols = len(blockvalues), len(blockvalues[0])
        out.append('pair %d %d %d %d %d %d %d %.12f' % (a, b, cell[0], cell[1], cell[2], rows, cols, dist))
        out.append(' '.join('%.15e' % x for row in blockvalues for x in row))
        count += rows * cols
    with open(args.output, 'w') as f:
        f.write('\n'.join(out) + '\n')
    print('wrote %s: %d species, %d atoms, %d pairs, %d matrix elements'
          % (args.output, len(species), len(atoms), len(pairs), count))
    return 0


if __name__ == '__main__':
    sys.exit(main())
