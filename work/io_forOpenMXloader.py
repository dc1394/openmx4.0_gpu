# -*- coding: utf-8 -*-
#    BoltzTraP2, a program for interpolating band structures and calculating
#                semi-classical transport coefficients.
#    Copyright (C) 2017-2022 Georg K. H. Madsen <georg.madsen@tuwien.ac.at>
#    Copyright (C) 2017-2022 Jesús Carrete <jesus.carrete.montana@tuwien.ac.at>
#    Copyright (C) 2017-2022 Matthieu J. Verstraete <matthieu.verstraete@ulg.ac.be>
#    Copyright (C) 2018-2019 Genadi Naydenov <gan503@york.ac.uk>
#    Copyright (C) 2020 Gavin Woolman <gwoolma2@staffmail.ed.ac.uk>
#    Copyright (C) 2020 Roman Kempt <roman.kempt@tu-dresden.de>
#    Copyright (C) 2022 Robert Stanton <stantor@clarkson.edu>
#
#    This file is part of BoltzTraP2.
#
#    BoltzTraP2 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    BoltzTraP2 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with BoltzTraP2.  If not, see <http://www.gnu.org/licenses/>.

import glob
import logging
import os
import xml.etree.ElementTree as et

import ase
import ase.data
import netCDF4 as nc
import numpy as np
import numpy.linalg
import scipy as sp
import scipy.constants
import scipy.spatial.distance

import BoltzTraP2.misc
from BoltzTraP2.misc import ffloat
from BoltzTraP2.sphere import calc_reciprocal_iksubset
from BoltzTraP2.units import *


def _normalize_element_case(mixed):
    """Put a string in lowercase, capitalize the first character, and return
    the result. This is intended as a preprocessing step for case-insensitive
    element names from input files.
    """
    if len(mixed) == 0:
        return ""
    else:
        return mixed[0].upper() + mixed[1:].lower()


def read_GENE_eneandmat(filename):
    """Read in a .energy file in GENE format, describing the electronic bands.

    The file contains information about k points, band energies, and
    (optionally) momentum matrix elements.

    Args:
        filename: path to the .energy file

    Returns:
        A 5-tuple. The first element is the Fermi level. The second is the
        maximum occupancy per state. The third is an (nkpoints, 3) array with
        the coordinates of the irreducible k points of the system. The fourth
        contains the energy bands in an array. The fifth is either an array
        with the derivatives of the bands, or None if such information is not
        contained in the file.
    """
    lines = open(filename, "r").readlines()
    # Line 1: title string
    # Line 2: nk, nspin, Fermi level(Ry)
    linenumber = 1
    tmp = lines[linenumber].split()
    nk, nspin, efermi = int(tmp[0]), int(tmp[1]), ffloat(tmp[2])

    minband = np.infty
    ebands1 = []
    mommat1 = []
    kpoints = []
    for ispin in range(nspin):
        for ik in range(nk):
            # k block: line 1 = kx ky kz nband
            linenumber += 1
            tmp = lines[linenumber].split()
            nband = int(tmp[3])
            if nband < minband:
                minband = nband
            kpoints.append([ffloat(i) for i in tmp[0:3]])
            eband = []
            vband = []
            for ib in range(nband):
                linenumber += 1
                fields = lines[linenumber].split()
                e = ffloat(fields[0])
                if len(fields) == 4:
                    v = [ffloat(i) for i in fields[1:]]
                else:
                    v = []
                eband.append(e)
                vband.append(v)
            ebands1.append(eband)
            mommat1.append(vband)
    kpoints = np.array(kpoints)
    ebands1 = np.array(ebands1)
    mommat1 = np.array(mommat1)
    # When several spin channels are present, the full list of k points is
    # redundant.
    kpoints = kpoints[: kpoints.shape[0] // nspin, :]

    # Lists of bands for different spin channels at the same k point are
    # concatenated.
    ebands = np.empty((nk, nspin, minband))
    for ispin in range(nspin):
        for ik in range(nk):
            ebands[ik, ispin, :] = ebands1[ispin * nk + ik][:minband]
    ebands = ebands.reshape((nk, nspin * minband))
    if mommat1.ndim == 3 and mommat1.shape[2] == 3:
        mommat = np.empty((nk, nspin, minband, 3))
        for ispin in range(nspin):
            for ik in range(nk):
                mommat[ik, ispin, :, :] = mommat1[ispin * nk + ik][:minband]
        mommat = mommat.reshape((nk, nspin * minband, 3))
    else:
        mommat = None

    # Convert everything to Ha and Ha / bohr
    efermi *= 0.5
    ebands *= 0.5
    if mommat is not None:
        mommat *= 0.5
    if nspin == 1:
        dosweight = 2.0
    else:
        dosweight = 1.0
    return efermi, dosweight, kpoints, ebands.T, mommat


def read_GENE_struct(filename):
    """Read in a GENE .structure file.

    Such files contain lattice vectors, elements and atomic positions.

    Args:
        filename: path to the .structure file
          NB: naming was previously .struct, and now .structure to distinguish
          a complete file with all information needed for ASE
    Returns:
        An ASE Atoms object with crystal unit cell, 3D periodic, and appropriate atoms and positions
    """
    lines = open(filename, "r").readlines()
    # Line 1: title
    # Line 2-4: lattice vectors in bohr
    cell = np.empty((3, 3))
    for i in range(3):
        cell[i, :] = np.array([ffloat(j) for j in lines[i + 1].split()])
    # Line 5: number of atoms
    natom = int(lines[4].split()[0])
    # All remaining lines: element names and Cartesian coordinates in bohr
    cart = []
    symbolstring = ""
    for i in range(natom):
        fields = lines[5 + i].split()
        symbolstring += fields[0]
        cart.append([ffloat(j) for j in fields[1:4]])
    cart = np.array(cart)
    # Bundle everything in an Atoms object and return it
    return ase.Atoms(
        symbolstring,
        positions=cart / Angstrom,
        cell=cell / Angstrom,
        pbc=[1, 1, 1],
    )

def read_OpenMX_eneandmat(filename):
    """Read in a .energy file in OpenMX format, describing the electronic bands.

    The file contains information about k points, band energies, and
    (optionally) momentum matrix elements.

    Args:
        filename: path to the .energy file

    Returns:
        A 5-tuple. The first element is the Fermi level. The second is the
        maximum occupancy per state. The third is an (nkpoints, 3) array with
        the coordinates of the irreducible k points of the system. The fourth
        contains the energy bands in an array. The fifth is either an array
        with the derivatives of the bands, or None if such information is not
        contained in the file.
    """
    lines = open(filename, "r").readlines()
    # Line 1: title string
    # Line 2: nk, nspin, Fermi level(Ry)
    linenumber = 1
    tmp = lines[linenumber].split()
    #tmp.append(4)
    nk, nspin, efermi, sW = int(tmp[0]), int(tmp[1]), ffloat(tmp[2]), int(tmp[3])
    #original_sW = tmp[3]
    minband = np.infty
    ebands1 = []
    mommat1 = []
    kpoints = []
    for ispin in range(nspin):
        for ik in range(nk):
            # k block: line 1 = kx ky kz nband
            linenumber += 1
            tmp = lines[linenumber].split()
            nband = int(tmp[3])
            if nband < minband:
                minband = nband
            kpoints.append([ffloat(i) for i in tmp[0:3]])
            eband = []
            vband = []
            for ib in range(nband):
                linenumber += 1
                fields = lines[linenumber].split()
                e = ffloat(fields[0])
                if len(fields) == 4:
                    v = [ffloat(i) for i in fields[1:]]
                else:
                    v = []
                eband.append(e)
                vband.append(v)
            ebands1.append(eband)
            mommat1.append(vband)
    kpoints = np.array(kpoints)
    ebands1 = np.array(ebands1)
    mommat1 = np.array(mommat1)
    # When several spin channels are present, the full list of k points is
    # redundant.
    kpoints = kpoints[: kpoints.shape[0] // nspin, :]

    # Lists of bands for different spin channels at the same k point are
    # concatenated.
    ebands = np.empty((nk, nspin, minband))
    for ispin in range(nspin):
        for ik in range(nk):
            ebands[ik, ispin, :] = ebands1[ispin * nk + ik][:minband]
    ebands = ebands.reshape((nk, nspin * minband))
    if mommat1.ndim == 3 and mommat1.shape[2] == 3:
        mommat = np.empty((nk, nspin, minband, 3))
        for ispin in range(nspin):
            for ik in range(nk):
                mommat[ik, ispin, :, :] = mommat1[ispin * nk + ik][:minband]
        mommat = mommat.reshape((nk, nspin * minband, 3))
    else:
        mommat = None

   # Convert everything to Ha and Ha / bohr
    efermi *= 0.5
    ebands *= 0.5
    if mommat is not None:
        mommat *= 0.5
    if sW == 1 :
        dosweight = 1.0
    else:
        dosweight = 2.0
    return efermi, dosweight, kpoints, ebands.T, mommat

def read_OpenMX_struct(filename):
    """Read in a OpenMX .MXstructure file.

    Such files contain lattice vectors, elements and atomic positions.

    Args:
        filename: path to the .MXstructure file
          NB: naming was previously .struct, and now .MXstructure to distinguish
          a complete file with all information needed for ASE
    Returns:
        An ASE Atoms object with crystal unit cell, 3D periodic, and appropriate atoms and positions
    """
    lines = open(filename, "r").readlines()
    # Line 1: title
    # Line 2-4: lattice vectors in bohr
    cell = np.empty((3, 3))
    for i in range(3):
        cell[i, :] = np.array([ffloat(j) for j in lines[i + 1].split()])
    # Line 5: number of atoms
    natom = int(lines[4].split()[0])
    # All remaining lines: element names and Cartesian coordinates in bohr
    cart = []
    symbolstring = ""
    for i in range(natom):
        fields = lines[5 + i].split()
        symbolstring += fields[0]
        cart.append([ffloat(j) for j in fields[1:4]])
    cart = np.array(cart)
    # Bundle everything in an Atoms object and return it
    return ase.Atoms(
        symbolstring,
        positions=cart / Angstrom,
        cell=cell / Angstrom,
        pbc=[1, 1, 1],
    )

def read_CASTEP_bands(filename):
    """Read in a CASTEP .bands file in GENE format, describing the electronic bands.

    The file contains information about k-points, Fermi, energy, band energies,
    and (optionally) momentum matrix elements.

    Args:
        filename: path to the .bands file

    Returns:
        A 7-tuple. The first element is the Fermi level. The second checks for
        different Fermi energy levels in a spin-polarized calculation. The third
        is the maximum occupancy per state. The fourth is an array (nkpoints, 3)
        with the coordinates of the irreducible k points of the system. The fifth
        contains the energy bands in an array. The sixth is the number of electrons
        in the system. The seventh is either an array with the derivatives of the
        bands, or None if such information is not contained in the file.
    """

    lines = open(filename, "r").readlines()
    kpoints = []
    for line in lines:
        if "Number of k-points" in line:
            nk = int(line.split()[3])
        elif "Number of spin components" in line:
            nspin = int(line.split()[4])
        elif "Number of electrons" in line:
            if nspin == 1:
                nelect = ffloat(line.split()[3])
            else:
                nelect_up = ffloat(line.split()[3])
                nelect_down = ffloat(line.split()[4])
                nelect = nelect_up + nelect_down
        elif "Number of eigenvalues" in line:
            nband = int(line.split()[3])
        elif "Fermi energy" in line:
            # .bands output is in Hartree
            efermi = ffloat(line.split()[5])
            diff_efermi = False
        # If spin polarised, use spin-up energy
        elif "Fermi energies" in line:
            efermi = ffloat(line.split()[5])
            # Perform a Fermi level check
            efermi_up = ffloat(line.split()[5])
            efermi_down = ffloat(line.split()[6])
            if efermi_up != efermi_down:
                diff_efermi = True
            else:
                diff_efermi = False
        elif "K-point" in line:
            kpoints.append([ffloat(line.split()[j]) for j in range(2, 5)])

    kpoints = np.array(kpoints)

    ebands = []
    ebands_up = []
    ebands_down = []
    for index, line in enumerate(lines):
        if "Spin component 1" in line:
            eig_start_line1 = index + 1
            ebands1 = []
            for i in range(nband):
                ebands1.append(ffloat(lines[eig_start_line1].split()[0]))
                eig_start_line1 += 1
            ebands1 = np.array(ebands1)
            ebands_up.append(ebands1)
        elif "Spin component 2" in line:
            eig_start_line2 = index + 1
            ebands2 = []
            for i in range(nband):
                ebands2.append(ffloat(lines[eig_start_line2].split()[0]))
                eig_start_line2 += 1
            ebands2 = np.array(ebands2)
            ebands_down.append(ebands2)

    # Lists of bands for different spin channels at the same k point are
    # concatenated.
    if nspin == 1:
        dosweight = 2.0
        ebands = np.array(ebands_up).T
    else:
        dosweight = 1.0
        ebands = np.concatenate(
            (np.array(ebands_up).T, np.array(ebands_down).T)
        )

    # Reading momentum matrices is not implemented at the moment.
    mommat = None

    return efermi, diff_efermi, dosweight, kpoints, ebands, nelect, mommat


def read_CASTEP_output(filename):
    """Read in a CASTEP structure in a .castep file.

    Such files contain lattice vectors, elements and atomic positions.

    Args:
        filename: path to the .castep file

    Returns:
        An ASE Atoms object with crystal unit cell, 3D periodic,
        and appropriate atoms and positions.
    """
    lines = open(filename, "r").readlines()

    # lattice vectors in Angs
    cell = []
    for index, line in enumerate(lines):
        if "Real Lattice(A)" in line:
            start = index + 1
            for j in range(3):
                cell.append(
                    [ffloat(lines[start].split()[k]) for k in range(0, 3)]
                )
                start += 1
            break  # avoid double counting in continuation runs
    cell = np.array(cell)

    # Check if symmtery operations are used in CASTEP.
    # Check if Hubbard U is present and if it reduces symmetry.
    noncollinear = False
    HubbU = False
    HubbU_warning = False
    for line in lines:
        if "treating system as non-collinear" in line:
            noncollinear = True
        elif "Units for Hubbard U values" in line:
            HubbU = True
        elif "Total number of ions in cell" in line:
            natoms = int(line.split()[7])
        elif "Number of symmetry operations" in line:
            nsymmops = int(line.split()[5])
            if nsymmops != 1:
                symmetry_castep = True
            else:
                symmetry_castep = False
        elif "There are no symmetry operations specified" in line:
            symmetry_castep = False
        elif "Hubbard U data has reduced the symmetry" in line:
            HubbU_warning = True

    # CASTEP Hubbard U calculations with symmetry_generate are not
    # currently implemented in BoltzTraP2.
    # Raise an error if the Hubbard U has reduced the symmetry and
    # if there is more than one symmetry operation.
    if HubbU_warning and symmetry_castep:
        raise ValueError(
            "Hubbard U found in CASTEP calculation. "
            "Please repeat the CASTEP run without symmetry_generate."
        )

    # All remaining lines: element names and fractional coordinates
    frac_coordinates = []
    symbolstring = ""
    for index, line in enumerate(lines):
        if "Cell Contents" in line:
            for i in range(natoms):
                fields = lines[5 + i].split()
                symbolstring += str(lines[index + 10 + i].split()[1])
                frac_coordinates.append(
                    [
                        ffloat(lines[index + 10 + i].split()[j])
                        for j in range(3, 6)
                    ]
                )
            break  # avoid double counting in continuation runs
    frac_coordinates = np.array(frac_coordinates)

    # Add magmom for collinear spin-polarised calculations
    # The are no symmerties for non-collinear calculations in CASTEP
    magmom = None
    for index, line in enumerate(lines):
        if "Initial magnetic" in line:
            magmom = []
            start = index + 3
            for j in range(natoms):
                magmom.append(ffloat(lines[start].split()[4]))
                start += 1
            magmom = np.array(magmom)
            break  # avoid double counting in continuation runs

    # Create fictitious "magmom" vectors for non-collinear
    # spin-polarised calculations to break the symmetry.
    fict_magmom = None
    if (noncollinear or HubbU) and not symmetry_castep:
        fict_magmom = []
        for i in range(natoms):
            temp_magmom = [i + 1.0, i + 2.0, i + 3.0]
            fict_magmom.append(temp_magmom)
        fict_magmom = np.array(fict_magmom)

    return (
        ase.Atoms(
            symbolstring,
            scaled_positions=frac_coordinates,
            cell=cell,
            pbc=[1, 1, 1],
        ),
        magmom,
        fict_magmom,
    )


def read_GPAW(filename):
    """
    Uses GPAW loader to load in a previously finished calculation.
    Args:
        filename: Name of file, must be in format "yourfilename.gpw"
    Returns:
        Everything is returned in one function to reduce time taken for GPAW loader to reload larger calculations (especially if the user has saved wavefunctions).
        atoms: ASE atoms object
        fermi: Fermi level in Ha
        dosweight: Max band occupation (1 if unpolarized calc, 2 if collinear, noncollinear not yet implemented)
        kpoints: (nkpt, 3) numpy array containing k points in the IBZ
        ebands: (nband, nkpt) numpy array containing eigenvalues at each k in kpoints
        nelect: Number of electrons
        sysname: Name of GPAW calculation,  filename[0:-4] : "yourfilename.gpw" -> "yourfilename" as sysname

    """

    from gpaw import GPAW

    calc = GPAW(filename)

    atoms = calc.atoms
    fermi = calc.get_fermi_level() * BoltzTraP2.units.eV
    nspin = calc.get_number_of_spins()

    if float(nspin) == 1.0:
        dosweight = 2
    elif float(nspin) == 2.0:
        dosweight = 1

    nkpoints = len(calc.get_ibz_k_points())

    kpoints = calc.get_ibz_k_points()

    nbands = calc.get_number_of_bands()
    ebands = np.zeros((nkpoints, nbands))
    for i in range(nkpoints):
        ebands[i] = calc.get_eigenvalues(i) * BoltzTraP2.units.eV
    sysname = filename[0:-4]
    nelect = calc.setups.nvalence - calc.parameters.charge

    return (atoms, fermi, dosweight, kpoints, ebands.T, nelect)


def W2Kmommat(filename, kpoints):
    """Read the contents of a Wien2k .mommat2 file.

    Args:
        filename: path to the .mommat2 file.
        kpoints: array with the k points of the same system.

    Returns:
        A 3-tuple. The first element is an array with the momentum matrix
        elements. The second and third are integers delimiting which bands
        have derivative information available.
    """
    nk = len(kpoints)

    fmat = open(filename, "r")
    matlines = fmat.readlines()
    fmat.close()
    il = 2
    mommat = []
    brk = []
    for ik in range(nk):
        nemin = int(matlines[il].split()[5])
        nemax = int(matlines[il].split()[6])
        il += 2
        brk += [[nemin, nemax]]
        ne = nemax - nemin + 1
        mmk = np.zeros((ne, 3), dtype=complex)
        ib1 = 0
        for ib in range(ne):
            line = matlines[il + ib1]
            for i in range(3):
                mmk[ib, i] = complex(
                    float(line[11 + 2 * i * 13 : 24 + 2 * i * 13]),
                    float(line[24 + 2 * i * 13 : 37 + 2 * i * 13]),
                )
            ib1 += ne - ib
        mommat += [mmk]
        il += ib1 + 1
    nemax = np.min(brk, axis=0)[1]
    nemin = np.max(brk, axis=0)[0]
    mommat2 = []
    for ik in range(nk):
        mommat2 += [mommat[ik][nemin - brk[ik][0] : nemax - brk[ik][0] + 1]]
    return np.array(mommat2), nemin, nemax


def W2Kene(filename, conv):
    """Read the contents of a Wien2k .energy file.

    Args:
        filename: path to the .energy file.
        conv: result of passing the lattice vectors of the system through
            ase.io.wien2k_c2p.

    Returns:
        A 2-tuple. The first element is an (nkpoints, 3) array with the
        coordinates of the irreducible k points of the system. The second
        contains the energy bands in an array.
    """
    f = open(filename, "r", encoding="ascii")
    lines = f.readlines()
    f.close()
    linenumber = 0
    k = np.zeros(3)
    while True:
        try:
            ll = lines[linenumber]
            k[0], k[1], k[2], nband = (
                float(ll[:19]),
                float(ll[19:38]),
                float(ll[38:57]),
                int(ll[73:79]),
            )
            break
        except Exception:
            linenumber += 1

    minband = np.infty
    ebands1 = []
    kpoints = []
    while True:
        try:
            ll = lines[linenumber]
            k[0], k[1], k[2], nband = (
                float(ll[:19]),
                float(ll[19:38]),
                float(ll[38:57]),
                int(ll[73:79]),
            )
            nband = int(nband)
            if nband < minband:
                minband = nband
            linenumber += 1
            eband = []
            for i in range(nband):
                e = ffloat(lines[linenumber].split()[1])
                eband += [e]
                linenumber += 1
            ebands1 += [np.array(eband)]
            kpoints += [k.copy()]
        except Exception:
            break
    kpoints = kpoints @ conv
    ebands = np.zeros((len(kpoints), minband))
    for i in range(len(kpoints)):
        ebands[i] = ebands1[i][:minband]
    return kpoints, ebands.T * 0.5  # To Ha


def W2Kfermi(filename):
    """Read the value of the Fermi level from a Wien2k .scf file.

    Args:
        filename: path to the .scf file.

    Returns:
        The value of the Fermi level.
    """
    nruter = None
    with open(filename) as f:
        for l in f:
            if l.startswith(":FER"):
                nruter = 0.5 * float(l[38:53])
    return nruter


# Small utility functions for parsing VASP's xml files.
def _parse_xml_array(line, cast=float):
    """Transform a line of text with fields separated by spaces into a list
    of homogeneous objects.
    """
    return [cast(i) for i in line.split()]


# Specialized parsers for each piece of information
def _detect_vasp_broken_interpolation(xml_tree):
    """Interpolation without group velocities is buggy in versions of VASP
    older than 5.4.4. Detect if the xml_tree passed in as argument comes from
    one of those calculations.
    """
    xml_eigen = xml_tree.find(
        './calculation/eigenvalues[@comment="interpolated_ibz"]/'
    )
    if xml_eigen is not None and not _parse_vasp_lvel(xml_tree):
        return True
    return False


def _detect_vasp_interpolated_velocities(xml_tree):
    """Return true if the XML tree provided as an argument originates from a
    VASP calculation using interpolation and providing group velocities. In
    that case, the list of k points on the interpolated grid will need manual
    reduction after reading it from vasprun.xml.
    """
    candidates = (
        './calculation/eigenvelocities[@comment="interpolated"]/',
        './calculation/eigenvalues[@comment="interpolated_ibz"]'
        "/electronvelocities",
    )
    for i in candidates:
        if xml_tree.find(i) is not None:
            return True
    return False


def _parse_vasp_name(xml_tree):
    """Extract the system name from an XML ElementTree representing a
    vasprun.xml file.
    """
    xml_general = xml_tree.find('./parameters/separator[@name="general"]')
    return xml_general.find('./i[@name="SYSTEM"]').text


def _parse_vasp_fermi(xml_tree):
    """Extract the Fermi level from an XML ElementTree representing a
    vasprun.xml file.
    """
    xml_dos = xml_tree.find("./calculation/dos")
    return float(xml_dos.find('./i[@name="efermi"]').text)


def _parse_vasp_magmom(xml_tree):
    """Return an appropriate value of the magnetic moments based on the values
    of ISPIN, LNONCOLLINEAR, MAGMOM and SAXIS in an XML ElementTree representing
    a vasprun.xml file.
    """
    xml_spin = xml_tree.find(
        "./parameters"
        '/separator[@name="electronic"]'
        '/separator[@name="electronic spin"]'
    )
    ispin = int(xml_spin.find('./i[@name="ISPIN"]').text.strip())
    noncollinear = (
        xml_spin.find('./i[@name="LNONCOLLINEAR"]').text.strip() == "T"
    )
    magmom = np.array(
        _parse_xml_array(xml_spin.find('./v[@name="MAGMOM"]').text.strip())
    )
    saxis = np.array(
        _parse_xml_array(xml_spin.find('./v[@name="SAXIS"]').text.strip())
    )
    if noncollinear:
        # Non-collinear calculation, SAXIS must be taken into account to
        # extract the Cartesian components of the magnetic moments.
        # Reference:
        # https://cms.mpi.univie.ac.at/wiki/index.php/SAXIS
        magmom0 = magmom.reshape((-1, 3))
        alpha = np.arctan2(saxis[1], saxis[0])
        beta = np.arctan2(np.hypot(saxis[1], saxis[0]), saxis[2])
        magmom = np.empty_like(magmom0)
        magmom[:, 0] = (
            np.cos(beta) * np.cos(alpha) * magmom0[:, 0]
            - np.sin(alpha) * magmom0[:, 1]
            + np.sin(beta) * np.cos(alpha) * magmom0[:, 2]
        )
        magmom[:, 1] = (
            np.cos(beta) * np.sin(alpha) * magmom0[:, 0]
            + np.cos(alpha) * magmom0[:, 1]
            + np.sin(beta) * np.sin(alpha) * magmom0[:, 2]
        )
        magmom[:, 2] = (
            -np.sin(beta) * magmom0[:, 0] + np.cos(beta) * magmom0[:, 2]
        )
    elif ispin == 1:
        # Unpolarized calculation
        magmom = None
    return magmom


def _parse_vasp_nelect(xml_tree):
    """Extract NELECT from an XML ElementTree representing a vasprun.xml file."""
    xml_electronic = xml_tree.find("./parameters").find(
        './separator[@name="electronic"]'
    )
    return float(xml_electronic.find('./i[@name="NELECT"]').text)


def _parse_vasp_kinter(xml_tree):
    """Extract KINTER from an XML ElementTree representing a vasprun.xml file.
    If this parameter is not present, return 0.
    """
    xml_incar = xml_tree.find("./incar")
    nruter = 0
    for e in xml_incar:
        attribs = e.attrib
        if (
            "type" in attribs
            and attribs["type"] == "int"
            and attribs["name"] == "KINTER"
        ):
            nruter = int(e.text.strip())
            break
    return nruter


def _parse_vasp_lvel(xml_tree):
    """Extract LVEL from an XML ElementTree representing a vasprun.xml file.
    If this parameter is not present, return False.
    """
    xml_incar = xml_tree.find("./incar")
    nruter = False
    for e in xml_incar:
        attribs = e.attrib
        if (
            "type" in attribs
            and attribs["type"] == "logical"
            and attribs["name"] == "LVEL"
        ):
            nruter = e.text.strip() == "T"
            break
    return nruter


def _parse_vasp_structure(xml_tree):
    """Extract the structural information from an XML ElementTree representing a
    vasprun.xml file, and return it as an ASE atoms object.
    """
    # Read in the lattice vectors, reduced positions and chemical elements.
    lattvec = np.empty((3, 3))
    positions = []
    elements = []
    xml_basis = xml_tree.find(
        './calculation/structure/crystal/varray[@name="basis"]'
    )
    for i, line in enumerate(xml_basis):
        lattvec[i, :] = _parse_xml_array(line.text)
    xml_positions = xml_tree.find(
        './calculation/structure/varray[@name="positions"]'
    )
    for line in xml_positions:
        positions.append(_parse_xml_array(line.text))
    positions = np.array(positions)
    xml_elements = xml_tree.iterfind('./atominfo/array[@name="atoms"]/set/rc')
    for element in xml_elements:
        elements.append(element.find("./c").text.strip())
    # Build and return an ase atoms object
    cartesian = positions @ lattvec
    atoms = ase.Atoms(
        elements, positions=cartesian, cell=lattvec, pbc=[1, 1, 1]
    )
    return atoms


def _get_vasp_kpoints_path(xml_tree):
    """Obtain the path to the list of irreducible k points to be used, which
    depends on whether interpolation was used, on the version of VASP, and on
    whether velocities are present.
    """
    kinter = _parse_vasp_kinter(xml_tree)
    # If the calculation does not use interpolation, use the regular list
    # of k points.
    if kinter == 0:
        return './kpoints/varray[@name="kpointlist"]'
    # Otherwise, try several possible datasets
    candidates = (
        './calculation/eigenvalues[@comment="interpolated"]/'
        'kpoints/varray[@name="kpointlist"]',
        './calculation/eigenvelocities[@comment="interpolated"]/'
        'kpoints/varray[@name="kpointlist"]',
        './calculation/eigenvalues[@comment="interpolated_ibz"]'
        '/electronvelocities/kpoints/varray[@name="kpointlist"]',
    )
    for i in candidates:
        xml_kpoints = xml_tree.find(i)
        if xml_kpoints is not None:
            return i
    raise ValueError(
        "vasprun.xml uses interpolation, but no suitable list of k "
        "points was found"
    )


def _get_vasp_energies_path(xml_tree):
    """Obtain the path to the list of eigenenergies, which depends on
    whether interpolation was used and on the version of VASP.
    """
    kinter = _parse_vasp_kinter(xml_tree)
    if kinter == 0:
        return "./calculation/eigenvalues/array/set"
    candidates = (
        './calculation/eigenvalues[@comment="interpolated"]/'
        "eigenvalues/array/set",
        './calculation/eigenvelocities[@comment="interpolated"]/'
        "eigenvalues/array/set",
        './calculation/eigenvalues[@comment="interpolated_ibz"]'
        "/electronvelocities/eigenvalues/array/set",
    )
    for i in candidates:
        xml_energ = xml_tree.find(i)
        if xml_energ is not None:
            return i
    raise ValueError(
        "vasprun.xml uses interpolation, but no suitable list of band "
        "energies was found"
    )


def _get_vasp_velocities_path(xml_tree):
    """Obtain the path to the list of group velocities, which depends on
    whether interpolation was used and on the version of VASP.
    """
    kinter = _parse_vasp_kinter(xml_tree)
    if kinter == 0:
        candidates = ("./calculation/electronvelocities",)
    else:
        candidates = (
            "./calculation/eigenvelocities",
            './calculation/eigenvalues[@comment="interpolated_ibz"]'
            "/electronvelocities",
        )
    for i in candidates:
        xml_vels = xml_tree.find(i)
        if xml_vels is not None:
            return i
    return None


def _parse_vasp_ikpoints(xml_tree):
    """Extract a list of "irreducible" k points from an XML ElementTree
    representing a vasprun.xml file, and return it as an (nkpoints, 3) numpy
    array. The list may actually need further reducing, specifically when
    KINTER != 0 and vasprun.xml contains velocities.
    """
    path = _get_vasp_kpoints_path(xml_tree)
    xml_kpoints = xml_tree.find(path)
    kpoints = []
    for p in xml_kpoints:
        kpoints.append(_parse_xml_array(p.text))
    kpoints = np.array(kpoints)
    return kpoints


def _parse_vasp_eigenvalues(xml_tree):
    """Extract a list of eigenvalues at each irreducible k point from an XML
    ElementTree representing a vasprun.xml file, and return it as an
    (nspin, nkpoints, nbands) numpy array.
    """
    path = _get_vasp_energies_path(xml_tree)
    xml_eigenvalues = xml_tree.find(path)
    data = []
    for spin in xml_eigenvalues:
        dspin = []
        for point in spin:
            dpoint = []
            for band in point:
                dpoint.append(_parse_xml_array(band.text)[0])
            dspin.append(dpoint)
        data.append(dspin)
    data = np.array(data)
    return data


def _parse_vasp_velocities(xml_tree):
    """Extract a list of k points and a list of group velocities at those
    points from an XML ElementTree representing a vasprun.xml file. Return
    them as a tuple of two numpy arrays, with shapes (nkpoints, 3) and
    (nspin, nkpoints, nbands, 3). Note   that these k points are not symmetry
    reduced. Return None if the tree does not contain information about group
    velocities.
    """
    path = _get_vasp_velocities_path(xml_tree)
    if path is None:
        return None
    xml_vels = xml_tree.find(path)
    xml_kpoints = xml_vels.find("./kpoints/varray")
    kpoints = []
    for p in xml_kpoints:
        kpoints.append(_parse_xml_array(p.text))
    # The three axes of the data are ordered like in (spin, point, band),
    # from most to least significant.
    xml_data = xml_vels.find("./eigenvalues/array/set")
    data = []
    for spin in xml_data:
        dspin = []
        for point in spin:
            dpoint = []
            for band in point:
                dpoint.append(_parse_xml_array(band.text)[1:])
            dspin.append(dpoint)
        data.append(dspin)
    return (np.array(kpoints), np.array(data))


def parse_vasprunxml(filename):
    """Parse a vasprun.xml file to extract all structural information, the
    coordinates of the q points used in the calculation, the energy eigenvalues
    at those points, and their gradients (group velocities) if available.

    For vasp to save the group velocities, both LOPTICS and LVEL must have been
    set to .TRUE. in the INCAR file.

    Return the results in a dictionary with the following keys:
    name: the system name (often listed as "unknown system" in vasprun.xml)
    atoms: an ASE atoms object representing the structure
    nelect: number of valence electrons in the simulation cell
    kpoints: numpy array with shape (nk, 3) containing direct coordinates of an
             irreducible set of k points.
    fermi: the Fermi level in eV
    magmom: the magnetic moments in a format suitable for the loaders in dft.py
    E: numpy array with shape (nspin, npoints, nbands) containing the energies
       of the electronic eigenstates at each point
    v: numpy array with shape (nspin, npoints, nbands, 3) containing the
       gradient of E with respect to k at each point. This key is only present
       if the vasprun.xml file contains group velocities.
    """
    # Open the file and parse it with the builtin parser
    xml_tree = et.parse(filename)
    # Check for interpolated bands created with problematic version of VASP.
    if _detect_vasp_broken_interpolation(xml_tree):
        raise ValueError(
            "the interpolated band data in vasprun.xml is "
            "incorrect: KINTER!=0 and LVEL=F, but the file was "
            "created with an old version of VASP"
        )
    # Obtain each piece of information using the specialized parsing.
    nruter = dict(
        fermi=_parse_vasp_fermi(xml_tree),
        name=_parse_vasp_name(xml_tree),
        atoms=_parse_vasp_structure(xml_tree),
        nelect=_parse_vasp_nelect(xml_tree),
        magmom=_parse_vasp_magmom(xml_tree),
        kpoints=_parse_vasp_ikpoints(xml_tree),
        E=_parse_vasp_eigenvalues(xml_tree),
    )
    # Check if the list of k points needs to be reduced, and do it
    toreduce = _detect_vasp_interpolated_velocities(xml_tree)
    if toreduce:
        subset = calc_reciprocal_iksubset(
            nruter["atoms"], nruter["magmom"], nruter["kpoints"]
        )
        nruter["kpoints"] = np.ascontiguousarray(nruter["kpoints"][subset, :])
    # Group velocities require special care. They may not be present, and
    # even if they are we need to extract the subset of data corresponding
    # to the irreducible k points.
    res = _parse_vasp_velocities(xml_tree)
    if res is not None:
        k, v = res
        # Obtain the index of those k points from the full array that are also
        # in the array of irreducible k points.
        reduk = nruter["kpoints"] % 1.0
        fullk = k % 1.0
        distances = sp.spatial.distance.cdist(reduk, fullk)
        indices = []
        for i in range(reduk.shape[0]):
            pos = distances[i, :].argmin()
            if distances[i, pos] > 1e-6:
                raise ValueError("inconsistent sets of k points")
            indices.append(pos)
        if len(set(indices)) != len(indices):
            raise ValueError("inconsistent sets of k points")
        nruter["v"] = v[:, indices, :, :]
        # In vasprun.xml files with KINTER != 0 and including velocities,
        # this also affects the list of energies.
        if toreduce:
            nruter["E"] = nruter["E"][:, indices, :]
    return nruter


# Small utility functions for parsing ABINIT's netcdf files.
# Specialized parsers for each piece of information
def _parse_abinit_structure(gsrfile):
    """Extract the structural information from a
    GSR netcdf file, and return it as an ASE atoms object.
    """
    # Read in the lattice vectors, reduced positions and chemical elements.
    typat = gsrfile["atom_species"][:].copy()
    atom_names = nc.chartostring(gsrfile["atom_species_names"][...])
    elements = []
    for ia in typat:
        elements.append(atom_names[ia - 1])
    # Build and return an ase atoms object.
    pcell_angstr = gsrfile["primitive_vectors"][:, :].copy() / Angstrom
    cartesian = gsrfile["reduced_atom_positions"][:, :] @ pcell_angstr
    atoms = ase.Atoms(
        elements, positions=cartesian, cell=pcell_angstr, pbc=[1, 1, 1]
    )
    return atoms


def _parse_abinit_velocities(gsrfile):
    """Extract a list of k points and a list of group velocities at those
    points from 1WF fortran files (?). Return
    them as a tuple of two numpy arrays, with shapes (nkpoints, 3) and
    (nspin, nkpoints, nbands, 3). Note that these k points are not symmetry
    reduced. Return None if the tree does not contain information about group
    velocities.
    """
    raise ValueError("Not implemented yet")
    return


def parse_abinitrun(filename):
    """Parse a abinit GSR.nc file to extract all structural information, the
    coordinates of the q points used in the calculation, the energy eigenvalues
    at those points, and their gradients (group velocities) if available.

    Return the results in a dictionary with the following keys:
    name = the system name
    atoms: an ASE atoms object representing the structure
    nelect: number of valence electrons in the simulation cell
    kpoints: numpy array with shape (nk, 3) containing direct coordinates of an
             irreducible set of k points.
    fermi: the Fermi level in eV
    E: numpy array with shape (nspin, npoints, nbands) containing the energies
       of the electronic eigenstates at each point
    """
    # To have the group velocities, abinit will have run another dataset with
    # 1WFx files containing the matrix elements. 03/2018 - a netcdf format for
    # these is in the works. TODO: add support.
    # v: numpy array with shape (nspin, npoints, nbands, 3) containing the
    #    gradient of E with respect to k at each point. This key is only present
    #    if there are also 1WF files with the band velocities

    # Open the file and parse it with the builtin parser
    gsrfile = nc.Dataset(filename, mode="r")

    # Obtain each piece of information from the corresponding function.
    nruter = dict(
        fermi=gsrfile["fermie"].getValue().copy(),
        name="abinit GSR file import for BT2",
        atoms=_parse_abinit_structure(gsrfile),
        nelect=gsrfile["nelect"].getValue().copy(),
        kpoints=gsrfile["reduced_coordinates_of_kpoints"][:, :].copy(),
        E=gsrfile["eigenvalues"][:, :, :].copy(),
    )

    gsrfile.close()

    # Group velocities require special care. They may not be present, and
    # even if they are we need to extract the subset of data corresponding
    # to the irreducible k points.
    res = None  # _parse_abinit_velocities(1WFfiles)
    if res is not None:
        k, v = res
        # Obtain the index of those k points from the full array that are also
        # in the array of irreducible k points.
        reduk = nruter["kpoints"] % 1.0
        fullk = k % 1.0
        distances = sp.spatial.distance.cdist(reduk, fullk)
        indices = []
        for i in range(reduk.shape[0]):
            pos = distances[i, :].argmin()
            if distances[i, pos] > 1e-6:
                raise ValueError("inconsistent sets of k points")
            indices.append(pos)
        if len(set(indices)) != len(indices):
            raise ValueError("inconsistent sets of k points")
        nruter["v"] = v[:, indices, :, :]

    return nruter


class AIMSReader:
    """Reader for AIMS calculations.

    This class simply wraps all reading functions for AIMS calculations,
    because not all parameters can be obtained from a single file.

    Args:
        directory (str): Directory for processing.

    Attributes:
        parameters (dict): Collection of parameters.
        atoms (atoms): ASE atoms object of the structure.
        kpoints (ndarray): A (nkpoints, 3) numpy array of the fractional
            k-coordinates folded into the 1st Brillouin Zone.
    """

    def __init__(self, directory):
        outputfile = self._get_AIMS_output(directory)
        self.parameters = self.read_AIMS_output(outputfile)
        self.atoms = self.read_AIMS_struct(
            os.path.join(directory, "geometry.in")
        )
        self.kpoints = self.read_AIMS_kpoints(
            os.path.join(directory, "Final_KS_eigenvalues.dat")
        )
        (
            self.fermi_level,
            self.dosweight,
            self.eigenvalues,
        ) = self.read_AIMS_energies(
            os.path.join(directory, "Final_KS_eigenvalues.dat")
        )
        toreduce = True
        if toreduce:
            logging.info("Reducing {} kpoints...".format(len(self.kpoints)))
            subset = calc_reciprocal_iksubset(self.atoms, None, self.kpoints)
            # currently I do not see a way to extract magnetic
            # moments correctly.
            if self.parameters["spin"] == "collinear":
                logging.warning(
                    """Post-SCF magnetic moments are not printed to the outputfile.
                Symmetry reduction is performed without taking these into account.
                Hence, results might be wrong."""
                )
            self.kpoints = np.ascontiguousarray(self.kpoints[subset, :])
            self.eigenvalues = np.ascontiguousarray(
                self.eigenvalues[:, subset, :]
            )
            logging.info("Reduced to {} kpoints.".format(len(self.kpoints)))

    def read_AIMS_struct(self, filename="geometry.in"):
        """Read in a geometry.in file in AIMS format.

        The file contains information about lattice vectors (in Angström),
            elements and positions (in angstrom), as well as initial magnetic
            moments.

        Args:
            filename (str): path to the geometry.in file

        Returns:
            atoms: An ASE Atoms object describing the structure contained in
                the file.
        """
        return ase.io.read(filename)

    def read_AIMS_kpoints(self, filename="Final_KS_eigenvalues.dat"):
        """Read in a Final_KS_eigenvalues.dat file in AIMS format.

        The file contains blocks of k-points with eigenvalues and occupations.
        FHI-AIMS writes out the eigenvalues for the entire first Billouin Zone.
        By default, it runs on a Gamma-centered grid.
        If SOC is enabled, an additional file called
        Final_KS_eigenvalues.dat.no_soc is present containing unperturbed
        eigenvalues.

        Args:
            filename (str): path to the Final_KS_eigenvalues.dat file

        Returns:
            An (nkpoints, 3) array with the coordinates with the k points in
                the file.
        """

        with open(filename, "r") as file:
            content = file.readlines()

        kpoints = [
            line.split()[-3:]
            for line in content
            if "k-point in recip. lattice units:" in line
        ]
        kpoints = np.array(kpoints, dtype=float)
        for row in range(kpoints.shape[0]):  # folding back into first BZ
            if kpoints[row][0] > 0.5:
                kpoints[row][0] -= 1
            if kpoints[row][1] > 0.5:
                kpoints[row][1] -= 1
            if kpoints[row][2] > 0.5:
                kpoints[row][2] -= 1

        self.parameters["nkpoints"] = len(kpoints)
        return kpoints

    def read_AIMS_energies(self, filename="Final_KS_eigenvalues.dat"):
        """Read in a Final_KS_eigenvalues.dat file in AIMS format.

        The file contains blocks of k-points with eigenvalues and occupations.
        If SOC is enabled, every eigenvalue is split and singly occupied.
        If collinear spin is enabled, there is an additional block for spin
        up / spin down.

        Args:
            filename (str): path to the Final_KS_eigenvalues.dat file

        Returns:
            A 3-tuple. The first element is the Fermi level in Hartree, the
            second is the spin degeneracy of each energy (1.0 or 2.0), and the
            third is an array with shape (nspins, nkpoints, nbands) containing the
            band energies in Ha.
        """
        from itertools import groupby

        nspins = 1 if (self.parameters["spin"] is None) else 2
        if (not self.parameters["SOC"]) and (nspins == 1):
            dosweight = 2.0
        else:
            dosweight = 1.0

        with open(filename, "r") as file:
            content = [
                line.strip().split()
                for line in file.readlines()
                if "k-point" not in line
                and "#" not in line
                and "occupation number (dn), eigenvalue (dn)" not in line
            ]
            content = [
                list(group)
                for k, group in groupby(content, lambda x: x == [])
                if not k
            ]  # this splits by empty lines
            if (not self.parameters["SOC"]) and (nspins == 2):
                spinup = (
                    np.array(content, dtype=float)[:, :, 2] * eV
                )  # (nkpoints, bands)
                spindown = (
                    np.array(content, dtype=float)[:, :, 4] * eV
                )  # (nkpoints, bands)
            else:
                content = (
                    np.array(content, dtype=float)[:, :, 2] * eV
                )  # (nkpoints, bands)

        if (not self.parameters["SOC"]) and (nspins == 2):
            nbands = spinup.shape[1]
            nks = spinup.shape[0]
            nbands2 = spindown.shape[1]
            assert (
                nbands == nbands2
            ), "Number of spin-up and spin-down bands is not the same."
        else:
            nbands = int(content.shape[1])
            nks = content.shape[0]

        ebands = np.empty((nspins, nks, nbands))

        if (not self.parameters["SOC"]) and (nspins == 2):
            ebands[0, :, :] = spinup
            ebands[1, :, :] = spindown
        else:
            ebands[0, :, :] = content
        return (self.parameters["fermi_level"], dosweight, ebands)

    def read_AIMS_output(self, outputfile):
        """Read in a .out file in AIMS format.

        The file contains information about the Fermi level, the number of spins, and the number of electrons.

        Args:
            filename (str): path to the Final_KS_eigenvalues.dat file

        Returns:
            dict: Parameters dictionary.
        """
        parameters = {"SOC": False, "spin": None}
        with open(outputfile, "r") as file:
            for line in file.readlines():
                if "include_spin_orbit" in line:
                    parameters["SOC"] = True
                if ("spin" in line) & ("collinear" in line):
                    parameters["spin"] = "collinear"
                if "Chemical potential" in line:
                    if parameters["spin"] is not None:
                        if "spin up" in line:
                            up_fermi_level = float(line.split()[-2])
                        elif "spin dn" in line:
                            down_fermi_level = float(line.split()[-2])
                            parameters["fermi_level"] = max(
                                [up_fermi_level, down_fermi_level]
                            )
                if "Chemical potential (Fermi level)" in line:
                    fermi_level = line.replace("eV", "")
                    parameters["fermi_level"] = float(fermi_level.split()[-1])
                if "Chemical potential is" in line:
                    parameters["fermi_level"] = float(line.split()[-2])
                if "number of electrons (from input files)" in line:
                    parameters["nelectrons"] = float(line.split()[-1])
                if "Number of k-points" in line:
                    parameters["k_points"] = int(line.split()[-1])
        parameters["fermi_level"] = (
            parameters["fermi_level"] * eV
        )  # reverting to atomic units
        return parameters

    def _get_AIMS_output(self, dirname):
        """Automatically looks for the correct AIMS .out file in dirname.

        Lists all .out file in directory and looks for "Have a nice day" line.

        Args:
            dirname (str): path to directory.

        Returns:
            str : path to outputfile.
        """
        with BoltzTraP2.misc.dir_context(dirname):
            # Find all the .out files in the directory, check for Have a nice day.
            filenames = sorted(
                [i for i in glob.glob("*.out") if os.path.isfile(i)]
            )
            if not filenames:
                return None
            if len(filenames) == 1:
                return os.path.join(dirname, filenames[0])
            if len(filenames) > 1:
                logging.warning(
                    "There is more than one .out file in the directory "
                    "- looking for 'Have a nice day.' in outputfiles."
                )
                for outfile in filenames:
                    check = os.popen(
                        "tail -n 10 {}".format(os.path.join(dirname, outfile))
                    ).read()
                    if "Have a nice day" in check:
                        return os.path.join(dirname, outfile)
                else:
                    raise LoaderError("Could not find outputfile.")


def parse_aims(directory):
    """Wrapper function to parse the AIMSReader class.

    Args:
        directory (str): Path to directory.

    Returns:
        <class> : AIMSreader class.
    """
    logging.info("Parsing AIMS ...")
    return AIMSReader(directory)


def _unpack_ESPRESSO_element(ESPRESSO_element_name):
    """Break down a Quantum ESPRESSO element name into an element and a suffix.

    Quoting from the manual, the acceptable syntax is:
    > chemical symbol X (1 or 2 characters, case-insensitive)
    > or chemical symbol plus a number or a letter, as in
    > "Xn" (e.g. Fe1) or "X_*" or "X-*" (e.g. C1, C_h;
    > max total length cannot exceed 3 characters)

    Return a tuple with the name of the chemical element and the suffix.
    """
    if len(ESPRESSO_element_name) > 3:
        raise ValueError(
            f"Quantum ESPRESSO element name {ESPRESSO_element_name} is too long"
        )
    # Start by breaking the name at the dash or underscore.
    if "_" in ESPRESSO_element_name or "-" in ESPRESSO_element_name:
        # If there is a dash or an underscore in the name, the first part
        # must be an element name without any suffix.
        fields = ESPRESSO_element_name.replace("-", "_").split("_")
        if len(fields) != 2:
            f"invalid Quantum ESPRESSO element name {ESPRESSO_element_name}"
        element_name = _normalize_element_case(fields[0])
        suffix = fields[1]
        if element_name not in ase.data.chemical_symbols:
            raise ValueError(
                f"Quantum ESPRESSO element name {ESPRESSO_element_name} "
                f'references the invalid chemical element "{element_name}"'
            )
    else:
        # If there is no dash or underscore, find the largest match from the
        # beginning of the string.
        normalized_name = _normalize_element_case(ESPRESSO_element_name)
        for i in range(len(normalized_name), 0, -1):
            substring = normalized_name[:i]
            if substring in ase.data.chemical_symbols:
                break
        else:
            raise ValueError(
                "no valid chemical symbol could be found in the"
                f"Quantum ESPRESSO element name {ESPRESSO_element_name}"
            )
        element_name = substring
        suffix = ESPRESSO_element_name[len(substring) :]
    if len(suffix) > 1:
        raise ValueError(
            f"Quantum ESPRESSO element name {ESPRESSO_element_name} "
            "contains too long a suffix"
        )
    return (element_name, suffix)


def _parse_ESPRESSO_title(xml_tree):
    """Extract the calculation tile from an XML ElementTree representing a
    Quantum ESPRESSO pw.x run.
    """
    nruter = xml_tree.find("./input/control_variables/title").text
    if nruter is None:
        nruter = ""
    return nruter


def _parse_ESPRESSO_rlattvec(xml_tree):
    """Extract the internal representation of the reciprocal-space basis used
    to list k points from n XML ElementTree representing a Quantum ESPRESSO
    pw.x run and return it as a NumPy array with each vector as a column.
    """
    xml_rlattvec = xml_tree.find("./output/basis_set/reciprocal_lattice")
    nruter = np.empty((3, 3))
    for i, line in enumerate(xml_rlattvec):
        nruter[:, i] = np.array(_parse_xml_array(line.text))
    return nruter


def _parse_ESPRESSO_structure(xml_tree):
    """Extract the structural information from an XML ElementTree representing a
    Quantum ESPRESSO pw.x run, and return it as an ASE atoms object. The "tags"
    attribute of that object is used to encode different "extended elements"
    with the same chemical symbol, as allowed by Quantum ESPRESSO, e.g. to
    create specific magnetic configurations.
    """
    xml_structure = xml_tree.find("./input/atomic_structure")
    xml_cell = xml_structure.find("./cell")
    lattvec = np.empty((3, 3))
    # The XML output comes in Hartree units.
    for i, line in enumerate(xml_cell):
        lattvec[i, :] = np.array(_parse_xml_array(line.text)) / Angstrom
    xml_positions = xml_structure.find("./atomic_positions")
    extended_elements = []
    elements = []
    positions = []
    tags = []
    for line in xml_positions:
        element = _unpack_ESPRESSO_element(line.attrib["name"])
        elements.append(element[0])
        if element not in extended_elements:
            extended_elements.append(element)
        tags.append(extended_elements.index(element))
        positions.append((np.array(_parse_xml_array(line.text)) / Angstrom))
    return ase.Atoms(
        elements, positions=positions, cell=lattvec, pbc=[1, 1, 1], tags=tags
    )


def _parse_ESPRESSO_bands(xml_tree):
    """Real the band_structure block of an XML ElementTree representing a
    Quantum ESPRESSO pw.x run, and return all the relevant information.
    """
    # Parse the internal representation of rlattvec to transform the k points
    # to actual fractional coordinates.
    rlattvec = _parse_ESPRESSO_rlattvec(xml_tree)
    # Read in the basic information about the k-point grid and band structure.
    xml_bands = xml_tree.find("./output/band_structure")
    xml_fermi = xml_bands.find("./fermi_energy")
    if xml_fermi is None:
        # If no explicit Fermi energy is present in the XML file, use the
        # average between the HOMO and LUMO.
        xml_lul = float(xml_bands.find("./lowestUnoccupiedLevel").text)
        xml_hol = float(xml_bands.find("./highestOccupiedLevel").text)
        fermi = 0.5 * (xml_lul + xml_hol)
    else:
        fermi = float(xml_fermi.text)
    nruter = dict(
        nelect=int(round(float(xml_bands.find("./nelec").text))), fermi=fermi
    )
    non_collinear = xml_bands.find("./noncolin").text == "true"
    if non_collinear:
        raise ValueError(
            "non-collinear ESPRESSO calculations are not supported yet"
        )
    spin_polarized = xml_bands.find("./lsda").text == "true"
    if spin_polarized:
        nbands_up = int(xml_bands.find("./nbnd_up").text)
        nbands_down = int(xml_bands.find("./nbnd_dw").text)
        if nbands_up != nbands_down:
            raise ValueError(
                "ESPRESSO runs with different number"
                " of up and down bands are not supported"
            )
        n_spin = 2
    else:
        n_spin = 1
    xml_kpoints = xml_bands.find("./starting_k_points")
    xml_mp = xml_kpoints.find("./monkhorst_pack")
    if xml_mp is None:
        # Other ways of generating k points are not supported.
        raise ValueError(
            "this ESPRESSO calculation does not use a Monkhorst-Pack grid"
        )
    # Iterate over irreducible k points and read their direct coordinates and
    # the band energies.
    kpoints = []
    bands = []
    for xml_kpoint in xml_bands.iter("ks_energies"):
        kpoints.append(
            np.linalg.solve(
                rlattvec, _parse_xml_array(xml_kpoint.find("k_point").text)
            )
        )
        bands.append(_parse_xml_array(xml_kpoint.find("eigenvalues").text))
    nruter["kpoints"] = np.array(kpoints)
    nruter["kpoints"] -= np.round(nruter["kpoints"])
    nruter["E"] = np.array(bands)
    nruter["E"] = nruter["E"].reshape((nruter["E"].shape[0], n_spin, -1))
    if not spin_polarized:
        nruter["magmom"] = None
    # TODO: Read the velocities if available.
    # TODO: Add support for non-colinear calculations.
    return nruter


def parse_ESPRESSO_xml(filename):
    """Parse an ESPRESSO XML file to extract all structural information, the
    coordinates of the q points used in the calculation, the energy eigenvalues
    at those points, and their gradients (group velocities) if available.

    Return the results in a dictionary with the following keys:
    title: the title of the calculation, which can be empty
    atoms: an ASE atoms object representing the structure
    nelect: number of valence electrons in the simulation cell
    kpoints: numpy array with shape (nk, 3) containing direct coordinates of an
             irreducible set of k points.
    fermi: the Fermi level in eV
    magmom: the magnetic moments in a format suitable for the loaders in dft.py
    E: numpy array with shape (nspin, npoints, nbands) containing the energies
       of the electronic eigenstates at each point
    v: numpy array with shape (nspin, npoints, nbands, 3) containing the
       gradient of E with respect to k at each point. This key is only present
       if the vasprun.xml file contains group velocities.
    """
    # Parse the whole file using the XML parser in the standard library.
    xml_tree = et.parse(filename)
    nruter = dict(
        title=_parse_ESPRESSO_title(xml_tree),
        atoms=_parse_ESPRESSO_structure(xml_tree),
    )
    # Do not use the actual magnetic moment, but just the atomic tags that
    # encode the extended ESPRESSO chemical symbol syntax. The purpose is
    # just to break the symmetry as required even, for instance, in a system
    # with no magnetization but with several different pseudopotentials for
    # the same element.
    nruter["magmom"] = (
        np.array(nruter["atoms"].get_tags(), dtype=np.float64) + 1.0
    )
    # Note that this can override nruter["magmom"].
    nruter.update(_parse_ESPRESSO_bands(xml_tree))
    # Check if the set of k points is indeed irreducible according to our own
    # symmetry code.
    subset = calc_reciprocal_iksubset(
        nruter["atoms"], nruter["magmom"], nruter["kpoints"]
    )
    if len(subset) != nruter["kpoints"].shape[0]:
        raise ValueError("inconsistent sets of k points")
    return nruter


def save_trace(
    filename,
    data,
    Tr,
    mur,
    N,
    sdos,
    cv,
    cond,
    seebeck,
    kappa,
    hall,
    scattering_model="uniform_tau",
):
    """Create a .trace file in a format similar to the original output of
    BoltzTraP. The optional parameter "scattering_model" enables other
    posibilities.
    """
    nmu = mur.shape[-1]
    if mur.ndim == 1:
        mur = np.broadcast_to(mur, (len(Tr), nmu))
    nformulas = data.get_formula_count()
    headerfmt = "#{:>9s} {:>9s}" + " ".join(8 * ["{:>25s}"])
    header = [
        "Ef[Ry]",
        "T[K]",
        "N[e/uc]",
        "DOS(ef)[1/(Ha*uc)]",
        "S[V/K]",
        "sigma/tau0[1/(ohm*m*s)]",
        "RH[m**3/C]",
        "kappae/tau0[W/(m*K*s)]",
        "cv[J/(mol*K)]",
        "chi[m**3/mol]",
    ]
    conductivity_factor = 1.0
    if scattering_model == "uniform_lambda":
        header[5] = "sigma/lambda0[1/(ohm*m**2)]"
        header[7] = "kappae/lambda0[W/(m**2*K)]"
        conductivity_factor = 1.0 / (sp.constants.alpha * sp.constants.c)
    elif scattering_model == "custom_tau":
        header[5] = "sigma[1/(ohm*m)]"
        header[7] = "kappae[W/(m*K)]"
        conductivity_factor = 1e-9
    elif scattering_model != "uniform_tau":
        raise ValueError("unsupported scattering model")
    rowfmt = "{:>10g} {:>9g}" + " ".join(8 * ["{:>25g}"])
    with open(filename, "w") as f:
        print(headerfmt.format(*header).strip(), file=f)
        for imu in range(nmu):
            for iT, T in enumerate(Tr):
                mu = mur[iT, imu]
                # cv is expressed as a quantity per unit cell. To create the
                # output, we reexpress it as a quantity per mole of unit
                # formula.
                ocv = cv[iT, imu] * AVOGADRO / nformulas
                # The equivalent to the trace of the Hall tensor is an average
                # over the even permutations of [0, 1, 2].
                ohall = (
                    hall[iT, imu, 0, 1, 2]
                    + hall[iT, imu, 2, 0, 1]
                    + hall[iT, imu, 1, 2, 0]
                ) / 3.0
                # Our estimate of the susceptibility comes directly from the
                # smoothed DOS.
                magsus = (
                    sdos[iT, imu]
                    * MU0
                    * MUB**2
                    * AVOGADRO
                    / (data.atoms.get_volume() * Meter**3)
                )
                print(
                    rowfmt.format(
                        mu * 2.0,
                        T,
                        N[iT, imu] + data.nelect,
                        sdos[iT, imu],
                        seebeck[iT, imu].trace() / 3.0,
                        cond[iT, imu].trace() * conductivity_factor / 3.0,
                        ohall,
                        kappa[iT, imu].trace() * conductivity_factor / 3.0,
                        ocv,
                        magsus,
                    ),
                    file=f,
                )


def save_condtens(
    filename,
    data,
    Tr,
    mur,
    N,
    cond,
    seebeck,
    kappa,
    scattering_model="uniform_tau",
):
    """Create a .condtens file in a format similar to the original output of
    BoltzTraP. The optional parameter "scattering_model" enables other
    posibilities.
    """
    nmu = mur.shape[-1]
    if mur.ndim == 1:
        mur = np.broadcast_to(mur, (len(Tr), nmu))
    headerfmt = "#{:>9s} {:>9s}" + " ".join(28 * ["{:>25s}"])
    header = [""] * 30
    header[0] = "Ef[Ry]"
    header[1] = "T[K]"
    header[2] = "N[e/uc]"
    header[3] = "sigma/tau0[1/(ohm*m*s)]"
    header[12] = "S[V/K]"
    header[21] = "kappae/tau0[W/(m*K*s)]"
    conductivity_factor = 1.0
    if scattering_model == "uniform_lambda":
        header[3] = "sigma/lambda0[1/(ohm*m**2)]"
        header[21] = "kappae/lambda0[W/(m*2*K)]"
        conductivity_factor = 1.0 / (sp.constants.alpha * sp.constants.c)
    elif scattering_model == "custom_tau":
        header[3] = "sigma[1/(ohm*m)]"
        header[21] = "kappae[W/(m*K)]"
        conductivity_factor = 1e-9
    elif scattering_model != "uniform_tau":
        raise ValueError("unsupported scattering model")
    rowfmt = "{:>10g} {:>9g}" + " ".join(28 * ["{:>25g}"])
    with open(filename, "w") as f:
        print(headerfmt.format(*header).strip(), file=f)
        for imu in range(nmu):
            for iT, T in enumerate(Tr):
                mu = mur[iT, imu]
                print(
                    rowfmt.format(
                        mu * 2.0,
                        T,
                        N[iT, imu] + data.nelect,
                        *cond[iT, imu, :, :].ravel(order="F")
                        * conductivity_factor,
                        *seebeck[iT, imu, :, :].ravel(order="F"),
                        *kappa[iT, imu, :, :].ravel(order="F")
                        * conductivity_factor,
                    ),
                    file=f,
                )


def save_halltens(filename, data, Tr, mur, N, hall):
    """Create a .halltensfile in a format similar to the original output of
    BoltzTraP.
    """
    nmu = mur.shape[-1]
    if mur.ndim == 1:
        mur = np.broadcast_to(mur, (len(Tr), nmu))
    headerfmt = "#{:>9s} {:>9s}" + " ".join(28 * ["{:>25s}"])
    header = [""] * 30
    header[0] = "Ef[Ry]"
    header[1] = "T[K]"
    header[2] = "N[e/uc]"
    header[3] = "RH[m**3/C]"
    rowfmt = "{:>10g} {:>9g}" + " ".join(28 * ["{:>25g}"])
    with open(filename, "w") as f:
        print(headerfmt.format(*header).strip(), file=f)
        for imu in range(nmu):
            for iT, T in enumerate(Tr):
                mu = mur[iT, imu]
                print(
                    rowfmt.format(
                        mu * 2.0,
                        T,
                        N[iT, imu] + data.nelect,
                        *hall[iT, imu, :, :, :].ravel(order="F"),
                    ),
                    file=f,
                )
