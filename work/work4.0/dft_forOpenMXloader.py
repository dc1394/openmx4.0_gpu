# -*- coding: utf-8 -*
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

import collections
import functools
import glob
import io
import logging
import math
import os
import os.path
import sys
import xml.etree.ElementTree as et

import ase.io
import ase.io.wien2k
import numpy as np

import BoltzTraP2.io
import BoltzTraP2.misc
import BoltzTraP2.sphere
from BoltzTraP2.units import *


class LoaderError(Exception):
    """Generic exception class for problems in loaders."""


# List of format names and loaders that will be tried for each directory
# A loader can be any callable that takes a mandatory argument (usually a
# directory) and returns an object with the following mandatory attributes:
#
# sysname: an arbitrary string
# atoms: an ASE atoms object representing the system
# dosweight: maximum occupancy of each state (normally 1 for spin-polarized
#            calculations and 2 otherwise)
# kpoints: irreducible set of k points in reduced coordinates, as a
#          (nkpoints, 3) array
# fermi: Fermi level from the original calculation, in Ha
# ebands: energy bands in Ha, as a (nbands, nkpoints) array
#
# And some or all of the following optinal attributes:
#
# self.magmom: initial magnetic moments, which can lower the symmetry of the
#              system. This attribute can be None (non-spin-polarized
#              calculations), a 1D array (for collinear spins) or a 2D array of
#              size (natoms, 3) for noncollinear spin configurations.
# self.nelect: number of valence electrons in the calculation
# self.mommat: momentum matrix elements in in Ha * bohr, as a
#              (nkpoints, nbands, 3) array
#
# The last loader to be added has the highest priority.
# The loader must raise a LoaderException if the argument cannot be processed.
# Note that the argument can be of any type. If the loader expects a string,
# it must explicitly check that fact.
# Implementing a new loader and adding it to this list is the
# way to add support for a new format in BoltzTraP2.
# To add a loader to the tuple, use BolzTraP2.dft.register_loader(name, loader)
# Example: register_loader("VASP", VASPLoader)
loaders = []


def register_loader(name, loader):
    """Add an element to the list of registered loaders."""
    loaders.append((name, loader))


class VASPLoader:
    """Loader for VASP calculations."""

    def __init__(self, directory):
        if not isinstance(directory, str):
            raise LoaderError("this loader only works with directories")
        with BoltzTraP2.misc.dir_context(directory):
            vasprunxml = "vasprun.xml"
            if not os.path.isfile(vasprunxml):
                raise LoaderError("vasprun.xml not found")
            r = BoltzTraP2.io.parse_vasprunxml(vasprunxml)
            self.sysname = r["name"]
            self.atoms = r["atoms"]
            BoltzTraP2.misc.info("lattice:")
            BoltzTraP2.misc.info(self.atoms.get_cell().T)
            self.kpoints = r["kpoints"]
            # Some unit conversions are necessary at this point.
            self.fermi = r["fermi"] * eV
            # If the calculation is spin-polarized, concatenate all the bands
            self.ebands = r["E"].transpose((0, 2, 1)) * eV
            if self.ebands.shape[0] > 1 or (
                r["magmom"] is not None and r["magmom"].shape[1] == 3
            ):
                self.dosweight = 1.0
            else:
                self.dosweight = 2.0
            self.ebands = self.ebands.reshape((-1, self.ebands.shape[-1]))
            self.nelect = r["nelect"]
            if "v" in r:
                self.mommat = r["v"].transpose((0, 2, 1, 3)) * eV * Angstrom
                self.mommat = self.mommat.reshape(
                    tuple([-1] + list(self.mommat.shape[2:]))
                )
                self.mommat = self.mommat.transpose((1, 0, 2))
            self.magmom = r["magmom"]


register_loader("VASP", VASPLoader)


class GenericWien2kLoader:
    """Generic Loader for Wien2k calculations, not intended for direct use.

    This class gives total control over the file names to be loaded.
    """

    def __init__(
        self, w2kname, dosweight, scffn, structfn, energyfn, derfn=""
    ):
        BoltzTraP2.misc.info("Wien2k system name:", w2kname)
        self.sysname = w2kname
        if not os.path.isfile(scffn):
            raise LoaderError(".scf file not found")
        self.dosweight = dosweight
        if not os.path.isfile(energyfn):
            raise LoaderError("energy(so) file not found")
        BoltzTraP2.misc.info("matching energy and scf files were found")
        self.atoms = ase.io.read(structfn)
        dum = ase.io.wien2k.read_struct(structfn, ase=False)
        latt = dum[1]
        if latt == "R":
            latt = "P"
        conv = ase.io.wien2k.c2p(latt)
        BoltzTraP2.misc.info("lattice:", self.atoms.get_cell().T)
        BoltzTraP2.misc.info("conv:", conv)
        BoltzTraP2.misc.info("conv:", latt)
        self.kpoints, self.ebands = BoltzTraP2.io.W2Kene(energyfn, conv)
        self.fermi = BoltzTraP2.io.W2Kfermi(scffn)
        if os.path.isfile(derfn):
            BoltzTraP2.misc.info("a matching .mommat2 file was found")
            self.mommat, nemin, nemax = BoltzTraP2.io.W2Kmommat(
                derfn, self.kpoints
            )
            self.ebands = self.ebands[nemin - 1 : nemax]


class Wien2kLoader(GenericWien2kLoader):
    """Loader for Wien2k calculations following the usual naming convention."""

    def __init__(self, directory):
        """Guess the name of the system and call the base cosntructor to do
        the real work.
        """
        if not isinstance(directory, str):
            raise LoaderError("this loader only works with directories")
        w2kname = _get_W2Ksystemname(directory)
        if w2kname is None:
            raise LoaderError("cannot determine a Wien2k system name")
        scffn = w2kname + ".scf"
        structfn = w2kname + ".struct"
        energyfn = w2kname + ".energyso"
        with BoltzTraP2.misc.dir_context(directory):
            if not os.path.isfile(energyfn):
                energyfn = os.path.splitext(energyfn)[0] + ".energy"
        if os.path.splitext(energyfn)[1] == ".energy":
            dosweight = 2.0
        else:
            dosweight = 1.0
        derfn = w2kname + ".mommat2"
        GenericWien2kLoader.__init__(
            self,
            w2kname,
            dosweight,
            *(
                os.path.join(directory, i)
                for i in (scffn, structfn, energyfn, derfn)
            )
        )


register_loader("Wien2k", Wien2kLoader)


def _get_W2Ksystemname(dirname):
    """Try to guess the Wien2k system name corresponding to a directory."""
    with BoltzTraP2.misc.dir_context(dirname):
        filenames = sorted(
            [i for i in glob.glob("*.struct") if os.path.isfile(i)]
        )
        if not filenames:
            return None
        if len(filenames) > 1:
            logging.warning(
                "there is more than one .struct file in the directory "
                "- using the first one"
            )
    return os.path.splitext(os.path.basename(filenames[0]))[0]


class GENELoader:
    """Loader for data in BoltzTraP's generic format."""

    def __init__(self, directory):
        if not isinstance(directory, str):
            raise LoaderError("this loader only works with directories")
        genename = _get_GENEsystemname(directory)
        if genename is None:
            raise LoaderError("cannot determine a GENE system name")
        structfn = genename + ".structure"
        energyfn = genename + ".energy"

        with BoltzTraP2.misc.dir_context(directory):
            BoltzTraP2.misc.info("GENE system name:", genename)
            if not os.path.isfile(energyfn):
                raise ValueError("energy file not found")
            self.atoms = BoltzTraP2.io.read_GENE_struct(structfn)
            BoltzTraP2.misc.info("lattice:", self.atoms.get_cell().T)
            (
                self.fermi,
                self.dosweight,
                self.kpoints,
                self.ebands,
                mommat,
            ) = BoltzTraP2.io.read_GENE_eneandmat(energyfn)
            if mommat is not None:
                self.mommat = mommat
        self.sysname = genename


register_loader("GENE", GENELoader)


def _get_GENEsystemname(dirname):
    """Try to guess the GENE system name corresponding to a directory."""
    with BoltzTraP2.misc.dir_context(dirname):
        filenames = sorted(
            [i for i in glob.glob("*.structure") if os.path.isfile(i)]
        )
        if not filenames:
            return None
        if len(filenames) > 1:
            logging.warning(
                "there is more than one .structure file in the directory"
                " - using the first one"
            )
    return os.path.splitext(os.path.basename(filenames[0]))[0]

class OpenMXLoader:
    """Loader for data in BoltzTraP's OpenMX refered generic format."""

    def __init__(self, directory):
        if not isinstance(directory, str):
            raise LoaderError("this loader only works with directories")
        openmxname = _get_OpenMXsystemname(directory)
        if openmxname is None:
            raise LoaderError("cannot determine a OpenMX system name")
        structfn = openmxname + ".MXstructure"
        energyfn = openmxname + ".MXenergy"

        with BoltzTraP2.misc.dir_context(directory):
            BoltzTraP2.misc.info("OpenMX system name:", openmxname)
            if not os.path.isfile(energyfn):
                raise ValueError("energy file not found")
            self.atoms = BoltzTraP2.io.read_OpenMX_struct(structfn)
            BoltzTraP2.misc.info("lattice:", self.atoms.get_cell().T)
            (
                self.fermi,
                self.dosweight,
                self.kpoints,
                self.ebands,
                mommat,
            ) = BoltzTraP2.io.read_OpenMX_eneandmat(energyfn)
            if mommat is not None:
                self.mommat = mommat
        self.sysname = openmxname


register_loader("OpenMX", OpenMXLoader)

def _get_OpenMXsystemname(dirname):
    """Try to guess the OpenMX system name corresponding to a directory."""
    with BoltzTraP2.misc.dir_context(dirname):
        filenames = sorted(
            [i for i in glob.glob("*.MXstructure") if os.path.isfile(i)]
        )
        if not filenames:
            return None
        if len(filenames) > 1:
            logging.warning(
                "there is more than one .MXstructure file in the directory"
                " - using the first one"
            )
    return os.path.splitext(os.path.basename(filenames[0]))[0]


class CASTEPLoader:
    """Loader for CASTEP calculations."""

    def __init__(self, directory):
        if not isinstance(directory, str):
            raise LoaderError("this loader only works with directories")
        castepname = _get_CASTEPsystemname(directory)
        if castepname is None:
            raise LoaderError("cannot determine a CASTEP system name")
        structfn = castepname + ".castep"
        energyfn = castepname + ".bands"
        with BoltzTraP2.misc.dir_context(directory):
            BoltzTraP2.misc.info("CASTEP system name:", castepname)
            if not os.path.isfile(energyfn):
                raise ValueError("energy file not found")
            (
                self.atoms,
                magmom,
                fict_magmom,
            ) = BoltzTraP2.io.read_CASTEP_output(structfn)
            BoltzTraP2.misc.info("lattice:", self.atoms.get_cell().T)
            (
                self.fermi,
                self.castep_fermi_mismatch,
                self.dosweight,
                self.kpoints,
                self.ebands,
                self.nelect,
                mommat,
            ) = BoltzTraP2.io.read_CASTEP_bands(energyfn)

            # Non-collinear spin-polarised runs don't use symmetry in CASTEP.
            # Hubbard U calculations with symmetry are not compatible with
            # BoltzTraP2 at the moment.
            # In both cases, use fictitious magmom instead to break the symmetry.
            if fict_magmom is not None:
                self.magmom = fict_magmom
            else:
                self.magmom = magmom
            # mommat is not implemented at the moment
            if mommat is not None:
                self.mommat = mommat
        self.sysname = castepname


register_loader("CASTEP", CASTEPLoader)


def _get_CASTEPsystemname(dirname):
    """Try to guess the CASTEP system name corresponding to a directory."""
    with BoltzTraP2.misc.dir_context(dirname):
        filenames = sorted(
            [i for i in glob.glob("*.castep") if os.path.isfile(i)]
        )
        if not filenames:
            return None
        if len(filenames) > 1:
            logging.warning(
                "there is more than one .castep file in the directory"
                " - using the first one"
            )
    return os.path.splitext(os.path.basename(filenames[0]))[0]


class ABINITLoader:
    """Loader for ABINIT calculations."""

    def __init__(self, directory):
        if not isinstance(directory, str):
            raise LoaderError("this loader only works with directories")
        abinitname = _get_ABINITsystemname(directory)
        if abinitname is None:
            raise LoaderError("no valid abinit run files found")

        # first chop off GSR.nc
        radixDS = abinitname.split("_GSR")[0]
        # then chop off eventual dataset reference, to get original radix for output files
        radix = radixDS.split("_DS")[0]

        with BoltzTraP2.misc.dir_context(directory):
            BoltzTraP2.misc.info("ABINIT GSR file name:", abinitname)
            r = BoltzTraP2.io.parse_abinitrun(abinitname)
            self.sysname = r["name"]
            self.atoms = r["atoms"]
            BoltzTraP2.misc.info("lattice:", self.atoms.get_cell().T)
            self.kpoints = r["kpoints"]
            # Some unit conversions are necessary at this point.
            self.fermi = r["fermi"] * Ha
            # If the calculation is spin-polarized, concatenate all the bands
            self.ebands = r["E"].transpose((0, 2, 1)) * Ha
            if self.ebands.shape[0] > 1:
                self.dosweight = 1.0
            else:
                self.dosweight = 2.0
            self.ebands = self.ebands.reshape(-1, self.ebands.shape[-1])
            self.nelect = r["nelect"]
            # TODO: for this option need to read other 1WFx files with velocity
            # matrix elements
            if "v" in r:
                BoltzTraP2.misc.info(
                    "ABINIT velocity reading is not implemented yet. Should not be here"
                )
                self.mommat = r["v"].transpose((0, 2, 1, 3)) * Ha / Bohr
                self.mommat = self.mommat.reshape(
                    tuple([-1] + list(self.mommat.shape[2:]))
                )
                self.mommat = self.mommat.transpose((1, 0, 2))


register_loader("ABINIT", ABINITLoader)


def _get_ABINITsystemname(dirname):
    """Try to guess the GENE system name corresponding to a directory."""
    with BoltzTraP2.misc.dir_context(dirname):
        # should run with a GS results netcdf file
        filenames = sorted(
            [i for i in glob.glob("*_GSR.nc") if os.path.isfile(i)]
        )
        if not filenames:
            return None
        if len(filenames) > 1:
            logging.warning(
                "there is more than one _GSR.nc file in the directory"
                " - using the first one"
            )
    # use the first one we find, TODO eventually add options to choose the GSR
    # file from a specific dataset.
    return os.path.basename(filenames[0])


class AIMSLoader:
    """Simple loader for AIMS calculations."""

    def __init__(self, directory):
        # Check that the argument is a directory
        if not isinstance(directory, str):
            raise LoaderError("this loader only works with directories")
        # Check if there may be a AIMS calculation in that directory
        structfn = "geometry.in"
        kpointfn = "Final_KS_eigenvalues.dat"
        energyfn = "Final_KS_eigenvalues.dat"
        # "cd" to the directory and check that all required files are present
        with BoltzTraP2.misc.dir_context(directory):
            BoltzTraP2.misc.info("AIMS system name:", directory)
            if not os.path.isfile(structfn):
                raise LoaderError("geometry.in file not found")
            if not os.path.isfile(kpointfn):
                raise LoaderError("Final_KS_eigenvalues.dat not found")
            # Initialize the AIMSReader,
            aimsreader = BoltzTraP2.io.parse_aims(directory)
            self.atoms = aimsreader.atoms
            # the raw k-point list
            self.kpoints = aimsreader.kpoints
            # and the band energies plus related parameters.
            self.fermi = aimsreader.fermi_level
            self.dosweight = aimsreader.dosweight
            ebands = aimsreader.eigenvalues
            # additional information
            self.nelect = aimsreader.parameters["nelectrons"]
            # Flatten the ebands array along the spin axes to comply
            # with the specification.
            self.ebands = ebands.transpose((0, 2, 1))
            self.ebands = self.ebands.reshape((-1, self.ebands.shape[-1]))
        # Finally, set the system name
        self.sysname = directory.split("/")[-1]


register_loader("AIMS", AIMSLoader)

_ESPRESSO_SCHEMA_KEY = (
    "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation"
)
_ESPRESSO_SCHEMA_PREFIX = "http://www.quantum-espresso.org/ns/qes/qes-1.0"


def _get_ESPRESSO_filename(dirname):
    """Try to guess the name of the ESPRESSO XML output file in a directory."""
    with BoltzTraP2.misc.dir_context(dirname):
        # Find all the xml files in the directory
        xml_filenames = sorted(
            [i for i in glob.glob("*.xml") if os.path.isfile(i)]
        )
        # Check if any of them is a QE output file.
        qe_xml_filenames = []
        for xml_fn in xml_filenames:
            try:
                tree = et.parse(xml_fn)
            except et.ParseError:
                pass
            attributes = tree.getroot().attrib
            try:
                if attributes[_ESPRESSO_SCHEMA_KEY].startswith(
                    _ESPRESSO_SCHEMA_PREFIX
                ):
                    qe_xml_filenames.append(xml_fn)
            except KeyError:
                pass
        if not qe_xml_filenames:
            return None
        if len(qe_xml_filenames) > 1:
            logging.warning(
                "there is more than one ESPRESSO XML output file "
                "in the directory - using the first one"
            )
        return os.path.basename(qe_xml_filenames[0])


class ESPRESSOLoader:
    """Loader for Quantum ESPRESSO calculations."""

    def __init__(self, directory):
        if not isinstance(directory, str):
            raise LoaderError("this loader only works with directories")
        filename = _get_ESPRESSO_filename(directory)
        if filename is None:
            raise LoaderError("ESPRESSO XML output file not found")
        with BoltzTraP2.misc.dir_context(directory):
            r = BoltzTraP2.io.parse_ESPRESSO_xml(filename)
            self.sysname = r["title"]
            self.atoms = r["atoms"]
            BoltzTraP2.misc.info("lattice:")
            BoltzTraP2.misc.info(self.atoms.get_cell().T)
            self.kpoints = r["kpoints"]
            self.fermi = r["fermi"]
            self.ebands = r["E"]
            if self.ebands.shape[1] > 1 or (
                r["magmom"] is not None and r["magmom"].ndim > 1
            ):
                self.dosweight = 1.0
            else:
                self.dosweight = 2.0
            self.ebands = self.ebands.reshape(
                (self.ebands.shape[0], -1)
            ).transpose()
            self.nelect = r["nelect"]
            # TODO: Parse starting_magnetization to build the correct magmon.
            self.magmom = r["magmom"]


register_loader("Quantum ESPRESSO", ESPRESSOLoader)


class GPAWLoader:
    """Loader for collinear and unpolarized GPAW calculations.
    Non-collinear calculations from gpaw.spinorbit are not explicitly included, but can be very easily forced into correct format to fit this leader."""

    def __init__(self, directory):
        # Check that arg is a directory
        if not isinstance(directory, str):
            raise LoaderError("This loader works with directories")

        # Check if GPAW calc is in directory
        gpaw_name = _get_GPAW_filename(directory)

        if gpaw_name is None:
            # Raise error if loader finds no GPAW calc
            raise LoaderError("Cannot find GPAW system in this directory")

        gpaw_file = os.path.join(directory, gpaw_name)

        (
            self.atoms,
            self.fermi,
            self.dosweight,
            self.kpoints,
            self.ebands,
            self.nelect,
        ) = BoltzTraP2.io.read_GPAW(gpaw_file)
        self.sysname = gpaw_name[0:-4]


register_loader("GPAW", GPAWLoader)


def _get_GPAW_filename(dirname):
    with BoltzTraP2.misc.dir_context(dirname):
        filenames = sorted(
            [i for i in glob.glob("*.gpw") if os.path.isfile(i)]
        )
        if not filenames:
            return None
        if len(filenames) == 1:
            return filenames[0]
        if len(filenames) > 1:
            logging.warning(
                "There is more than one gpw file " "- Using the first one"
            )
            return filenames[0]


class DFTData:
    """Objects of this class hold structural and dynamical information from DFT
    results in any supported format.
    """

    def __init__(self, directory, derivatives=False, *args, **kwargs):
        """Create a DFTData object."""
        for label, loader in loaders[::-1]:
            BoltzTraP2.misc.info("looking for a {} calculation".format(label))
            try:
                loaded = loader(directory, *args, **kwargs)
            except LoaderError as e:
                BoltzTraP2.misc.info("error in {} loader: {}".format(label, e))
                continue
            self.source = label
            break
        else:
            raise ValueError(
                "no calculation found in directory {}".format(directory)
            )
        BoltzTraP2.misc.info(
            "successfully loaded a {} calculation".format(self.source)
        )
        # Try to copy all relevant attributes from the loader
        if derivatives:
            try:
                self.mommat = loaded.mommat
            except AttributeError:
                raise ValueError(
                    "no derivative information found in directory {}".format(
                        directory
                    )
                )
        else:
            try:
                loaded.mommat
            except AttributeError:
                pass
            else:
                BoltzTraP2.misc.info(
                    "derivative information will be discarded"
                )
            self.mommat = None
        try:
            self.sysname = loaded.sysname
            self.atoms = loaded.atoms
            self.dosweight = loaded.dosweight
            self.kpoints = loaded.kpoints
            self.fermi = loaded.fermi
            self.ebands = loaded.ebands
        except AttributeError:
            raise ValueError(
                "some essential piece of information was not loaded"
            )

        # Warn the user if the spin up and spin down Fermi energies
        # are different in CASTEP.
        try:
            self.castep_fermi_mismatch = loaded.castep_fermi_mismatch
            if self.castep_fermi_mismatch:
                BoltzTraP2.misc.info(
                    "CASTEP WARNING: "
                    "Different spin up and spin down Fermi energy."
                    "\nProceeding with spin up Fermi energy. Transport results might"
                    " be inaccurate."
                )
        except AttributeError:
            pass

        BoltzTraP2.misc.info("Fermi energy:", self.fermi)
        # If no initial magnetic moments are provided by the loader, assume a
        # non-spin-polarized calculation.
        try:
            self.magmom = loaded.magmom
        except AttributeError:
            self.magmom = None
            BoltzTraP2.misc.info("Assuming a non-spin-polarized calculation")
        # If the number of valence electrons has not been set yet, compute it
        # from the bands.
        try:
            self.nelect = loaded.nelect
        except AttributeError:
            degeneracies = BoltzTraP2.sphere.calc_reciprocal_degeneracies(
                self.atoms, self.magmom, self.kpoints
            )
            weights = degeneracies.astype(np.float64) / degeneracies.sum()
            occupancy = (loaded.ebands < loaded.fermi).astype(np.intc)
            self.nelect = round(self.dosweight * (occupancy * weights).sum())

    def bandana(self, emin=-np.inf, emax=np.inf):
        bandmin = np.min(self.ebands, axis=1)
        bandmax = np.max(self.ebands, axis=1)
        ntoolow = np.count_nonzero(bandmax <= emin)
        accepted = np.logical_and(bandmin < emax, bandmax > emin)
        BoltzTraP2.misc.info("BANDANA output")
        for iband in range(len(self.ebands)):
            BoltzTraP2.misc.info(
                iband, bandmin[iband], bandmax[iband], accepted[iband]
            )
        self.ebands = self.ebands[accepted]
        if self.mommat is not None:
            self.mommat = self.mommat[:, accepted, :]
        # Removing bands may change the number of valence electrons
        self.nelect -= self.dosweight * ntoolow
        return accepted

    def get_lattvec(self):
        try:
            self.lattvec
        except AttributeError:
            self.lattvec = self.atoms.get_cell().T * Angstrom
        return self.lattvec

    def get_volume(self):
        try:
            self.UCVol
        except AttributeError:
            lattvec = self.get_lattvec()
            self.UCvol = np.abs(np.linalg.det(lattvec))
        return self.UCvol

    def get_formula_count(self):
        """Return the number of irreducible formulas in the unit cell.

        Useful for computing molar quantities.
        """
        counts = collections.Counter(self.atoms.get_chemical_symbols())
        return functools.reduce(math.gcd, counts.values())
