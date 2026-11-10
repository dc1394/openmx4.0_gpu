Release note: May 08, 2026

***** How to apply the patch4.0.1: *****

cp ./patch4.0.1.tar.gz openmx4.0/source
cd openmx4.0/source
tar zxvf patch4.0.1.tar.gz
mv GaAs.dat ../work/
make clean
make install

***** patch4.0.1.tar.gz *****
contains 

Band_DFT_Dosout.c
Mulliken_Charge.c
GaAs.dat

***** purpose of patch4.0.1.tar.gz *****

Related to Band_DFT_Dosout.c:
The units in bxsf and frmsf files are Hartree and bohr in the previous version.
According to the convension in XCrySDen, the units are changed to eV and Angstrom,
respectively, in the patch.  

Related to Mulliken_Charge.c:
The index of HOMO is overwritten in Mulliken_Charge.c in an incorrect manner.
The effect appears for non-collinear calculations with the gamma-point. 
The problem is resolved by applying the patch. 

Related to GaAs.dat:
The calculation explained in the Sec. 38.1 of the manual cannot be reproduced by
GaAs.dat stored in work. The problem is resolved by applying the patch.
