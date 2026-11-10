#!/usr/bin/env bash
set -euo pipefail
source /home/ec2-user/openmx-build/scripts/environment.sh
root=/home/ec2-user/openmx-build
prefix=/home/ec2-user/.local/opt/openmx-4.0.1-sve256
source_dir=/home/ec2-user/openmx4.0
make -C "$source_dir/source" -j12 all > "$root/logs/openmx-build.log" 2>&1
mkdir -p "$prefix/bin" "$prefix/share/openmx" "$prefix/share/doc/openmx" /home/ec2-user/.local/bin
install -m755 "$source_dir/source/openmx" "$prefix/bin/openmx"
cp -a "$source_dir/DFT_DATA19" "$prefix/share/openmx/"
for tool in DosMain cohp jx analysis_example esp polB calB Z2FH bandgnu13 bin2txt cube2xsf intensity_map md2axsf tp kSpin BandDispersion ADenBand FermiLoop GridCalc MulPOnly MulPCalc gcube2oned cif2omx diff_gcube add_gcube diff_geo; do
  install -m755 "$source_dir/source/$tool" "$prefix/bin/$tool"
done
# The official make all installs utilities in work; replace its executable
# with a link to the versioned installation so both entry points agree.
ln -sfn "$prefix/bin/openmx" "$source_dir/work/openmx"
ln -sfn "$prefix/bin/openmx" /home/ec2-user/.local/bin/openmx
ln -sfn "$prefix/bin/openmx" /home/ec2-user/.local/bin/openmx-4.0.1
ldd "$prefix/bin/openmx" > "$root/logs/installed-ldd.txt"
readelf -d "$prefix/bin/openmx" > "$root/logs/installed-dynamic.txt"
sha256sum "$prefix/bin/"* > "$root/logs/installed-binaries-SHA256SUMS"
printf 'Installed at %s\n' "$prefix"
