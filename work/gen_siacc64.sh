#!/bin/bash
# P6 accuracy-dump pair (plan sec. 13.2-C): Si 64-atom Gamma cluster
# (2x2x2 of sidia.dat's conventional cell, n = 64*13 = 832 > cluster
# switch-num 400 -> GPU dense path).  Decks/jobs cloned from the
# sip_ccol360_{o,g}1 production skeletons; OPENMX_ACC_DUMP=1 dumps the
# orthonormal H~ / Y~ / ko at SCF iteration 3 on the dense owner.
#   ./gen_siacc64.sh          -> siacc64_o1, siacc64_g1
set -u
cd "$(dirname "$0")" || exit 1
fail() { echo "FAIL: $*" >&2; exit 1; }

TPL=../sidia.dat
[ -f "$TPL" ] || fail "sidia.dat not found"

mkcase() { # $1=tag $2=srcdir
  local tag=$1 src=$2
  local dst="siacc64_${tag}"
  [ -d "$src" ] || fail "source $src missing"
  rm -rf "$dst"; mkdir "$dst"

  sed "s/${src}/${dst}/g" "$src/$src.sh" > "$dst/$dst.sh"
  cp -p "$src/mps_node.sh" "$dst/" 2>/dev/null

  # deck: clone the solver/basis/eigen sections, splice the si64 geometry
  awk -v tpl="$TPL" '
    BEGIN {
      # read the 8-atom fractional basis and lattice constant from sidia.dat
      na = 0; a = 0; want_a = 0
      while ((getline line < tpl) > 0) {
        n = split(line, f, " ")
        # the template also carries an EMPTY <Atoms.UnitVectors stub in its
        # keyword list; keep looking until an actual numeric row appears
        if (f[1] == "<Atoms.UnitVectors") { want_a = (a == 0) }
        else if (want_a && n >= 3 && f[1] + 0 > 0) { a = f[1] + 0; want_a = 0 }
        if (n >= 5 && f[2] == "Si" && f[1] + 0 >= 1 && f[1] + 0 <= 8 && index(line, "0.") > 0 && na < 8) {
          na++; fx[na] = f[3] + 0; fy[na] = f[4] + 0; fz[na] = f[5] + 0
        }
      }
      close(tpl)
      if (na != 8 || a <= 0) { print "GEOM_PARSE_FAIL" > "/dev/stderr"; exit 1 }
    }
    /^Atoms.Number/         { print "Atoms.Number                       64"; next }
    /^<Atoms.UnitVectors/   { print; print "    " 2*a "     0.0     0.0";
                              print "     0.0    " 2*a "     0.0";
                              print "     0.0     0.0    " 2*a; skip=1; next }
    /^Atoms.UnitVectors>/   { skip=0; print; next }
    /^<Atoms.SpeciesAndCoordinates/ {
      print; k=0
      for (i=0;i<2;i++) for (j=0;j<2;j++) for (l=0;l<2;l++) for (m=1;m<=8;m++) {
        k++
        printf "  %3d  Si  %11.7f  %11.7f  %11.7f  2.0 2.0\n", k, (fx[m]+i)/2, (fy[m]+j)/2, (fz[m]+l)/2
      }
      skip=1; next
    }
    /^Atoms.SpeciesAndCoordinates>/ { skip=0; print; next }
    skip==1 { next }
    { print }
  ' "$src/$src.dat" | sed -e "s/${src}/${dst}/g" \
        -e "s/^scf\.maxIter\([[:space:]]\{1,\}\)[0-9]\{1,\}/scf.maxIter\110/" \
        > "$dst/$dst.dat"

  sed -i -e "s/^#PBS -N .*/#PBS -N a64${tag}/" \
         -e "s/^#PBS -l elapstim_req=.*/#PBS -l elapstim_req=00:20:00/" \
         -e "s/^ulimit -c 0/ulimit -c 0\nexport OPENMX_ACC_DUMP=1\nexport OPENMX_ACC_DUMP_SCF=3/" \
         -e "s/^MPS_X=\"\(.*\)\"/MPS_X=\"\1 -x OPENMX_ACC_DUMP -x OPENMX_ACC_DUMP_SCF\"/" \
         "$dst/$dst.sh"

  # ---- assertions ----
  [ "$(grep -cE '^ +[0-9]+ +Si +0\.' "$dst/$dst.dat")" -eq 64 ] || fail "$dst: atom count != 64"
  grep -q "Atoms.Number                       64" "$dst/$dst.dat" || fail "$dst: Atoms.Number"
  grep -q "10.862" "$dst/$dst.dat" || fail "$dst: cell not doubled"
  grep -qE "^scf\.maxIter[[:space:]]+10" "$dst/$dst.dat" || fail "$dst: maxIter"
  grep -q "scf.EigenvalueSolver       cluster" "$dst/$dst.dat" || fail "$dst: not cluster solver"
  grep -q "^export OPENMX_ACC_DUMP=1$" "$dst/$dst.sh" || fail "$dst: dump env"
  grep -q -- "-x OPENMX_ACC_DUMP -x OPENMX_ACC_DUMP_SCF" "$dst/$dst.sh" || fail "$dst: mpirun -x"
  grep -q "$src" "$dst/$dst.dat" && fail "$dst: stale name in deck"
  chmod +x "$dst/$dst.sh"
  echo "OK  $dst (from $src)"
}

mkcase o1 sip_ccol360_o1
mkcase g1 sip_ccol360_g1
# sanity: the two decks must differ ONLY in names and the gemmul8 line
d=$(diff <(sed 's/siacc64_o1/CASE/g' siacc64_o1/siacc64_o1.dat) \
         <(sed 's/siacc64_g1/CASE/g' siacc64_g1/siacc64_g1.dat) | grep -cE "^[<>]" )
echo "deck diff lines (expect <=2, the scf.gemmul8.enable pair): $d"
