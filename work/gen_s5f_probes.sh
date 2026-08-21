#!/bin/bash
# RTX 5080 campaign (rtx5080_procedure.md sec.3): create the two NC band
# probe decks missing from the Pegasus set, byte-identical to sib_nc216_p
# except System.Name / Atoms.Number / cell / coordinates:
#   sib_nc64_p  : conv 2x2x2 diamond Si, a = 10.862 Ang cubic (geometry taken
#                 verbatim from siacc64_o1, the gen_siacc64.sh cell).
#   sib_nc128_p : prim 4x4x4 diamond Si -- fcc primitive cell replicated 4^3.
#                 Supercell matrix w.r.t. the conventional cell (a=5.431 Ang):
#                     [[0,2,2],[2,0,2],[2,2,0]]   (det=16 conv cells = 128 atoms)
#                 i.e. cell vectors (0,2a,2a),(2a,0,2a),(2a,2a,0), |v|=15.361 Ang,
#                 angles 60 deg (isotropic in the plan sec.7.1 sense).
#                 Coordinates carry the same cartesian a/8(1,1,1) origin shift
#                 as the OpenMX-Viewer 64/216 decks (= +1/32 in supercell frac).
# Usage from work/:  ./gen_s5f_probes.sh
set -u
cd "$(dirname "$0")" || exit 1

TPL=sib_nc216_p
[ -s "$TPL/$TPL.dat" ] || { echo "FATAL: $TPL deck missing"; exit 1; }
[ -s siacc64_o1/siacc64_o1.dat ] || { echo "FATAL: siacc64_o1 deck missing"; exit 1; }

mkgeom() {  # $1=name $2=natoms  ; coordinate + cell blocks come on stdin
  local name=$1 nat=$2
  mkdir -p "$name"
  # template with the three geometry-bearing blocks stripped
  awk '
    /^<Atoms.SpeciesAndCoordinates/ {drop=1; print "@COORDS@"; next}
    /^Atoms.SpeciesAndCoordinates>/ {drop=0; next}
    /^<Atoms.UnitVectors/           {drop=1; print "@CELL@";   next}
    /^Atoms.UnitVectors>/           {drop=0; next}
    drop {next}
    {print}
  ' "$TPL/$TPL.dat" \
  | sed -e "s/^System.Name  *${TPL}\$/System.Name                     ${name}/" \
        -e "s/^\(Atoms.Number  *\)216\$/\1${nat}/" > "$name/$name.tpl"
}

# ---- sib_nc64_p: geometry verbatim from siacc64_o1 ------------------------
mkgeom sib_nc64_p 64
sed -n '/^<Atoms.SpeciesAndCoordinates/,/^Atoms.SpeciesAndCoordinates>/p' \
    siacc64_o1/siacc64_o1.dat > sib_nc64_p/coords.blk
sed -n '/^<Atoms.UnitVectors/,/^Atoms.UnitVectors>/p' \
    siacc64_o1/siacc64_o1.dat > sib_nc64_p/cell.blk

# ---- sib_nc128_p: generated fcc-primitive 4x4x4 ---------------------------
mkgeom sib_nc128_p 128
python3 - <<'EOF'
a = 5.431
cell = [(0.0, 2*a, 2*a), (2*a, 0.0, 2*a), (2*a, 2*a, 0.0)]
with open("sib_nc128_p/cell.blk", "w") as f:
    f.write("<Atoms.UnitVectors\n")
    for v in cell:
        f.write("    %10.7f  %10.7f  %10.7f\n" % v)
    f.write("Atoms.UnitVectors>\n")
shift = 1.0/32.0                    # cartesian a/8(1,1,1) origin shift
rows = []
for n1 in range(4):
    for n2 in range(4):
        for n3 in range(4):
            for b in (0.0, 0.25):
                rows.append(tuple((n + b)/4.0 + shift for n in (n1, n2, n3)))
assert len(rows) == 128
with open("sib_nc128_p/coords.blk", "w") as f:
    f.write("<Atoms.SpeciesAndCoordinates\n")
    for i, r in enumerate(rows, 1):
        f.write("  %3d  Si   %11.8f  %11.8f  %11.8f   2.0  2.0\n" % (i, *r))
    f.write("Atoms.SpeciesAndCoordinates>\n")

# sanity: minimum-image nearest neighbour must be a*sqrt(3)/4 for every atom
import itertools, math
cart = [[sum(rows[i][k]*cell[k][d] for k in range(3)) for d in range(3)]
        for i in range(128)]
dmin = [1e9]*128
for i, j in itertools.combinations(range(128), 2):
    best = 1e9
    for s1 in (-1,0,1):
        for s2 in (-1,0,1):
            for s3 in (-1,0,1):
                d = math.dist(cart[i], [cart[j][k]+s1*cell[0][k]+s2*cell[1][k]+s3*cell[2][k]
                                        for k in range(3)])
                best = min(best, d)
    dmin[i] = min(dmin[i], best); dmin[j] = min(dmin[j], best)
ref = a*math.sqrt(3)/4
assert all(abs(d-ref) < 1e-6 for d in dmin), (min(dmin), max(dmin), ref)
print("128-atom geometry OK: NN = %.6f Ang for all atoms (ref %.6f)" % (dmin[0], ref))
EOF
[ $? -eq 0 ] || { echo "FATAL: 128-atom generation failed"; exit 1; }

# ---- assemble + assert -----------------------------------------------------
fail=0
for nm in sib_nc64_p:64 sib_nc128_p:128; do
  name=${nm%%:*}; nat=${nm##*:}
  awk -v cb="$name/coords.blk" -v ub="$name/cell.blk" '
    /^@COORDS@$/ {while ((getline l < cb)>0) print l; next}
    /^@CELL@$/   {while ((getline l < ub)>0) print l; next}
    {print}
  ' "$name/$name.tpl" > "$name/$name.dat"
  rm -f "$name/$name.tpl" "$name/coords.blk" "$name/cell.blk"
  got_nat=$(awk '$1=="Atoms.Number"{print $2}' "$name/$name.dat")
  got_n=$(sed -n '/^<Atoms.SpeciesAndCoordinates/,/^Atoms.SpeciesAndCoordinates>/p' "$name/$name.dat" | grep -c " Si ")
  got_name=$(awk '$1=="System.Name"{print $2}' "$name/$name.dat")
  got_iter=$(awk '$1=="scf.maxIter"{print $2}' "$name/$name.dat")
  got_spin=$(awk '$1=="scf.SpinPolarization"{print $2}' "$name/$name.dat")
  ok=1
  [ "$got_nat" = "$nat" ] || { echo "FAIL $name Atoms.Number=$got_nat"; ok=0; }
  [ "$got_n" = "$nat" ]   || { echo "FAIL $name coord rows=$got_n"; ok=0; }
  [ "$got_name" = "$name" ] || { echo "FAIL $name System.Name=$got_name"; ok=0; }
  [ "$got_iter" = "3" ]   || { echo "FAIL $name scf.maxIter=$got_iter"; ok=0; }
  [ "$got_spin" = "NC" ]  || { echo "FAIL $name spin=$got_spin"; ok=0; }
  [ $ok -eq 1 ] && echo "OK $name ($nat atoms)" || fail=1
done
exit $fail
