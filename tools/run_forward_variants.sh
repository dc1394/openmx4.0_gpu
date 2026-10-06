#!/bin/bash
# Run OpenMX once per case and variant and compare the runs: the SCF-level
# measurement of the forward-transform settings of the cluster solver
# (precision stage, reuse of the prepared X, precision controller).  Works on
# a workstation and inside a batch job; one GPU node.
#
#   tools/run_forward_variants.sh -o out -n 18 -v tools/forward_variants/stage1.txt \
#       Pt63:large_example GFRAG:large_example
#
#   -o DIR     output directory; one sub-directory per case and variant
#   -b FILE    openmx binary (default: source/openmx of this repository)
#   -n RANKS   MPI ranks (default 16)
#   -v FILE    variants file, one variant per line: <label> [NAME=VALUE ...]
#                NAME with a dot is an input keyword (replaced or appended in
#                the copied input file), @append=FILE appends a file from the
#                directory of the variants file (e.g. a precision policy),
#                any other NAME is an environment variable of the run.
#   -s "..."   settings of the same form applied to every variant, e.g.
#              -s "scf.EigenvalueSolver=cluster" to run an input written for
#              another solver through the cluster solver
#   -r LABEL   reference variant of the summary (default: the first one)
#   -m "CMD"   MPI launcher with its options (default: "mpirun --bind-to core")
#   case       <input>:<example directory under work/> of this repository
#
# The summary (tools/summarize_forward_variants.py) goes to DIR/RESULTS.md.
# Start an MPS daemon beforehand if the ranks are to share the GPU through it.
set -u
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
out= binary=$repo/source/openmx ranks=16 variants= reference= common= launcher="mpirun --bind-to core"
while getopts o:b:n:v:s:r:m: option; do
  case $option in
    o) out=$OPTARG ;;
    b) binary=$OPTARG ;;
    n) ranks=$OPTARG ;;
    v) variants=$OPTARG ;;
    s) common=$OPTARG ;;
    r) reference=$OPTARG ;;
    m) launcher=$OPTARG ;;
    *) sed -n '2,27p' "$0"; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
if [ -z "$out" ] || [ -z "$variants" ] || [ $# -lt 1 ]; then sed -n '2,27p' "$0"; exit 2; fi
[ -x "$binary" ] || { echo "no executable $binary" >&2; exit 2; }
[ -f "$variants" ] || { echo "no variants file $variants" >&2; exit 2; }
python=${PYTHON:-python3}

mkdir -p "$out" || exit 1
out=$(CDPATH= cd -- "$out" && pwd)
variants_dir=$(CDPATH= cd -- "$(dirname -- "$variants")" && pwd)
cp "$variants" "$out/variants.txt"
[ -n "$reference" ] || reference=$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "$out/variants.txt")
export OMP_NUM_THREADS=1

{
  echo "date: $(date)"
  echo "host: $(hostname)"
  echo "cpu: $(sed -n 's/^model name[^:]*: *//p' /proc/cpuinfo 2>/dev/null | head -1)"
  echo "memory: $(awk '/^MemTotal/ {printf "%.0f GiB", $2 / 1048576}' /proc/meminfo 2>/dev/null)"
  echo "binary: $binary"
  echo "sha256: $(sha256sum "$binary" | cut -d' ' -f1)"
  echo "commit: $(git -C "$repo" log --oneline -1 2>/dev/null)"
  echo "modified: $(git -C "$repo" status --short -uno -- source tests tools 2>/dev/null | tr '\n' ';')"
  echo "ranks: $ranks, launcher: $launcher"
  echo "settings of every variant: ${common:-none}"
  echo "libraries: $(ldd "$binary" 2>/dev/null | awk '/cublas|cusolver|cudart|libmpi[.]/ {print $3}' | tr '\n' ' ')"
  echo "MPS daemon: $(pgrep -x nvidia-cuda-mps > /dev/null 2>&1 && echo running || echo not running)"
  echo "environment: $(env | grep -E '^(CUBLAS|CUSOLVER|CUDA|NVHPC|GEMMUL8|OPENMX|OMP)_' | sort | tr '\n' ';')"
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv 2>/dev/null
} > "$out/environment.txt"
cat "$out/environment.txt"

for item in "$@"; do
  name=${item%%:*}; dir=${item#*:}
  input=$repo/work/$dir/$name.dat
  [ -f "$input" ] || { echo "no input $input" >&2; exit 2; }
  echo
  echo "== $name"
  mkdir -p "$out/$name"
  ln -sfn "$repo/DFT_DATA19" "$out/$name/DFT_DATA19"
  while read -r label settings; do
    case "$label" in ''|'#'*) continue ;; esac
    run=$out/$name/$label
    mkdir -p "$run"
    cd "$run" || exit 1
    cp "$input" "$name.dat"
    exports=
    for setting in $common $settings; do
      key=${setting%%=*}; value=${setting#*=}
      case "$key" in
        @append) printf '\n' >> "$name.dat"; cat "$variants_dir/$value" >> "$name.dat"; cp "$variants_dir/$value" "$out/" ;;
        *.*) if grep -q -i "^$key " "$name.dat"; then
               # values may hold slashes (paths): '|' delimits, and | & \ in the value are escaped
               escaped=$(printf '%s' "$value" | sed -e 's/[|&\\]/\\&/g')
               sed -i "s|^$key .*|$key   $escaped|I" "$name.dat"
             else printf '\n%s   %s\n' "$key" "$value" >> "$name.dat"; fi ;;
        *)   exports="$exports $key=$value" ;;
      esac
    done
    start=$(date +%s.%N)
    env $exports $launcher -np "$ranks" "$binary" "$name.dat" -nt 1 > "$name.std" 2>&1 < /dev/null
    code=$?
    printf '  %-14s exit=%d wall=%7.1fs SCF=%s %s\n' "$label" $code \
      "$(awk -v a="$start" -v b="$(date +%s.%N)" 'BEGIN {print b - a}')" \
      "$(grep -c -E '^ +SCF= +[0-9]+ ' "$name.out" 2>/dev/null)" \
      "$(grep -E '^ *Utot\.' "$name.out" 2>/dev/null | head -1)"
    grep -h 'forward transform' "$name.std" | cut -c1-230 | sed 's/^/      /'
    grep -h -E 'forward GEMMs: (stage [0-9]+ .* from the next|final stage|.*rejected)' "$name.std" | sed 's/^/      /' | head -12
    cd "$out" || exit 1
  done < "$out/variants.txt"
done

echo
"$python" "$repo/tools/summarize_forward_variants.py" --ref "$reference" "$out"/*/ | tee "$out/RESULTS.md"
