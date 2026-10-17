# CUDA / HIP 統合ビルド

このリポジトリから NVIDIA 用 CUDA 版と AMD 用 HIP 版をビルドできます。
バックエンドはビルド時に選択し、実行ファイルごとに固定します。
CUDA と HIP には別々のビルドディレクトリを使用してください。
元の `~/openmx4.0_amdgpu` チェックアウトは不要です。

## 必要なツールと構成

CMake 3.20 以上、Python 3.8 以上、GNU make に加え、選択したバックエンドの
コンパイラ、MPI、数値計算ライブラリが必要です。最初にリポジトリのルートで
サブモジュールを初期化します。

```sh
git submodule update --init --recursive
```

共通ソースは `source/`、AMD 固有の実装と設定は `source/backends/hip/` にあります。
`source/backends/hip/sources.json` が HIP ビルドで置き換えるソースを記録します。
MI300A と Radeon の判定、固有のメモリ管理、ネイティブ FP64 / GEMMul8 の選択は
取り込んだ AMD 実装で処理します。

CMake は既存の Makefile にあるファイル別の最適化・オフロード設定を使います。
`build/<backend>/native/source/` に選択したソースをリンクし、GEMMul8、FFTW、MAGMA、
ELPA/COSMA などの変更される依存ソース・生成物をビルドごとに分離します。
共通アーカイブは `source/third_party/dist/`、AMD 用 MAGMA パッチなどは
`source/third_party/hip/` に保存しています。
選択結果は各ビルドディレクトリの `selected-sources.json` で確認できます。

## NVIDIA / CUDA

既定の Makefile は NVIDIA HPC SDK 26.9 と CUDA 13.4 を使用します。
その構成がインストールされている場合は、リポジトリのルートで次のようにビルドします。
`120` は RTX 5080 の例です。A100 には `80`、H100 には `90` を指定します。

```sh
cmake -S . -B build/cuda \
  -DOPENMX_GPU_BACKEND=CUDA \
  -DOPENMX_GPU_ARCH=120 \
  -DOPENMX_BUILD_JOBS=4 \
  '-DOPENMX_MAKE_OPTIONS=GPUSOLVER2_JOBS=4'
cmake --build build/cuda
```

別の SDK、CUDA、MPI を選ぶ場合はネイティブ Makefile の変数を渡します。
次のパスは実際のインストール先に置き換えてください。

```sh
cmake -S . -B build/cuda \
  -DOPENMX_GPU_BACKEND=CUDA -DOPENMX_GPU_ARCH=120 \
  '-DOPENMX_MAKE_OPTIONS=NVHPC_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5;NVHPC_CUDA_VERSION=13.2;NVHPC_CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2;GPUSOLVER2_JOBS=4'
cmake --build build/cuda
```

分離された CUDA 数学ライブラリには `NVHPC_MATH_DIR`、別の MPI には
`NVHPC_MPI_BIN` と `NVHPC_MPI_LIBDIR` も指定できます。
CUDA の MAGMA は従来どおり任意です。必要なら `WITH_MAGMA=1` を追加します。
従来の NVIDIA ビルドも使用できます。

```sh
make -C source -j4 GEMMUL8_GPU_ARCH=120 GPUSOLVER2_JOBS=4 openmx
```

## AMD / HIP

HIP 版には ROCm の clang/clang++、hipcc、flang、OpenMP オフロードと、
AOCL の BLIS、libFLAME、ScaLAPACK、FFTW3、AOCL Utilities が必要です。
ELPA の Fortran `mpi` モジュールを読むため、MPI は使用する ROCm flang と
互換のコンパイラで構築されている必要があります。gfortran 用の `mpi.mod` は
ROCm flang 用として使用できません。AOCL が依存する AOCC の Fortran ランタイムも
`AOCC_LIB` に指定します。

AMD Makefile の既定値には特定の計算機向けの `/system/apps/...` パスが含まれます。
下記の `ROCM`、`MPI_HOME`、`AOCL_HOME`、`AOCC_LIB` を実際の環境に合わせて指定します。
`gfx942` は MI300A の例です。Radeon には対象 GPU のアーキテクチャを指定してください。

```sh
cmake -S . -B build/hip \
  -DOPENMX_GPU_BACKEND=HIP \
  -DOPENMX_GPU_ARCH=gfx942 \
  -DOPENMX_BUILD_JOBS=4 \
  '-DOPENMX_MAKE_OPTIONS=ROCM=/opt/rocm-7.2.0;MPI_HOME=/path/to/openmpi-rocmflang;AOCL_HOME=/path/to/aocl/aocc;AOCC_LIB=/path/to/aocc/lib;GPUSOLVER2_JOBS=4'
cmake --build build/hip
```

標準配置と異なる環境では `ROCM_LLVM`、`MPI_LIBDIR`、`AOCL_INC`、`AOCL_LIB`、
`CLANG`、`CPPLANG`、`HIPCC`、`FLANG` も上書きできます。
初回には GEMMul8、パッチ適用済み MAGMA、ELPA/COSMA もビルドします。
ELPA/COSMA の構築には autoconf、automake、libtoolize、m4、unzip も必要です。

## CMake の設定と生成物

| 変数 | 内容 |
| --- | --- |
| `OPENMX_GPU_BACKEND` | `CUDA`（既定）または `HIP`。別バックエンドには別ディレクトリを使用。 |
| `OPENMX_GPU_ARCH` | CUDA は `80` / `90` / `120` など、HIP は `gfx942` / `gfx1200` など。未指定時はネイティブ Makefile の検出・既定値を使用。 |
| `OPENMX_BUILD_JOBS` | ネイティブ make の並列数。既定は `4`。 |
| `OPENMX_MAKE_OPTIONS` | `NAME=value;NAME=value` 形式のネイティブ make 変数。シェルで引用符を付ける。 |

`cmake --build ... -jN` の外側の並列数では、内部の make の並列数は変更されません。
`OPENMX_BUILD_JOBS` で指定してください。依存スタックの独立した並列数は
`GPUSOLVER2_JOBS`、CUDA の任意 MAGMA は `MAGMA_JOBS` で指定します。
`GPUSOLVER2_JOBS` を指定しない場合は `OPENMX_BUILD_JOBS` の値を使います。

実行ファイルは `build/cuda/openmx` または `build/hip/openmx` に生成されます。
`cmake --build build/hip --target native-clean` で HIP 側のネイティブ生成物を清掃できます。
コンパイラや依存ライブラリの構成を変更する場合は、新しいビルドディレクトリの使用が確実です。

```sh
cmake --install build/hip --prefix /path/to/install
```

インストール対象は `bin/openmx` のみです。DFT データ、入力ファイル、起動スクリプト、
依存ライブラリはコピーしません。実行時に参照するビルド内の依存ライブラリと、
ROCm / CUDA / MPI / AOCL のライブラリ配置を維持してください。

## 実行時の入力パスと MI300A

入力の `DATA.PATH` は実行時の作業ディレクトリから解釈されます。
任意の場所から実行する場合は入力に絶対パスを設定します。

```text
DATA.PATH /absolute/path/to/openmx4.0_gpu/DFT_DATA19
```

既存の `DATA.PATH ../DFT_DATA19` を使う入力は、元の `work/` から実行できます。

```sh
cd work
mpirun -np 1 ../build/cuda/openmx Si8-LNO.dat -nt 1
```

MI300A の GPU 実行では `HSA_XNACK=1` を設定します。ランクごとの CPU 配置を
ROCm 初期化前に固定する起動スクリプトは
`source/backends/hip/tools/mi300a_rank_launcher.py` にあり、HIP ビルド時には
`build/hip/native/tools/` にもコピーされます。スクリプトは OpenMPI のランク情報と
`OPENMX_AFFINITY_DIR` を必要とし、割り当てられた CPU 数と `-nt` の一致を確認します。
次は 24 コアを確保したジョブ内で、絶対 `DATA.PATH` を持つ入力を使う例です。

```sh
export HSA_XNACK=1 HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0
export OPENMX_AFFINITY_DIR="$PWD/work/mi300a-affinity-run1"
mpirun -np 8 --map-by ppr:8:node:PE=3 --bind-to core \
  python3 source/backends/hip/tools/mi300a_rank_launcher.py \
  "$PWD/build/hip/openmx" /absolute/path/to/input.dat -nt 3
```

この統合時の検証環境には ROCm/HIP がないため、HIP 版の実ビルドと AMD GPU 上の
数値・性能検証は実施していません。

## 検証

```sh
ctest --test-dir build/cuda --output-on-failure
ctest --test-dir build/hip --output-on-failure
```

共通テストはソースの選択、取り込み時のチェックサム、CUDA ヘッダーと HIP 互換
ヘッダーの分離、依存物の独立性、両 Makefile のコンパイル指示を検証します。
HIP の CTest はさらに MI300A/Radeon の演算方針、非共線バンドの復旧処理、
MPI ランク起動処理をホスト上で検証します。AMD GPU を使用する実機試験とは別です。
AMD 版から取り込んだ追加テストは `build/hip/native/tests/` から実行できます。
