# CUDA / HIP 統合ビルド

このリポジトリから NVIDIA 用 CUDA 版と AMD 用 HIP 版をビルドできます。
バックエンドはビルド時に選択し、実行ファイルごとに固定します。
CUDA と HIP には別々のビルドディレクトリを使用してください。
元の `~/openmx4.0_amdgpu` チェックアウトは不要です。

## 必要なツールと構成

CMake 3.25.2 以上、C++20 対応コンパイラ・標準ライブラリ、Python 3.8 以上、GNU make に加え、選択したバックエンドの
コンパイラ、MPI、数値計算ライブラリが必要です。最初にリポジトリのルートで
使用するバックエンドのサブモジュールを初期化します。HIP は GEMMul8 のみを使い、
FFTW3 には AOCL のライブラリを使用します。

```sh
# AMD / HIP
git submodule update --init --recursive source/third_party/GEMMul8

# NVIDIA / CUDA（FFTW3 のソースも初期化）
git submodule update --init --recursive
```

ソースアーカイブから依存物を展開済みの場合、そのディレクトリに Git メタデータが
なくてもビルドできます。特に HIP の構築で、使用しない FFTW3 の展開済み
ディレクトリをサブモジュールに変換する必要はありません。

共通ソースは `source/`、AMD 固有の実装と設定は `source/backends/hip/` にあります。
`source/backends/hip/sources.json` が HIP ビルドで置き換えるソースを記録します。
MI300A と Radeon の判定、固有のメモリ管理、ネイティブ FP64 / GEMMul8 の選択は
取り込んだ AMD 実装で処理します。

CPU と GPU の固有値計算は、同じ ELPA 2026.02.002 ライブラリを使用します。
`scf.eigen.lib elpa1` は一段階法、`elpa2` は二段階法を CPU 上で実行します。
`OPENMX_GPU=0` で GPU を無効にした場合も新しい ELPA の CPU 経路に切り替わります。
OpenMX が以前内蔵していた古い ELPA のソースとビルドルールは除去しました。

CMake は既存の Makefile にあるファイル別の最適化・オフロード設定を使います。
`build/<backend>/native/source/` に選択したソースをリンクし、GEMMul8、FFTW、MAGMA、
ELPA/COSMA などの変更される依存ソース・生成物をビルドごとに分離します。
共通アーカイブは `source/third_party/dist/`、AMD 用 MAGMA パッチなどは
`source/third_party/hip/` に保存しています。
選択結果は各ビルドディレクトリの `selected-sources.json` で確認できます。

`core/` の新しい C++ モジュールは CMake が構築し、両本体へ共有ライブラリとして
リンクします。直接 make を使う場合も、この部分には CMake が必要です。
初回の構成時に Kokkos 5.2.2 を取得して SHA-256 を検証します。オフラインでは
`FETCHCONTENT_SOURCE_DIR_KOKKOS` で展開済みソースを指定できます。
コアの既定値は `OPENMX_CORE_BACKEND=SERIAL` です。PAO 動径変換も GPU で実行する場合は
`CUDA` / `HIP` と、対応するコンパイラ・Kokkos アーキテクチャを指定します。
詳細は [C++ / Kokkos 移行手順](CPP_CORE_MIGRATION.md) を参照してください。

## 共通コードとバックエンド境界

CUDA/HIP で重複していた 56 ファイルを `source/` の単一ソースへ統合しました。
HIP の置換リストは 105 件から 49 件になり、共通化した元ファイルは
`source/backends/hip/src/` と `include/` から削除しています。

| 共通部分 | 主なソース |
| --- | --- |
| GPU API とデバイス割り当ての呼び出し | `openmx_gpu_backend.h`、`openmx_gpu_runtime.h`、`openmx_gpu_blas.h`、`openmx_gpu_solver.h`、`openmx_gpu_complex.h`、`openmx_gpu_device.h` |
| メモリ容量判定・作業分割・失敗時の復旧 | `openmx_gpu_workers.c`、`openmx_gpu_views.c`、`openmx_gpu_task.c`、`openmx_gpu_vector.c`、`openmx_gpu_bilinear.c`、collective 系 |
| 密度行列・密行列変換の GPU カーネル | `openmx_gpu_dm.cu`、`openmx_gpu_transform.cu` |
| 複数 GPU の実行制御・行列積・MAGMA 接続 | `openmx_mgpu_runtime.cu`、`openmx_mgpu_gemm.cu`、`openmx_magma_mgpu.cpp` |
| 数値計算・入力と後処理 | CWF/DMmu 派生ソルバー、ELPA/COSMA 接続、分極、混合、重なり積分、全エネルギーの呼び出し部分、`truncation.c`、`openmx_common.c/h`、`Input_std.c`、`Free_Arrays.c` など |

共通 API はネイティブの関数・型・定数へのコンパイル時の別名です。
実行時の切り替えや追加のデータ転送は行いません。HIP Makefile が定義する
`__HIP_PLATFORM_AMD__` から `OPENMX_GPU_HIP` を決定します。GPU を使わない
判定コードからも参照できるよう、`openmx_gpu_backend.h` は SDK に依存しません。

OpenACC と OpenMP target の指示、物理メモリ・UUID・GPU 間接続の取得、
ELPA の GPU オプションと HIP 小行列対策はバックエンドごとの分岐に保持しています。
CUDA の弱 FP64 GPU 向け MAGMA FP32 補正と共有 Band グループ、HIP の MAGMA 共通ロック、
入力切り替え時のキャッシュ破棄、GPU 不在時の後処理も元の方針を維持します。
主要 SCF ソルバー、Hamiltonian・密度・軌道・力などの専用実装と、
ツールチェーン・依存ライブラリの構築は引き続きバックエンド別です。

`sources.json` の `shared`、`overrides` と `import_sha256` は変更していない取り込み元を記録し、
共通実装は `integrated_shared` に登録します。取り込み後に改修したバックエンド固有の
実装は `integrated_overrides` に登録し、移植元の不変ファイルと区別します。共通コードを修正すると
両ビルドへ反映されます。既存の HIP ビルドを再開したときも、準備処理が
削除済みの置換ソースへのリンクを共通ソースへ張り替えます。
両 Makefile は共通 GPU インターフェースヘッダーの変更を再コンパイルへ反映します。

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
CUDA 版の MAGMA は、複数 GPU を使う Cluster／Band の密行列固有値計算に必要です。
ビルドするノードで `nvidia-smi` が GPU を 2 枚以上表示した場合は自動でリンクし、未導入なら先にビルドします
（`WITH_MAGMA=auto`、既定。導入済みのものはそのまま使います）。GPU のないログインノードで複数 GPU ノード向けに
ビルドする場合や、ツールチェーンの変更後に作り直す場合は `WITH_MAGMA=1`、MAGMA なしでビルドする場合は
`WITH_MAGMA=0` を `OPENMX_MAKE_OPTIONS` に追加します。
MAGMA 2.10.0 は HIP 用に同梱したアーカイブ（SHA-256 を検証）から作るので、ネットワーク接続は不要です。
その構築には CUDA 対応の CMake 3.18 以上が必要です。
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

HIP Makefile は、選択した `ROCM_LLVM_LIB` を `LIBRARY_PATH` の先頭に置きます。
ROCm clang は環境変数のライブラリ検索先を明示した `-L` より先に使うため、
この設定で Intel oneAPI や AOCC の別の OpenMP ランタイムが選ばれることを防ぎます。
追加の検索先は維持します。

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

バンドルした ELPA 2026.02.002 には、ROCm flang で `type(c_ptr)` の
割当出力先が誤って渡される場合があるため、構築スクリプトがホスト／デバイスの
割当ラッパーを `integer(c_intptr_t)` 経由に修正します。元アーカイブは変更せず、
ビルド内の ELPA ソースへ再適用可能な形で反映します。対応しないソースでは
構築を停止し、意図しない書き換えを防ぎます。

`cmake` が `PATH` にない場合、以前の依存スタック構築で生成した
`cmake-bootstrap/bin/cmake` など、既存の CMake を絶対パスで利用できます。
次の `CMAKE_BIN` を実際の実行ファイルに置き換え、上記の構成・ビルドコマンドの
`cmake` を `"$CMAKE_BIN"` として実行してください。

```sh
CMAKE_BIN=/absolute/path/to/cmake-bootstrap/bin/cmake
"$CMAKE_BIN" --version
```

ELPA/COSMA の構築スクリプトは `PATH` を再設定するため、CMake 3.24 以上を
再利用するには `OPENMX_MAKE_OPTIONS` に `GPUSOLVER2_CMAKE=/absolute/path/to/cmake-bootstrap/bin/cmake`
も追加します。これは CMake の再構築を省略するための指定で、依存ライブラリの
生成物は引き続きこの HIP ビルド内に作られます。

## CMake の設定と生成物

| 変数 | 内容 |
| --- | --- |
| `OPENMX_GPU_BACKEND` | `CUDA`（既定）または `HIP`。別バックエンドには別ディレクトリを使用。 |
| `OPENMX_CORE_BACKEND` | 新しい C++ コアの実行空間。`SERIAL`（既定）、`OPENMP`、`CUDA`、`HIP`。GPU は本体と同じバックエンドを選択。 |
| `OPENMX_BUILD_APPLICATION` | `OFF` なら MPI や GPU SDK を必要としないコア単独ビルド。既定は `ON`。 |
| `OPENMX_GPU_ARCH` | CUDA は `80` / `90` / `120` など、HIP は `gfx942` / `gfx1200` など。未指定時はネイティブ Makefile の検出・既定値を使用。 |
| `OPENMX_BUILD_JOBS` | ネイティブ make の並列数。既定は `4`。 |
| `OPENMX_MAKE_OPTIONS` | `NAME=value;NAME=value` 形式のネイティブ make 変数。シェルで引用符を付ける。 |

`cmake --build ... -jN` の外側の並列数では、内部の make の並列数は変更されません。
`OPENMX_BUILD_JOBS` で指定してください。依存スタックの独立した並列数は
`GPUSOLVER2_JOBS`、CUDA の MAGMA は `MAGMA_JOBS`（既定 3）で指定します。
`GPUSOLVER2_JOBS` を指定しない場合は `OPENMX_BUILD_JOBS` の値を使います。

実行ファイルは `build/cuda/openmx` または `build/hip/openmx` に生成されます。
`cmake --build build/hip --target native-clean` で HIP 側のネイティブ生成物を清掃できます。
コンパイラや依存ライブラリの構成を変更する場合は、新しいビルドディレクトリの使用が確実です。

```sh
cmake --install build/hip --prefix /path/to/install
```

OpenMX のインストール対象は `bin/openmx` と `lib/libopenmx_core.so` です。
Kokkos のインストール規則も実行されます。DFT データ、入力ファイル、起動スクリプト、
既存の数値計算ライブラリはコピーしません。実行時に参照するビルド内の依存ライブラリと、
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

次は C++ コア導入前の AMD 検証記録です。新コアの HIP デバイス実行はまだ検証していません。

2026-10-17 に Radeon RX 9060 XT（gfx1200、1 GPU）、ROCm 7.2.0、
ROCm clang 版 OpenMPI 5.0.9、AOCL 5.2.0 で HIP 版を依存ライブラリから構築し、
4 MPI × 4 スレッド／8 MPI × 2 スレッド（最大 16 CPU コア）で検証しました。
CPU/GPU の数値比較、S 全例、L3 代表例、Cr2 強制 GPU 対角化、ELPA/COSMA、
容量復旧経路の強制試験、同一 MPI プロセス内での S 全例を合わせて 88 計算、CTest 8 件、
追加ホストテスト 19 件が合格しました。小規模 S 例では既定の閾値による CPU 密対角化を
許容し、強制 GPU 例では GPU 固有値分解の実行証跡を別途確認しています。
容量試験は `OPENMX_SETHAM_GPU=0` でエネルギー・双極子の復旧経路に限定しました。
32Si の全段復旧を強制した予備試験は Hamiltonian 積分が遅く、300 秒で終了しました。
その診断ログは試験記録内の `archives/capacity-allpaths-timeout/` に保存し、合格数には含めません。
最終結果は `work/codex_hip_integration_20261017/final/validation-report.txt` と
`final-audit.json` に保存しています。MI300A、物理マルチ GPU、複数ノードでの実行と、
L/L2/L3 全例の実行・性能評価はこの検証に含みません。

## 検証

新コアの単体テストは [移行手順](CPP_CORE_MIGRATION.md#テスト) を参照してください。
以下の CTest は、コアを含むビルドの完了後に実行します。コアの GPU テストは実機を使います。

```sh
ctest --test-dir build/cuda --output-on-failure
ctest --test-dir build/hip --output-on-failure
```

共通テストはソースの選択、取り込み時のチェックサム、CUDA ヘッダーと HIP 互換
ヘッダーの分離、依存物の独立性、両 Makefile のコンパイル指示を検証します。
共通化後は、両バックエンドが同じ実体を参照すること、置換元の重複が残らないこと、
既存ビルドのリンク更新も検証します。
HIP の CTest はさらに MI300A/Radeon の演算方針、非共線バンドの復旧処理、
MPI ランク起動処理をホスト上で検証します。AMD GPU を使用する実機試験とは別です。
AMD 版から取り込んだ追加テストは `build/hip/native/tests/` から実行できます。

共通 GPU 処理のホスト試験は、GPU API を模擬して数値結果、メモリ境界、
割り当て失敗時の巻き戻し、デバイスとスコープの復元を検証します。
CUDA には `tests/`、HIP には準備済みの `build/hip/native/tests/` を指定します。
実機のストリーム、P2P 転送、GPU コンパイラによるオフロードの検証は別途必要です。

```sh
# ASan を使う試験ではスタックサイズを unlimited にしないでください。
ulimit -s 8192
for backend_tests in tests build/hip/native/tests; do
  for test in gpu_vector gpu_task gpu_bilinear gpu_workers gpu_dense_workers \
              mgpu_gemm mgpu_runtime mgpu_band_plan mgpu_band_group magma_mgpu \
              dm_capacity grid_recovery energy_capacity hamiltonian_capacity \
              projector_recovery setpro_capacity density_capacity; do
    python3 "$backend_tests/run_${test}_cpu.py" || exit 1
  done
done
```

2026-10-18 の共通化では、RTX 5080 環境で CUDA 本体をビルドし、4 MPI × 1 スレッドの
CPU／CUDA `-runtest` 全 14 件を、変更前の実行ファイルと付属参照値の両方に対して
検証しました。変更前との差は CPU 経路で出力精度内のゼロ、CUDA 経路で最大
`7e-12 Ha`（エネルギー）、`2e-12 Ha/bohr`（力）です。
両バックエンドの上記ホスト試験、メモリ・容量判定・ELPA/COSMA の MPI 試験、
CPU ELPA 接続試験も合格しています。HIP 向け共通 C ソース 38 件は OpenMP を有効にした
ホスト構文検査を行いました。この環境には ROCm と AMD GPU がないため、今回の変更の
HIP デバイスコンパイルと AMD 実機検証は未実施です。

CPU の ELPA 接続部は、実際の MPI と LAPACK を使って検証できます。
次の試験は実数・複素数、一段階法・二段階法、部分固有ベクトル、長方形のプロセス
グリッド、MPI 部分コミュニケータ、空のローカル行・列を確認します。
HIP ビルドでは `--native-source build/hip/native/source` を指定します。

```sh
python3 tests/run_elpa_cpu_mpi.py --native-source build/cuda/native/source
```

標準の CPU 回帰試験は、既存の入力と参照値を読み取り、別のディレクトリに結果を
保存して実行します。各出力先には未使用のディレクトリを指定してください。
`--reference` は全エネルギーに加え、各原子の力の全成分を参照値と比較します。

```sh
python3 tools/run_gpu_suites.py --binary build/cuda/openmx \
  --output build/cpu-elpa-runtest --suite S --ranks 18 --threads 1 \
  --env OPENMX_GPU=0 --together
python3 tools/compare_gpu_suites.py --reference build/cpu-elpa-runtest

python3 tools/run_gpu_suites.py --binary build/cuda/openmx \
  --output build/cpu-elpa-runtestL --suite L --ranks 18 --threads 1 \
  --env OPENMX_GPU=0 --together
python3 tools/compare_gpu_suites.py --reference build/cpu-elpa-runtestL
```
