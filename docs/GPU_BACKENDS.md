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
単一構成の CMake ビルドは、種別を省略すると C++ コアも含めて `Release` を使います。
以前の空の `CMAKE_BUILD_TYPE` も再構成時に `Release` となり、最適化なしのコアが
意図せず計算に使われることを防ぎます。明示した `Debug` などの指定は維持します。
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
コア単独では Intel の SYCL コンパイラによる `OPENMX_CORE_BACKEND=SYCL` も選択できます。
選択したデバイスに FP64 がなければ動径変換を CPU で実行します。
OpenMX 本体の Intel GPU・Metal バックエンドはまだありません。
対応範囲と検証結果は [2026-10-20 レビュー対応](CODE_REVIEW_20261020.md) に記録しています。

## 共通コードとバックエンド境界

CUDA/HIP で重複していた 56 ファイルを `source/` の単一ソースへ統合しました。
HIP の置換リストは 105 件から 49 件になり、共通化した元ファイルは
`source/backends/hip/src/` と `include/` から削除しています。

| 共通部分 | 主なソース |
| --- | --- |
| GPU API とデバイス割り当ての呼び出し | `openmx_gpu_backend.h`、`openmx_gpu_runtime.h`、`openmx_gpu_blas.h`、`openmx_gpu_solver.h`、`openmx_gpu_complex.h`、`openmx_gpu_device.h` |
| メモリ容量判定・作業分割・失敗時の復旧 | `openmx_gpu_workers.c`、`openmx_gpu_views.cpp`、`openmx_gpu_task.c` / `openmx_gpu_task_worker.cpp` / `openmx_gpu_task_cpu.cpp` / `openmx_gpu_task_packet.cpp`、`openmx_gpu_vector.c` / `openmx_gpu_vector_worker.cpp` / `openmx_gpu_vector_packet.cpp`、`openmx_gpu_bilinear.c`、collective 系 |
| 密度行列・密行列変換の GPU カーネル | `openmx_gpu_dm.cu`、`openmx_gpu_transform.cu` |
| 複数 GPU の実行制御・行列積・MAGMA 接続 | `openmx_mgpu_runtime.cu`、`openmx_mgpu_gemm.cu`、`openmx_magma_mgpu.cpp` |
| 数値計算・入力と後処理 | CWF/DMmu 派生ソルバー、ELPA/COSMA 接続、分極、混合、重なり積分、全エネルギーの呼び出し部分、`truncation.c`、`openmx_common.c/h`、`Input_std.c`、`Free_Arrays.c` など |

共通 API はネイティブの関数・型・定数へのコンパイル時の別名です。
実行時の切り替えや追加のデータ転送は行いません。HIP Makefile が定義する
`__HIP_PLATFORM_AMD__` から `OPENMX_GPU_HIP` を決定します。GPU を使わない
判定コードからも参照できるよう、`openmx_gpu_backend.h` は SDK に依存しません。
明示的なバックエンド識別子も受け付け、未実装の SYCL・METAL・HOST を
CUDA 専用アダプターへ渡すとコンパイルを停止します。
コアの `openmx_accelerator.h` は SDK から独立して、バックエンド、FP64 能力、
メモリモデルを記述します。実行方針は能力とバックエンド別の検証結果に基づきます。

`openmx_gpu_views.cpp` は C++17 の共通転送アダプターです。読み取りビューは同じGPU・peerでは
借用し、それ以外ではデバイス領域を所有します。C++ではmoveとスコープ終了による解放に対応し、
Cでは明示的に解放します。所有領域は確保元のデバイスで解放し、呼び出し元のデバイスを復元します。
peerが使えない転送の一時領域は16 MiBを上限とします。
詳細と検証範囲は [C++移行の第35段階](CPP_CORE_MIGRATION.md#第35段階-共通gpu読み取りビューとデバイス間転送のraii化) を参照してください。

密度行列・密行列変換の一時GPU領域は、`openmx_gpu_memory.hpp` のC++17の型付き所有者が管理します。
管理用のホスト確保や余分なゼロ初期化を行わず、確保元のデバイスで解放します。
読み取りビューも同じデバイス復元・解放の実装を使用します。公開前の失敗で出力を維持する
ワーカーの契約と、GeForce・データセンター・AMDの実行方針は変更していません。
[第36段階の検証](CPP_CORE_MIGRATION.md#第36段階-密度行列密行列変換の一時gpu領域を型付きraiiへ移行) に対象と実機・モデルの範囲を記載しています。

GEMMul8の常駐作業領域もCUDA/HIP共通の `DeviceBuffer` が所有し、CUDAの再利用用行列にも
同じ所有者を使います。キャッシュのヒット時に所有者のためのホスト確保・SDK照会・ゼロ初期化を
追加しません。明示的な解放が失敗した作業領域は保持して再試行し、解放後に呼び出し元のGPUを復元します。
ホスト側のキャッシュ登録がメモリ不足になった場合も既存の容量不足の回復経路へ戻します。
メモリの上限・予約量・ランク共有率、CUDAのprepared行列の有効性判定、MI300A/Radeonと
GeForce/A100向けの演算選択は維持します。キャッシュ表と設定の大域実体は残っています。
[第37段階の検証](CPP_CORE_MIGRATION.md#第37段階-gemmul8の常駐gpuキャッシュの所有化) を参照してください。

範囲タスクはCの制御・MPI通信と、C++17のGPUワーカー・CPU回復・パケット処理に分けています。
GPU領域、一時結果とCSR添字はワーカー呼び出し内の所有者が管理し、通常の解放処理を通らない
終了でも回収します。レイアウトの計算と結果の公開は共通の内部ヘッダーを使用し、CPUの演算順序・
64 MiBの作業分割と通信形式は維持しています。C/C++間はC ABIで接続します。
詳細は [第38段階](CPP_CORE_MIGRATION.md#第38段階-範囲タスクのgpuワーカーと所有権の分離) を参照してください。

CPU回復の作業領域もチャンク内の所有者が管理します。CPUカーネルのC ABI、CSR配置と縮約順序、
64 MiBの分割と確保失敗時の範囲縮小を保ちます。GPU SDKに依存しないC++17のホストモジュールで、
CUDA/HIP固有の状態を持ちません。検証範囲は
[第40段階](CPP_CORE_MIGRATION.md#第40段階-cpuタスクの作業領域をスコープ内の所有へ移行) を参照してください。

パケットの生成とremote側の解釈もSDK不要のC++17へ移しました。C/C++共通の内部wireヘッダーを使い、
16-byteの配置・CSR検証・コピーを保ちます。生成中のパケットとワーカー実行中の結果は所有者が保持し、
通常の失敗で返した結果も含め、Cの通信側がfreeする契約を維持します。
[第41段階](CPP_CORE_MIGRATION.md#第41段階-範囲タスクの通信パケットを所有するモジュールへ移行) に検証範囲を記載します。

ホスト側の履歴を使う重み付きGram積・線形結合も、Cの制御・通信・CPU回復とC++17のGPUワーカーに
分けています。GPU arenaと未公開の部分結果を所有し、添字表はarena内の位置からその都度求めます。
CUDAではNVHPC C++のOpenACC、HIPではClang C++のOpenMP targetを使い、両者のループと指示は
移行前のままです。CPUの算術・通信形式・デバイス別の方針も保ちます。
[第39段階](CPP_CORE_MIGRATION.md#第39段階-ベクトル縮約のgpuワーカーと型付き所有) に検証範囲を記載します。

ベクトルのパケット処理もSDK不要のC++17へ切り出し、ワーカー実行中の結果を所有者が保持します。
wire配置・数値データのコピーと検証を保ち、ヘッダーの余白だけをゼロに統一します。旧パケットの余白も
受理するためwire形式の変更はありません。Cの制御・CPU算術・commitとGPUワーカーは保ち、C側の不要な
OpenACC/OpenMPヘッダーを除去しました。[第42段階](CPP_CORE_MIGRATION.md#第42段階-ベクトル通信パケットの所有とヘッダーの初期化) に詳細を記載します。

コリニアクラスターの永続メタデータは、CUDA/HIPとも明示的な計算コンテキストの資源グループが
所有します。本体・精度制御・深部補助関数・Force前の解放へ借用状態を渡し、破棄時にはその所有元と
状態を受け取るコールバックで同期・解放します。変換済みXの世代番号も計算ごとに持ち、CUDAの
GEMMul8再利用キーには所有元を含めます。既存APIの再利用とは独立したキー空間です。
Cの演算とバックエンドごとの最適化を維持し、状態取得でGPUを初期化しません。GEMMul8等の大域キャッシュ表は今後の移行対象です。
[第43段階](CPP_CORE_MIGRATION.md#第43段階-コリニアクラスターの永続状態を計算コンテキストへ移行) に検証範囲を記載します。

ノンコリニアクラスターの永続状態も同じ計算所有へ移しました。collinear/noncollinearの借用状態をDFTが
受け取り、深部補助関数・精度制御・EDM・scatter・解放へ渡します。密度格子のメモリ不足時の固有ベクトル退避は、
所有元付きの短命なコールバックで要求し、密度格子に暗黙の現在ソルバーを置きません。
CUDAのarenaとHIPのdevice/host保持・MI300A方針を保ちます。
[第44段階](CPP_CORE_MIGRATION.md#第44段階-ノンコリニアクラスターと密度格子のキャッシュ連携を明示化) に詳細を記載します。

バンド両経路も、k点のoverlap・固有ベクトル・精度改善・容量判定を計算ごとの状態へ移しました。
DFTとNEGFの初期バンド計算は、SDK不要のopaque型を借用してソルバーと深部補助関数へ渡します。
HIP NCのプロセス終了コールバックは、MPI終了前に動く所有者の解放へ統一しました。
常駐・直列化・k点が多い経路の保持方針は維持しています。NC DC/DC-LNOの未設定エネルギー成分も
両バックエンドで修正しています。[第45段階](CPP_CORE_MIGRATION.md#第45段階-バンド状態の計算所有と非共線dcの出力契約) を参照してください。

DC・DC-LNO・Krylovも同じ計算所有へ移し、DFTから全7ソルバーへ借用状態を明示しています。
深部の確保・解放と呼び出しpacketも状態を受け取り、現在の所有元を指す可変な別名マクロは撤去しました。
DC-LNOのnode/device群とHIP Krylovのpreflight communicatorはMPI終了前に計算所有者が解放します。
clear/finishは状態を使用する処理が止まり、該当rank群が同じ終了境界に到達してから行います。
実コアの寿命試験と実MPI 4 rankの部分コミュニケータ試験で検証しています。閾値・メモリ判定・再試行・
AMDのGEMMul8既定値は維持します。[第46段階](CPP_CORE_MIGRATION.md#第46段階-dcdc-lnokrylovの永続状態とmpi資源を計算所有へ移行) に範囲を記録します。
全体の同時実行には、残る共通大域状態と共有キャッシュ表の移行が必要です。

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

GeForce の MPS 実行では、入力・出力・作業領域をすべて確保した共通 GPU 処理が、
計算開始前にメモリ確保の排他ロックを解放します。A100/H100 と HIP の既定は
従来の実行方針を維持します。ハードウェアごとの方針、切り替え方法と計測条件は
[性能回帰検証](PERFORMANCE_REGRESSION.md) に記載しています。

GPU 占有ロックには、プロセス内・プロセス間とも待機上限があります。
`OPENMX_GPU_LOCK_TIMEOUT_MS` の既定は `300000`（5分）で、`0` は待機しません。
`OPENMX_GPU_LOCK_DIR` にはロックファイルを置く既存の絶対パスを指定できます。
既定は `/tmp` です。同じユーザーで GPU を共有する全ランク・ジョブから見える、
同じノード内のディレクトリを指定してください。コンテナごとに異なる `/tmp` を
使う構成では、共有したローカルディレクトリへ変更する必要があります。
期限切れやロック取得失敗は診断を出して、その GPU 処理の開始を辞退します。
呼び出し元は既存の容量回復・CPU 経路、または明示的な異常終了を選びます。

## NVIDIA / CUDA

既定の Makefile は NVIDIA HPC SDK 26.9 と CUDA 13.4 を使用します。
その構成がインストールされている場合は、リポジトリのルートで次のようにビルドします。
`sm_120` は RTX 5080 の例です。A100 には `sm_80`、H100 には `sm_90` を指定します。

```sh
cmake -S . -B build/cuda \
  -DOPENMX_GPU_BACKEND=CUDA \
  -DOPENMX_GPU_ARCH=sm_120 \
  -DOPENMX_BUILD_JOBS=4 \
  '-DOPENMX_MAKE_OPTIONS=GPUSOLVER2_JOBS=4'
cmake --build build/cuda
```

数値だけの `120` も指定できます。指定は NVHPC、nvcc、ELPA/COSMA と
Kokkos/CUDA に伝わり、GPU を搭載していないログインノードでも構成できます。
`OPENMX_GPU_ARCH` が未指定または `auto` の場合は、CMake の構成時に
`nvidia-smi --query-gpu=compute_cap --format=csv,noheader` を実行し、
最初の有効な compute capability からアーキテクチャを決定します。
コマンドが存在しない場合、実行に失敗した場合、有効な GPU を認識できない場合は
構成をエラー終了します。キャッシュ済みの `CMAKE_CUDA_ARCHITECTURES` があっても
検出は省略しません。ログインノードなどでは `-DOPENMX_GPU_ARCH=sm_80` のように
計算ノードの GPU に合わせて指定してください。CPU コアだけのビルドは GPU を検出しません。
複数アーキテクチャの生成には、それぞれ別のビルドディレクトリを使用してください。

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

CMake 4 系をインストール済みでも古いバージョンが表示される場合は、
`type -a cmake` で実際の検索順を確認してください。例えば `~/bin/cmake` や
`/usr/local/bin/cmake` が、更新した `/usr/bin/cmake` より先に選ばれることがあります。
選んだ実行ファイルで `--version` を確認してから、構成とビルドの両方に同じパスを使います。
シェル設定で古い版の `CMAKE_ROOT` を固定している場合も解除し、各 CMake に
対応するモジュールを自動検出させてください。

NVIDIA の直接 make は、コンパイラ用に `PATH` を変更する前に呼び出し元の CMake を
絶対パスで記録し、C++ コアと MAGMA の構築に使います。個別に選ぶ場合は
`OPENMX_CORE_CMAKE=/absolute/path/to/cmake` と `MAGMA_CMAKE=/absolute/path/to/cmake`
を make に渡せます。

ELPA/COSMA の構築スクリプトは `PATH` を再設定するため、CMake 3.24 以上を
再利用するには `OPENMX_MAKE_OPTIONS` に `GPUSOLVER2_CMAKE=/absolute/path/to/cmake-bootstrap/bin/cmake`
も追加します。これは CMake の再構築を省略するための指定で、依存ライブラリの
生成物は引き続きこの HIP ビルド内に作られます。

## CMake の設定と生成物

| 変数 | 内容 |
| --- | --- |
| `OPENMX_GPU_BACKEND` | `CUDA`（既定）または `HIP`。別バックエンドには別ディレクトリを使用。 |
| `OPENMX_CORE_BACKEND` | 新しい C++ コアの実行空間。`SERIAL`（既定）、`OPENMP`、`CUDA`、`HIP`、`SYCL`。GPU は本体と同じバックエンドを選択。`SYCL` はコア単独で使用。 |
| `OPENMX_BUILD_APPLICATION` | `OFF` なら MPI や GPU SDK を必要としないコア単独ビルド。既定は `ON`。 |
| `OPENMX_GPU_ARCH` | CUDA は `sm_80` / `sm_90` / `sm_120`（数値のみも可）、HIP は `gfx942` / `gfx1200` など。CUDA コアの Kokkos 設定にも反映。CUDA で未指定または `auto` の場合は `nvidia-smi` で自動認識し、検出に失敗したら構成をエラー終了。 |
| `OPENMX_BUILD_JOBS` | ネイティブ make の並列数。既定は `4`。 |
| `OPENMX_MAKE_OPTIONS` | `NAME=value;NAME=value` 形式のネイティブ make 変数。シェルで引用符を付ける。 |
| `OPENMX_COPY_TO_WORK` | 成功した本体ビルドを作業ディレクトリへコピー。既定は `ON`。 |
| `OPENMX_WORK_DIR` | コピー先。既定はリポジトリの `work`。実行ファイル名は `openmx`。 |
| `CMAKE_INSTALL_PREFIX` | 未指定なら `OPENMX_WORK_DIR` にローカル配置し、実行ファイルは直下の `openmx`。明示指定時は `<prefix>/bin/openmx`。`cmake --install` の `--prefix` でも変更可能。 |

`cmake --build ... -jN` の外側の並列数では、内部の make の並列数は変更されません。
`OPENMX_BUILD_JOBS` で指定してください。依存スタックの独立した並列数は
`GPUSOLVER2_JOBS`、CUDA の MAGMA は `MAGMA_JOBS`（既定 3）で指定します。
`GPUSOLVER2_JOBS` を指定しない場合は `OPENMX_BUILD_JOBS` の値を使います。

実行ファイルは `build/cuda/openmx` または `build/hip/openmx` に生成されます。
ビルド成功後、既定では `work/openmx` も自動更新されます。コピーは一時ファイルを
経由して置き換えるため、実行中の古い実行ファイルへ書き込みません。
CUDA/HIP の両方をビルドする場合、最後に成功した側が `work/openmx` になります。
別々のコピー先には `OPENMX_WORK_DIR`、自動コピーの無効化には
`OPENMX_COPY_TO_WORK=OFF` を指定します。
`cmake --build build/hip --target native-clean` で HIP 側のネイティブ生成物を清掃できます。
コンパイラや依存ライブラリの構成を変更する場合は、新しいビルドディレクトリの使用が確実です。

```sh
# 既定: work/openmx と work/lib/libopenmx_core.so
cmake --install build/cuda
# HIP 版: cmake --install build/hip

# 別の prefix を指定する場合: <prefix>/bin/openmx と <prefix>/lib/libopenmx_core.so
cmake --install build/cuda --prefix "$HOME/.local"
```

`--install` には `cmake -B` で構成したビルドディレクトリを指定します。
上の例では `build` は親ディレクトリなので、`cmake --install build` は失敗します。
`-B build` で構成した場合には `--install build` を使えます。
インストールはビルドを実行しないため、先に `cmake --build` を完了してください。
prefix を指定しない場合は `work/openmx` に配置します。ローカルの配置先は
`OPENMX_WORK_DIR` で変更できます。構成時の `-DCMAKE_INSTALL_PREFIX=...` または
インストール時の `--prefix` で明示指定した場合は、`<prefix>/bin/openmx` に配置します。

既存のビルドに以前の `/usr/local` などの prefix がキャッシュされている場合は、
次のように一度リセットしてからビルド・インストールしてください。

```sh
cmake -S . -B build/cuda -UCMAKE_INSTALL_PREFIX
cmake --build build/cuda
cmake --install build/cuda
```

OpenMX のインストール対象は実行ファイルと `lib/libopenmx_core.so` です。
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

C++ コアの HIP 実機検証は [C++ / Kokkos 移行記録](CPP_CORE_MIGRATION.md) の
「AMD 実機検証（2026-10-19）」を参照してください。以下はコア導入前の検証記録です。

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

2026-10-19 の Force 改善では、216 Si、非共線 Cluster、18 MPI × 1 スレッド、
RTX 5080・MPS 有効の同一入力で比較しました。Force は `147.03 s` から `20.85 s`、
VNA の case-1 ローカル計算は `135.22 s` から `9.06 s` に短縮しました。
全 25 SCF 反復と力の計算は正常終了し、変更前との差は全エネルギー `1.03e-12 Ha`、
力の最大成分差 `1.01e-12 Ha/bohr` 以下でした。GPU に全ランク分が同時に収まらない
場合も、1 ランク分が収まれば配列を一度だけ転送して既存の力カーネルで一括計算します。
メモリ確保に失敗した場合は分割計算へ復旧します。比較とログは
`build/build-force-d3-20261019/force-comparison.json` と同ディレクトリに保存しています。

DFT-D3 の C6 補間は、提供された修正版の最大指数を引く処理を取り込みました。
参照値から離れた配位数でも重みのアンダーフローによる C6 の欠落を防ぎ、
エネルギーと力に使う比・配位数微分を保ちます。CUDA/HIP 共通の処理です。
`dftd3-interpolation` は高精度計算と有限差分でこの補間と微分を検証します。

新コアの単体テストは [移行手順](CPP_CORE_MIGRATION.md#テスト) を参照してください。
7ソルバー・混合の状態寿命、密度混合の作業領域と非共線エネルギー出力の12件は、アプリケーションを無効にした
CPUのみの構成でも実行します。SDK呼び出しは模擬し、実際のCメタデータ・終了処理とそのビルドの
C++コアを接続します。MPIのある構成にはDC-LNO/Krylovの実4 rank試験も登録します。
ホストCIの対象とローカル検証は [第47段階](CPP_CORE_MIGRATION.md#第47段階-ソルバー状態の寿命試験をgpu-sdk不要のciへ移行) に記載します。
Hamiltonian混合の履歴も計算ごとの状態として所有し、9件目の寿命試験を追加しています。
GPU常駐・host履歴・増分CPUの切替とworker方針は維持します。
所有・数値比較の範囲は [第48段階](CPP_CORE_MIGRATION.md#第48段階-hamiltonian混合のgpucpu履歴を計算所有へ移行) を参照してください。
新しい共通ヘッダーもHIPの `sources.json` の `integrated_shared` に登録します。
ネイティブのCIは元の `source/` 自体をinclude探索先に加えず、隔離したソースの19単位を構築します。
同じ共有実装の一時数値領域も呼び出し単位のC++所有へ移し、作業行列には連続した型付きビューを使います。
大きな実数領域は既存の上書き用テンソルを再利用し、以前の `calloc` のゼロ初期化は取得ごとに保ちます。
GPU常駐・host履歴・増分CPU・旧CPUの選択とハードウェア別の実行方針は維持します。
試験と実計算の範囲は [第49段階](CPP_CORE_MIGRATION.md#第49段階-hamiltonian混合の一時行列とスレッド作業領域をraiiへ移行) に記載します。
k空間のDIIS/Kerker混合も共有状態を計算所有へ移し、両バックエンドから同じ不透明な状態を渡します。
DGEMV/DGEMMのキャッシュ方針と `OPENMX_MIX_FAST` は維持します。
隔離したネイティブ構築は、この二つの混合実装を含む14単位ずつを検査します。
所有・CPU/GPUの数値比較は [第50段階](CPP_CORE_MIGRATION.md#第50段階-k空間の密度混合状態を計算所有へ移行) を参照してください。
DIIS/Kerkerの一時領域も呼び出し単位のC++所有へ移し、実数・複素数の行列を型付きの連続ビューにしました。
元のLAPACK LU分解、残差の初期化、履歴回転とデバイス別方針を保ちます。
NEGFの分岐と逆行列のモデル試験は、電極付きNEGFの実計算の代わりにはなりません。
[第51段階](CPP_CORE_MIGRATION.md#第51段階-k空間密度混合の一時領域を型付きraiiへ移行) に検証範囲を記録します。
実空間のDIIS/ADIIS/GR-Pulayも、一時行列をC++の呼び出し所有へ移しました。
既存の電子行列の形状を使うため、原子表を再走査する所有ポインタ木を作りません。
通常のCPU経路と高速経路は同じ平坦ビューを借用し、精度・容量の方針を維持します。
[第52段階](CPP_CORE_MIGRATION.md#第52段階-実空間の密度混合を連続した行列の所有へ移行) に数値・所有試験の範囲を記録します。
実空間DIIS/ADIIS/GR-Pulayの非ピボットLUは、SDKに依存しない単一のC++/mdspan実装を共有します。
各workerが作業領域を所有し、旧NVHPC/ROCmの積和の丸め方を3か所の明示的な`std::fma`で保ちます。
他の逆行列方式、デバイス選択、GeForceとA100/H100・AMDの精度/容量方針は維持します。
[第53段階](CPP_CORE_MIGRATION.md#第53段階-実空間密度混合の逆行列を共通のc数値モジュールへ移行) に元の丸め差と修正後の検証を記録します。
KSポテンシャルのRMM-DIISVも、両バックエンドで明示した計算状態と呼び出し所有の作業領域を共有します。
逆FFTの実空間容量、ピボットLAPACK、履歴の高速/旧設定とハードウェア別の実行方針を保ちます。
`Mixing_V.c`は共通実装の区分へ移し、隔離SDK構築は両側の呼び出し元を含む22単位を検査します。
[第54段階](CPP_CORE_MIGRATION.md#第54段階-ksポテンシャル混合の状態と一時行列を計算所有へ移行) に所有・計時・実計算の範囲を記録します。
k空間/KSポテンシャル混合のピボットLUは、明示したnative演算とcontextを受ける共通C++/mdspan実装を使います。
LAPACKのABIと選択は共通のネイティブアダプターに置き、共有コアにBLAS/LAPACKやGPU SDKの依存を加えません。
[第55段階](CPP_CORE_MIGRATION.md#第55段階-ピボットlu逆行列を明示した演算付きのcモジュールへ共通化) に所有と数値・ABIの検証範囲を記録します。
適応混合の上限・連続判定・エネルギー履歴も、両バックエンドが明示した計算コンテキストから借用します。
固定サイズのC++所有値を4混合方式で共有し、別計算への大域履歴の持越しと追加確保を避けます。
単純混合を含め、隔離SDK構築はCUDA/HIP各23単位を検査します。
[第56段階](CPP_CORE_MIGRATION.md#第56段階-適応混合の3大域変数を計算所有の固定サイズ履歴へ移行) に所有・実計算の範囲を記録します。
混合重み・残差履歴も同じ計算所有の値へ移し、入力・各混合方式・Krylov・SCF出力が明示した状態を共有します。
ADIISの状態引数と両backendのKrylovを含め、隔離SDK構築はCUDA/HIP各26単位を検査します。
[第57段階](CPP_CORE_MIGRATION.md#第57段階-混合重みと残差履歴の大域状態を計算所有へ移行) に所有・入力/出力・実計算の検証範囲を記録します。
適応混合8経路の履歴シフトと更新はSDK非依存のC++/mdspanモジュールへ共通化しました。
残差比とKerkerポテンシャルの固定倍率を区別し、nativeの符号・最大値・clampを呼び出しごとに受け取ります。
CUDA/HIP各26単位、新モジュールのsm_80/gfx942コンパイル、両nativeコンパイラと固定コアのビット比較を確認しています。
詳細は[第58段階](CPP_CORE_MIGRATION.md#第58段階-適応混合の更新処理をcmdspanへ共通化)に記録します。
電子エネルギーの5大域値も計算所有へ移し、両backendのDFT/バンドと各混合・SCF/MD出力を明示して接続しました。
隔離SDK検査はRuntest/API・MD・エネルギー分解を含む各34単位です。所有・MD/restartと未検証範囲は
[第59段階](CPP_CORE_MIGRATION.md#第59段階-電子エネルギーの5大域値を計算所有へ移行)に記録します。
全エネルギーの19内訳も計算所有に移し、両DFTの転記/加算順・SCF/MD出力を維持しました。
無確保の所有取得、通常/Solver=12形式の出力とMD/restartを検査しています。
[第60段階](CPP_CORE_MIGRATION.md#第60段階-全エネルギーの19内訳を計算所有へ移行)に検証範囲を記録します。
全エネルギー・イオン運動エネルギー・圧力体積項も計算所有に移し、CUDA/HIPのDFTと
MD・Stress・出力・XANES・NEBへ明示的に渡しています。SDK検査は初期化などを含む40単位へ拡張しました。
[第61段階](CPP_CORE_MIGRATION.md#第61段階-全エネルギーイオン運動エネルギー圧力体積項の所有)に検証範囲を記録します。
温度とNosé–Hooverの6状態も計算所有へ移し、MD・出力・継続入力に明示的に渡しています。
SDK検査は継続入力出力を含む各41単位です。CPU/GPUで非零熱浴状態を読み戻すMD継続計算も確認しました。
[第62段階](CPP_CORE_MIGRATION.md#第62段階-温度とnoséhoover状態の所有)に検証範囲を記録します。
化学ポテンシャルとXANES用の計3値もC++計算所有に移し、両バックエンドの主要ソルバーと出力に渡しています。
CUDA/HIPのSDK検査はCWF・解析も含む各74単位です。CWF出力で見つかった既存不具合とDIISの反復数差は
[第63段階](CPP_CORE_MIGRATION.md#第63段階-化学ポテンシャルの計算単位の所有)に、合格した範囲とともに記録しています。
非共線クラスターCWFの共役・投影修正と17個の一時バッファのRAII所有は、両バックエンドで共通のソースを使います。
[第64段階](CPP_CORE_MIGRATION.md#第64段階-非共線クラスターcwfの投影と一時領域)では変更単位を両SDKで検査し、
RTX 5080/CPUのAO・HO・MO実計算も確認しました。AMD実機での検証は別途必要です。
コリニアCWFも16バッファをinvocation所有へ移し、共通4ソースの未使用GPUキャッシュを除去しました。
[第65段階](CPP_CORE_MIGRATION.md#第65段階-コリニアcwfの一時領域と未使用gpu状態の除去)は361 CTestと変更4単位の両SDK検査を通過しています。
18組のSCF比較と20回の対照を記録し、hoppingの不一致5組は既存の再現性調査として継続しています。
CWFのグリッド・記述子60個もC++計算所有へ移し、共通の16関数へ型付きビューを渡しています。
[第66段階](CPP_CORE_MIGRATION.md#第66段階-cwfグリッド記述子60大域の計算所有)は367 CTest、4ホスト構成とCUDA/HIP各74単位を通過しました。
92組のSCF比較・CWFバンドのFourier整合性を確認し、hopping要素の不一致14/28組は再現性調査として保持しています。
以下の CTest は、コアを含むビルドの完了後に実行します。コアの GPU テストは実機を使います。

```sh
ctest --test-dir build/cuda --output-on-failure
ctest --test-dir build/hip --output-on-failure
```

CUDA の `vna-archive-recovery` は、VNA 行列の一括計算と容量不足時の復旧を
ASan/UBSan で検証します。FP32/FP64、ハロー、空ランク、メモリ確保・転送の失敗、
ロックと一時配列の解放、CPU 復旧後の数値結果を確認します。
実際の GPU カーネルを使う同じ試験は次のように実行できます。

```sh
OPENMX_TEST_GPU=1 CC=nvc python3 tests/run_setpro_archived_cpu.py
```

多数の MPI ランクで大きなセルを計算すると、VNA の全ランク分の配列が GPU に
同時には収まらないことがあります。CUDA 版はその場合、GPU の使用をランク間で
調整し、各ランクの配列を一度だけ転送して一括計算します。単一ランク分も収まらない
場合は、分割計算に進みます。`OPENMX_SETPRO_VERBOSE=1` を設定すると、
`HVNA archived GPU batch` を含む各段階の進行状況を標準エラーへ出力します。

2026-10-19 に RTX 5080、NVHPC 26.9 / CUDA 13.4 で、216 原子の Si
（Si7.0-s2p2d1 / Si_PBE19、200 Ry、非共線 Cluster）を検証しました。
18 MPI × 1 スレッド、MPS 有効の GPU 計算と `OPENMX_GPU=0` の CPU 計算は、
いずれも入力の全 25 SCF 反復と力の計算を正常終了しました。CPU/GPU の差は
全エネルギー `2.13e-10 Ha`、力の最大成分差 `1.28e-10 Ha/bohr` で、
両者とも許容誤差 `1e-7` 内です。CTest 全 35 件と、上記の GPU カーネル試験も合格しました。
入力、実行ログ、全エネルギー成分・648 成分の力の照合結果は
`build/sidia333-freeze/validation.json` と同ディレクトリに保存しています。

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
