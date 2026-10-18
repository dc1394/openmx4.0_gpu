# OpenMX 本体の C++ / Kokkos 移行

作業ブランチは `refactor/cpp-kokkos-core`。移行開始点は
`4d8c7e0f01133927dbd76ce24f0e44d2ceb2f87e` です。
この文書は完了した移行単位と、残っている作業を記録します。
OpenMX 本体全体の C++ 化、大域変数ゼロ、全カーネルの Kokkos 化はまだ完了していません。

## 導入した構成

`core/` は、MPI や OpenMX の大域変数ヘッダーを必要とせずにビルド・テストできる
C++ ライブラリです。ヘッダーと実装を分離した通常の C++ モジュール構成を使い、
C++20 の言語機能である named modules は使用していません。

| 責務 | 実装 | 所有権と依存関係 |
| --- | --- | --- |
| 多次元ホスト配列 | `core/include/openmx/core/tensor.hpp` | `std::vector` の単一割り当てを所有し、`mdspan` で参照する。コピーは独立、ムーブは所有権を移す。 |
| PAO の計算状態 | `core/src/pao_storage.cpp` | 格子・原子密度・波動関数・Fourier 表を計算ケースごとに所有。数値配列は連続配置し、数値モジュールには `mdspan` を渡す。 |
| 球 Bessel 関数 | `core/include/openmx/core/bessel.hpp` | 入出力と角運動量を引数で受ける純粋なカーネル。CPU/CUDA/HIP で同じ実装。 |
| 動径表と補間 | `core/src/radial_table.cpp` | 化学種ごとの入力表をコピーして所有。PAO、非局所 projector、中性原子ポテンシャルの境界条件を明示して補間する。 |
| 波数空間の補間 | `core/src/radial_table.cpp` | 波数を `i*step` で求め、PAO/projector の共通補間へ渡す。格子配列を所有せず、`NormK` を必要としない。 |
| PAO/NLP/VNA 動径 Fourier 変換 | `core/src/radial.cpp` | 半径、積分の測度、波数、チャネル、サンプルを明示的に受け取る。ホスト配列は `Tensor`、実行用配列は `Kokkos::View` が所有。 |
| FFT グリッド検証 | `core/include/openmx/core/fft_grid.hpp` | 正の整数を 2、3、5、7 の積に分解できるか判定。大域の可変テーブルを持たない。 |
| 周期セル逆引き | `source/openmx_periodic.h` | セルの半径と座標から添字を直接計算。逆引き用の所有配列を持たない。 |
| 既存 C との境界 | `source/openmx_core*.h`、`core/src/c_api.cpp`、`core/src/pao_compat.hpp` | 不透明なコンテキストと C ABI。入力表、出力先、実行デバイス、エラー出力先を引数で受け取る。C の互換ポインタ表はここで構築する。 |

`source/FT_PAO.c`、`FT_NLP.c`、`FT_VNA.c` は入出力・MPI 通信と新ライブラリの接続を担います。
従来の `double*** TmpRF` / `TmpNRF`、`double** SphB`、VNA の一時配列と
その手動確保・解放を除去しました。変換中のサンプリングに大域配列を読むコールバックは使いません。
`RadialF.c`、`Nonlocal_RadialF.c`、`VNAF.c` の他の利用箇所も共通 C++ 補間処理を呼びます。
スカラーの互換呼び出しは既存の行を借用し、呼ぶたびに表全体をコピーしません。
`Spherical_Bessel.c` も共通 C++ カーネルを呼ぶ薄い C アダプターです。

通常計算・標準スイート・NEB では、`Spe_PAO_XV`、`Spe_PAO_RV`、`Spe_Atomic_Den`、
`Spe_Atomic_Den2`、`Spe_PAO_RWF`、`Spe_RF_Bessel` の数値領域を `PaoStorage` が所有します。
読み込み前に確保し、入力ファイルの読み込み、変換、MPI 配布、restart の読み書き、SCF 中の
参照まで同じ領域を使います。各配列は 1 つの連続領域で、未使用の容量と行末もゼロ初期化します。
既存の `Spe_*` 大域ポインタは借用ビューとして残り、宣言の除去は参照側の移行時に行います。
互換ポインタ表には RAII の `std::vector` を使い、数値データの所有と分離します。

`Spe_VNL`、`Spe_Vna`、NLP/VNA の変換結果、化学種メタデータはまだ C 側が所有します。
PAO ファイルの解析、MPI 配布と restart の制御も C 側に残ります。変換 API が作る
`RadialTable` は引き続き変換スコープ中の入力コピーであり、所有値の直接参照への切り替えは
今後の作業です。多重ポインタは移行用の境界で受け渡し、数値モジュールには渡しません。

`RF_BesselF` と CUDA/HIP 両方の `NLRF_BesselF` は同じ C++ 補間を呼びます。
最初の区間での外挿、上端の仮想点、projector の微小刻み幅での線形処理を維持します。
1 点・2 点の表も明示的に処理し、元の PAO 実装の無限ループ・範囲外参照を避けます。
`NormK` の宣言・確保・生成・解放をすべて除去し、restart 時も格子の再生成を必要としません。

Kokkos 経路は、独立した波数と軌道に対して並列化します。積分の加算は元と同じ
動径グリッド順序を保ち、作業領域は波数 32 点ずつに制限します。
CUDA/HIP の API やカーネルの複製はこの数値モジュールにはありません。
`Kokkos::View<double**>` 等の `*` はランクの型表現であり、行ごとに `malloc` する
多重ポインタ構造ではありません。

PAO/NLP は等間隔の半径と波数を使い、VNA は `r=x*x` の格子・`2*r*r*x` の測度と
Gauss–Legendre の波数を使います。測度を事前計算し、Bessel 値との積、端点の `0.5`、
加算、最後の刻み幅の順序を保持します。NLP の projector 番号と角運動量は別々に扱い、
同じ角運動量の複数 projector と 1/2 スピン成分を明示的に並べます。
補間は旧実装の最初の区間・外側・原点付近の扱いを維持しています。
VNA の上端では、従来の範囲外読み取りをせずに端点用の値を選びます。

## 実行状態と RAII

```text
main
  └─ openmx_core_run: スタック上の Context を所有
       └─ 通常計算 / Runtest / NEB
            ├─ readfile_with_context → SetPara_DFT → ReadPara_DFT
            │    └─ Allocate_Arrays_with_context → PAO 所有ストレージ → 読み込み
            ├─ FT_PAO / FT_NLP / FT_VNA → C++ 動径変換 → 結果の格納
            └─ Free_Arrays_with_context → GPU キャッシュ破棄 → PAO ストレージ解放
       └─ openmx_core_finish → MPI_Finalize → Context の破棄
```

コンテキストのプロセス大域ポインタやシングルトンは置きません。
`-runtest` のケース間と NEB の画像間では同じコンテキストを渡し、Kokkos の
初期化・終了を繰り返しません。PAO の所有ストレージは計算ケースごとに解放し、
同じコンテキストで次の形状を確保できます。確保済みのビューを無効にする上書き確保は拒否します。
確保途中の例外では完成した配列を自動解放し、C 側には部分的なビューを公開しません。
コンテキストの破棄と `openmx_core_finish` でも残った PAO 配列を解放します。
変換用の一時配列は各変換のスコープで解放し、非同期処理は
RAII の fence により解放前に完了させます。コピー途中の例外でも `Tensor` の形状と
ストレージの整合性を維持します。

GPU 対応の Kokkos は初期化時に GPU を使うため、GPU 無効時の変換は GPU の初期化を
必要としないホスト実装を呼びます。既存コードが後の入力でランクと GPU の対応を
変えた場合も、異なるデバイスを前提とする実行空間の再利用を避け、ホストで計算します。
別の所有者が Kokkos を初期化している場合も、そのランタイムを終了せずホストで計算します。

旧 `readfile` / `run_main` / `openmx_api` の呼び出しは、null コンテキストで
ホスト実装を使えます。この互換経路の PAO 確保・解放は従来の C 実装を維持し、
コンテキストを隠れた大域変数で補いません。コンテキスト付きの確保と解放は同じ所有者を
渡す必要があります。新 C ABI から C++ 例外を外へ出さず、数値計算が成功するまで
既存出力配列を更新しません。新しい表変換 API は出力行も先に検証し、成功後の反映には
割り当てやコールバックを使いません。初期段階の `openmx_core_pao_transform` 互換 API は
引き続き提供し、その格納コールバック自体は例外を投げないことが前提です。
既存 C 本体の深いエラー処理には `exit`、`MPI_Abort`、手動解放が残っています。
この段階で OpenMX 全体の例外安全性や複数計算の同時実行を保証するものではありません。

## ビルド

CMake 3.25.2 以上と C++20 対応のコンパイラ・標準ライブラリが必要です。
Kokkos は [5.2.2 の公式リリース](https://github.com/kokkos/kokkos/releases/tag/5.2.2) に固定し、
FetchContent が SHA-256 を検証します。

```text
d6557aaef39302282a15f9c770433d1fcdf4e961dfd6d9ed726b9d0d0f546b9f
```

`Kokkos/mdspan` は同リリースに同梱されたものを使い、別のコピーをダウンロードしません。
オフラインでは、検証・展開済みの Kokkos ソースを
`-DFETCHCONTENT_SOURCE_DIR_KOKKOS=/absolute/path/to/kokkos-5.2.2` で指定します。
この上書き時には FetchContent のアーカイブ検証を通らないため、指定先の版を管理してください。

### CPU だけでコアを開発する

```sh
cmake -S . -B build/core-host -DOPENMX_BUILD_APPLICATION=OFF \
  -DOPENMX_CORE_BACKEND=SERIAL -DCMAKE_BUILD_TYPE=Debug
cmake --build build/core-host --parallel 4
ctest --test-dir build/core-host --output-on-failure

# OpenMP の実行空間を使う場合
cmake -S . -B build/core-openmp -DOPENMX_BUILD_APPLICATION=OFF \
  -DOPENMX_CORE_BACKEND=OPENMP -DCMAKE_BUILD_TYPE=Debug
cmake --build build/core-openmp --parallel 4
OMP_NUM_THREADS=2 OMP_PROC_BIND=false ctest --test-dir build/core-openmp --output-on-failure
```

この構成は OpenMX 実行ファイルを作らず、MPI・ELPA・CUDA・ROCm を必要としません。
`cmake -S core ...` でもコアを単独で構築できます。
ホストビルドは C++23 と `std::mdspan` の両方が使えれば自動選択します。
それ以外は C++20 と Kokkos の mdspan を使います。
`-DCMAKE_CXX_STANDARD=23` だけでは、標準ライブラリに存在しない `std::mdspan` は
利用できません。`core-tensor` テストの出力で `mdspan=std` / `mdspan=Kokkos` を確認できます。
`-DOPENMX_CORE_FORCE_KOKKOS_MDSPAN=ON` で後者を明示的に検証できます。

### OpenMX 本体と CUDA

本体の `OPENMX_GPU_BACKEND` と新しいコアの `OPENMX_CORE_BACKEND` は別の設定です。
コアの既定値は `SERIAL` で、既存 GPU ソルバーの選択には影響しません。
PAO/NLP/VNA 変換も Kokkos/CUDA で動かすには、例えば RTX 5080 で次を指定します。

```sh
cmake -S . -B build/cuda -DOPENMX_GPU_BACKEND=CUDA \
  -DOPENMX_GPU_ARCH=120 -DOPENMX_BUILD_JOBS=4 \
  -DOPENMX_CORE_BACKEND=CUDA \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DKokkos_ARCH_BLACKWELL120=ON
cmake --build build/cuda --parallel 4
ctest --test-dir build/cuda --output-on-failure
```

GPU コアの既定規格は C++20 です。CMake の CUDA 言語を使い、同一の `.cpp` を
CUDA コンパイラで構築します。実行時は既存の GPU 利用可否・ランク割り当てに従います。
`OPENMX_GPU=0` や `scf.eigen.lib elpa2` ではコアも CPU 経路を使います。

### OpenMX 本体と HIP

以下は MI300X (`gfx942`) 用の構成例です。ROCm / MPI / AOCL の設定は
[既存のビルド手順](GPU_BACKENDS.md#amd--hip) と合わせて指定してください。
このブランチの新コアは AMD 実機ではまだ検証していません。

```sh
cmake -S . -B build/hip -DOPENMX_GPU_BACKEND=HIP \
  -DOPENMX_GPU_ARCH=gfx942 -DOPENMX_CORE_BACKEND=HIP \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DKokkos_ARCH_AMD_GFX942=ON \
  '-DOPENMX_MAKE_OPTIONS=ROCM=/opt/rocm;MPI_HOME=/path/to/mpi;AOCL_HOME=/path/to/aocl;AOCC_LIB=/path/to/aocc/lib'
cmake --build build/hip --parallel 4
ctest --test-dir build/hip --output-on-failure
```

HIP でも CMake の HIP 言語で同じカーネルを構築します。
MI300A や Radeon では対象に合う CMake と Kokkos のアーキテクチャを指定します。
Kokkos の [設定ガイド](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html) も参照してください。
本体 CUDA とコア HIP、またはその逆の組み合わせは構成時に拒否します。

### 直接 make を使う場合

両ネイティブ Makefile が `source/openmx_core.mk` を読み込み、C++ コアだけを
CMake で構築します。CMake ドライバー経由では、このルールに完成した共有ライブラリを渡します。

```sh
make -C source -j4 openmx OPENMX_CORE_CMAKE=/absolute/path/to/cmake \
  OPENMX_CORE_CXX=g++ OPENMX_CORE_BACKEND=SERIAL
```

`OPENMX_CORE_CMAKE_OPTIONS` で CMake オプションを追加できます。
構成やコンパイラを変える場合は `OPENMX_CORE_BUILD_DIR` も分けてください。
本体のリンクには `libopenmx_core.so` が必要です。CMake のインストールでは
`bin/openmx` と `lib/libopenmx_core.so` を配置し、実行ファイルはビルド時の場所と
`$ORIGIN/../lib` を探索します。従来の MPI・数値計算ライブラリへの依存も残ります。

## 大域変数の棚卸し

`tools/audit_common_globals.py` は Clang の C 構文木から、`openmx_common.h` 自体が
宣言・定義するファイルスコープのオブジェクトを数えます。SDK や MPI ヘッダーの
変数を除外し、同じヘッダーのベースラインを `docs/core-migration-baseline.json` に保存しています。

| 対象 | 開始時 | この段階 |
| --- | ---: | ---: |
| 共通ヘッダーのオブジェクト | 1,197 | 1,192 |
| そのうち多重ポインタ（深さ 2 以上） | 253 | 252 |

削除したものは `ratv`、`NfundamentalNum`、`fundamentalNum`、`Version_OpenMX`、`NormK` です。
第2段階では 1,193 宣言を維持し、第3段階で `NormK` を除去しました。
新しい大域変数は追加していません。PAO の所有権を移した 6 つの借用ポインタは
残存宣言として数えており、所有権の移行だけで大域変数が消えたとは扱いません。
`ratv` は格納先を移すのではなく配列そのものを除去し、元の番号順序を計算で再現します。
LPQ 出力の配列寸法・番号順序は維持し、以前未初期化だった領域には `-1` を書きます。
FFT 格子検証の移行では、元の再帰関数が戻り値を無視して `22` などを受理していた問題と、
ゼロ入力の無限再帰も修正しました。球 Bessel 関数は原点での `sb[lmax+1]` を明示的に初期化します。

```sh
python3 tools/audit_common_globals.py \
  --baseline docs/core-migration-baseline.json --check-no-new \
  --output build/globals-current.json -- \
  -I /path/to/cuda/include -I /path/to/mpi/include
```

HIP では対象ビルドの互換ヘッダー・定義も渡してください。
これは共通ヘッダーの宣言数であり、アプリケーション全体の大域変数の総数ではありません。
各 `.c` のファイルスコープ変数、関数内 `static`、transport ヘッダー等は別途棚卸しが必要です。
ポインタ深さは型表記上の `*` の数であり、typedef 内部や別名の所有関係までは表しません。

## テスト

CTest に 18 件の独立テストを登録しています。所有権と例外時の整合性、周期セルの番号順序・
整数境界、FFT 格子、球 Bessel 関数の値・導関数、動径積分の独立参照との比較、
32 点バッチ境界、異なる形状の連続実行、C ABI の失敗時の出力保護、
Kokkos ランタイムの所有・借用・破棄、GPU 構成での CPU 指定を検証します。
動径表には旧実装から固定した境界値、非一様格子の二次関数、解析解を持つ動径積分、
平方根格子、スピン・projector の対応、コピー後の独立性、出力範囲と部分更新の防止を追加しました。
PAO ストレージには連続配置、ゼロ初期化、独立コンテキスト、異なる形状での再確保、
途中確保失敗、二重解放の防止、終了後の利用拒否、所有ストレージからの変換を追加しました。
波数補間は旧実装の固定値、解析的な二次関数、1 点・2 点・3 点、微小刻み幅、
不正入力時に C 出力を変更しないことを検証します。
スカラー借用 API は形状と座標を定数時間で検証し、格子の単調性と入力値の有限性は呼び出し側の
前提とします。所有する `RadialTable` の構築時には格子・全行を検証します。

```sh
cmake -S . -B build/core-sanitize -DOPENMX_BUILD_APPLICATION=OFF \
  -DOPENMX_CORE_BACKEND=SERIAL -DOPENMX_CORE_SANITIZERS=ON \
  -DCMAKE_CXX_STANDARD=23 -DCMAKE_BUILD_TYPE=Debug
cmake --build build/core-sanitize --parallel 4
ctest --test-dir build/core-sanitize --output-on-failure
```

`.github/workflows/core.yml` は、ホスト Serial / OpenMP / ASan+UBSan の CI を定義します。
GPU の CTest は実機が必要で、ホスト CI の合格は HIP/CUDA 実行の代わりになりません。
既存の `run_integral_threading_cpu.py` は現在、実物の C アダプターと構築済みコアを使います。
`OPENMX_CORE_LIBRARY=/absolute/path/to/libopenmx_core.so` で共有ライブラリを指定できます。

### 第1段階の検証記録（2026-10-18、170c51d3）

CMake 3.28.3、GCC 14.2、Clang/libc++ 21、CUDA 13.4、NVIDIA HPC SDK 26.9、
GeForce RTX 5080 で検証しました。以下は正しさの検証で、性能測定結果ではありません。

| 構成・試験 | 結果 |
| --- | --- |
| C++20 / Serial / Kokkos mdspan | コア 10 件合格 |
| C++20 / OpenMP / Kokkos mdspan | コア 10 件合格（2 スレッド） |
| C++23 / Serial / ASan+UBSan | コア 10 件合格 |
| Clang/libc++ 21 / C++23 / `std::mdspan` | コア 10 件合格。自動選択と `mdspan=std` の出力を確認 |
| CUDA / Kokkos mdspan | コア 10 件合格。うちホスト用 6 件は GPU を不可視にして実行 |
| CUDA Compute Sanitizer / 動径変換 | エラー 0、リーク 0 |
| バックエンド分離・HIP ホスト試験 | CTest 9 件合格（AMD デバイス実行を含まない） |
| 既存の積分スレッド試験 | CUDA 用・HIP 用のホスト試験が合格 |
| 直接 make でのコア構築 | Serial / OpenMP の切り替え、変更時の再リンク、無変更時の再リンク省略を確認 |
| OpenMX CPU 経路 / CUDA 経路 | `-runtest` 各 14 件合格、4 MPI × 1 スレッド、同一ジョブ内の連続実行 |

本体は `OPENMX_CORE_BACKEND=CUDA` でビルドし、CPU 検証では `OPENMX_GPU=0` を指定しました。
変更前の同一バイナリ SHA-256 を持つ検証記録と比較し、原子の対応・座標・積分グリッドを
確認した上で、最終エネルギーと全原子の力の全成分を比較しました。

| 比較 | 最大絶対エネルギー差 (Ha) | 最大絶対力成分差 (Ha/bohr) |
| --- | ---: | ---: |
| 変更前 → 変更後 CPU | 1.13e-10 | 1.20e-11 |
| 変更前 → 変更後 CUDA | 1.40e-11 | 3.00e-12 |
| 変更後 CPU → CUDA | 4.91e-11 | 2.70e-11 |
| 同梱参照値 → 変更後 CPU | 7.80e-11 | 7.80e-11 |
| 同梱参照値 → 変更後 CUDA | 4.90e-11 | 7.90e-11 |

許容値はそれぞれ `1e-7` です。Kokkos Tools のイベントも記録し、PAO の Bessel・
積分カーネルが CUDA 実行空間で起動し、`Cuda` メモリ領域を確保していることを確認しました。
単に既存 GPU ソルバーが動くことだけを合格条件にはしていません。

作業機のログは `build/core-migration/` にあります（Git 管理外）。
`cpu-S-final/`、`gpu-S-final/` が入力・出力・実行条件、`validation.json` が比較結果、
`binary-fingerprints.json` が本体と共有ライブラリの SHA-256 です。
変更前のバイナリは `c4110d756f45bd5adf27b47a04e84893b848cc1aafb81f75afaeb75476a56d96` です。

```sh
python3 tools/run_gpu_suites.py --binary build/cuda/openmx \
  --output build/check-cpu --suite S --ranks 4 --threads 1 --together --env OPENMX_GPU=0
python3 tools/run_gpu_suites.py --binary build/cuda/openmx \
  --output build/check-gpu --suite S --ranks 4 --threads 1 --together
python3 tools/compare_gpu_suites.py --reference build/check-cpu
python3 tools/compare_gpu_suites.py --reference build/check-gpu
python3 tools/compare_gpu_suites.py build/check-cpu build/check-gpu
```

本体と同じ MPI の `mpirun` を使ってください。
AMD/ROCm はこの作業機にないため、HIP のデバイスコンパイルと実機実行は未検証です。
新コア導入後の L/L2/L3 全例、restart、NEB、マルチ GPU・マルチノードの全体回帰も
この検証には含みません。CI 定義は追加しましたが、リモートの CI 実行結果はまだありません。

### 第2段階の検証記録（2026-10-18、動径表と PAO/NLP/VNA）

比較元は第1段階の `170c51d3` です。同じコンパイラ・実機で次を確認しました。

| 構成・試験 | 結果 |
| --- | --- |
| C++20 / Serial、C++20 / OpenMP | コア各 14 件合格（OpenMP は 2 スレッド） |
| C++23 / ASan+UBSan、Clang/libc++ 21 / `std::mdspan` | コア各 14 件合格 |
| CUDA | コア 14 件とバックエンド分離が合格。ホスト用 8 件は GPU 不可視でも合格 |
| CUDA Compute Sanitizer / projector・VNA 変換 | エラー 0、リーク 0 |
| HIP のホスト・分離試験、既存の積分スレッド試験 | CTest 9 件と CUDA/HIP 用のホスト積分試験が合格 |
| ネイティブ make の依存関係 | C ABI ヘッダー変更で 6 つの動径アダプターを再コンパイル対象にすることを確認 |
| OpenMX CPU 経路 / CUDA 経路 | `-runtest` 各 14 件合格、4 MPI × 1 スレッド、同一ジョブ内で連続実行 |

最終ビルドで、全原子の力・エネルギーを比較しました。許容値はどちらも `1e-7` です。

| 比較 | 最大絶対エネルギー差 (Ha) | 最大絶対力成分差 (Ha/bohr) |
| --- | ---: | ---: |
| 第1段階 → 第2段階 CPU | 4.50e-11 | 5.00e-12 |
| 第1段階 → 第2段階 CUDA | 2.00e-12 | 2.00e-12 |
| 第2段階 CPU → CUDA | 4.91e-11 | 2.60e-11 |
| 同梱参照値 → 第2段階 CPU | 3.30e-11 | 7.80e-11 |
| 同梱参照値 → 第2段階 CUDA | 4.80e-11 | 7.90e-11 |

restart に保存した変換表も、各経路につき PAO 70,200 値、NLP 241,200 値、VNA 7,800 値を
第1段階と比較しました。最大絶対差は、それぞれ `1.78e-15`、`1.06e-15`、`1.78e-14` です。
PAO/NLP の既存ファイルには行末に未使用のパディングがあるため、比較は計算対象の波数のみです。
ファイル長・行の並びも一致しています。restart からの再実行は今回の検証には含めません。

Kokkos Tools は CUDA 実行空間の Bessel / 積分イベントを各 1,768 回記録しました。
波数 900 点の PAO/NLP 用デバイス割り当てが計 52 回、300 点の VNA 用が 26 回あり、
今回追加した変換も GPU 実行経路に入っていることを確認しています。

ログは `build/core-radial-migration/`（Git 管理外）の `cpu-S-final/`、`gpu-S-final/`、
`trace-final/`、`validation.json` にあります。最終バイナリの SHA-256 は次の通りです。

```text
openmx:            462f60c53d13ad959c89d222513639e72060336e94c0cd31740619a1ff8238e7
libopenmx_core.so:  525a52fadb8085a0a59225c23615fede08d4ccd7bc49a455ea95d16e9403e2c3
```

AMD/ROCm のデバイス検証、L/L2/L3、restart 読み込み、NEB、マルチ GPU・マルチノードの
全体回帰は引き続き未実施です。ここに記録した値は正しさの検証で、ベンチマークではありません。

### 第3段階の検証記録（2026-10-18、PAO の所有権と波数補間）

比較元は第2段階の `6c681182` です。同じコンパイラ・実機で次を確認しました。

| 構成・試験 | 結果 |
| --- | --- |
| C++20 / Serial、C++20 / OpenMP | コア各 18 件合格（OpenMP は 2 スレッド） |
| C++23 / ASan+UBSan、Clang/libc++ 21 / `std::mdspan` | コア各 18 件合格 |
| CUDA | コア 18 件とバックエンド分離が合格。ホスト用 12 件は GPU 不可視でも合格 |
| HIP のホスト試験 | 8 件合格。改修済み HIP 固有ファイルの登録を更新し、分離検査も合格 |
| 既存の積分スレッド試験 | CUDA 用・HIP 用のホスト試験が合格 |
| 大域宣言の監査 | 共通ヘッダー 1,192 個、移行開始点から 5 個削除、追加なし |
| OpenMX CPU 経路 / CUDA 経路 | `-runtest` 各 14 件合格、4 MPI × 1 スレッド、同一ジョブ内で連続実行 |
| 前段階の restart の読み込み | Methane を CPU/CUDA の通常実行経路で検証。エネルギーと全力成分が旧バイナリの restart 実行と出力精度で一致 |

全 14 ケースで原子・座標・積分格子を照合し、最終エネルギーと全力成分を比較しました。
許容値はそれぞれ `1e-7` です。

| 比較 | 最大絶対エネルギー差 (Ha) | 最大絶対力成分差 (Ha/bohr) |
| --- | ---: | ---: |
| 第2段階 → 第3段階 CPU | 8.00e-12 | 3.00e-12 |
| 第2段階 → 第3段階 CUDA | 1.10e-10 | 2.60e-11 |
| 第3段階 CPU → CUDA | 8.80e-11 | 1.10e-11 |
| 同梱参照値 → 第3段階 CPU | 2.50e-11 | 7.90e-11 |
| 同梱参照値 → 第3段階 CUDA | 6.30e-11 | 7.80e-11 |

保存された変換表の計算対象値は、各経路の PAO 70,200 値、NLP 241,200 値、
VNA 7,800 値すべてで前段階と一致しました。新たに保存する PAO 表は未使用の行末も
ゼロであることを確認しました。Kokkos Tools では Bessel / 積分の CUDA イベントを
各 1,768 回記録し、所有権移行後も共通カーネルを使うことを確認しました。

restart 検証では前段階の Methane ファイル一式を別ディレクトリへコピーし、
旧・新それぞれの本体と対応するコア共有ライブラリを指定して実行しました。
両経路で 5 種類の Fourier 表は再書き込みされず、動径変換カーネルも起動していません。
この検証は PAO/NLP/VNA の保存表を再計算せずに読み戻す経路を含みます。

ログと入力・出力は `build/core-pao-migration/`（Git 管理外）にあります。
`validation.json` に全スイートと変換表の比較、`restart-cpu.json` / `restart-gpu.json` に
restart の比較、`globals.json` に大域宣言の監査を保存しました。

```text
openmx:            422db5e9bbed36bc32b19ae710ed1225f9a7d0e568823d40db1ff9090ee6aef3
libopenmx_core.so:  5b4f72073e7f060d55264bce31f1ec8b06f72f1782efb08bc3862e1885513d3f
```

AMD/ROCm のデバイスビルド・実行、L/L2/L3 全例、NEB、マルチ GPU・マルチノード、
旧 `openmx_api` の全体回帰は今回の検証には含みません。restart は Methane のみで、
異なる MPI 分割や非共線スピン間の restart 互換性を網羅するものではありません。
ここに記録した値は正しさの検証で、性能ベンチマークではありません。

## 次の移行単位と完了条件

1. 残る VPS/NLP/VNA と化学種メタデータを C++ の所有値にする。PAO を含む参照側に
   明示的なビューを渡し、借用状態で残る `Spe_*` 大域ポインタと null コンテキストの
   手動確保経路を段階的に除去する。補間・PAO/NLP/VNA の入力コピーを所有値への参照に変える。
   `FT_ProExpn_VNA` / `FT_ProductPAO`、導関数、入力解析、MPI 配布、restart の制御も移す。
2. 周期セル、原子・近接リスト、MPI による原子分配を独立したオブジェクトにする。
   一時的に旧配列を参照する場合も寿命を明示し、隠れた既定コンテキストを置かない。
3. 可変長の原子・軌道ブロックを「オフセット + 連続値」にし、ブロックごとに `mdspan` を渡す。
   密行列には `layout_left` / `layout_right` を明示し、BLAS の leading dimension を保持する。
4. Hamiltonian、重なり、密度、混合、力の順に、CPU 参照実装と独立した数値テストを用意して
   Kokkos 実装へ移す。既存の CUDA/HIP 実装は各段階の同等性確認後に削除する。
5. SCF / MD / NEB の状態と MPI communicator・BLAS/solver ハンドル・GPU バッファを
   明示的な所有オブジェクトへ集約する。ランタイムの寿命と計算ケースの寿命を分離する。
6. 入出力、restart、輸送、解析機能まで引数で依存を受け渡し、`openmx_common.h` の可変
   大域宣言と各翻訳単位の可変 `static` を除去する。旧 C ABI が不要になった部分を閉じる。

完了判定には、アプリケーションの可変大域状態ゼロ、同一プロセス内の独立した計算状態、
例外・途中終了を含む解放保証、CPU/CUDA/HIP の数値同等性、全標準スイートと restart / NEB /
MPI 分割の検証が必要です。第三者ライブラリが内部に持つ状態はアプリケーションの
大域変数とは区別し、初期化・終了の境界を管理します。
