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
| VPS/NLP/VNA の計算状態 | `core/src/vps_storage.cpp` | VPS 格子・ポテンシャル・projector と Fourier 表を所有。VNA 展開の有効時だけ追加の 4 配列を所有し、すべて `mdspan` で参照する。 |
| 可変長の軌道積表 | `core/src/product_storage.cpp` | 化学種と軌道対のオフセットを持ち、全数値を 1 つの連続領域に所有。各軌道対は `[2*l2+1, 波数容量]` の `mdspan` で参照する。 |
| VNA 分離射影子の生成 | `core/src/neutral_projectors.cpp` | PAO・VPS 格子・中性原子ポテンシャル・積分節点を引数で受け取り、初期基底、正規化、VNA を計量とする直交化と再正規化を行う。作業領域・射影子・エネルギーは連続配置の `Tensor` が所有する。 |
| 球 Bessel 関数 | `core/include/openmx/core/bessel.hpp` | 入出力と角運動量を引数で受ける純粋なカーネル。CPU/CUDA/HIP で同じ実装。 |
| 原子対の Bessel 表 | `core/src/pair_bessel.cpp` | 波数格子と角運動量の容量を明示した `PairBesselWorkspace`。値・導関数を連続配置の `Tensor` と `mdspan` で管理し、距離・角運動量が一致した表はコピーせず借用する。 |
| 重なり・運動エネルギーの作業領域 | `core/src/overlap_workspace.cpp` | 波数格子、RF 表、原子対の Bessel キャッシュ、動径積分、複素行列と変換用の行列を OpenMP ワーカーごとに所有。磁気量子数を含む軌道番号で連続配置し、C++ では `mdspan`、C 境界では平坦なビューを借用する。 |
| 非局所積分の作業領域と射影子メタデータ | `core/src/nonlocal_workspace.cpp` | 化学種ごとの射影子数・角運動量・磁気成分のオフセットをコピーして所有。角度組立ての行列は `[channel, orbital, projector component]` の `mdspan`。CPU 用の動径表・積分・Bessel キャッシュは CPU 作業領域だけが所有する。 |
| 動径表と補間 | `core/src/radial_table.cpp` | `RadialTableView` が格子と全入力値を検証して借用する。独立したコピーが必要な場合は `RadialTable` が所有。PAO、非局所 projector、中性原子ポテンシャル、VNA 分離射影子の境界条件を明示して補間する。 |
| 波数空間の補間 | `core/src/radial_table.cpp` | 波数を `i*step` で求め、PAO/projector の共通補間へ渡す。格子配列を所有せず、`NormK` を必要としない。 |
| PAO/NLP/VNA・軌道積の Fourier 変換 | `core/src/radial.cpp` | 半径、積分の測度・方式、波数、チャネル、サンプルを明示的に受け取る。ホスト配列は `Tensor`、実行用配列は `Kokkos::View` が所有。 |
| FFT グリッド検証 | `core/include/openmx/core/fft_grid.hpp` | 正の整数を 2、3、5、7 の積に分解できるか判定。大域の可変テーブルを持たない。 |
| 周期セル逆引き | `source/openmx_periodic.h` | セルの半径と座標から添字を直接計算。逆引き用の所有配列を持たない。 |
| 既存 C との境界 | `source/openmx_core*.h`、`core/src/c_api.cpp`、`core/src/{pao,vps,product}_compat.hpp` | 不透明なコンテキストと C ABI。入力表、出力先、実行デバイス、エラー出力先を引数で受け取る。固定寸法の互換ポインタ表は共通の `legacy_rows.hpp`、可変長の軌道積表は `product_compat.hpp` で構築する。 |

`source/FT_PAO.c`、`FT_NLP.c`、`FT_VNA.c`、`FT_ProductPAO.c`、`FT_ProExpn_VNA.c` は入出力・MPI 通信と新ライブラリの接続を担います。
従来の `double*** TmpRF` / `TmpNRF`、`double** SphB`、VNA の一時配列と
その手動確保・解放を除去しました。変換中のサンプリングに大域配列を読むコールバックは使いません。
`RadialF.c`、`Nonlocal_RadialF.c`、`VNAF.c`、`PhiF.c` の他の利用箇所も共通 C++ 補間処理を呼びます。
スカラーの互換呼び出しは既存の行を借用し、呼ぶたびに表全体をコピーしません。
`Spherical_Bessel.c` も共通 C++ カーネルを呼ぶ薄い C アダプターです。

通常計算・標準スイート・NEB では、`Spe_PAO_XV`、`Spe_PAO_RV`、`Spe_Atomic_Den`、
`Spe_Atomic_Den2`、`Spe_PAO_RWF`、`Spe_RF_Bessel` の数値領域を `PaoStorage` が所有します。
読み込み前に確保し、入力ファイルの読み込み、変換、MPI 配布、restart の読み書き、SCF 中の
参照まで同じ領域を使います。各配列は 1 つの連続領域で、未使用の容量と行末もゼロ初期化します。
既存の `Spe_*` 大域ポインタは借用ビューとして残り、宣言の除去は参照側の移行時に行います。
互換ポインタ表には RAII の `std::vector` を使い、数値データの所有と分離します。

同じコンテキストが `VpsStorage` を所有し、`Spe_VPS_XV`、`Spe_VPS_RV`、`Spe_Vna`、
`Spe_VH_Atom`、`Spe_Atomic_PCC`、`Spe_VNL`、`Spe_VNLE`、`Spe_VPS_List`、
`Spe_NLRF_Bessel` の 9 配列も連続領域へ移しました。
`scf.ProExpn.VNA` が有効な場合は `NeutralProjectorStorage` が `Projector_VNA`、
`VNA_proj_ene`、`Spe_VNA_Bessel`、`Spe_CrudeVNA_Bessel` の 4 配列を追加で所有します。
無効時はこの 4 つの借用ポインタを null にし、確保しません。スピン成分数と容量は
計算ケースごとに決め、NLP/VNA の未使用領域もゼロ初期化します。

PAO/NLP/VNA の変換 API は `RadialTableView` で所有領域を借用し、入力表全体のコピーを
除去しました。格子・全行を検証してから、積分用のサンプルだけを新しく所有します。
借用する格子・チャネル記述・数値行はサンプリング中に変更せず、ビューより長く生存する
必要があります。変換 API は呼び出し後に入力の参照を保持しません。
すべてのサンプリングと計算が終わってから結果を書き込むため、入力と出力の重なりも扱えます。

`Spe_ProductRF_Bessel` の数値領域も `ProductStorage` へ移しました。
`ProductLayout` は各化学種の基底数をコピーし、`l1 <= l2` の軌道対だけを保持します。
restart と同じ `(l1,m1,l2,m2,l,k)` の順序で連続配置し、行末 2 点を含めてゼロ初期化します。
空の角運動量グループと `l1 > l2` の未使用ブロックは確保せず、互換表では null にします。
7 重ポインタの大域宣言は借用表として残りますが、行ごとの数値領域の確保・解放はしません。
この表の寿命は PAO/VPS 所有ストレージと独立し、基底メタデータを C 側で解放しても破棄できます。

化学種メタデータ、PAO/VPS ファイルの解析、原子ポテンシャルの構築、MPI 配布、
restart の制御は C 側に残ります。
多重ポインタは移行用の境界で受け渡し、数値モジュールには渡しません。

`SetPara_DFT.c` 内の VNA 分離射影子の数値処理は `expand_neutral_projectors` に移しました。
全 PAO 基底を借用する `RadialTableView` と VPS 格子・VNA・展開寸法を明示的に受け取り、
初期基底の 4 分岐、正規化、古典 Gram–Schmidt の加算順序を維持します。
旧実装の再正規化（動径ノルムを掛け、射影子エネルギーをその二乗で補正）もそのままです。
この依存のある生成処理はホストで実行し、Kokkos の初期化を必要としません。
`phi`、`phi2`、`pe` の手動確保を除去し、計算結果全体も C++ が所有してから C 側へ反映します。
不正な入力・途中のオーバーフロー・確保失敗では、射影子とエネルギーをどちらも変更しません。

`RF_BesselF` と CUDA/HIP 両方の `NLRF_BesselF` は同じ C++ 補間を呼びます。
最初の区間での外挿、上端の仮想点、projector の微小刻み幅での線形処理を維持します。
1 点・2 点の表も明示的に処理し、元の PAO 実装の無限ループ・範囲外参照を避けます。
`NormK` の宣言・確保・生成・解放をすべて除去し、restart 時も格子の再生成を必要としません。
VNA 用の `GL_NormK` も大域宣言を除去しました。CUDA/HIP 両方の `Set_ProExpn_VNA`
呼び出し内で波数を自動配列に生成し、CPU 処理と同期 GPU バッチへ引数で渡します。
Fourier 変換や restart の読み込みが大域格子を初期化するという依存はなくなりました。

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

軌道積は半径と波数の双方に Gauss–Legendre の節点を使います。台形則の端点係数を
適用せず、`r*r*weight`、Bessel 値、左軌道、右軌道の順に掛け、最後に区間幅の半分を掛けます。
PAO のサンプルは各軌道 1 行だけを所有し、軌道対のチャネルは左右の行番号と角運動量を持ちます。
積を先に別表へ展開せず、通常の動径変換と同じホスト処理・Kokkos カーネルを使います。
旧 `FT_ProductPAO` のスレッドごとの `GL_PAO`、`SphB` と手動確保・解放は除去しました。

VNA 分離射影子の Fourier 変換も同じ Gauss 積分カーネルを使い、旧 `FT_ProExpn_VNA` の
`TmpPhiF`、`SumTmp`、Bessel 作業領域と手動確保・解放を除去しました。
補間には `RadialKind::neutral_projector` を追加し、`PhiF` の原点側の二次延長、
最初の区間の外挿、表の外側でのゼロを維持します。VNA ポテンシャル用の cutoff は適用しません。
上端の仮想点を選んでから値を読むため、旧 `PhiF` の範囲外読み取りは起こりません。
原点側のステンシルに必要な最小 6 点も検証します。旧スカラー ABI `PhiF` は共通処理への
薄い接続として残し、不正な形状・座標では NaN を返します。

## 実行状態と RAII

```text
main
  └─ openmx_core_run: スタック上の Context を所有
       └─ 通常計算 / Runtest / NEB
            ├─ readfile_with_context → SetPara_DFT → ReadPara_DFT
            │    └─ Allocate_Arrays_with_context → PAO / VPS / 任意の VNA・軌道積ストレージ → 読み込み
            ├─ FT_PAO / FT_NLP / FT_VNA / FT_ProductPAO / FT_ProExpn_VNA → C++ 動径変換 → 結果の格納
            └─ Free_Arrays_with_context → GPU キャッシュ破棄 → 計算ストレージ解放
       └─ openmx_core_finish → MPI_Finalize → Context の破棄
```

コンテキストのプロセス大域ポインタやシングルトンは置きません。
`-runtest` のケース間と NEB の画像間では同じコンテキストを渡し、Kokkos の
初期化・終了を繰り返しません。PAO、VPS/VNA、軌道積の所有ストレージは計算ケースごとに解放し、
同じコンテキストで次の形状を確保できます。確保済みのビューを無効にする上書き確保は拒否します。
確保途中の例外では完成した配列を自動解放し、C 側には部分的なビューを公開しません。
PAO、VPS、軌道積の解放は独立しており、1 つの解放は他のビューや Kokkos の寿命に影響しません。
コンテキストの破棄と `openmx_core_finish` でも残った所有配列を解放します。
変換用の一時配列は各変換のスコープで解放し、非同期処理は
RAII の fence により解放前に完了させます。コピー途中の例外でも `Tensor` の形状と
ストレージの整合性を維持します。

GPU 対応の Kokkos は初期化時に GPU を使うため、GPU 無効時の変換は GPU の初期化を
必要としないホスト実装を呼びます。既存コードが後の入力でランクと GPU の対応を
変えた場合も、異なるデバイスを前提とする実行空間の再利用を避け、ホストで計算します。
別の所有者が Kokkos を初期化している場合も、そのランタイムを終了せずホストで計算します。

旧 `readfile` / `run_main` / `openmx_api` の呼び出しは、null コンテキストで
ホスト実装を使えます。この互換経路の PAO/VPS/VNA・軌道積の確保・解放は従来の C 実装を維持し、
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

### 古い NVIDIA コンパイラとの組み合わせ

OpenMX 本体の C / Fortran と C++ コアは別々のコンパイラで構築し、C ABI で接続します。
本体の `nvc` / `nvfortran` が古い場合でも、`SERIAL` / `OPENMP` コアには別途用意した
GCC / Clang と対応する標準ライブラリを使えます。CMake では
`CMAKE_CXX_COMPILER`、直接 make では `OPENMX_CORE_CXX` を指定します。
コアを `SERIAL` にしても、本体の既存 CUDA ソルバーは引き続き利用できます。

ただし、C++20 未対応のコンパイラだけでこのブランチをビルドする経路はありません。
`std::span` と Kokkos 5.2.2 が C++20 を必要とし、C++17 への自動切り替えは行いません。
`std::mdspan` がない場合の Kokkos/mdspan への切り替えも、C++20 を最低条件とします。
固定した Kokkos 5.2.2 の構成チェックでは GCC 10.4 以上、CUDA コアには nvcc 12.2 以上が必要です。
[Kokkos の公式要件](https://kokkos.org/kokkos-core-wiki/get-started/requirements.html) と
各 CUDA Toolkit が対応するホストコンパイラの範囲も確認してください。

既存 CUDA アダプター用の `NVCC_HOST` と、コア用の `OPENMX_CORE_CXX` は別設定です。
`Makefile.kugui` は両方にサイトの GCC 12.2 を指定し、COSMA 用の C / C++ コンパイラと
GCC ランタイムのリンク・実行時パスも揃えます。通常の Makefile で同様に指定する場合は、例えば次を使います。

```sh
make -C source -j4 openmx OPENMX_CORE_BACKEND=SERIAL \
  OPENMX_CORE_CXX=/path/to/gcc/bin/g++ NVCC_HOST=/path/to/gcc/bin/g++ \
  GPUSOLVER2_HOST_CC=/path/to/gcc/bin/gcc GPUSOLVER2_HOST_CXX=/path/to/gcc/bin/g++
```

非標準の GCC 配置では、`Makefile.kugui` のように対応する `libstdc++` の検索パスも設定します。
CUDA コアを有効にする場合は、さらに `CMAKE_CUDA_COMPILER` と
`CMAKE_CUDA_HOST_COMPILER` で対応する nvcc / ホスト C++ コンパイラを指定します。
`CMAKE_CUDA_HOST_COMPILER` はネイティブ Makefile の `NVCC_HOST` を変更しません。
CUDA 11 系など古い Toolkit 全体への後方互換は保証しません。既存 CUDA アダプターにも
`-std=c++20` を使う箇所があるため、コアをホストにするだけで全体の要件が下がるわけではありません。

2026-10-19 に CMake 4.4.4 / GCC 12.4 でホストコアの全 33 テスト、
NVHPC 24.1 に同梱された nvcc 12.3.103 と同じ GCC で CUDA コアの全 33 テストが合格しました。
CUDA は `sm_80` / PTX を指定してビルドし、実行機は RTX 5080 です。
NVHPC 23.3 の C11 コードから GCC 12 の共有コアを呼ぶ試験では、
VNA 射影子の Fourier 変換を解析値と照合し、コンテキストの生成・解放も確認しました。
GCC 9 / nvcc 12.0 は Kokkos の最低版検査で構成時に拒否されることを確認しています。
ログと C ABI の試験コードは `build/compiler-compatibility/`（Git 管理外）にあります。
これはコアと C ABI の検証であり、古い SDK を使った OpenMX 全体や Kugui 実機での再検証は含みません。

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
PAO/NLP/VNA・軌道積の変換も Kokkos/CUDA で動かすには、例えば RTX 5080 で次を指定します。

```sh
cmake -S . -B build/cuda -DOPENMX_GPU_BACKEND=CUDA \
  -DOPENMX_GPU_ARCH=sm_120 -DOPENMX_BUILD_JOBS=4 \
  -DOPENMX_CORE_BACKEND=CUDA \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc
cmake --build build/cuda --parallel 4
ctest --test-dir build/cuda --output-on-failure
```

GPU コアの既定規格は C++20 です。CMake の CUDA 言語を使い、同一の `.cpp` を
CUDA コンパイラで構築します。`OPENMX_GPU_ARCH` は本体と Kokkos の CUDA
アーキテクチャを同時に設定するので、構成時に GPU がなくても指定できます。
未指定または `auto` の場合は `nvidia-smi` による自動認識を行い、コマンドが
存在しない場合や GPU の compute capability を取得できない場合は構成をエラー終了します。
実行時は既存の GPU 利用可否・ランク割り当てに従います。
`OPENMX_GPU=0` や `scf.eigen.lib elpa2` ではコアも CPU 経路を使います。

### OpenMX 本体と HIP

以下は MI300X (`gfx942`) 用の構成例です。ROCm / MPI / AOCL の設定は
[既存のビルド手順](GPU_BACKENDS.md#amd--hip) と合わせて指定してください。
Radeon 実機での検証条件と結果は、後述の「AMD 実機検証（2026-10-19）」を参照してください。

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

Radeon RX 9060 XT (`gfx1200`) では、本体の `OPENMX_GPU_ARCH` と
`CMAKE_HIP_ARCHITECTURES` を `gfx1200`、`Kokkos_ARCH_AMD_GFX1200=ON` にします。
固定している Kokkos 5.2.2 はこのアーキテクチャの登録を欠くため、
`cmake/OpenMXKokkosGfx1200.cmake` が FetchContent の展開先に登録と
32-lane wave の特性を追加します。配布アーカイブのバージョンと SHA-256 は保持します。
パッチは対象バージョン・置換箇所を全て確認してから書き込み、再適用では変更しません。

`FETCHCONTENT_SOURCE_DIR_KOKKOS` で外部の展開済みソースを指定した場合は
FetchContent のパッチ工程を通らないため、`gfx1200` を使う前にそのコピーへ適用します。

```sh
cmake -DKOKKOS_SOURCE_DIR=/path/to/kokkos-5.2.2 \
  -P cmake/OpenMXKokkosGfx1200.cmake
```

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
既定で `work/openmx` と `work/lib/libopenmx_core.so` を配置します。prefix を
明示指定した場合は `<prefix>/bin/openmx` と `<prefix>/lib/libopenmx_core.so` です。
実行ファイルはビルド時の場所、`$ORIGIN/lib`、`$ORIGIN/../lib` を探索します。
従来の MPI・数値計算ライブラリへの依存も残ります。

## 大域変数の棚卸し

`tools/audit_common_globals.py` は Clang の C 構文木から、`openmx_common.h` 自体が
宣言・定義するファイルスコープのオブジェクトを数えます。SDK や MPI ヘッダーの
変数を除外し、同じヘッダーのベースラインを `docs/core-migration-baseline.json` に保存しています。

| 対象 | 開始時 | この段階 |
| --- | ---: | ---: |
| 共通ヘッダーのオブジェクト | 1,197 | 1,191 |
| そのうち多重ポインタ（深さ 2 以上） | 253 | 252 |

削除したものは `ratv`、`NfundamentalNum`、`fundamentalNum`、`Version_OpenMX`、`NormK`、`GL_NormK` です。
第2段階では 1,193 宣言を維持し、第3段階で `NormK` を除去しました。
第4段階の VPS/NLP/VNA 所有権移行では宣言数は変わらず、新しい大域変数も追加していません。
第5段階で `GL_NormK` を除去しました。PAO の 6 配列、VPS/NLP/VNA の 13 配列、
軌道積表の借用ポインタは残存宣言として数えており、
所有権の移行だけで大域変数が消えたとは扱いません。
第6段階の VNA 射影子生成・変換の移行では宣言数を維持し、新しい大域変数は追加していません。
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

CTest に 47 件の独立コアテストを登録しています。所有権と例外時の整合性、周期セルの番号順序・
整数境界、FFT 格子、球 Bessel 関数の値・導関数、動径積分の独立参照との比較、
32 点バッチ境界、異なる形状の連続実行、C ABI の失敗時の出力保護、
Kokkos ランタイムの所有・借用・破棄、GPU 構成での CPU 指定を検証します。
動径表には旧実装から固定した境界値、非一様格子の二次関数、解析解を持つ動径積分、
平方根格子、スピン・projector の対応、コピー後の独立性、出力範囲と部分更新の防止を追加しました。
PAO ストレージには連続配置、ゼロ初期化、独立コンテキスト、異なる形状での再確保、
途中確保失敗、二重解放の防止、終了後の利用拒否、所有ストレージからの変換を追加しました。
VPS/VNA ストレージでは 1/2 スピン成分と VNA 展開の有効／無効、PAO との独立した解放、
追加配列の確保失敗時の巻き戻し、コンテキストの破棄時の解放、CPU/GPU 変換と未使用領域を
検証します。借用ビューでは所有コピーとの同等性・寿命の違い、不正値の拒否、
入力と出力を共有する C ABI 呼び出しも検証します。
軌道積表は異なる化学種・基底数での連続配置と restart 行順、空のグループ、
未使用の軌道対、寸法のオーバーフロー、独立した寿命と途中確保失敗を検証します。
Gauss 積分は解析的な多項式積分と独立した Bessel 級数を参照にし、
左右の軌道・角運動量の対応、32 点のバッチ境界、出力パディングと失敗時の出力保護を確認します。
VNA 射影子の生成は第5段階の C 実装から取得した固定値と、定数 PAO・定数ポテンシャルの
解析解を参照します。空原子・ゼロポテンシャル・小さいノルム・積分オーバーフロー・
入力と出力の重なり・遅い段階の入力不備での出力保護も検証します。
Fourier 変換は独立した Bessel 級数と解析積分、`PhiF` は旧値と二次関数・6 点の端点を検証します。
波数補間は旧実装の固定値、解析的な二次関数、1 点・2 点・3 点、微小刻み幅、
不正入力時に C 出力を変更しないことを検証します。
スカラー借用 API は形状と座標を定数時間で検証し、格子の単調性と入力値の有限性は呼び出し側の
前提とします。`RadialTable` / `RadialTableView` の構築時には格子・全行を検証します。

原子対の Bessel 表は独立した標準ライブラリの球 Bessel 関数と導関数の恒等式を参照し、
原点、角運動量 0〜29、キャッシュ上限、無効化、角運動量と距離によるキー、
波数格子の所有、借用表のアドレスと寿命、独立ワークスペース、C ABI の失敗時の出力保護を検証します。
既存の積分スレッド試験も新しい C ABI を使い、8 スレッドの独立キャッシュを照合します。

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

### 第4段階の検証記録（2026-10-18、VPS/NLP/VNA の所有権と借用ビュー）

比較元は第3段階の `11e3f998` です。同じコンパイラ・実機で次を確認しました。

| 構成・試験 | 結果 |
| --- | --- |
| C++20 / Serial、C++20 / OpenMP | コア各 23 件合格（OpenMP は 2 スレッド） |
| C++23 / ASan+UBSan、Clang/libc++ 21 / `std::mdspan` | コア各 23 件合格。ASan のリーク検査も有効 |
| CUDA | コア 23 件とバックエンド分離が合格。ホスト用 16 件は GPU 不可視でも合格 |
| HIP のホスト・分離試験 | 9 件合格（AMD デバイス実行を含まない） |
| 既存の積分スレッド試験 | CUDA 用・HIP 用のホスト試験が合格 |
| 大域宣言の監査 | 共通ヘッダー 1,192 個、移行開始点から 5 個削除、追加なし |
| OpenMX CPU 経路 / CUDA 経路 | `-runtest` 各 14 件合格、4 MPI × 1 スレッド、同一ジョブ内で連続実行 |
| 第3段階の restart の読み込み | Methane を CPU/CUDA で検証。エネルギーと全力成分が旧バイナリの restart 実行と出力精度で一致 |
| VNA 展開を無効にした通常計算 | Methane を CPU/CUDA で検証し、旧バイナリとの比較が合格 |

全 14 ケースで原子・座標・積分格子を照合し、最終エネルギーと全力成分を比較しました。
許容値はそれぞれ `1e-7` です。標準入力の Cr2 と GaAs ではスピン軌道結合が有効であり、
VPS ストレージの 1/2 スピン成分を同じジョブ内で切り替える経路も含みます。

| 比較 | 最大絶対エネルギー差 (Ha) | 最大絶対力成分差 (Ha/bohr) |
| --- | ---: | ---: |
| 第3段階 → 第4段階 CPU | 0（出力精度内） | 0（出力精度内） |
| 第3段階 → 第4段階 CUDA | 1.10e-10 | 1.20e-11 |
| 第4段階 CPU → CUDA | 4.91e-11 | 1.10e-11 |
| 同梱参照値 → 第4段階 CPU | 2.50e-11 | 7.90e-11 |
| 同梱参照値 → 第4段階 CUDA | 4.80e-11 | 7.80e-11 |

保存された Fourier 表は各経路・各種類につき 14 ファイルを比較し、計算対象の全値が
前段階と一致しました。対象は PAO 70,200 値、NLP 241,200 値、VNA 7,800 値、
VNA projector 展開 788,400 値、軌道積 185,700 値です。
PAO/NLP の旧ファイルの未使用パディングは数値比較から除外し、新しい両表では
行末 2 点がゼロであることを確認しました。Kokkos Tools の Bessel / 積分 CUDA イベントは
各 1,768 回で、所有権の移行後も共通カーネルで実行しています。

restart 検証では旧・新それぞれの本体と対応するコア共有ライブラリを明示しました。
両経路で 5 種類の Fourier 表は再書き込みされず、動径変換カーネルも起動していません。
VNA 展開無効の通常計算は、CPU で旧出力と一致し、CUDA の最大差はエネルギー
`4.90e-11 Ha`、力成分 `6.00e-12 Ha/bohr` でした。

ログと入力・出力は `build/core-vps-migration/`（Git 管理外）にあります。
`validation.json` にスイート・保存表・CUDA イベントの比較、
`restart-{cpu,gpu}.json` と `vna-off-{cpu,gpu}.json` に追加計算の比較を保存しました。

```text
openmx:            400a0e5296573773c5cbb2c43b29f6620494cf6337a01eee9a2e63935d296130
libopenmx_core.so:  15fa7a587a89bd612a7e962de1f40185da326929dac87870d04e196209322e21
```

AMD/ROCm のデバイスビルド・実行、L/L2/L3 全例、NEB、マルチ GPU・マルチノード、
旧 `openmx_api` の全体回帰は今回の検証には含みません。restart と VNA 展開無効の
追加計算は Methane のみです。ここに記録した値は正しさの検証で、性能ベンチマークではありません。

### 第5段階の検証記録（2026-10-18、可変長の軌道積表と Gauss 積分）

比較元は第4段階の `038b99c5` です。同じコンパイラ・実機で次を確認しました。

| 構成・試験 | 結果 |
| --- | --- |
| C++20 / Serial、C++20 / OpenMP | コア各 28 件合格（OpenMP は 2 スレッド） |
| C++23 / ASan+UBSan、Clang/libc++ 21 / `std::mdspan` | コア各 28 件合格。ASan のリーク検査も有効 |
| CUDA | コア 28 件とバックエンド分離が合格。ホスト用 20 件は GPU 不可視でも合格 |
| CUDA Compute Sanitizer | 軌道積の実行テストでエラー 0、リーク 0 |
| HIP のホスト・分離試験 | 9 件合格（AMD デバイス実行を含まない） |
| 既存の積分スレッド試験 | CUDA 用・HIP 用のホスト試験が合格 |
| 大域宣言の監査 | 共通ヘッダー 1,191 個、移行開始点から 6 個削除、追加なし |
| OpenMX CPU 経路 / CUDA 経路 | `-runtest` 各 14 件合格、4 MPI × 1 スレッド、同一ジョブ内で連続実行 |
| 第4段階の restart の読み込み | Methane を CPU/CUDA で検証。エネルギーと全力成分が旧バイナリの restart 実行と出力精度で一致 |
| VNA 展開を無効にした通常計算 | Methane を CPU/CUDA で検証し、旧バイナリとの比較が合格 |

全 14 ケースで原子・座標・積分格子を照合し、最終エネルギーと全力成分を比較しました。
許容値はそれぞれ `1e-7` です。

| 比較 | 最大絶対エネルギー差 (Ha) | 最大絶対力成分差 (Ha/bohr) |
| --- | ---: | ---: |
| 第4段階 → 第5段階 CPU | 2.00e-12 | 1.00e-12 |
| 第4段階 → 第5段階 CUDA | 4.00e-12 | 2.00e-12 |
| 第5段階 CPU → CUDA | 4.91e-11 | 1.20e-11 |
| 同梱参照値 → 第5段階 CPU | 2.70e-11 | 7.90e-11 |
| 同梱参照値 → 第5段階 CUDA | 5.10e-11 | 7.80e-11 |

保存された Fourier 表は各経路・各種類につき 14 ファイルを比較しました。
今回移行した軌道積 185,700 値の最大絶対差は CPU/CUDA とも `3.06e-15` 未満で、
ファイル長と行の並びも一致しています。PAO 70,200 値、NLP 241,200 値、VNA 7,800 値、
VNA projector 展開 788,400 値は前段階と完全に一致しました。
PAO/NLP の旧ファイルの未使用パディングは数値比較から除外し、新しい両表では
行末 2 点がゼロであることを確認しました。

Kokkos Tools は Bessel / 積分の CUDA イベントを各 2,028 回記録しました。
前段階から増えた各 260 回は、26 化学種分の軌道積変換を波数 32 点ずつに分割したものです。
波数用のデバイス割り当ては 900 点の PAO/NLP が計 52 回、300 点の VNA/軌道積が計 52 回で、
新しい変換も共通の CUDA 実行経路に入っていることを確認しました。

restart 検証では旧・新それぞれの本体と対応するコア共有ライブラリを明示しました。
両経路で 5 種類の Fourier 表は再書き込みされず、動径変換カーネルも起動していません。
VNA 展開無効の通常計算は、CPU で旧出力と一致し、CUDA の最大差はエネルギー
`9.01e-12 Ha` 未満、力成分 `1.01e-12 Ha/bohr` 未満でした。

ログと入力・出力は `build/core-product-migration/`（Git 管理外）にあります。
`validation.json` にスイート・保存表・CUDA イベントの比較、
`restart-{cpu,gpu}.json` と `vna-off-{cpu,gpu}.json` に追加計算の比較を保存しました。

```text
openmx:            552ada19dae0bd192c082a556444a4af371a3e25f5a2d0ef72840c92b9dd7abd
libopenmx_core.so:  8423c2015c79599582bfd480ee0f330e7eef69876dd9e16fed342bd94c5a86ed
```

AMD/ROCm のデバイスビルド・実行、L/L2/L3 全例、NEB、マルチ GPU・マルチノード、
旧 `openmx_api` の全体回帰は今回の検証には含みません。restart と VNA 展開無効の
追加計算は Methane のみです。ここに記録した値は正しさの検証で、性能ベンチマークではありません。

### 第6段階の検証記録（2026-10-19、VNA 分離射影子の生成と変換）

比較元は第5段階の `85d03acc` です。同じコンパイラ・実機で次を確認しました。

| 構成・試験 | 結果 |
| --- | --- |
| C++20 / Serial、C++20 / OpenMP | コア各 33 件合格（OpenMP は 2 スレッド） |
| C++23 / ASan+UBSan、Clang/libc++ 21 / `std::mdspan` | コア各 33 件合格。ASan のリーク検査も有効 |
| CUDA | コア 33 件とバックエンド分離が合格。ホスト用 24 件は GPU 不可視でも合格 |
| CUDA Compute Sanitizer | VNA 射影子の変換テストでエラー 0、リーク 0 |
| HIP のホスト・分離試験 | 9 件合格（AMD デバイス実行を含まない） |
| 既存の積分スレッド試験 | CUDA 用・HIP 用のホスト試験が合格 |
| 大域宣言の監査 | 共通ヘッダー 1,191 個を維持、追加なし |
| OpenMX CPU 経路 / CUDA 経路 | `-runtest` 各 14 件合格、4 MPI × 1 スレッド、同一ジョブ内で連続実行 |
| 第5段階の restart の読み込み | Methane を CPU/CUDA で検証。エネルギーと全力成分が旧バイナリの restart 実行と出力精度で一致 |
| VNA 展開を無効にした通常計算 | Methane を CPU/CUDA で検証し、旧バイナリとの比較が合格 |

全 14 ケースで原子・座標・積分格子を照合し、最終エネルギーと全力成分を比較しました。
許容値はそれぞれ `1e-7` です。

| 比較 | 最大絶対エネルギー差 (Ha) | 最大絶対力成分差 (Ha/bohr) |
| --- | ---: | ---: |
| 第5段階 → 第6段階 CPU | 1.07e-10 | 2.50e-11 |
| 第5段階 → 第6段階 CUDA | 1.40e-11 | 4.00e-12 |
| 第6段階 CPU → CUDA | 1.17e-10 | 2.50e-11 |
| 同梱参照値 → 第6段階 CPU | 8.00e-11 | 7.90e-11 |
| 同梱参照値 → 第6段階 CUDA | 4.80e-11 | 7.80e-11 |

保存された Fourier 表は各経路・各種類につき 14 ファイルを比較しました。
PAO 70,200 値、NLP 241,200 値、VNA 7,800 値、軌道積 185,700 値は前段階と完全に一致しています。
PAO/NLP の旧ファイルの未使用パディングは数値比較から除外し、新しい両表では
行末 2 点がゼロであることを確認しました。

今回移行した VNA 射影子表 788,400 値には、前段階との最大絶対差
`2.2562e-7` 未満があり、表の比較基準 `1e-10` を超えました。
旧 C 処理をコンパイルする NVC は FMA（積和演算の縮約）を使い、既存 C++ コアは
`-ffp-contract=off` / CUDA `--fmad=false` を使うためです。
ほぼ線形従属な基底の直交化では、この丸めの違いが射影子の差として増幅されます。
GaAs と Methane の実入力を取り出し、旧 C 処理の FMA だけを無効にした再計算で
生成値の差が丸め誤差まで縮小することを確認しました。

さらに、第5段階から抽出した生成・直交化・補間・Fourier 積分の C 処理を
NVC `-Mnofma` でビルドし、新しい C API の位置で呼び出す参照ライブラリを作成して
全 14 例を再計算しました。この条件を揃えた旧 C 処理との射影子表の最大絶対差は
CPU `1.39e-16` 未満、CUDA `1.14e-13` 未満で、全値が基準 `1e-10` を満たしました。
前段階の通常バイナリに対する物理量の比較は上表のとおりで、許容値を変更していません。
コアの演算設定と従来の古典 Gram–Schmidt 法も変更していません。

Kokkos Tools は Bessel / 積分の CUDA イベントを各 2,288 回記録しました。
前段階から増えた各 260 回は、26 化学種分の VNA 射影子変換を波数 32 点ずつに分割したものです。
波数用のデバイス割り当ては 900 点の PAO/NLP が計 52 回、300 点の
VNA/軌道積/VNA 射影子が計 78 回でした。旧 C 処理を呼ぶ参照計算では各 2,028 回に戻り、
移行した変換が共通の CUDA 実行経路に入っていることも確認しました。

restart 検証では旧・新それぞれの本体と対応するコア共有ライブラリを明示しました。
両経路で 5 種類の Fourier 表は再書き込みされず、動径変換カーネルも起動していません。
VNA 展開無効の通常計算は、CPU で旧出力と一致し、CUDA の最大差はエネルギー
`2.68e-10 Ha` 未満、力成分 `3.01e-11 Ha/bohr` 未満でした（許容値はともに `1e-7`）。

ログと入力・出力は `build/core-neutral-migration/`（Git 管理外）にあります。
`validation.json` にスイート・保存表・CUDA イベントの比較、
`restart-{cpu,gpu}.json` と `vna-off-{cpu,gpu}.json` に追加計算の比較を保存しました。
演算条件の調査は `replay.json`、参照 C コードと実行結果は
`legacy_reference.c`、`legacy-reference.so`、`no-fma-reference-S/` にあります。

```text
openmx:            497c66d7413431eddffc4eba65d7e679fce579a3bae771090431c6005206b226
libopenmx_core.so:  14ae376ba17ff2a6411eebd58f8b478150e07c753b4747d1717998dcda406468
```

AMD/ROCm のデバイスビルド・実行、L/L2/L3 全例、NEB、マルチ GPU・マルチノード、
旧 `openmx_api` の全体回帰は今回の検証には含みません。restart と VNA 展開無効の
追加計算は Methane のみです。ここに記録した値は正しさの検証で、性能ベンチマークではありません。

## 第7段階：原子対 Bessel 表の所有権と借用

`Set_OLP_Kin` と CUDA/HIP 両方の `Set_Nonlocal` の CPU 経路は、
OpenMP ワーカーごとに `PairBesselWorkspace` を所有します。格子は作成時に検証してコピーし、
生成時には距離・角運動量・漸化式の整数範囲だけを検証します。数値ループに動的な
コールバックやプロセス大域の既定コンテキストはありません。

表は `[値/導関数, 角運動量, 波数]` の連続配置です。C 境界には読み取り専用の平坦な
借用ポインタを渡し、C++ モジュール内では `mdspan` で参照します。`Set_OLP_Kin` の
原子対ごとの Bessel 表・導関数表・行ポインタ表・一時配列の確保と、キャッシュヒット時の
表全体のコピーを除去しました。非局所ポテンシャルの CPU 経路も同じ生成処理とキャッシュを使います。
元の Bessel 漸化式、積分順序、数値精度と GPU の既存高速経路は維持します。

キャッシュは距離の厳密な `==` と角運動量をキーにし、ワーカーごと・関数呼び出しごとに
寿命を限定します。MD の異なる構造へ古い表を持ち越しません。既定の上限は 128 表で、
`OPENMX_OLPKIN_BESSEL_CACHE_MAX` と `OPENMX_SETNL_BESSEL_CACHE_MAX` で各経路を設定できます。
`0` は表を保持せず、最初に確保した作業領域を繰り返し使います。上限到達時も同じ動作です。
キャッシュ追加時の確保失敗は、既存の表を保持したまま作業領域での生成へ切り替えます。

キャッシュされた借用表はワークスペースの破棄まで有効で、後続の追加によって数値領域は
移動しません。作業領域を借用した表は次のサンプリングまで使用できます。呼び出し側は
表の参照中に同じワークスペースを変更せず、ワーカー間で共有しません。
旧 C の Bessel キャッシュ管理を削除し、プロファイルの 3 つの可変大域状態も
`Set_OLP_Kin` の呼び出し内の集計へ移しました。`PrintMemory` の初回表示用 `static` と、
その他の積分配列・大域入力の移行は残ります。

性能検証は [性能回帰の検証手順](PERFORMANCE_REGRESSION.md) に従い、
Intel コンパイラ版 CPU、移行前 GPU、`v3.1` の GPU と比較します。
GeForce 上の目標時間を達成したことを A100/H100 の性能検証の代わりにはしません。

単一構成の CMake では、空のビルド種別を `Release` にし、C++ コアも最適化します。
GPU の密度行列モジュールは、要求された出力チャネルだけをホストへ転送します。
既に計算済みの密行列から上書きする gather は、状態和の作業領域とゼロ初期化を必要としない
専用カーネルを使い、非公開の出力に計算してから反映する境界は維持します。
gather は全 GPU ワーカーの完了と同期を確認してから、非公開の GPU 出力を
ホストの反映先へ直接転送し、大きな中間ホスト領域と追加コピーを除去します。
共通 task/vector と XC 格子は、全入力と作業領域の準備後に GeForce の MPS 実行を
許可します。A100/H100 と HIP の既定の排他制御は維持します。
Force4B は、GPU に全ランク分を常駐できない GeForce では既存の融合 CPU trace を
選択し、native FP64 の強い GPU では従来の GPU archive 経路を保持します。
VNA の GeForce バッチは同じ物理 GPU の共有数から容量を計算し、全ランクの共通サイズへ
揃えます。GPU の可視番号や空き容量がランク間で異なっても collective の回数を変えません。

### 第7段階の検証

通常の CTest 46 件（独立コア 37 件を含む）、コアの ASan/UBSan 37 件、
RTX 5080 上の Kokkos/CUDA コア 37 件が通過しました。Bessel 表の解析値、
キャッシュの借用寿命と独立性、無効入力時の出力保護、pthread による独立キャッシュと
タイミング集計を確認しています。GPU admission では準備・実行の失敗時の巻き戻しと、
結果反映・後片付けに失敗したときに重複する CPU 再計算へ進まないことを検証しました。

`-runtest` の標準 14 例を CPU/GPU とも 4 MPI × 1 OpenMP、同じプロセス内で連続実行しました。
同梱参照に対する最大差は CPU/GPU とも Force `7.9e-11 Ha/bohr` 以下、
エネルギー `9.2e-11 Ha` 以下です。CPU–GPU 間の最大差はエネルギー `1.72e-10 Ha`、
Force `1.9e-11 Ha/bohr` でした。

共通密度行列カーネルは CUDA/HIP の CPU エミュレーションで各 13,674 比較、
RTX 5080 実機で 4,480 比較が通過し、最大誤差は `1.78e-15` です。
sm_80 / sm_90 でも GPU worker・gather のコンパイルを確認しました。
HIP の worker、XC slab、積分スレッドのホスト検証も通過しました。
A100/H100 および AMD 実機の動作・速度、L/L2/L3 全例、NEB、マルチノードの確認は含みません。

実行ログ・入力・結果は `build/refactor-performance-20261019/` に保存しています。
`suite-validation.json` と `suite-{cpu,gpu}-reference.txt`、`suite-cpu-gpu.txt` に
標準テストの比較結果を、性能の実行条件と結果は [性能回帰検証](PERFORMANCE_REGRESSION.md) に記録しています。

## 第8段階：重なり・運動エネルギー作業領域の連続配置

`Set_OLP_Kin` の各 OpenMP ワーカーが `OverlapWorkspace` を所有します。
格子の区間・点数、角運動量・多重度、旧 C カーネルの整数添字範囲を確保前に検証し、
RF 表、4 チャネルの動径積分、組合せの RF オフセット、8 チャネルの複素行列と
変換用行列、Bessel キャッシュを呼び出しの寿命にまとめました。
数値領域の `Tensor` は外へ公開せず、形状を変更できない借用 `mdspan` を返します。
C 境界でもすべての確保が成功してからビューを公開します。

複素行列の旧 `[l0][radial0][m0][l1][radial1][m1]` を、
`multiplicity*l*l + radial*(2*l+1) + magnetic_index` による軌道番号の
`[channel, orbital0, orbital1]` へ移しました。`magnetic_index` は旧 `-l..l` の順です。
最大角運動量 `L`、多重度 `M` の容量では、片側の軌道数が旧 `M*(L+1)*(2*L+3)` から
`M*(L+1)^2` になり、小さい角運動量の磁気成分に付いていた未使用の容量も除去します。
6 重ポインタの 8 配列と変換用の 2 重ポインタの 8 配列について、行ポインタ表と
手動の確保・解放を除去しました。C の内側ループでは算出した添字を使い回し、
配列参照ごとの C ABI 呼び出しや動的確保を行いません。

4 重ポインタの `SumS0`、`SumK0`、`SumSr0`、`SumKr0` は削除しました。
角度方向の組立ては、同じ基底の順で詰めた動径積分の結果を直接読みます。
このための散布コピーと、不要になった組合せメタデータの 4 配列も除去しました。
Bessel の漸化式、動径・Gaunt・複素実変換の加算順序、安定化用の乱数呼び出し、
MPI/OpenMP による原子対の分配を維持します。

CUDA/HIP の両方が同じ C アダプターと所有モジュールを使います。
今回の作業領域はホスト所有であり、Kokkos ランタイムや GPU の初期化を必要としません。
既存の任意の動径 GPU 処理にも同じ平坦なビューを渡し、GeForce と A100/H100 の
実行・精度・メモリ制御の方針は変更しません。

複素数の C ABI 型 `dcomplex` は独立した `openmx_complex.h` に分けました。
従来と同じ 2 つの `double` で、数値モジュールから `openmx_common.h` を読み込む必要はありません。
新しい可変大域状態は追加していません。共通ヘッダーの大域オブジェクト数は 1,191 のままです。
`Comp2Real`、化学種・原子の大域入力、原子対分配表の C による確保、`PrintMemory` の
初回表示用 `static` は残り、OpenMX 全体の大域状態ゼロや例外安全性の達成とは扱いません。

### 第8段階の検証

作業領域の 5 テストでは、角運動量 0〜14 の軌道番号に衝突・隙間がないこと、
各チャネルの連続配置・ゼロ初期化、Bessel キャッシュ更新中のビューの寿命、
独立したワークスペースと異なる形状での連続実行、無効形状・整数境界を確認します。
確保の各位置で `bad_alloc` を発生させ、途中の領域がすべて解放され、C ABI の
既存ビューと数値が変更されず、例外が境界を越えないことも検証します。
この確保失敗の注入はテスト実行ファイルだけにあり、製品ライブラリには含めません。

通常 CTest 51 件、ホスト ASan/UBSan のコア 42 件、RTX 5080 の CUDA 構成のコア
42 件が通過しました。サニタイザーはスタック上限を 8 MiB にして実行しています。
CUDA/HIP 両方の積分スレッド試験は、8 スレッドそれぞれの Bessel キャッシュと
重なり積分の作業領域を実物の C ABI で作り、異なる格子と形状を繰り返して照合します。
CUDA 構成の作業領域 5 テストは GPU を不可視にしても通過しました。
`Methane` は 4 MPI × 1 OpenMP で任意の動径 GPU 処理を有効にし、Bessel キャッシュを
無効にした場合も参照値と一致しました。NVIDIA の起動通知で実際の
`OLPKin_Radial_Kernel` の CUDA カーネル起動も確認しています。
AMD 実機でのビルド・実行の確認は含みません。

標準 `-runtest` の 14 例は CPU/GPU とも 4 MPI × 1 OpenMP、同じプロセス内の
連続実行で通過しました。同梱参照に対する最大差は CPU がエネルギー `7.1e-11 Ha`、
GPU が `8.1e-11 Ha`、Force は両方とも `7.8e-11 Ha/bohr` 以下です。
移行前との最大差は CPU がエネルギー `1.51e-10 Ha`・Force `1.7e-11 Ha/bohr`、
GPU が `1.1e-11 Ha`・`2e-12 Ha/bohr` でした。
CPU–GPU 間はエネルギー `4.91e-11 Ha`・Force `1.3e-11 Ha/bohr` 以下です。
すべての原子と格子も照合し、エネルギーと全 Force 成分の許容値を `1e-7` としています。

同じ sidia333 入力、18 MPI × 1 OpenMP で GPU＋MPS は 104.608 秒、CPU は
753.254 秒でした。原子・格子・25 SCF を揃え、CPU は前段階の 756.165 秒、Intel 版の
800.460 秒と比較しています。詳細は [性能回帰検証](PERFORMANCE_REGRESSION.md) に記録します。

`cmake --install build/cuda` も成功し、`work/openmx` と `work/lib/libopenmx_core.so` は
測定したビルドと SHA-256 が一致しています。実行条件とログは
`build/overlap-workspace-migration/`、各比較・ハッシュ・検証範囲の集計は同ディレクトリの
`validation.json` に保存しています。L/L2/L3、NEB、マルチノードは今回の確認に含みません。

## AMD 実機検証（2026-10-19）

`ff9fcafc` に以下の修正を加えた作業ツリーを、Radeon RX 9060 XT
（`gfx1200`、1 GPU）、ROCm 7.2.0 / Clang 22、OpenMPI 5.0.9、AOCL 5.2.0、
CMake 3.31.12 で構築しました。`OPENMX_CORE_BACKEND=HIP` を選び、
`OPENMX_COPY_TO_WORK=OFF` で `build/hip/openmx` とコアを検証しています。

- Kokkos 5.2.2 の `gfx1200` 登録不足を、上記のビルド時パッチで補いました。
  未変更の配布アーカイブからの適用、自動認識、再適用時の全ファイルの
  SHA-256 と更新時刻の不変を確認しています。
- HIP の Force4B の 2 つの target contraction に、既存ワーカーと共通の
  物理 GPU UUID による排他制御を適用しました。GPU を共有する MPI ランクの
  長い contraction を同時実行せず、カーネルの完了後に解放します。
  容量復旧経路の MPI 集団通信へロックを持ち越しません。
- ROCm ランタイム探索のホスト試験で、一時ディレクトリから本番 Makefile を読む際も
  `openmx_core.mk` を見つけられるよう、Makefile の親を make の include 探索先に加えました。

CTest は 56/56 件（コア 42 件、うち GPU ラベル 9 件）を通過しました。
別の Serial コアの Debug ビルドも AddressSanitizer / UndefinedBehaviorSanitizer 付きで
42/42 件を通過しています。サニタイザー検査の対象はホスト C++ コアです。

修正版では同梱参照に対し、次の通常設定の計算が通過しています。
S は各 MPI ジョブで 14 例を連続実行し、L は `CG15c-DC-LNO`、`ZrB2_2x2`、
`nsV4Bz5` を同じ MPI ジョブで連続実行しています。

| 計算 | 合格数 |
| --- | --- |
| S 全14例・CPU、4 MPI × 4 OpenMP | 14 / 14 |
| S 全14例・GPU、4 MPI × 4 OpenMP | 14 / 14 |
| S 全14例・GPU、8 MPI × 2 OpenMP | 14 / 14 |
| L 代表3例・GPU、4 MPI × 4 OpenMP | 3 / 3 |

終了表示に加え、全エネルギー履歴、全原子の Force 成分、原子種・座標、FFT 格子を照合しました。
エネルギー・Force の許容差は `1e-7`、座標は出力桁に合わせた `1e-5` です。
全45計算での同梱参照との差は、エネルギー `1.2e-9 Ha`、Force `8.0e-10 Ha/bohr` 未満でした。
小さな行列の密対角化には既定の CPU フォールバックを許容しています。

計算時間も同じ 4 MPI × 4 OpenMP で記録しています。次は単発の観測値で、
ジョブ全体の値は起動・終了を含みます。修正版 GPU S は Kokkos Tools の記録も含みます。
CPU S と ZrB2 では時間の増加が見られました。

| 対象 | 移行前（秒） | 修正版（秒） |
| --- | ---: | ---: |
| S全14例・CPU（ジョブ全体） | 86.76 | 108.33 |
| S全14例・GPU（ジョブ全体） | 87.68 | 80.79 |
| L代表3例・GPU（ジョブ全体） | 405.53 | 420.74 |
| ZrB2（OpenMX内計測） | 162.86 | 178.87 |

通常設定・4 MPI × 4 OpenMP の `ZrB2_2x2` 単独実行は 177.99 秒で正常終了し、
移行前との差はエネルギー `9.1e-13 Ha`、全 Force 成分 `1.1e-12 Ha/bohr` 未満でした。
デバッガ、診断用 preload、詳細な時間記録を使った試行は、通常実行の検証と区別しています。
追加の検証も通過しました。

| 計算 | 合格数 |
| --- | --- |
| 移行前の restart を読む通常計算（旧版・修正版） | 2 / 2 |
| `scf.ProExpn.VNA off`（旧版・修正版） | 2 / 2 |
| Serial コアに差し替えた legacy HIP の ZrB2 | 1 / 1 |
| Cr2 の非共線 GPU 固有値分解強制（旧版・修正版） | 2 / 2 |

Cr2 は `OPENMX_CLUSTER_NONCOL_GPU_SWITCH_NUM=1` と
`OPENMX_CLUSTER_GPU_DIAG=1` を指定しました。各実行で MAGMA の成功記録を 22 件確認し、
`n=20` の実行と `n=40` の非共線実行で `info=0`、`mout=n` を確認しました。
強制 GPU 同士のエネルギーは出力桁で一致し、Force の最大差は `1.0e-12 Ha/bohr` 未満です。

Kokkos Tools では HIP の Bessel / 積分カーネルを各 2,288 回、HIP メモリ確保を
910 回記録しました。4 ランク全てに両カーネルの実行記録があります。
旧版の比較用計算を含む合計 84 計算が数値検証を通過しています。
同梱参照との差はエネルギー `1.2e-9 Ha`、Force `8.0e-10 Ha/bohr` 未満、
旧版や CPU/GPU・MPI 分割間の比較はエネルギー `1.7e-10 Ha`、Force `7.8e-11 Ha/bohr` 未満です。

検証条件、ソース・入力・実行ファイルのハッシュ、数値比較とログは
`work/codex_refactor_hip_20261019/summary.json` と同ディレクトリに保存しています。
実機確認は単一ノード・RX 9060 XT 1 GPU、L は代表 3 例です。
L2/L3、NEB、複数 GPU、MI300A 実機は今回の確認に含みません。

## Radeon 向け半径積分の最適化（2026-10-19）

`99ff9d29` を基準に、同じ RX 9060 XT (`gfx1200`) / ROCm 7.2.0 環境で
半径方向 Fourier–Bessel 積分を最適化しました。C API、所有権、Kokkos による
実行空間の抽象化を維持し、GPU ベンダーや製品名による分岐は追加していません。

- デバイス側の Bessel 表は波数方向を連続にし、積分する隣接レーンの読み込みをまとめました。
  ホスト実行では従来どおり半径方向を連続にします。
- GPU の固定32波数バッチを、Bessel 表と出力の合計で原則8 MiB以内のバッチへ変更し、
  転送・同期回数を減らしました。1波数だけで上限を超える入力では、その1波数分を確保します。
  ホスト実行のバッチ上限は32のままです。
- 全て上書きする View とミラーのゼロ初期化を省きました。バッチ末尾の余剰領域は計算に使いません。
  非同期処理を終えてから View を解放する RAII ガードは維持しています。
- 値専用 Bessel カーネルを追加しました。`j0(x)=sin(x)/x`（原点は1）と、
  `x > 2*(lmax+1)` の領域で前進漸化式を使い、それ以外は従来の後退漸化式を使います。
  根拠は [DLMF 10.49](https://dlmf.nist.gov/10.49)、
  [10.51.1](https://dlmf.nist.gov/10.51.E1)、
  [10.74(iv)](https://dlmf.nist.gov/10.74.iv) です。

半径方向の加算順序と `-ffp-contract=off` は維持しています。
従来の値・微分用 Bessel 関数、GPUなし要求時の `transform_host` と実行経路の選択は変更していません。

### 単体性能の再測定

既存の HIP ビルド設定に、任意ベンチマークを追加できます（既定では無効）。

```sh
cmake -S . -B build/hip -DOPENMX_CORE_BUILD_BENCHMARKS=ON
cmake --build build/hip --target openmx_radial_benchmark --parallel 8
HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0 taskset -c 0 \
  build/hip/core/openmx_radial_benchmark --device 0 --repetitions 11
```

`--device -1` は GPU を初期化しないホスト経路です。各ケースで2回ウォームアップし、
サンプリング済み入力からの変換時間を JSON Lines で出力します。時刻計測には View の確保、
転送、カーネル、同期、内部 View の解放とホスト結果の生成を含み、入力の生成と Kokkos 初期化は含みません。
毎回、従来のホスト積分との全出力比較を行い、最大絶対誤差 `2e-12` 超過で失敗します。

保存した変更前・変更後のコアを同じ実行ファイルへ `LD_PRELOAD` して交互に3組測定しました。
表は各組11回の中央値から、さらに3組の中央値を取った値です。GPUクロックは固定していません。
表は合成入力による積分変換の時間です。OpenMX 計算全体の時間は次節に示します。

| 入力（半径点 × 波数、出力行数） | 変更前 (ms) | 変更後 (ms) | 倍率 |
| --- | ---: | ---: | ---: |
| ゼロ次 (901 × 900、1行) | 15.998 | 1.164 | 13.74 |
| PAO (901 × 900、8行、lmax=3) | 13.998 | 2.470 | 5.67 |
| 軌道積 (200 × 500、252行、lmax=6) | 3.053 | 1.626 | 1.88 |
| 端数バッチ (65 × 257、8行、lmax=3) | 1.294 | 0.411 | 3.15 |

この測定でのホスト参照との最大絶対誤差は `2.23e-16` 未満でした。

### 計算全体の検証

S全14例を4 MPI × 4 OpenMPで、変更前・変更後を交互に各3回測定しました。
次の時間は OpenMX の `runtest.result` に出る全ケースの合計で、MPI 起動を含みません。

| 対象 | 1回目 (s) | 2回目 (s) | 3回目 (s) | 中央値 (s) |
| --- | ---: | ---: | ---: | ---: |
| 変更前 | 78.63 | 79.48 | 80.17 | 79.48 |
| 変更後 | 78.18 | 79.43 | 80.25 | 79.43 |

S全体では測定のばらつきが差を上回り、明確な短縮は確認できませんでした。
今回速くした初期化時の積分以外に、SCF中の行列構築・対角化などの時間を含みます。
既定の小行列CPU密対角化の選択条件は変更していません。

HIP の CTest 56件、Serial Release 42件、ホスト C++ コアの ASan/UBSan 42件を通過しています。
全次数0～29、原点・微小引数・零点近傍・漸化式の切替境界、複数タイルと最後の端数、
台形・Gauss の軌道積、高次数の実デバイス実行を追加の数値検査で確認しました。

S全14例のGPU実行（4 MPI × 4 OpenMPを各3回、8 MPI × 2 OpenMPを1回）、
GPUなしCPU実行（4 MPI × 4 OpenMPを1回）、L代表3例のGPU実行を数値比較しました。
L は `CG15c-DC-LNO`、`ZrB2_2x2`、`nsV4Bz5` を同じ MPI ジョブで連続実行しています。
変更前の比較用実行を含む115計算が、終了状態に加えて全エネルギー履歴、全原子のForce成分、
原子種・座標、FFT格子の照合を通過し、NaN/Inf・異常終了はありませんでした。
同梱参照との差はエネルギー `1.126e-9 Ha`、Force成分 `7.97e-10 Ha/bohr` 未満、
変更前やCPU/GPU・MPI分割間との差はエネルギー `2.75e-10 Ha`、Force成分 `3.01e-11 Ha/bohr` 未満です。

8 MPI × 2 OpenMP の記録では、Fourier変換を担当する4ランクでHIPの
Bessel・積分カーネルをそれぞれ合計212回確認しました。半径積分のゼロ初期化カーネルは0回です。
最終レビューではホスト実行の端数バッチに不要なパディングを付けないよう調整し、
ホストの42件・サニタイザー42件とHIPの56件を再確認しました。
このホスト用調整の前後でRadeon用コアのバイナリは完全一致しており、計測結果との対応も確認しています。

全S/L入力と参照60ファイルは前回検証時から不変です。
検証条件、ソース・バイナリのハッシュ、数値比較、単体計測の生データとログは
`work/codex_radeon_perf_20261019/summary.json` と同ディレクトリに保存しています。
ビルド済みアプリケーションは `build/hip/openmx`、ベンチマークは
`build/hip/core/openmx_radial_benchmark` です。実機での性能測定はRadeon 1 GPUで行い、
MI300A・CUDAの性能測定とL全16例・L2/L3の再実行は今回の確認に含みません。

## 第9段階：非局所積分の所有権と CPU 処理の共通化

比較元は Radeon 向け最適化を含む `d0355507` です。半径積分・値専用 Bessel の
最適化、`gfx1200` のビルド時パッチ、HIP の Force4B の共有 GPU 制御を保持しています。

`Set_Nonlocal` の CPU 経路は CUDA/HIP で同一だった 693 行を
`source/openmx_nonlocal_cpu.h` の共通処理へまとめました。動径・Gaunt・複素実変換の
式、加算順序、スピン・射影子・原子対の番号順序と MPI/OpenMP の分配は維持しています。
デバイス側の Bessel・動径積分カーネルは各バックエンドの既存実装を使います。

CPU と GPU 計算後の角度組立ては、それぞれの OpenMP ワーカーが
`NonlocalWorkspace` を所有します。化学種ごとの射影子数と角運動量をコピーし、
射影子の磁気成分の累積オフセットも所有します。従来の射影子番号 1〜N と
未使用の 0 番はメタデータに残し、数値配列には各射影子の実際の `2*l+1` 成分だけを詰めます。
異なる化学種の上限に合わせる容量は残りますが、全射影子を最大角運動量の幅へ広げません。

4 チャネルの 5 重ポインタ `TmpNL` / `TmpNLr` / `TmpNLt` / `TmpNLp` は、
`[channel, packed orbital, projector component]` の連続領域へ移しました。
変換用の 4 行列も連続領域へ移し、C++ は借用 `mdspan`、C は平坦なビューを使います。
内側ループには C ABI 呼び出しや確保・解放を追加せず、原子対の添字を使い回します。
軌道番号の計算は `openmx_core_orbitals.h` に共通化し、重なり積分でも同じ計算を使います。

CPU 経路の 3 重ポインタの積分表、RF/NLRF 表と Bessel キャッシュも同じ所有者へまとめました。
NLRF は従来と同じ 2 スピン領域を独立に持ちます。GPU 計算後の角度組立てでは、
CPU 用の格子・動径表・積分表・Bessel キャッシュを確保しません。
どちらのモードも Kokkos ランタイムや GPU の初期化を必要としないホスト所有です。
C ABI は全確保が成功してからビューを公開し、失敗時に既存のビューを変更しません。

`SetNonlocal_ProfileEnabled` の可変 `static` キャッシュも削除しました。
関数呼び出しごとに設定を読み、同じプロセスで設定を変えても古い値を使いません。
共通ヘッダーの大域オブジェクト数は 1,191 のままで、新しい可変大域状態は追加していません。
出力の `DS_NL`・`HNL`、基底や幾何の大域入力、`Comp2Real`、初回表示用 `static` と
GPU の容量不足通知用の状態は残ります。共通 CPU 処理も引き続き C アダプターであり、
OpenMX 全体の大域変数・多重ポインタの除去や数値カーネルの全面 C++ 化の完了とは扱いません。

### 第9段階の検証

通常 CTest 56 件、ホスト ASan/UBSan のコア 47 件、RTX 5080 の CUDA 構成の
コア 47 件が通過しました。新しい 5 テストは、角運動量 0〜14 の可変長レイアウト、
空の化学種、メタデータのコピーと独立性、連続配置・チャネルの分離、Bessel 借用表の寿命、
CPU/GPU 角度組立ての確保範囲、無効入力、確保途中の全位置での失敗と解放を検証します。
CUDA 構成の新しい 5 テストは GPU を不可視にしても通過しました。
CUDA/HIP のホスト積分試験は、実物の C ABI を 8 スレッドから使い、異なる格子と形状を
繰り返して照合しています。プロファイル設定を 1→0→1 と変更する試験も通過しています。

標準 S 全14例を CPU/GPU それぞれ 4 MPI × 1 OpenMP で同じ MPI ジョブ内に連続実行し、
同梱参照・第8段階の保存済み結果・CPU/GPU 間の全エネルギー最終値と全 Force 成分、
原子種・座標・FFT 格子を照合しました。全28計算が完走し、同梱参照との最大差は
エネルギー `4.91e-11 Ha` 以下、Force `7.81e-11 Ha/bohr` 以下でした。
CPU/GPU 間の差はそれぞれ `4.91e-11 Ha` 以下・`9.01e-12 Ha/bohr` 以下です。
さらに Cr2 を 4 MPI × 2 OpenMP で、通常の GPU 経路と
`OPENMX_SETNL_OPENACC=0`・`OPENMX_SETNL_BESSEL_CACHE_MAX=0` の CPU 切替経路で実行し、
同梱参照と相互比較を通過しました。標準テストの CPU/GPU 実行は CPU コアを分けて並行に行い、
その時間を性能値には使っていません。

`sidia333_nc_cluster.dat` は他の計算と重ねず 18 MPI × 1 OpenMP で計測し、
RTX 5080＋MPS が 104.011 秒、CPU が 760.342 秒でした。前者は比較元の
`d0355507` の 103.702 秒と同程度で、約105秒の目標を維持しています。
CPU は第8段階の保存済み結果との数値比較を通過し、同じ条件で保存した Intel 版の
800.460 秒に比肩する性能でした。CPU の今回値と第8段階との差はエネルギー
`6.11e-11 Ha` 以下・Force `1.20e-10 Ha/bohr` 以下、Intel 版との差は
それぞれ `7.30e-11 Ha` 以下・`1.23e-10 Ha/bohr` 以下です。
いずれも単一回の確認で、速度向上の主張には使いません。

`cmake --install build/cuda` も成功し、`work/openmx` と `work/lib/libopenmx_core.so` が
CPU/GPU の両性能測定で使ったファイルと SHA-256 で一致することを確認しました。
入力・実行条件・ハッシュ・全物理量とログは `build/nonlocal-workspace-migration/`、
確認結果の一覧は同ディレクトリの `validation.json`、時間の比較は
`docs/PERFORMANCE_REGRESSION.md` に記録しています。
この作業機で Radeon、MI300、A100/H100 の実機を再検証した結果は含みません。
L/L2/L3 全例、restart、MD/NEB、多ノードの再実行も今回の確認範囲には含みません。

## 次の移行単位と完了条件

1. 化学種メタデータを C++ の所有値にする。PAO/VPS/VNA・軌道積表の参照側に
   明示的なビューを渡し、借用状態で残る `Spe_*` 大域ポインタと null コンテキストの
   手動確保経路を段階的に除去する。
   導関数、原子ポテンシャルの構築、入力解析、MPI 配布、restart の制御も移す。
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
