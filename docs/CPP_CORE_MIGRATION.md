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
| VNA2/VNA3 の作業領域 | `core/src/vna_workspace.cpp` | 4 チャネルの軌道対行列、複素実変換行列、Bessel 値・導関数と漸化式の一時領域をワーカーごとに所有。C++ は `mdspan`、CUDA/HIP 共通の C アダプターは平坦なビューを借用する。 |
| VNA 前処理と基底メタデータ | `core/src/vna_preparation.cpp` | 化学種ごとの角運動量・基底数・軌道数と射影子番号をコピーして所有。重み付き動径表は `[channel, radial row, point]` の連続領域に置き、空の基底に数値行を確保しない。 |
| DS_VNA の CPU 作業領域 | `core/src/dsvna_workspace.cpp` | 4 チャネルの角度行列、動径和、複素実変換と Bessel の一時領域をワーカーごとに所有。化学種内の実在する基底を軌道・動径オフセットで並べ、C++ は `mdspan`、CUDA/HIP 共通 CPU 処理は平坦なビューを借用する。 |
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

CTest に 62 件の独立コアテストを登録しています。所有権と例外時の整合性、周期セルの番号順序・
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

## 第10段階：VNA2/VNA3 の作業領域とバッチ状態の所有権

比較元は `67ac8d9d` です。CUDA/HIP の VNA2・VNA3 で重複していた CPU 処理を
`source/openmx_vna_cpu.h` の 1 つのアダプターへまとめました。両方向は、軌道積を持つ化学種と
中性原子ポテンシャルを持つ化学種を交換し、元と同じ順序の座標の引き算を選びます。
単なる変位の符号反転へ置き換えず、丸めの順序も保持しています。Gaunt・球面調和関数、
動径積分、上半分から下半分への共役コピー、複素実変換と Cartesian 導関数の式・加算順序は維持します。

4 つの 6 重ポインタ行列は `[channel, packed orbital, packed orbital]` の連続領域へ移し、
変換用の 4 行列も `VnaWorkspace` が所有します。磁気成分の添字は重なり・非局所積分と
同じ `openmx_core_orbitals.h` を使い、内側ループで C ABI を呼びません。
原子対ごとの Bessel 表・導関数と漸化式の一時領域の `malloc/free` を除去し、
各ワーカーが所有する領域を再利用します。VNA の `Spherical_Bessel2` は
既存の `PairBesselWorkspace` と漸化式の開始次数が異なるため、元の計算式を維持します。
GPU の初期化を必要とせず、C ABI は確保成功後だけビューを公開します。
CPU のメモリ使用量の記録は、実際の連続領域の容量と `sizeof(dcomplex)` を使い、
VNA2 のワーカー0が各呼び出しで1回記録します。

`SetPro_gpu3` はファイルスコープから削除し、`Set_ProExpn_VNA` のローカル状態として
begin・VNA2・VNA3・end へ明示的に渡します。CUDA/HIP 共通の
`openmx_vna_gpu_context.h` は GPU ベンダーの型に依存しない C のバッチ記述で、同期完了まで呼び出し元が保持します。
GPU テーブルの確保・解放は引き続き各バックエンドの begin/end で行います。
`SetPro_verbose` の可変キャッシュと VNA2/VNA3 の初回表示用 `static` も削除しました。
CUDA/HIP それぞれ4個のファイル内の可変状態を減らし、共通ヘッダーの大域オブジェクトは
1,191 個のままです。VNA2/VNA3 の原子ごとの時間加算には OpenMP の atomic を追加し、
同じ原子の原子対が複数スレッドへ分割された場合の計測値の競合を修正しました。

CUDA/HIP のデバイスカーネル、DS_VNA と HVNA の構築、共有 GPU のメモリ予算と容量復旧、
Radeon 向け半径積分・Force4B、GeForce/A100/H100 の演算方針は保持しています。
バックエンドごとに既存の12関数の本体が比較元と一致することも確認しました。
DS_VNA/HVNA の他の静的コンテキスト、幾何・基底の大域入力、出力の多重ポインタ、
原子対リストの手動確保は残ります。アプリケーション全体の再入可能性や大域状態ゼロを達成した段階ではありません。

### 第10段階の検証

新しい5個のコア試験は角運動量0〜14の配置、const ビュー、所有領域の独立性、
C ABI、無効入力・整数範囲と、確保途中の全位置での失敗時の解放を検証します。
`tests/run_vna_workspace_cpu.py` は本番の共通 CPU 処理と C++ 所有者を使い、
解析解のある s 軌道の積分・Cartesian 導関数と比較します。1/3/11積分点、
1/2/8スレッド、異なる化学種・基底数・空の基底、両方向、極軸上・同一原子対の90条件を確認します。
2つの独立した GPU バッチ状態の片方を解放しても、もう片方の表と設定が保たれること、
解放を繰り返せること、診断表示設定の切り替えも同じ試験で確認します。

通常 CTest 62件、ASan/UBSan のコア52件、CUDA 構成のコア52件を通過しました。
新しい5件は CUDA 構成で GPU を不可視にしても通過しています。
解析解の90条件は通常構成・ASan/UBSan・HIP ソースを選択したホスト構成で通過し、
最大絶対誤差は `2.23e-16` 未満でした。CUDA/HIP の既存の容量復旧カーネルも、
固定した旧カーネルとの各32条件の比較を通過しました。
サニタイザーの実行はスタック上限を8192 KiBに設定し、`LD_LIBRARY_PATH` と
`LD_PRELOAD` を外しています。アプリケーションのビルドや実行条件は変更しません。

標準 S 全14例を CPU/GPU それぞれ4 MPI × 1 OpenMPで同じ MPI ジョブ内に連続実行し、
同梱参照・第9段階・CPU/GPU 間の最終エネルギー、全 Force 成分、原子種・座標・FFT 格子を
照合しました。全28計算が完走し、同梱参照との最大差はエネルギー `7.61e-11 Ha` 以下、
Force `7.81e-11 Ha/bohr` 以下です。CPU の最終エネルギー・全 Force 成分は
第9段階と出力桁で一致しました。CPU/GPU 間の最大差はそれぞれ `4.91e-11 Ha` 以下・
`8.01e-12 Ha/bohr` 以下です。Cr2 は4 MPI × 2 OpenMPでも通常の GPU 経路と
`OPENMX_SETPRO_GPU=0` の CPU 切替経路を実行し、同梱参照と相互比較を通過しています。
標準テストの CPU/GPU 実行は CPU コアを分けて並行に行い、その時間を性能値には使いません。

`sidia333_nc_cluster.dat` は他の計算と重ねず18 MPI × 1 OpenMPで計測しました。
RTX 5080＋MPS は103.802秒で、前回の104.011秒と同程度、約105秒の目標を維持しています。
CPU は755.005秒で、前回760.342秒、保存済みの同条件の Intel 版800.460秒に比肩する性能でした。
CPU のエネルギー・全 Force 成分は前回と出力桁で一致し、Intel 版との差も
エネルギー `7.30e-11 Ha` 以下・Force `1.23e-10 Ha/bohr` 以下です。
いずれも単一回の確認で、計算全体の速度向上の主張には使いません。

`cmake --install build/cuda` も成功し、`work/openmx` と `work/lib/libopenmx_core.so` が
CPU/GPU の両性能測定で使ったファイルと SHA-256 で一致することを確認しました。
入力、実行条件、バイナリとソースのハッシュ、物理量とログは
`build/vna-workspace-migration/`、確認結果の一覧は同ディレクトリの `validation.json`、
時間の比較は `docs/PERFORMANCE_REGRESSION.md` に記録しています。
Radeon、MI300、A100/H100 の実機と L/L2/L3 全例、restart、MD/NEB、多ノードの再実行は
今回の確認範囲には含みません。

## 第11段階：VNA 前処理の所有権と DS_VNA の呼び出し状態

比較元は `59aa6b3c` です。`VnaPreparation` は化学種ごとの基底の角運動量・基底数・
磁気成分を含む軌道数と、VNA 射影子の角運動量・動径番号をコピーして所有します。
入力メタデータの変更や解放に依存せず、空の角運動量グループや基底を扱います。
`RadialBasisLayout` のオフセットで実在する基底だけを並べ、2 種類の重み付き動径表を
`[channel, radial row, integration point]` の連続した `Tensor` に置きます。
従来の `Bessel_Pro00/01` の 4 重ポインタと射影子番号表の手動確保・解放を除去しました。
C++ は `mdspan`、C 境界は寿命を所有者に結び付けた平坦なビューを使います。

CUDA/HIP の重複した前処理ループは `source/openmx_vna_preparation.h` にまとめました。
重みの `0.50*Dk*weight*k*k`、RF 補間との積、追加の `k` の積の順序と、
元の OpenMP の点分割を維持します。内側ループに新たな C ABI 呼び出しは加えません。
既存のスカラー `RF_BesselF` は Fourier 表を借用する互換アダプターとして残ります。
新しい所有モジュールには大域変数、MPI、GPU ランタイムへの依存がありません。
形状・整数範囲を検証し、C ABI は全確保に成功した場合だけビューを公開します。

DS_VNA の `SetPro_gpu2` も CUDA/HIP 双方のファイルスコープから削除しました。
`Set_ProExpn` 内のローカル状態を begin・run・end へ明示的に渡し、
型の定義は `openmx_vna_gpu_context.h` で共有します。GPU 用のパック表は
所有済みの基底メタデータと重み付き動径表から構築し、同期処理の完了後に解放します。
GPU 表の確保・解放は引き続きネイティブの begin/end が担います。

デバイスカーネル・GPU メモリ予算・容量復旧と HVNA の演算は保持しました。
比較元との関数本体の照合は CUDA 18 関数、HIP 25 関数で一致しています。
Radeon 向けの半径積分・Force4B と GeForce/A100/H100 の演算方針も変更していません。
共通ヘッダーの大域オブジェクトは 1,191 個のままで、新しい可変大域状態はありません。
DS_VNA の CPU 角度組立てに使う 5 重ポインタ、HVNA の静的コンテキスト、
幾何・Fourier 表の大域入力、出力の多重ポインタは今後の移行対象です。

### 第11段階の検証

新しい5件のコア試験は、角運動量0〜14、異なる化学種・基底数・空の基底、
射影子の順序と末尾2枠、const ビュー、コピー後の独立性、不正形状と整数範囲、
C ABI の出力保護、確保途中の全位置での失敗時の解放を検証します。
通常 CTest 68件、ASan/UBSan のコア57件、CUDA 構成のコア57件が通過し、
新しい5件は CUDA 構成で GPU を不可視にしても通過しました。

`tests/run_vna_preparation_cpu.py` は本番の C++ 所有者、共通の前処理ループ、
各バックエンドの DS_VNA の begin/end を組み合わせ、GPU を使わずに検証します。
1/3/17積分点、1/2/8スレッド、射影子の動径多重度0/1/3、空の基底の有無の54条件で、
補間の呼び出し回数、重み付き動径表、GPU 転送用の全有効行、射影子と軌道のオフセット、
複素実変換表が期待値と完全一致しました。入力の基底数・化学種数を変えても所有済みの
メタデータが保たれること、2つの独立した DS_VNA 状態と繰り返し解放、GPU 無効時の
ゲートも確認します。同じ54条件は ASan/UBSan と HIP ソースを選択したホスト試験でも通過しました。
既存の容量復旧カーネルと旧カーネルとの各32条件の比較、HIP の HVNA ホスト試験も通過しています。

標準 S 全14例を CPU/GPU それぞれ4 MPI × 1 OpenMPで同じジョブ内に連続実行し、
同梱参照・第10段階・CPU/GPU 間で最終エネルギー、全 Force 成分、原子種・座標・FFT 格子を
照合しました。全28計算が完走し、同梱参照との最大差はエネルギー `9.21e-11 Ha` 以下、
Force `7.81e-11 Ha/bohr` 以下です。CPU は第10段階と出力桁で一致しました。
CPU/GPU 間の最大差はそれぞれ `4.81e-11 Ha` 以下・`9.01e-12 Ha/bohr` 以下です。
Cr2 は4 MPI × 2 OpenMPでも通常の GPU 経路と `OPENMX_SETPRO_GPU=0` の CPU 切替経路を
実行し、同梱参照と相互比較を通過しました。
標準スイートの CPU/GPU 実行はコアを分けて並行に行い、その時間を性能値には使いません。

`sidia333_nc_cluster.dat` は18 MPI × 1 OpenMPで他の計算・ビルドと重ねず実行しました。
RTX 5080＋MPS は103.891秒で約105秒の目標を維持し、CPU は753.585秒でした。
保存済みの同条件の Intel 版800.460秒に比肩する性能です。
CPU のエネルギーと全 Force 成分は第10段階と出力桁で一致しました。
GPU の前回との差はエネルギー `1.03e-12 Ha` 以下、Force `1.01e-12 Ha/bohr` 以下で、
CPU と Intel 版の差もそれぞれ `7.30e-11 Ha` 以下・`1.23e-10 Ha/bohr` 以下です。
いずれも単一回の確認で、計算全体の速度向上の主張には使いません。

最後に読取り専用の引数修飾を反映した最終ビルドも、ホストの全実行領域と CUDA の
デバイスコードは測定版と一致しています。差はコンパイラの一時オブジェクト名と
モジュール識別子だけで、標準スイートは最終ビルドで実行しました。
`cmake --install build/cuda` も成功し、`work/openmx` と `work/lib/libopenmx_core.so` が
最終ビルドと SHA-256 で一致することを確認しました。
入力・実行条件・ハッシュ・全物理量とログは `build/vna-preparation-migration/`、
確認結果の一覧は同ディレクトリの `validation.json`、測定版との照合は
`final-binary-equivalence.json`、時間の比較は `docs/PERFORMANCE_REGRESSION.md` にあります。
Radeon、MI300、A100/H100 の実機と L/L2/L3 全例、restart、MD/NEB、多ノードの再実行は
今回の確認範囲には含みません。

## 第12段階：DS_VNA の CPU 作業領域と角度組立ての共通化

比較元は `36ad25d9` です。DS_VNA の CPU 経路に残っていた4組の5重ポインタを、
`DsvnaWorkspace` が所有する `[channel, orbital, projector component]` の
連続領域へ移しました。2組の動径和、4組の複素実変換行列、Bessel 値・導関数と
漸化式の一時配列も同じワーカー単位の C++ 所有者で管理します。
C++ は `mdspan`、C 境界は借用する平坦なビューを使い、確保途中の例外では
完成済みの領域を自動解放します。C ABI は全確保が成功するまで出力を更新しません。
このホスト作業領域の生成・破棄に Kokkos や GPU の初期化は必要ありません。

`RadialBasisLayout` に化学種内の軌道オフセットを追加しました。空の角運動量群を
飛ばして実在する軌道・動径行だけを並べ、ワーカーの容量は化学種ごとの最大値にします。
射影子と PAO の角運動量上限は独立に扱います。作業領域が保持するのは容量だけで、
前処理オブジェクトの破棄後も所有配列は有効です。計算時には所有済みの基底数・
射影子番号・積分点数を明示的に渡し、元の `List_YOUSO` と基底数配列を参照しません。

CUDA/HIP で重複していた CPU 組立てを `source/openmx_dsvna_cpu.h` にまとめました。
積分の加算順序、Gaunt 係数・球面調和関数からの組立て、複素実変換、
`Type_DS_VNA` へのキャスト位置と Cartesian 導関数の式は維持しています。
各 `Set_ProExpn` にあった診断用の `static firsttime` は除去し、ワーカー0が
その呼び出しで所有する行列のサイズを報告します。原子対を複数スレッドへ分配するときの
`time_per_atom` の加算は atomic にし、同じ原子への同時更新の競合を除きました。

ホスト側の `Spherical_Bessel2` は、波数ゼロでも `sb[lmax+1]` を初期化します。
最高次数の導関数がこの隣接次数を読むため、以前は未初期化値を使う場合がありました。
GPU 側は既にこの次数まで初期化しており、変更していません。
GPU の計算・メモリ予算・容量復旧に関する関数本体は比較元と CUDA 18関数・HIP 25関数で
一致しています。共通ヘッダーの大域オブジェクトは1,191個で、新規追加はありません。

### 第12段階の検証

新しい5件のコア試験で、可変長の基底、角運動量0〜14の非対称な PAO/射影子容量、
空の基底・射影子、const ビュー、独立した寿命、C ABI のオフセットと失敗時の出力保護、
確保途中の全位置での失敗時の解放を確認しました。通常 CTest 74件、ASan/UBSan の
コア62件、CUDA 構成のコア62件が通過し、新しい5件は GPU を不可視にしても通過しました。

`tests/run_dsvna_workspace_cpu.py` は本番の共通 CPU 処理と C++ 所有者を、
`tests/fixtures/dsvna_cpu_before_raii.c` に固定した第11段階の CPU 組立てと照合します。
1/3/11積分点、1/2/8スレッド、4種類の s/p/d・空基底と3種類の射影子構成の
108条件を float/double 各精度で実行し、未使用の出力領域を含めて完全一致しました。
旧処理の計時競合を参照試験へ持ち込まないよう、参照側は1スレッドで実行します。
p/d の角度組立ては決定的な試験用係数で旧処理と比較し、s 軌道は独立した解析式でも
積分と Cartesian 導関数を照合します。解析値との差は float で0、double で
`5.56e-17` 以下でした。ゼロ波数の Bessel 値・導関数、所有後に変更した元の
メタデータへの非依存性、同一原子への計時加算も確認します。同じ216条件が
ASan/UBSan と HIP ソースを選択したホスト試験でも通過しました。

標準 S 全14例を CPU/GPU それぞれ4 MPI × 1 OpenMPで同じジョブ内に連続実行し、
同梱参照・第11段階・CPU/GPU 間でエネルギー、全 Force 成分、原子種・座標・FFT 格子を
照合しました。全28計算が完走し、同梱参照との差はエネルギー `4.90e-11 Ha` 以下、
Force `7.80e-11 Ha/bohr` 以下です。CPU は第11段階と全例で出力桁まで一致しました。
GPU の第11段階との差はそれぞれ `8.91e-11 Ha` 以下・`2.60e-11 Ha/bohr` 以下、
CPU/GPU 間の差は `4.90e-11 Ha` 以下・`2.60e-11 Ha/bohr` 以下でした。
Cr2 は4 MPI × 2 OpenMPでも通常の GPU 経路と `OPENMX_SETPRO_GPU=0` の CPU 切替経路を
実行し、同梱参照との照合を通過しました。両経路の結果は出力桁で一致しています。
標準スイートの CPU/GPU 実行はコアを分けて並行に行い、その時間を性能値には使いません。

`sidia333_nc_cluster.dat` は18 MPI × 1 OpenMPでほかの計算・ビルドと重ねず実行しました。
RTX 5080＋MPS は104.540秒で約105秒の目標を維持し、CPU は765.814秒でした。
CPU の全体時間は前回より約1.62%増ですが、変更対象を含む VNA 処理の最大時間は
27.339秒から25.838秒でした。保存済みの同条件の Intel 版800.460秒に比肩する性能です。
CPU のエネルギーと全 Force 成分は第11段階と出力桁で一致しました。
GPU の前回との差はエネルギー `1.03e-12 Ha` 以下、Force `1.01e-12 Ha/bohr` 以下で、
CPU と Intel 版の差もそれぞれ `7.30e-11 Ha` 以下・`1.23e-10 Ha/bohr` 以下です。
いずれも単一回の比較であり、全体の速度向上の主張には使いません。

`cmake --install build/cuda` も成功し、`work/openmx` と `work/lib/libopenmx_core.so` が
CPU/GPU の性能測定と標準スイートに使ったビルドと SHA-256 で一致しました。
入力・実行条件・ハッシュ・全物理量とログは `build/dsvna-workspace-migration/`、
確認結果の一覧は同ディレクトリの `validation.json`、時間の内訳と比較上の注意点は
`docs/PERFORMANCE_REGRESSION.md` に記録しています。
Radeon、MI300、A100/H100 の実機と L/L2/L3 全例、restart、MD/NEB、多ノードの再実行は
今回の確認範囲には含みません。

HVNA の静的コンテキスト、幾何・Fourier 表の大域入力、最終出力の多重ポインタと
ネイティブ GPU バッファの begin/end 所有は、引き続き移行対象です。

## 第13段階：HVNA の配置表と GPU 呼び出し状態の共通化

比較元は `4b612571` です。CUDA と HIP の HVNA に残っていた静的コンテキスト、
HIP の容量復旧用コンテキストと通常経路の選択フラグの計4個を、`Set_ProExpn` の
ローカル状態へ移しました。begin・halo 受信・実行・end に同じ状態を明示的に渡します。
HIP の経路選択も呼び出し単位なので、別の呼び出しの選択や配列を上書きしません。
ネイティブアーカイブの寿命は引き続き C の begin/end で管理します。

`ProjectorArchiveLayout` は、ローカル原子と受信原子それぞれの近接行オフセットと
原子ごとの開始行を所有します。`HvnaLayout` は両配置表、射影子の磁気成分数、
`[species, projector]` のエネルギー表をまとめて C++ RAII で管理します。
元の多重ポインタ表と原子マップは C ABI 境界で読み取り、所有済みの基底・射影子情報と
照合して平坦な配列へ写します。エネルギー表は mdspan で参照できます。
生成後に入力や VNA 前処理を破棄しても所有配列は有効です。生成に GPU の初期化は不要です。
原子・化学種の範囲、近接行の int 上限、float アーカイブのバイト数のオーバーフローを検査し、
失敗時は例外を C 境界で止め、既存の出力ビューを変更しません。

GPU 上またはホスト上の大きな射影子アーカイブは、従来のネイティブ確保を維持しました。
メタデータの生成は既存の集団的な容量判定後に行います。CUDA の常駐・再アップロード・
小分け復旧、HIP のランク順次実行と容量復旧、ホスト RAM の上限と MPI 同期位置は同じです。
数値収縮の加算順序も維持しています。CUDA/HIP 各14関数の本体が比較元と一致し、
さらに CUDA 2関数・HIP 4関数は静的状態への参照を引数に置き換えただけです。
3経路の容量判定も照合済みです。共通ヘッダーの大域オブジェクトは1,191個で追加はありません。

### 第13段階の検証

新しい5件のコア試験で、順番を入れ替えた原子マップ、異なる基底数・近接数、
ローカル/halo の境界、空ランク・ゼロ射影子、入力破棄後の寿命、複数所有者の独立性、
不正入力・整数上限、全確保位置での失敗と解放を確認しました。通常 CTest 79件、
ASan/UBSan のコア67件、CUDA 構成のコア67件が通過しています。

CUDA のアーカイブ復旧試験は単精度/倍精度の46条件と空ランク・空 halo を通過しました。
HIP のホスト試験は実際の C++ 所有者をリンクし、前処理を直後に破棄してから実行します。
2組のコンテキストの同時保持、デバイス確保失敗時の CPU 収縮、ランク順次実行、
ホスト RAM 上限、両数値アーカイブの確保失敗と集団的な切替、環境変数の経路選択を
ASan/UBSan 付きで確認し、収縮結果は参照値と一致しました。
この HIP 試験を HIP 構成の CTest にも登録しました。

標準 S 全14例を CPU/GPU それぞれ4 MPI × 1 OpenMPで実行し、全28計算が完走しました。
同梱参照との差はエネルギー `4.81e-11 Ha` 以下、Force `7.80e-11 Ha/bohr` 以下です。
CPU は第12段階と全例で出力桁まで一致し、GPU の前回との差はエネルギー
`3.40e-11 Ha` 以下、Force `2.60e-11 Ha/bohr` 以下でした。CPU/GPU 間も
エネルギー `4.81e-11 Ha` 以下、Force `8.00e-12 Ha/bohr` 以下です。
原子種・座標・FFT 格子も照合しています。スイートはコアを分けた並行実行のため、
その時間を性能値には使いません。

Cr2 は4 MPI × 2 OpenMPで通常 GPU、`OPENMX_SETPRO_GPU=0`、
`OPENMX_GPU_CAPACITY_TEST_RECOVERY=1` の3経路を実行しました。すべて同梱参照と
一致する許容範囲に収まり、CPU 切替は通常 GPU と出力桁まで一致しました。
強制復旧は実際にホストアーカイブを使い、通常 GPU とのエネルギー差0、
Force 差 `1.00e-12 Ha/bohr` 以下で、空の MPI ランクを含めて完走しました。

`sidia333_nc_cluster.dat` は18 MPI × 1 OpenMPで、ほかの計算・ビルドと重ねずに
実行しました。RTX 5080＋MPS は103.211秒で約105秒の目標を維持し、CPU は752.036秒でした。
CPU のエネルギーと全 Force 成分は第12段階と出力桁で一致しました。
GPU はエネルギー差0、Force 差 `1.01e-12 Ha/bohr` 以下です。保存済みの Intel 版は
800.460秒で、CPU 性能も比肩しています。Intel 版との数値差はエネルギー
`7.30e-11 Ha` 以下、Force `1.23e-10 Ha/bohr` 以下でした。
いずれも単発の比較であり、速度向上の主張には使いません。

`cmake --install build/cuda` は成功し、`work/openmx` と `work/lib/libopenmx_core.so` が
CPU/GPU の性能測定と標準スイートに使ったビルドと SHA-256 で一致しました。
入力・条件・全物理量・ログと検証一覧は `build/hvna-layout-migration/validation.json`、
時間の内訳は `docs/PERFORMANCE_REGRESSION.md` に記録しています。
Radeon、MI300、A100/H100 の実機、L/L2/L3 全例、restart、MD/NEB、多ノードの再実行は
今回の確認範囲には含みません。

幾何・Fourier 表の大域入力、HIP の GPU 使用診断フラグ、最終出力の多重ポインタ、
ネイティブ GPU バッファの begin/end 所有は残っており、次の移行対象です。

## 第14段階：HVNA の CPU 収縮とワーカー作業領域の共通化

比較元は `6227226a` です。CUDA/HIP で重複していたローカル原子と受信原子の
CPU 収縮を `source/openmx_hvna_cpu.h` にまとめました。`HvnaProjectors` は基底軌道数、
射影子の磁気成分数、`[species, group]` のエネルギー表を所有し、GPU の `HvnaLayout` も
同じクラスを使います。元の前処理とエネルギー入力の寿命から独立しています。

`HvnaWorkspace` は種ごとの最大実軌道数を辺とする連続した正方行列を所有します。
C++ 側は mdspan、C 側は平坦なビューを使い、旧 `NLH` の行ポインタと行別の確保を
除去しました。受信処理では1組を再利用し、ローカル処理ではワーカーごとに確保します。
数値ループは借用ビューを直接参照し、原子対ごとの確保や C++ 境界の呼び出しはありません。
CPU の所有者は GPU 経路を使わないときに作り、GPU 実行時の不要な `NLH` 確保も除きました。
生成と破棄に GPU の初期化は必要ありません。確保途中の失敗は RAII で解放し、
C ABI は完了前に出力ビューを変更しません。

原子分配・近接表・交差表・距離・カットオフ・原子別計時は `SetProHvnaGeometry` で
明示的に渡します。これらの寿命はまだアプリケーションが管理する借用ビューです。
CPU 収縮内から `List_YOUSO`、`Spe_Total_NO`、`VNA_List` とエネルギーの大域表を直接読む
必要はなくなりました。近接項・射影子群・磁気成分の加算順序と、単精度積を double へ
変換する位置を維持しています。受信時の行方向の並列化とローカル原子の分配も維持します。

旧処理は `k==0` で一時行列へ代入するため、その項が交差表で除外されると未初期化値や
前の原子対の値へ加算し得ました。新処理は最初の有効項で代入し、共通項が一つもない
場合はゼロを出力します。通常の先頭項が有効な場合は旧処理と同じ演算順序です。

HIP の GPU 診断値は呼び出し単位のコンテキストで保持し、`Set_ProExpn_VNA_WithStatus` から
DFT のローカル変数へ返します。既存の `Set_ProExpn_VNA` は互換入口として残しました。
診断値の判定方針を保ち、バッチ終了で状態を消す前に回収します。CUDA/HIP の未使用の
ワーカーランク変数も除去し、今回の可変 static オブジェクトの削減は計3個です。
両 `Set_ProExpn_VNA.c` にファイル範囲の可変 static は残っていません。
HIP の `DFT.c` は改修済みソースとして `integrated_overrides` に移し、取り込み元の
不変スナップショットとのハッシュ比較から分離しました。

### 第14段階の検証

新しい5件のコア試験は、可変長基底、mdspan のストライド、元入力・前処理・射影子表を
破棄した後の寿命、独立した作業領域、空基底・ゼロ射影子、C ABI の不正入力と出力保護、
メタデータと作業領域の全確保位置での失敗・解放を確認します。通常 CTest 86件、
ASan/UBSan のコア72件、CUDA 構成のコア72件が通過しました。

`tests/fixtures/hvna_cpu_before_raii.c` に比較元の CPU ループを固定し、
`tests/run_hvna_workspace_cpu.py` で本番の共通 CPU 処理と照合しています。
s/p/d・空基底、射影子の角運動量0/2/14とゼロ射影子、1/2/8スレッド、
0/1/3/6ローカル原子、単精度積/倍精度積を組み合わせ、float/double の DS 格納型で
各384条件、計768条件が未使用の出力領域を含めて完全一致しました。
受信スロットを直後に NaN で汚し、元の基底数・射影子番号・エネルギーも変更してから
所有済みメタデータで実行しています。計時の加算と、両積精度が異なる値を返すことも
確認します。先頭・全交差項を除いた各32条件、計64条件は独立した参照計算とも一致します。
同じ試験が C++ 所有者まで ASan/UBSan を付けた HIP ソース選択でも通過しました。

HIP のネイティブ HVNA ホスト試験も ASan/UBSan で参照値と一致し、
メモリ不足時の CPU 切替、ランク順次実行、RAM 上限、独立したコンテキストを確認しました。
診断値の入口・出口は本番ドライバーを使う192呼び出しで検査し、1/2/4/8スレッドでの
独立性、CPU 呼び出しでのリセット、終了処理前の値の回収と互換入口を確認しました。
CUDA の18関数・HIP の20関数の本体は比較元と一致し、HIP の別の2関数は診断値の保存先を
引数のコンテキストに変えただけです。GPU の数値式・メモリ予算・容量復旧の演算方針は
維持しています。共通ヘッダーの大域オブジェクトは1,191個で、新規追加はありません。

標準 S 全14例を CPU/GPU それぞれ4 MPI × 1 OpenMPで実行し、全28計算が完走しました。
同梱参照との差はエネルギー `4.81e-11 Ha` 以下、Force `7.80e-11 Ha/bohr` 以下です。
CPU は第13段階と全例で出力桁まで一致し、GPU の前回との差はエネルギー
`9.01e-12 Ha` 以下、Force `1.01e-12 Ha/bohr` 以下でした。CPU/GPU 間も
エネルギー `4.81e-11 Ha` 以下、Force `9.00e-12 Ha/bohr` 以下で、
原子種・座標・FFT 格子を照合しています。標準スイートはコアを分けた並行実行のため、
その時間を性能値には使いません。

Cr2 は4 MPI × 2 OpenMPで通常 GPU、`OPENMX_SETPRO_GPU=0`、
`OPENMX_GPU_CAPACITY_TEST_RECOVERY=1` を実行しました。全経路が同梱参照との照合を通過し、
CPU 切替は通常 GPU と出力桁まで一致しました。強制復旧も実際にホストアーカイブを使い、
通常 GPU とのエネルギー差0、Force 差 `1.00e-12 Ha/bohr` 以下で、空 MPI ランクも完走しました。
CUDA 構成の新しい5件のコア試験は、GPU を不可視にしても通過しました。

`sidia333_nc_cluster.dat` は18 MPI × 1 OpenMPで、ほかの計算・ビルドと重ねずに
実行しました。RTX 5080＋MPS は103.244秒で約105秒の目標を維持し、CPU は749.603秒でした。
CPU のエネルギー・全 Force 成分・座標・FFT 格子は第13段階と一致しています。
GPU はエネルギー差0、Force 差 `1.01e-12 Ha/bohr` 以下でした。
CPU の `Set_ProExpn_VNA` 最大時間は25.868→24.308秒で、保存済みの Intel 版全体800.460秒に
比肩する性能を保っています。Intel 版との数値差はエネルギー `7.30e-11 Ha` 以下、
Force `1.23e-10 Ha/bohr` 以下でした。いずれも単発の比較で、速度向上の主張には使いません。

`cmake --install build/cuda` も成功し、`work/openmx` と `work/lib/libopenmx_core.so` が
性能測定・標準スイートに使ったビルドと SHA-256 で一致しました。
入力・条件・全物理量・ログと検証一覧は `build/hvna-cpu-migration/validation.json`、
時間の比較は `docs/PERFORMANCE_REGRESSION.md` に記録しています。
Radeon、MI300、A100/H100 の実機と L/L2/L3 全例、restart、MD/NEB、多ノードの再実行は
今回の確認範囲には含みません。

幾何・Fourier 表と MPI の進捗配列の大域入力、最終出力の多重ポインタ、
通信バッファの手動確保とネイティブ GPU バッファの begin/end 所有は次の移行対象です。

## 第15段階：VNA の一括所有と射影子通信の横断的な移行

比較元は `842eca7d` です。今回の単位は VNA 内だけに閉じず、CUDA/HIP の
`Set_ProExpn_VNA`、`Set_Nonlocal`、`Force_HNL`、`Force4B`、`Force_CoreHole` と、
共通の `Stress4B`、`Stress_HNL`、`Set_CoreHoleMatrix` の計13関数を対象にしました。

`ProjectorExchange<float/double>` は進捗とメッセージ寸法を `[4, peers]` の連続配列で
所有し、送信用・受信用の数値バッファを独立に保持します。C++ は mdspan、C は明示的な
ビューを使います。必要容量が増えたときだけ確保するため、通信ループ内の38か所の
確保と対応する解放を除去しました。送信中に受信バッファが拡張されても送信データの
ポインタと値を維持し、各 `MPI_Wait` の後で送信領域を再利用します。
communicator は C アダプターへ明示的に渡し、コアの生成・破棄は MPI/GPU を初期化しません。
MPI リクエスト自体は C 側の同期処理が保持し、終了前に完了させます。

`F_Snd_Num_WK` と `F_Rcv_Num_WK` の大域宣言・起動時の確保・終了時の解放を削除し、
全利用箇所を呼び出し単位の状態へ移しました。各関数内で既に同名のローカル配列を
使っていた `Snd_DS_NL_Size`、`Rcv_DS_NL_Size` の未使用の大域配列も削除しました。
共通ヘッダーの宣言は1,191→1,187個で、新規の大域オブジェクトはありません。
既存の通信先リストや原子分配そのものは、まだアプリケーションの借用入力です。

`AtomPairPlan` はローカル原子ごとの CSR オフセット、順序付き原子対、原子・化学種の
番号と、正向き・逆向きそれぞれの座標差を所有します。逆向きの座標差は旧コードと同じ
`a-b-t` の順で計算し、正向きの値の符号反転による丸め順序の変更を避けます。
元の行ポインタや座標表を破棄した後も有効です。原子・並進番号、不正な近接数、
非有限の座標、int を超えるタスク数を検査し、空ランクは幾何表がなくても構築できます。
座標差は `[pair, orientation, xyz]` の mdspan で参照できます。

`OpenMXVnaInvocation` は基底・射影子の前処理、原子対計画、波数を RAII でまとめて所有し、
DS_VNA、VNA2、VNA3 の全段階が同じスナップショットを使います。
両バックエンドに重複していた三つの処理手順を `source/openmx_vna_assembly.h` に統合し、
原子対表の3回の構築も1回にしました。VNA2 の対角ブロックへの加算順序も保ちます。
CPU の DS_VNA と VNA2/3 は、基底と幾何の所有ビューおよび `SetProVnaTables` の明示的な
Fourier 入力・出力ビューから読み、大域の座標・原子マップ・基底配列を直接参照しません。
Fourier 表と最終出力の多重ポインタは、まだ所有元を移していない借用境界です。

VNA の MPI 処理は寸法通知→データ転送の二段階と、受信直後の CPU 収縮/GPU アーカイブを
共通化しました。MPI の int に収まらない寸法や受信寸法の不一致は、格納前に検出します。
CUDA/HIP のディスパッチアダプターは、常駐アーカイブ、ランク順次実行、容量復旧と
各デバイスの方針を保持します。C ABI は失敗時に既存の出力ビューを変更せず、
構築途中の確保とバッファ拡張の失敗を RAII で処理します。

幾何スナップショットは原子対当たり整数5個と double 6個、原子ごとの CSR オフセットを
保持します。従来のタスク番号2個に比べてメタデータ容量が増えるため、
将来の大きな系の検証でも、その分を含めてホストメモリ使用量を確認する必要があります。
GPU の大きな数値アーカイブとその容量判定はネイティブ実装の管理を維持しています。

### 第15段階の検証

新しい10件のコア試験は、可変長・順序を入れ替えた原子対、CSR/mdspan の配置、
正逆の演算順序、入力破棄後の寿命、空ランク、不正な番号・寸法・非有限値、
float/double の独立したバッファ、全構築位置での確保失敗と拡張失敗時の出力保護を
検査します。通常 CTest は98件、ASan/UBSan と CUDA 構成のコアは各82件が通過しました。
CUDA 構成の新しい10件は、GPU を不可視にしても通過しています。

本番の VNA リング処理を抽出した MPI 試験は、`842eca7d` の処理を固定した参照と
全送受信イベント・待機の順序・受信値を比較します。1/4 MPI、float/double、
不均等な原子分配・空ランク、空の基底・射影子、CPU 収縮/アーカイブを確認しました。
受信スロットは使用直後に NaN で汚し、次のメッセージに前の値が残る誤りを検出します。
別の実 MPI 試験では、送信中に受信バッファを拡張し、大きなメッセージ、ゼロ件、
独立した communicator と繰り返しの生成・破棄も確認しました。

共有 CPU 積分の再検証は、DS_VNA の float/double 計216条件が凍結参照と完全一致し、
VNA2/3 の解析解90条件も最大誤差 `2.23e-16` 以下でした。いずれも所有者を含めた
ASan/UBSan 構成でも通過しました。HVNA は旧収縮との計768条件の完全一致と、
先頭/全交差項がない64条件の検査を通過しています。
HIP のネイティブ HVNA ホスト試験も ASan/UBSan で参照値との差0を維持し、
容量不足の CPU 切替、ランク順次実行、RAM 上限、環境変数の選択を確認しました。
共有ドライバーの192呼び出しで、終了処理前の診断値の回収と呼び出し間の独立性も検査します。

VNA 以外の11関数は、所有操作と進捗の変数名を除いた数値処理・MPI 呼び出しのトークンが
比較元と一致しました。VNA のネイティブ補助関数も CUDA 28個、HIP 42個の本体が
比較元と一致し、GPU の数値式とデバイスごとの実行方針は維持しています。
記録は `build/projector-exchange-migration/preservation.json` に保存しました。

標準 S 全14例を CPU/GPU 各4 MPI × 1 OpenMPで実行し、全28計算が完走しました。
同梱参照との最大差は、CPU がエネルギー `4.61e-11 Ha`、Force `7.80e-11 Ha/bohr`、
GPU がエネルギー `2.90e-10 Ha`、Force `7.80e-11 Ha/bohr` 以下です。
前段階との差は CPU がエネルギー `1.01e-12 Ha`、Force `2.01e-12 Ha/bohr`、
GPU がエネルギー `3.36e-10 Ha`、Force `3.71e-11 Ha/bohr` 以下でした。
原子種・座標・全 Force 成分・FFT 格子を照合しています。スイートはコアを分けた
並行実行のため、その時間を性能値には使いません。

Cr2 は4 MPI × 2 OpenMPで通常 GPU、`OPENMX_SETPRO_GPU=0`、
`OPENMX_GPU_CAPACITY_TEST_RECOVERY=1` を実行しました。全経路が完走して同梱参照と一致する
許容範囲に収まり、CPU 切替の出力は通常 GPU と完全一致しました。強制復旧も実際に
ホストアーカイブを使い、通常 GPU とのエネルギー差0、Force 差 `1.00e-12 Ha/bohr` 以下です。
Ndia2 は `scf.stress.tensor on`、4 MPI × 2 OpenMPで旧版 CPU・新版 CPU/GPU を実行しました。
Stress 全9成分と Force は旧版と出力桁で一致し、エネルギー差は CPU が0、
GPU が `4.01e-12 Ha` 以下でした。いずれも空 MPI ランクを含みます。

`sidia333_nc_cluster.dat` は18 MPI × 1 OpenMPで、ほかの計算・ビルドと重ねずに
実行しました。RTX 5080＋MPS は103.609秒で約105秒の目標を維持し、MPS サーバーへの
18クライアントの接続も確認しました。CPU は745.277秒で、前段階749.603秒と
保存済み Intel 版800.460秒に比肩する性能を保っています。
前段階との最大差は、CPU がエネルギー `7.41e-11 Ha`、Force `1.23e-10 Ha/bohr`、
GPU がエネルギー `1.03e-12 Ha`、Force `1.01e-12 Ha/bohr` 以下でした。
Intel 版との差もエネルギー `1.03e-12 Ha`、Force `1.35e-10 Ha/bohr` 以下です。
原子種・座標・FFT 格子と25回の SCF 反復を照合しています。いずれも単発の比較で、
時間差を今回の変更だけによる速度向上とはみなしません。

`cmake --install build/cuda` は成功し、`work/openmx` と `work/lib/libopenmx_core.so` が
標準スイートと CPU/GPU の性能測定に使ったビルドと SHA-256 で一致しました。
全検証・入力とソースのハッシュ・物理量の照合は
`build/projector-exchange-migration/validation.json`、処理別時間は
`docs/PERFORMANCE_REGRESSION.md` に記録しています。
Radeon、MI300、A100/H100 の実機、L/L2/L3 全例、restart、MD/NEB、多ノードと
CoreHole 入力の再実行は今回の確認範囲には含みません。CoreHole の通信は
数値処理・MPI 呼び出しの保存を検査し、本体ソースを再ビルドしました。

次は Fourier 表・最終出力の所有移行、アプリケーション全体の原子分配・近接表の
オブジェクト化、ネイティブ GPU 資源の寿命管理と Kokkos 実行経路への移行が必要です。

## 第16段階：VNA 最終行列の所有と計算経路の大域依存の除去

比較元は `c26ffcc3` です。今回は最終出力の寿命まで対象を広げ、VNA の構築、
DFT、Hamiltonian、全エネルギー、Force、Stress と、通常計算・標準スイート・NEB・
数値微分検査・外部 API の呼び出し経路を一括して変更しました。

### 行列モジュールと計算コンテキスト

`AtomBlockLayout` は原子別 CSR オフセット、ブロックの行数・列数、数値領域の
オフセットを所有します。`AtomBlockMatrix<Scalar>` は成分ごとの値を1本の連続領域に
格納し、各ブロックを `layout_right` の mdspan で返します。ホストの数値型はテンプレートで、
`std::mdspan` が使える環境ではそちらを使い、それ以外は Kokkos/mdspan を使います。
メタデータは入力と独立して保持し、移動後の空レイアウトも検査可能です。
行列本体のコピー・移動・代入は許可せず、借用ビューのアドレスを安定させます。

`OpenMXCoreContext` が `VnaMatrixOwner` を所有します。未収縮の `HVNA`、`HVNA2`、
`HVNA3`、`DS_VNA` の4種と、必要な場合だけ収縮側の `CntHVNA2`、`CntHVNA3`、
`CntDS_VNA` の3種を生成します。各種類の値はそれぞれ1回の確保で保持します。
`HVNA` が使う `M2G` と、残りの行列が使う `F_M2G` の区別、原子0のダミー、
4成分の並び、最後の受信スロット、収縮時も維持する受信容量を保っています。
`DS_VNA` の永続形式は従来と同じ float です。独立した CPU 評価器は float/double の
両方を検査します。

ネイティブカーネルと未移行の C ループには、4本の平坦なポインタ表からなる借用アダプターを
渡します。アダプターは数値値を所有せず、個々の行を確保・解放しません。
CPU の DS_VNA、HVNA 収縮、VNA2/3 と、Hamiltonian・エネルギー・Force・Stress の
`HVNA` 参照は、オフセットと行ストライドを持つ平坦なビューへ変更しました。
数学的な積・加算の順序、単精度積を double に変換する位置は維持しています。

旧 `truncation.c` と `Free_Arrays.c` に重複していた確保・二つの解放処理を除去しました。
終了時と MD の原子再分配時の解放は、古い原子数・基底数・近接表を読みません。
確保途中の失敗は RAII で解放し、C ABI に例外を出しません。新しい所有者の構築が
成功してから置き換えるため、失敗した再確保は既存の行列と出力ビューを維持します。
負の寸法、不正な原子・化学種、受信容量不足、成分・オフセットのオーバーフローも検出します。

7個の大域宣言を削除し、Clang の構文木による共通ヘッダーの数は1,187→1,180個でした。
新しい大域オブジェクトはありません。これは `openmx_common.h` の範囲の数であり、
アプリケーション全体の大域変数がゼロになったことを意味しません。

### 明示的な呼び出しとバックエンドの境界

`truncation` と `DFT` は計算コンテキストを受け取り、DFT から VNA 行列ビューを
Hamiltonian、Force、Stress、全エネルギーへ渡します。Force 内の融合収縮、GPU 結果の
集約、ストリーミングとランク順次実行にも引数で伝播します。
隠れた既定コンテキストや、古い変数名を大域オブジェクトへ展開するマクロは置きません。

`Runtest`、`neb_run`、`Check_Force`、`Check_Stress` も同じ経路です。
外部 `openmx_api` の入口の引数は維持し、内部で `openmx_core_create/destroy` を使って
VNA 出力用の計算コンテキストを明示的に所有します。通常の API は従来の stateless な
radial 入出力・解放を保ち、呼び出すたびに Kokkos を初期化・終了しないようにします。
VNA の所有者は旧配列を解放した後に破棄でき、NEB 入口にも明示的に渡します。
Kokkos を使うライブラリ呼び出しは、複数入力を処理する間同じコンテキストを保持する必要が
あります。内部の DFT/VNA 等の C 関数は新しい明示引数を要求します。

CUDA/HIP の VNA ネイティブ補助関数の本体は変更していません。Force、Stress、
Hamiltonian、エネルギー、DFT の計461関数についても、引数と借用アドレス表現の置換を
除いた数値処理・MPI 呼び出しのトークンが比較元と一致しました。
GeForce、A100/H100、Radeon/MI300 のメモリ予算・ランク順次実行・容量復旧の方針は
維持しています。HIP の改修済み Hamiltonian と共通の数値微分検査は manifest の
統合済みソースへ移し、取り込み元の不変スナップショットとは区別しています。

### 第16段階の検証

新しい9件のコア試験は、異なる原子順序・可変長行列・mdspan の行ストライド、
全7種類の要素数と旧ビューとの同一領域、入力破棄後の寿命、空 MPI ランク、空基底・
ゼロ射影子、MD に相当する再確保・収縮モード変更を確認します。
22種類の不正入力、サイズのオーバーフロー、全構築位置での確保失敗と既存値の保護、
終了済みコンテキスト、独立した4スレッドの計算所有者も検査します。
テスト用の確保カウンターは、この並行検査で競合しないよう atomic にしました。

通常 CTest は107件、ASan/UBSan と CUDA 構成のコアは各91件が通過しました。
CUDA 構成の新しい9件は、GPU を不可視にしても通過します。
DS_VNA の float/double 計216条件は凍結参照と完全一致し、VNA2/3 は解析解90条件で
最大差 `2.23e-16` 以下でした。HVNA は旧収縮との768条件の完全一致と、
先頭/全交差項がない64条件を確認しました。平坦な出力ビューを使う評価器と C++ 所有者を
含めた ASan/UBSan 構成でも同じ比較を通過しています。

実 MPI の1/4ランク試験は、float/double、不均等な分配と空ランクについて、
VNA の全転送イベント・待機の順序・受信値を凍結参照と比較します。
受信した平坦なビューも検査し、消費直後に値を NaN で汚します。
GPU 使用診断の192呼び出し、CUDA/HIP 双方の Force4B ホスト試験、
HIP の HVNA ネイティブホスト試験も通過しました。ストリーミング、ランク順次実行、
上限と確保失敗時の CPU 切替を検査し、HIP の HVNA 参照値との差は0でした。
外部 API のソースも、ネイティブ Makefile のコンパイラ・ヘッダーでコンパイルしました。

標準 S 全14例は CPU/GPU 各4 MPI × 1 OpenMPで実行し、全28計算が完走しました。
CPU のエネルギー・全 Force 成分・座標・FFT 格子は前段階と全例で出力桁まで一致しました。
GPU の前段階との差はエネルギー `3.44e-10 Ha`、Force `3.81e-11 Ha/bohr` 以下でした。
同梱参照との差は CPU がエネルギー `4.61e-11 Ha`、GPU が `5.40e-11 Ha` 以下で、
両経路の Force は `7.80e-11 Ha/bohr` 以下です。CPU/GPU 間の最大差もエネルギー
`4.81e-11 Ha`、Force `9.00e-12 Ha/bohr` 以下でした。
CO は収縮軌道を使う3ステップを完走し、同一計算コンテキスト内での行列の再構築と解放を
実際の計算で確認しました。スイートの並行実行の時間は性能比較には使いません。

Ndia2 は Stress 有効、4 MPI × 2 OpenMPで、CPU/GPU とも前段階の同じ経路と
エネルギー・Force・Stress 全9成分が出力桁で一致しました。
Cr2 の4 MPI × 2 OpenMPも通常 GPU、`OPENMX_SETPRO_GPU=0`、強制容量復旧の3経路が
完走し、同梱参照の許容差内でした。CPU 切替は通常 GPU と出力桁まで一致し、
強制復旧は実際にホストアーカイブを使って、エネルギー差0、Force 差 `1.00e-12 Ha/bohr`
以下でした。これらは空 MPI ランクも含みます。

`sidia333_nc_cluster.dat` は18 MPI × 1 OpenMPで、ほかの計算・ビルドと重ねずに
実行しました。RTX 5080＋MPS は103.974秒で約105秒の目標を維持し、
18クライアントの接続も記録しました。CPU は747.408秒で、前段階745.277秒と
保存済み Intel 版800.460秒に比肩する性能を保っています。
CPU は前段階とエネルギー・全 Force 成分・座標・FFT 格子が出力桁で一致し、
GPU はエネルギー差0、Force 差 `1.01e-12 Ha/bohr` 以下でした。
Intel 版との差もエネルギー `1.03e-12 Ha`、Force `1.35e-10 Ha/bohr` 以下です。
全経路で同じ216原子・140³格子・25 SCFを照合しています。
`Set_ProExpn_VNA` の最大時間は CPU が24.538→24.066秒、
GPU が7.558→7.583秒でした。単発の測定であり、
全体時間の差を今回の変更だけによる速度向上・低下とはみなしません。

VNA 無効の H2 も4 MPIで CPU/GPU とも前段階の同じ経路とエネルギー・Force・座標・
FFT 格子が出力桁で一致しました。VNA 行列の未確保状態を DFT が渡す経路を確認しています。
H2 の NEB は中間像2個・最適化1ステップ・4 MPIで、旧版 CPU・新版 CPU/GPU を実行し、
端点を含む4像のエネルギー・NEB Force・座標と出力された格子情報が一致しました。
外部 API は実際の本体オブジェクトへリンクした検査プログラムで、VNA 有効→無効→有効の
3計算を同じ4 MPIプロセス内で完走しました。最終結果は通常の CLI の CPU 計算とも
エネルギー・全 Force 成分・座標・FFT 格子が出力桁で一致しています。

`cmake --install build/cuda` は成功し、`work/openmx` と `work/lib/libopenmx_core.so` は
CPU/GPU の測定と標準スイートに使ったビルドと SHA-256 で一致しました。
全検証・ソースと入力のハッシュ・物理量の照合は
`build/vna-matrix-migration/validation.json`、時間の比較は
`docs/PERFORMANCE_REGRESSION.md` に記録しています。CRLF の旧ソース3ファイルは
元の改行を保ち、Git の whitespace 属性でも正しく扱うようにしました。
Radeon、MI300、A100/H100 の実機と HIP コンパイラは利用できず、実機再検証は含みません。
L/L2/L3 全例、多ノード、restart、CoreHole、NVE/NVT 等の MD 全手法と本格的な NEB 収束は
今回の確認範囲には含みません。大きな系でのホストメモリ使用量も引き続き検査が必要です。

幾何・Fourier 入力とその他の行列・SCF 状態の大域所有、未移行 C/GPU 消費側の
借用ポインタ表、ネイティブ GPU 資源の所有と Kokkos 実行経路への移行は残っています。

## 第17段階：Fourier 入力の明示化と restart・MPI 行配布の共通化

比較元は AMD 側の並列化・容量制御の更新を含む `e8e6af27` です。
この更新の `Mixing_H.c`、共通 GPU ベクトル処理、HIP Hamiltonian とその回帰試験は
変更せず、Fourier 入力から重なり・非局所積分、VNA、Wannier、Force・Stress までを
一括して変更しました。

### 所有者と平坦なビュー

`Spe_RF_Bessel`、`Spe_NLRF_Bessel`、`Projector_VNA`、`VNA_proj_ene`、
`Spe_VNA_Bessel`、`Spe_CrudeVNA_Bessel`、`Spe_ProductRF_Bessel` の7個の大域宣言を
削除しました。Clang の構文木で共通ヘッダーは1,180→1,173個で、追加した大域変数は
ありません。アプリケーション全体の大域状態の除去は引き続き必要です。

数値領域は既存の `PaoStorage`、`VpsStorage`、`NeutralProjectorStorage`、
`ProductStorage` の連続領域を使い、二重に保持しません。
`openmx_core_reciprocal_get` は計算コンテキストから、寸法を持つ平坦なビューと
未移行の入出力用の借用行アダプターを返します。確保も Kokkos の初期化も行わず、
不正なコンテキストや終了済み状態では出力を変更しません。
参照は各所有者の release、計算の finish、コンテキストの破棄で無効になります。

軌道積表は化学種ごとの軌道オフセットと対の行番号を所有し、
上三角の可変長行を平坦な数値領域に対応させます。s/p/d 基底、空の角運動量群、
空の化学種と、異なる軌道数を扱います。CPU の PAO・非局所射影子の補間、DS_VNA、
VNA2/3 と Force・Stress の射影エネルギー参照は多重ポインタをたどりません。
DFT が入力ビューを取得して明示的に渡し、内側の原子対・軌道・波数ループでは
コンテキストの検索や確保を行いません。Wannier の射影重なりにも計算状態を渡します。

`Allocate_Arrays(6/7)` と `Free_Arrays` に残っていた PAO/VPS・VNA・軌道積表の
null コンテキスト用の手動確保・解放を削除しました。残る実空間の `Spe_*` 参照は
従来通り同じ所有者からの借用ですが、Fourier 表を大域変数へ再公開しません。
内部の `readfile` は明示的なコンテキストを必須にしました。

外部 `openmx_api` の3引数は維持します。`openmx_core_create_host` が入力・Fourier 表・
VNA 行列を所有し、radial 計算には stateless なホスト実装を使います。
Kokkos が未初期化、外部から初期化済み、または終了済みのどの状態でも、
この所有者は Kokkos の初期化・終了を行いません。通常の CLI は従来のランタイム方針を
保ちます。CPU の通常計算を一律に stateless 実装へ切り替えたものではありません。

### restart と MPI 配布

`ReciprocalArchive` は、化学種別の CSR オフセットと保存・配布する行を所有します。
数値は計算所有者から借り、入力の基底数・化学種表・ポインタ表を保持しません。
PAO、非局所射影子、VNA 射影子、粗い VNA ポテンシャル、軌道積の5種類が同じ
行プランを使います。化学種・spin・角運動量・多重度・積の行順序を維持し、
PAO/VNL は `List_YOUSO[15]`、VNA/軌道積は `GL_Mesh` 個という保存長も維持します。
新しいヘッダーやメタデータを restart ファイルへ挿入しません。

ファイルは C++ RAII で管理し、例外、短いファイル、書き込み・close の失敗でも閉じます。
読み込みは長さを先に検査し、大きな表をもう一つ確保せず各行へ直接読みます。
長さ検査後の I/O 自体が失敗した場合は途中の行が変わる可能性があり、呼び出し側で
再計算します。全ランクで読み込みの成否を合意するため、1ランクだけ読めない場合も
全ランクが同じ再計算・配布経路へ進みます。PAO の Bcast 後、軌道積の Bcast 前の
既存 Barrier と、root の順序も維持しました。

実ファイルの比較で、初回の PAO 保存先ディレクトリ作成を共通化時に落としていた箇所を
検出し、復元しました。実 MPI 試験にも、本体の保存ブロックを使って初回から全5種類を
生成する検査を追加しています。

### 第17段階の検証

新しい14件のコア試験は、平坦な添字と借用アダプターの同一領域、余白の初期化、
所有者の独立性、release・再確保、入力メタデータ破棄後の寿命、全確保位置での失敗と
出力の保護、4スレッドの独立した所有者を検査します。
restart の凍結した行順序、バイナリ形式、余分な末尾データ、切り詰め、`/dev/full`、
空表、外部が所有する Kokkos ランタイムと終了後の連続ホスト計算も確認しました。

本体 CTest は122件、ASan/UBSan と CUDA 構成のコアは各105件が通過しました。
新しい試験のうちランタイムを起動しない13件は、CUDA 構成で GPU を不可視にしても
通過しました。DS_VNA の float/double 計216条件は凍結参照と完全一致し、
VNA2/3 の解析解90条件、radial 準備・GPU パックの54条件も ASan/UBSan で通過しました。
HIP の HVNA ホスト試験は、容量不足、ホストアーカイブ、ランク順次実行、APU 判定と
CPU 切替を含め参照値との差0でした。数値・通信トークンは380関数と条件付きコンパイルの
3翻訳単位で、明示引数と平坦なアドレスへの置換を除いて一致しました。

実 MPI 試験は1/4ランク、5種類×4条件の計40計算です。非対称な化学種分配、空ランク、
空基底、spin 依存、正常・欠落・切り詰め・最終ランクだけの読み込み失敗を検査し、
全 Bcast の root・行アドレス・要素数と Barrier の位置が従来通りであることを確認しました。
実際の H2 計算でも全5種類の restart ファイルが比較元とバイト単位で一致し、
同じ旧ファイルから再開した CPU/GPU のエネルギー・全 Force 成分・座標・FFT 格子は
出力桁で一致しました。

標準 S 全14例は CPU/GPU 各4 MPI × 1 OpenMPで、最終版と今回の比較元が全28計算ずつ
完走しました。最大差は CPU がエネルギー `2.80e-11 Ha`、
Force `2.60e-11 Ha/bohr`、GPU がエネルギー
`1.50e-11 Ha`、Force `2.70e-11 Ha/bohr` でした。
全140個の Fourier restart ファイルもバイト単位で一致しました。
Ndia2 の Stress 全9成分と Force は保存済みの前段階と一致し、CO の収縮軌道の3 MDステップ、
Cr2 の CPU VNA切替と強制容量復旧、VNA 無効入力、小さな1ステップの NEB も通過しました。

`sidia333_nc_cluster.dat` は同じ216原子・140³格子・25 SCF、18 MPI × 1 OpenMPで、
ほかの計算・ビルドと重ねずに比較しました。RTX 5080＋MPS は 103.819 秒で
約105秒の目標を維持し、18クライアントの接続も記録しています。
CPU は 751.942→752.660 秒で、保存済み Intel 版800.460秒に比肩します。
比較元との差は CPU がエネルギー `4.40e-11 Ha`、
Force `1.19e-10 Ha/bohr`、GPU がエネルギー差0、
Force `1.00e-12 Ha/bohr` でした。
単発の測定であり、時間差を今回の変更だけによる速度向上・低下とはみなしません。
詳細は `docs/PERFORMANCE_REGRESSION.md`、全検証・入力とソースのハッシュは
`build/reciprocal-state-migration/validation.json` に記録しています。
`cmake --install build/cuda` も成功し、`work/openmx` と `work/lib/libopenmx_core.so` は
測定したビルドと SHA-256 で一致しました。

外部 API の VNA 有効→無効→有効の3計算は同じ4 MPIプロセス内で完走し、
通常の CLI とエネルギー・全 Force 成分・座標・FFT 格子が出力桁で一致しました。

Radeon/MI300、A100/H100 の実機と HIP コンパイラはこの環境では利用できません。
CUDA/HIP の容量制御・APU 判定・GeForce と A100/H100 の既存の実行方針は維持し、
この段階ではネイティブ数値カーネルを新しい Kokkos カーネルに置き換えていません。
L/L2/L3 全例、多ノード、CoreHole、すべての MD 手法と本格的な NEB 収束、
大規模系のピークホストメモリは今回の確認範囲に含みません。

## 第18段階：化学種の基本情報の所有と実空間入力の明示化

比較元は第17段階の `bfccc29c` です。化学種の基本情報23表を計算コンテキストの
`SpeciesStorage` へ移し、PAO/VPS の実空間表を入力解析から初期密度、Fourier 変換、
全エネルギー、Force・Stress まで明示的に渡すようにしました。

### 化学種情報の寿命

名称・基底指定・PAO/VPS ファイル名の4表、整数13表、実数4表、原始・収縮基底の
2表を C++ が所有します。整数・実数は化学種方向が連続する SoA、名称は固定容量の
連続領域、基底数は `layout_right` の `mdspan` で参照します。質量の初期値 -1 と
基底数・OpenCore フラグの初期値0を保ち、その他の余白も0で初期化します。
名称の容量と従来の C 入力形式は変更しません。

`Input_std` は明示的な計算コンテキストを受け取り、`Allocate_Arrays(0/2)` にあった
これらの表の手動確保をなくしました。確保は数値領域と借用行アダプターの全構築に
成功してから公開し、途中失敗は RAII で巻き戻します。`Free_Arrays` では、未移行の
可変長表が基底数を使い終わった最後に所有者を解放し、借用大域ポインタを null にします。
通常の終了処理を通らずコンテキストを破棄する場合も、この所有領域は解放されます。

`openmx_core_species_get` と `openmx_core_atomic_get` は確保や Kokkos の初期化を
行いません。未確保の所有者は空ビュー、終了済み・null コンテキストは失敗とし、
失敗時には出力を変更しません。取得した記述子は自動更新されず、後から別の表を
確保した場合は取り直します。参照の寿命は対応する所有者の release、finish、destroy
までで、別の所有者の確保・解放では数値領域のアドレスを変えません。

### 実空間表と参照側

`Spe_Atomic_Den`、`Spe_Atomic_Den2`、`Spe_Atomic_PCC`、`Spe_VH_Atom`、`Spe_Vna`、
`Spe_VPS_XV`、`Spe_VPS_RV`、`Spe_VNL` の8個の大域宣言を削除しました。
`OpenMXAtomicViews` は既存の PAO/VPS 所有領域と化学種情報を参照し、数値を二重に
保持しません。密度・Hartree・PCC の +2 余白と、spin・射影子・動径方向の stride を
明示します。入力の読み込み・規格化・ゴースト原子の処理も同じ平坦な領域を更新します。

DFT がビューを一度取得し、初期密度、NEGF の電極初期密度、Hamiltonian の局所
ポテンシャル構築、全エネルギー、Force・Stress へ渡します。CPU の内側のループでは
平坦なアドレスを使い、コンテキスト検索、数値表のコピー、追加確保を行いません。
CUDA と HIP のエネルギー用ホストパック、VNA 無効時の Force・Stress、Fourier の
入力側も明示的なビューを使います。未移行の変換 ABI に渡す行アダプターは同じ所有者
から借り、プロセス大域の表へ再公開しません。

`VH_AtomF` と `Dr_VH_AtomF` は遠方の電荷を引数で受け、動径行だけを参照します。
後者の原点側の分岐も、引数の `rv` と同じ行を使うようにしました。本体の既存の
呼び出しは同じ化学種の行と電荷を渡すため、計算式と補間順序を保ちます。

AMD 側の密度・混合・容量制御、GeForce と A100/H100 の実行方針、CUDA/HIP の
数値カーネル・縮約順序・通信順序を変更していません。HIP の全エネルギー専用ソースは
新しい明示引数に対応させ、更新済みファイルとしてビルド manifest に登録しました。

### 第18段階の検証

所有権・ビューの新規12試験は、全フィールドの独立性、平坦な添字と借用行の同一領域、
名称の終端と余白、release・再確保、全確保位置での失敗と巻き戻し、終了後のアクセス
拒否と出力保護、独立した4スレッドの計算を確認します。新しい Hartree 補間の試験は
異なる2格子の原点側・内部・遠方を、解析値と導関数で428比較します。

本体 CTest は135件、ASan/UBSan と CUDA 構成のコアは各117件が通過しました。
新規12コア試験は CUDA 構成で GPU を不可視にしても通過しました。
HIP 選択後の26個の C ソースも GCC のホスト構文・型・引数検査を通過しました。
これは ROCm のオフロードコード生成の検査ではありません。
CUDA/HIP のホスト実行によるエネルギー・EH0・Exc0・双極子の容量復旧試験は
各336比較、実際の OpenMP チーム数を変えるエネルギー・Stress の試験は各112条件が
ASan/UBSan で通過しました。数値・MPI トークンは534関数・条件付き翻訳単位で、
明示引数と同じ領域を指すアドレスの置換を除いて一致しました。
Clang の共通ヘッダーの構文木は1,173→1,165個で、追加した大域変数はありません。
削除した8シンボルが実行ファイルから消えたことも確認しました。

サニタイザー試験では、この環境の unlimited stack と NVIDIA OpenMP ランタイムの
探索パスによる初期化・終了時の衝突を避けるため、試験プロセスの stack を8 MiBにし、
`LD_LIBRARY_PATH` を外して GNU ランタイムを使いました。本体の実行設定は同じです。

標準 S 全14例は最終ビルドの CPU/GPU 各4 MPI × 1 OpenMPで完走し、同梱参照を
満たしました。第17段階との最大差は CPU がエネルギー `1.30e-11 Ha`、Force
`2.60e-11 Ha/bohr`、GPU が `1.30e-11 Ha`、`1.40e-11 Ha/bohr` です。
Ndia2 の Stress 全9成分は4 MPI × 2 OpenMPで CPU/GPU とも出力桁で一致しました。
CO の収縮軌道の3 MDステップ、Cr2 の CPU VNA切替と強制容量復旧、VNA 無効の H2、
中間像2個の1ステップ NEB も通過しました。外部 API の VNA 有効→無効→有効の
3計算は同じ4 MPIプロセス内で完走し、最終エネルギー・Force・座標・格子は CLI と一致しました。

Fourier restart 全140ファイルのサイズと行配置を維持し、PAO・非局所射影子・軌道積の
84ファイルはバイト単位で一致しました。VNA ポテンシャルの変換には最大 `1.24e-13`、
規格化された VNA 射影子の変換には最大 `1.40e-6`、`|差|/(1+|旧値|)` で
`4.76e-8` の差があります。後者は小さなポテンシャルの差に対する古典的
Gram–Schmidt と規格化の感度が現れたと考えられますが、原因を分離した測定ではありません。
全140表のビット一致を達成したとは扱いません。保存形式と所有領域・行プランの検査に加え、
旧版が生成した H2 の全5表から CPU/GPU で再開し、エネルギー・Force・座標・格子の
一致を確認しました。表そのものの差と最終物理量の比較は別々に記録しています。

`sidia333_nc_cluster.dat` は18 MPI × 1 OpenMPで、RTX 5080＋MPS が
103.429→103.763秒、CPU が
759.159→747.610秒でした。GPU の約105秒の目標と、
比較元・保存済み Intel 版に比肩する CPU 性能を維持しています。単発測定であり、
差を今回の変更だけの効果とはみなしません。詳細は `docs/PERFORMANCE_REGRESSION.md`、
全検証とソース・成果物のハッシュは `build/atomic-state-migration/validation.json` に記録しています。
`cmake --install build/cuda` で配置した `work/openmx` と `work/lib/libopenmx_core.so` は
測定した最終ビルドと SHA-256 で一致しました。

Radeon/MI300、A100/H100 の実機と HIP コンパイラは利用できません。
HIP のホスト回帰は実施しましたが、ROCm のデバイスコード生成・実機再検証は含みません。
L/L2/L3 全例、多ノード、CoreHole、すべての MD 手法と本格的な NEB 収束、
大規模系のピークホストメモリも今回の確認範囲に含みません。化学種の基本情報の所有を
移した段階であり、名称・基底数・電荷等の借用大域宣言、実空間 PAO 表の一部、
Hamiltonian・SCF 状態とネイティブ GPU 資源の所有、Kokkos 数値カーネルへの移行は残ります。

## 第19段階：軌道評価・導関数と非局所表の明示化

比較元は第18段階の `04e0b451` です。軌道評価の呼び出し経路を、グリッド生成、
Force・Stress、非局所 Hamiltonian、光学応答・XANES、輸送解析と出力まで移行しました。
`OpenMXAtomicViews` を引数で渡し、PAO の格子長・基底数・軌道数等も同じ記述子の
化学種情報から読みます。再利用する評価関数のグリッド点ループでは所有者の検索、
表のコピー、作業領域の確保を行いません。

### 実空間表と呼び出し境界

`Spe_PAO_XV`、`Spe_PAO_RV`、`Spe_PAO_RWF`、`Spe_VPS_List`、`Spe_VNLE` の5個の
大域宣言と Allocate/Free の別名公開を削除しました。Clang の共通ヘッダーの構文木は
1,165→1,160個で、追加した大域変数はありません。旧実行ファイルにあった5シンボルが
新しい実行ファイルから消えたことも確認しました。軌道評価関数に残っていた未使用の
関数内 `static firsttime` も7個削除しています。

PAO 表と射影子の角運動量・エネルギーは既存の C++ 所有領域を参照します。
値・導関数・収縮軌道、Voronoi、EXX、LPQ、最適化軌道の出力、SCF 行列の出力、
運動量行列、NEGF の電極軌道と電流密度へ同じ明示ビューを伝えます。
CUDA/HIP の表構築・オンザフライ評価・Force 用パックと容量復旧も対応しました。
固定シグネチャの GPU ワーカーコールバックには、その呼び出しのレコードから
記述子を借用させ、隠れた既定コンテキストや thread-local 状態を置いていません。

main、runtest、外部 API は入力で表の確保が終わってから記述子を取得します。
DFT は従来通り呼び出しごとに取得し、readfile の PAO/VPS 読み込みも確保後の
記述子を使います。所有者の release・finish・destroy で借用参照が無効になる規約は
維持しています。同一プロセス内の API 再実行でも、前の記述子を再利用しません。

非局所積分の `OpenMXNonlocalShape` は、角運動量の多重ポインタの代わりに平坦な
先頭アドレスと行 stride を受けます。射影子0の余白と化学種・射影子の順序を保ち、
短すぎる行 stride とアドレス範囲のオーバーフローを、参照・確保の前に拒否します。
既存の積分ワークスペースが持つメタデータのスナップショットは引き続き独立です。
この内部 C ブリッジの構造体は拡張しているため、利用する C コードとライブラリは
同じヘッダーで再ビルドする必要があります。

### 軌道作業領域の所有

`OrbitalWorkspace` が動径と角度の2個の連続領域を所有し、`layout_right` の
mdspan を返します。値だけの場合は RF・AF、導関数を使う場合は RF・dRF と
AF・dAFQ・dAFP を、領域内の別チャネルとして参照します。従来の角運動量・多重度の
容量と角度側の余白を保ち、確保時には全要素を0で初期化します。

スカラー・収縮軌道の評価、Set_Orbitals_Grid の CPU ワーカー、Force3・Stress3 の
再利用領域から、RF/AF 系の行ごとの malloc/free と `WORK_DORBITAL` を除去しました。
最大角運動量4の例では、値評価の12回、導関数評価の30回の小さな確保が、
ハンドルと2領域の計3回になります。再利用する導関数の評価中は確保しません。
所有者のコピー・移動を禁止して借用アドレスを固定し、全構築が成功してから C に
バッファを公開します。途中の失敗は RAII で巻き戻し、失敗した API は既存の出力を
変えません。各 OpenMP ワーカーは独立した所有者を持ちます。
C 呼び出し側の正常な境界では destroy が必要であり、MPI_Abort 等が未移行の C 配列まで
自動的に巻き戻すという意味ではありません。

### 第19段階の数値・所有権検証

新しい6コア試験は mdspan の添字と余白、チャネルの非重複、所有者の独立性、
null・不正寸法・オーバーフロー、3か所すべての確保失敗と巻き戻し、4スレッドでの
反復生成を検査します。VPS 角運動量・エネルギーの平坦な行と旧借用アダプターの
同一領域、解放後の空ビュー、非局所積分の不正 stride も既存試験に追加しました。

実際の C 評価関数は、移行前の凍結コードと独立した空間差分を合わせて224,205比較しました。
2化学種、異なる格子・基底数、原点、軸、内部、格子点と端、範囲外、収縮軌道、
g 軌道と再利用領域を含みます。従来の格子長・基底数・容量の大域値に異なる値を入れ、
新しい評価が引数のビューだけに依存することも確認しています。
既存の CPU インライン、Force3、Voronoi、収縮軌道の halo の補間試験は CUDA 側210条件、
HIP 側196条件が通過しました。CUDA の表準備の確保失敗と84条件の compact gather、
HIP の1,166補間比較・9,072軌道値、アップロード失敗35条件・準備90条件・表構築24条件も
ホスト上で確認しています。HIP のデバイス呼び出しをホストスタブに置き換えた試験は、
実機でのコード生成・実行の検証とは区別します。

本体 CTest は145件、ASan/UBSan と CUDA 構成のコアは各123件が通過しました。
CUDA 構成のうち新しい作業領域等の16件は GPU を不可視にしても通過しています。
数値・MPI トークンは55個の C ファイル、1,148関数・条件付き翻訳単位で、明示引数、
同じ領域の平坦なアドレス、作業領域の確保・解放の置換を除いて一致しました。
40個のネイティブ CUDA/HIP カーネルファイルも改修前と同じハッシュです。

HIP で選択される48個の C ファイルは、ホストの C コンパイラで型と呼び出し引数を
確認しました。CUDA/HIP のホスト容量試験は各336条件、実際に OpenMP チームを変更する
Energy・Stress の試験は各112条件が ASan/UBSan を含めて通過しています。

標準 S 全14例を CPU/GPU 各4 MPI × 1 OpenMPで実行しました。保存済みの第18段階との
エネルギー・Force の差は CPU で0、GPU で最大 `1.10e-11 Ha`・`2.60e-11 Ha/bohr` です。
同梱参照も全例で許容差内で、CPU/GPU の Fourier restart 全140ファイルは比較元と
バイト単位で一致しました。Ndia2 の Stress 全9成分、CO の3 MDステップ、Cr2 の
CPU VNA切替・容量復旧、VNA 無効化、小さな1ステップの NEB、外部 API の連続3計算、
旧版が生成した H2 restart からの CPU/GPU 再開も通過しました。

`sidia333_nc_cluster.dat` は18 MPI × 1 OpenMPで測定し、CPU は804.105→795.525秒、
RTX 5080＋MPS は2組の比較で107.352→108.263秒、108.116→108.391秒でした。
GPU の約105秒という目標には今回の測定では届いていません。CPU は保存済み Intel 版の
800.460秒と同程度ですが、Intel 版は再測定していません。時間差をリファクタリングの
効果だけとは判断しません。条件・処理別時間は [性能記録](PERFORMANCE_REGRESSION.md)、
最終ソース・成果物の SHA-256 と検証結果は `build/orbital-state-migration/validation.json`
に記録しています。インストールした `work/openmx` と共有ライブラリも検証済み成果物と一致します。

Radeon/MI300・A100/H100 の実機と HIP コンパイラは利用できず、ROCm のデバイスコード生成は
未検証です。L/L2/L3 全例、多ノード、CoreHole、全 MD 手法、本格的な NEB 収束、
大規模系のピーク RAM は今回の確認範囲に含みません。

### 独立した差分検査で確認した既存の制限

第19段階では、一般角度の g 軌道の導関数に、所有権移行前から空間差分と一致しない例を
確認しました。この問題は次節の数値修正で対応済みです。
合成入力の位置 `(0.35,-0.29,0.41)`、軌道26、x 方向では、旧版と改修版はいずれも
`0.1508452919739493`、値の空間差分からは `0.14964432628120083` でした。
一般角運動量の分岐の dAFP と、明示式の l≤3 の分岐では sin(theta) の取り込み方が
揃っていませんでした。第19段階の所有権移行ではこの式を変えず、g 軌道の一般位置は
旧版との回帰比較、独立した差分は
s/p/d/f の一般位置と g の赤道面で確認しています。全角度での g 導関数の正しさを
新たに保証したとは扱いません。再現結果は検証ディレクトリの `review-notes.md` に記録しています。

## g 軌道導関数の数値修正

比較元は `2499cf00` です。軌道の Cartesian 勾配は方位角成分に `e_phi/r` を掛けるため、
角度側で `(∂Y/∂phi)/sin(theta)` が必要です。旧 g 分岐では `ComplexSH` の未スケールの
方位角微分を渡していたため、一般角度で係数が不足していました。加えて、
`Associated_Legendre` が極で theta 微分を0にし、`acos(z/r)` が極付近で横方向の
座標を丸め落とす問題もありました。

`openmx_orbital_harmonics.h` に、軌道専用の複素球面調和関数と角度微分を集約しました。
`a=|m|` として `P_l^a(cos(theta))=sin(theta)^a T_l^a(cos(theta))` を使い、
`T` とその微分を同じ漸化式から求めます。この実装は sin(theta) で除算せず、
両極の `|m|=1` の有限で非零の導関数も返します。規格化と位相は
[NIST DLMF 14.30.1](https://dlmf.nist.gov/14.30.E1)、漸化式は
[14.10.3](https://dlmf.nist.gov/14.10.E3) に基づき、その式を因数分解して微分しました。
g を含む基底では横方向の長さと座標から角度の sin/cos を求めます。

通常・収縮軌道、正負の勾配、再利用ワークスペース、CUDA/HIP の Force3 内の
CPU 評価、両バックエンドの軌道グリッド生成に同じ処理を適用しました。
原点の動径補外、軌道の並びと収縮時の既存の位相規約は維持しています。
他の積分が使う `ComplexSH` の微分規約は変更していません。
g は従来どおり GPU 経路でもホストで軌道・導関数を評価します。

実際の C 関数と CUDA/HIP のホスト評価を合わせて473,916比較し、
うち24,630成分を独立した導関数と比較しました。一般角度、赤道、両極、
横方向の成分が `1e-14` の極付近、9個すべての g 成分、複数の動径関数、
収縮・符号反転・再利用を含みます。g の有限差分との差は最大 `1.5e-11`、
独立した Cartesian 多項式の complex-step 微分との差は最大 `4.51e-15` です。
元の再現例の修正後は `0.14964432627680258` になりました。
g を含まない基底の極付近の座標変換は今回の修正範囲に含めていません。

回帰入力は `tests/fixtures/Cl2_g_orbitals.dat` です。一般方向に置いた Cl₂ 分子と
`Cl7.0-s2p2d1f1g1` を使い、CPU/GPU 各4 MPI × 2 OpenMPでエネルギー・全 Force 成分・
座標・FFT 格子が出力桁で一致しました。導関数修正なので、旧版との Force の一致は
合格条件にせず、全エネルギーの差分との整合性も別に検査しました。

NVHPC/CUDA でビルド・インストールを完了し、CTest 全145件を通過しました。
初回に2件で ASan の shadow memory とプロセスメモリ配置が衝突したため、
その2件だけスタック上限を8 MiBにして再実行しています。
標準 S スイート14例を CPU/GPU 各4 MPI × 1 OpenMPで完走し、
同梱参照、変更前、CPU/GPU 間の全比較に合格しました。
HIP 選択ソース6本のホスト C 構文検査と共通導関数テストも通過しましたが、
ROCm コンパイラ・AMD 実機での検証は行っていません。

Cl₂ の `-forcetest2 0` は、変位幅0.0003 bohrで第1原子の3方向を調べました。
250 Ryは1 MPI × 2 OpenMP、500 Ryは4 MPI × 2 OpenMPです。
解析 Force と全エネルギー差分による Force の最大差は次のとおりです（Ha/bohr）。

| 指定 cutoff | 変更前 CPU | 修正後 CPU | 修正後 GPU |
| --- | ---: | ---: | ---: |
| 250 Ry | 1.38847e-4 | 1.19648e-4 | 1.19648e-4 |
| 500 Ry | 5.59300e-5 | 5.59300e-5 | 5.59300e-5 |

両 cutoff で修正後 CPU/GPU の解析 Force は出力桁で一致しました。
ただし、総エネルギー差分の検証用許容値 `5e-5 Ha/bohr` は未達です。
500 Ryの最大差は今回の修正で変化しない z 成分にあり、変更前から残っています。
また500 Ryの x/y 成分の差は変更前の約 `1.07e-6 / 2.06e-6` から
修正後の約 `2.17e-5 / 1.22e-5` に増えており、全成分の改善を確認したものではありません。
格子を細かくすると最大差は減りましたが、残差の原因は未確定です。
独立した解析式で確認済みの g 軌道導関数の修正と、総 Force の差分整合性は
別の検証結果として扱います。失敗した許容値判定もログに保存しています。
記録は `build/g-orbital-derivative-fix/` に保存しています。

## 次の移行単位と完了条件

1. 化学種の基本情報23表と PAO/VPS・Fourier の数値領域の所有、実空間・Fourier 表の
   大域参照の除去、軌道・導関数の明示化は移行済み。名称・基底数・電荷等の借用大域
   ポインタを除去し、Hubbard・spin-orbit 係数等の残る化学種情報、原子種・分配も所有化する。
   軌道の出力・収縮係数・解析側に残る多重ポインタも平坦な所有領域へ移す。
   MPI 配布、restart と入力解析の制御も独立したモジュールへ移す。
2. 周期セル、原子・近接リスト、MPI による原子分配を独立したオブジェクトにする。
   一時的に旧配列を参照する場合も寿命を明示し、隠れた既定コンテキストを置かない。
3. VNA 以外の Hamiltonian・重なり・密度等の可変長原子・軌道ブロックも
   「オフセット + 連続値」にし、ブロックごとに `mdspan` を渡す。
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
