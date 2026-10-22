# OpenMX 本体の C++ / Kokkos 移行

作業ブランチは `refactor/cpp-kokkos-core`。移行開始点は
`4d8c7e0f01133927dbd76ce24f0e44d2ceb2f87e` です。
この文書は完了した移行単位と、残っている作業を記録します。
OpenMX 本体全体の C++ 化、大域変数ゼロ、全カーネルの Kokkos 化はまだ完了していません。

2026-10-20 のレビュー対応では、C ABI ヘッダーを `core/include/` へ移し、
コアの `source/` 依存を除去しました。`source/` の同名ヘッダーは互換インクルードです。
本体の共通ヘッダーにあった外部変数と `g_exx` の定義計1,155個は
`source/openmx_globals.c` に一本化しました。大域状態の使用を減らす作業は継続します。
また、SDK に依存しないデバイス能力記述とコア単独の SYCL ビルドを導入し、
macOS 上の CPU 検査を CI に追加しました。修正項目・実行結果・未検証の範囲は
[レビュー対応記録](CODE_REVIEW_20261020.md) を参照してください。

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
| onsite・DC・LNO・縮約係数 | `core/src/local_basis_state.cpp` | 9系統を計算コンテキストが所有。原子ごとの不均一な行列はオフセット付きの連続領域、係数は矩形 Tensor。数値処理へ平坦な行ビューを渡し、再分配後は旧トポロジーを使わず解放する。 |
| 軌道最適化 | `core/src/orbital_optimization.cpp` | DFT 呼び出しごとに履歴・Hessian・トポロジーの写し・再利用する作業領域を所有。atoms/species、DIIS/EF、勾配・正規化を共通化し、演算対象は平坦な行列ビュー、MPI/LAPACK は明示的なサービスとして受け取る。 |
| core-hole の係数と Force のスピノル演算 | `core/src/core_hole.cpp` | Force 段階の呼び出しが基底配置・スピン・状態番号と正負の係数を所有。係数は一度生成し、全ワーカーが不変ビューを借用する。重なりの平坦な行と出力の `mdspan` を使い、SDK・MPI・大域変数に依存しない。 |
| 非局所 Force の射影子とスピノル演算 | `core/src/projector_force.cpp` | 化学種ごとの角運動量・成分オフセット・エネルギー・スピン係数を段階ごとにコピーして所有。ワーカーは不変ビューを共有し、入力・出力の行間隔を明示した矩形ブロックを共通 C++/mdspan 演算へ渡す。 |
| 計算段階の native 資源と非局所 GPU アーカイブ | `core/src/{solver_workspace,native_resources}.cpp`、`source/openmx_force_hnl_gpu.h` | 段階の呼び出しが名前付き資源グループと POD 状態を所有。ホスト配列・SDK ハンドルは独自 deleter で管理し、非局所 Force は大域の GPU コンテキストを持たない。 |
| 動径表と補間 | `core/src/radial_table.cpp` | `RadialTableView` が格子と全入力値を検証して借用する。独立したコピーが必要な場合は `RadialTable` が所有。PAO、非局所 projector、中性原子ポテンシャル、VNA 分離射影子の境界条件を明示して補間する。 |
| 波数空間の補間 | `core/src/radial_table.cpp` | 波数を `i*step` で求め、PAO/projector の共通補間へ渡す。格子配列を所有せず、`NormK` を必要としない。 |
| PAO/NLP/VNA・軌道積の Fourier 変換 | `core/src/radial.cpp` | 半径、積分の測度・方式、波数、チャネル、サンプルを明示的に受け取る。ホスト配列は `Tensor`、実行用配列は `Kokkos::View` が所有。 |
| FFT グリッド検証 | `core/include/openmx/core/fft_grid.hpp` | 正の整数を 2、3、5、7 の積に分解できるか判定。大域の可変テーブルを持たない。 |
| 周期セル逆引き | `core/include/openmx_periodic.h` | セルの半径と座標から添字を直接計算。逆引き用の所有配列を持たない。 |
| 既存 C との境界 | `core/include/openmx_core*.h`、`core/src/c_api.cpp`、`core/src/{pao,vps,product}_compat.hpp` | 不透明なコンテキストと C ABI。入力表、出力先、実行デバイス、エラー出力先を引数で受け取る。固定寸法の互換ポインタ表は共通の `legacy_rows.hpp`、可変長の軌道積表は `product_compat.hpp` で構築する。 |

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

### Intel / SYCL コア

```sh
cmake -S core -B build/core-sycl \
  -DCMAKE_CXX_COMPILER=icpx -DOPENMX_CORE_BACKEND=SYCL \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/core-sycl --parallel 4
```

コア単独の SYCL 実行空間を構築します。OpenMX 本体の Intel GPU 対応は未実装です。
選択した SYCL デバイスの FP64 能力を確認し、FP64 が使えなければ動径変換を
CPU で実行します。`Context::last_fallback()` で理由を取得でき、
`OPENMX_CORE_VERBOSE=1` では理由を一度ずつ表示します。
外部が所有する Kokkos ランタイムや、コンテキストと異なるデバイスを
要求された場合の CPU 実行も区別できます。
IntelLLVM には `-fp-model=precise -ffp-contract=off` を適用し、
公開インライン関数を使う側にも同じ演算規則を伝えます。

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

以下は初期移行の記録です。第21段階では共通ヘッダーの宣言が1,128から1,118へ、
深さ2以上のポインタ宣言が193から183へ減りました。実際の `openmx_globals.o` の定義は
1,123から1,113へ減り、今回の対象10シンボルはリンク済みバイナリからも消えています。

| 対象 | 開始時 | 第6段階時点 |
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

宣言数とは別に、`tools/audit_global_usage.py` は各翻訳単位の LLVM IR から
「どの関数がどの大域オブジェクトを読む・書く・ポインタを渡すか」を関数単位で数え、
ファイルスコープ `static`、関数内 `static`、環境変数の参照も含めて棚卸しします。
結果と、書き手に基づく所有権の分類、ゲートの使い方は
[大域状態の利用監査](GLOBAL_STATE_AUDIT.md) と、生成物の
[利用一覧](GLOBAL_STATE_INVENTORY.md) にあります。
環境変数は `source/openmx_env_registry.h` のレジストリ（名前ごとの id と種類、`openmx_env_flag_at()` などの
型付き O(1) 読み取り、MPI_Init 直後の写し取りと rank 0 からの配布）経由で読み、`getenv` の直接呼び出しと
レジストリに無い名前は同じゲートで増加を止めます（`tools/update_env_registry.py` が表を保守し、
`tools/env_registry_doc.py` が実装文書 §10.2 の表を生成します）。

## テスト

独立コアの CTest は、所有権と例外時の整合性、周期セルの番号順序・
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
アプリケーションを無効にしても、7ソルバー、Hamiltonian・密度混合の状態、密度混合の作業領域、
非共線エネルギー出力を実際のC++所有者に接続する11件の `native-state` 試験を登録します。
MPIを検出した場合は、実4 rankの部分コミュニケータを使う
DC-LNO/Krylovの2件も登録します。SDK呼び出しは模擬し、共有コアはそのビルドの実物を使います。
登録漏れ・重複・別ビルドのライブラリの使用は、次の検査で拒否します。

```sh
python3 tests/check_native_state_ctest.py --build build/core-sanitize
ctest --test-dir build/core-sanitize -L '^native-state$' --no-tests=error --output-on-failure
# MPIを検出した構成
python3 tests/check_native_state_ctest.py --build build/core-sanitize --require-mpi
ctest --test-dir build/core-sanitize -L '^native-state-mpi$' --no-tests=error --output-on-failure
```

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

## 第20段階：SCF 行列32系統の所有と大域依存の除去

比較元はレビュー修正後の `c223a7a4` です。VNA 行列に続き、主要な電子状態行列を
計算コンテキストの C++ オブジェクトへ移しました。32個の大域変数の宣言・定義を
削除し、SCF、混合、対角化、Hamiltonian 構築、エネルギー、Force、Stress、
restart、輸送、解析・出力にまたがる147関数が `OpenMXElectronicViews` を
引数で受け取り、5か所の計算境界がコンテキストからビューを取得するようにしました。
CUDA/HIP のネイティブ経路も同じ所有者を参照します。

### 所有する行列と平坦な領域

`RaggedMatrix` は component → atom → neighbor の3段の CSR と、各ブロックの
行数・列数を持ちます。有効な行列系統ごとに数値領域を1回確保し、全要素を初期化します。
ブロックは `layout_right` の `mdspan`、C 側ではオフセットと連続値のビューで参照できます。
履歴と spin は component にまとめ、局所原子・halo・作業用スロットの異なる形状を
パディングで均一化せずに保持します。

`ElectronicMatrices` が次の32系統の配置と所有を担当します。

| 分類 | 移行した配列 |
| --- | --- |
| 運動・非局所積分 | `H0`, `CntH0`, `HNL`, `iHNL`, `iCntHNL`, `iHNL0` |
| core hole・相互作用 | `HCH`, `iHCH`, `H_XMCD`, `H_Hub` |
| 重なり | `OLP_L`, `OLP`, `OLP_CH`, `CntOLP` |
| Hamiltonian・射影子 | `H`, `CntH`, `DS_NL`, `CntDS_NL` |
| 密度 | `DM0`, `DM`, `ParDM`, `iParDM`, `Partial_DM`, `EDM`, `PDM`, `iDM` |
| 混合履歴・残差 | `HisH1`, `HisH2`, `ResidualH1`, `ResidualH2`, `ResidualDM`, `iResidualDM` |

旧来の数値ループ用に `ElectronicMatrixOwner` が借用行ポインタを構成します。
ポインタ表も階層ごとの連続領域であり、各数値行の `malloc/free` は不要です。
数値の複製や、グリッド点・原子対の内側でのコンテキスト検索は追加していません。
この互換ビューは所有権を持たず、呼び出し側による解放・再代入は禁止です。
既存の多重ポインタ構文が消費側に残っているため、その完全除去はまだ完了していません。

### 再構築と寿命

`openmx_core_electronic_allocate` は形状・数値領域・互換ビューの構築をすべて終えてから
新しい所有者を公開します。途中の割り当てや入力検査が失敗しても、旧所有者と呼び出し側の
ビューを保持します。所有者は構築時に必要な形状情報をコピーするため、解放時には
旧 `M2G/F_M2G/S_M2G`、近接リスト、化学種情報を参照しません。
`truncation` の再配置と `Free_Arrays`、コンテキスト終了のいずれも同じ解放 API を使います。

DFT と Wannier 生成は呼び出し単位でビューを借用します。入力解析と最上位の計算・
標準テストのループは、その間に `truncation` が所有者を置換できるため、各呼び出し境界で
ビューを取り直します。これにより MD ステップや連続するテストで古い行ポインタを保持しません。
Force3 の容量回復コールバックも、呼び出しごとの状態にビューを持たせています。
隠れた既定コンテキストや大域配列を置き換えるマクロは導入していません。

旧実装の形状には例外があるため、次の点を明示して維持しています。

- `Partial_DM` は spin 数によらず2成分。`iDM` も常に2成分と密度履歴を持つ。
- `iResidualDM` が不要な経路にも、履歴ごとに1要素のダミー領域を持つ。
- `OLP` / `OLP_CH` の作業スロット、halo 導関数の縮小形状、射影子列数の `+2` を保持する。
- 密度・Hamiltonian・射影子・履歴それぞれの M/F/S 原子マップと収縮基底を区別する。
- 混合法0〜7を受け入れ、Hamiltonian 履歴を使わない経路では履歴数0を許容する。

メモリ使用量の表示も各所有者の実要素数から求めます。従来の一部の密度・履歴行列の表示で
`H0` の要素数を流用していた箇所を解消しました。

### 第20段階の検証

新たな8件の CTest は32系統の独立した要素数、CSR と旧形状ビューの一致、mdspan の
stride と境界、spin・混合法・オプション、入力寿命からの独立、再構築と終了を検査します。
割り当て失敗を各確保箇所で発生させ、旧状態の保存とリークがないことを確認します。
空の MPI rank に相当する入力、ゼロ長ブロック、不正入力26種、4スレッドでの独立した
コンテキストも含みます。

比較元から取り出した実際の旧アロケータと、spin・混合法・オプション・halo 数・
halo 導関数を変えた288通りで全階層の形状を照合しました。異なる M/F/S マップを使った
32系統の要素数を回帰テストの固定値にも保存しています。

GCC の ASan/UBSan、Intel C++ の Serial、Kokkos CUDA の各 core ビルドで
132件すべてが通過しました。Intel SYCL でも core 全体をコンパイルし、今回の所有・形状に
関するホストテスト8件が通過しました。CUDA/HIP の実コンパイラ検査は各11翻訳単位が通過し、
ROCm clang では HIP 用に選択される全287本の C ソースの構文・型・引数検査も通過しました。
ネイティブ OpenMP のエネルギー・Stress 112例と Hamiltonian の容量回復336例も
ASan/UBSan 付きで通過しています。

NVHPC/CUDA のアプリケーション全体をビルドし、最終の CTest は158件すべて通過しました。
ASan のメモリ配置衝突を避けるため、テスト実行時はスタック上限を8 MiBにし、
`LD_LIBRARY_PATH` / `LD_PRELOAD` を外しています。通常の `cmake --install build/cuda`
も成功し、`work/openmx` と `work/lib/libopenmx_core.so` がビルド成果物と一致しています。

標準 S 全14例を CPU/GPU 各4 MPI × 1 OpenMPで、同一プロセス内の連続ケースとして
実行しました。変更前との比較、同梱参照との比較、CPU/GPU 間の比較はすべて、
エネルギー `1e-7 Ha`、Force `1e-7 Ha/bohr` の許容値を満たしました。
収縮基底で3 MD ステップを行う CO も含みます。

`sidia333_nc_cluster.dat` は RTX 5080、18 MPI × 1 OpenMP、MPS 有効で比較しました。
変更前の起動・終了を含む実時間は113.68秒と113.43秒、変更後の再測定は114.18秒で、
再測定時の差は約0.7%です。全エネルギーと座標は出力桁で一致し、Force の最大差は
約 `1e-12 Ha/bohr` でした。変更後の OpenMX 内部の最大計算時間は110.68秒、
Force は16.50秒です。初回の変更後測定には129.61秒という結果もあり、
`Others` と対角化・出力の時間増加が見られました。実行順を逆にし、他の検証処理と
重ねない再測定では再現しませんでしたが、原因を特定したものではないため両方のログを
保持しています。目標の約105秒への到達を示す測定ではありません。

CPU 経路も C60 を18 MPI × 1 OpenMPで順に実行し、変更前10.99秒、変更後11.14秒でした。
エネルギー・全 Force 成分・座標は出力桁で一致しました。これは同じ NVHPC ビルドの
移行前後の比較であり、Intel コンパイラ版 `~/openmx4.0` との性能比較は今回再実施していません。
リンク済みバイナリの `nm` 検査でも対象の32大域シンボルが消えていることを確認し、
`openmx_globals.o` の定義数は1155から1123に減りました。

この層はベンダー SDK 型を持たず、将来の Intel/Apple GPU 経路からも利用できる設計です。
現段階の行列所有領域と互換ビューはホスト上にあり、Intel/Apple 向けの数値カーネルや
デバイス転送を実装したものではありません。AMD 実機、Intel GPU、Apple GPU、A100/H100
での実行・性能は今回未検証です。既存の GeForce・A100/H100・AMD の実行方針は維持しています。
検証入力・バイナリのハッシュとログは `build/electronic-matrix-refactor-20261021/` に保存しています。

## 第21段階：onsite・LNO・縮約係数の所有と平坦な参照

比較元は A100 最適化後の `855ffe3c` です。次の9系統を `LocalBasisState` と
計算コンテキストの所有へ移しました。対応する9個の大域宣言・定義と、数値処理から
参照されていなかった `H_Zeeman_NCO` の大域変数・確保・解放を除去しました。
Zeeman 場の物理的な寄与は `NC_v_eff` 等の既存処理に保持しています。

| 数値領域 | 連続配置と形状 |
|---|---|
| `DM_onsite` | 2履歴 × spin成分 × 局所原子ブロック。行と列の化学種を旧実装どおり区別する。 |
| `v_eff` | spin成分 × 第1 halo までの原子ごとの正方行列。 |
| `NC_OcpN` / `NC_v_eff` | 複素数の8成分 / 4成分と原子ごとの正方行列。 |
| `S12` | DC/recursion の原子別行列。近接原子の縮約基底数の合計に3を足す旧寸法を保持。 |
| `LNO_coes` / `LNO_pops` | spin × 第2 halo までの原子 × 最大基底数² / 最大基底数。 |
| `CntCoes` / `CntCoes_Species` | 原子 / 化学種 × 最大軌道数 × 最大縮約係数数。 |

不均一な行列は既存の `AtomBlockMatrix`、矩形領域は `Tensor<double,3>` を用い、
C++ は `mdspan`、C 境界は要素オフセット・形状・stride を持つビューを借用します。
数値領域は各系統につき一つの連続した確保です。原子マップと近接リストは構築中だけ
借用し、解放時には参照しません。部分的な確保失敗では旧所有者と呼び出し側のビューを
保持します。空の MPI rank、原子0の padding、halo、無効な機能の空ビューも明示します。

CUDA/HIP の91関数に依存を伝播し、88関数は `OpenMXLocalBasisViews` を引数で受け取り、
3つの入口が既存の計算コンテキストから取得します。986箇所の数値行参照を、ポインタ表を
たどる形から連続領域の行ビューへ置き換えました。SCF、Hubbard補正、力、Stress、
DC/LNO、軌道最適化、光学計算、出力まで含みます。グリッド点ごとの所有者検索や確保は
追加していません。MD と標準テストの入口は再分配後にビューを取得し直します。

Force/Stress のポテンシャル導関数と一部の軌道最適化ヘルパーは、同じ連続領域への
互換行ポインタをまだ借用しています。SCF 行列32系統の消費側も含め、アプリケーション全体の
多重ポインタ除去は未完了です。所有者・ビュー・C ABI は CUDA/HIP の SDK 型に依存せず、
将来の Intel/Apple 経路からも利用できるホスト側の境界です。デバイス転送や新しい
Intel/Metal カーネルを追加した段階ではありません。

移行中に `Contract_Hamiltonian` の非共線スピンで、4成分の和を2要素の作業配列に
書き込む既存の境界外アクセスも修正しました。実際の関数を UBSan で実行して修正前の
範囲外アクセスを再現し、修正後は独立した密行列積との117個の数値比較を通過しています。
異なる化学種、並べ替えた局所原子、halo 係数、spin=0/1/3 と虚数 SOC 成分を含みます。
入力パーサーが非共線スピンと軌道最適化の同時指定を拒否する既存の制限は変更していません。

### 第21段階の検証

所有・形状テスト8件を追加しました。旧アロケータから独立に求めた要素数、全行の
stride と重複・欠落、48通りの spin/機能組合せ、23種の不正入力、再分配、終了、
全確保箇所への失敗注入、4スレッドでの独立コンテキストを検証します。
さらに比較元から抜き出した実際の旧アロケータと384条件で照合し、9系統の全階層の
寸法が一致しました。局所マップと第1 halo のマップの順序が異なる場合も含みます。

GCC ASan/UBSan、Intel C++ Serial、Kokkos CUDA の各 core 構成で140件が通過しました。
Intel SYCL でも core 全体をコンパイルし、新しいホスト側所有権テスト8件が通過しました。
実コンパイラによる CUDA sm_80 / HIP gfx942 の各11翻訳単位と、HIP に選択される
全287本の C ソースの構文・型・引数検査も通過しています。軌道 spline の CUDA/HIP
CPU fallback と、OpenMP の energy/Stress 112例・Exc0 Stress 32例も確認しました。
Intel の実行にはコンパイラ付属のランタイムライブラリを指定し、GCC のサニタイザー実行は
スタック上限8 MiB、`LD_LIBRARY_PATH` / `LD_PRELOAD` を外した環境を使用しています。

標準 S 全14例を CPU/GPU 各4 MPI × 1 OpenMPで同一プロセス内の連続ケースとして実行しました。
変更前との比較では、CPU のエネルギーと全 Force 成分は出力桁まで一致し、GPU の最大差は
エネルギー `2.41e-10 Ha`、Force `2.6e-11 Ha/bohr` でした。同梱参照との比較、CPU/GPU 間の
比較も `1e-7 Ha` / `1e-7 Ha/bohr` を満たしました。CO の縮約軌道最適化と3 MDステップも含みます。

最終の NVHPC/CUDA アプリケーションのビルドと CTest 168件が通過しました。
`cmake --install build/cuda` も成功し、`work/openmx` と `work/lib/libopenmx_core.so` の
ハッシュがビルド成果物と一致しています。Hamiltonian の境界外アクセス修正後にも
CO を CPU/GPU で再実行し、同梱参照との比較を通過しました。
Cluster-LNO は最終 `Utot.` 行を出力しないため、比較ツールはこの場合に限り最後の
MD ステップの `Utot =` を読みます。通常のサマリー優先、複数 MD、値の欠落・非有限値を
確認するテストも追加しました。

追加8入力を各4 MPI × 1 OpenMPで実行しました。NiO の一般 DFT+U、非共線の
onsite / dual / full、spin・orbital Zeeman を有効にした dual、Si8 の DC / DC-LNO /
Cluster-LNO です。NiO は k-grid を2³、Si8 は150 Ryにし、full の入力は
Ni の s1p1d1 / O の s1p1 基底へ縮小しています。元の入力を上書きせず、派生入力と
ハッシュを保存しました。full と Zeeman の同時指定は既存パーサーで非対応のため、
両者を分けて検証しています。

CPU の変更前後は最大差がエネルギー `1.01e-11 Ha`、Force `1.61e-11 Ha/bohr`、
非共線 NiO と Cluster-LNO の GPU も変更前の同じ GPU 経路との比較を通過しました。
GPU 側の比較全体の最大差はエネルギー `1.52e-9 Ha`、Force `1.44e-9 Ha/bohr` です。
正常終了、SCF 終了の表示、反復回数、座標、格子と全 Force 成分を確認しています。
追加計算の一部は並行実行しているため、その実時間を性能比較には使っていません。

非共線 NiO の CPU/GPU 間には変更前から差があり、今回の入力ではエネルギーで最大
約 `2.24e-6 Ha`、Force で約 `4.70e-6 Ha/bohr` でした。厳しい `1e-7` の許容値で
すべての追加入力の CPU/GPU 一致を確認できたとは扱いません。full と Cluster-LNO では
GPU の SCF 反復が CPU より1回多いことも確認しました。同じ GPU 経路の変更前後では
反復回数も一致し、初回の比較エラーを含むログと再比較結果を保存しています。

性能確認は、他の計算・ビルドと重ねず、RTX 5080、18 MPI × 1 OpenMP、MPS 有効で
元の `sidia333_nc_cluster.dat` を実行しました。起動・終了を含む実時間は変更前113.131秒、
変更後114.132秒で、差は約0.9%です。変更後の OpenMX 内部最大時間は110.780秒、
Force は16.815秒でした。エネルギー・座標・FFT格子は一致し、Force の最大差は
約 `1e-12 Ha/bohr` です。MPS の18クライアントの接続も記録しました。
目標の約105秒への到達を示す結果ではありません。

CPU 経路は C60 を18 MPI × 1 OpenMPで順に実行し、変更前10.738秒、変更後10.887秒、
OpenMX 内部最大時間は8.839秒と8.847秒でした。エネルギー・全 Force 成分・座標は
出力桁まで一致しました。これは同じ NVHPC ビルドの前後比較であり、Intel コンパイラ版
`~/openmx4.0` との比較や、CPU の sidia 全計算は今回再実施していません。

A100 用 FP64 Hamiltonian タイル、multi-GPU GEMM の転送順序と結果回収、密度 packing の
容量制限、ELPA/FFTW の既存方針は維持しています。A100 最適化で既に変更された
`openmx_gpu_bilinear.h` / `openmx_gpu_bilinear_pack.cpp` は、AMD の取り込み時から不変な
ファイルとして残っていた manifest を「共通化後に変更するファイル」へ訂正しました。
元のファイル内容を復元したり、取り込み時のハッシュを現物に書き換えたりしていません。

AMD 実機、A100/H100、Intel GPU、Apple GPU の実行・性能はこの環境では未検証です。
入力・バイナリ・保護対象ソースのハッシュと検証ログは
`build/local-basis-refactor-20261021/` に保存しています。

## 第22段階: 軌道最適化の C++ 化と局所基底の互換ポインタ表の除去

`Opt_Contraction.c` の約2,800行の実装を共通 C++ コアへ移し、ネイティブ側は
MPI・既存 LAPACK・入力状態を接続する約115行のアダプターにしました。
atoms / species と atoms2 / species2、DIIS / EF、勾配、正規化、履歴・Hessian の更新を
同じモジュールが処理します。呼び出しのなかった旧 species EDIIS 実装は除去しました。
固有値計算の選択、スカラー演算と加算の順序は維持しています。

`OpenMXCoreContext` が `OrbitalOptimization` を所有し、DFT 入口で構築、出口で破棄します。
コンテキスト終了時にも解放します。履歴4系統、Hessian、エネルギー履歴は `Tensor` と
`std::vector` の連続領域になり、原子モードで不要な化学種履歴・Hessian を確保しません。
反復内で作っていた勾配、DIIS 行列、逆行列・固有値計算の作業領域も同じ所有者が再利用します。
DFT の外部変数6個と `Oopt_NormD` を除去し、ノルムは DFT の局所変数から出力へ渡します。
変更前のバイナリにあった7シンボルが、変更後から消えていることも確認しました。

化学種・縮約係数の対応、局所原子と halo の写像、近接リスト、送受信パケット計画は
構築時にコピーします。最適化中の H / OLP / DM / EDM は `OpenMXElectronicMatrixView`、
係数は `OpenMXLocalTensor3` を借用し、数値データの多重ポインタ表を使いません。
原子モードの MPI 通信は既知の要素数と順序を再利用し、反復ごとのサイズ交換を除去しました。
ネイティブ側は `MPI_Sendrecv` で両方向を処理し、送受信のない方向は `MPI_PROC_NULL` とします。
要素数は MPI の `int` 範囲、平坦配列のオフセットは `size_t` で検証します。

モジュールの入力・C ABI は MPI communicator やベンダー SDK 型に依存しません。
集約・broadcast・exchange・固有値・逆行列のサービスを引数で受け取ります。
構築時だけ借用する既存メタデータのポインタ表と、既存固有値関数への小さい行アダプターは
この境界に残ります。構築失敗は旧所有者・出力ビューを保持します。一方、反復途中で
サービスが失敗した場合は係数・履歴が途中まで更新されるため、その計算を停止します。
ネイティブ側は診断を出して MPI 計算を終了し、C ABI から例外を漏らしません。

ゼロ勾配での除算を避け、係数を変更せずにノルム0を返すようにしました。また species DIIS で
勾配の二乗ノルムが `1e-6` 以下の場合に未初期化だった降下係数を、隣接する分岐と同じ
`0.4` に定義しました。s/p の磁気成分間で係数数が異なる不正なメタデータも拒否します。

第21段階の9系統から残りの互換行ポインタ表をすべて除去しました。Force / Stress も
ポテンシャルの平坦ビューを受け取ります。CUDA/HIP で同一だった Hubbard full-projector の
実・非共線力導関数は `openmx_hubbard_force.h` へ共通化し、実数経路の x/y/z 作業行列を
1個の C++ Tensor 所有領域へまとめました。独立した C ABI で確保・解放し、SDK や
Kokkos ランタイムの初期化を必要としません。SCF 行列32系統等の互換表はまだ残ります。
今回の最適化処理は共通ホスト実装であり、Intel/Metal のデバイス実装や軌道最適化の
GPU オフロードを追加した段階ではありません。

### 第22段階の検証

軌道最適化のテスト12件を追加しました。変更前 `6e7a8dad` の有効な数値処理を
テスト専用名前空間に保存し、atoms/species × spin=0/1 × DIIS/EF の8条件を各4反復、
係数・履歴・Hessian・ノルムまで逐次比較します。別に、縮約エネルギーの独立した式を
有限差分し、s/p の勾配、磁気成分の結合、複数原子の化学種勾配、最後の動径関数だけを
最適化する制約を検証しました。ゼロ勾配と小勾配 DIIS の修正も個別に確認しています。
形状、構築後の入力破棄、全確保箇所への失敗注入、置き換え・終了、4スレッドの独立状態、
halo パケット、不足したサービスや行列、不正設定と失敗時の出力保持も含みます。

共通化した実際の Hubbard 導関数を、変更前の関数と216条件で照合しました。
異なる化学種、縮約の有無、spin=0/1/3、halo、欠けた近接ブロック、x/y/z と複素成分を
ASan/UBSan で確認しています。局所基底の所有・全行配置を検証する既存8件も、平坦ビューを
使う検査へ更新し、旧寸法から独立に求めた要素数とオフセットを引き続き照合します。

GCC ASan/UBSan、Intel C++ Serial、Kokkos CUDA の core 全152件が通過しました。
Intel SYCL では core をビルドし、変更箇所のホスト側20件が通過しました。
追加した磁気成分・小勾配の検査も各構成で確認しています。実コンパイラによる
CUDA sm_80 / HIP gfx942 の各11翻訳単位と、HIP に選択される C 全287本の
構文・型・引数検査も通過しています。アプリケーションの NVHPC/CUDA ビルドと
CTest 全181件も通過しました。

標準 S 全14例を CPU/GPU 各4 MPI × 1 OpenMPで、同一 MPI プロセス内の連続ケースとして
変更前後に実行しました。変更前、同梱参照、変更後の CPU/GPU 間の全5比較が
`1e-7 Ha` / `1e-7 Ha/bohr` を満たしました。CO の軌道最適化と3 MDステップも含みます。
変更前後の最大差は CPU でエネルギー0・Force `1e-12 Ha/bohr`、GPU で
エネルギー `1.2e-11 Ha`・Force `2.7e-11 Ha/bohr` でした。
追加7入力は CO の species DIIS / species EF / species2 EF / atoms2 DIIS、
NiO の一般 DFT+U と非共線 full、Si8 の DC-LNO です。入力とハッシュを保存し、
各入力を CPU/GPU・変更前後の計28回実行しました。同じ経路の変更前後で、正常終了、
SCF 反復数、原子・座標・格子、エネルギー・全 Force 成分を照合しています。
追加7入力の最大差は CPU で `2.01e-12 Ha` / `9.14e-10 Ha/bohr`、GPU で
`4.61e-9 Ha` / `1.34e-9 Ha/bohr` でした。
追加計算と標準スイートには並行実行を含むため、その時間を性能比較には使いません。

性能測定は他の計算・ビルドと重ねず、元の `sidia333_nc_cluster.dat`、RTX 5080、
18 MPI × 1 OpenMP、既存 MPS サーバーで行いました。最適化処理の実装後と、最後の
寸法上限検査を含む最終ビルドで測定し、両方とも25 SCF、140³格子で正常終了しました。

| GPU＋MPS | 内部最大計算時間 (秒) | 起動・終了を含む実時間 (秒) | Force (秒) |
| --- | ---: | ---: | ---: |
| 変更前 `6e7a8dad` | 110.347 | 113.839 | 16.732 |
| 変更後の初回 | 104.832 | 108.370 | 15.474 |
| 最終ビルド | 105.874 | 109.417 | 15.964 |

いずれも変更前とエネルギー・座標・格子が一致し、Force の最大差は約 `1e-12 Ha/bohr`。
MPS への18クライアントの接続も記録しました。計算部分は約105秒ですが、起動を含む
実時間105秒には届いていません。この環境・入力での比較です。
CPU は C60 を18 MPI × 1 OpenMPで順に実行し、変更前10.736秒、変更後10.635秒、
内部最大時間は8.804秒、8.476秒でした。エネルギー・全 Force・座標は出力桁まで一致しました。
同じ NVHPC ビルドの前後比較で、Intel 版 `~/openmx4.0` や CPU の sidia 全計算は
今回再測定していません。

最終ビルドの `cmake --install build/cuda` も成功し、`work/openmx` と
`work/lib/libopenmx_core.so` のハッシュがビルド成果物および最終性能計測の使用物と一致します。
AMD 実機、A100/H100、Intel GPU、Apple GPU の実行・性能は未検証です。
A100 用 Hamiltonian タイル、multi-GPU GEMM、密度 packing の保護対象5ソースは
比較元と同一で、Makefile の既存最適化設定も維持しています。検証ログ、入力・成果物の
ハッシュ、保護対象の比較は `build/orbital-optimization-refactor-20261021/` に保存します。

## 第23段階: Cluster・Band・DC・DC-LNO・Krylov の作業領域の所有化

優先対象7ファイルの CUDA/HIP 両方に、共通の C++ `solver_workspace` を導入しました。
コリニア・非コリニア Cluster / Band、DC / DC-LNO の実・複素経路、Krylov の通常・
スレッド経路の計20入口が `OpenMXSolverInvocation` を通して数値本体を呼びます。
所有者は C++ のスタック上にあり、正常復帰、途中復帰、C++ 例外のいずれでも破棄されます。
C ABI から例外は漏らしません。MPI のエラー終了は従来どおりプロセスを終了します。
この所有層は MPI communicator、CUDA/HIP 型や既定の計算コンテキストに依存せず、
Kokkos ランタイムの初期化・終了も行いません。

通常のホスト作業領域の `malloc/free` を、worker ごとの名前付き `Tensor` に置き換えました。
実・複素行列、整数の範囲・写像、MPI 送受信バッファ、密度蓄積、固有値計算の作業領域を
同じ呼び出し内で再利用します。MPI_Request/Status と LAPACK INTEGER は、各ネイティブ
ABI のサイズを用いた整列済み領域です。同じ形状では再確保・コピーせず、小さくした後も
容量を保持します。worker が同時に使う領域は独立しています。寸法積のオーバーフローを
検査し、確保失敗は以前の領域と出力を保持します。固有値計算・BLAS が必要とする行表も
C++ が所有し、数値データへの借用アダプターとしてだけ渡します。

DC / DC-LNO / Krylov の固有値、PDOS、DC の Fermi 重みは、成分・原子ごとの長さを持つ
一次元 Tensor と固定長の行記述子になりました。残差は構築時に近接・化学種情報をコピーし、
成分・原子ごとの `[orbital-pair,state]` Tensor へ格納します。異なる軌道数の原子・隣接を
オフセットで表し、旧6段ポインタの数値領域を除去しました。遅延確保は原子・成分ごとに
一度行い、内側ループは借用した行を使います。破棄は変更後の原子分配や近接リストに依存しません。
実数経路と複素3成分経路、ダミー原子の配置、DOS 出力側も同じビューを使います。

非コリニア Band の `ReEVec/ImEVec_{i1,i2,j1,j2}` は8個の大域配列から呼び出しごとの
平坦な作業領域になり、密度計算へ明示的に渡します。CWF 側には同じ名前の未使用配列の
定義・確保・解放だけが残っていたため、合わせて削除しました。Krylov の LO/HO は整数 Tensor、
作業行列は平坦ビュー、逆重なりの3次元作業領域は連続領域と C++ 所有の ABI 行表です。
分岐で使わないビューは空で初期化します。

既存の数値式・加算順、MPI メッセージの順序、ELPA の選択、GPU 呼び出しは維持します。
GeForce の容量制限付き任意キャッシュ、AMD の共有 packing、A100 の既存カーネル・
転送・実行方針を、この所有化に合わせて変更していません。Band の任意の固有ベクトル/
packing キャッシュ、DC-LNO の反復間 LO/HO・重なりキャッシュ、GPU の永続ハンドル・
作業領域、Krylov 内部の基底生成等の補助関数の確保は引き続きネイティブ側に残ります。
SCF 行列の借用ポインタや全アプリケーションの大域変数も未完です。ソルバー全体を
C++ / Kokkos カーネルへ置き換えた段階ではありません。

### 第23段階の検証

共通作業領域の8件を追加し、バッファの再利用・縮小、実/複素/整数の行配置、数値コピーを
伴わない ABI 行表の生成と再結合、整列済みバイト領域、異なる原子・軌道数の残差、
トポロジー入力の破棄、確保失敗、部分構築・途中復帰・例外時の解放、4 worker の独立性を
検査します。残差の要素数は固定した独立の寸法から照合します。

GCC ASan/UBSan、Intel C++ Serial、Kokkos CUDA の core 全160件が通過しました。
Intel SYCL はビルドと今回に関連するホスト側28件が通過しました。実コンパイラによる
CUDA sm_80 / HIP gfx942 の各11翻訳単位と、HIP の C 全287本の構文・型・引数検査も
通過しました。NVHPC/CUDA のアプリケーションをビルドし、CTest 全189件を検査しました。
最初の実行では5件が ASan の shadow mapping / alternate signal stack 初期化で失敗し、
スタックを8 MiBにした環境で、その5件も通過しています。

標準 S の14入力を CPU/GPU 各4 MPI × 1 OpenMPで、同じ MPI プロセス内の連続ケースとして
変更前後に実行しました。変更前後、同梱参照、変更後の CPU/GPU 間の全5比較が
`1e-7 Ha` / `1e-7 Ha/bohr` を満たしました。所有層の変更後も、CO の軌道最適化と
複数 MD ステップを含む連続実行が正常終了しています。

対象経路の9入力は Si8 の Cluster / Band / DC / DC-LNO / Krylov と、GaAs の
非コリニア Cluster / Band、Si8 の非コリニア DC / DC-LNO です。各入力を CPU/GPU で
変更前後に実行しました。CPU の9入力はエネルギー・全 Force・SCF反復数が一致しました。
GPU の9入力も正常終了し、エネルギー最大差 `2.80e-10 Ha`、Force 最大差
`1.97e-10 Ha/bohr` です。元入力・原子・座標・格子も一致します。

ただし GaAs の GPU 非コリニア Cluster / Band は、厳密な「SCF反復数一致」には不合格です。
今回の比較は Cluster が51→57、Band が16→19反復で、最終エネルギー・Force は上記の
許容値内です。Band は同じ変更前の再実行でも16→15反復でした。厳密比較の失敗記録は
残し、反復数を除く数値結果は別の比較として記録しています。この2入力の時間を性能比較に
使わず、元の sidia 入力の25 SCF計算で性能を確認します。

追加で、スピン分極 DC を CPU/GPU、近似逆重なりを使う Krylov を4 MPI×2 OpenMPと
8 MPI×2 OpenMP、非コリニア DC-LNO の分散経路を8 MPI×1 OpenMPで変更前後に実行し、
全5条件が厳密比較を通りました。近似 Krylov のエネルギー差は最大 `1.66e-9 Ha`、
Force 差は `1e-12 Ha/bohr`、反復数は一致します。これらと標準スイートには並行実行を
含むため、性能比較には使いません。

最後の不要な解放ループの除去後も、Krylov の4 MPI×2 OpenMPと8 MPI×2 OpenMPを
最終ビルドで再確認しました。非コリニア Band の8作業配列の ELF シンボルが、変更前には
存在し、最終アプリケーションには存在しないことも確認しています。

性能は他の計算・ビルドと重ねず、元の `sidia333_nc_cluster.dat` を RTX 5080・
18 MPI×1 OpenMP・MPS で前後比較しました。両方25 SCF・140³格子で正常終了し、
MPS の18クライアント接続を記録しています。

| 条件 | 内部最大計算時間 (秒) | 起動・終了を含む実時間 (秒) | Force (秒) |
| --- | ---: | ---: | ---: |
| sidia GPU＋MPS・変更前 `7106180f` | 105.389 | 108.915 | 15.625 |
| sidia GPU＋MPS・最終ビルド | 105.397 | 108.722 | 15.532 |
| C60 CPU・変更前 `7106180f` | 8.505 | 10.437 | 0.200 |
| C60 CPU・最終ビルド | 8.469 | 10.588 | 0.212 |

sidia はエネルギー・座標・格子が一致し、Force 最大差は約 `1e-12 Ha/bohr`。
CPU の C60 も18 MPI×1 OpenMPで順に実行し、27 SCF、エネルギー・Force・座標が一致しました。
いずれも同じ NVHPC ビルドでの前後比較です。計算部分の約105秒は維持しましたが、
起動を含む実時間105秒には届いていません。Intel コンパイラ版 `~/openmx4.0` との
性能比較や、CPU の sidia 全計算は今回再実施していません。

`cmake --install build/cuda` も成功しました。`work/openmx` / `work/lib/libopenmx_core.so` は
最終計測に使ったビルド成果物とハッシュが一致します。A100 用 Hamiltonian タイル、
multi-GPU GEMM、密度 packing の保護対象5ソースは変更前と同一で、既存コンパイル設定も
維持しています。AMD / A100 / H100 / Intel / Apple の実機動作・性能は未検証です。

入力、変更前の `7106180f` バイナリ・コア、比較結果、検証ログは
`build/solver-raii-refactor-20261021/` に保存します。

## 第24段階: 永続 GPU キャッシュと深部補助関数の所有化

Cluster / Band のコリニア・非コリニア、DC、DC-LNO、Krylov の7モジュールに、
計算コンテキストが所有する C++ `native_resources` を導入しました。CUDA/HIP の
計14翻訳単位の確保・解放・SDK オブジェクト生成/破棄、計2,052箇所を共通所有層へ
接続しています。ホスト領域は `unique_ptr`、GPU・pinned 領域、stream、BLAS/solver
handle は解放コールバック付き `unique_ptr` が所有します。既存の C キャッシュ構造体と
数値関数のポインタは、その所有領域への借用です。

DFT 入口で明示的な `OpenMXCoreContext` に名前付きグループを結び付けます。
隠れた既定コンテキストを設けず、CPU のみの空のグループ生成で GPU を初期化しません。
ケース終了時には従来のキャッシュ終了処理を実行してから残る所有領域を解放し、
`finish` / コンテキスト破棄でも MPI・ランタイム終了より先に回収します。
従来の Force 前の解放、stream の待機、OpenACC の unmap、借用別名の無効化を保持します。
部分初期化後に残った領域も、stream を待ってから buffer・handle・stream の順に回収します。

SDK アダプターは確保した device を記録し、別の worker/main thread から解放する場合に
一時的にその device を選択して、呼び出し元の選択を戻します。OpenACC の device 選択も
別に復元します。所有層への登録失敗では新しい領域を回収し、既存の SDK エラーと CPU
再試行の契約を維持します。ホスト・device・handle のアドレス領域を区別し、同じ数値の
アドレスが別の種類の所有権を上書きしないようにしています。ホスト `realloc` は失敗時に
旧領域を保持し、移動後の登録には既存の map node を再利用します。寸法積と最大領域長を
検査し、メタデータ用ストレージも整列・ゼロ初期化した C++ 所有領域です。

DC-LNO の重なり履歴・LO/HO と確保フラグは、関数内 static からケースごとの状態へ
移しました。CUDA の非コリニア Cluster に残っていた関数内 static GEMM handle も、
同じ寿命を持つ状態と所有者に移しました。異なる入力へ移った後に解放済みの handle や
前ケースの確保フラグを再利用しません。キャッシュの容量上限・退避・再利用条件は保持し、
GeForce、AMD Radeon、A100/H100 の既存の選択方針を変更していません。

Krylov の基底生成、近似逆重なり、埋め込み行列、LAPACK 補助関数は、CUDA/HIP 合計
20補助入口・102確保箇所を呼び出し単位の RAII に移しました。2/3次元作業領域は連続 Tensor
と所有された行表で構成し、同じ関数内の重複変数名にも確保箇所ごとに別の領域を割り当てます。
従来の SIMD が各 malloc 行に要求していた16 byte整列は、実行用行列の物理 stride を
偶数にすることで保持します。数値式・論理寸法・加算順は変えていません。補助関数の終了時に
作業領域を解放するため、一時領域を計算終了まで保持する構成にもしていません。

固有値精密化の CUDA/HIP 補助関数にも呼び出し側の所有者を渡します。確保コールバックを
所有化したまま補助関数で直接 `free` / SDK free を使うと、登録に古いアドレスが残ります。
18 MPI の sidia 検証でこの不整合によるヒープ破損を検出し、精密化内の一時領域、永続
固有値・occupation、作業領域の拡張と終了処理を同じ所有者の解放 API に揃えました。
ケースをまたぐ状態の借用も終了時に無効化します。追加の連続精密化検証では、CUDA の
非コリニア Band の既存ケース終了処理が精密化キャッシュを解放していないことも検出しました。
ケース終了時にその終了処理を呼び、CUDA の Cluster/Band 4経路の精密化方針も初期値へ
戻します。HIP 側に既にあるケース境界の扱いに揃え、回収済みの領域への借用を残しません。

共通 C++ コアの所有型と C ABI に CUDA/HIP SDK の型を持ち込んでいません。
ただし、今回の移行で全アプリケーションの大域変数や生ポインタを除去したわけではありません。
ネイティブのキャッシュ記述子・借用別名・設定値の static、行ポインタ ABI、対象外の補助
モジュールと第三者ライブラリ内部の所有は残ります。ネイティブ実行は従来の単一計算の
制御構造であり、複数コンテキストの同時実行を保証する段階でもありません。
数値カーネル全体の C++ / Kokkos 化は次の作業です。

### 第24段階の検証

共通コアに所有層8件と実 SDK アダプターを SDK モデルへ接続する5件を追加しました。
部分初期化、ケース寿命、解放順、device 復元、OpenACC の選択、登録失敗、SDK エラー、
再確保、アドレス領域の分離、4 worker の同時確保・解放を検査します。
GCC ASan/UBSan、Intel C++ Serial、Kokkos CUDA は core 全173件が通過し、
Intel SYCL はビルドと関連するホスト41件が通過しました。最後のメタデータ・上限検査の
修正後も GCC 全173件と、他の各構成の新規13件を確認しています。

CUDA/HIP 精密化の実際の解放関数を C++ コアに接続する回帰テストも追加しました。
各バックエンド1,000回の状態生成・破棄と二度の終了呼び出し、CUDA 非コリニア Band の
実際のケース終了関数を使う100ケース、実/複素の20成功・失敗条件で、
解放済みの作業領域が所有登録に残らないことを検査します。ASan/UBSan でも通過しました。
アプリケーションの203テストは202件が最終一括実行で通過し、残り1件の HIP 取り込み
manifest 検査も、改修済みの精密化2ファイルを履歴保存対象から統合済み対象へ分類して
再検査が通過しました。取り込み元のハッシュを書き換える扱いにはしていません。

実 SDK による CUDA sm_80 / HIP gfx942 の各11翻訳単位と、HIP C 全287本の構文検査が
通過しました。最後の精密化改修後にも、CUDA の Cluster/Band 4本と精密化本体、HIP の
同4本と精密化 kernel 本体を実コンパイラで再コンパイルしました。CUDA DC-LNO は
60障害注入・78数値比較、HIP DC-LNO NC は1,140障害注入・949数値比較が通過しました。
HIP Krylov の24 GEMM失敗・38数値比較、device なしの解放、MI300A workspace の243設定・
7回復条件、非コリニア Band 精密化の142制御条件も通過しています。これらの HIP 検査は
ホスト SDK モデルと実コンパイラによる検査で、AMD 実機検証ではありません。

標準 S 全14入力を CPU/GPU 各4 MPI×1 OpenMPで、同一 MPI プロセス内の連続ケースとして
変更前後に実行しました。変更前後、同梱参照、CPU/GPU 間の全5比較が
`1e-7 Ha` / `1e-7 Ha/bohr` を満たしました。精密化の所有契約修正後にも同じ全5比較が
通過しました。最後のケース終了処理の修正後にも CPU/GPU 全14入力が正常終了し、全5比較が通過しました。

対象経路の9入力はすべて CPU/GPU で正常終了し、変更前後のエネルギー最大差は CPU
`1.40e-11 Ha`、GPU `5.22e-10 Ha`、Force 最大差は CPU 0、GPU `3.91e-10 Ha/bohr` でした。
CPU の SCF 回数も一致します。ただし GPU の GaAs 非コリニア Cluster は48→53反復、
Band は20→18反復で、厳密な SCF 回数一致には不合格です。この失敗記録は残し、
最終数値の一致とは区別します。これらの入力の時間を性能比較には使用していません。

スピン分極 DC の CPU/GPU、近似逆重なり Krylov の4 MPI×2 OpenMP・8 MPI×2 OpenMP、
小規模 GPU 起動を明示した Krylov の4 MPI×2 OpenMP、非コリニア DC-LNO の8 MPI分散も
全6条件が厳密比較を通りました。最大差は `1.27e-9 Ha` / `1e-12 Ha/bohr` です。
同一 MPI プロセスで DC-LNO col→Cluster NC→DC-LNO NC→Krylov→DC-LNO col→
Cluster NC→Cluster col の7入力を CPU/GPU で連続実行し、ケース終了後の再初期化も検査しました。
精密化の修正後の GPU 連続実行も通過しました。さらに、小規模問題でも GPU / FP32精密化を
明示して Cluster col→Band col→Cluster NC→Band NC を2回繰り返す8入力を実行しました。
同じ明示設定の変更前 GPU 計算と比較して、エネルギー最大差 `1.91e-10 Ha`、Force 最大差
`1.02e-10 Ha/bohr` でした。これは所有寿命を調べるために通常のサイズ選択を上書きする検証で、
通常設定の CPU/GPU 同等性は標準 S と個別9入力で確認しています。並行検証の時間は
性能値に使用していません。

性能は他の計算・ビルドと重ねず、元の sidia 入力、RTX 5080、18 MPI×1 OpenMP、
既存 MPS サーバーで測定しました。変更前後とも25 SCF・140³格子で正常終了し、
MPS の18クライアント接続も記録しました。

| 条件 | 内部最大計算時間 (秒) | 起動・終了を含む実時間 (秒) | Force (秒) |
| --- | ---: | ---: | ---: |
| sidia GPU＋MPS・変更前 `16849867` | 105.488 | 108.968 | 15.596 |
| sidia GPU＋MPS・所有契約修正後 | 105.211 | 108.723 | 15.526 |
| sidia GPU＋MPS・ケース終了処理を含む最終ビルド | 105.704 | 109.215 | 15.651 |
| C60 CPU・変更前 `16849867` | 8.445 | 10.386 | 0.202 |
| C60 CPU・最終ビルド | 8.508 | 10.436 | 0.204 |

sidia はエネルギー・座標・格子が一致し、Force 最大差は約 `1e-12 Ha/bohr`。
C60 は27 SCF、エネルギー・Force・座標が一致します。各1回の同じ NVHPC ビルドでの
前後比較で、CPU の実時間差は約0.05秒です。GPU 計算部分の約105秒は維持しましたが、
起動を含む105秒は未達です。Intel 版 `~/openmx4.0` との比較、CPU の sidia 全計算、
AMD / A100 / H100 / Intel / Apple の実機動作・性能は未検証です。

`cmake --install build/cuda` は成功しました。`work/openmx` と
`work/lib/libopenmx_core.so` は、ビルド成果物および最終測定の固定コピーとハッシュが一致します。
A100 用 Hamiltonian タイル、multi-GPU GEMM、密度 packing の保護対象5ソースも
比較元と同一です。既存 Makefile の最適化指定は維持しています。

入力、変更前の `16849867` バイナリ・コア、厳密比較の失敗を含む比較結果、所有契約修正前の
ヒープ破損ログとスタック、最終検証ログは `build/native-resources-refactor-20261021/` に
保存します。途中の性能測定には `validated/`、ケース終了処理を含む最終測定には `completed/` の固定
バイナリ・コアを使用しました。

## 第25段階: Hamiltonian・密度格子の共通ホスト積分

`refactor/cpp-kokkos-core` で `Set_Hamiltonian` と `Set_Density_Grid` の
CPU 格子積分を `core/src/grid_assembly.cpp` に移しました。CUDA/HIP に重複していた
スピン1/2/4成分の数値処理をまとめ、ネイティブ側には軌道・行列・格子の借用ビューを
作る共通アダプター `source/openmx_grid_assembly.c` を設けました。DFT が明示的な
`OpenMXCoreContext` から作業領域を渡し、この移行ではグローバル変数を追加していません。

計算ごとの所有者が、連続密度領域、atom-to-B の送受信値と行表、原子・ペアの実行計画、
worker ごとの行列・軌道パネル・積分 accumulator、MPI request/status 領域を管理します。
密度と送受信値には Tensor、数値行列には mdspan を使い、多重ポインタは C ABI 境界の
借用行表に留めます。寸法が同じ SCF では確保せずに再利用し、寸法変更は新しい領域の
構築成功後に切り替えます。容量・オフセット積を検査し、失敗時には従来のビューと値を
保持します。ケース終了時は GPU キャッシュと MPI の利用終了後に解放し、`finish` と
コンテキスト破棄でも回収します。

密度は HIP 版の原子単位の分担を共通化し、CUDA 版のペアごとの OpenMP バリアと
worker ごとの全格子複製・集約を除去しました。1ペアの密度行列をスピン交互配置で
一度だけ pack し、格子点を SIMD lane として縮約します。Hamiltonian は小さい
スピン非分極行列の16点ブロック積分と、一般形の既存行列を初期値とする積分を保持します。
各成分の隣接ペア・格子・軌道に対する加算順を維持し、core の積和非縮約の設定も保持します。
native compiler と C++ compiler の違いによる丸めの差は、実計算比較で別途検査します。

OpenMP チームは native 側で作り、実際の worker 番号・人数を core に渡します。
予約人数より小さい動的チームも扱い、core 内で別の OpenMP runtime を起動しません。
`OPENMX_SETHAM_SIMD`、`OPENMX_DENSITY_GRID_COL_SIMD`、
`OPENMX_DENSITY_GRID_NC_SIMD` は CPU/OS が実行可能な ISA のみを選び、
`scalar` / `avx2` で上限を指定できます。ホストの ISA 判定を SYCL/CUDA/HIP の
デバイスコンパイルから分離し、x86 以外では可搬な scalar 経路を使用します。

GPU kernel、on-the-fly 経路、resident table、GPU メモリ配分、MPS、
GeForce と A100/H100 の既存の計算精度・実行方式は変更していません。
native adapter が参照する既存の大域配列、B-to-C/B-to-D の通信関数、
軌道最適化中の native 一時領域、格子 GPU キャッシュの所有化は継続課題です。

worker の Tensor 内に64 byte整列の部分領域を設け、軌道パネルの stride は8要素単位に
揃えました。13軌道などでも小行列の accumulator は16列を更新し、scalar tail を
避けます。標準 C++ の所有配列内で整列するため、専用 allocator・SDK を追加せず、
確保失敗時の巻き戻しも同じ仕組みで検証できます。HIP ヘッダーがホストでも定義する
`__AMDGCN__` を CPU の ISA 判定の条件にせず、デバイスコンパイルのマクロで分離します。
ROCm の実コンパイラで、ホストの CPU feature 判定を含むオブジェクトを確認しました。

最終検証ではルートの CTest 全212件、GCC ASan/UBSan・Intel C++・Kokkos CUDA の
core 全182件、SYCL の新しい格子試験9件が通過しました。新しい格子試験は
float/double 軌道、縮約・halo 軌道、全スピン成分、奇数寸法、16軌道を超える行列、
端数・重複格子点、空の原子/rank、実行人数の変動、容量再利用、失敗時の旧領域保持を
独立した long double 基準と照合します。CUDA sm_80 と HIP の native 6翻訳単位を
実SDK・O0でコンパイルし、共通 C++ 積分も HIP/gfx942 でコンパイルしました。

CPU・GPU の `-runtest` は各14例を同じ MPI プロセス内で連続実行し、ケースごとの
所有領域の解放・再生成を確認しました。変更前後、CPU対GPU、同梱基準値との
エネルギー・全Force成分・原子・格子比較はいずれも `1e-7` 以内です。最終版の同梱基準値
との差は、エネルギーで最大 `6.1e-11 Ha`、Forceで最大 `1.1e-11 Ha/bohr` でした。
Cluster/Band の collinear/noncollinear、DC、DC-LNO、Krylov の9経路×CPU/GPU、
スピン分極DC、2 OpenMP の Krylov、空rankを含むDC-LNOの比較も保存しています。

厳密な SCF 回数一致は、小規模 NC の2例では成立しません。最終版の NC Cluster/CPU は
46→65 SCF、NC Band/GPU は19→16 SCFでした。前者のエネルギー最大差は約
`5.8e-9 Ha`、Forceは約 `3.56e-9 Ha/bohr`、後者は約 `2.9e-11 Ha` と
`1.3e-11 Ha/bohr` で、どちらも数値許容範囲内です。変更前バイナリの CPU 再実行は
46 SCFを再現しました。積和縮約を試した別ビルドでも反復回数は一致せず、採用していません。
この2例の反復回数を含む厳密比較の失敗は、エネルギー・Forceの比較結果と分けて残します。

性能は他のビルド・計算と重ねず、18 MPI×1 OpenMP、RTX 5080＋既存 MPS で測定しました。
元の sidia 入力を使用し、変更前後とも25 SCF・140³格子で正常終了、MPSの18クライアント
接続を記録しています。CPUのC60は27 SCF・84³格子です。

| 条件 | 内部最大計算時間 (秒) | 起動・終了を含む実時間 (秒) | Set_Hamiltonian (秒) | Set_Density_Grid (秒) |
| --- | ---: | ---: | ---: | ---: |
| sidia GPU＋MPS・変更前 `c2ab0e5e` | 105.742 | 109.270 | 20.428 | 13.718 |
| sidia GPU＋MPS・最終版 | 103.834 | 107.416 | 20.436 | 12.544 |
| C60 CPU・変更前 `c2ab0e5e` | 8.579 | 10.485 | 0.601 | 0.490 |
| C60 CPU・最終版 | 8.539 | 10.637 | 0.657 | 0.540 |
| C60 CPU・Intel版 `~/openmx4.0` | 9.841 | 10.847 | 0.482 | 0.429 |

sidiaのエネルギー・座標・格子は一致、Force最大差は約 `1e-12 Ha/bohr`。
C60の変更前後はエネルギー・全Force・座標が一致し、Intel版との差も約 `1e-11` です。
各条件1回の比較で、CPU全体の時間はほぼ維持しましたが、CPUのこの2区間には低下が
残ります。Intel版はコンパイラ・MPI・ライブラリ構成も異なり、このC60の結果だけで
CPU全般の優劣を判断しません。sidiaの計算部分で約105秒の目標に達しましたが、
起動・終了を含む105秒は未達です。CPUのsidia全計算、AMD・A100/H100・Intel/Apple GPUの
実機性能は今回測定していません。

`cmake --install build/cuda` は成功しました。`work/openmx` と
`work/lib/libopenmx_core.so` には実計算で検証した固定成果物を配置しています。
最終のHIPホスト判定修正後もCUDAのcoreは同一ハッシュです。A100用Hamiltonianタイル、
multi-GPU GEMM、密度packingの保護対象5ソースも変更前と同一です。
検証前後のバイナリ、入力、反復回数一致の失敗を含む数値結果と全ログは
`build/grid-assembly-refactor-20261021/` に保存します。最終計測・連続テストには
`optimized/` の固定バイナリとcoreを使用しています。

## 第26段階: 重なり・非局所・VNA積分の共通基底変換

`Set_OLP_Kin`、`Set_Nonlocal`、`Set_ProExpn_VNA` の VNA2/3 CPU fallback に
重複していた複素球面調和基底から実基底への変換を `core/src/two_center.cpp` へ移しました。
`Set_Nonlocal` は CPU の動径積分後と CUDA/HIP の GPU 動径積分後の両方から、同じ
`SetNonlocal_TransformDSNL` を呼び出します。VNA2/3 の両方向も共通モジュールを使います。
既存の Gaunt 積分、Bessel 関数、動径積分、GPU デバイス上の基底変換はこの段階では
変更していません。

新しい `TwoCenterWorkspace` を既存の3種類の worker 所有領域に組み込みました。
複素変換の中間領域、実数の出力、重なりの軌道角運動量を連続した `Tensor` で所有し、
Kokkos/mdspan または対応環境の `std::mdspan` で参照します。4成分・8成分を分けて
コンパイルし、積分のブロック内ではメモリ確保しません。C 側は寿命が明示された平坦な
借用ビューだけを受け取ります。

入力には変換表の幅・角運動量数、元行列の寸法・ブロック位置、距離・三角関数値・
オンサイト指定を渡します。モジュール自身は `Comp2Real`、原子・化学種の大域表、
MPI communicator、GPU SDK、既定の計算 context を参照しません。実行は呼び出し側の
OpenMP チームで行い、別の OpenMP/Kokkos ランタイムを起動しません。変換の縮約順、
極軸での分岐、オンサイトのゼロ微分、重なり8倍・運動エネルギー4倍の規格化、
軌道角運動量の符号を保ちました。固有値安定化用 `rnd` は native 側に残し、呼び出す
軌道・ペア・スレッドの順序を変えていません。コンパイラ間の積和縮約の差は実計算で
別途検査します。

`Set_OLP_Kin` と CUDA/HIP の `Set_Nonlocal` は、手作業の `OneD2Mc_AN` /
`OneD2h_AN` 配列を既存の `OpenMXAtomPairPlan` へ置き換えました。局所原子・近接原子の
順序を保持した所有スナップショットを作り、空 rank も同じインターフェースで扱います。
`Set_Nonlocal` の波数格子と複素 Hamiltonian scratch は C++ の invocation が所有します。
ハロー積の `NLH_halo` とローカル積の `NLH` を、同じ worker ごとの連続テンソルに
統合しました。深部の `Multiply_DS_NL` も寸法付きビューで行を参照するため、数値用の
3重・4重ポインタと行ごとの malloc/free は不要です。MPI の送受信順・待機位置は維持しました。

ユニットテストは、独立した long double 複素行列積、行列として構築した角運動量演算子、
Cartesian 有限差分を基準にしています。s〜g の全組合せ、上限の角運動量14、異なる
左右寸法、非ゼロのブロック位置、極軸付近、オンサイト、独立 invocation、並列実行、
確保失敗時の解放、計算中の確保ゼロ、不正入力時の旧出力保持を検査します。
既存 VNA2/3 の解析的 s 波テストは、1/2/8 OpenMP と両方向を引き続き検証します。

ルートの既存 CTest 全219件と追加の非局所 invocation 寿命テスト1件、
GCC ASan/UBSan・Intel C++・Kokkos CUDA の core 全189件が通過しました。
新しい寿命テストは両 backend の実際の wrapper と C++ owner を使い、1/2/4/8 OpenMP、
呼び出し間の scratch 初期化、同じ呼び出し内の再利用、HIP の GPU→CPU 使用表示リセットを
確認します。このテスト自体も ASan/UBSan で実行しました。
SYCL は新しい数値試験7件と複素テンソル行ビューの試験を確認しています。
実SDKで CUDA sm_80・HIP の対象3翻訳単位ずつを O0 コンパイルし、新しい C++ 数値処理と
3種類の workspace は HIP/gfx942 でもコンパイルしました。AMD・A100/H100・Intel/Apple GPU
の実機実行は今回行っていません。

CPU/GPU の `-runtest` 各14例を同じ MPI プロセスで連続実行しました。変更前後、
CPU対GPU、同梱基準との5比較は全て `1e-7` 以内です。同梱基準との差はエネルギーで
最大 `6.1e-11 Ha`、Force で最大 `7.8e-11 Ha/bohr` でした。
Cluster/Band の collinear/noncollinear、DC・DC-LNO・Krylov の9経路×CPU/GPU、
スピン分極DC、2 OpenMP の Krylov、8 MPI の空rankを含むDC-LNOの計23比較でも、
最大差はエネルギー `1.74e-9 Ha`、Force `4.73e-10 Ha/bohr` で許容値内です。
原子、座標、格子も一致しました。

小規模 NC の GPU 2例は SCF 回数一致の厳密比較では失敗を残しています。
Cluster は53→49回、Band は15→16回で、エネルギー差はそれぞれ `2.30e-10 Ha` と
`1.53e-10 Ha`、Force 最大差は `4.73e-10` と `3.00e-11 Ha/bohr` です。
この結果を数値一致と分け、元の失敗した `result.json` を変更せずに別ファイルで監査しました。
CPUのNC Clusterは変更前後とも65回です。単純な一括成功として反復回数の差を隠していません。

他の計算・ビルド・テストと重ねず、18 MPI×1 OpenMP で性能を確認しました。
RTX 5080＋既存 MPS の sidia は、変更前後とも18クライアント・25 SCF・140³格子です。
CPU の C60 は27 SCF・84³格子を保ち、初回に見えた約3%の時間差を調べるため
実行順を入れ替えて各3回測定しました。下表の C60 は各区間の中央値です。

| 条件 | 内部最大計算時間 (秒) | 起動・終了を含む実時間 (秒) | Set_OLP_Kin (秒) | Set_Nonlocal (秒) | Set_ProExpn_VNA (秒) |
| --- | ---: | ---: | ---: | ---: | ---: |
| sidia GPU＋MPS・変更前 `2dfb1811` | 104.344 | 107.924 | 0.561 | 0.917 | 7.664 |
| sidia GPU＋MPS・変更後 | 104.280 | 107.718 | 0.535 | 0.890 | 7.702 |
| C60 CPU・変更前、3回中央値 | 8.696 | 10.786 | 0.110 | 0.172 | 0.819 |
| C60 CPU・変更後、3回中央値 | 8.716 | 10.837 | 0.113 | 0.175 | 0.857 |
| C60 CPU・Intel版 `~/openmx4.0`、1回 | 9.912 | 10.436 | 0.502 | 0.907 | 0.604 |

sidia のエネルギー・座標・格子は一致し、Force 最大差は約 `1e-12 Ha/bohr` でした。
重なり基底変換部分の rank 最大時間は0.060→0.033秒でした。CPU の変更前後は全6回とも
エネルギー・全Force・座標が一致し、Intel版との差はエネルギー約 `1e-11 Ha`、
Force 約 `1.4e-11 Ha/bohr` です。全体の CPU 中央値は内部で約0.23%、実時間で約0.47%の差で、
この入力では概ね同等の性能です。ただし VNA の CPU 区間は0.819→0.857秒で、改善余地が
残ります。sidia は計算部分の約105秒を維持しましたが、起動・終了を含む105秒は未達です。
GPU は各1回、Intel版はコンパイラ・MPI・ライブラリ・出力処理も異なるため、
これらの結果を全入力・全 GPU での性能向上とは解釈しません。CPU の sidia 全計算は
今回測定していません。

`cmake --install build/cuda` は成功し、`work/openmx` と `work/lib/libopenmx_core.so` は
実計算・性能測定に使った `candidate/` の固定成果物と同一ハッシュです。A100向けの
Hamiltonian タイル、multi-GPU GEMM、密度packingの保護対象5ソースも変更前と同一です。

この段階でも native の基底・幾何表の大域参照、`DS_NL` 等の借用行ポインタ、
`Multiply_DS_NL` の相対論・core-hole の数値ループ、Gaunt/動径の C 実装と一部 GPU の
一時領域は残っています。今回の共通化はホストの数値処理と所有の境界であり、アプリ全体の
Kokkos GPU 移行や Intel/Apple GPU 実装の完了を意味しません。

検証成果物は `build/two-center-refactor-20261021/` に保存します。

## 第27段階: Force の呼び出し所有・段階別 scratch と共通縮約

`Force` のハロー密度行列 `CDM0` / `iDM0` と原子ごとの力を
`ForceMatrices` の所有へ移しました。`F_M2G`、原子種、縮約軌道数、近接リストから
不変のブロック配置を作り、スピン成分ごとの数値を連続した `AtomBlockMatrix` で保持します。
原子0の1×1パディングを明示するため、旧 allocator の `FNAN[0]=0` という大域表への
書き込みも不要です。残る C 数値ループには数値を所有しない互換行ビューを貸し出します。
この行ビュー生成を既存の VNA/SCF 行列所有と共有し、所有と借用を分離しました。

主処理、`Force3` の両 OpenMP 実装、`Force_HNL`、`Force4B`、`Force_CoreHole`、
深部の `dHCH` と `MPI_OLP` に明示的な scratch を渡します。複素微分3成分、Hubbard の
実数3成分・2×2スピン行列、VNA の微分、グリッド微分、ペアの添字、worker の縮約値、
MPI のメッセージとカウントは再利用する C++ テンソルが所有します。行ごとの確保・解放を
削除し、同じ worker が原子・受信ハローを切り替えても容量を再利用します。
MPI 送信と受信の領域は独立させ、未完了の `MPI_Isend` を待ってから再利用する順序を保ちました。

`openmx_force_stages.h` の共通 dispatcher が各段階を入れ子の C++ invocation で実行します。
大きな `Force3` のグリッド微分領域は同段階の終了時に解放し、後続の VNA/HNL の
GPU archive とホストメモリのピークを重ねません。通常 return、早期 return、例外、
途中の確保失敗でも同じ所有規則が働きます。公開 `Force` の返す時間は外側で計測し、
密度配置の生成と最終の解放も含めます。

Force#2 のスピン縮約と Force#5 の重なりの縮約は、CUDA/HIP 共通の
`core/src/force_workspace.cpp` を使います。密度と各 Cartesian 微分を mdspan で参照し、
非スピンの縮退、対称ペアの係数、複素対角・非対角密度の符号を維持します。
微分テンソルの確保行数・行ストライドを実際の積分ブロックの寸法と別に渡します。
重なりは h=0 を除外し、元の重みの乗算・差し引き順を保ちます。縮約計算では EDM が
primitive 基底の大きなブロックを確保している場合があり、CntOLP の寸法を使って EDM の
行ストライドを推定しません。大きな EDM ブロック内の先頭の縮約領域を、そのブロック自身の
寸法で参照します。実際の縮約 Methane と余白付き行列の単体試験で確認しました。公開 C 引数に
別の OLP/EDM 行ビューが渡された場合も既存の挙動を保ちます。共通数値処理は大域変数、
MPI、GPU SDK、既定 context を参照せず、呼び出し側の worker 上で動作します。

追加した単体テストは、独立した long double のスピン密度行列積と有限差分を基準に、
スピン0/1/3、spin-orbit、対称ペア係数、異なる原子種・ハロー・矩形寸法、余白付き行と
成分ストライド、オンサイト除外、不正入力時に部分的な力を残さないことを検査します。
寿命・確保失敗・例外・並列 invocation と、CUDA/HIP 両方の実 wrapper による
1/2/4/8 worker・段階ごとの scratch 解放と呼び出し間の再初期化も検証します。

ルートの CTest 全228件、GCC ASan/UBSan・Intel C++・Kokkos CUDA の core 全196件が
通過しました。SYCL は Force と共通行アダプターに関係する24件を通過しました。
実 wrapper の寿命試験も ASan/UBSan で確認しています。CUDA sm_80 と HIP の両方で、
通常版と `f3omp` 版の Force をコンパイルしました。Force の新 C++ 数値処理と既存の
SCF/VNA 行列アダプターも HIP/gfx942 でコンパイルしました。

CPU/GPU の `-runtest` 各14例を、それぞれ同じ MPI プロセスで連続実行しました。
変更前後・CPU対GPU・同梱基準との5比較は全て `1e-7` 以内です。同梱基準との差は
エネルギー最大 `6.1e-11 Ha`、Force 最大 `7.8e-11 Ha/bohr` でした。
Cluster/Band の collinear/noncollinear、DC・DC-LNO・Krylov の9経路×CPU/GPU、
分極DC・2 OpenMPのKrylov・空rankを含む8 MPIのDC-LNOの5比較、core-holeの
collinear/noncollinear・Hubbard dual・縮約Methaneの6比較を実行しました。
計29比較のエネルギーと全Force成分は `1e-7` 以内です。縮約Methaneの通常のSCF表示は
専用のため、汎用runnerの反復数欄だけに頼らず、18回の `G-SCF/L-SCF` の順序もCPU/GPUの両経路で
変更前後に一致することを別途検査しました。

緩いSCF条件 `1e-6` の NiO Hubbard dual では、Force 最大差が `7.4103e-8 Ha/bohr`、
エネルギー差が `2.5780e-9 Ha` で、今回の29比較の最大でした。この例ではForceの前の
最終 `Uele` も約 `1.21e-7 Ha` 異なっているため、より厳しい収束条件と再現性も追加検査しました。
同じNiO入力を `1e-10` とした追加試験では、4 MPI×1 OpenMPは変更前後とも176 SCFで
エネルギー・全Forceが完全一致しました。一方、4 MPI×2 OpenMPでは、同じ旧バイナリ・入力の
再実行だけでも159→175 SCFと変わりました。新バイナリの追加試験は200 SCFの上限に達し、
この条件での収束完了は確認できていません。エネルギー差 `2.5080e-9 Ha`、Force最大差
`3.8636e-8 Ha/bohr` は許容値内ですが、この試験を収束成功や反復数一致として扱いません。
並列SCFの再現性と、この高精度Hubbard条件の安定化は残る検証課題です。

他の例ではForce最大差は `1.264e-9 Ha/bohr` 以下でした。厳密な反復回数比較は
小規模NCの3例で失敗を残しています。GPU Clusterは62→53、GPU Bandは16→19、
2 OpenMPのCPU core-hole NCは21→20回です。対応するエネルギー差はそれぞれ
`1.0101e-10`、`1.6996e-11`、`3.7005e-11 Ha`、Force最大差は `6.11e-10`、
`3.9e-11`、`1.264e-9 Ha/bohr` でした。元の `result.json` の失敗を残し、
別の数値監査で評価しています。

`OPENMX_FORCE4B_FUSED=0` の従来の微分行列経路と、GPU固有値計算から
`OPENMX_FORCE_GPU=0` でForceだけCPUへ切り替える経路も4 MPI×2 OpenMPで比較し、
エネルギー・全Forceが許容差内でした。上記の追加Hubbard比較とCPU再測定を含めた
個別の数値監査35組は全て `1e-7` 以内です。反復数の不一致4件と、そのうち高精度Hubbardの
未収束1件は数値の差と分けて記録しています。

他の計算・ビルド・テストと重ねず、18 MPI×1 OpenMPで性能を比較しました。
sidiaは既存MPSに18クライアントが接続し、25 SCF・140³格子を保ちました。
C60のCPU経路は27 SCF・84³格子で、実行順を変えて各3回測定した中央値です。

| 条件 | 内部最大計算時間 (秒) | 起動・終了を含む実時間 (秒) | Force (秒) |
| --- | ---: | ---: | ---: |
| sidia GPU＋MPS・変更前 `e6fd469c` | 104.142 | 107.668 | 15.652 |
| sidia GPU＋MPS・変更後 | 103.927 | 107.267 | 15.515 |
| C60 CPU・変更前、3回中央値 | 8.708 | 10.786 | 0.208 |
| C60 CPU・変更後、3回中央値 | 8.675 | 10.786 | 0.201 |
| C60 CPU・Intel版 `~/openmx4.0`、1回 | 9.809 | 10.336 | 0.178 |

sidiaのエネルギー・Force最大差はいずれも約 `1e-12`、C60 CPUの変更前後は全6回で
エネルギー・全Forceが一致しました。Intel版との差はエネルギー約 `1e-11 Ha`、
Force約 `1.4e-11 Ha/bohr` です。Force時間にはC++所有の生成・解放も含め、計算部分の
約105秒とCPUの性能を維持しました。sidiaは各1回であり、GPU全般の高速化を示す結果では
ありません。起動・終了を含むsidiaの105秒は未達です。CPUのIntel版も構成・MPI・出力処理が
異なるため、このC60の結果だけで全CPU計算の性能を判断しません。

今回も NVIDIA の GPU 数値カーネル、AMD の APU 判定・FP64 方針、GeForce と
A100/H100 のバッチ・メモリ・精度選択は保持しています。GPU 補助関数のトークン比較と、
A100 向け Hamiltonian・multi-GPU GEMM・密度 packing の保護対象5ソースのハッシュで
確認しました。AMD・A100/H100・Intel/Apple GPU の実機確認はこの環境では行っていません。

GPU archive の一部の static context、GPU packet の手作業の確保、Force3 の軌道補助
handle、native の幾何・分配・Hubbard 表と借用の多重ポインタは引き続き残っています。
今回の段階で Force 全体の大域変数ゼロや、GPU カーネルの Kokkos 化が完了したわけでは
ありません。検証成果物は `build/force-refactor-20261021/` に保存します。

## 第28段階: HNL・core-hole の共通縮約と軌道・Hubbard scratch の再利用

AMD側の `d437109f` の確認・最適化を引き継ぎ、HIPの原子内チーム並列縮約を維持しました。
`Force_HNL` と `Force_CoreHole` の各13か所、CUDA/HIP合計52か所に残っていた
スカラー・共線・スピノルの密度と微分の積和を、既存の共通C++ `force_spin_trace` に移しました。
数値処理は多重ポインタを辿らず、密度と微分を寸法・ストライド付きのmdspanで参照します。
順方向・逆方向の原子対、オンサイト・対称ペア・DC系の係数の選択はnative側に残し、
各ループの積和順序、虚部の符号、逆向きの密度ブロック・微分行列を保持しています。

`OpenMXSolverInvocation` の各workerが名前付き `OrbitalWorkspace` を `unique_ptr` で所有する
APIを追加しました。Force3の通常処理と遅延GPU回復はこの領域を借用し、手動での
`orbital_create` / `orbital_destroy` を必要としません。同じ形状なら確保せずに再利用し、
形状変更の失敗時には従来の所有領域と出力を保持します。呼び出し終了・早期return・例外で
解放され、MPIやGPU SDK、可変大域キャッシュには依存しません。公開の軌道確保APIと
このAPIのビュー生成は `OrbitalWorkspace::buffers()` に一本化しました。
nativeヘルパーは出力引数でビューを受け取ります。大きな構造体の値返しを使った初版は
この環境のNVHPC 26.9のForce3 OpenMP領域で不正なグリッド添字による異常終了を再現し、
出力引数へ変更した最終版でCPU実計算を再検証しました。

Hubbard full-projectorの共線導関数は、原子対ごとのtranspose領域の確保・解放を除去しました。
呼び出し側のworkerが一度確保した平坦なテンソルを明示的な引数で渡します。
補助関数内の `List_YOUSO` 参照も取り除きました。再利用領域をNaNで埋めてから旧実装と
比較する試験で、初期ゼロ値や前の原子対の値に依存しないことを確認しています。

追加試験では実際の両backend・両関数の呼び出し式を抽出し、旧13形式のループと比較します。
25種類の矩形寸法、余白付き微分、逆向き原子対、スカラーの縮退係数、共線NC、スピノル、
スレッド別の力配列を対象とします。軌道cacheは無確保での再利用、形状変更の失敗、
worker間の独立性、全確保位置での失敗、例外・早期returnでの解放を検証しました。

ルートCTest全230件、GCC ASan/UBSan・Intel C++・Kokkos CUDAのコア各197件が通過しました。
新しい縮約呼び出し試験、実Force wrapperの寿命試験、旧Hubbard導関数との216条件の
比較もASan/UBSanで通過しました。CUDA sm_80、HIPの通常版・`f3omp`版、
共通Force・軌道・solverのC++3翻訳単位のHIP/gfx942コンパイルを確認しました。
実wrapperの84呼び出し×両backend・1/2/4/8 workerは、native軌道ヘルパーも通して
NVHPCで確認しました。Intel SYCL構成もビルドし、関連22試験を通過しました。

CPU/GPUの `-runtest` 各14例は変更前後・CPU対GPU・同梱基準との5比較で `1e-7` 以内でした。
Cluster/Bandの共線・非共線、DC・DC-LNO・Krylovの9経路×CPU/GPU、core-holeの共線・
非共線、Hubbard dual、縮約Methane、Force4Bの非融合経路、GPU固有値解法からの
CPU Forceへの切り替え、高精度Hubbardの計27比較もエネルギー・全Forceが `1e-7` 以内でした。
最大差はHubbard dualのエネルギー `1.2820e-9 Ha`、Force `1.2726e-8 Ha/bohr` です。
この例を収束条件 `1e-10`・4 MPI×1 OpenMPで確認したところ、前後とも176 SCFで
エネルギー・全Forceが完全一致しました。各計算のSCF完了表示も確認し、縮約Methaneは
CPU/GPUとも18回の `G-SCF/L-SCF` の並びを照合しました。

厳密なSCF反復数の比較はGPU Cluster NCの59→51、GPU Band NCの20→16、
2 OpenMPのCPU core-hole NCの20→21の3件で不一致を残しました。対応するForce最大差は
`1.056e-9`、`5.4e-11`、`6.71e-10 Ha/bohr` です。GPUの2例では同一の旧バイナリ・入力・
coreでも前回の実行の53/19回と今回の59/20回が異なっています。反復数の不一致を
合格に書き換えず、数値の比較と区別して保存しています。標準スイートと個別経路の
並行検証中の時間は性能評価に使いません。

他の実計算・ビルド・試験を止め、18 MPI×1 OpenMPで性能を比較しました。
sidiaはRTX 5080の既存MPSに両版とも18クライアントが接続し、25 SCF・140³格子でした。
元入力の `scf.maxIter=25`、`scf.criterion=1e-13` を保持し、両版とも
`The SCF was not fully achieved` と終了しています。この測定は同条件での速度比較であり、
収束完了の検証とは区別します。C60 CPUは各3回、実行順を変えた中央値で、
全6回が27 SCF・84³格子で収束し、エネルギー・全Forceが完全一致しました。

| 条件 | 内部最大計算時間 (秒) | 起動・終了を含む実時間 (秒) | Force (秒) |
| --- | ---: | ---: | ---: |
| sidia GPU＋MPS・変更前 `d437109f` のCUDA経路 | 104.024 | 107.516 | 15.684 |
| sidia GPU＋MPS・変更後 | 103.729 | 107.069 | 15.604 |
| C60 CPU・変更前、3回中央値 | 8.690 | 10.687 | 0.202 |
| C60 CPU・変更後、3回中央値 | 8.644 | 10.737 | 0.203 |
| C60 CPU・Intel版 `~/openmx4.0`、1回 | 9.759 | 10.285 | 0.193 |

sidiaのエネルギー差は0、Force最大差は約 `1e-12 Ha/bohr` でした。Intel版との差は
エネルギー約 `1e-11 Ha`、Force約 `1.4e-11 Ha/bohr` です。今回の例では性能を維持し、
sidiaの内部計算時間は約105秒以内でした。起動・終了込みの105秒は未達です。
C60の実時間中央値は約0.05秒増えており、汎用的な高速化や全CPU経路でのIntel版との
同等性能を示す結果ではありません。Intel版との構成・MPI・出力処理の差も残ります。

CUDA/HIPのMakefileにForceの共有ヘッダ依存を追加し、ヘッダだけを変更しても
`Force.o` が更新されることを8条件のdry-runで確認しました。依存追記後の再ビルドでは
NVHPCの一時fatbin名・module IDだけが変わり、CPU命令・GPU画像・定数・配置は
測定版と同一でした。最終インストール版でもCPU/GPUのCluster実計算が完走し、
変更前とのエネルギー・全Force・SCF反復数が基準を満たしました。

GPUカーネルを含むnative補助関数はCUDAの103関数・HIPの109関数のトークン比較、保護対象5ファイルは
ハッシュで不変を確認しました。AMDの最適化済み縮約、GeForceとA100/H100の精度・
バッチ・容量方針を変更していません。AMD・A100/H100・Intel/Apple GPUの実機確認は
この環境では行っていません。検証成果物は `build/force-deep-refactor-20261021/` に保存します。

nativeの幾何・原子分配・Hubbard係数の大域参照、借用の多重ポインタ、GPU packetの確保と
一部のstatic contextは残っています。この段階でForce全体の大域変数ゼロや、
GPU数値カーネルのKokkos移行が完了したわけではありません。

## 第29段階: HNL・core-hole の共有化と不変な係数の所有

`0216ce1d` を基準に、CUDA/HIP の `Force.c` に重複していた `dHNL`、
`dHNL_SO_rows`、`dHCH`、`dHCH_SO` を分離しました。native の射影子処理は
`source/openmx_force_projectors.h` の一つの実装を両バックエンドで使い、
core-hole の係数生成とスピノル演算は `core/src/core_hole.cpp` に移しました。
HNL の加算順序と、バックエンド固有の GPU 実装・精度・容量方針を保持しています。

`CoreHole` は対象原子、スピン、spin-orbit、角運動量、状態番号、基底の配置と
penalty を明示的に受け取ります。基底数・磁気成分のオフセットを検証し、
入力メタデータを保持せず、必要な値と連続配置の係数だけを所有します。
`OpenMXSolverInvocation` が `unique_ptr` で所有し、後続の準備で所有者リストが
拡張されても公開済みビューのアドレスは変わりません。正常終了、途中終了、例外、
確保途中の失敗で自動解放し、不完全なビューを公開しません。

従来は `dHCH` の原子対ごとにコリニア係数を準備し、`dHCH_SO` の呼び出しごとに
基底配置と相対論的係数を組み直していました。新しい準備は `Force_CoreHole` 段階の
開始時に一度だけ行い、ワーカー間では不変ビューを共有します。数値ループ内では
係数の確保・初期化・所有者検索を行いません。s/p/d/f と全状態番号を扱い、
対象の基底が存在しない場合は従来どおり寄与を加えません。

正負の penalty は別々に元の除算・乗算順序で生成します。元の式には導関数の符号に
依存しない対角項があるため、penalty の反転を導関数の符号へまとめていません。
演算本体は平坦な重なりの4行と `[spin component, Cartesian direction]` の
複素 `mdspan` を受け取り、元の実部・虚部・加算順序を維持します。
MPI、SDK 型、`openmx_common.h`、Kokkos の初期化は必要としません。

native の `dHCH` は化学種の軌道数、収縮軌道数、cutoff、相対論的設定と
projector 成分数を `OpenMXAtomicViews` から受け取り、core-hole の設定を
`OpenMXCoreHoleView` から受け取ります。`dHNL_SO_rows` のスピンも引数にしました。
既存 C の多重ポインタは薄い接続で行ビューへ変換し、core-hole の C++ 演算には
渡しません。native の接続は出力引数を使い、NVHPC で問題になった大きな構造体の
値返しを追加していません。

### 第29段階の検証

コアの6テストは係数、入力の寿命、ビューの安定性、不正入力時の出力保持、
正常・途中終了・例外の寿命、すべての確保地点での失敗、並行参照を検査します。
`tests/fixtures/core_hole_spinor_before_views.c` は変更前のスピノル処理を固定した
比較対象です。実際の native 接続と C++ 演算を使う5,184条件の比較を ASan/UBSan で
実行しました。s/p/d/f、全状態、スピン0/1/3、正負の penalty、導関数の符号、
複数の同一角運動量基底、対象基底の欠如、別原子を含みます。物理式の正しさを
独立に証明する検査ではなく、従来の式との数値同等性と境界・寿命の検査です。

本体 CTest 237件、GCC ASan/UBSan・Intel C++・Kokkos CUDA のコア各203件、
SYCL の関連28件が通過しました。CUDA/HIP の native Force 寿命検査は GCC の
ASan/UBSan と NVHPC の双方で通過し、原子ビューの受け渡しも検査しました。
CUDA sm_80 と HIP の通常/代替 Force3 のコンパイル、および C++ の2翻訳単位の
HIP/gfx942 コンパイルも通過しました。

標準 S 全14例は CPU/GPU それぞれ4 MPI×1 OpenMPで変更前後を実行し、
前後・CPU対GPU・同梱参照の5比較が通過しました。同梱参照との差はエネルギー最大
`6.3e-11 Ha`、Force最大 `7.8e-11 Ha/bohr` です。Cluster/Band の共線・非共線、
DC/DC-LNO/Krylov、core-hole、Hubbard dual、縮約基底、非融合 Force と GPU から
CPU Force への切り替えの計26比較は、エネルギー・全Forceが既定の `1e-7` 以内でした。
個別計算52回はすべてSCF収束を確認しました。最大差は、収束条件 `1e-6` の
Hubbard dual のエネルギー `1.1153e-8 Ha`、Force `5.2298e-8 Ha/bohr` です。

厳密な反復数の比較は GPU Cluster NC の51→53、GPU Band NC の19→18、
CPU core-hole NC・2 OpenMP の20→21で不一致でした。Force最大差は順に
`1.05e-10`、`1.7e-11`、`3.702e-9 Ha/bohr` です。不一致の結果は保存し、
数値が許容差内であることと区別しています。他の計算を止めた性能比較は
[性能検証記録](PERFORMANCE_REGRESSION.md) に記載します。

共有化した4関数以外では、CUDA の102定義・HIP の108定義がトークン単位で
不変でした。共有化した HNL の2関数も明示的なスピン引数以外の演算は不変です。
保護対象5ソースはハッシュで不変を確認しました。AMD の最適化済み原子内チーム縮約、
GeForce と A100/H100 の既存の切り替え方針を変更していません。
AMD・A100/H100・Intel/Apple GPU の実機検証はこの環境では行っていません。
検証成果物は `build/force-projector-refactor-20261022/` に保存します。

HNL/core-hole の native 処理には、原子分配、近接リスト、幾何、Hubbard 等の
大域参照と借用の多重ポインタが残っています。今回の共有化は host の重複除去と
core-hole の数値・所有境界の分離であり、GPU 数値カーネルの Kokkos 移行や
Force 全体の大域変数ゼロは引き続き未完了です。

## 第30段階: 非局所 Force の射影子所有とスピノル演算の C++ 化

`6b57a7fc` を基準に、HNL の射影子メタデータと j 依存のスピノル演算を
`core/src/projector_force.cpp` へ移しました。`Force_HNL` 段階の開始時に
`ProjectorForce` を一度準備し、対象化学種の射影子数、角運動量、成分オフセット、
エネルギーとスピン係数を所有します。既存 `NonlocalProjectors` の配置処理を再利用し、
数値ループで原子表・エネルギー表を繰り返し引き直しません。

所有者は `OpenMXSolverInvocation` の `unique_ptr` で管理します。入力の表を後から
変更・解放しても、準備済みの係数は変わりません。所有者リストの拡張でも以前のビューは
有効です。確保途中の失敗では出力を保持し、正常終了・途中終了・例外で自動解放します。
個々の係数への数値ポインタは C の借用ビューに限り、C++ の所有に生ポインタを使いません。
MPI、GPU SDK、Kokkos の初期化、隠れた可変コンテキストには依存しません。

従来の非相対論的4ループと相対論的5ループは、原子対の矩形ブロックを C++ へ渡します。
計算ケースが所有する `OpenMXElectronicMatrixView` から入力の平坦な領域と行間隔を取得し、
射影子の `+2` の余白と halo 作業領域の異なる幅も明示します。1セルごとの C ABI 呼び出し・
検証・出力コピーを避け、各ブロックを一度検証してから全セルを処理します。
従来の行アダプターしかない互換経路は、同じ係数と C++ スピノル演算を使う回復経路を残します。

`dHNL_SO_rows` の演算本体は平坦な8行と複素出力の `mdspan` を使い、
p/d/f の実部・虚部・加算順序を保持します。スピンと角運動量ごとにテンプレートを特殊化し、
不要な分岐をコンパイル時に除去します。導関数の符号とエネルギーの反転は別々に扱い、
符号に依存しない対角項も維持します。順方向・逆方向の2項は元と同じセル内で加算し、
最後に出力へ反映します。後続の項やチャネルに不正な入力があっても、出力を途中まで
書き換えません。セルのループ内で領域確保や所有者検索を行いません。

非相対論的な積和も C++/mdspan へ移し、エネルギー・導関数・値を掛ける順序と、
元の `+=` / `-=` を保ちます。転置によって値を先に掛ける項は、順序を入れ替えません。
`Force_HNL` と `dHNL` のスピン・spin-orbit 参照も明示的なビューへ移しました。
OpenMP はビュー全体を共有し、個々のワーカーは不変の情報だけを読みます。
native の接続は出力引数を使い、大きな C 構造体の値返しを追加していません。

相対論的射影子の host 演算は、従来の式が実装されている s/p/d/f を対象にします。
それ以上の角運動量は未定義の計算へ進まず明示的に拒否します。これは PAO 基底の
g 軌道とは別の制限です。非相対論的な射影子配置は既存の最大角運動量14の検証を使います。

### 第30段階の検証

コアの6テストは配置・係数、入力の寿命、ビューの安定性、不正入力時の出力保持、
正常・途中終了・例外での解放、すべての確保地点での失敗、並行参照を検査します。
変更前の `dHNL_SO_rows` と9か所の矩形ループを固定した
`tests/fixtures/projector_force_before_views.c` を比較対象にし、実際の native 接続と
C++ 演算を31,104条件で比較しました。新しいブロック処理と行アダプターの互換経路の双方を
検査し、s/p/d/f、混在・重複する射影子、スピン0/1/3、
正負・ゼロのエネルギー、導関数とエネルギーの反転、空の化学種・行・列、異なる重なり行、
入力の行間隔と出力の余白を含みます。
ASan/UBSan 付きで通過しました。この検査は旧式との数値同等性と寿命・境界の確認であり、
物理式を独立に証明するものではありません。

CTest244件、GCC ASan/UBSan・Intel C++・Kokkos CUDA のコア各209件、SYCL関連34件が
通過しました。既存のスピン縮約検査も明示的なスピンビューを持つ fixture へ追従しました。
native Force の寿命検査は GCC ASan/UBSan と NVHPC の双方で通過しました。
CUDA sm_80・HIP の通常/代替 Force3 と、新しい射影子・solver の C++2翻訳単位の
HIP/gfx942 コンパイルも確認しました。

CPU/GPU の標準 S 各14例は変更前後・CPU対GPU・同梱参照との5比較を通過しました。
同梱参照との差はエネルギー最大約 `7.1e-11 Ha`、Force最大 `7.8e-11 Ha/bohr` です。
Cluster/Band の共線・非共線、DC/DC-LNO/Krylov、core-hole、Hubbard dual、縮約基底、
非融合 Force と CPU Force への切り替えの26比較中25組は、エネルギー・全Forceが
既定の `1e-7` 以内でした。緩い収束条件 `1e-6`、4 MPI×2 OpenMP の Hubbard dual は、
両版69 SCFで収束したものの、エネルギー差 `1.4373e-8 Ha`、Force最大差
`1.09752e-7 Ha/bohr` となり、既定のForce許容差を超えました。この失敗と元の出力を
保持し、全例合格とは扱いません。

同一の旧バイナリ・旧コア・入力を再実行した対照でもForceが `7.0614e-8 Ha/bohr`
変動しました。他の計算を止めた同じ入力・並列数の前後比較は両版69 SCF、
エネルギー差 `2.2499e-10 Ha`、Force最大差 `1.1910e-9 Ha/bohr` で通過しました。
別に `1e-10`・4 MPI×1 OpenMPとした比較は両版176 SCFで、エネルギー・全Forceが
完全一致しました。これらは元の失敗を置き換える再試行ではなく、緩い並列SCFでの
変動を調べる追加条件です。追加2組を含む28比較中27組が数値許容差内で、
個別56計算はすべてSCF収束を確認しました。

GPU Cluster NC の51→52、GPU Band NC の20→17という反復数不一致も厳密比較の
失敗として保存しました。対応するForce最大差は `1.103e-9`、`1.9e-11 Ha/bohr` です。
最初のHubbard dualの許容差超過と、これら2件の反復数不一致を区別して記録します。

共有ヘッダーを除く native 関数では `Force_HNL` だけを変更し、CUDA103定義・HIP109定義が
トークン単位で不変でした。共有 core-hole の2関数と保護対象5ソースも不変です。
AMD の原子内チーム縮約、GeForce・A100/H100 の精度・容量方針を保持しています。
他の計算を止めた性能比較は [性能検証記録](PERFORMANCE_REGRESSION.md) に記載します。
検証成果物は `build/force-hnl-refactor-20261022/` に保存します。

原子分配・近接リスト・幾何・Hubbard 設定等の native 大域参照と借用ポインタは残ります。
Force 全体の大域変数ゼロや GPU 数値カーネルの Kokkos 移行は未完了です。
AMD・A100/H100・Intel/Apple GPU の実機、L/L2/L3全例、多GPU・多ノードは再検証していません。

## 第31段階: 非局所 GPU アーカイブの段階所有と大域状態の除去

比較元は `c1ba2f0d` です。CUDA/HIP の `FHNL_gpu` を除去し、`Force_HNL` が受け取る
`OpenMXSolverInvocation` に GPU アーカイブの所有を結び付けました。準備・halo 保存・
第2ケースの保存・解放の4処理を `source/openmx_force_hnl_gpu.h` へ共有化しています。
後続の実行処理にもコンテキストを引数で渡し、ファイル内の既定状態を検索しません。

`openmx_core_solver_resources` は、既存の native 資源管理を段階の呼び出しにも提供します。
計算全体の永続グループと同じ C++ 実装を使い、同名グループを別の呼び出しへ持ち越しません。
ゼロ初期化した POD 状態はグループが所有し、同じ名前・サイズでの取得は値とアドレスを
維持します。グループはワーカー領域に入る前に取得し、個々の資源登録・解放は既存の
mutex で保護します。コア自身は MPI やデバイスランタイムを初期化しません。

非局所 GPU 処理のホスト11配列と任意のデバイスアーカイブを、この所有者へ登録します。
通常の段階終了では従来と同じ位置で明示的に解放し、途中の callback return や C++ 例外では
所有者の破棄が未解放分を回収します。`enabled` 設定前の途中状態も対象です。
SDK の同期 callback はホスト領域を含む資源の回収より先に実行し、資源グループを
通常の solver 作業領域より先に破棄します。解放時には確保したデバイスへ一時的に戻り、
CUDA/OpenACC の両デバイス選択も呼び出し側へ戻します。

CPU 経路や既存方針で GPU 段階を無効にした場合は、アーカイブのグループを作りません。
デバイス確保や登録に失敗しても、ホストアーカイブは保持して既存の容量回復処理を使います。
ホスト容量による辞退では部分的な配列を解放し、未完成の状態を公開しません。
明示的な解放の再実行、同じ段階での再準備、入れ子の別呼び出しも独立して扱います。

### 第31段階の検証

コアの6テストは正常・途中終了・例外での解放、同期と破棄の順序、solver 作業領域の寿命、
状態の再利用・整列・入れ子、不正入力時の出力保持、C++ 確保地点での失敗、並行登録を
確認します。既存の SDK モデルもテスト用ヘッダーにまとめ、実際の native アダプターを
同じモデルで検査します。

新しい `force-hnl-gpu-archives` は実際の共有処理と資源アダプターを使い、CUDA/HIP それぞれ
64条件を1〜8 OpenMP workerで実行しました。異なる化学種・軌道数、入力行の余白、
local/halo/第2ケースの保存、空の rank・射影子、容量不足、GPU 確保・登録失敗、
段階の無効化、再準備、途中終了・例外・入れ子と解放時のデバイス切り替えを含みます。
加えて双方で C++ 側の17確保地点の失敗を注入し、実際の ASan/UBSan 付きコアでも
通過しました。SDK モデルと C++ 例外による検査であり、実機の強制終了や `MPI_Abort` が
C++ の巻き戻しを行うという意味ではありません。

CTest252件、GCC ASan/UBSan・Intel C++・Kokkos CUDA のコア各215件、SYCL関連47件が
通過しました。移動した構造体を読む既存の Force packet 失敗検査も、新しい配置へ
追従しました。CUDA/HIP の容量回復検査も新しい型の配置へ追従し、それぞれ96条件を
通過しました。CUDA 側は通常の CTest にも登録しました。
ASan 検査は外部の `LD_LIBRARY_PATH` / `LD_PRELOAD` を解除し、
スタック上限を8 MiBにした環境で実行しています。CUDA sm_80・HIP の通常/代替 Force3 と、
変更したC++2翻訳単位の HIP/gfx942 コンパイルも通過しました。

CPU/GPU の標準S各14例は変更前後・CPU対GPU・同梱参照との5比較を通過しました。
同梱参照との差はエネルギー最大 `6.1e-11 Ha`、Force最大 `7.8e-11 Ha/bohr` です。
主要9経路、core-hole、Hubbard dual、縮約基底、非融合 Force と CPU Force への切り替えの
26比較に、厳しい収束条件の Hubbard 比較を加えた27組もすべて `1e-7` 以内でした。
個別54計算はすべて SCF 収束を確認しました。緩い条件 `1e-6` の Hubbard dual の最大差は
エネルギー `7.7510e-9 Ha`、Force `3.0764e-8 Ha/bohr` です。
`1e-10`・4 MPI×1 OpenMP では両版176 SCFでエネルギー・全Forceが完全一致しました。
前段階で記録した緩い並列 SCF の変動が解消したとは扱いません。

GPU Cluster NC の57→55、GPU Band NC の20→16という反復数不一致2件は、元の厳密比較の
失敗を保持しています。Force最大差はそれぞれ `8.65e-10`、`3.3e-11 Ha/bohr` です。
移動した4処理以外の native 関数では、呼び出し側の3関数だけを変更し、CUDA97定義・
HIP103定義がトークン単位で不変でした。GPU 数値カーネル、AMD のチーム縮約、
保護対象5ソースと GeForce・A100/H100 の方針も保持しています。

性能比較は [性能検証記録](PERFORMANCE_REGRESSION.md) に記載します。入力、固定した
比較前後のバイナリ・コア、全Force、コンパイル・試験・所有権の監査は
`build/force-gpu-scope-refactor-20261022/` に保存します。

Force4B の GPU 状態、他モジュールの native 資源所有者の大域エイリアス、原子分配・幾何等の
大域参照と借用多重ポインタは残ります。GPU カーネルの Kokkos 化も未完了です。
AMD・A100/H100・Intel/Apple GPU 実機、L/L2/L3全例、多GPU・多ノードは再検証していません。

## 第32段階: Force4B の GPU 作業領域と通信子の所有（2026-10-22）

`Force.c` と HIP 版の `F4B_gpu`、`F4B_stream1`、`F4B_turns1`、`F4B_stream` を廃止しました。
常駐アーカイブ、case 1 の分割処理、ランク順次処理、case 2 の分割処理は、それぞれ
明示的な `OpenMXSolverInvocation` の資源グループに属します。補助関数にはその呼び出しの
状態を渡し、別の呼び出しが同じファイル内の状態を上書きしない形にしています。

`source/openmx_force_gpu_resources.h` に7種類の配置情報・状態の型、資源アダプター、
4つの終了処理を共通化しました。ホスト配列、通常のバッチで使う一時メタデータとGPU arena、
HIP の固定ページ領域を C++ 所有者へ登録します。CUDA の `acc_malloc/acc_free` と
HIP のデバイス確保・固定ページ確保の組、ページ固定に失敗した場合のゼロ初期化した
ホスト領域への切り替えは維持しています。通常時は従来の段階で解放し、callback の
途中 return や C++ 例外では未解放分を所有者が回収します。資源の同期と確保した
デバイスへの切り替えは、前段階の所有者・アダプターの契約を使います。

ノード内・デバイス内の MPI 通信子も登録しました。ハンドルの置き場所は所有者の状態領域で、
返却済みの C スタックを破棄 callback が参照しません。`MPI_COMM_NULL` は明示的に設定し、
ビット列のゼロとは仮定しません。ランク順次処理の Begin が偽を返しても、ほかのランクのために
通信子を持つ場合があります。その状態は呼び出し側へ渡し、従来どおり同期に参加します。
CPU 経路や段階自体を無効にした場合は新しい資源グループを作りません。

GeForce のアーカイブ使用条件、A100/H100 の native FP64 方針、AMD の O(N)・APU・
固定ページ転送・チーム縮約は変更していません。通常の数値カーネルも変更していません。
共有ヘッダーへ移した終了処理と、20個の Force4B 関数以外では、CUDA73定義・HIP79定義が
トークン単位で不変でした。HIP の旧ホスト確保・解放2関数は共通アダプターへ置き換えました。
保護対象の5ソースも、この段階の比較元 `5579586f` から不変です。

### 第32段階の検証

`force-vna-gpu-resources` は実際の CUDA/HIP の準備・保存・終了処理と C++ 所有者を
SDK モデルで接続します。双方の各224条件を1/2/4/8 OpenMP workerで検査し、各57か所の
C++ 確保失敗も注入しました。4種類の状態、空のランク、混在する軌道数、アーカイブの配置、
第1・第2ケースの寿命の切り替え、容量不足、GPU確保・登録失敗、再準備、入れ子、
途中終了・例外、デバイス復元、計算しないランクの通信子を含みます。
実際の ASan/UBSan 付きコアでも通過しました。MPI は非ゼロの null ハンドルを持つモデルであり、
実機の `MPI_Abort` が C++ の巻き戻しを行うという検査ではありません。

既存の CPU 縮約を比較対象とする保存・分割処理・ランク順次処理の数値検査も、
CUDA/HIP 双方で ASan/UBSan を付けて通過しました。FP32 使用の有効・無効を検査し、
HIP の常駐経路では各432比較と、常駐・分割の144組のビット一致を確認しました。
既存の3つの CUDA 分割処理検査も通常の CTest に登録しました。

変更前の CTest253件中11件は、環境変数レジストリへの移行にテストが追従しておらず
失敗していました。抽出した処理にも実際の `openmx_env.c` とヘッダーを接続し、
レジストリの真偽値規則に沿って検査するよう修正しました。最終の CTest257件はすべて通過しました。
CUDA sm_80 と HIP の通常/代替 Force3 の4コンパイルも通過しています。

Cluster/Band の共線・非共線、DC/DC-LNO/Krylov の主要9経路は、CPU/GPU の前後18比較で
エネルギー・全Forceが `1e-7` 以内でした。個別36計算の SCF 収束を確認しています。
GPU Cluster NC の48→51、GPU Band NC の19→20という反復数の不一致2件は、厳密比較の
失敗として保存しました。エネルギー最大差は `1.5402e-10 Ha`、Force最大差は
`7.08e-10 Ha/bohr` です。同じ変更前バイナリ・コアを再実行した対照でも、それぞれ
48→56、19→18と変動しました。この対照で元の失敗記録を置き換えてはいません。

標準Sの CPU/GPU 各14例も、前後・CPU対GPU・同梱参照との5比較を通過しました。
同梱参照との差はエネルギー最大 `6.1e-11 Ha`、Force最大 `7.8e-11 Ha/bohr` です。
性能の測定値と適用範囲は [性能検証記録](PERFORMANCE_REGRESSION.md) に記載します。
入力、固定したバイナリ・コア、全Force、厳密比較の失敗、ソースの照合は
`build/force-vna-scope-refactor-20261022/` に保存します。

[大域状態の監査](GLOBAL_STATE_AUDIT.md) の設定298個・構造表278個の一括移行は、
今回の対象に含めません。NVIDIA版のファイル内 static は451→447個となり、HIP版でも
同じ4つの状態を除去しました。共通ヘッダーの1,051オブジェクトと12,912参照組は不変です。
所有権の移行後に利用一覧・ベースラインを更新しました。

一時的な GPU アーカイブの admission lease や worker の packet、他モジュールの
大域エイリアス、原子分配・幾何の大域参照と借用多重ポインタは残ります。
GPU カーネルの Kokkos 化、AMD・A100/H100・Intel/Apple GPU 実機、L/L2/L3全例、
多GPU・多ノードでの再検証も未完了です。

## 第33段階: Force 回復パケットの C++ 所有と一時アーカイブの寿命（2026-10-22）

比較元は `e6663ece` です。Force4B と非局所 Force の容量回復経路に残っていた
ホスト9配列の手動確保・解放を、バッチ単位の `ForceVnaRecoveryPacket` と
`ForceHnlRecoveryPacket` へ移しました。浮動小数点値、密度、項目記述子、射影子数と
CSR オフセットは、それぞれ型付きの連続 `Tensor` が所有します。多重ポインタ表は作りません。
バッファを借りる C callback の正常終了、途中 return、C++ 例外で所有者を破棄し、
確保途中で失敗した場合も、完成していた配列を回収してから C 側へ失敗を返します。

`source/openmx_force_recovery.h` に、両バックエンドのパッキング、タスクの組み立てと
MPI 参加制御を共通化しました。数値カーネルは既存の CUDA/HIP 実装を使い、加算順序、
FP32/FP64 の選択と AMD のチーム縮約は変更していません。256項目と値領域の上限、
単独で上限を超える項目の扱い、ローカル GPU・ほかのランク・CPU への回復順序も維持します。
空の rank や処理済みの rank も、ほかの rank が必要とする集団通信へ参加します。
callback は通信子を引数で受け取り、CPU 完了を示すタスクの返り値0も成功として扱います。
パケットは同期したタスクが完了してから、次の回復 collective より前に破棄します。

CUDA の `Force4BArchiveLease` は、呼び出しの資源グループ内の安定した記述子になりました。
入場制御を取得した直後に登録し、GPU の確保・転送途中の失敗や callback の途中終了でも、
確保したデバイスでバッファを解放してから入場制御を返します。呼び出し側のデバイス選択も
復元します。通常時は従来どおり、回復 collective へ入る前に明示的に終了します。
一時ポインタを常駐 `Force4BGpuContext` に代入する必要もなくなりました。
記述子・登録の確保に失敗した場合は、既存の回復処理へ辞退します。
GeForce と A100/H100 の選択方針、HIP の常駐・分割・ランク順次実行は維持しています。

コアの所有者に CUDA/HIP、MPI、Kokkos 実行環境の初期化は必要ありません。
ホスト領域の `Tensor` / mdspan と、ネイティブ実行の SDK アダプターを分けた境界は、
将来の Intel/Apple GPU 経路でも同じパケット所有を利用するためのものです。
これらの GPU バックエンドを実装したという意味ではありません。

### 第33段階の検証

CTest263件が通過しました。新しいコア6テストは、空・最大項目数、独立したパケット、
不正な形状、途中 return・例外、入れ子、並行呼び出しと9か所の確保失敗を検査します。
GCC ASan/UBSan、Intel C++、Kokkos/CUDA でも、この6件を通過しました。
サニタイザー検査では外部の `LD_LIBRARY_PATH` / `LD_PRELOAD` を解除し、スタック上限を8 MiBにしました。
実際の C++ 所有者と native 回復処理を接続した確保失敗9条件も、漏れや dispatch がないことを
確認しました。ASan/UBSan を付けた実際のコアとの接続も通過しています。

CUDA/HIP の回復パッキングは、17項目と513項目の各96比較で、従来の縮約と照合しました。
halo、可変射影子数、非局所第2ケースの3種類の微分、FP32/FP64、ローカル・ほかの rank・
CPU の回復を含みます。空・処理済み rank に集団通信の参加が必要な条件も検査しました。
HIP の常駐経路では FP32 有効・無効の各432比較と、常駐・分割144組のビット一致も通過しました。
これらは SDK/MPI のモデルによるホスト検査です。
実際のアプリケーションでも、容量回復を強制した共線 Si クラスターの4 MPI実行で、
変更前後の SCF 収束と反復回数の一致を確認しました。エネルギー最大差 `1.5e-11 Ha`、
Force最大差 `1e-12 Ha/bohr` で、数値許容差内です。

一時アーカイブは実際の C++ 資源所有者を使い、104寿命条件、128の GPU 種別・設定条件、
12か所の C++ 確保失敗を通過しました。正常・辞退・同期/転送/確保失敗、再準備、重複準備、
途中 return・例外、デバイス復元と入場制御の解放順序を含みます。実際の ASan/UBSan 付き
コアでも通過しました。MPI_Abort やプロセス強制終了の巻き戻しを保証する検査ではありません。

CUDA sm_80 と HIP SDK のホスト OpenMP の、通常/代替 Force3 の4コンパイルを通過しました。
native 関数は変更した2つの CUDA batch 実行と共通ヘッダーへ移した処理を除き、
CUDA86定義・HIP96定義がトークン単位で不変です。容量回復の GPU 数値カーネル、
core-hole の native 本体と保護対象5ソースは不変です。

主要9経路は CPU/GPU の前後18比較で、エネルギー・全Forceが `1e-7` 以内でした。
個別36計算の SCF 収束を確認し、最大差はエネルギー `4.4070e-9 Ha`、
Force `2.61e-10 Ha/bohr` です。GPU Cluster NC の53→54、GPU Band NC の17→19という
反復数不一致2件は、厳密比較の失敗として保存しました。同じ変更前バイナリ・コアの再実行でも
53→50、17→21と変動しましたが、元の失敗をこの対照で置き換えてはいません。

標準Sの CPU/GPU 各14例は、前後・CPU対GPU・同梱参照との5比較を通過しました。
同梱参照との差はエネルギー最大 `6.8e-11 Ha`、Force最大 `7.8e-11 Ha/bohr` です。
性能の値と適用範囲は [性能検証記録](PERFORMANCE_REGRESSION.md) に記載します。
固定したバイナリ・コア、入力、全Force、厳密比較の失敗、監査・試験の成果物は
`build/force-recovery-refactor-20261022/` に保存します。

[大域状態の監査](GLOBAL_STATE_AUDIT.md) の分類は不変で、設定298個・構造表278個を
変更していません。共通ヘッダー1,051個・12,912参照組、ファイル内 static 447個も不変です。
所有権移行後に利用一覧・ベースラインを更新しました。
環境変数レジストリも移動した profile 読み手の参照元一覧に追従させ、値の解釈は維持しました。

worker の一時 GPU view・packet、Force3 の native GPU 領域、ほかのモジュールの大域別名、
幾何等の大域参照と借用多重ポインタは残ります。GPU カーネルの Kokkos 移行、
AMD・A100/H100・Intel/Apple GPU 実機、L/L2/L3全例、多GPU・多ノードの再検証は未完了です。

## 第34段階: 非局所 Force の通常 GPU バッチを所有化し、ホスト処理を共通化

`Force_HNL_GpuCase1Run_Part` / `Force_HNL_GpuCase2Run_Part` の通常バッチを対象にしました。
実際の呼び出し経路を調べ、未使用の `Force4B_PairDistribute` ではなく、常駐アーカイブからの
縮約と回復経路の両方で使うホスト領域を移しています。

- `core/force_hnl_batch` が、ケース別の原子・近接・プロジェクター情報、密度重み、力を
  `OverwriteTensor` の連続領域で所有します。従来の10 / 11個の手動確保を、各ケース6個の確保にまとめました。
  行メタデータと3成分の力は形状を持つ2次元領域です。既存の Kokkos/mdspan、または利用可能な環境の
  `std::mdspan` に対応します。上書き前提の領域に `make_unique_for_overwrite` を使い、
  余分なゼロ初期化を除きました。他モジュールの `Tensor` の初期化契約は変えていません。
- C ABI は一時的な平坦ビューを callback に渡します。割り当て途中の失敗、callback の失敗返却、
  C++例外でも所有領域を破棄します。終了 callback が GPU 処理を終えてからホストのビューを失効させます。
  コア単体では MPI・CUDA・HIP・Kokkos ランタイムを初期化しません。
- `source/openmx_force_hnl_batch.h` に数え上げ、密度重み・ローカル/halo 行のパッキング、
  減衰・HNL補正、原子別の力への加算を集約しました。幾何・近接表・スピン設定は明示的な借用入力です。
  従来の大域別名を読む境界を `_Part` の2関数に残し、共通 callback と GPU dispatch は入力を受け取ります。
- 一時 GPU アリーナは、既存の `OpenMXNativeResources` に登録します。通常は同期ダウンロード後に
  解放し、途中で抜けた場合だけ終了 callback がデバイスを同期してから解放します。
  C のスタック記述子を資源グループに登録しません。容量不足と登録失敗は従来の回復経路へ進みます。
- `Force4BGpuItem` のワイヤ配置を `OpenMXForcePairItem` に一本化しました。
  CUDA の OpenACC 縮約、HIP の128スレッド・チーム内軌道積の並列縮約、FP64演算式と加算順序、
  GeForce とデータセンター向けの入場制御・精度の方針は維持しています。

検証は次の範囲で行いました。

- 全 CTest **270/270**。新しいコアの6試験は、GCC、GCC ASan/UBSan、Intel C++、
  Kokkos CUDA の各構成で通過しました。入れ子・並行呼び出し、空入力、添字上限、
  callback の失敗・例外、6個すべての部分確保失敗を含みます。配列 `new[]` もテスト用の
  故障注入に登録し、ASan でも配列確保の失敗・解放を数えられるようにしました。
- CUDA/HIP それぞれ **580例** の SDK モデル試験で、移行前 `ea8d1fad` から記録した全Forceに一致しました。
  スピン0/1/3、通常/平坦ソルバー、ローカル/halo、減衰導関数、常駐/容量不足/ホスト回復、
  大域/外部出力、空ランクを確認します。各バックエンドで6確保失敗・10転送失敗も試し、
  途中の同期、デバイス復帰、ホストとGPUの解放を確認しました。ASan/UBSan を有効にした
  実際の C++ コアをリンクした同じ試験も通過しています。AMD実機の試験を代替するものではありません。
- CUDA の通常/`f3omp` と ROCm SDK Clang による HIP のホスト OpenMP 通常/`f3omp` の4構成で
  `Force.c` をコンパイルしました。CUDA/HIP の2つの GPU 縮約ループは字下げ以外不変です。
  その他の native 関数はCUDA86個・HIP94個、core-hole本体と保護対象5ソースは不変です。
- 既存の FP32/FP64、回復パケット、HIP 常駐/ストリーム試験も通過しました。
  HIP の常駐縮約は精度ごと432比較、常駐/ストリームの144組はビット一致です。
- 主要9経路の CPU/GPU 前後18比較で、エネルギーと全Forceが `1e-7` 以内でした。
  個別36計算は SCF 収束し、最大差はエネルギー `9.3200e-10 Ha`、Force `1.015e-9 Ha/bohr` です。
  GPU Cluster NC の54→52、Band NC の17→20は反復数不一致として厳密比較の失敗を保存しました。
  同じ変更前バイナリ・コアの対照でも、それぞれ54→52、17→18の変動を確認しています。
  この対照結果で元の失敗を置き換えてはいません。
- 標準Sの CPU/GPU 各14例は、前後・CPU対GPU・同梱参照の5比較を通過しました。
  GPU の容量回復を強制した8原子Si・4 MPIでも、17 SCFで収束し、前後差は
  エネルギー `2.00e-12 Ha`、Force `1.00e-12 Ha/bohr` 以内でした。

18 MPI の性能結果は [性能検証記録](PERFORMANCE_REGRESSION.md) に記載します。
最終版の固定バイナリ・コア、入力、全Forceと比較・性能記録は
`build/force-hnl-batch-refactor-20261022/final-validation/` に保存します。
初版の測定・比較結果、最初の失敗を含む試験、監査とソースの最終照合は上位ディレクトリに保存します。

[大域状態の監査](GLOBAL_STATE_AUDIT.md) の分類（設定298個・構造表278個）、
共通ヘッダー1,051個・12,912参照組、ファイル内 static 447個は不変です。
幾何と近接表の大域実体・借用多重ポインタ、GPU worker の一時 view、Force3 の native GPU 領域は
残ります。GPU カーネルの Kokkos 化、AMD・A100/H100・Intel/Apple GPU 実機、
L/L2/L3全例、多GPU・多ノードは今回再検証していません。

## 第35段階: 共通GPU読み取りビューとデバイス間転送のRAII化

比較元は `baf33d56` です。Hamiltonian、密度・軌道格子、密行列変換・密度行列で使う
`openmx_gpu_views.c` を `openmx_gpu_views.cpp` へ移し、CUDA/HIP 共通の所有境界を設けました。
ネイティブ側は C++17 でコンパイルし、C の関数インターフェースを維持します。
このモジュールのために C++20 対応の NVIDIA コンパイラは要求しません。

- `OpenMXGpuReadView` は C++ からは move-only な所有者です。同じGPU・peer access は借用し、
  コピーが必要な場合だけ専用のデバイス領域を所有します。通常の明示的な解放に加え、
  途中 return・例外でもデストラクターが解放します。C からは従来どおり明示的に解放します。
- 所有領域には確保元のデバイスを記録します。呼び出し側が別GPUへ移っていても、
  確保元で解放してから呼び出し側のデバイスを復元します。同じGPU間の明示的コピーも、
  呼び出し時のデバイスが異なる場合に正しいデバイスへ切り替えます。
- 転送中のSDKエラーやホスト確保失敗では、一時ホスト領域と準備途中のGPU領域を回収します。
  C ABI を C++ 例外が通過しないようにし、準備失敗の出力は空にします。
  source の所有と、producer の同期は従来どおり呼び出し側の責任です。
- peer の共有方針と16 MiB上限の分割転送を維持します。一時ホスト領域は `unique_ptr` で所有し、
  ダウンロードで全要素を上書きするため、追加のゼロ初期化を行いません。
  同一GPU・peer借用の経路には新しい確保や同期を加えていません。
- C/C++ の配置を別々の翻訳単位で照合しました。LP64ではデバイス番号が従来の末尾パディングに入り、
  構造体サイズは16バイトのままです。C++でのコピーは拒否し、move代入では既存の所有領域を先に解放します。

CUDA/HIP のビルドルール、ソース選択と単独試験のビルダーを新しい拡張子へ追従させました。
単独ビルダーの環境変数読み取りには、実際の `openmx_env.c` もリンクします。
SDK の呼び出しは既存の中立な runtime 別名を使用し、GPU固有の数値カーネルは変更していません。
Intel/Apple GPU の実行アダプターを追加したという意味ではありません。

### 第35段階の検証

- CTest **271/271** が通過しました。新しい C++17 / ASan / UBSan 試験は、実際の転送実装を
  独立したSDKモデルへ接続します。C11の別翻訳単位からも同じAPIを呼び、配置と明示的解放を検査します。
  借用・所有・move・例外、別GPUからの解放、16 MiB超の末尾チャンク、14か所のSDK失敗、
  2条件のホスト確保失敗、デバイス復元の一度の失敗と再試行、320回の並行ビューを確認しました。
- CUDA/HIP の密行列ワーカーのCPUモデルは、それぞれ **13,694比較** を通過しました。
  実際のRTX 5080では単一GPUの密行列 **4,480比較**、Hamiltonian・密度格子の **78条件** と
  出力範囲 **96検査** を通過し、後者の Compute Sanitizer はエラー0件でした。
  ワーカーの途中失敗、結果の公開と動的チームも検査しています。
- CUDA `sm_80` のビルドと、ROCm SDK Clang による HIP のC++17ホストOpenMPコンパイルを通過しました。
  Cから使うHamiltonianワーカーの単独ビルダーもコンパイル・リンクを確認しました。
  そのソース抽出が古い戻り値型を仮定していた問題と、HIP試験の環境変数読み取りの未解決リンクを修正しました。
  初回の失敗ログも保存しています。これらはAMD・A100実機や物理的な多GPUの検証を代替しません。
- 主要9経路のCPU/GPU前後 **18比較** で、エネルギー・全Forceが `1e-7` 以内でした。
  個別36計算はSCF収束し、最大差はエネルギー `7.1481e-9 Ha`、Force `1.864e-9 Ha/bohr` です。
  GPU Cluster NC の52→48、Band NC の20→17は反復数不一致による厳密比較の失敗として保存しました。
  同じ変更前バイナリ・コアの対照でも52→47、20→16となり、すべて収束しています。
  この対照で元の失敗を置き換えてはいません。
- 標準SのCPU/GPU各14例は、前後・CPU対GPU・同梱参照との5比較を通過しました。
  同梱参照との差はエネルギー最大 `9.4e-11 Ha`、Force最大 `7.8e-11 Ha/bohr` でした。

設定298個・構造表278個の一括移行とは範囲を分けています。
大域状態監査の共通ヘッダー1,051個・12,912参照組、ファイル内static447個は不変です。
C専用監査の対象は290→289翻訳単位、3,032→3,027関数となりますが、これは共通ビューを
C++へ移したためであり、大域変数や5関数を削除したという意味ではありません。
所有権の検証後に利用一覧・ベースラインを再生成しています。

性能検証は [性能検証記録](PERFORMANCE_REGRESSION.md) に記載します。
固定バイナリ・コア、入力、全Force、最初の失敗を含む試験・比較・監査結果は
`build/gpu-view-refactor-20261022/` に保存します。L/L2/L3全例、AMD・A100/H100・Intel/Apple GPU実機、
多GPU・多ノードは再検証していません。Cの呼び出し側の明示的解放、ほかのworkerの一時領域、
Force3のnative GPU領域と、大域実体・借用多重ポインタ、GPUカーネルのKokkos移行は残ります。

## 第36段階: 密度行列・密行列変換の一時GPU領域を型付きRAIIへ移行

比較元は `e5f0699e` です。共通の `openmx_gpu_dm.cu` と `openmx_gpu_transform.cu` に残っていた
SDK領域の手動確保・解放を、C++17の `openmx::gpu::DeviceBuffer<T>` へ移しました。
密度行列の詰め直した固有ベクトル、添字、重み、位相、出力と、密行列変換の出力が対象です。

`source/openmx_gpu_memory.hpp` が、型付きの連続領域・要素数・確保元のデバイスを所有します。
`unique_ptr` とデバイス用deleterで実装し、管理用のホスト確保や追加のゼロ初期化を行いません。
空の所有者はSDKを呼ばず、要素数と型サイズの積の桁あふれや、使用中の領域への重複確保を拒否します。
所有権はmoveでだけ渡せます。明示的な `reset` とスコープ終了の両方で、確保元のGPUで解放して
呼び出し側のデバイスを復元します。通常のRELEASEで空にするため、その後のデストラクターは二重解放しません。
GPU領域の消費が完了してから解放するというワーカー実行器の同期契約は維持します。

前段階の `DeviceBinding` と `DeviceFree` はこのヘッダーへ移し、読み取りビューでも同じ実装を
使います。移動した2型の実装はバイト単位で不変です。デバイス領域の確保数・サイズ、
部分結果を全ワーカーの成功後に公開する順序、GPU不足時の行単位CPU回復を維持しています。
COMMIT中の失敗は従来どおり実行器の致命的エラーであり、公開後の巻き戻しを保証する変更ではありません。
幾何・近接表やSCF状態の新しい大域実体は置きません。CUDA/HIPの呼び出しはコンパイル時の別名で、
GPUカーネルのKokkos化やIntel/Apple GPUバックエンドの追加は今回の範囲に含みません。

CMakeのソース投影とHIP試験のstagingを `.hpp` に対応させ、両Makefileに依存関係を追加しました。
CUDA/HIPのどちらでも同じヘッダーを使用し、ネイティブ部分にC++20を追加要求しません。

検証は次の範囲で行いました。

- CTest **272/272** が通過しました。新しいC++17 / ASan / UBSan試験はCUDA/HIPそれぞれで、
  空・move・再利用・型サイズの桁あふれ、2条件の確保失敗、例外、256回の並行所有を確認します。
- 実際の密度行列ワーカーにSDK失敗 **80か所**・ホスト確保失敗 **18か所**、密行列変換に
  SDK失敗 **9か所**・ホスト確保失敗 **2か所**を注入しました。公開前の失敗で出力が不変となり、
  ホスト/GPUの領域とデバイス選択が回収・復元されることを確認しました。
  通常のRELEASEを省いて実行器が戻る条件でもスコープ所有を検査し、遅いGPU OOMからのCPU回復も通過しました。
  これは独立したSDKモデルによる検査で、GPU実機の致命的故障を模擬するものではありません。
- 既存のCUDA/HIP密行列CPUモデルは各 **13,694比較** を通過しました。
  RTX 5080の単GPU密度行列経路は **4,480比較** を通過し、Compute Sanitizerはエラー0件です。
  複数GPU専用の密行列変換・spinor・scatterはモデルで検証し、実機では再検証していません。
- CUDA `sm_80` のビルドと、ROCm SDKの `gfx942` / `gfx1200` 向けの両モジュール計4コンパイルを
  C++17で通過しました。HIPオブジェクトのデバイスコード格納も確認しました。
  SDKのホストコンパイルだけに限った検査ではありませんが、AMD実機での動作検証は未実施です。
- 数値カーネル5本、補助・公開関数18個と保護対象5ソースはバイト単位で不変です。
  主要9経路のCPU/GPU前後18比較で、エネルギー・全Forceは `1e-7` 以内でした。
  個別36計算はSCF収束し、最大差はエネルギー `1.3860e-9 Ha`、Force `1.215e-9 Ha/bohr` です。
  GPU Cluster NCの48→51、Band NCの17→21は反復数不一致による厳密比較の失敗として保存しました。
  同じ変更前バイナリ・コアの対照でも48→50、17→21となり、すべて収束しています。
- 標準SのCPU/GPU各14例は、前後・CPU対GPU・同梱参照の5比較を通過しました。
  同梱参照との差はエネルギー最大 `1.83e-10 Ha`、Force最大 `7.8e-11 Ha/bohr` です。

設定298個・構造表278個の分類は維持しています。所有権の検証後に監査の一覧・ベースラインを更新し、
監査対象の289翻訳単位・3,027関数、共通ヘッダー1,051個・12,912参照組、全staticの実体は不変でした。
今回はC++側の所有だけを移しており、C専用監査の対象から追加の関数を外していません。

性能は [性能検証記録](PERFORMANCE_REGRESSION.md) に記載します。固定バイナリ・コア、入力・全Force、
最初の試験失敗を含む検証・監査結果は `build/gpu-worker-memory-refactor-20261022/` に保存します。
最初のモデル試験ではCOMMIT中の確保失敗まで未公開を要求していたため、公開前の故障注入へ修正しました。
AMD・A100/H100・Intel/Apple GPU実機、L/L2/L3全例、多GPU・多ノードは再検証していません。
ネイティブの常駐キャッシュ・Force3・Cワーカーの残る一時領域と、大域実体・借用多重ポインタの移行は残ります。

## 第37段階: GEMMul8の常駐GPUキャッシュの所有化

比較元は `b6579f88` です。CUDA/HIPの `gemmul8_bridge.cu` が保持するデバイス・ストリーム別の
作業領域を、前段階のC++17の `DeviceBuffer<unsigned char>` へ移しました。
CUDAのFixed変換で再利用するprepared行列も同じ所有者が管理します。CUDA/HIPの演算方針は
別々に保持し、デバイス確保・解放・一時的なデバイス切り替えを共通化しています。

所有者は確保元のGPUとバイト数を記録します。キャッシュのヒット時に管理用のホスト確保や
所有者によるSDK照会を追加せず、領域のゼロ初期化も行いません。通常の明示解放に加え、
所有する表の破棄と例外によるスコープ終了でも領域を回収します。消費側の処理を同期してから
解放するという従来の呼び出し契約は維持します。

`allocate_status` と `try_reset` はSDKの状態を返します。明示解放が失敗した場合は所有権を
保持するため、作業領域を後で解放し直せます。解放に成功してデバイス復元だけが失敗した場合は
先に所有権を消費し、復元の再試行による二重解放を防ぎます。デストラクターの解放は従来の
一時所有者と同じbest effortです。CUDAのprepared行列を拡張する際は、解放失敗を無視して
古いポインタを捨てず、有効フラグを落として回復経路へ戻します。

キャッシュ表への登録が `std::bad_alloc` で失敗しても、確保関数は既存の容量不足を返し、
ネイティブBLAS等の従来の回復経路へ戻します。任意の外部ライブラリ例外を捕捉する変更ではありません。
予約メモリ・容量上限・ランク共有率、A100の大行列閾値、GeForceとMI300A/Radeonの選択、
prepared行列のバージョン・形状・精度の判定は維持します。表・設定・カウンターの大域実体を
新しいコンテキストに移す作業は残っています。今回もGPUカーネルのKokkos化やIntel/Apple GPU
実装を追加した段階ではありません。

HIP固有のbridgeを改修したため、`sources.json` では `overrides` と `import_sha256` の
未変更取り込み元から `integrated_overrides` へ移しました。選ぶソースは同じで、元の取り込みは
Git履歴に残ります。取り込み元のハッシュを改修後の値で上書きしていません。
両Makefileは共通ヘッダーへの依存を記録します。新しい所有機構にC++20は必要ありません。

検証は次の範囲で行いました。

- CTest **274/274** が通過しました。SDKスタブを共通fixtureへ整理した後も、影響する2試験を
  再実行して通過しました。GPUを持たない環境でもキャッシュとアーキテクチャ選択を検査できます。
- 実際のCUDA/HIP bridgeをC++17 / ASan / UBSanで検査しました。バックエンドごとに
  作業領域のSDK準備失敗 **6か所**、管理用ホスト確保失敗 **4か所**を注入しています。
  再利用・拡張、GPU/stream別の分離、256 MiB超のtrim、異なるGPUからの解放、
  解放失敗後の再試行、例外による表の破棄、予約量・共有率による辞退を確認しました。
  所有者の解放・バインド・復元失敗4条件も検査し、CUDAのprepared行列では有効性・バージョンの
  維持と無効化、拡張、解放の再試行、管理領域不足、GPU別の回収を確認しました。
  SDKモデルによる検査であり、GPU実機の致命的故障を再現したものではありません。
- 既存のA100/GeForce選択試験は、大行列の自動選択、境界値、環境変数の優先順位、
  容量不足、4 GPUモデルへの精度伝達の6条件を通過しました。
  RTX 5080の実数・複素数 **39 GEMM** は、CPUのlong double参照に対する最大相対誤差
  `3.702e-13` でした。fast/accurate、N/T/C、paddingと長いKの分割を含みます。
  prepared行列も左右・実数/複素数・N/T/C・fast/accurateの24条件で、初回準備・再利用・内容更新後の
  再準備の **72 GEMM** をCPU参照と照合しました。準備48回・再利用24回・容量回復0回を確認し、
  最大相対誤差は `4.188e-14` です。解放後の常駐バイト数も0となりました。
- CUDA `sm_80`、HIP `gfx942` / `gfx1200` のbridgeを実SDKでC++17コンパイルしました。
  HIPは必要なBLASLtヘッダーを検証済みアーカイブから補って再実行しています。
  AMD・A100/H100実機での動作と性能を測った結果ではありません。
- 両bridgeの補助・演算・方針関数 **68個** と保護対象5ソースはバイト単位で不変です。
  CUDAのforward演算関数はpreparedポインタの取得だけを所有者経由へ変更しています。
- 主要9経路のCPU/GPU前後 **18比較** はエネルギー・全Forceが `1e-7` 以内で、個別36計算は
  SCF収束しています。最大差はエネルギー `1.4050e-9 Ha`、Force `1.063e-9 Ha/bohr` でした。
  GPU Cluster NCの51→59、Band NCの21→19は反復数不一致による厳密比較の失敗として保存します。
  同じ変更前バイナリ・コアの対照も51→48、21→18となりました。対照で元の失敗は置き換えていません。
- 標準SのCPU/GPU各14例は前後・CPU対GPU・同梱参照の5比較を通過しました。
  同梱参照との差はエネルギー最大 `6.1e-11 Ha`、Force最大 `7.8e-11 Ha/bohr` です。

設定298個・構造表278個の分類を維持し、所有権の試験後に監査を更新しました。
C専用監査の289翻訳単位・3,027関数、共通ヘッダー1,051個・12,912参照組、staticと環境変数は
revision以外の全項目が不変です。Cの関数を監査対象から外したことによる減少はありません。

性能は [性能検証記録](PERFORMANCE_REGRESSION.md) に記載します。固定バイナリ・コア、入力・全Force、
SCF不一致と対照、SDK試験の初回失敗を含む記録は `build/gemmul8-cache-refactor-20261022/` に保存します。
初回の試験コンパイルで不足していた標準ヘッダー・環境変数実装へのリンク、モデルの型名衝突を修正しました。
AMD・A100/H100・Intel/Apple GPU実機、L/L2/L3全例、多GPU・多ノードは再検証していません。
Force3、Cワーカーの一時領域、残る常駐キャッシュと大域実体・借用多重ポインタの移行を続けます。

## 第38段階: 範囲タスクのGPUワーカーと所有権の分離

比較元は `c2e9322f` です。範囲タスクのGPU側を `openmx_gpu_task_worker.cpp` へ切り出し、
呼び出し内の `TaskWork` がGPU領域・未公開の一時結果・CSRの正規化添字を所有するようにしました。
デバイス領域は共通の `DeviceBuffer`、ホスト領域は `openmx_host_buffer.hpp` のmalloc用deleterを
持つ `unique_ptr` が管理します。新しい所有者のための追加のホスト確保や、数値領域のゼロ初期化はありません。
キャッシュ・計算設定の大域実体も追加していません。

C側の `openmx_gpu_task.c` は制御、通信のpack/remote、CPU回復を担当します。
配置・サイズ・CSRのコピー・結果の公開は `openmx_gpu_task_layout.h` のC/C++共通実装です。
CPU回復の演算順序、最大64 MiBの作業分割、通信パケットの定義と16-byte alignmentを維持します。
制御側は共通のC ABIからGPUワーカーを呼びます。GPUカーネルは元のC/OpenACC・C/OpenMP targetの
ままで、HIPのteam/thread分割とGeForce向けの容量・ロック方針を変更していません。

明示的なRELEASEでは確保元のGPUで解放します。失敗時は所有権を保持し、途中終了・C++スコープの
巻き戻りでも所有者が回収します。実行器のCOMMIT・RELEASE失敗が致命的である契約は維持しており、
公開途中の巻き戻しや任意の外部例外からの回復を保証する変更ではありません。
公開前の失敗では全ワーカーの成功を待ってから結果を公開する従来の契約を保ちます。

両MakefileにC++17のワーカーと依存ヘッダーを登録し、HIPの共通ソース一覧を更新しました。
C++用のlinkage指定を加えた公開ヘッダーは、不変取り込み元から `integrated_shared` へ移しています。
C++20をネイティブ部分へ追加要求していません。ホスト用の所有者はGPU SDKから独立していますが、
本体のIntel/Apple GPU対応やGPUカーネルのKokkos化を実装した段階ではありません。

既存のC数値試験はCのままで、SDK呼び出しのモデルを介して実際のC++ワーカーへ接続しています。
本番の所有者をテスト内で生ポインタへ置き換えません。HIPのHamiltonian/Force実機試験も、
C製カーネルのコンパイル設定を保ち、所有者へのテスト専用C ABIで接続します。

検証は次の範囲で行いました。

- CTest **275/275** を通過し、試験側の環境変数実装へのリンクを整えた後も対象3試験を通過しました。
- C++17 / ASan / UBSanで、CUDA/HIPそれぞれ **18か所のSDK準備失敗** と
  **6か所のホスト確保失敗** を注入しました。dense/CSR/lookupの混合入力、scratch、固定出力の
  reduction、ガード、解放処理を省く途中終了、実行器の例外によるスコープ解放を検査しています。
  確保元と違うGPUからの解放と、解放失敗後に所有権を保って再試行する条件も通過しました。
- CUDA/HIPの従来のCPUモデルは各、範囲タスク96、疎DM回復1,281、energy/dipole等336、
  grid/radial/VNA 144、Hamiltonian 336、Force回復96×2、VNA projector 32比較を通過しました。
  HIPのForce4B residentはFP32設定ごとに432比較と、resident/stream 144結果の同一性を通過しています。
- RTX 5080の単GPU実行では、通常GPU、強制CPU、GPU容量不足、実際のcudaMallocの遅いOOMからの
  CPU回復の4条件が通りました。タスクのcudaMalloc/cudaFreeを数え、各条件の終了時に未解放領域が
  0となることを検査します。複数GPU・別ノードを必要とする条件は単GPUの成功件数に含めません。
  Compute SanitizerはMPIをホスト通信に限定すると不正アクセス/API診断がなくなりますが、
  NVHPC OpenACCのlaunch引数用領域5,120 bytesが1件残ります。同じ試験を変更前のCワーカーでも
  実行し、同じ1件・同じサイズになることを確認しました。検査全体をエラー0とは記録していません。
- HIPワーカーの `gfx942` / `gfx1200` 向けC++17コンパイル、実機試験用C++ probeのホストコンパイル、
  Hamiltonian/ForceのC製OpenMP target試験の `gfx942` コンパイルを実SDKで通過しました。
  AMD実機での実行・性能検証ではありません。
- CPU回復とpackの3関数はバイト単位で不変、配置と公開の5関数はinline指定とC/C++共通のcastだけの
  変更です。制御・remoteはローカルワーカーの呼び出し名だけが変わっています。
  通信パケット、容量実行器・collective、両Force/Hamiltonianと保護対象5ソースは不変です。
- 主要9経路のCPU/GPU前後18比較はエネルギー・全Forceが `1e-7` 以内で、個別36計算は収束しています。
  最大差はエネルギー `1.5070e-9 Ha`、Force `1.597e-9 Ha/bohr` でした。
  GPU Cluster NCの59→46、Band NCの19→18は反復数不一致の厳密比較失敗として保存します。
  同じ変更前バイナリ・コアの対照でも59→51、19→17となりました。対照で元の失敗は置き換えていません。
- 標準SのCPU/GPU各14例は前後・CPU対GPU・同梱参照の5比較を通過しました。

設定298個・構造表278個の分類を維持し、所有権の試験後に監査を更新しました。
C専用監査の対象は289翻訳単位・3,022関数です。ワーカー3関数とヘッダーの2関数がC++側へ
移ったためCの関数数が5減りました。大域実体の削除による減少ではありません。
共通ヘッダー1,051個・12,912参照組、static・環境変数とすべての分類は不変です。

固定バイナリ・コア、入力・全Force、初回失敗を含む試験と監査は `build/task-worker-refactor-20261022/`
に保存します。初回のC++接続で試験用MPIヘッダーに必要なstdlibと環境変数実装へのリンクを補いました。
スタック無制限環境のASan起動失敗と、単GPUでは成立しない複数ランク条件を試した失敗も記録しています。
メモリ検査と性能の条件・結果は [性能検証記録](PERFORMANCE_REGRESSION.md) に記載します。
AMD・A100/H100・Intel/Apple GPU実機、L/L2/L3全例、多GPU・多ノードは再検証していません。
ベクトル縮約、CPUタスクのscratchと通信パケット、残る常駐キャッシュの所有権移行を続けます。

## 第39段階: ベクトル縮約のGPUワーカーと型付き所有

比較元は `f850fe2a` です。ホスト側の履歴を使う重み付きGram積と線形結合のGPU側を、
`openmx_gpu_vector_worker.cpp` へ切り出しました。`VectorWork` がGPU arenaと未公開の部分結果を
型付き所有者で管理します。GPU側は `DeviceBuffer<double>`、ホスト側は前段階のmalloc用deleterを
持つ `unique_ptr` を使い、追加のホスト確保・数値領域のゼロ初期化はありません。

arena内の添字は各フェーズで位置から求め、重複して保持していた生ポインタ表を除去しました。
重み付きGram積だけで添字ビューを作り、線形結合では作りません。確保サイズと配置、結果を全ワーカーの
成功後に公開する順序、GPU容量不足からの再試行・通信・CPU回復は維持しています。
空のローカル呼び出しはSDK・ヒープを使わず、桁あふれした容量要求は確保前に辞退します。

C側の `openmx_gpu_vector.c` は制御、通信パケット、CPU回復を担当します。
C/C++はC ABIで接続します。CPUの演算順序と通信形式は不変です。GPUのEXECUTE本体は、
CUDAのOpenACC、HIPのteam/thread reductionを含めバイト単位で移動しました。
HIPの既存の長いベクトルに対する並列分割や、GeForce/A100の容量・占有・精度方針を変更していません。
通常のRELEASEに加え、実行器が解放を省いて戻る場合と、C++スコープの巻き戻りでも回収します。
実行器のCOMMIT・RELEASE失敗を致命的とする契約は保ち、公開途中の巻き戻しを保証していません。

両MakefileとHIPの選択一覧にC++17のモジュールを登録しました。CUDAのGPUループにはNVHPC C++の
OpenACCを使い、HIPにはClang C++のOpenMP targetを使います。元の最適化・GPU指定・オフロード設定を
引き継ぎ、C専用の言語・診断指定だけを置き換えます。C++20をGPU部分へ追加要求しません。
公開ヘッダーのC++ linkage指定も加え、不変取り込み元から `integrated_shared` へ移しています。

試験用のC/C++接続とSDK・ヒープ失敗モデルは、前段階のタスクと共通化しました。
タスク試験が読み取る既存のCモデルのprefixは両バックエンドとも不変です。HIP実機試験では
residentのCカーネルをCのまま、ワーカーをC++17で構築し、テスト専用C ABIで接続します。
GPUの数値試験は必須オフロードを維持しています。

検証は次の範囲で行いました。

- CTest **277/277** が通過しました。従来のベクトル容量回復試験も登録し、C製の制御・通信・
  CPU回復と実際のC++ワーカーを接続する96条件を毎回検査します。HIP版の96条件も通過しました。
- C++17 / ASan / UBSanで、CUDA/HIPそれぞれ、Gram積のSDK準備失敗 **24か所**・ホスト確保失敗
  **3か所**、線形結合のSDK準備失敗 **18か所**・ホスト確保失敗 **3か所**を注入しました。
  公開前の出力不変、ガード、解放を省く途中終了、例外による回収、確保元GPUへの復元と解放の再試行を
  確認しました。共有モデルへ移したタスクの18 SDK・6ホスト失敗の試験も両バックエンドで通過しました。
- RTX 5080の実GPUで **48構成・296,400比較** をCPUのlong double参照と照合し、最大相対誤差は
  `3.87894e-16` でした。両演算、0/1要素、長さの端数、padding、添字の並べ替え、相殺を含みます。
  構成ごとにGPU確保・解放の釣り合いを検査しています。この専用試験はMPIを初期化しません。
  Compute Sanitizerの不正アクセス・APIエラー・リークはすべて **0件** でした。
- CUDA `sm_80` をC++17 / OpenACCでコンパイルしました。HIPは `gfx942` / `gfx1200` のワーカー、
  `gfx942` のC++実機probeとC製resident比較試験を実SDKでコンパイルしました。
  OpenMP targetオブジェクトへのデバイスコード格納も照合します。AMD実機での実行・性能検証ではありません。
- CPU・制御はローカルワーカーの呼び出し名だけの変更、pack・サイズ検証・結果公開の3関数と通信パケットは
  バイト単位で不変です。容量計算はvoid castだけを変更し、GPUのEXECUTE本体・pragma、
  容量実行器・collective、`Mixing_H.c` と保護対象5ソースを保っています。
- 主要9経路のCPU/GPU前後18比較はエネルギー・全Forceが `1e-7` 以内で、個別36計算は収束しています。
  最大差はエネルギー `1.2800e-10 Ha`、Force `1.418e-9 Ha/bohr` でした。
  GPU Cluster NCの46→55、Band NCの18→16は反復数不一致の厳密比較失敗として保存します。
  同じ変更前バイナリ・コアの対照も46→66、18→21となりました。対照で元の失敗を置き換えていません。
- 標準SのCPU/GPU各14例は、前後・CPU対GPU・同梱参照の5比較を通過しました。
  同梱参照との差はエネルギー最大 `6.1e-11 Ha`、Force最大 `1.1e-11 Ha/bohr` です。

設定298個・構造表278個の分類を維持し、所有権の試験後に監査を更新しました。
C専用監査は289翻訳単位・3,017関数です。ワーカー3関数とヘッダーの2関数がC++側へ移ったため、
Cの関数数はさらに5減りました。大域実体の削除数ではありません。共通ヘッダー1,051個・12,912参照組、
static・環境変数と分類は不変で、新しいワーカーに可変な大域実体はありません。

性能は [性能検証記録](PERFORMANCE_REGRESSION.md) に記載します。固定バイナリ・コア、入力・全Force、
SCF不一致と対照、実機とモデルの範囲は `build/vector-worker-refactor-20261022/` に保存します。
GPUカーネルのKokkos化とIntel/Apple GPUの本体実装を追加した段階ではありません。
AMD・A100/H100・Intel/Apple GPU実機、L/L2/L3全例、多GPU・多ノードは再検証していません。
CPUタスクのscratchと通信パケット、残る常駐キャッシュの所有権移行を続けます。

## 第40段階: CPUタスクの作業領域をスコープ内の所有へ移行

比較元は `0c559947` です。範囲タスクのCPU回復を `openmx_gpu_task_cpu.cpp` へ切り出しました。
CSRの正規化、固定長の縮約出力とscratchを入れるarenaは、各チャンクのmalloc用所有者が管理します。
Cの制御・通信から内部C ABIで呼び出し、数値カーネルはCのまま使います。追加のヒープ確保と
数値領域のゼロ初期化はありません。C++17のホストモジュールで、GPU SDKに依存しません。
CUDA/HIPに共通のCPU回復であり、将来のバックエンド固有のSDK状態も持ち込みません。

64 MiBを超える分割可能な作業範囲の半減と、確保失敗時の半減を保ちます。分割不能な1要素が上限を
超える場合は従来どおり確保を試し、それも確保できなければ同じ診断でabortします。桁あふれした
要求はmallocへ渡しません。空タスクは確保しません。scratchの無い非空タスクが1 byteを確保する
既存の方針も維持しています。チャンク順、CSR配置とコピー、入力の参照、縮約順序は不変です。

カーネル呼び出しがC++の例外で巻き戻る場合もarenaを回収します。CPUカーネルは直接出力へ書くため、
途中の出力を巻き戻す契約は追加していません。本体のC呼び出し元が例外から回復できるという保証でもありません。
GPUワーカー、通信パケットとremote側の検証・公開、GPU容量・占有・精度方針を変更していません。

既存のC数値モデルはCのまま、実際のC++ CPUモジュールとGPUワーカーへリンクします。
HIPのresident Force試験ではCPU実装のテキスト抽出を廃止し、実際のCPUモジュールへ接続しました。
HIP実機用Force/Hamiltonian試験もCカーネルとC++ CPU回復を別々に構築し、必須オフロードを維持します。

検証は次の範囲で行いました。

- CTest **278/278** が通過しました。新しいSDK不要のC++17 / ASan / UBSan試験では、dense・CSR・
  lookup・scratch・縮約を独立したlong double参照と照合し、入力不変と出力ガードを確認しました。
  初回の確保失敗と容量上限からの範囲縮小、40 MiB×3要素の分割、64 MiB超の単一要素、空タスクと
  scratch無しも検査しています。カーネルが出力を書いた後に例外を投げても、未解放領域は残りません。
  桁あふれ時の確保0回と、単一要素のOOM時のSIGABRT・診断も別プロセスで検査します。
- CUDA/HIPのCモデルはそれぞれタスク96、疎DM1,281、energy336、grid144、Hamiltonian336、Force192、VNA32条件を
  通過しました。HIPのresident/streamの432比較もFP32設定の有効・無効それぞれで通り、144結果がbit一致です。
  SDKを模擬したCPU上の試験であり、AMD実機での性能検証ではありません。
- CPUモジュールはGCC・NVHPC・ROCm Clangの3コンパイラでC++17として、GPU SDKのinclude無しで
  構築しました。HIP実機用のC製Force/Hamiltonian試験は実SDKでgfx942向けにコンパイルし、デバイスコードの
  格納を確認しました。C/C++の部分リンクでCPU bridgeが解決することも確認しています。
  AMD向け本体の最終リンク・実機実行を通過したという意味ではありません。
- 通信パケット・pack・remoteはバイト単位で不変、制御はCPU呼び出し名だけの変更です。
  CPU容量計算はバイト単位で不変、CPU本体は所有者・get・解放と関数名だけを変更しました。
  GPUワーカー・共通配置・実行器・collectiveと保護対象5ソースは不変です。
- 主要9経路のCPU/GPU前後18比較はエネルギー・全Forceが `1e-7` 以内で、個別36計算は収束しています。
  最大差はエネルギー `5.9300e-10 Ha`、Force `5.0700e-10 Ha/bohr` でした。
  GPU Cluster NCの55→53、Band NCの16→17は反復数不一致の厳密比較失敗として保存します。
  同じ変更前バイナリ・コアの対照も55→61、16→19となりました。対照で元の失敗を置き換えていません。
- 標準SのCPU/GPU各14例は、前後・CPU対GPU・同梱参照の5比較を通過しました。
  同梱参照との差はエネルギー最大 `6.1e-11 Ha`、Force最大 `7.8e-11 Ha/bohr` です。

設定298個・構造表278個の分類を維持し、所有権の試験後に監査を更新しました。
C専用監査は289翻訳単位・3,015関数で、CPUの2関数がC++へ移った分だけ対象から減っています。
大域実体の削除数ではありません。共通ヘッダー1,051個・12,912参照組、static・環境変数と分類は不変です。
新しいCPUモジュールに可変な大域実体はありません。

性能は [性能検証記録](PERFORMANCE_REGRESSION.md) に記載します。固定バイナリ・コア、入力・全Force、
SCF不一致と対照、初回の試験失敗は `build/task-cpu-refactor-20261022/` に保存します。
初回の単体実行では無制限stackとASanのshadow領域が競合したため、8 MiBのstackで再実行しました。
両バックエンドの模擬MPIヘッダーにはabortの宣言用includeを補い、HIPは更新したprojectionで数値モデルを通過しています。
GPUカーネルのKokkos化とIntel/Apple GPUの本体実装を追加した段階ではありません。
AMD・A100/H100・Intel/Apple GPU実機、L/L2/L3全例、多GPU・多ノードは再検証していません。
通信パケットと残る常駐キャッシュの所有権移行を続けます。

## 第41段階: 範囲タスクの通信パケットを所有するモジュールへ移行

比較元は `964e3658` です。Cの制御とMPI通信を保ち、範囲タスクのpack/remoteをSDK不要の
`openmx_gpu_task_packet.cpp` へ切り出しました。C++17のmalloc用所有者が生成中のパケットと
ワーカー実行中の結果を管理し、C ABIで戻るときに通信側へ所有権を渡します。通常のGPU失敗でも
確保した結果はC側へ渡し、通信側がcommitせずfreeする契約を保ちます。ワーカーがC++例外で
巻き戻る場合は未返却の結果を回収しますが、MPI全体の例外回復を保証する契約は追加していません。

wireヘッダーはC/C++共通の内部ヘッダーへ移し、plain C layoutとtrivial性をコンパイル時に検査します。
calloc領域上のheaderはplacement newで寿命を開始します。既存のcallocによるゼロ埋めを保ち、
追加確保・追加初期化はありません。16-byteの配置、パラメーター、dense/CSR/lookupのコピー、overflow・CSR・長さの検証は
所有者とC++ cast以外の変更をしていません。C制御はcallback名だけの変更です。CPU回復、GPUワーカー、
collectiveのプロトコル、Force/Hamiltonian、保護対象5ソースとデバイス別の方針も不変です。

既存のC数値モデルはCのまま、実際のC++パケット・CPU回復・GPUワーカーをリンクします。
HIP実機用のC製Force/Hamiltonian試験にもC++のホストモジュールを登録しています。
GPUカーネルにC++20以降の要件を追加せず、パケットモジュールにもCUDA/HIPのSDK状態を持ち込みません。

検証は次の範囲で行いました。

- CTest **280/280** が通過しました。SDK不要のC++17 / ASan / UBSan試験では、独立したCの既知データと
  パケット全byteを照合し、そのCパケットを解釈したdense/CSR/lookup・縮約の値とcommitの範囲も確認しました。
  切り詰めた全 **1,136 prefix** は実際にその長さの領域を渡して検査し、追加の **22不正条件** とともに
  確保・ワーカー実行前に拒否します。pack/resultのcalloc失敗、空タスク、通常失敗時のCへの所有権移譲と、
  private出力を書いた後の例外による回収を検査しました。
- 実MPI **6プロセス・3ノード模擬の11条件** が通過しました。実際のC制御・通信、C++パケットとCPU回復を
  接続し、複数の依頼元、ヘルパー辞退、pack/result/受信のOOM、単一要素、空タスク、GPU無効化・同一ノードの
  回復を検査します。管理対象のホスト確保は各rank・各条件の終了時に回収でき、縮約・入力不変・ガードも
  照合しました。NVHPCのMPI wrapperと、GCCへ切り替えたwrapperのUBSanで同じ11条件が通っています。
  デバイス実行と物理ノード分類だけを模擬しており、実GPUによるノード間の容量回復試験ではありません。
- CUDA/HIPのCモデルはそれぞれタスク96、疎DM1,281、energy336、grid144、Hamiltonian336、Force192、
  VNA32条件を通過しました。両バックエンドの数値カーネルをC++として再解釈せずに接続しています。
- CPU/packetの両ホストモジュールをGCC・NVHPC・ROCm ClangでC++17コンパイルしました。
  HIP実機用C試験はgfx942向けのデバイスコードを格納し、C/C++の部分リンクでCPU/pack/remoteのbridgeが
  解決しました。AMD向け本体の最終リンク・実機実行を通過したという意味ではありません。
- 主要9経路のCPU/GPU前後18比較はエネルギー・全Forceが `1e-7` 以内で、個別36計算は収束しています。
  最大差はエネルギー `1.7060e-9 Ha`、Force `4.6600e-10 Ha/bohr` でした。
  GPU Cluster NCの53→62、Band NCの17→19は反復数不一致の厳密比較失敗として保存します。
  同じ変更前バイナリ・コアの対照も53→51、17→20となりました。対照で元の失敗を置き換えていません。
- 標準SのCPU/GPU各14例は前後・CPU対GPU・同梱参照の5比較を通過しました。
  同梱参照との差はエネルギー最大 `6.1e-11 Ha`、Force最大 `7.8e-11 Ha/bohr` です。

設定298個・構造表278個の分類を維持し、所有権の検証後に監査を更新しました。C専用監査は
289翻訳単位・3,010関数で、移動した2関数とCから使わなくなったheaderの3関数分が減っています。
大域実体の削除数ではありません。大域実体・参照組・staticの数と環境変数名は不変で、新しいC++側に
可変な大域実体はありません。読み取り専用のcallback表は同じ実体のまま新しいC ABI名を参照します。
監査記述に含まれるそのinitializer名の変更も明示して照合しました。

性能は [性能検証記録](PERFORMANCE_REGRESSION.md) に記載します。固定バイナリ・コア、入力・全Force、
SCF不一致と対照、初回の試験失敗は `build/task-packet-refactor-20261022/` に保存します。
初回の空タスク試験はテスト側のCSR位置を修正しました。MPI試験のUBSanは対応compilerで選択する形にし、
NVHPCのwrapperへ未対応のsanitize指定を強制しません。
AMD・A100/H100・Intel/Apple GPU実機、L/L2/L3全例、多GPU・物理多ノードは再検証していません。
ベクトル通信パケットと残る常駐キャッシュの所有権移行を続けます。

## 第42段階: ベクトル通信パケットの所有とヘッダーの初期化

比較元は `b0d78c6f` です。ベクトルのサイズ計算・pack/remoteをSDK不要の
`openmx_gpu_vector_packet.cpp` へ切り出し、C/C++共通の内部wireヘッダーを使います。
生成中のパケットとワーカー実行中の結果をmalloc用所有者で管理し、C ABIで戻る時点で通信側へ渡します。
通常の失敗で返す結果もCがcommitせずfreeする契約を保ち、例外では未返却の結果を回収します。
MPI全体の例外回復を追加する契約ではありません。

従来のheaderにはmallocで未初期化のまま送られるtail paddingがありました。現在の24-byte headerは
全体をゼロにしてからtrivialなオブジェクトの寿命を開始し、値を設定します。追加ヒープ確保と数値payloadの
初期化・コピーはありません。余白だけの初期化を変更し、wire配置と各field・payloadの値は維持しています。
以前の任意の余白を持つパケットも解釈できます。overflow・長さ・pair indexの検証は不変です。

Cの制御・CPU算術はcallback名だけの変更、commitはバイト単位で不変です。GPUワーカー、範囲タスク、
collectiveプロトコル、保護対象5ソースとGeForce/A100/AMDの方針も保ちます。C側で不要になった
OpenACC/OpenMPヘッダーを除去しました。GPUカーネルのC++20要件は追加していません。

検証は次の範囲で行いました。

- CTest **282/282** が通過しました。SDK不要のC++17 / ASan / UBSan試験では、重み付きGram積と線形結合の
  独立したC wireデータと全byteを照合しました。malloc領域を事前にpoisonしてheaderの余白ゼロも検査し、
  旧headerの任意の余白も受理することを確認しています。数値出力は既知の値と照合します。
  両モードで **152/120通り** の切り詰めをその長さの領域で検査し、追加の **15/10不正条件**、
  overflow・不正範囲、malloc/calloc失敗、空タスク、通常失敗の所有権移譲と例外による回収を確認しました。
- 実MPI **6プロセス・3ノード模擬** でベクトル各モード11条件、既存タスク11条件が通過しました。
  管理対象のホスト領域は各rank・各条件で回収でき、数値・入力不変・ガードを検査します。
  共通の転送・確保失敗fixtureとcompiler/launcherを整理し、NVHPC wrapperとGCC wrapperのUBSanで
  同じ33条件を再確認しています。デバイス実行と物理ノード分類だけを模擬し、実際のC制御・MPI通信と
  C++パケットを接続しています。実GPUによる物理ノード間の容量回復試験ではありません。
- CUDA/HIPのCモデルはそれぞれベクトル96、タスク96、疎DM1,281、energy336、grid144、Hamiltonian336、
  Force192、VNA32条件を通過しました。7種類のタスク試験が読み取るprefixは両バックエンドとも不変です。
- packetモジュールはGCC・NVHPC・ROCm ClangでC++17構築しました。HIP実機用C試験とC++ probeは
  gfx942のデバイスコードを格納し、C/C++の部分リンクでpack/remote/localを解決しました。
  AMD向け本体の最終リンク・実機実行を通過したという意味ではありません。
- 主要9経路のCPU/GPU前後18比較はエネルギー・全Forceが `1e-7` 以内で、個別36計算は収束しています。
  最大差はエネルギー `1.9590e-9 Ha`、Force `2.4400e-9 Ha/bohr` でした。
  GPU Cluster NCの62→56、Band NCの19→20は反復数不一致の厳密比較失敗として保存します。
  同じ変更前バイナリ・コアの対照も62→47、19→15となりました。対照で元の失敗を置き換えていません。
- 標準SのCPU/GPU各14例は前後・CPU対GPU・同梱参照の5比較を通過しました。

設定298個・構造表278個の分類を維持し、所有権の検証後に監査を更新しました。
C専用監査は289翻訳単位・3,007関数で、移動した3関数分が減っています。大域実体の削除数ではありません。
大域実体・参照組・staticの数と環境変数名は不変で、新しいC++側に可変な大域実体はありません。
読み取り専用callback表の実体は維持し、監査記述に含まれる新しいinitializer名だけを明示して照合しました。

性能は [性能検証記録](PERFORMANCE_REGRESSION.md) に記載します。固定バイナリ・コア、入力・全Force、
SCF不一致と対照は `build/vector-packet-refactor-20261022/` に保存します。
AMD・A100/H100・Intel/Apple GPU実機、L/L2/L3全例、多GPU・物理多ノードは再検証していません。
この段階でKokkos化や大域状態ゼロを達成したという意味ではありません。次はC++/CUDA側を含む
残る永続状態と、明示的な計算所有者への移行を優先します。

## 第43段階: コリニアクラスターの永続状態を計算コンテキストへ移行

比較元は `2f697a70` です。CUDA/HIPの `Cluster_DFT_Col` が定義していた永続メタデータを、
明示的な `OpenMXCoreContext` の資源グループへ移しました。CUDAの21変数、HIPの32変数を
計算ごとのPOD状態に保持し、解放先を指す可変なモジュール別名も除去します。さらに診断用の
静的ポインタを自動変数へ変え、HIPのGPU切替閾値は設定レジストリからその都度読みます。
結果としてソルバー内の可変な大域定義はCUDAで23個、HIPで35個減りました。
両ファイルは同時にリンクされるものではなく、共通ヘッダーの大域参照は依然として残ります。

`openmx_core_native_resources_state` は、ゼロ初期化したPODとその解放コールバックを一括して
所有します。登録が失敗したときは出力を変えず、未公開のコールバックを呼びません。
解放時にはそのグループと状態を渡し、メタデータ・バッファが有効なうちに、ロックを保持せず
従来の同期・解放処理を実行します。古い `void(void)` コールバックのC ABIは、未移行モジュールのため
維持します。計算中の暗黙の既定コンテキスト、TLS、可変な大域状態ポインタは追加していません。

DFTの入口で借用する不透明な `OpenMXClusterColState` を、本体、深部補助関数、精度制御とForce前の
解放へ明示的に渡します。run境界の解放は資源グループのコールバックで行い、破棄中に状態を取得し直す
処理はありません。バッファとSDKハンドルは既存のC++資源所有者が回収し、状態内のポインタは借用です。
精度改善ソルバーの確保コールバックにもuserdataを追加し、未移行ソルバーの旧コールバックを保ちます。
SCF間の再利用、初期値の `-1` / `1`、union BLACS grid、デバイス復元とAMDのhost退避方針を維持しています。

CUDAの変換済みXは、計算ごとの世代番号だけでは別計算のGEMMul8キャッシュと衝突します。
prepared-Xのキーへ所有元の同一性を加え、状態を破棄する前にその所有元のエントリーを解放します。
従来APIのキー空間は独立しており、他の計算やband用キャッシュを選択的な解放で無効にしません。
GEMMul8のキャッシュ表・設定・カウンターそのものの大域状態は、この段階では残ります。

検証は次の範囲で行いました。

- CTest **289/289** が通過しました。実コアの通常版・ASan/UBSan版で、所有元を受け取る解放の
  初期値・再利用・別名・2計算の独立性・途中失敗・例外・解放順序・二重解放を12試験で確認しました。
  CUDA有効コアとIntel C++のSERIALコアでも同じ12試験を通過し、ホスト側の所有権を確認しています。
- 両バックエンドの実際の型定義・初期化・精度制御・深部解放処理を使うSDKモデルで、各100回の
  二重コンテキスト試験を通過しました。確保コールバックの登録失敗、CUDAの有限回再試行、
  Force前の解放、union grid、次の入力への切替も確認します。GPUと1 rankのMPI/BLACS操作は模擬です。
  実OpenMX全体の同時実行可能性を保証する試験ではありません。
- CPU密度行列のスレッド数・分割・部分電荷と大きな添字、cuSOLVERのGeForce/高速FP64方針、
  精度改善の故障注入・反復解放も通過しました。実GPUのGEMMul8 forward試験は229条件を通過し、
  同じ世代番号を持つ別の所有元、旧API、選択的解放と所有元の番地再利用を数値で確認しました。
- CUDAの85関数、HIPの100関数は、所有権受け渡しの変更を逆変換した後の演算・制御トークンが一致します。
  HIPの切替閾値取得だけが静的キャッシュから通常の読み取りに変わり、その入力値・既定値を別途試験します。
  精度改善のCUDA/HIP実装も、確保の呼び出し変更を逆変換すると全トークンが一致します。
- HIPのcollinear/noncollinear cluster・band、DFT、Free_ArraysのCオブジェクトと、gfx942デバイスコードを
  含む精度改善ソルバーをROCm Clangで構築しました。AMD向け本体の最終リンクや実機試験ではありません。
- 主要9経路のCPU/GPU前後18比較は、エネルギー・全Forceが `1e-7` 以内です。最大差は
  `3.0600e-10 Ha` / `1.2230e-9 Ha/bohr`、個別36計算はいずれも収束しています。
  GPU Cluster NCの56→55、Band NCの20→17は反復数不一致の厳密比較失敗として保存します。
  同じ変更前バイナリとコアの対照も56→50、20→19でした。元の失敗は対照で置き換えていません。
- 標準SのCPU/GPU各14例は前後・CPU対GPU・同梱参照の5比較を通過しました。
  実GPUの追加試験は、FP32有効/無効入力・直列スピン・prepared-X再利用を繰り返す4入力を前後で実行し、
  すべて収束しました。最大差は `2.8500e-10 Ha` / `3e-12 Ha/bohr` です。
  メモリ受入れを辞退させる分散GS2のspin-union gridも前後17 SCFで収束し、
  エネルギー差 `1.301e-11 Ha`、Force差ゼロでした。

所有権の検証後に監査を更新しました。C専用監査は289翻訳単位・3,011関数、共通大域実体1,051個・
12,912参照組と設定298個・構造表278個を維持しています。ファイルstaticは447→432、関数内staticは
386→380です。外部大域名は358→357で、`Hex`は別モジュールにも定義があるため、当ファイルの2定義除去が
一意な名前の集計では1減になります。CからC++へ演算を隠して数を減らした変更ではありません。

性能は [性能検証記録](PERFORMANCE_REGRESSION.md) に記載します。入力・固定した本体とコア、全Force、
厳密比較・対照と試験は `build/cluster-col-state-refactor-20261022/` に保存します。
残る6ソルバーの別名、共通ヘッダーの大域状態、C++/CUDA/HIP内のキャッシュ・設定も引き続き移行対象です。
AMD・A100/H100・Intel/Apple GPU実機、L/L2/L3全例、多GPU・物理多ノードは再検証していません。

## 第44段階: ノンコリニアクラスターと密度格子のキャッシュ連携を明示化

比較元は `15fc760f` です。CUDAの22変数、HIPの31変数を、計算コンテキストが所有する
`OpenMXClusterNonColState` へ移しました。可変な資源グループ別名、診断用の静的ポインタと
HIPのGPU切替閾値キャッシュも整理し、ソルバーが定義する可変大域実体はCUDAで24個、HIPで34個減りました。
別々のバックエンドの定義数を足して、単一バイナリの削減数とはしません。共通ヘッダーの大域参照は残ります。

DFTはcollinear/noncollinearの借用状態を `OpenMXNativeSolverStates` で受け取り、ソルバー本体・
深部補助関数・精度制御・Force前の解放・post-SCF scatterとMAGMA経由のEDMへ明示的に渡します。
公開する内部C APIはSDK不要のヘッダーへ集めました。状態・解放コールバックの一括所有は第43段階の
コアAPIを使い、SCF/MD間の再利用と入力境界の解放を維持します。状態を取得し直す解放処理や、
既定コンテキスト・TLS・可変な大域状態ポインタはありません。今回コア共有ライブラリは不変です。

密度格子のGPUメモリ受入れ処理は、最後の固有ベクトルをホストへ退避させるため、従来は大域ソルバーへ
直接アクセスしていました。現在はDFTが所有元付きの `OpenMXDensityCacheHooks` を作り、
`Set_Density_Grid` からGPU serviceへ借用します。メモリ不足時だけ、その計算のコールバックを呼びます。
密度格子側は状態やコールバックを保持せず、ワークスペース返却→空き容量の再確認→固有ベクトル退避→
CPUへの回復という従来の手順を保ちます。別の計算のキャッシュを破棄する暗黙の依存を除去しました。

CUDAのarena・OpenACC対応表・変換済みoverlap・cuSOLVER作業領域、HIPのdevice solve・page-locked host
退避・MI300Aの保持方針、DM/EDM・プロファイル・精度制御も同じ計算の状態として扱います。
バッファとハンドルは既存のC++資源所有者が回収し、POD内のポインタは借用です。
Cの演算、GPU転送・カーネル、精度・容量・デバイス別の方針と保護対象5ソースを維持しています。
GPUコードへ新しいC++標準の要件を追加していません。

検証は次の範囲で行いました。

- CTest **291/291** が通過しました。両バックエンドの実際の状態定義・初期化・精度制御・
  密度格子からの退避・深部解放をコンパイルし、実コアで各100回の二重コンテキスト試験を行いました。
  ASan/UBSanで、SCF間の再利用、確保を許さない解放、次の入力への切替、退避要求の所有元と
  ホスト側の固有ベクトル保持を確認します。SDKと1 rankのMPIは模擬で、OpenACC転送は実行しません。
  実OpenMX全体の同時実行・物理多GPUの保証ではありません。
- CPUの疎密変換は3種類の非対称な配置を照合しました。HIPのhost back-transformはCPU BLASと
  GPUショートカットの12形状を通過し、CPU形の最大相対差は `8.13e-16` です。
  MI300Aとその他GPUの既定値・明示上書き・保持/解放・GEMM診断48条件も通過しました。
  精度選択のCUDAモデルと、HIPの4ソルバーの入力切替/バンドcycle試験も通っています。
- CUDAの83関数、HIPの103関数は、所有権受け渡しだけを逆変換すると演算・制御トークンが一致します。
  関数ヘッダーと本体の間にマクロ定義があるXANESも含めて確認しました。HIPの閾値取得だけが
  プロセス共通の静的キャッシュから通常の読み取りへ変わり、閾値・既定値・parserを保ちます。
- HIPのNC、DFT、Free_Arrays、両密度格子ファイルを実SDKで構築しました。NCと2つの密度格子ファイルは
  gfx942向けOpenMP targetでもコンパイルし、`.llvm.offloading` を持ちます。
  AMD向け本体の最終リンクや実機検証ではありません。
- 主要9経路のCPU/GPU前後18比較はエネルギー・全Forceが `1e-7` 以内です。最大差は
  `2.0691e-9 Ha` / `1.0910e-9 Ha/bohr` で、個別36計算はすべて収束しています。
  GPU Cluster NCの55→50、Band NCの17→20は反復数不一致の厳密比較失敗として保存します。
  同じ変更前バイナリ・コア・入力の対照も55→51、17→16でした。元の失敗は置き換えていません。
- 標準SのCPU/GPU各14例は、前後・CPU対GPU・同梱参照の5比較を通過しました。
  GPUのNC小行列を明示的に有効化した精度改善経路と、密度格子のメモリ不足を強制する経路も前後で収束し、
  最後のFP64解き直しを確認しました。最大差は `1.399e-11 Ha` / `4.8e-11 Ha/bohr` でした。
  こちらの40→41、41→42も反復数不一致として保存しています。

C専用監査は289翻訳単位・3,015関数で、共通大域実体1,051個・12,912参照組、設定298個・構造表278個、
環境変数251名を維持しています。ファイルstaticは432→412、関数内staticは380→376へ減り、外部大域名は
357で不変です。C++へ演算を移して解析範囲を減らした数ではありません。所有権の検証後にベースラインを更新しました。

性能は [性能検証記録](PERFORMANCE_REGRESSION.md) に記載します。入力・固定本体/コア・全Force・
厳密比較と対照・故障試験は `build/cluster-noncol-state-refactor-20261022/` に保存します。
残る5ソルバーの別名、共通ヘッダーとC++/CUDA/HIPの大域状態も引き続き対象です。
AMD・A100/H100・Intel/Apple GPU実機、全L/L2/L3、多GPU・物理多ノードは再検証していません。

## 第45段階: バンド状態の計算所有と非共線DCの出力契約

比較元は `3a4799a5` です。collinear/noncollinear bandの永続メタデータ、k点のoverlap・固有ベクトル、
精度改善の二段階処理、容量・プロファイル・診断の状態を、計算コンテキストが所有するPODへ移しました。
移した変数と、資源別名等を含む可変な大域定義の削減数は次のとおりです。バックエンド間の数は合算しません。

| バックエンド・経路 | 計算状態へ移した変数 | 除去した可変大域定義 |
| --- | ---: | ---: |
| CUDA collinear band | 51 | 52 |
| CUDA noncollinear band | 39 | 42 |
| HIP collinear band | 51 | 53 |
| HIP noncollinear band | 41 | 44 |

DFTの `OpenMXNativeSolverStates` は両バンドの借用状態も保持します。通常の呼び出し、FP64の最終解き直し、
mixing後の制御とサイクル終了、深部補助関数、精度改善の確保コールバックまで同じ状態を渡します。
NEGFの初期バンド計算にもDFTから渡し、API宣言はSDK不要のopaque型のヘッダーへ集めました。
C++資源所有者がバッファ・ハンドルとPODを回収し、PODのポインタは借用です。既定コンテキストやTLS、
可変な大域状態ポインタによる取り直しはありません。コア共有ライブラリは第44段階と同じです。

SCF/MD間の再利用、常駐・直列化・k点が多い経路の精度改善、k点ごとのホスト保存、overlapと固有ベクトルの
容量制限を保ちます。HIPの閾値取得は同じparser・既定値・過剰共有時のCPU切替を通常の読み取りへ変えました。
HIP NCに残っていた `atexit` の登録・フラグ・終了関数は除去し、計算所有者の解放へ統一しています。
状態の消滅後に動くプロセス終了コールバックへ、借用ポインタを残しません。読み取り専用の初期値と
文字列ポインタ表も局所化しました。GPUコードのC++標準要件と、GeForce/A100/H100・AMDの数値方針は維持します。

検証中にNC DCのGPU経路とNC DC-LNOのCPU経路で300秒のタイムアウトを検出しました。初回候補とスタックを
保存しています。DC-LNOの再実行も60秒で止まり、一部rankがSCF終了後の `OM_dual`、別rankが次のSCFの
`Poisson` に進んでいました。コードを追うと、NC DCは `Eele1[1]`、NC DC-LNOは `Eele0[1]` / `Eele1[1]` を
返却せず、DFTが二成分を加算していました。バンド移行で変わったスタック配置から、未設定値を読む問題が現れました。
両バックエンドのSCF入口で未使用成分を0に確定し、出力契約を満たすようにしました。DOSの出力、数値カーネル、
SCFの計算式・閾値・反復上限は変更していません。MPIの停止判定を上書きして回避する処理も追加していません。

最終候補を固定して、次を確認しました。

- CTest **294/294** が通過しました。実コアと、両バックエンド・両バンドの実際のPOD、初期化、確保アダプター、
  精度制御、推移的な深部解放を使い、各100回の二重コンテキスト試験をASan/UBSanで実行しています。
  SCFの確保なしの再利用、番兵配列、FP64制御、サイクル終了時に捨てるwarm basisと保持するoverlap、入力境界の
  解放を照合しました。SDK・1 rank MPIと他モジュールの共通ワークスペースは模擬し、OpenACC/OpenMP転送は実行しません。
- DC/DC-LNOの実C入口・引数packet・callback・実C++呼び出し所有者を使う出力試験は、4入口×32条件（計128）を通過しました。
  出力領域の第2成分をNaN・無限大・巨大値等にしてもSCFの総和・停止判定が有限で一致し、大文字小文字のSCF指定と
  DOSの出力保持も検査します。演算カーネルは模擬です。旧入口へ戻す対照は4種類すべて失敗し、回帰を検出できています。
- CUDAの全218関数（collinear 102 / noncollinear 116）は、所有権受け渡しを逆変換すると演算・制御トークンが一致します。
  HIPはcollinear 106関数、noncollinear 149関数が一致し、残る差分は閾値取得と終了登録の撤去です。
  DFT/NEGFの計45関数も引数受け渡し以外が一致します。DC/DC-LNOの計230関数は、未使用成分の初期化だけが差分です。
- 主要9経路のCPU/GPU前後18比較は、エネルギー・全Forceが `1e-7` 以内で、個別36計算はすべて収束しました。
  最大差は `8.5601e-10 Ha` / `9.33e-10 Ha/bohr` です。修正後はNC DC・NC DC-LNOも停止しませんでした。
  GPU Cluster NCの50→51、Band NCの20→17は反復数不一致の厳密比較失敗として保存しています。
  同じ変更前バイナリ・コア・入力の対照は50→55、20→20でした。後者は反復数が同じですが、参照結果が以前の
  反復数不一致を保持しているため厳密検証器は失敗を返します。この由来も記録し、元の失敗を置き換えていません。
- 標準SのCPU/GPU各14例は、前後・CPU対GPU・同梱参照の5比較を通過しました。
  実GPUの常駐、直列化、k点が多い経路と同一プロセスでのFP32有効/無効入力切替は、8組・16計算すべて収束し、
  各組の反復数も一致しました。最大差は `5.7201e-10 Ha` / `1.5e-11 Ha/bohr` です。
  直列化時のホスト保存とk点overlapの再利用は実行ログも照合しています。
- HIPのoverlapキャッシュ73条件、精度改善142条件、MI300A等の方針・解放243条件、実SDKヘッダーでのworkspace query、
  7失敗/回復とoverflow、deviceなし/部分確保の解放、4ソルバーの入力制御と旧順序の対照を通過しました。
  CUDAのcuSOLVER方針は旧SDK相当65条件・新SDK相当各68条件を通過しています。
- HIPの主要8 Cソースを実SDKで構築しました。両バンドとwave-function probeはgfx942向けOpenMP targetで構築し、
  `.llvm.offloading` セクションを確認しました。AMD向け本体の最終リンク・実機の検証ではありません。

C専用監査は289翻訳単位・3,019関数、共通大域実体1,051個・12,912参照組です。設定298個・構造表278個、
外部大域名357、環境変数251名を維持しています。ファイルstaticは412→352、関数内staticは376→343へ減りました。
ソース上のもう1個の `firsttime` は変更前のLLVM IRに実体がなく、監査の削減数に加えていません。
C++へ演算を移して解析範囲を減らした数ではありません。所有権を検証してから一覧とベースラインを更新しました。

性能は [性能検証記録](PERFORMANCE_REGRESSION.md) に記載します。固定本体・コア、全Force、厳密比較と対照、
タイムアウトと修正後の結果は `build/band-state-refactor-20261023/` に保存します。初回のfixture不足、
対象フラグの欠落をELF検査で検出したオブジェクト、k点数がrank数を超えた試験を直列化経路と誤認した検査失敗も
区別して保持しています。数値を計算済みの例は入力・設定を照合して再利用し、結果を上書きしていません。
残る3ソルバーの可変別名、共通ヘッダーとC++/CUDA/HIPの大域状態は引き続き対象です。
他モジュールのHIP共有ワークスペース等も残るため、実OpenMX全体の同時実行・物理多GPUの保証ではありません。
AMD・A100/H100・Intel/Apple GPU実機、電極付きNEGF、全L/L2/L3、多GPU・物理多ノードは再検証していません。

## 第46段階: DC・DC-LNO・Krylovの永続状態とMPI資源を計算所有へ移行

`b7bc4a8f` を基準に、両バックエンドのDC・DC-LNO・Krylovから最後の3つの可変な資源別名を
除去しました。SDK不要の `OpenMXDCState` / `OpenMXDCLNOState` / `OpenMXKrylovState` をDFTから
入口・既存のC++呼び出しpacket・深部補助関数・Force前のキャッシュ解放へ渡します。
状態のPODとGPU/host領域は、既存のC++資源グループが計算ごとに所有します。
全7ソルバーの借用状態が明示され、隠れた現在コンテキストを持つマクロは撤去しました。

DCのoverlapキャッシュ・GPUハンドル・メモリ判定、DC-LNOの実数/複素数作業領域・MPI proxy群、
Krylovのスレッド別作業領域・KUキャッシュ・再計算履歴・初回制御を移しています。
CUDAの状態フィールドはDC 11、DC-LNO 15、Krylov 17、HIPは19、21、27です。
初期値の1・−1と `MPI_COMM_NULL`、AMDのGEMMul8無効化既定値も保持します。
既存のHistory領域も同じ明示された所有元から取得します。借用ポインタは残りますが、領域の所有元はC++へ統一されています。

HIP Krylovのpreflightが生成していたdevice communicatorは、従来解放されていませんでした。
計算所有者の終了処理で解放するようにし、DC-LNOのnode/device群とともにMPI終了前に破棄します。
これらの解放は該当rank群で集団的に実行する必要があり、状態を使用中のままclear/finishできません。
SCFの演算・閾値・反復上限、GPUの切替・容量・再試行方針は変えていません。
KrylovのOpenMP領域は、従来の共有staticの代わりに状態ポインタを共有します。

- CTestは299件が通過しました。実C++コアに実際のPOD・初期化・推移的解放を接続し、
  3ソルバー×両バックエンドで各100回の二重コンテキストをASan/UBSanで検証しました。
  再取得時の確保なしの再利用、番兵、別所有元への非干渉、部分確保と二重解放防止、
  C++確保失敗注入下の終了・デバイス復元を確認しています。SDKは模擬です。
- DC-LNO/Krylovは実MPI 4 rankでも両バックエンド各100サイクル検査しました。
  2 rankずつの部分コミュニケータを使い、別計算の資源を保持したまま片方だけを解放します。
  HIP Krylovの追加した解放を取り除く対照は、communicator残留を検出して失敗しました。
  MPIを模擬する検査でもNULLを非零にし、ゼロ初期化だけに依存できないことを確認します。
- 前段階のNC DC/DC-LNO出力契約は維持し、実入口・packet・callback・コアを使う4入口×32条件と、
  旧処理の4失敗対照が通りました。状態の受け渡しも照合しています。
- 6実装の361関数は、状態参照と共有指定を逆変換すると演算・制御トークンが一致しました。
  残る1関数の変更はHIP Krylovのcommunicator解放です。DFTの計40関数も引数の変更以外が一致します。
- 主要9経路のCPU/GPU前後18比較は全エネルギー・全Forceが `1e-7` 以内で、36計算が収束しました。
  最大差は `1.0260e-9 Ha` / `5.42e-10 Ha/bohr` です。GPU NC cluster 48→55とband 21→20の
  反復数不一致は厳密比較の失敗として保存しています。同じ変更前の本体・コア・入力の対照も48→53、21→18でした。
  対照で元の失敗を置き換えていません。標準SのCPU/GPU各14例は前後・CPU対GPU・同梱参照の5比較を通過しました。
- 同一プロセスでCPU 6入力・GPU 8入力を前後に実行した14比較・28計算も収束し、反復数が全組一致しました。
  閾値を試験用に下げ、DCのS-cacheとKrylovのKU-cacheの再生成、DC-LNOの実GPU taskをログで確認しています。
  最大差は `5.6440e-9 Ha` / `1e-12 Ha/bohr` でした。旧suiteの入力探索に必要なseed出力の診断は合否に使わず、
  新しく計算した前後を比較します。KrylovはOpenMP 2 threadでもCPU/GPUの4組・8計算を通過しました。
  4 MPIの通常経路と8 MPIの1原子/rank経路を使い、GPUのpool/cacheが実行されたことも照合しています。
- HIPのDC数値/失敗試験、DC-LNO collinear/noncollinearの84/1,140失敗と126/949数値照合、
  Krylovの38数値照合・24 GEMM失敗、MI300A guardの66,231数値・54 dispatch照合が通りました。
  所有境界のSDKは模擬し、数値の基準にはCPU LAPACKを使っています。
  変更した5 C単位を実ROCm SDKでホスト構築し、3ソルバーはgfx942 OpenMP targetでも構築しました。
  オブジェクト内のイメージのarchとproducerを確認しています。AMD向け本体の最終リンク・実機試験ではありません。

C専用監査の対象は289単位・3,019関数のまま、ファイルstaticが352→322、関数内staticが343→325に減りました。
状態フィールド43、資源別名3、診断用の局所化2に対応する48定義です。HIP側は67フィールド・別名3・局所化2を
扱っています。バックエンドの数は合算しません。設定298個・構造表278個、共通大域実体1,051・12,912参照組、
外部大域名357・環境変数251名は維持しています。LLVMのOpenMP補助関数4個の番号が変わり、旧ゲートは
27組を新規参照として報告しました。4組の名前を個別に対応付けると全参照組と、各関数の共通大域への
read/write/write-through/escape/addressの集合が一致します。この証拠と所有権検証後にベースラインを更新し、
更新後の増加防止ゲートも通過しました。C++へ解析範囲を移して数を減らした変更ではありません。

性能・同一プロセス試験の結果は [性能検証記録](PERFORMANCE_REGRESSION.md) に記載します。
固定本体・コアと検証元は `build/local-solvers-state-refactor-20261023/` に保存します。
初回の引数変換・fixture不足と監査の名前差分を残し、末尾空白の整理後にGPU bundleが変わったため、
最終バイナリで全主要経路とSを再実行しています。実OpenMX全体の再入可能性、GEMMul8/MAGMA等の
共有キャッシュ表、共通大域状態の除去は未完了です。AMD・A100/H100・Intel/Apple GPU実機、
電極付きNEGF、全L/L2/L3、多GPU・物理多ノードは再検証していません。

## 第47段階: ソルバー状態の寿命試験をGPU SDK不要のCIへ移行

`a66314df` の状態所有を継続的に検査できるよう、ルートのアプリケーション用登録から
共通の `cmake/OpenMXNativeStateTests.cmake` へ8件の状態・出力試験と2件の実MPI試験を移しました。
`OPENMX_BUILD_APPLICATION=OFF` と `cmake -S core` の両方で登録され、アプリ付き構成の
299件の名前・件数は維持します。`BUILD_TESTING=OFF` の構成にはPythonを要求しません。
共有コアのコンパイラとフラグをfixtureに渡し、GCCとClangのsanitizer runtimeの混在を防ぎます。
出力契約のfixtureはC++ドライバを使う場合も実入口をC11として構築します。
固定アドレス実行形式の指定はLinuxに限定し、Darwinでは通常のPIEを使います。

CIは全ホスト構成で8件の存在・重複・実fixture・ビルド内の実ライブラリを検査し、
Ubuntu Serialの1構成ではOpen MPIを導入して4 rankのDC-LNO/Krylov試験も必須にします。
CIのCPU数が4未満でも起動できるよう、そのジョブだけOpen MPIのoversubscribeを許可します。
Open MPI 4系と[5系の環境変数](https://docs.open-mpi.org/en/v5.0.x/mca.html#converting-mapping-parameters)を設定します。
Open MPI 5で2スロットに制限した4 rank試験も両ソルバー・両バックエンドが通りました。
登録の6種類の不備を作った対照は、それぞれ意図した理由で失敗しました。
既存のGPUエラー境界試験には、生成したMPIスタブをincludeする指定を追加しました。
環境のIntel MPIヘッダーで偶然構築できていた依存を除き、GPU SDK・MPIヘッダーのない環境でも通します。

ローカルのLinuxでは、GCC C++20 Serial 241件（実MPI2件を含む）、GCC C++20 OpenMP 239件、
GCC C++23 ASan/UBSan 239件、独立コアのClang C++23 ASan/UBSan 239件が通りました。
GPU検出・SDKコンパイラのコマンドを失敗するスタブに置き換え、include/libraryの環境変数も除いて検証しています。
各ライブラリの依存物にもCUDA/HIP runtimeはありません。GCC/Clangの直接実行16組と、
既存のGPU契約6スクリプトも通過しました。macOS実機での結果はこのローカル検証に含めません。
アプリ付き構成も、MPIスタブのinclude修正後に299件すべて通過しました。

アプリケーション・数値コアの実装、監査ベースライン、既存の本体・コアのバイナリは変更していません。
大域状態の削減数は第46段階のままです。この変更のために科学計算・性能測定は再実行していません。
試験・登録一覧・失敗対照・バイナリ照合は `build/native-state-host-ci-20261023/` に保存します。

プッシュ後のCIではLinuxの3構成とGPU契約が成功しました。macOSの最初の実行は、
既存の原子対Bessel試験がlibc++にない `std::sph_bessel` を使うため、構築段階で失敗しました。
この試験の引数範囲0〜6には独立した冪級数を使い、標準ライブラリへの依存を除いています。
本体の下降漸化式とは異なる参照計算を維持し、検査する角運動量・値・導関数を減らしていません。
次のmacOS CIは構築を通過し、8件中7件が通りました。残るHIP collinear clusterの試験では、
ASan付きfixtureの `operator new` 差し替えが共有コアの確保に届かず、注入した失敗を検出できませんでした。
Darwinの試験用確保フックにはdyldのinterposeを使います。最初の実行形式に含める方法では
同じ失敗が残ったため、フックを試験専用dylibとして接続します。確保失敗・解放の検査条件は維持します。
ASanとの接続順を明示するため、SDK模擬試験の子プロセスではそのdylibをpreloadします。
実行後の診断では、通常のmacOSコア試験231件は通過しましたが、ASanがある子プロセスでは
そのinterposeも優先されず、確保失敗の注入が残りました。macOSではCIに指定したsanitizer OFFを
fixtureにも適用し、実際の確保失敗・解放の検査を維持します。Linuxのfixtureは引き続きASan/UBSanを
使います。macOSでのASanと共有コアへの確保失敗注入の組合せは、この検証の保証に含めません。
試験専用の変更で、配布する本体・コアのメモリ管理には影響しません。
修正後の[macOS CI](https://github.com/dc1394/openmx4.0_gpu/actions/runs/39829412048/job/120001262458)は、
8件の状態試験と231件の独立コア試験をすべて通過しました。

## 第48段階: Hamiltonian混合のGPU/CPU履歴を計算所有へ移行

`75082a5b` の本体を基準に、CUDA/HIPで共有する `Mixing_H.c` の7個の可変staticを除去しました。
GPUの論理履歴・分割arena・ワーカー配置、ホストに保持する履歴と重み、容量予約、
増分CPUのブロックGramキャッシュを `OpenMXMixingHState` にまとめます。
DFTは明示したコンテキストから借用状態を取得し、入口を含む21個の状態を使う関数へ渡します。
容量callbackにはその状態を、worker callbackには状態を含むpacketを渡し、現在の計算を指す隠れた変数を持ちません。

PODと永続領域の所有元は既存のC++資源グループです。10か所のhost確保とGPU shardの確保を
所有用アダプターへ接続し、計算の終了・clearと部分確保の後にも解放します。
Force前にはその計算の履歴を明示して解放します。`Free_Arrays_with_context` は既存のグループclearで
混合状態も破棄するため、古い引数なしの解放と重複宣言を除いています。
解放時のworker実行は、確保時のデバイスを使い、呼び出し元のデバイス設定を復元します。
clear/finishには利用者とGPU処理が停止している必要があり、MPI終了前に実行します。

演算・履歴の論理的な回転・SCFの巻き戻し判定・fingerprint、GPU/host履歴/増分CPU/旧CPUの選択、
容量の予約・MPIの合意・再試行と、GeForce/データセンター/AMDのworker方針は維持しています。
35関数は状態・所有用アダプターの受け渡しを逆変換すると、演算と制御のトークンが一致しました。
各反復の数値作業配列と内側のループは、この単位では移行していません。

- 実際のPOD・初期化・preflight・worker・解放を実C++コアに接続し、両バックエンド各100回の
  二重コンテキストをASan/UBSanで検査しました。SDK・MPI・worker dispatchは模擬です。
  0/1/3のspin、空rank、GPU常駐・host履歴・増分CPU・旧CPU、C++登録とSDKの確保失敗、
  別所有元への非干渉、確保なしの再取得、二重解放防止・デバイス復元を確認しています。
  実際のGPU Gram/残差関数も模擬メモリ上で実行し、重み・履歴の置換を含めて独立参照と比較します。
  この9件目の寿命試験は、SDK不要のホストCIにも登録しています。
- RMM-DIISHを指定した収束例のCPU/GPU・4経路・復旧経路の前後12組、24計算は反復数も一致し、
  最大差は全エネルギー `1.2002e-11 Ha`、全Force `1.54e-10 Ha/bohr` でした。
  GPU常駐、host履歴、強制した増分CPUの実行をログで確認しています。
- GaAsの非共線clusterは、変更前から100反復でも未収束でした。GPUの3経路は変更後の比較が
  `1e-7` を超えたため失敗として保存しています。同じ変更前本体・コア・入力・設定をもう一度使った
  対照も3経路すべて許容差を超えました。これらを収束した例や成功した比較に数えず、元の失敗を対照で置き換えていません。
- DC・DC-LNO・KrylovにもRMM-DIISHを指定し、CPU/GPUの前後6組・12計算が収束しました。
  CPU3組は反復数・全エネルギー・全Forceが出力精度内で一致します。GPUは混合履歴とソルバーの
  GPU閾値を明示して実行し、全エネルギーの最大差 `4.81e-10 Ha`、全Forceは出力精度内で一致しました。
  GPUの反復数はDC 10→11、DC-LNO 11→10、Krylov 10→11で、厳密比較の失敗を保存しています。
  同じ変更前を再実行した対照はDC 10→10、DC-LNO 11→10、Krylov 10→10でした。
  この対照で元の失敗を置き換えていません。
- 同一プロセスで4経路を2巡したCPU/GPUの前後32計算もすべて収束し、16組の反復数が一致しました。
  全エネルギーの最大差は `1.6002e-11 Ha`、全Forceは `1.55e-10 Ha/bohr` です。
  各入力で実GPUの混合履歴を再生成したことを確認し、入力探索用のseed出力は数値比較に使いません。
- CTest300件が通りました。初回の無制限stackで既存の軌道試験がASanのshadow配置に失敗したログも残し、
  CIと同じ8 MiBのstackで全件を再実行しました。
  HIPの共有混合実装・DFT・Free_Arraysを実ROCm SDKでホスト構築し、混合実装はgfx942 OpenMP targetでも
  構築しています。埋め込んだイメージのarch/producerも検査しました。AMD本体の最終リンク・実機検証ではありません。

プッシュ後のHIP CIで、新しい `openmx_mixing_h_state.h` が隔離したソースへ投影されていないことを検出しました。
HIPの `integrated_shared` 一覧に追加しています。最初のローカル構築は `-Isource` が不足を隠していたため、
元ソースのinclude指定とinclude環境変数を外し、3 C単位とgfx942を暗黙宣言エラーも有効にして再構築しました。
CIのネイティブ構築対象にも `Mixing_H.c` を加え、実SDKを使う隔離した12単位と強いシンボルの検査を通しています。
この追加修正は計算コード・本体・共有コアのバイナリを変更せず、数値・性能結果は同じ固定バイナリの記録です。

C専用監査では、対象289単位を保ってファイルstaticが322→315、関数内staticは325のままです。
2個のbind/resetと6個の所有用アダプターにより関数数は3,019→3,027となります。
設定298個・構造表278個と、共通大域1,051・12,912参照組、外部大域名357・環境名251を維持しました。
OpenMP補助関数5個の番号が変わって旧ゲートが報告した30組は、関数名を個別に対応付けると
全参照組と各関数のread/write/write-through/escape/addressが一致します。
所有権とこの対応を検証してから一覧・ベースラインを更新し、更新後のゲートも通っています。

固定本体・コアと試験・失敗対照は `build/mixing-h-state-refactor-20261023/` に保存します。
性能は [性能検証記録](PERFORMANCE_REGRESSION.md) に記載します。実OpenMX全体の再入可能性、
共通大域・共有キャッシュの除去、AMD・A100/H100・Intel/Apple GPU実機、電極付きNEGF、
全L/L2/L3、多GPU・物理多ノードの再検証は未完了です。

## 第49段階: Hamiltonian混合の一時行列とスレッド作業領域をRAIIへ移行

`fd233779` を基準に、CUDA/HIP共通の `Mixing_H.c` に残る一時領域の
`malloc` / `calloc` 60か所を、呼び出し単位のC++所有へ移しました。
PulayのCPU・GPU・増分CPU、MultiSecant・One-shot、逆行列の作業行列を連続領域と型付きビューにし、
各OpenMPスレッドの累積配列も独立した所有元から借ります。永続履歴の所有元と寿命は第48段階のままです。
単純混合には作業領域を作らず、既存の計時区間内でPulayの所有元を生成・破棄します。

既存の `OverwriteTensor` を名前付きの実数作業領域にも使います。大きな行列を確保するたびに
不要な全要素ゼロ初期化を追加せず、以前の `calloc` に相当するGram部分和と物理slotの係数は
取得ごとに明示してゼロにします。履歴次元の縮小・再利用で古い係数が残らないことも検査します。
確保・拡張に失敗した場合は既存領域と出力を維持し、通常終了・callbackの早期return・例外では
C++所有元が解放します。Cの入口はC ABIを使い、NVIDIAのネイティブC/C++17経路へKokkosの型を持ち込みません。

既存37関数の演算・制御トークンは、所有・配列アクセスの変更を除くと一致します。
GPUの常駐・host履歴・増分CPU・旧CPUの選択、回転・巻き戻し・容量方針、MPI、
GeForceとデータセンターの実行方針、AMDの設定は維持します。
本体全体の大域参照や電子行列の借用多重ポインタを除去した段階ではありません。

- 実共有コアの確保をpoisonして、上書き用領域がゼロ初期化されないことと通常の実数領域が
  ゼロ初期化されることを対照で検査します。名前・workerの分離、確保なしの再利用、縮小、
  拡張失敗、範囲・overflow、途中終了時の解放も試験します。
- 両バックエンドの各100回の状態試験を拡張し、実際のGram/残差wrapperの再利用と
  0/1/3 spin・空rank・物理slot置換を含む増分Gramを独立参照と比較します。
  逆行列は解析的な1x1/2x2固有系をモデルとして、正則化した逆行列とLUの行列積恒等式を確認します。
  GPU/MPI/固有値ライブラリはこのfixtureでは模擬で、OpenMPのhostループは直列実行です。
- RMM-DIISHの収束した4主要経路・復旧経路12組と、DC/DC-LNO/KrylovのCPU/GPU6組を比較しました。
  CPUの局所ソルバー3組は出力精度内で一致します。GPU DCは10→11、DC-LNOは11→10となり、
  反復数の厳密比較は失敗として保持します。別の数値照合では18組すべて許容差内で、
  最大差は `4.621e-10 Ha` / `1.341e-10 Ha/bohr` です。
  同じ旧バイナリ・コアによる対照はDC 10→10、DC-LNO 11→11でした。
  対照が元の反復数差を再現しなかったことも残し、厳密比較の成功とは扱いません。
- 同一プロセスで4主要経路を2巡するCPU/GPU前後32計算と、2 OpenMPスレッドの
  コリニア・非共線clusterのCPU/増分GPU前後8計算が収束し、反復数が一致しました。
  後者の最大Force差は `2.358e-9 Ha/bohr` です。
- GaAs非共線clusterの未収束100 SCF比較はGPU3経路で再び失敗しました。
  同じ旧バイナリ・コアを再実行した3経路の対照も許容差を超えます。
  元の失敗を残し、収束した比較へ含めません。
- Linuxの独立ホスト構成はGCC C++20 SERIAL（243件）/OpenMP（241件）、
  GCC C++23とClang C++23のASan/UBSan（各241件）が通過しました。
  アプリケーション構成の301試験では、初回にASanとNVIDIA版 `libgomp.so.1` の
  loader競合が4件で発生しました。旧コアでも再現し、元の失敗とloaderの差を保存しています。
  試験用loader/include環境を分離すると同じ4件がASanを有効のまま通過しました。
  最終ソースでも環境を分離して全301件を再実行し、GCC/Clangのsanitizer付き実コアでも
  拡張した混合fixtureを再確認しました。
- 実SDKでCUDA sm_80・HIP gfx942の隔離した12単位ずつを構築し、
  共有混合実装のgfx942 OpenMP targetイメージも検査しました。
  AMDの本体リンク・実機測定ではありません。

監査範囲は289 C単位のままです。既存関数の共通大域へのアクセスは一致し、
OpenMP補助関数5個の番号だけを個別に対応付けています。新しい共通エラー処理が
`MPI_Abort` 用にcommunicatorを読む1参照が加わり、12,912→12,913組です。
ファイルstatic315・関数内static325、設定298個・構造表278個と外部大域・環境名は不変です。
所有権と参照差を検証した後にベースラインを更新しました。

固定バイナリ・コア、元の失敗と対照は `build/mixing-h-workspace-refactor-20261023/` に保存します。
末尾空白の整理でもNVIDIA本体のハッシュが変わったため、最初と最終の候補を別々に保持します。
最終候補の収束した12組も反復数が一致し、最大差は `2.801e-11 Ha` / `1.651e-10 Ha/bohr` でした。
インストール後の `work/openmx` と共有コアも、検証した最終候補と同じハッシュです。
性能と最終バイナリの比較は [性能検証記録](PERFORMANCE_REGRESSION.md) に記録します。

## 第50段階: k空間の密度混合状態を計算所有へ移行

`f7da6ce6` を基準に、共有 `DIIS_Mixing_Rhok.c` のGramキャッシュ1個と、
DIIS/Kerkerの通常・NEGF経路にある初回表示フラグ4個を `OpenMXMixingRhokState` へ移しました。
DFTの両バックエンドから状態を明示して渡し、CPUの旧経路とGPU用の高速経路も同じ所有境界を使います。
元のGram領域はC++資源グループに登録し、部分確保・clear・finishで解放します。
型の内部表現は二つの密度混合実装だけで共有し、他の呼び出し元には不透明な型を渡します。
現在の計算を指す大域ポインタや既定コンテキストは追加していません。

高速経路の継続・巻き戻し・履歴数とspinの判定、DGEMV/DGEMM、MPI合意、
slotの回転、Kerker重みとNEGFの演算は維持します。既存55関数の計算・制御は、
状態引数と所有用アダプターを逆変換すると一致し、共通ヘッダーのバイト変更は
状態型のincludeと3個のプロトタイプだけです。通常とNEGFの初回表示は計算ごとに独立します。

- 実共有コアを使う100回の二重コンテキスト試験で、四つの初回表示、キャッシュの分離、
  確保なしの再取得、SCF継続・巻き戻し・gap、履歴数とspinの変更、
  高速経路の無効化、確保の登録失敗、途中のclear/finishを検査しました。
  productionのpreflightと表示ブロックをそのままコンパイルし、MPI abortだけを模擬します。
  前段階から変更していないGCC C++20 SERIAL/OpenMP、GCC/Clang C++23 ASan/UBSanの
  実共有コアにも接続して通過しました。
- この10件目のホスト寿命試験をアプリケーションを構築しないCIにも登録しました。
  登録漏れ・重複・別fixture・別コア・label漏れの5つの対照をすべて拒否します。
- RMM-DIISKの4主要経路についてCPU/GPU、高速設定と `OPENMX_MIX_FAST=0` の12組を比較し、
  DC/DC-LNO/KrylovのCPU/GPU6組も照合しました。全36計算が13 SCFで収束し、
  最大差は `4.001e-12 Ha` / `2.511e-10 Ha/bohr` です。
- 同一プロセスで4主要経路を2巡するCPU/GPU前後32計算もすべて収束し、16組の反復数が一致しました。
  最大差は `1.200e-11 Ha` / `3.220e-10 Ha/bohr` です。
- 302試験の初回は301件が通過しました。残る1件は、移行した2ファイルを不変なAMD取り込み用の
  hash一覧に残したmanifest不整合でした。共通統合の一覧へ移し、不変取り込み用の2項目を除いて、
  元の失敗を保持したまま隔離構築の12検査を再確認しました。検査条件は変更していません。
- 実SDKでCUDA sm_80・HIP gfx942の隔離した14単位ずつを構築し、HIPのDFTとMixing_DMも
  元ソースのinclude探索やinclude環境変数に頼らず構築しました。AMDの最終リンク・実機検証ではありません。

解析範囲は289 C単位で、追加した所有用の3関数により3,036→3,039関数です。
ファイルstaticは315→314、関数内staticは325→321です。共通大域1,051・参照12,913組と
各関数のread/write/write-through/escape/addressが一致し、設定298個・構造表278個を含む全分類も不変です。
所有権とこの差分を検証してから一覧・ベースラインを更新しました。

固定した本体・コア、数値・寿命試験、元のmanifest失敗は
`build/mixing-rhok-state-refactor-20261023/` に保持します。
18 MPI・MPSのsidia333は内部計算時間103.365→103.260秒で、条件を変えず約105秒の目標内を維持しました。
CPUのC60とIntel版を含む記録は [性能検証](PERFORMANCE_REGRESSION.md) に記載します。
SCFごとの一時配列と電子密度履歴の借用多重ポインタは残っています。
実OpenMX全体の再入可能性、電極付きNEGFの実計算、AMD・A100/H100・Intel/Apple GPU実機、
全L/L2/L3、多GPU・物理多ノードは本段階の保証に含めません。

## 第51段階: k空間密度混合の一時領域を型付きRAIIへ移行

`086dece5` を基準に、共有 `DIIS_Mixing_Rhok.c` / `Kerker_Mixing_Rhok.c` の
一時確保47か所をC++の呼び出し所有へ移しました。通常・NEGFの各入口が明示的な
invocationを深部関数へ渡し、途中return・例外・部分確保の失敗でも所有領域を解放します。
永続Gram履歴は第50段階の計算状態が引き続き所有し、一時配列と寿命を分けています。

実数のGram行列・最適残差と複素数のLU退避行列は、連続した型付きビューを借用します。
所有する行ポインタ表は作りません。実数には既存の上書き用テンソルを再利用し、
複素数にも同じ所有者を使う独立した作業領域を追加しました。不要な全領域の初期化はせず、
元の明示的なゼロ化は維持します。配列要素数の積では元の `sizeof` と同様に `size_t` へ広げます。

Rhokの実際の逆行列は、元と同じ `method_flag=1` のピボット付きLAPACK LUです。
Hamiltonian混合の固有値分解経路へ置き換えていません。leading dimension、work/pivotの寸法、
BLAS/MPI、残差の更新順序、slot回転とCPU/GPUの選択も保ちます。
所有とビューの変更を逆変換すると、既存17関数の計算・制御本体が一致します。

- 実共有コアの複素数領域をpoisonして、上書き用とゼロ初期化用の分離、workerの独立性、
  確保なしの再利用、縮小、失敗時の保持、overflow・空領域を検査しました。
  呼び出しの解放・確保失敗の全位置・並行workerの試験にも追加しています。
- productionの通常/NEGF dispatch、実数・複素数の逆行列を実共有コアにつなぎました。
  引数全体の受け渡し、途中終了・例外・全確保位置の失敗と解放を確認します。
  LUの二段階の呼び出し契約、複数寸法・strideの逆行列を独立した行列積恒等式と照合し、
  未使用の複素LUと固有系の処理・入力復元・境界canaryも検査しました。
  MPI abort・LAPACK/固有系はこのfixtureではモデルです。
- この11件目の `native-state` 試験をSDK不要のCIにも登録し、登録漏れ・重複・label漏れ・
  別fixture・別コアの5つの対照をすべて拒否しました。
- アプリケーション構成の304試験が通過しました。独立したGCC C++20 SERIALは246件、
  OpenMPは244件、GCC C++23とClang C++23のASan/UBSanは各244件が通過しました。
  試験のloader/include環境はSDKから分離し、SERIAL構成では実MPIの2件も実行しています。
- 実SDKの隔離したCUDA sm_80・HIP gfx942は各14単位の構築に通過しました。
  元の `source/` のinclude探索やinclude環境変数へ依存しません。
- RMM-DIISKの4主要経路のCPU/GPU・高速設定/旧設定12組と、DC/DC-LNO/KrylovのCPU/GPU6組は、
  全36計算が13 SCFで収束しました。反復数が一致し、最大差は `6.999e-12 Ha` /
  `1.850e-10 Ha/bohr` です。同一プロセスで4経路を2巡する前後32計算もすべて収束し、
  16組の反復数が一致し、最大差は `4.996e-12 Ha` / `5.300e-10 Ha/bohr` です。
  OpenMP 2 threadのコリニア・非共線clusterもCPU/GPU前後8計算が13 SCFで収束し、
  4組の反復数が一致しました。最大Force差は `8.394e-9 Ha/bohr` です。

監査範囲は289 C単位、static314/321と全分類を維持します。
OpenMP補助関数1個の番号の対応を除き、既存関数の共通大域への全アクセスが一致します。
確保エラー用のcommunicator読み取りが2組加わり、参照組は12,913→12,915です。
所有・失敗・再利用と実計算を検証してから一覧・ベースラインを更新しました。
証拠と固定本体・コアは `build/mixing-rhok-workspace-refactor-20261023/`、
性能の記録は [性能検証](PERFORMANCE_REGRESSION.md) に保存します。

借用した密度履歴の多重ポインタ、全体の再入可能性、電極付きNEGFの実計算、
AMD・A100/H100・Intel/Apple GPU実機、全L/L2/L3、多GPU・物理多ノードの検証は残ります。

## 次の移行単位と完了条件

1. 化学種の基本情報23表と PAO/VPS・Fourier の数値領域の所有、実空間・Fourier 表の
   大域参照の除去、軌道・導関数の明示化は移行済み。名称・基底数・電荷等の借用大域
   ポインタを除去し、Hubbard・spin-orbit 係数等の残る化学種情報、原子種・分配も所有化する。
   軌道の出力・解析側の残る多重ポインタを除去する。
   MPI 配布、restart と入力解析の制御も独立したモジュールへ移す。
2. 周期セル、原子・近接リスト、MPI による原子分配を独立したオブジェクトにする。
   一時的に旧配列を参照する場合も寿命を明示し、隠れた既定コンテキストを置かない。
3. VNA、主要 SCF 行列32系統、onsite・DC・LNO/収縮係数9系統は連続領域の所有へ移行済み。
   消費側の借用行ポインタを減らし、ブロックごとに `mdspan` または平坦なビューを渡す。
   軌道最適化の履歴・微分も所有化済み。原子種別の Hubbard 相互作用係数等を移す。
   密行列には `layout_left` / `layout_right` を明示し、BLAS の leading dimension を保持する。
4. Hamiltonian・密度格子、二中心積分の基底変換、Forceのスピン・重なり縮約は共通CPU実装と
   独立テストへ移行済み。HNL/core-holeのhost実装も共有化し、両者の係数と
   スピノル演算、HNLの矩形ブロック処理をC++/mdspanへ移行済み。
   動径/Gaunt積分、混合等にも同じ境界を設け、Kokkos実装へ移す。
   既存CUDA/HIP実装は各段階の同等性確認後に削除する。
5. SCF / MD / NEB の状態と MPI communicator・BLAS/solver ハンドル・GPU バッファを
   明示的な所有オブジェクトへ集約する。ランタイムの寿命と計算ケースの寿命を分離する。
6. 入出力、restart、輸送、解析機能まで引数で依存を受け渡し、`openmx_common.h` の可変
   大域宣言と各翻訳単位の可変 `static` を除去する。旧 C ABI が不要になった部分を閉じる。

完了判定には、アプリケーションの可変大域状態ゼロ、同一プロセス内の独立した計算状態、
例外・途中終了を含む解放保証、CPU/CUDA/HIP の数値同等性、全標準スイートと restart / NEB /
MPI 分割の検証が必要です。第三者ライブラリが内部に持つ状態はアプリケーションの
大域変数とは区別し、初期化・終了の境界を管理します。
