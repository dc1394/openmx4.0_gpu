# OpenMX Basis

OpenMX の基底処理を独立して利用する C++20 ライブラリです。
現在の公開範囲は PAO、非局所射影子、中性原子ポテンシャル、VNA 射影子の
動径補間、PAO/射影子の波数空間補間、PAO/VPSファイルの読み込み、
原子密度の補間・動径導関数・球対称Hartree積分、Gauss–LegendreとLebedev積分則、
球Bessel関数と2中心積分用Besselテーブル、球面調和関数と角度導関数、複素から実軌道への変換、
Gaunt角度積分と不変の係数表です。
本体もこの実装を使用します。
Fourier-Bessel 変換、完全な2中心積分は
まだこのパッケージには含まれません。

CMake 3.25.2 以上と C++20 のコンパイラ・標準ライブラリでビルドできます。
このディレクトリだけを取り出して利用でき、MPI、BLAS、Kokkos、GPU SDK、
OpenMX の実行ファイル・大域変数ヘッダー、ネットワーク接続を必要としません。
コードのライセンスは元の OpenMX と同じです。

## ビルドと外部利用

リポジトリのルートで実行します。

```sh
cmake -S modules/basis -B build/basis -DCMAKE_BUILD_TYPE=Release
cmake --build build/basis --parallel
ctest --test-dir build/basis --output-on-failure
cmake --install build/basis --prefix "$PWD/build/basis-install"

cmake -S modules/basis/examples/radial -B build/basis-example \
  -DCMAKE_PREFIX_PATH="$PWD/build/basis-install"
cmake --build build/basis-example
./build/basis-example/radial_example

cmake -S modules/basis/examples/pao -B build/pao-example \
  -DCMAKE_PREFIX_PATH="$PWD/build/basis-install"
cmake --build build/pao-example
./build/pao-example/pao_example DFT_DATA19/PAO/C6.0.pao

cmake -S modules/basis/examples/vps -B build/vps-example \
  -DCMAKE_PREFIX_PATH="$PWD/build/basis-install"
cmake --build build/vps-example
./build/vps-example/vps_example DFT_DATA19/VPS/C_PBE19.vps

cmake -S modules/basis/examples/density -B build/density-example \
  -DCMAKE_PREFIX_PATH="$PWD/build/basis-install"
cmake --build build/density-example
./build/density-example/density_example DFT_DATA19/PAO/C6.0.pao
```

外部プロジェクトの `CMakeLists.txt` は以下だけでリンクできます。

```cmake
find_package(OpenMXBasis 0.1 CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE OpenMX::basis)
```

動径補間の公開ヘッダーは `<openmx/basis/radial_table.hpp>`（C++）と
`<openmx/basis/radial.h>`（C ABI）です。C の呼び出し元でもリンク時には
C++ ランタイムが必要なので、CMake の `project()` で `C CXX` を有効にします。
静的ライブラリは PIC で生成します。CMake のビルド中の利用には
`add_subdirectory(path/to/basis)` と同じ `OpenMX::basis` ターゲットが使えます。
0.1 系は初期 API であり、C++ バイナリ ABI の固定は約束しません。

## PAO ファイルから使う

`<openmx/basis/pao_file.hpp>` は、ファイルまたは `std::istream` から
メタデータ・格子・原子密度・軌道・任意の生成時固有値を読み込む入口です。

```cpp
#include <openmx/basis/pao_file.hpp>

const auto pao = openmx::basis::PaoFile::from_file("C6.0.pao");
const double value = pao.sample(1, 0, 0.85); // l=1, mu=0, r=0.85 bohr
const auto radial_values = pao.orbital(1, 0); // 借用する span<const double>
```

`PaoFile` は連続領域に全データを所有し、入力ストリーム・ファイルの寿命に依存しません。
コピーは独立、ムーブは所有権を移します。取得した span は所有者の破棄・代入まで有効です。
並行した読み込みはそれぞれの入力を使い、構築後の同じオブジェクトは複数スレッドから読めます。
補間は既存の PAO 規約を使い、追加確保を行いません。

- 軌道は `l`、多重度 `mu`、動径点の順で連続配置します。`mu` は0始まりです。
  密度は物理点だけを返し、本体で使う前後の余白は含めません。
- 半径は bohr、生成時固有値は Hartree。`log_radii()` はファイルの対数格子をそのまま保持します。
  `cutoff` はメタデータであり、PAO の補間範囲を格子上端より内側へ切り詰めません。
- 原子番号は従来通り `AtomSpecies` の整数部分です。`1e-15 < AtomSpecies < 1` は1とし、
  仮想原子の小数値も `metadata().atom_species` に保存します。
- `PaoReadMode::metadata` は数値表を要求・確保せず、メタデータと任意の生成時固有値だけを返します。
  このモードやムーブ元のオブジェクトから軌道を評価することはできません。
- 大文字小文字、CRLF、空白、`#` コメントを扱います。生成条件など未知の節は無視します。
  スカラーの重複は最初の値を使います。`radial.cutoff.pao` の既定値は5 bohrです。
  生成時固有値の `Lmax=` 節がない仮想原子では `orbital_energy()` は `nullopt` を返します。
- 通常の読み込みでは必須節・終了タグ、数値・寸法、正の半径、半径・対数格子それぞれの狭義単調増加、
  全軌道節の格子一致を検証します。異なる印字精度には `1e-10*(1+abs(reference))` の許容差を使い、
  最後の軌道節の格子を採用する従来規約を維持します。
  欠落・余分な表要素、非有限値、重複する固有値チャネル等は例外になります。
  旧本体が不完全な入力を継続したり正常終了コードで中断した場合も、明示的に失敗を返します。

この入口は C++ API です。補間用の公開 C ABI にファイル API はまだ追加していません。
本体は `core/` 内の C アダプターを通じて同じ読み込み実装を利用し、検証を終えてから
計算所有の配列へ値を渡します。ライブラリ自身は MPI の停止やプロセス終了を行いません。

## VPS ファイルから使う

`<openmx/basis/vps_file.hpp>` の `VpsFile::from_file()` / `read()` は、
原子種・電子数、動径格子、局所ポテンシャル、非局所射影子・エネルギー、任意のPCC密度を所有します。
PAO と同じく、ファイルや入力ストリームの寿命に依存せず、コピー・ムーブ・並行読み出しが可能です。

```cpp
#include <openmx/basis/vps_file.hpp>

const auto vps = openmx::basis::VpsFile::from_file("C_PBE19.vps");
const double value = vps.sample_projector(0, 0, 0.85); // projector, spin channel, bohr
const double energy = vps.projector_energy(0, 0);       // Hartree
```

- `angular()` の順で射影子を保持します。1チャネルのスカラーデータと、2チャネルの
  j依存データを扱い、j依存の並びはファイル通り `j=l+1/2`、`j=l-1/2` です。
  `projector_values()` は `[projector][spin channel][radial point]`、
  `projector_energies()` は `[projector][spin channel]` の連続領域です。
- 局所ポテンシャルと射影子エネルギーは Hartree、半径は bohr。
  射影子はファイルの値をそのまま保持し、補間には `RadialKind::projector` を使います。
  スカラー相対論のj平均、SO係数、指定電子数による補正を読取時に暗黙適用しません。
- 新しい `<project.energies` 形式と、`number.vps` / `<pseudo.NandL` /
  `<projection.energies` を使う旧形式に対応します。旧形式はスカラー用です。
  `local.part.vps` と一致する0始まりの記述行を除き、他の行をその順で射影子にします。
  範囲外のlocalラベルは行を除外しない従来規約を維持します。
- `charge.pcc.calc` が無効なら `partial_core_density()` は空です。有効なら物理格子点だけを返します。
  本体で使う余白・端点の延長は本体のアダプターの責務です。
  格子は6点以上、正で狭義単調増加の半径と狭義単調増加の対数座標を検証し、
  PCCの格子もポテンシャルと一致することを確認します。許容差はPAO読取と同じです。
- `VpsReadMode::metadata` は動径数値表を要求・確保しません。射影子数と角運動量は保持します。
  新形式では射影子エネルギーも保持し、旧形式では全表読取時に取得します。
  射影子が0個のファイルも扱えます。
- `AtomSpecies` の小数値を保持し、整数原子番号の規則はPAOと同じです。
  `total.electron` の既定値は `AtomSpecies`、`valence.electron` は0です。
  `core_charge()` は `valence.electron + AtomSpecies - total.electron` を返します。
  論理値は従来の `on/yes/true/.true./ok`、`off/no/false/.false./ng` を扱います。

PAO/VPSは同じ内部テキスト解析器を使います。VPSの公開ファイルAPIも現在はC++用です。
本体のC境界は、C++所有者の生存中に同期コールバックへ借用データを渡します。
局所ポテンシャルの補正用領域はC++が所有し、利用側の早期終了・失敗・例外でも解放します。
コールバックを抜けたあとにデータを保持してはいけません。解析・確保の失敗では呼び出さず、
コールバック内で本体の配列を変更した後の失敗をロールバックする契約にはしていません。

## 原子密度とHartree積分

`<openmx/basis/density_table.hpp>` は密度・格子を所有する `LogDensityTable` と、
既存の配列を借用する `LogDensityView` を提供します。

```cpp
#include <openmx/basis/pao_file.hpp>
#include <openmx/basis/density_table.hpp>

const auto pao = openmx::basis::PaoFile::from_file("C6.0.pao");
const openmx::basis::LogDensityTable density(pao.log_radii(), pao.radii(), pao.density());
const double rho = density.sample(0.85);
const double drho = density.derivative(0.85);
const openmx::basis::GaussLegendreRule quadrature(1500);
const double hartree = openmx::basis::atomic_hartree_potential(density.view(), 0.85, quadrature.view());
```

- 半径はbohr、密度は電子数/bohr³、動径導関数は電子数/bohr⁴、ポテンシャルはHartreeです。
  密度の正規化・電子数補正・PCC加算は暗黙には行いません。
- 対数格子は6点以上で等間隔、半径は正で狭義単調増加とします。
  `log(r[i])` と格子、等間隔性を `1e-10*(1+abs(reference))` の許容差で構築時に検証します。
  密度は有限値を要求します。所有表には物理点だけを渡し、線形延長した前後の余白を自動で追加します。
  借用ビューにはN+2要素の密度を渡し、物理点は+1から配置します。
- `sample()` と `derivative()` は原点を含む有限・非負の半径を取ります。
  `sample_log()` は自然対数を取り、負の無限大は原点、正の無限大は上端延長です。
  原点側は従来の `KumoF` / `Dr_KumoF` の動径ステンシルを維持します。
  上端外の値は一定ですが、導関数は従来通り端点の対数座標での傾き/半径です。
  **上端外では、導関数APIは一定延長したsampleの数学的な微分とは一致しません。**
  この規約を物理的に変更する場合は、リファクタリングとは別の検証が必要です。
- Hartree積分の半径は有限かつ正とします。節点は[-1,1]、重みは正・有限で、
  同数の非空配列を借用します。積分次数と節点の順序は利用者が指定します。
  対数格子下端から評価半径までの内側積分と、評価半径から上端までの外側積分を使い、
  格子外でも元実装の境界・延長規則を適用します。無限領域積分や原点からの積分ではありません。
- 所有表のコピーは独立し、借用配列はビューの寿命中に変更してはいけません。
  検証は構築時に行い、補間・導関数はO(1)、Hartreeは節点数に比例し、評価時の確保はゼロです。
  同じ不変の表・積分則を複数スレッドで使用できます。

`<openmx/basis/density.h>` は本体のCコードも利用するスカラー評価ABIです。
内側のループで格子全体を再検証しないため、配列の寸法・格子の有効性は呼び出し元の責任です。
例外・大域エラー状態を使わず、不正な点数・null配列・NaN座標などはNaNを返します。
本体のHartree生成はC++所有の一時配列で全物理点を計算し、成功後だけ出力へコピーします。
空原子はゼロを返し、失敗時は出力を保持します。MPIとKokkosの初期化は要求しません。

## 積分則を生成する

`<openmx/basis/quadrature.hpp>` の `GaussLegendreRule(count)` は[-1,1]の積分点・重みを
連続配列に所有し、`view()` で確保なしの借用ビューを返します。点数は `1 <= count < INT_MAX`、
生成コストはO(count²)、ビュー取得はO(1)です。次数は利用者が選び、Hartreeの例の1500点は
本体の細密積分と同じです。外部から作った積分則には引き続き `GaussLegendreView(nodes, weights)` が使えます。

節点は昇順です。元のOpenMXと同じNewton反復・停止判定1e-13・最終反復時の導関数による重みを
維持し、奇数点数の中央節点に残るごく小さい丸め値も勝手にゼロへ置き換えません。
非有限な反復・100反復での未収束は例外になります。コピーは独立、ムーブは所有権を移し、
同じ不変の積分則を複数スレッドで使用できます。借用配列の寿命は所有者の破棄・代入までです。

`<openmx/basis/quadrature.h>` の `openmx_basis_gauss_legendre()` はC用の入口です。
呼出側が用意した、重ならない各count要素の出力に全結果を渡します。内部の確保・生成が失敗した場合は
0を返し、両方の出力を保持します。余分なパディングは要求せず、MPIやKokkosの初期化も不要です。
本体の古い `Gauss_Legendre` C関数は、この入口を使って従来の余白・制御引数の規約を保つアダプターです。

本体では細密積分の所有者をC++の同期コールバック境界内に置き、原子データの準備後に破棄します。
コールバック中にだけ節点・重みを借用し、入れ子の呼び出しや例外でも所有領域を共有・保持しません。
従来の `FineGL_Abscissae` / `FineGL_Weight` 大域配列は不要になりました。
粗・中密度の積分則も計算ごとに同じC++クラスで所有し、Energy・Stress・Fourier変換・
VNA組み立てへ不変ビューを渡します。本体の3種類の積分表で大域配列を廃止しました。

## 球Bessel関数と2中心積分用テーブル

`SphericalBesselSample(x, maximum_order)` は、球Bessel関数 `j_l(x)` と
無次元引数 `x` による導関数を固定長配列で所有します。ヒープ確保は不要です。
`values()` は `l=0..maximum_order+1`、`derivatives()` は `l=0..maximum_order` の
読み取り専用spanを返します。所有者をコピーすると結果も独立します。

```cpp
#include <openmx/basis/bessel.hpp>
#include <openmx/basis/pair_bessel.hpp>
#include <array>

const openmx::basis::SphericalBesselSample scalar(1.0, 4);
const double j0 = scalar.values()[0];
const std::array wavevectors{0.0, 0.5, 1.0, 2.0}; // bohr^-1
openmx::basis::PairBesselWorkspace workspace(4, wavevectors, 32);
const auto table = workspace.sample(3.0, 4);       // distance in bohr
const auto index = 2 * table.points() + 2;         // l=2, k=1 bohr^-1
const double j2 = table.values()[index];
const double radial_derivative = wavevectors[2] * table.derivatives()[index];
```

`PairBesselWorkspace` は波数格子をコピーし、連続した作業配列と上限付きキャッシュを所有します。
表の並びは `[角運動量][波数]` です。値と導関数はそれぞれ有効な行だけをspanとして返し、
mdspan接続用の `storage()` / `row_capacity()` は余裕を持たせた領域の形状も表します。
`sample()` の導関数も引数 `x=k*r` に関する値です。距離による導関数には波数を掛けます。

- 最大角運動量は0〜29、波数格子は非空で有限・非負、距離も有限・非負です。
  `maximum_order + int(1.5*x) + 10` がintに収まる範囲を検証します。
  無効な入力では例外を返し、既存のテーブルとキャッシュ統計を保持します。
- 元のOpenMXの下降漸化式、正規化、演算順序と `j_0'` の減算規則を維持します。
  原点では解析的な極限値を返します。極端な浮動小数点入力でのunderflow/overflowや
  原点付近の減算による精度の制約も継承します。1点の評価コストはO(maximum_order+x)です。
- キーは距離と角運動量の完全一致です。キャッシュ済みの借用表は所有者の破棄まで有効です。
  キャッシュが無効・満杯なら事前確保済みの作業表を再利用し、その借用は次の非キャッシュ評価まで有効です。
  同じ計算内で一時表を保存しておく場合は、次の評価前に必要な結果をコピーしてください。
- キャッシュの確保が失敗すると成長を止め、作業表で評価を続けます。
  既存のキャッシュは維持し、ヒット時・作業表の再利用時に確保を行いません。
  同時に評価する各スレッドが自分のworkspaceを所有します。

本体の重なり・非局所射影子処理もこの所有者を使い、従来のmdspanとC ABIへ借用表を渡します。
Fourier変換用のCUDA/HIPカーネルは同じ漸化式にKokkosの数学関数を接続し、大引数での値専用高速経路を維持します。
公開入口はC++ APIで、MPIやKokkosの初期化は不要です。単独の利用例は `examples/bessel/` にあります。

## Lebedev球面積分則

`<openmx/basis/lebedev.hpp>` の `LebedevRule` は、590・1202・2030点の既存OpenMX格子を所有します。
座標は単位球面上の無次元値で、重みの総和は1です。球面積分には4πを掛けます。
点の順番と印字された係数を維持し、正規化し直したり点を並べ替えたりしません。

```cpp
#include <openmx/basis/lebedev.hpp>

openmx::basis::LebedevRule rule(2030);
const auto point = rule.point(7); // span<const double,4>: x, y, z, weight
const auto flat = rule.xyzw();   // [point][4] の連続配置
rule.reset(590);                 // 容量内では確保せず再利用
```

- 借用したspanは所有者の破棄・代入・成功した `reset()` まで有効です。
  容量内でのリセットはアドレスを保ち、同じ点数の繰り返しはO(1)です。
  異なる点数ではO(点数)で表を入れ替えます。容量を超える場合だけ新しく確保します。
- 無効な点数や確保失敗は例外となり、前の格子・アドレス・借用ビューを維持します。
  コピーは独立し、ムーブ元は空になります。ムーブ元も `reset()` で再利用できます。
  点の範囲外アクセスは例外になります。
- 同じ格子を複数のスレッドから読むことができます。書き換え・リセットと同時に読むことはできません。
  共有する可変大域状態、MPI・GPU・Kokkosの初期化はありません。
- Cでは `<openmx/basis/lebedev.h>` の `openmx_basis_lebedev_fill()` が、呼び出し元の連続配列へ
  確保なしで表を書きます。容量はdouble要素数で指定します。不正入力では数値配列を変更しません。
  `OpenMXLebedevRuleData` は読み取り専用の借用ビューで、`openmx_lebedev_point()` は
  検証済みの添字を使う軽い行アクセスです。診断バッファと数値配列は重ねないでください。

本体のEnergy／Stressは、呼び出しの寿命に結び付けた同じ `LebedevRule` を使います。
Cの生成関数と互換用の行ポインタ配列を廃止し、GPU用の既存転送配列にもこの連続ビューから値を渡します。
CPUへ戻る場合も同じ係数を使います。単独の利用例は `examples/lebedev/` です。

## 球面調和関数と角度微分

`<openmx/basis/harmonics.hpp>` は、値と導関数を所有するC++の戻り値型を提供します。
計算に確保・アプリケーション状態は不要で、別々のスレッドから同時に利用できます。

```cpp
#include <openmx/basis/harmonics.hpp>
#include <cmath>

const double theta = 0.7, phi = -0.3;
const auto integral = openmx::basis::integral_harmonic(4, 1, theta, phi);
const auto orbital = openmx::basis::orbital_harmonic(4, 1,
    std::sin(theta), std::cos(theta), phi);
```

- `integral_harmonic()` の `IntegralHarmonicSample` は複素Y、`polar_derivative` = dY/dθ、
  `azimuthal_derivative` = dY/dφを返します。本体の `ComplexSH()` と同じ積分用規約です。
  θは0〜πで、`sqrt(1-cos²θ) <= 1e-12` のときθ微分をゼロにする従来処理も保ちます。
- `orbital_harmonic()` の `OrbitalHarmonicSample` は、最後の成分が
  `azimuthal_over_sine` = (dY/dφ)/sinθです。g軌道の修正済み実装に基づき、両極での有限な方向微分を保ちます。
  半径r>0での角度部分の空間勾配は `(polar_derivative*e_theta + azimuthal_over_sine*e_phi)/r` です。
  位相φは極での接線方向を選びます。2種類の出力は混同を防ぐため別の型にしています。
- 極付近では `acos(z/r)` からsinθを復元せず、`hypot(x,y)/r` を渡します。
  sinθは0〜1、cosθは−1〜1で、二乗和が1からdoubleの16 epsilon以内であることを検証します。
  入力を再正規化せず、cosθが±1に丸められたときも小さなsinθを保持します。
- `associated_legendre(l,m,cosθ)` はm>=0のPとθ微分を返します。x=cosθに関する微分ではありません。
  積分用と同じ極の打切り規約です。
- 公開入口はl=0〜64、|m|<=l、有限な角度と有限なmφを受け入れ、無効な入力は例外にします。
  次数の上限はlong doubleとdoubleが同じ範囲の環境でも階乗の中間値を表せる範囲です。
  [規格化と位相](https://dlmf.nist.gov/14.30.E1)はCondon–Shortley規約に従います。

本体のC側は同じ内部計算核をインラインで使い、ホストC++への呼び出しを各軌道へ追加しません。
積分用と軌道用の式・演算順序を維持し、積分用の極処理を軌道用の処理で置き換えてはいません。
公開C++ APIは入力エラーでプロセス終了せず、本体の互換関数だけが従来のエラー処理を持ちます。
独立した利用例は `examples/harmonics/` にあります。

## 複素から実球面調和関数へ変換する

`<openmx/basis/real_harmonics.hpp>` の `RealHarmonicTransform(maximum_degree)` は、
次数0〜64までの変換係数を連続配列で所有します。入力の複素調和関数は前節と同じ
Condon–Shortley位相で、磁気量子数m=-l,...,+lの順に渡します。

```cpp
#include <openmx/basis/real_harmonics.hpp>
#include <openmx/basis/harmonics.hpp>
#include <array>

const openmx::basis::RealHarmonicTransform conversion(4);
std::array<std::complex<double>, 3> complex_p, real_p;
for (int m=-1; m<=1; ++m)
  complex_p[m+1] = openmx::basis::integral_harmonic(1, m, 0.7, -0.3).value;
conversion.transform(1, complex_p, real_p); // px, py, pz
```

- 出力順はs、p=(x,y,z)、d=(z²,x²−y²,xy,xz,yz)、l>=3ではm=0,cosφ,sinφ,...,cos(lφ),sin(lφ)です。
  本体の正規化とCartesian成分の符号を維持します。変換行列はユニタリで、一般の複素入力に
  対する出力の虚部を捨てません。実球面調和関数の虚部には丸め誤差が残る場合があります。
- `transform()` は両spanにちょうど2l+1要素を要求します。完全・部分的な重なりを許し、
  有界のstack領域だけで計算します。不正な次数・寸法は例外になり、出力を保持します。
- `coefficient(l,row,column)` は有効な2l+1四方の範囲を検証します。
  `coefficients()` は読み取り専用span、`view()` はCにも渡せる不変ビューです。
  配置は `[degree][real row][complex column]`、全次数の幅は2*maximum_degree+1で、余白はゼロです。
- コピーは独立し、ムーブ元は空になります。借用ビューは所有者の破棄・代入まで有効です。
  同じ所有者の並行読み取りが可能で、係数参照・変換時の追加確保はありません。
- `<openmx/basis/real_harmonics.h>` の `openmx_basis_real_harmonics_fill()` は、呼出側の
  `dcomplex` 連続配列に係数を生成するC入口です。容量は複素要素数で指定し、成功時も確保しません。
  不正入力では0を返して数値出力を保持し、診断バッファは数値配列と重ねない契約です。
  `dcomplex` と `std::complex<double>` のレイアウトが同一であるという仮定は置きません。

本体では計算ごとに一度所有し、重なり・非局所積分、VNA、Force/gridのCPU処理、
Voronoi軌道モーメント、Wannier射影へ不変ビューを渡します。GPUへの既存の転送配置は保持します。
旧 `Comp2Real` 大域配列は不要になりました。独立利用例は `examples/real_harmonics/` です。

## Gaunt角度積分と係数表

`<openmx/basis/gaunt.hpp>` の `gaunt(bra,left,right)` は、規格化された複素球面調和関数の
`∫conj(Y_bra) Y_left Y_right dΩ` を返します。位相は `integral_harmonic()` と同じ
Condon–Shortley規約です。状態の初期化やヒープ確保を必要としません。

```cpp
#include <openmx/basis/gaunt.hpp>
const double integral = openmx::basis::gaunt({2,0}, {1,0}, {1,0});
// sqrt(5/pi)/5
const openmx::basis::GauntTable table(4, 4, 8);
const double same = table.coefficient(2, 0, 1, 0, 1);
```

- `HarmonicIndex{degree,order}` は整数の軌道角運動量です。各次数は0〜38、磁気量子数は
  −degree〜degreeです。三角条件を満たす組では次数の合計を38以下に制限します。
  これは元の階乗表0!〜39!を越えないための上限で、高次数一般の精度を保証するAPIではありません。
  三角条件を満たさない組はゼロなので、各次数38まで受け付けます。
- `GauntTable(l0max,l1max,llmax)` は `G(l0,m0,l1,m1,LL,m0−m1)` を1本の配列に所有します。
  配置は `[l0*(l0+1)+m0][LL][l1][l1+m1]`、最終幅は2*l1max+1です。
  `row(l0,m0,LL,l1)` は有効な2*l1+1要素の読み取り専用spanを返します。
  `view()` はC互換の不変ビューで、余白はゼロ、選択則でゼロになる有効要素は旧演算の符号も保持します。
- 表の全ての三角条件を満たす組が次数合計38以下であることを検証します。
  `(4,12,24)` のような長方形の表は、高いLLの行が選択則でゼロになるため利用できます。
  単に3つの最大次数の合計で拒否する制約にはしていません。
- コピーは独立した所有、ムーブ元は空になります。借用は所有者の破棄・代入まで有効です。
  参照・評価は無確保で、並行読み取りが可能です。不正入力では例外を返します。
- `<openmx/basis/gaunt.h>` にスカラー評価と、呼出側のdouble配列を満たすC APIがあります。
  成功時は確保せず、失敗時は数値出力・ビューを保持して0を返します。
  診断バッファは数値出力と重ねないでください。表の容量はdouble要素数で指定します。

本体の `Gaunt()` も同じ計算核を使います。VNAの準備所有者は、CPUで必要になったときに
射影子用とVNA2/3用の表を作り、両向きのVNA2/3計算で再利用します。再取得は無確保・同一アドレスで、
初回の並行取得と確保失敗後の再試行にも対応します。GPUだけで処理する場合はホスト表を作りません。
独立した利用例は `examples/gaunt/` です。
積分と3j記号の関係は [DLMF §34.3](https://dlmf.nist.gov/34.3)、
三角条件と階乗和は [DLMF §34.2](https://dlmf.nist.gov/34.2) を参照してください。

## 動径・波数表のデータと数値規約

- 角運動量 `angular` は `0 <= l <= 29`。チャネルの並び順は呼び出し元が指定し、
  同じ `l` を持つ複数の動径関数も扱えます。磁気量子数の展開は行いません。
- 格子と入力値は有限、半径は非負で狭義単調増加。OpenMX の単位は bohr、波数は逆 bohr。
  値はそのままの動径関数 `R_l(r)` として扱い、`r` の掛け算・正規化・位相の変更はしません。
- PAO/projector は3点以上、neutral potential/projector は6点以上必要です。
  最大格子半径より外側はゼロ、neutral potential は指定した cutoff より外側もゼロです。
- 最初の点より内側の PAO は `l=0` なら最初の値、それ以外はゼロ。
  projector の `l=0` は最初の補間区間を延長します。
  neutral potential/projector は従来の原点側二次継続を使用します。
- PAO と neutral 系の最初の区間は元の OpenMX と同じ「2番目のステンシルで外挿」する規約です。
  最初の格子点の値に必ず一致する通常の補間とは異なります。波数の非負の最初の区間も同様です。
  この動作を変更する場合は、単なる整理と区別して数値仕様を変更する必要があります。
- 波数格子は `k[i] = i*step`。1点・2点の表の特殊処理、projector の微小刻みでの線形処理、
  負の波数に対する従来の延長も維持します。最後の点より外側はゼロです。

`RadialTable` は全入力を検証して連続領域へコピーする不変オブジェクトで、
コピーは独立、ムーブは所有権を移します。`RadialTableView` は検証だけを行って借用します。
借用する格子、チャネル記述、各行はビューより長く生存し、サンプリング中に変更してはいけません。
構築・検証は入力サイズに比例し、各サンプリングは `O(log N)`、追加確保ゼロです。
所有済みの表は複数スレッドから同時に読み出せます。

`sample_radial_row` と `sample_reciprocal_row` は高速な借用関数です。
毎回の全入力走査はせず、有限性・単調性は呼び出し元の前提条件です。
外部データを直接利用する場合は、まず `RadialTable` / `RadialTableView` で検証してください。
C++ API の入力エラーは例外で報告し、C ABI は例外を捕捉して失敗を返します。
C ABI は呼び出しごとのエラーバッファを使い、失敗時に数値出力を書き換えません。

## 数秒で検証する

`ctest --test-dir build/basis -L unit --output-on-failure` で、解析的な多項式、
旧実装から固定した参照値、端点、無効入力、コピー・借用・並行性、
確保失敗を検証します。PAO は解析値を持つ s〜g・各2多重度の小入力で、全軌道・密度、
不正形式、並行読み込みと所有権を検証します。VPSはスカラー/j依存・旧形式・PCC有無・
射影子0個の組合せを検証します。密度では解析的な値・導関数、有限差分、旧実装の参照値、
一定密度のHartree解析解、異常入力と出力保持を検証します。
スカラー評価とHartree積分が確保を行わないことも検査します。
生成したGauss–Legendre則は多項式の厳密積分、節点の順序・対称性、次数1500までの規格化、
コピー・ムーブ・並行生成と確保失敗時の出力保持を検査します。
球Besselは独立したべき級数・解析式・Legendre積分と有限差分で検証します。
キャッシュのキー・借用寿命・無確保の再利用、確保失敗後の継続、独立した並行評価も検査します。
Lebedevは全15,288係数の凍結した指紋、球面上の混合多項式積分、所有・移動・容量再利用、
確保故障時の保持、並行読み取りとC APIで検証します。
球面調和関数はRodrigues多項式、有限差分、球面上の直交性、加法定理、両極の解析的微分、
無効入力、無確保・並行評価を検査します。本体のC互換境界は
`python3 tests/run_harmonics_cpu.py --sanitize` で凍結した旧実装とビット単位で比較します。
これはリポジトリ側の試験であり、独立パッケージの試験には不要です。
実調和関数への変換は旧係数とのビット一致、次数64までのユニタリ性、独立したs〜gの
Cartesian多項式、所有・移動・確保失敗・重なり・並行評価・実C呼び出しで検証します。
本体のVoronoi角運動量行列は `python3 tests/run_angular_momentum_cpu.py /path/to/libopenmx_core.so` で
旧実装との一致と交換関係・角運動量の二乗を検証します。
Gauntは凍結した旧C実装とのビット一致、解析的なs/p/d積分と選択則・対称性、独立した
Lebedev球面積分で検証します。表の余白、長方形の選択則ゼロ、階乗表の上限、所有・確保故障・
並行評価と実C APIも検査します。本体境界は `python3 tests/run_gaunt_cpu.py --sanitize` で
約203万個の係数を旧C実装と比較します。

`basis-package-consumer` はこのディレクトリだけをコピーして独立ビルド・インストールし、
インストール先と元のソース・ビルドディレクトリを移動してから、外部 C/C++ プログラムを
ビルド・実行します。上位の `core/` や SDK への依存を持ち込むと、この試験が失敗します。
サニタイザーは `-DOPENMX_BASIS_SANITIZERS=ON` で有効にできます。
