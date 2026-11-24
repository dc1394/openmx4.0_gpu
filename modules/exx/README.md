# OpenMX Exx

ハイブリッド汎関数（HSE06 と PBE0）の厳密交換を、局所 RI（LRI）で計算するための C++20 ライブラリーです。
大域変数も状態も持ちません。libxc は使いません。

いま入っているのは、構造ごとの準備（`docs/EXX_IMPLEMENTATION_GUIDE.md` の 6.2 節）、交換行列の構築（6.4 節）、原子の位置についての $E_x$ の微分（5.9 節）、C の入口です。
密度行列の収集、ハミルトニアンへの加算、MPI での分担は、本体に組み込むときに加えます。式の番号は、この指示書のものです。

| ヘッダー | 内容 |
|---|---|
| `openmx/exx/kernel.hpp` | 相互作用のカーネル $\tilde v(\kappa)$ 。重なり、クーロン、短距離（HSE06）、打ち切りクーロン（PBE0）の 4 つ |
| `openmx/exx/two_center.hpp` | 動径関数の球ベッセル変換（式 (11)）、二中心積分（式 (12)）と変位についてのその微分、クーロンポテンシャルの動径部分（式 (13)） |
| `openmx/exx/abf.hpp` | PAO の動径関数を等間隔の格子に移す関数と、補助基底関数（ABF）の生成 |
| `openmx/exx/lri.hpp` | 元素ごとの表、三中心積分（5.7 節）、式 (7) の解と、対の変位についてのその微分（式 (18)） |
| `openmx/exx/system.hpp` | site、近傍リスト、T ブロックと V ブロック。対の 2 通りの記述のうち、準備が計算する側を返す関数もあります |
| `openmx/exx/exchange.hpp` | 交換行列の構築（6.4 節）。site の形式と Γ 点の形式、 $E_x$ （式 (4)）、外側反復の判定に使う量（式 (15)）、 $E_x$ の T ブロックと V ブロックについての導関数（式 (16)、(17)）。導関数には、中心の原子 1 個の配列だけで計算する形（2 つの形式）もあります |
| `openmx/exx/forces.hpp` | 密度行列を固定した $E_x$ の、原子の位置についての微分（5.9 節）。中心の原子ごとの導関数に、対ごとの連鎖律を適用します |
| `openmx/exx/dense_backend.hpp` | 行列積の関数表（7.2 節）を包むクラス。表を渡さなければ、ライブラリーの中のループで計算します |
| `openmx/exx/exx.h` | C の入口。関数表の型、構造の準備、Γ 点の形式と site の形式の構築、式 (4) と (15)、原子の位置についての微分 |

依存するのは `OpenMX::basis`（球ベッセル関数、Gaunt 係数、PAO の補間）だけです。
MPI、BLAS、LAPACK、GPU SDK、OpenMX の実行ファイルは要りません。
式 (7) のコレスキー分解と、小さい対称行列の固有値分解は、ライブラリーの中に実装してあります。
交換行列の構築に使う行列積は、呼び出し側が関数表で渡せます（BLAS の DGEMM など）。
CMake 3.25.2 以上、C++20 のコンパイラー、C のコンパイラーでビルドします。C のコンパイラーは、C の入口を C のプログラムから呼ぶ試験に使います。
ライセンスは OpenMX と同じ GNU General Public License です。

## 規約

- 単位は Hartree 原子単位です。長さは bohr、波数は 1/bohr で表します。
- 実球面調和関数の順序は、OpenMX の軌道と同じです。p は (x, y, z)、d は (z², x²−y², xy, xz, yz)、l が 3 以上では m = 0、cos φ、sin φ、cos 2φ、sin 2φ、…の順に並びます。
- 二中心積分の変位は、1 つ目の関数の中心から 2 つ目の関数の中心へ向かうベクトルです。
- 成分の番号は、動径関数の順に 2l+1 個ずつ振ります。PAO は本体と同じ順（l の昇順、同じ l では多重度の順）で渡します。
- 動径の格子は、0 からカットオフ半径までを等分します。逆空間の格子は、0 から波数の上限までを等分します。積分はどちらも台形則です。
- 格子の設定は、本体の重なり積分と同じにします。動径の区間の数は `1DFFT.NumGridR`（既定値 900）、波数の上限は `1DFFT.EnergyCutoff` の平方根（既定値 60/bohr）、逆空間の区間の数は `1DFFT.NumGridR` です。
- ABF の半径は、PAO のカットオフ半径と同じです。
- T ブロックは、原子ごとに C 近傍の順に横に並べた 1 つの配列で持ちます。要素は `t[a][(P*u_a + mu)*ell_a + c_offset[a][i] + lambda]` です。`c_offset[a][i]` は i 番目の C 近傍の最初の列、`ell_a` は列の総数（C 近傍の PAO の数の和）です。`t_block(system, a, i)` は 1 個のブロックを取り出します。
- V ブロックは `v[a][i][P*n_b + Q]` の形で持ちます。近傍リストの 0 番目は原子自身で、残りは原子の番号、セルの 3 つの整数の順に並びます。
- 密度行列と交換行列は、1 スピン分です。密度行列の占有数は 0 から 1 です。スピン分極のない計算では、呼び出し側が $E_x$ を 2 倍します。
- site の形式のブロック $M_{xy}(\mathbf R)$ は、行が原点のセルの原子 $x$ 、列がセル $\mathbf R$ の原子 $y$ です。Γ 点の形式は $N\times N$ の配列で、軌道は原子の順に並びます。
- site の形式には、ブロックを 1 本の配列に並べた形もあります。並べる順は `BlockLayout` が持ちます。キーは $(x,y,\mathbf R)$ の昇順で、`offset[i]` が i 番目のブロックの最初の要素です。一覧に入るのは、ある中心の原子 $A$ について、行の site が $\mathrm{loc}(A)$ 、列の site が $S(A)$ に属するブロックです。構築が読む $D$ と書く $K$ は、どちらもこの一覧に収まります。一覧は近傍リストだけで決まるので、同じ構造を準備したプロセスは同じ配列を持ち、要素ごとに足し合わせられます。
- 配列は行優先です。関数表の `dgemm` だけは、BLAS と同じ列優先で受け取ります。

## 使い方

```cpp
#include <openmx/exx/abf.hpp>
#include <openmx/exx/system.hpp>
using namespace openmx::exx;

// 元素ごと：PAO を格子に移し、ABF と表を作る
const RadialGrid grid{0.0, cutoff/900, 900};
const ReciprocalGrid reciprocal = ReciprocalGrid::uniform(60.0, 900);
const std::vector<RadialFunction> orbitals = sample_orbitals(grid, pao_mesh, pao_functions);
const AuxiliaryBasis abf = make_auxiliary_basis(grid, orbitals, 1e-4);   // scf.exx.abf.pca
std::vector<SpeciesTable> species;
species.emplace_back(grid, orbitals, abf.functions, reciprocal);

// 構造ごと：近傍リスト、T ブロック、V ブロック
SystemOptions options;                               // 既定は HSE06 の短距離カーネル（ω = 0.11/bohr）
options.interaction = truncated_coulomb_kernel(12.0); // PBE0 の場合
const System system = prepare_system(structure, species, reciprocal, options);
```

`prepare_system()` は、単位胞の全ての原子を 1 回の呼び出しで処理します。MPI での分担は、本体に組み込むときに加えます。

交換行列は、準備した `System` と密度行列から作ります。

```cpp
#include <openmx/exx/exchange.hpp>

// Γ 点だけの計算：N x N の密度行列（1 スピン分）から K と E_x を作る
const std::vector<double> k = exchange_gamma(system, density, 0.25);   // 混合比 α = 0.25
const double e_x = exchange_energy_gamma(density, k);                  // 式 (4)

// 原子 1 個の寄与だけを足す（係数は -α）。原子をランクに配るときに使う
add_exchange_gamma(system, atom, -0.25, density, k_part);

// k 点のある計算：ブロック D_xy(R) を書く関数を渡し、ブロック K_xy(R) を受け取る
const DensityBlocks blocks = [&](const BlockKey& key, std::span<double> block) { /* D_xy(R) を書く */ };
const BlockMatrix k_blocks = exchange_blocks(system, blocks, 0.25);

// 行列積を BLAS に任せる
const OpenMXExxDenseBackend table{&my_dgemm, nullptr};
const std::vector<double> k_blas = exchange_gamma(system, density, 0.25, DenseBackend(&table));
```

行列積の回数は、中心の原子 1 個につき 2 回と、V 近傍 1 個につき 1 + 4n 回です（n は中心の原子の ABF の個数）。
V 近傍ごとの 4 つの項は、中心の原子の ABF を 1 個ずつ計算します。
関数表の `dgemm` は、β が 0 のとき結果の行列を読みません（BLAS の DGEMM と同じ約束です）。ライブラリーは、β が 0 の積に、確保したままの配列を渡します。
Γ 点の形式は、同じ原子の周期像を行列積の前にまとめます。V ブロックは周期像について足し合わせ、T ブロックの列は同じ軌道に属するものを足し合わせます。
このため、Γ 点の形式で数える V 近傍は、site ではなく原子です。セルが近傍の範囲より小さくても、行列積は増えません。

原子の位置についての微分は、準備に使った引数と `System` から作る `PositionGradient` で計算します。

```cpp
#include <openmx/exx/forces.hpp>

// 密度行列を固定した dE_x/dτ。gradient[3*原子 + 軸] に、中心の原子ごとの寄与を足す（係数は -α）
const PositionGradient derivative(structure, species, reciprocal, options, system);
std::vector<double> gradient(3*structure.positions.size(), 0.0);
for (std::size_t atom = 0; atom < structure.positions.size(); ++atom)
  derivative.add_gamma(atom, -0.25, density, gradient);          // Γ 点の形式
// site の形式では derivative.add_blocks(layout, atom, -0.25, density_blocks, gradient)
```

力は、この微分の符号を変えたものです。対ごとの寄与を一方の原子に足し、他方の原子から引くので、原子についての和は 0 になります。
微分は、近傍リストを固定したときのものです。対が近傍の打ち切りをまたぐと $E_x$ は不連続に変わり、微分はその跳びを含みません。
密度行列は対称なもの（site の形式では $D_{xy}(\mathbf R)=D_{yx}(-\mathbf R)^{\mathsf T}$ ）を渡します。軌道の応答は含みません。本体では、既存の Pulay 項が担います。
対は、準備が計算した側の記述で扱います。式 (7) の行列の分解も準備と同じになります。コレスキー分解の経路は保存した T ブロックを使い、固有値分解の経路は右辺を計算し直します。
1 回の呼び出しは、その原子の C 近傍の全ての対を分解します。対は両方の原子から 1 回ずつ訪れるので、分解の回数は準備の 2 倍です。

C からは `exx.h` を使います。Γ 点の形式と、1 本の配列に並べた site の形式を扱えます。

```c
#include <openmx/exx/exx.h>

OpenMXExxOptions options;
openmx_exx_default_options(&options);           /* HSE06。キーワードの既定値 */
OpenMXExxState *state = openmx_exx_create();
char message[256];
if (openmx_exx_prepare(state, &options, &geometry, message, sizeof message) != 0) { /* message に理由 */ }
const int n = openmx_exx_orbital_count(state);
/* k を 0 で埋めてから、全ての原子の寄与を係数 -α で足す */
openmx_exx_add_gamma(state, 0, openmx_exx_atom_count(state), -0.25, density, k, NULL, message, sizeof message);
const double e_x = openmx_exx_energy_gamma((size_t)n*n, density, k);
openmx_exx_destroy(state);
```

site の形式は、ブロックの一覧を作ってから使います。k 点の計算では、呼び出し側が式 (5) で $D_{xy}(\mathbf R)$ を配列に入れ、できた $K_{xy}(\mathbf R)$ から式 (14) で $K(\mathbf k)$ を作ります。

```c
openmx_exx_prepare_blocks(state, message, sizeof message);      /* 一覧を作る。準備をやり直すと消える */
const long blocks = openmx_exx_block_count(state);
const size_t elements = openmx_exx_block_elements(state);       /* 配列の要素数 */
for (long b = 0; b < blocks; ++b) {
  int x, y, cell[3];
  size_t offset;
  openmx_exx_block(state, b, &x, &y, cell, &offset);            /* density[offset ...] に D_xy(R) を行優先で入れる */
}
openmx_exx_add_blocks(state, 0, openmx_exx_atom_count(state), -0.25, density, k, NULL, message, sizeof message);
double asymmetry;                                               /* K_xy(R) と K_yx(-R) の転置の差の最大値 */
openmx_exx_symmetrize_blocks(state, k, &asymmetry, message, sizeof message);
const double e_x = openmx_exx_energy_gamma(elements, density, k);
```

`openmx_exx_find_block()` は、原子の対とセルからブロックの番号を返します。式 (4) と式 (15) は配列の要素についての和なので、Γ 点の形式と同じ関数を使います。

原子の位置についての微分は、`openmx_exx_add_position_gradient_gamma()` と `openmx_exx_add_position_gradient_blocks()` が足し込みます。

```c
double gradient[3*ATOMS] = {0.0};                               /* gradient[3*原子 + 軸] = dE_x/dτ */
openmx_exx_add_position_gradient_gamma(state, 0, openmx_exx_atom_count(state), -0.25, density, gradient,
                                       NULL, message, sizeof message);
```

中心の原子の範囲を分けて呼んだ結果は、足し合わせると全体の微分になります。失敗した呼び出しは `gradient` を変えません。

`openmx_exx_prepare()` は、元素ごとの表を状態に残します。作り直すのは、PAO、格子、しきい値のどれかが変わったときだけです。
失敗した呼び出しは、状態を変えません。関数は例外を外に出さず、0 以外の返り値とメッセージで知らせます。

## ビルドと外部利用

リポジトリーのルートで実行します。`modules/basis` が隣にあれば、一緒にビルドされます。

```sh
cmake -S modules/exx -B build/exx -DCMAKE_BUILD_TYPE=Release
cmake --build build/exx --parallel
ctest --test-dir build/exx --output-on-failure
cmake --install build/exx --prefix "$PWD/build/exx-install"

cmake -S modules/exx/examples/lri -B build/exx-lri -DCMAKE_PREFIX_PATH="$PWD/build/exx-install"
cmake --build build/exx-lri
build/exx-lri/lri_example DFT_DATA19/PAO/C6.0.pao s2p2d1 DFT_DATA19/PAO/H6.0.pao s2p1 2.06
```

例題は 2 つあります。`examples/kernel` はカーネルの値を表にします。
`examples/lri` は `.pao` ファイルを読み、ABF の個数と LRI の精度の目安を出します。
`--generators s2p2d1f1` のように指定すると、ABF を PAO より大きい組の積から作ります。

## 精度の目安

次の値は、GCC 13.3.0 でビルドした `lri_example` と単体試験の実測値です（既定の格子、DFT_DATA19 の PAO）。

主成分分析のしきい値が既定値 1e-4 のときの ABF の個数（動径関数の数 / 成分を数えた数）です。

| 元素と基底 | 個数 | 間引く前 |
|---|---|---|
| H6.0-s2p1 | 7 / 15 | 7 / 15 |
| C6.0-s2p2d1 | 19 / 69 | 22 / 78 |
| N6.0-s2p2d1 | 17 / 65 | 22 / 78 |
| O6.0-s2p2d1 | 20 / 72 | 22 / 78 |
| Si7.0-s2p2d1 | 19 / 69 | 22 / 78 |
| Si7.0-s2p2d1f1 | 28 / 140 | 35 / 169 |

- 同じ原子の PAO の積は、ABF を間引かなければ丸め誤差の範囲で再現されます。既定のしきい値では、自己相互作用の和の不足が 1e-5 以下です。
- 2 つの原子の PAO の積は、同じ原子の積から作った ABF では完全には表せません。結合距離の C–C、C–H、Si–Si では、自己相互作用の和が 0.2% から 0.5% 不足します。比べた相手は、f 関数を含む大きい組（s5p5d4f3）の積から作った ABF です。この相手も下からの近似なので、実際の不足はこれ以上です。
- 式 (7) の行列は、対角成分を 1 に揃えた後の条件数が 6e6 に達します。係数の成分ごとの値は 1e-10 程度までしか決まりません。展開した密度は、クーロンノルムで 1e-12 まで決まります。
- コレスキー分解の枢軸が `LriOptions::pivot_tolerance`（既定値 1e-7）以下になると、固有値分解に切り替えます。上の元素の組では、切り替えは起きませんでした。
- 動径の格子を 4 倍に細かくしても、2 原子の和は相対 1e-8 しか変わりません。波数の上限を 2 倍にしても、12 桁まで変わりません。

## 構築の時間の目安

重なり行列のフィクスチャーのセルで、構築 1 回を測りました。
セルは 3 原子（Si7.0-s2p2d1f1、C6.0-s2p2d1、H6.0-s2p1）で、軌道は 38 個、ABF は 140、69、15 個、C 近傍は合計 227 site、V 近傍は合計 335 site です。
しきい値と打ち切りは既定値、計算機はクラウドのコンテナの 1 コア、コンパイラーは GCC 13.3.0 の `-O2` です。

| 形式 | 行列積 | 回数 | 時間 |
|---|---|---|---|
| Γ 点の形式 | OpenBLAS 0.3.26（1 スレッド）を関数表で渡す | 51 | 0.03 秒 |
| Γ 点の形式 | ライブラリーの中のループ | 51 | 0.04 秒 |
| site の形式 | OpenBLAS 0.3.26（1 スレッド）を関数表で渡す | 1,681 | 12.6 秒 |

site の形式には、同じ密度行列を全てのセルのブロックとして渡しました。Γ 点だけの計算に当たる条件です。
$K$ のブロックは 5,723 個（7.1 MB）で、 $\sum_{\mathbf R}K(\mathbf R)$ は Γ 点の形式の $K$ と 2.0e-15 まで一致しました。
Γ 点の形式も、周期像を行列積の前にまとめない形では、行列積が 1,681 回で 6.9 秒（ライブラリーの中のループでは 33 秒）でした。
大きい系では測っていません。

原子の位置についての微分の時間を、Si の 8 原子のセル（格子定数 10.26 bohr、Si7.0-s2p2d1、ABF は原子あたり 69 個）で測りました。
格子、しきい値、打ち切りは既定値で、C 近傍は合計 696 site、V 近傍は合計 792 site です。計算機とコンパイラーは上と同じで、行列積はライブラリーの中のループです。

| 処理 | 時間 |
|---|---|
| 構造ごとの準備（式 (7) の系は 345 個） | 3.3 秒 |
| Γ 点の形式の構築 1 回 | 0.47 秒 |
| 中心の原子ごとの導関数（8 原子の合計） | 0.86 秒 |
| 対ごとの連鎖律（8 原子の合計） | 8.6 秒 |

連鎖律の時間は準備の 2.6 倍です。対を 1 回だけ訪れる形にすれば、分解と積分の回数は半分になります（未実装）。

## テスト

- `exx-kernel-analytic`: 4 つのカーネルの値と $\kappa\to0$ の極限を、50 桁で評価した参照値（`tests/fixtures/kernel_reference.txt`）と比べます。相対 1e-13。
- `exx-two-center-gaussian`: ガウス関数の二中心積分を解析値と比べます。s 同士は 1e-10、p を含む組は 1e-6。
- `exx-two-center-symmetry`: 対を逆から見た積分の転置と、座標系の回転に対する不変性。1e-10。
- `exx-two-center-gradient`: 二中心積分の変位についての微分を、値の 4 次の中心差分（刻み 0.002 bohr）と比べます。4 つのカーネル、l が 6 と 4 までの全ての組、座標軸の上の変位、変位 0 で確かめます。差の上限は、最大の成分の 1e-9 倍です。s のガウス関数は閉じた式と比べます（1e-10）。
- `exx-two-center-overlap`: PAO の重なり積分を、本体が書き出した重なり行列（`tests/fixtures/overlap_reference.txt`）と比べます。本体と同じ離散化で 1e-8。
- `exx-abf-pca`: ABF の正規直交性（1e-12）、射影の残差と捨てた固有値の一致（相対 1e-8）、軌道の順序と符号を変えたときの再現性。
- `exx-three-center-gaussian`: ガウス関数の三中心積分を解析値と比べます。1e-8。
- `exx-lri-identity`: ABF の線形結合で作った関数の係数が再現されること。1 原子と離れた 2 原子では係数ごとに 1e-10、結合距離の 2 原子ではクーロンノルムで 1e-10。
- `exx-lri-convergence`: ABF を増やしたときの自己相互作用の和を記録します。和が減らないことを確かめます。
- `exx-lri-gradient`: 式 (18) を中心差分と比べます。計量と右辺の微分は 1e-8、展開係数を通した微分は、重みを $G R$ の形に取って 1e-8 です。乱数の重みでは、計量が特異に近い方向の桁落ちが差分に入るので 1e-5 にしています。固有値分解の経路は、固有ベクトルを捨てる場合も確かめます（1e-8）。対を逆の原子から見た結果とは 1e-9 で合います。式 (7) の解法を分解と求解に分けた `MetricFactorization` の解が、`solve_metric` とビット単位で同じことも確かめます。
- `exx-system-neighbours`: 近傍リストを、セルの総当たりの探索と比べます。
- `exx-system-blocks`: T ブロックと V ブロックの 2 通りの見方の一致、結晶の記述を変えたときの一致、展開した積の電荷と重なり行列の比較。
- `exx-dense-backend`: 行列積を、総和を書き下した値と比べます（1e-13）。DGEMM が拒否する引数、試験用の関数表、失敗を返す関数表も確かめます。
- `exx-exchange-fixture`: site の形式の $K$ のブロックと $E_x$ を、参照実装のフィクスチャー（`tests/fixtures/kbuild_reference.txt`）と比べます。絶対 1e-12。
- `exx-exchange-layout`: 1 本の配列に並べた site の形式を確かめます。ブロックの一覧はフィクスチャーの $K$ のキーと同じで、 $K$ はブロックの表を使う構築とビット単位で同じです。 $K_{xy}(\mathbf R)$ と $K_{yx}(-\mathbf R)$ の転置の差は 1e-12 以下で、平均を取った後は 0 です。
- `exx-exchange-gamma`: Γ 点の形式を、site の形式の $\sum_{\mathbf R}K(\mathbf R)$ と、密行列で書いた定義（式 (8)、(9)）の両方と比べます。1e-12。
- `exx-exchange-gradient`: 式 (16) と (17) を、T ブロックと V ブロックの全ての要素の中心差分と比べます。相対 1e-7。
- `exx-exchange-centre-gradient`: 中心の原子ごとに計算した導関数を確かめます。Γ 点の形式は式 (16)、(17) の実装と、site の形式は Γ 点の形式と比べます（最大の成分の 1e-12 倍）。セルごとに違う密度行列では、site の形式の全ての要素を中心差分と比べます（1e-9）。行列積の回数と、関数表を渡した結果も確かめます。
- `exx-exchange-backend`: 試験用の関数表を渡した結果を、渡さない結果と比べます（1e-12）。行列積の回数も確かめます。
- `exx-exchange-system`: PAO から作った T ブロックと V ブロックで $K$ を確かめます。孤立した原子では T ブロックを使わない式と、周期構造と大きいセルでは定義と比べます（相対 1e-12）。C の入口の結果が C++ の入口とビット単位で同じことも、2 つの形式で確かめます。site の形式の $\sum_{\mathbf R}K(\mathbf R)$ は Γ 点の形式と比べます（相対 1e-12）。
- `exx-exchange-forces`: 原子の位置についての $E_x$ の微分を、原子を動かして準備をやり直した $E_x$ の 4 次の中心差分（刻み 0.002 bohr）と比べます。密度行列は固定し、2 つの形式で確かめます。差の上限は、最大の成分の 1e-6 倍です。対象は、周期像を多く含む小さいセル、大きいセルの 3 原子（打ち切りクーロンのカーネル）、式 (7) を固有値分解で解く場合、固有ベクトルを 4 個捨てる場合です。微分の原子についての和が 0 であること（1e-12）、 $D(\mathbf R)=D$ を渡した site の形式が Γ 点の形式と合うこと（1e-12）、関数表を渡した結果、C の入口の結果も確かめます。動かした構造の近傍リストが元と同じことも確かめます。
- `exx-c-api`: C99 のプログラムから C の入口を呼びます。2 つの形式の構築、ブロックの一覧、準備をやり直した後の状態、原子の位置についての微分（中心差分との比較を 1 成分）を確かめます。
- `exx-package-consumer`: インストールしたパッケージから、2 つの例題と C の入口の試験をビルドします。

## フィクスチャーの作り直し

- `tests/fixtures/kbuild_reference.txt` は `python3 tools/exx_reference.py --dump modules/exx/tests/fixtures/kbuild_reference.txt` が書きます（NumPy が要ります）。試験は保存したファイルを読みます。
- `tests/fixtures/kernel_reference.txt` は `python3 tools/exx_reference.py --kernels modules/exx/tests/fixtures/kernel_reference.txt` が書きます（mpmath が要ります）。
- `tests/fixtures/overlap_reference.txt` は、CPU 版の本体で `tests/fixtures/overlap_reference.dat` を実行し、できた `.scfout` を `tools/exx_overlap_fixture.py` に渡して作ります。手順はスクリプトの冒頭にあります。
