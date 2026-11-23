# OpenMX Exx

ハイブリッド汎関数（HSE06 と PBE0）の厳密交換を、局所 RI（LRI）で計算するための C++20 ライブラリーです。
大域変数も状態も持ちません。libxc は使いません。

いま入っているのは、構造ごとの準備までです（`docs/EXX_IMPLEMENTATION_GUIDE.md` の 6.2 節）。
交換行列の構築（6.4 節）と C の入口は、まだありません。式の番号は、この指示書のものです。

| ヘッダー | 内容 |
|---|---|
| `openmx/exx/kernel.hpp` | 相互作用のカーネル $\tilde v(\kappa)$ 。重なり、クーロン、短距離（HSE06）、打ち切りクーロン（PBE0）の 4 つ |
| `openmx/exx/two_center.hpp` | 動径関数の球ベッセル変換（式 (11)）、二中心積分（式 (12)）、クーロンポテンシャルの動径部分（式 (13)） |
| `openmx/exx/abf.hpp` | PAO の動径関数を等間隔の格子に移す関数と、補助基底関数（ABF）の生成 |
| `openmx/exx/lri.hpp` | 元素ごとの表、三中心積分（5.7 節）、式 (7) の解 |
| `openmx/exx/system.hpp` | site、近傍リスト、T ブロックと V ブロック |

依存するのは `OpenMX::basis`（球ベッセル関数、Gaunt 係数、PAO の補間）だけです。
MPI、BLAS、LAPACK、GPU SDK、OpenMX の実行ファイルは要りません。
式 (7) のコレスキー分解と、小さい対称行列の固有値分解は、ライブラリーの中に実装してあります。
CMake 3.25.2 以上と、C++20 のコンパイラーでビルドできます。ライセンスは OpenMX と同じ GNU General Public License です。

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

## テスト

- `exx-kernel-analytic`: 4 つのカーネルの値と $\kappa\to0$ の極限を、50 桁で評価した参照値（`tests/fixtures/kernel_reference.txt`）と比べます。相対 1e-13。
- `exx-two-center-gaussian`: ガウス関数の二中心積分を解析値と比べます。s 同士は 1e-10、p を含む組は 1e-6。
- `exx-two-center-symmetry`: 対を逆から見た積分の転置と、座標系の回転に対する不変性。1e-10。
- `exx-two-center-overlap`: PAO の重なり積分を、本体が書き出した重なり行列（`tests/fixtures/overlap_reference.txt`）と比べます。本体と同じ離散化で 1e-8。
- `exx-abf-pca`: ABF の正規直交性（1e-12）、射影の残差と捨てた固有値の一致（相対 1e-8）、軌道の順序と符号を変えたときの再現性。
- `exx-three-center-gaussian`: ガウス関数の三中心積分を解析値と比べます。1e-8。
- `exx-lri-identity`: ABF の線形結合で作った関数の係数が再現されること。1 原子と離れた 2 原子では係数ごとに 1e-10、結合距離の 2 原子ではクーロンノルムで 1e-10。
- `exx-lri-convergence`: ABF を増やしたときの自己相互作用の和を記録します。和が減らないことを確かめます。
- `exx-system-neighbours`: 近傍リストを、セルの総当たりの探索と比べます。
- `exx-system-blocks`: T ブロックと V ブロックの 2 通りの見方の一致、結晶の記述を変えたときの一致、展開した積の電荷と重なり行列の比較。
- `exx-package-consumer`: インストールしたパッケージから 2 つの例題をビルドします。

## フィクスチャーの作り直し

- `tests/fixtures/kernel_reference.txt` は `python3 tools/exx_reference.py --kernels modules/exx/tests/fixtures/kernel_reference.txt` が書きます（mpmath が要ります）。
- `tests/fixtures/overlap_reference.txt` は、CPU 版の本体で `tests/fixtures/overlap_reference.dat` を実行し、できた `.scfout` を `tools/exx_overlap_fixture.py` に渡して作ります。手順はスクリプトの冒頭にあります。
