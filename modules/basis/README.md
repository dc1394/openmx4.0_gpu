# OpenMX Basis

OpenMX の基底処理を独立して利用する C++20 ライブラリです。
現在の公開範囲は PAO、非局所射影子、中性原子ポテンシャル、VNA 射影子の
動径補間と PAO/射影子の波数空間補間です。本体もこの実装を使用します。
PAO/VPS ファイルの読み込み、球面調和関数、Fourier-Bessel 変換、2中心積分は
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
```

外部プロジェクトの `CMakeLists.txt` は以下だけでリンクできます。

```cmake
find_package(OpenMXBasis 0.1 CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE OpenMX::basis)
```

公開ヘッダーは `<openmx/basis/radial_table.hpp>`（C++）と
`<openmx/basis/radial.h>`（C ABI）です。C の呼び出し元でもリンク時には
C++ ランタイムが必要なので、CMake の `project()` で `C CXX` を有効にします。
静的ライブラリは PIC で生成します。CMake のビルド中の利用には
`add_subdirectory(path/to/basis)` と同じ `OpenMX::basis` ターゲットが使えます。
0.1 系は初期 API であり、C++ バイナリ ABI の固定は約束しません。

## データと数値規約

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
確保失敗を検証します。スカラー補間が確保を行わないことも検査します。

`basis-package-consumer` はこのディレクトリだけをコピーして独立ビルド・インストールし、
インストール先と元のソース・ビルドディレクトリを移動してから、外部 C/C++ プログラムを
ビルド・実行します。上位の `core/` や SDK への依存を持ち込むと、この試験が失敗します。
サニタイザーは `-DOPENMX_BASIS_SANITIZERS=ON` で有効にできます。
