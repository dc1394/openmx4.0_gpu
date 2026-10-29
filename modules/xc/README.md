# OpenMX Xc

OpenMX の交換相関汎関数を、大域変数も状態も持たない純粋な関数として独立して利用する
C++20 ライブラリです。公開範囲は Ceperley–Alder LDA（Perdew–Zunger 形、エネルギー密度・
ポテンシャル・その差・密度微分）、同じ形の LSDA、Perdew–Wang 92 相関、一様電子ガスの
Slater 交換、スピン分解した GGA-PBE（エネルギー密度と、密度・密度勾配についての導関数）です。
本体（`Set_XC_Grid.c`、`Force.c`、`Stress.c`、`Total_Energy.c` の CPU 経路）もこの実装を使用します。
GPU 側の OpenACC カーネル（`Set_XC_Grid.c` の `SXG_` 関数）は本体に残っています。

演算は OpenMX 4.0 の C ルーチンを演算の順序ごと移したもので、密度の下限
（1e-14 bohr^-3 とスピン分極の clamp）も含めて、縮約なしでコンパイルした元のコードと
bit 一致します。元の大域変数 `dc_Type`（LDA+U の cFLL 規則: 両スピンを平均し分極を 0 に
する）は引数 `symmetrize_spins` になりました。元の 5 つの C ソースは `tests/fixtures/` に
凍結し、テストがライブラリとの bit 一致を確かめます。単位は Hartree 原子単位です。

CMake 3.25.2 以上と C++20 のコンパイラ・標準ライブラリでビルドできます。
このディレクトリだけを取り出して利用でき、MPI、BLAS、Kokkos、GPU SDK、
OpenMX の実行ファイル・大域変数ヘッダー、ネットワーク接続を必要としません。
コードのライセンスは元の OpenMX と同じです。

## ビルドと外部利用

リポジトリのルートで実行します。

```sh
cmake -S modules/xc -B build/xc -DCMAKE_BUILD_TYPE=Release
cmake --build build/xc --parallel
ctest --test-dir build/xc --output-on-failure
cmake --install build/xc --prefix "$PWD/build/xc-install"

cmake -S modules/xc/examples/pbe -B build/xc-example \
  -DCMAKE_PREFIX_PATH="$PWD/build/xc-install"
cmake --build build/xc-example
```

C++ からは `openmx/xc/functionals.hpp`（`openmx::xc::pbe` など）、C からは
`openmx/xc/xc.h`（`openmx_xc_pbe` など。配列の形は元の OpenMX ルーチンと同じ）を使います。
計算核そのものは `openmx/xc/kernels.h` の `static inline` な C 関数（`openmx_xc_kernel_pbe` など）で、
ライブラリも本体の GPU デバイスコード（nvc の OpenACC `routine seq`、hipcc/clang の OpenMP
`declare target`。後者は include 前に `OPENMX_XC_KERNELS_OMP_TARGET` を定義）もこれを使います。
インライン展開されるため、ライブラリと bit 一致するのは include 側も浮動小数点の縮約なし
（`-ffp-contract=off`）でコンパイルしたときだけです（GPU のデバイスコードや既定で縮約する
コンパイラでは最終桁が異なります）。

## テスト

- `xc-legacy`: 凍結した元のルーチンとの bit 一致（密度の下限、分極の clamp、rs<1 と rs>=1、
  対称化の有無、負の密度を含む格子）。
- `xc-analytic`: 一様電子ガスの Slater 交換の閉じた式、LDA と LSDA の非分極極限の一致、
  PW92 の論文の式（別の形）との一致、勾配ゼロでの PBE の LDA への帰着。
- `xc-derivatives`: ポテンシャルと導関数を中心差分と比較（密度微分、スピンごとの微分、
  PBE の勾配微分）。
- `xc-invalid`: 境界条件（下限以下の密度、不明な量の指定、NULL）。
- `xc-c-api`、`xc-package-consumer`: C からの利用と、インストールしたパッケージからの利用。
