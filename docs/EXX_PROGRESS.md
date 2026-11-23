# EXX 実装の進捗

## 現在の状態

- 完了したマイルストーン: M0, M1
- 作業中: なし（次は M2）
- 基準コミットからのブランチ: `feature/exx`（基準 `4e0ee247438a3c9358e95adee00a6ff60a4c618b`）。リモートは `origin/feature/exx` で、2026-11-23 に最初のプッシュをした。コミットとプッシュは、別のブランチや PR を作らずに `feature/exx` へ直接行う（2026-11-23、利用者の指示）。強制プッシュと履歴の書き換えはしない。
- M1 の短距離部分は、2026-11-23 に定義を書き直した（利用者の指示）。libxc 7.1.2 の形をやめ、Quantum ESPRESSO 7.6 と同じ形にしている。コミットは `615ad67f`（指示書の改訂 3）と `e8c8bf6a`（参照実装、フィクスチャー、計算核、試験）である。下の表で「M1（書き直しの後）」とした行が、今のコードでの結果である。書き直す前の記録は、このファイルの `e66e3cff` の版にある。

## 受け入れ試験の結果

| マイルストーン | 試験 | 結果 | 日付 | 条件（計算機、ビルド） |
|---|---|---|---|---|
| M0 | 参照実装の自己試験（`python3 tools/exx_reference.py`） | `self-test passed`。16 項目が全て `ok`。差の最大は T1 の 3.7e-15、T6（相対）は 2.4e-12 と 9.7e-12 | 2026-11-23 | RTX 5080 の計算機のホスト。Python 3.14.6、NumPy 2.4.6 |
| M0 | 参照実装の積分の検査（`--integrals`） | `integral checks passed`。18 項目が全て `ok`。s 同士は 1.8e-15 以下、p を含む組は 6.3e-8 以下、動径のポアソンの公式は 7.8e-8、三中心は 2.7e-10 と 5.2e-10 | 2026-11-23 | 同上。SciPy 1.17.1 |
| M0 | `--dump` の出力と付属のフィクスチャーの比較 | バイト単位で一致（$K$ のブロック 84 個） | 2026-11-23 | 同上 |
| M0 | CPU 版のビルド（GCC、OPENMPI、OPENBLAS） | 成功。変更の前後とも | 2026-11-23 | 下の「ビルドと実行」のコンテナ |
| M0 | 指示書の `ctest`（正規表現で選んだ 18 件） | 変更の前後とも 18 件が全て合格 | 2026-11-23 | 同上 |
| M0 | `scf.XcType HSE` の入力（`Methane.dat` の 1 行だけを変えたもの） | `scf.XcType=HSE is not implemented.` を表示して終了。出力ファイルは作られない。`NLX` も同じ | 2026-11-23 | 同上。1 ランク |
| M0 | `work/input_example/Methane.dat` の全エネルギー | 変更の前後とも `Utot = -8.033540953477` Hartree。SCF は 14 回。`.out` と標準出力の差は日付と時間の行だけで、ほかの出力（`.ene`、cube、`.xyz`、再開ファイルなど）はバイト単位で一致 | 2026-11-23 | 同上。1 ランク、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1` |
| M0 | 大域状態のゲート（`tools/audit_global_usage.py --check-no-new`） | `scf.XcType` の拒否（`71f7561c`）の前後で報告が同じで、増えた組はない。ベースラインの更新（`bab89d8e`）の後は合格。286 単位を全て解析 | 2026-11-23 | ホスト。clang 21.1.8、NVHPC 26.9 |
| M0 | 試作用のハイブリッド入力の削除（`bcf52225`）の後の再確認 | `Methane.dat` は `Utot = -8.033540953477` Hartree のままで、出力の差は日付と時間の行だけ。`scf.XcType HSE` は同じメッセージで終了。`ctest` の 18 件とゲートは合格 | 2026-11-23 | コンテナ（`ctest` と実行）、ホスト（ゲート） |

M1 の結果です。「書き直しの後」は、短距離部分を Quantum ESPRESSO 7.6 の形に書き直した後のコードでの結果、「書き直す前」は、その前のコード（`5215d212`）での結果です。

| マイルストーン | 試験 | 結果 | 日付 | 条件（計算機、ビルド） |
|---|---|---|---|---|
| M1（書き直しの後） | 手順 1：参照実装の自己検査（`tools/wpbeh_reference.py`） | `self-test passed`（15 項目。値は指示書の 10.4 節の表と同じ）。`--dump` で作り直したフィクスチャーは、保存したファイルとバイト単位で一致 | 2026-11-23 | クラウドのコンテナ（Ubuntu 24.04.5）の仮想環境。Python 3.12.3、mpmath 1.4.1 |
| M1（書き直しの後） | 手順 2：Quantum ESPRESSO 7.6 のルーチンとの照合（`--compare-qe`） | `comparison passed`。フィクスチャーの 320 チャンネルの全てが許容差の中にあり、許された差に対する比の最大は 0.043。ルーチンのエラーの返り値は全て 0。区分ごとの差は指示書の M1 の表 | 2026-11-23 | 同上。gfortran 13.3.0、`-O1`、倍精度。NumPy 2.5.5 |
| M1（書き直しの後） | 参照実装の倍精度の試作（`--double`） | 160 点の全てが許容差の中にあり、許された差に対する比の最大は 0.12 | 2026-11-23 | 同上。NumPy 2.5.5、SciPy 1.18.1 |
| M1（書き直しの後） | `xc-wpbeh-special` | 合格。1e-30 から 1e8 の 781 点で、相対誤差の最大は $e^{x}E_1(x)$ が 1.5e-15、`erfcx` が 4.3e-16（要求は 1e-14）、導関数が 1.8e-15 と 2.6e-15。特殊関数のコードは書き直しで変えていない | 2026-11-23 | クラウドのコンテナ。GCC 13.3.0（Release）と Clang 18.1.3（サニタイザー付き） |
| M1（書き直しの後） | `xc-wpbeh-fixture` | 合格。160 点で、許された差に対する比の最大は 0.075。区分ごとの差は下の表 | 2026-11-23 | 同上 |
| M1（書き直しの後） | `xc-hybrid` | 合格（比較は 14,112 個）。混合比 0 の出力は PBE の関数とビット単位で同じ。混合比 0.25、1、−0.5 の出力は「PBE − 混合比 × 短距離部分」と同じ | 2026-11-23 | 同上 |
| M1（書き直しの後） | 既存の xc の試験（`xc-legacy` など 6 件） | 合格。上の 3 件と合わせて、xc の 9 件が全て合格 | 2026-11-23 | 同上 |
| M1（書き直しの後） | `libxc` の語の検索（`grep -rniw libxc modules source tools`） | 該当なし | 2026-11-23 | クラウドのコンテナ |
| M1（書き直す前） | ハイブリッド汎関数を指定しない計算（`Set_XC_Grid` の変更の前後） | 6 つの入力（`Methane`、`Mol_MnO`、`Crys-MnO`、`GaAs`、`Graphite4`、`H2O`）で全エネルギーが一致。出力の差は日付と時間の行だけ。`Methane.dat` は `Utot = -8.033540953477` Hartree のまま | 2026-11-23 | コンテナ。1 ランク、1 スレッド |
| M1（書き直す前） | 指示書の `ctest` の 18 件と xc の 9 件 | 27 件が全て合格 | 2026-11-23 | コンテナ |
| M1（書き直す前） | 大域状態のゲート | 合格。報告は変更の前と同じ | 2026-11-23 | ホスト |
| M1（書き直す前） | GPU 版のコンパイル検査（`tools/check_gpu_compilation.py`） | CUDA（100 単位）と HIP（101 単位）が合格。一覧にない `Set_XC_Grid.c`、`OutData.c`、`OutData_Binary.c` も、同じ指定でコンパイルできた | 2026-11-23 | CUDA はホストの NVHPC 26.9。HIP は `openmx-hip-compile:7.2` のコンテナ（ROCm 7.2） |
| M1（書き直す前、参考） | 混合比 0.25 と ω 0.11 を呼び出し側で直接渡した一時的なバイナリー（コミットしていない） | `Methane` と `Mol_MnO` で SCF が収束。1 スレッドと 4 スレッドの全エネルギーの差は 5e-11 Hartree 以下。`Methane` の H 原子の力は、解析値と全エネルギーの数値微分の差が 1.0e-5 Hartree/bohr（PBE では 1.1e-5） | 2026-11-23 | コンテナ。1 ランク |

「書き直す前」の 5 行の試験は、xc の 9 件を除いて、書き直しの後には実行していません。書き直しで変えたのは、`modules/xc` の短距離部分の計算核、その試験とフィクスチャー、参照実装、ヘッダーと README の説明、文書です。関数の引数と `source/` の下のファイルは変えていません。混合比が 0 のときは計算核を呼ばないので（`xc-hybrid` が確かめています）、ハイブリッド汎関数を指定しない計算の結果は変わりません。大域状態のゲートが調べる `source/` のファイルも同じままです。本体のビルド、指示書の `ctest`、GPU 版のコンパイル検査は、プッシュのたびに GitHub Actions が実行します（`.github/workflows/core.yml`、`gpu-compile.yml`）。混合比 0.25 での SCF と力（参考の行）は、新しい定義では確かめていません。入力からハイブリッド汎関数を選べるようになる M4 で確かめます。

`xc-wpbeh-fixture` での、実装と参照値の相対差の最大です。`dexdgd` はフィクスチャーの尺度での値、エネルギーは 2 つのチャンネルの大きい方の `tol` で分けた値です。

| `tol` の区分 | $w$ | チャンネル数 | `dexdd` | `dexdgd` | エネルギー |
|---|---|---|---|---|---|
| 1.0e-12 | 0.017 から 0.36 | 160 | 2.9e-15 | 4.2e-15 | 9.3e-16 |
| 1.4e-12 | 0.77 | 32 | 1.0e-14 | 8.3e-15 | 2.9e-15 |
| 3.0e-11 | 1.7 | 32 | 3.3e-13 | 1.7e-13 | 2.1e-14 |
| 2.2e-8 | 3.6 | 32 | 8.5e-11 | 7.7e-11 | 1.9e-11 |
| 9.9e-6 | 8.0 | 32 | 5.2e-8 | 1.8e-7 | 3.6e-8 |
| 1.0e-9（ $w>14$ ） | 16.5 | 32 | 2.5e-12 | 7.5e-10 | 5.5e-13 |

Quantum ESPRESSO のルーチンと参照値の差は、指示書の M1 の「Quantum ESPRESSO との照合」の表にあります。 $w$ が 0.36 以下の 160 チャンネルでは、エネルギーの相対差が 7.1e-13 以下でした。 $w$ が 4 から 14 の領域では、差が Fortran の倍精度の丸め誤差に埋もれます（ $w=8.0$ のエネルギーで 1.5e-6）。ルーチンを 4 倍精度でコンパイルした照合は、していません。

M0 と、書き直す前の M1 の試験の入力、標準出力、ログは、ホストの `build/exx-m0/` と `build/exx-m1/` にあります（git の管理外）。書き直しの作業はクラウドのコンテナで行いました。そのログは残していませんが、下の「ビルドと実行」の手順で同じ結果を作り直せます。

## ビルドと実行

ホストには CI と同じ apt パッケージがありません。足りないのは `libomp-dev`、`openmpi-bin`、`libopenmpi-dev`、`libscalapack-openmpi-dev`、`libfftw3-dev`、`libtool-bin` で、`sudo` にはパスワードが要ります。CPU 版は、同じパッケージを入れた Docker イメージ `openmx-cpu-toolchains:ubuntu24.04` の中でビルドして実行します。イメージの中身は Ubuntu 24.04.5、GCC 13.3.0、Open MPI 4.1.6、OpenBLAS 0.3.26、ScaLAPACK 2.2.1、FFTW 3.3.10、CMake 3.28.3 です。イメージがなくなった場合は、Ubuntu 24.04 に `.github/workflows/core.yml` の `cpu-application` と同じパッケージを入れて作り直します。

```sh
# リポジトリーのルートで実行する。コンテナにはリポジトリーを同じパスで見せる
D="docker run --rm --user $(id -u):$(id -g) -e HOME=/home/ubuntu -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -v $PWD:$PWD"
IMG=openmx-cpu-toolchains:ubuntu24.04
$D -w $PWD $IMG cmake -S . -B build/exx-cpu -DOPENMX_GPU_BACKEND=CPU -DOPENMX_CPU_COMPILER=GCC \
  -DOPENMX_CPU_MPI=OPENMPI -DOPENMX_CPU_LINALG=OPENBLAS -DOPENMX_COPY_TO_WORK=OFF -DOPENMX_BUILD_JOBS=8
$D -w $PWD $IMG cmake --build build/exx-cpu --parallel 12
$D -w $PWD $IMG ctest --test-dir build/exx-cpu --output-on-failure \
  -R '^(core-(grid-assembly|host-simd-boundary)|build-backend-isolation|cpu-build-accelerator-boundary|cpu-numerical-toolchain|environment-registry)'
# 実行例。build/exx-m0/DFT_DATA19 は ../../DFT_DATA19 へのシンボリックリンク
$D -w $PWD/build/exx-m0/run_after $IMG mpirun -np 1 ../../exx-cpu/openmx Methane.dat
```

初回のビルドは 12 並列で約 5 分です。

大域状態のゲートはホストで実行します。この計算機の `source/third_party/gpusolver2-install` には OpenMP 版 ELPA のヘッダーがないので、CMake の CUDA ステージのスタックを指定します。指定しないと `elpa_cosma_bridge.c` と `openmx_elpa_cpu.c` が解析されません。

```sh
python3 tools/audit_global_usage.py --output build/exx-m0/globals-usage.json \
  --baseline docs/global-usage-baseline.json --check-no-new \
  --make-arg GPUSOLVER2_PREFIX=$PWD/build/cuda/native/source/third_party/gpusolver2-install
```

M1 で使った環境と手順です。仮想環境は `build/exx-m1/venv` に作ります（git の管理外）。ホストの `build/exx-m1/venv` は、書き直す前に PySCF（libxc 7.1.2 を含みます）を入れて作ったものです。使う前に、下の 3 行で作り直します。

```sh
# mpmath（参照値）と、NumPy、SciPy（--double と --compare-qe）。PySCF と libxc は入れない
rm -rf build/exx-m1/venv
uv venv build/exx-m1/venv --python 3.12
uv pip install --python build/exx-m1/venv/bin/python mpmath numpy scipy
PY=build/exx-m1/venv/bin/python
$PY tools/wpbeh_reference.py                      # self-test passed
$PY tools/wpbeh_reference.py --double             # 倍精度の試作とフィクスチャーの比較
$PY tools/wpbeh_special_reference.py --check modules/xc/tests/fixtures/wpbeh_special_reference.txt
# Quantum ESPRESSO 7.6 のルーチンとの照合。QE は、タグ qe-7.6 のソースを置いたディレクトリー
$PY tools/wpbeh_reference.py --qe-files build/exx-m1/qe-check
gfortran -O1 -J build/exx-m1/qe-check -o build/exx-m1/qe-check/wpbeh_qe_driver build/exx-m1/qe-check/kind_l.f90 \
  $QE/XClib/qe_funct_exch_gga.f90 build/exx-m1/qe-check/wpbeh_qe_driver.f90
build/exx-m1/qe-check/wpbeh_qe_driver < build/exx-m1/qe-check/wpbeh_qe_input.txt > build/exx-m1/qe-check/wpbeh_qe_output.txt
$PY tools/wpbeh_reference.py --compare-qe build/exx-m1/qe-check/wpbeh_qe_output.txt   # comparison passed
# モジュールだけのビルドと試験（ホスト）。Clang とサニタイザーは CI の 2 番目の構成と同じ
cmake -S modules/xc -B build/exx-xc -DCMAKE_BUILD_TYPE=Release
cmake --build build/exx-xc --parallel 8 && ctest --test-dir build/exx-xc --output-on-failure
cmake -S modules/xc -B build/exx-xc-clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DOPENMX_XC_SANITIZERS=ON
cmake --build build/exx-xc-clang --parallel 8 && ctest --test-dir build/exx-xc-clang --output-on-failure
# GPU 版のコンパイル検査。CUDA はホスト、HIP はコンテナ
N=/opt/nvidia/hpc_sdk/Linux_x86_64/26.9
python3 tools/check_gpu_compilation.py --backend CUDA --sdk $N/cuda/13.3 --cc $N/compilers/bin/nvc \
  --cxx $N/cuda/13.3/bin/nvcc --include $N/math_libs/13.3/targets/x86_64-linux/include \
  --arch 80 --build build/exx-m1/cuda-compile --jobs 8
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:$PWD -w $PWD openmx-hip-compile:7.2 \
  python3 tools/check_gpu_compilation.py --backend HIP --sdk /opt/rocm --cc /opt/rocm/llvm/bin/clang \
  --cxx /opt/rocm/llvm/bin/clang++ --arch gfx942 --build build/exx-m1/hip-compile --jobs 8
```

Quantum ESPRESSO のソースは、リポジトリーに入れません。照合に要るファイルは `XClib/qe_funct_exch_gga.f90` の 1 つだけです（指示書の文献 [16]）。2026-11-23 の照合に使ったファイルは 74,253 バイトで、SHA-256 は `9ab1a693661ac5c4697f79cd63f3c10529ce926c1b8cf733019429eaae2a9a66` でした。

クラウドのコンテナ（Ubuntu 24.04.5、GCC 13.3.0、Clang 18.1.3、CMake 3.28.3）で同じ手順を実行するときは、`gfortran` と `libclang-rt-18-dev` を apt で入れます。後者がないと、サニタイザーを付けた Clang の構成がリンクで失敗します。書き直しの後の結果は、この環境で `modules/xc` だけをビルドして得たものです。

変更の前後を比べるときは、前のコミットを `git archive` で `build/` の下に書き出し、そこで CPU 版をビルドします。書き出した先には `DFT_DATA19` へのシンボリックリンクが要ります（ないと構成が失敗します）。

## 決めたこと

- （2026-11-23）CPU 版は上のコンテナの中でビルドして実行する。指示書の `sudo apt-get install` は実行していない。利用者の了承済み。
- （2026-11-23）ビルドディレクトリーは `build/exx-cpu` にした。指示書の `build/cpu` には 2026-10-28 の別の構成（MKL）のビルドが残っていて、構成を変えた再利用を CMake が拒否する。
- （2026-11-23）`-DOPENMX_COPY_TO_WORK=OFF` を付ける（CI と同じ指定）。この作業ツリーの `work/openmx` は GPU 版の実行ファイルなので、CPU 版で上書きしない。実行には `build/exx-cpu/openmx` を直接使う。
- （2026-11-23）試験は `build/exx-m0/` の中で実行し、`work/` に出力を作らない。
- （2026-11-23）`HSE` と `NLX` の拒否は、`scf.XcType` を読んだ直後に置いた。書き方は既存の入力エラーと同じ（ホストのランクが表示、`MPI_Finalize()`、`exit(0)`）なので、終了コードは 0 になる。
- （2026-11-23）指示書、参照実装、フィクスチャーは、指示書の冒頭の表の場所に置いた。フィクスチャーの置き場所 `modules/exx/tests/fixtures/` には、まだこのファイルしかない（モジュール本体は M2 で作る）。
- （2026-11-23）`.gitignore` の `*.md`（MD の出力用の規則）が Markdown の文書にも当たるので、新しい文書は `git add -f` で追加した。既存の `docs/*.md` と同じ扱い。
- （2026-11-23、利用者の指示）試作用のハイブリッド入力（`scf.xmin.Rec.Coulomb` など 4 つのキーワードと `Allocate_Arrays(15)` の呼び出し）を `Input_std.c` から削除した（`bcf52225`）。
- （2026-11-23、利用者の指示）大域状態のベースラインを更新した（`bab89d8e`）。ベースラインになかった 16 組は `main` の変更によるもので、内訳は `docs/GLOBAL_STATE_AUDIT.md` の末尾に書いた。試作用入力の削除で減った 4 組も `bcf52225` で反映した。以後は `--check-no-new` の合否で確認する。
- （2026-11-23）ベースラインの更新と入力の削除は別のコミットにした。`bab89d8e` は EXX の変更に依存しないので、`main` に単独で取り込める。

- （2026-11-23、M1。同じ日の書き直しで取り消した）短距離部分の定義は、指示書のとおり libxc 7.1.2 の `GGA_X_WPBEH` にした。Quantum ESPRESSO の形には変えていない。
- （2026-11-23、M1、利用者の指示。表示と出典は同じ日の書き直しで改めた）短距離部分の計算核は `modules/xc/src/wpbeh_kernel.cpp` に置き、MPL 2.0 の表示と libxc の出典を付けた。密度の下限や出力の規約を付ける部分（`functionals.cpp`）、特殊関数、ハイブリッド用の関数は、モジュールのライセンスのままである。
- （2026-11-23、M1）公開した関数は、C++ の `wpbeh_short_range_exchange`、`pbe_screened_hybrid`、`e1_scaled`、`erfcx` と、C の `openmx_xc_wpbeh_short_range`、`openmx_xc_pbe_screened_hybrid` である。PBE0 用の関数は作っていない（M8）。
- （2026-11-23、M1）導関数は、値と 2 つの偏微分（ $\rho_s$ と $\lvert\nabla\rho_s\rvert$ ）を一緒に運ぶ演算で求めた。手で微分した式は書いていない。
- （2026-11-23、M1）特殊関数は、小さい引数では級数（ $e^{x}E_1(x)$ 、 $x<1$ ）と `exp(x*x)*erfc(x)`（`erfcx`、 $x<2$ ）、それ以上では深さを固定した連分数で計算する。導関数も同じ連分数から取る。 $x$ が大きいと $e^{x}E_1(x)-1/x$ が桁落ちするためである。
- （2026-11-23、M1）特殊関数の参照値を作るスクリプト `tools/wpbeh_special_reference.py` と、そのフィクスチャー `modules/xc/tests/fixtures/wpbeh_special_reference.txt` を加えた。指示書にないファイルである。試験から mpmath を呼べないので、値を保存した。
- （2026-11-23、M1）密度の下限（スピンごとに 0.5e-14）、勾配の大きさの下限（0.5e-14）、勾配が 0 の点、対称化（`symmetrize_spins`）の扱いは、PBE の交換と同じにした。 $s$ の下限は、書き直しで 1e-15（libxc と同じ値）から 1e-10 に変えた（下の「M1 の書き直し」の項）。
- （2026-11-23、M1）混合比がちょうど 0 のとき、ハイブリッド用の関数は短距離部分を計算せず、PBE の出力をそのまま返す。
- （2026-11-23、M1）`Set_XC_Grid` の新しい引数は、`XC_switch` の直後の `exx_alpha` と `exx_omega` である。7 か所の呼び出しは、どちらにも 0 を渡している。M4 で EXX の状態の値に置き換える。`exx_alpha` が 0 でないときは GPU の経路に入らず、ホストのループを使う。
- （2026-11-23、M1）PBE の関数を直接呼ぶ箇所は、`Set_XC_Grid.c` の中だけである（ホストのループと、GPU の経路の `SXG_XC_PBE`）。`Force.c` などは `Set_XC_Grid` の結果を使う。
- （2026-11-23、M1）HIP 版の `Set_XC_Grid.c` は、新しい引数を受け取って無視する。HIP 版の複製を変えたときは、`source/backends/hip/sources.json` の `import_sha256` を更新する。更新しないと `build-backend-isolation` の試験が落ちる。
- （2026-11-23、M1）差分商との比較は、丸め誤差の小さい領域に限った。 $w$ が 4 から 14 の領域と、 $w>14$ での勾配についての導関数は、フィクスチャーとの比較で確かめている。
- （2026-11-23、M1）`openmx_common.h`、`OutData.c`、`OutData_Binary.c` は改行が CRLF のファイルである。バイト単位の置換で編集し、改行を保った。

- （2026-11-23、M1 の書き直し、利用者の指示）libxc を一切使わない。短距離部分の定義、定数、照合から、libxc に由来するものを外した。指示書は改訂 3 にした（1.2 節に規則を加えた）。
- （2026-11-23、M1 の書き直し、利用者の指示）短距離部分の定義は、Quantum ESPRESSO 7.6 の `pbexsr` と同じ形にした。論文の丸めた定数、 $s\le0.08$ での多項式、還元勾配が 8.3 を超えたときの付け替え、 $w>14$ でのガウス関数を使う。この定義では、式が切り替わる 3 か所で増強因子が不連続になる（指示書の M1 の表）。
- （2026-11-23、M1 の書き直し）Quantum ESPRESSO のルーチンとは 3 点を変えた。導関数には付け替えの微分を入れた。特殊関数は、全ての引数で倍精度の精度まで計算する。付け替え前の $s$ には下限 1e-10 を入れた。この下限は自前の保護で、Quantum ESPRESSO にも論文にもない値である。
- （2026-11-23、M1 の書き直し）計算核と参照実装は、Quantum ESPRESSO 7.6 のソースの式と定数を書き写した新しい実装である。冒頭に出典を書き、MPL 2.0 の表示を外して、リポジトリーのライセンス（GPLv3）で配布する。Quantum ESPRESSO のファイルの表示は GNU General Public License である。ライセンスの扱いを変えるときは、利用者が決める。
- （2026-11-23、M1 の書き直し）照合の相手は、Quantum ESPRESSO のルーチンを gfortran でコンパイルした出力にした。参照実装は駆動プログラムと入力を書き出し（`--qe-files`）、出力を比べる（`--compare-qe`）。コンパイルと実行は、スクリプトの外で行う。PySCF を通した照合（`--compare-pyscf`）は削除した。
- （2026-11-23、M1 の書き直し）フィクスチャーの還元勾配の 16 個を取り直した（0.08 と 8.3 の両側、15、40 を含む）。点の数は 160 のままで、形式の版を 2 にした。
- （2026-11-23、M1 の書き直し）`xc-wpbeh-fixture` を定義に合わせて変えた。フィクスチャーが式の切り替えの両側を含むことを、チャンネルの数（ $s\le0.08$ が 60、還元勾配が 8.3 を超えるものが 80、 $w>14$ が 32）で確かめる。差分商との比較の点は、切り替えの両側に取り直した。 $\omega=0$ の一様電子ガスの極限では、増強因子と 1 の差が 4.7e-9 から 4.9e-9 の間にあることを確かめる。「還元勾配が 15 を超えると、勾配についての導関数が 0 になる」という以前の検査は、新しい定義では成り立たないので外した。
- （2026-11-23、利用者の指示）コミットとプッシュは `feature/exx` に直接行う。別のブランチや PR は作らない。

## 未解決の問題

- （2026-11-23）旧い厳密交換のコードが残っている。試作用入力が書いていた 4 個の変数の定義（`xmin_Rec_Coulomb`、`xmax_Rec_Coulomb`、`Ng1_Rec_Coulomb`、`Nrank_Rec_Coulomb`）、`Allocate_Arrays()` の `case 15`、`Free_Arrays.c` の 4 つの解放、ビルドに入っていない `Expand_ReCoulomb_Kernel.c`、到達しない `Band_DFT_Col_NEGF.c:743` の `if (5==XC_switch)`、`source/exx*` である。どれも動作には影響しない。削除するときは利用者に確認する（指示書の 1.3 節）。
- （2026-11-23、M1）短距離部分を含む計算は、PBE の関数の約 3.6 倍の時間がかかる。この計算機の 1 スレッドで、1 点あたり 1.5 マイクロ秒と 0.42 マイクロ秒だった（密度を 1e-8 から 1 まで対数で一様に取った点での値）。混合比 0 では PBE と同じ時間である。GPU への対応は M6 以降になる。この時間は書き直す前の計算核での値で、書き直しの後は測っていない。
- （2026-11-23、M1）`work/input_example/H2O.dat`（構造最適化）の再開ファイル `H2O.rst4gopt` は、同じ実行ファイルで 2 回計算しても 10 バイトが違う。M1 の変更とは関係がない。出力を比べるときは、このファイルを除く。
- （2026-11-23、M1）ハイブリッド汎関数を入力から選ぶ手段は、まだない（M4）。本体の中でホストのループが混合比 0.25 で動くことは、呼び出し側を書き換えた一時的なバイナリーで確かめただけである。この確認は書き直す前の定義で行った。
- （2026-11-23、M1 の書き直し）Quantum ESPRESSO のルーチンとの照合は倍精度で行った。 $w$ が 4 から 14 の領域では、差が Fortran の丸め誤差（エネルギーで 1.5e-6 まで）に埋もれる。この領域で定数が一致することは、 $w$ の小さい点で 1e-12 まで合うことと、ソースとの行ごとの照合に依っている。ルーチンを 4 倍精度でコンパイルした照合はしていない（指示書の 14 節）。
- （2026-11-23、M1 の書き直し）新しい定義の増強因子は、 $w=14$ で相対 2.6e-4 まで跳ぶ。 $\omega=0.11$ では、スピン分極のない全密度 1.6e-8 bohr⁻³ の点に当たる。SCF の収束と力への影響は確かめていない（M4、M7）。
- （2026-11-23、M1 の書き直し）交換孔の模型の出典（指示書の文献 [9]）は読んでいない。Heyd の学位論文（文献 [11]）の付録 A のコードも、全文は確認していない。定数は Quantum ESPRESSO 7.6 のソースから取った。
- （2026-11-23、M1 の書き直し）書き直しの後、本体の CPU 版のビルド、指示書の `ctest` の 18 件、ハイブリッド汎関数を指定しない計算の全エネルギー、大域状態のゲート、GPU 版のコンパイル検査を、手元では実行していない。理由は「受け入れ試験の結果」の表の下に書いた。

## 利用者への確認待ち

- なし。2026-11-23 の 3 件（試作用入力の削除、ベースラインの更新、コンテナでのビルド）は回答を受けて対応した（「決めたこと」）。同じ日の「短距離部分を libxc の形のまま進めるか」の確認は、書き直すという回答を受けて対応した。
