# EXX 実装の進捗

## 現在の状態

- 完了したマイルストーン: M0, M1, M2
- 作業中: なし（次は M3）
- 基準コミットからのブランチ: `feature/exx`（基準 `4e0ee247438a3c9358e95adee00a6ff60a4c618b`）。リモートは `origin/feature/exx` で、2026-11-23 に最初のプッシュをした。コミットとプッシュは、別のブランチや PR を作らずに `feature/exx` へ直接行う（2026-11-23、利用者の指示）。強制プッシュと履歴の書き換えはしない。
- M1 の短距離部分は、2026-11-23 に定義を書き直した（利用者の指示）。libxc 7.1.2 の形をやめ、Quantum ESPRESSO 7.6 と同じ形にしている。コミットは `615ad67f`（指示書の改訂 3）と `e8c8bf6a`（参照実装、フィクスチャー、計算核、試験）である。下の表で「M1（書き直しの後）」とした行が、今のコードでの結果である。書き直す前の記録は、このファイルの `e66e3cff` の版にある。
- M2（`modules/exx` の前半）は 2026-11-23 に終えた。コミットは `6064704`（モジュールとカーネル）、`05da230`（CI のジョブ `exx`）、`b56ba6c`（動径の変換と二中心積分）、`2d567b5`（重なり行列との照合）、`60d01d7`（ABF）、`7abfd0d`（三中心積分と LRI の係数）、`e1c9391`（site、近傍リスト、T ブロックと V ブロック）、`d4c0b44`（例題 `lri_example`）、`d518d44`（`exx-lri-identity` に離れた 2 原子の場合を追加）である。`source/` と `core/` は変えていない。指示書は改訂 4 にした（M2 の実測値を 12 節から 14 節に書いた）。

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
| M1（書き直しの後） | CI の `host` ジョブの試験 `total-energy-workspace` と `stress-grid` | 修正（`19e874b9`）の前は 2 件ともコンパイルエラーで失敗し、後は 2 件とも合格。失敗は `Set_XC_Grid` の引数の追加（`5215d212`）から続いていた（GitHub Actions の実行 `45110318332`） | 2026-11-23 | クラウドのコンテナ。`host` ジョブと同じ構成（`OPENMX_BUILD_APPLICATION=OFF`、OpenMP、Debug、GCC 13.3.0）。実行したのはこの 2 件だけ |
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

M2 の結果です。値は、クラウドのコンテナ（Ubuntu 24.04.5）で GCC 13.3.0（Release）を使ってビルドしたものです。Clang 18.1.3 にサニタイザーを付けた構成でも、`modules/exx` の 11 件が全て合格しました。

| マイルストーン | 試験 | 結果 | 日付 | 条件（計算機、ビルド） |
|---|---|---|---|---|
| M2 | `exx-kernel-analytic` | 合格。4 つのカーネルの $\tilde v(\kappa)$ と $\kappa^2\tilde v(\kappa)$ で、相対差の最大は 6.9e-15 と 7.0e-15（要求は 1e-13）。 $\kappa\to0$ の値は 14 個が全て一致 | 2026-11-23 | クラウドのコンテナ。GCC 13.3.0 |
| M2 | `exx-two-center-gaussian` | 合格。3 つのカーネルの s 同士は 8.4e-15（要求は 1e-10）、p を含む組は 2.7e-15（要求は 1e-6）。重なりは 2.2e-16 以下、球ベッセル変換は 6.1e-16 | 2026-11-23 | 同上 |
| M2 | 重なり行列の照合（`exx-two-center-overlap`） | 合格。45 ブロックの 6,741 要素で、本体と同じ離散化では差の最大が 6.2e-11（要求は 1e-8）。変換を積分点で直接求める方法（モジュールが自分の関数に使う方法）では 8.2e-8 | 2026-11-23 | 同上。フィクスチャーは CPU 版の本体の出力 |
| M2 | `exx-two-center-symmetry` | 合格。 $V_{ab}(\mathbf R)=V_{ba}(-\mathbf R)^{\mathsf T}$ は 7.1e-15、座標系の回転は 1.4e-12（要求は 1e-10）。実数の Gaunt 係数と数値積分の差は 2.0e-14 | 2026-11-23 | 同上 |
| M2 | `exx-three-center-gaussian` | 合格。ABF が 1 つ目の原子、2 つ目の原子、PAO と同じ原子にある場合の差の最大は、6.5e-11、2.4e-11、4.1e-11（要求は 1e-8）。動径のポテンシャルと解析値の差は、刻み 0.0025 bohr で 1.3e-10 | 2026-11-23 | 同上 |
| M2 | `exx-abf-pca` | 合格。正規直交性は 4.4e-15（要求は 1e-12）。射影の残差と捨てた固有値の和の相対差は 2.9e-15（要求は 1e-8）。軌道の順序と符号を変えても、固有値は 1.2e-15、関数は 4.0e-15 まで同じ | 2026-11-23 | 同上 |
| M2 | `exx-lri-identity` | 合格。1 原子では、ABF の張る空間にある積の係数が 3.4e-14 まで再現された（要求は 1e-10）。離れた 2 原子（Si–C の 9.2 bohr、C–H の 8.0 bohr）では、 $b=Gc$ から求めた係数が成分ごとに 5.0e-14 まで再現された。結合距離の 2 原子では、密度の差がクーロンノルムで 4.3e-13、係数の成分ごとの差が最大 1.3e-10 で、試験は後者を 1e-8 で判定している（表の下の説明） | 2026-11-23 | 同上 |
| M2 | `exx-lri-convergence`（記録） | 合格。和が減らないことと、最後の増分が最初の増分以下であることを確かめる。値は下の表 | 2026-11-23 | 同上 |
| M2 | `exx-system-neighbours`、`exx-system-blocks`（指示書にない試験） | 合格。近傍リストは総当たりの探索と一致。T ブロックは対の解とビット単位で同じで、逆から見た解との差は 2.1e-14。V ブロックは転置とビット単位で同じ。結晶の記述を変えたときの差は、T が 2.8e-13、V が 1.2e-14 | 2026-11-23 | 同上 |
| M2 | `exx-package-consumer` | 合格。インストールしたパッケージから、例題 `kernel_example` と `lri_example` をビルドして実行できた | 2026-11-23 | 同上 |
| M2 | `libxc` の語の検索（`grep -rniw libxc modules source tools`） | 該当なし | 2026-11-23 | クラウドのコンテナ |
| M2 | GitHub Actions（`2d567b5` までの 4 コミット） | C++ core（実行 `45127585889`）の 19 ジョブと、Native GPU compilation が全て成功。`exx` の 3 ジョブ（Ubuntu の g++、Ubuntu の clang++ とサニタイザー、macOS の clang++）を含む | 2026-11-23 | GitHub のランナー |
| M2 | GitHub Actions（`d4c0b44` までの 4 コミット） | C++ core（実行 `45134699816`）の 19 ジョブと、Native GPU compilation（実行 `45134699710`）が全て成功。`exx` の 3 ジョブは、`modules/exx` の 11 件を実行している | 2026-11-23 | GitHub のランナー |

`exx-lri-identity` の 2 原子の場合について。指示書の要求は「元の係数が 1e-10 で再現されること」です。1 原子では、PAO の積から係数を求める経路の全体で、この要求を満たしました。2 原子の PAO の積は ABF の張る空間に入らないので、2 原子の場合は係数 $c$ から $b=Gc$ を作って解き直す形で確かめています。離れた 2 原子（関数がほとんど重ならない距離）では、係数が成分ごとに 5.0e-14 まで戻り、要求を満たしました（`d518d44` で加えた場合）。結合距離の 2 原子では、式 (7) の行列の条件数が、対角成分を 1 に揃えた後も 6e6 に達します（C–C、2.3 bohr）。 $b$ の丸め誤差だけで係数が 1e-10 程度動くので、成分ごとの差は 1e-10 に届きません。試験は、この場合の展開した密度の差をクーロンノルムで測って 1e-10 で判定し、成分ごとの差は 1e-8 で判定しています。結合距離の場合のこの読み方でよいかは、利用者への確認待ちに入れました。

元素ごとの ABF の個数です（動径関数の数 / 成分を数えた数）。主成分分析のしきい値の既定値は 1e-4 です。`lri_example` の出力で、PAO は DFT_DATA19 のものです。

| 元素と基底 | 1e-2 | 1e-3 | 1e-4（既定） | 1e-5 | 1e-6 | 間引く前 |
|---|---|---|---|---|---|---|
| H6.0-s2p1 | 6 / 14 | 7 / 15 | 7 / 15 | 7 / 15 | 7 / 15 | 7 / 15 |
| C6.0-s2p2d1 | 14 / 54 | 15 / 59 | 19 / 69 | 20 / 72 | 22 / 78 | 22 / 78 |
| N6.0-s2p2d1 | 13 / 51 | 14 / 56 | 17 / 65 | 21 / 77 | 22 / 78 | 22 / 78 |
| O6.0-s2p2d1 | 16 / 60 | 18 / 68 | 20 / 72 | 22 / 78 | 22 / 78 | 22 / 78 |
| Si7.0-s2p2d1 | 12 / 50 | 15 / 59 | 19 / 69 | 21 / 77 | 22 / 78 | 22 / 78 |
| Si7.0-s2p2d1f1 | 17 / 99 | 23 / 123 | 28 / 140 | 31 / 155 | 35 / 169 | 35 / 169 |

指示書の 12 節は、s2p2d1 で原子あたり 32 個を仮定していました。実測は 69 個（C、Si）で、約 2 倍です。

1 原子の PAO の積を展開したときの、自己相互作用の和の不足（1 − 展開の値 / 厳密な値）です。厳密な値は、動径積分で求めました。ABF を間引かなければ、不足は丸め誤差（1e-14 以下）になります。

| 元素と基底 | 1e-2 | 1e-3 | 1e-4（既定） | 1e-5 | 1e-6 |
|---|---|---|---|---|---|
| H6.0-s2p1 | 8.7e-4 | 丸め誤差 | 丸め誤差 | 丸め誤差 | 丸め誤差 |
| C6.0-s2p2d1 | 9.2e-4 | 1.1e-4 | 1.4e-6 | 4.9e-7 | 丸め誤差 |
| N6.0-s2p2d1 | 4.4e-4 | 1.7e-4 | 9.6e-6 | 3.9e-7 | 丸め誤差 |
| O6.0-s2p2d1 | 1.4e-3 | 3.4e-5 | 5.1e-6 | 丸め誤差 | 丸め誤差 |
| Si7.0-s2p2d1 | 4.1e-4 | 2.9e-5 | 3.0e-6 | 1.4e-7 | 丸め誤差 |
| Si7.0-s2p2d1f1 | 5.9e-4 | 2.5e-5 | 2.8e-6 | 6.2e-7 | 丸め誤差 |

`exx-lri-convergence` の 2 原子の記録です。値は、2 つの原子の PAO の積の全てについて足した $(\tilde\rho\vert g\vert\tilde\rho)$ （Hartree）で、基底はフィクスチャーのもの（Si7.0-s2p2d1f1、C6.0-s2p2d1、H6.0-s2p1）です。右端は、しきい値を 1e-4 から 1e-6 にしたときの増加を、1e-6 の値で割ったものです。

| 対 | 距離（bohr） | 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 増加 |
|---|---|---|---|---|---|---|---|
| Si–Si | 4.44 | 4.041040 | 4.047083 | 4.047680 | 4.048594 | 4.048904 | 3.0e-4 |
| C–C | 2.90 | 3.142915 | 3.145322 | 3.146568 | 3.146650 | 3.146844 | 8.8e-5 |
| C–H | 2.06 | 2.569948 | 2.571358 | 2.572546 | 2.572606 | 2.572702 | 6.0e-5 |
| H–H | 1.40 | 1.967212 | 1.967827 | 1.967827 | 1.967827 | 1.967827 | 0 |
| Si–C | 3.56 | 3.358701 | 3.361789 | 3.362881 | 3.363351 | 3.363653 | 2.3e-4 |
| Si–H | 2.80 | 2.523725 | 2.526275 | 2.526624 | 2.527024 | 2.527145 | 2.1e-4 |
| Si–Si | 7.26 | 0.888234 | 0.889456 | 0.890052 | 0.890681 | 0.891204 | 1.3e-3 |

和は 7 つの対の全てで減りませんでした。ただし、しきい値を 1e-6 にすると同じ原子の積を全て使い切るので、この表からは 2 原子の積の誤差が分かりません。そこで、ABF を PAO より大きい組の積から作って比べました（`lri_example --generators`）。PAO は C と Si が s2p2d1、H が s2p1 です。「不足」は、最も大きい組（s5p5d4f3、しきい値 1e-6）の値を基準にした相対値です。基準も下からの近似なので、実際の不足はこの値以上です。

| 対（距離） | ABF を作る組（しきい値） | 2 原子の ABF の数 | $(\tilde\rho\vert 1/r\vert\tilde\rho)$ の和 | 不足 | 短距離カーネル（ω = 0.11）での和 | 不足 | 電荷の誤差の最大 |
|---|---|---|---|---|---|---|---|
| C–C（2.90） | s2p2d1（1e-4）。既定の作り方 | 138 | 3.146568 | 2.4e-3 | 2.764275 | 2.9e-3 | 3.4e-4 |
| | s2p2d1（1e-6） | 156 | 3.146844 | 2.3e-3 | 2.764679 | 2.7e-3 | 9.0e-4 |
| | s3p3d2（1e-4） | 320 | 3.150606 | 1.1e-3 | 2.768573 | 1.3e-3 | 1.1e-4 |
| | s2p2d1f1（1e-4） | 268 | 3.152472 | 5.6e-4 | 2.770410 | 6.5e-4 | 3.7e-4 |
| | s3p3d2f1（1e-4） | 458 | 3.153622 | 1.9e-4 | 2.771599 | 2.2e-4 | 1.8e-5 |
| | s4p4d3f2（1e-4） | 714 | 3.154102 | 4.0e-5 | 2.772086 | 4.1e-5 | 2.0e-5 |
| | s5p5d4f3（1e-6） | 1100 | 3.154228 | 基準 | 2.772200 | 基準 | 1.4e-6 |
| Si–Si（4.44） | s2p2d1（1e-4）。既定の作り方 | 138 | 1.745228 | 4.6e-3 | 1.471689 | 5.6e-3 | 6.1e-4 |
| | s2p2d1（1e-6） | 156 | 1.745851 | 4.2e-3 | 1.472484 | 5.0e-3 | 5.7e-4 |
| | s2p2d1f1（1e-4） | 280 | 1.752415 | 4.5e-4 | 1.479154 | 5.2e-4 | 2.6e-4 |
| | s3p3d2f1（1e-4） | 406 | 1.752838 | 2.1e-4 | 1.479541 | 2.6e-4 | 5.7e-5 |
| | s4p4d3f2（1e-4） | 606 | 1.753162 | 2.5e-5 | 1.479877 | 3.5e-5 | 2.7e-5 |
| | s5p5d4f3（1e-6） | 1070 | 1.753206 | 基準 | 1.479928 | 基準 | 1.3e-6 |
| C–H（2.06） | s2p2d1 と s2p1（1e-4）。既定の作り方 | 84 | 2.572546 | 2.0e-3 | 2.205893 | 2.5e-3 | 1.5e-3 |
| | s2p2d1f1 と s2p1d1（1e-4） | 184 | 2.577154 | 1.6e-4 | 2.210547 | 3.6e-4 | 9.5e-4 |
| | s3p3d2f1 と s3p2d1（1e-4） | 311 | 2.577463 | 4.3e-5 | 2.211208 | 5.7e-5 | 8.1e-5 |
| | s5p5d4f3 と s5p5d4f3（1e-6） | 1188 | 2.577575 | 基準 | 2.211333 | 基準 | 9.7e-7 |

この表から分かることです。

- 既定の作り方（PAO の積、しきい値 1e-4）では、2 原子の積の自己相互作用の和が 0.20% から 0.46% 不足します。短距離カーネルでは 0.25% から 0.56% です。
- しきい値を小さくしても、不足はほとんど減りません。同じ原子の PAO の積だけでは、2 原子の積を表しきれないためです。
- 角運動量の 1 つ高い関数を 1 個加えた組（s2p2d1f1、H は s2p1d1）から ABF を作ると、不足は 0.02% から 0.06% になります。ABF の数は、C と Si で約 2 倍、H で 15 個から 50 個です。動径関数を増やすだけの組（s3p3d2）は、ABF の数の割に効きません。
- 2 番目に大きい組（s4p4d3f2）でしきい値を 1e-6 以下にした値と基準との差は、相対 1.4e-5（Si–Si）と 3.3e-5（C–C）です。この行は表にありません。基準そのものの誤差も、この程度と考えられます。
- 式 (7) の解は、どの場合もコレスキー分解で求まりました。枢軸の最小は 1.1e-4 です（C–H、s3p3d2f1 と s3p2d1、しきい値 1e-6）。

格子を変えたときの、既定の作り方（しきい値 1e-4）での $(\tilde\rho\vert 1/r\vert\tilde\rho)$ の和です。既定の設定は 1 行目です。

| 動径の区間の数 | 波数の上限（1/bohr）と区間の数 | C–C（2.90） | Si–Si（4.44） |
|---|---|---|---|
| 900 | 60、900 | 3.146568457724 | 1.745227653548 |
| 450 | 60、900 | 3.146568548957 | 1.745227673824 |
| 1800 | 60、900 | 3.146568436931 | 1.745227641060 |
| 3600 | 60、900 | 3.146568432829 | 1.745227638111 |
| 900 | 60、1800 | 3.146568457724 | 1.745227653548 |
| 900 | 120、1800 | 3.146568457724 | 1.745227653548 |
| 900 | 30、450 | 3.146568457729 | 1.745227653556 |

動径の区間を 900 から 3600 にしたときの変化は、相対 7.9e-9（C–C）と 8.8e-9（Si–Si）です。波数の上限と刻みを変えたときの変化は、相対 5e-12 以下でした。既定の `1DFFT.*` の設定（`Input_std.c:3510-3520` の既定値は 900、3600、900）は、PAO の積に対して足りています。LRI の誤差（1e-3 の桁）より 5 桁小さい値です。

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

CI の `host` ジョブの試験（ラベル `native-state`）は、本体なしの構成で実行します。MPI も BLAS も要りません。`openmx_core` のビルドは、4 並列で約 1 分 20 秒でした。`Set_XC_Grid` の代役を使う 2 件は、次のコマンドで実行できます。

```sh
cmake -S . -B build/exx-core -DOPENMX_BUILD_APPLICATION=OFF -DOPENMX_CORE_BACKEND=OPENMP -DCMAKE_CXX_STANDARD=20 \
  -DOPENMX_CORE_SANITIZERS=OFF -DOPENMX_TEST_REQUIRE_OPENMP=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build/exx-core --target openmx_core --parallel 4
ctest --test-dir build/exx-core -R '^(total-energy-workspace|stress-grid)$' --output-on-failure
```

変更の前後を比べるときは、前のコミットを `git archive` で `build/` の下に書き出し、そこで CPU 版をビルドします。書き出した先には `DFT_DATA19` へのシンボリックリンクが要ります（ないと構成が失敗します）。

M2 の手順です。クラウドのコンテナで実行しました。`modules/exx` のビルドには、コンパイラーと CMake だけが要ります。`modules/basis` を含めた最初のビルドは、GitHub のランナーの g++（2 並列）で 33 秒と 36 秒でした（2 回の実行）。Clang にサニタイザーを付けた構成は、最初のビルドに約 10 分かかります（このコンテナの 4 並列）。GitHub のランナーの 2 並列では、8 分 12 秒と 3 分 58 秒でした。ランナーによって 2 倍の差があります。2 回目からは、変えたファイルだけがコンパイルされます。

```sh
# モジュールだけのビルドと試験。2 つ目の構成は CI の 2 番目と同じ
cmake -S modules/exx -B build/exx -DCMAKE_BUILD_TYPE=Release
cmake --build build/exx --parallel 4 && ctest --test-dir build/exx --output-on-failure
cmake -S modules/exx -B build/exx-clang -DCMAKE_CXX_COMPILER=clang++ -DOPENMX_EXX_SANITIZERS=ON
cmake --build build/exx-clang --parallel 4 && ctest --test-dir build/exx-clang --output-on-failure
# 例題。ABF の個数と LRI の精度の目安を出す
cmake --install build/exx --prefix "$PWD/build/exx-install"
cmake -S modules/exx/examples/lri -B build/exx-lri -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/build/exx-install"
cmake --build build/exx-lri
P=DFT_DATA19/PAO
build/exx-lri/lri_example $P/Si7.0.pao s2p2d1                                   # 1 元素の ABF の個数
build/exx-lri/lri_example $P/C6.0.pao s2p2d1 $P/C6.0.pao s2p2d1 2.90            # 対。既定の作り方
build/exx-lri/lri_example $P/C6.0.pao s2p2d1 $P/C6.0.pao s2p2d1 2.90 --generators s5p5d4f3 --thresholds 1e-4,1e-6
build/exx-lri/lri_example $P/C6.0.pao s2p2d1 $P/H6.0.pao s2p1 2.06 --generators s2p2d1f1,s2p1d1 --thresholds 1e-4
build/exx-lri/lri_example $P/C6.0.pao s2p2d1 $P/C6.0.pao s2p2d1 2.90 --thresholds 1e-4 --radial 3600 --reciprocal 60,900
# カーネルのフィクスチャー（mpmath が要る。M1 の仮想環境を使う）
$PY tools/exx_reference.py --kernels modules/exx/tests/fixtures/kernel_reference.txt
```

重なり行列のフィクスチャーは、CPU 版の本体の出力から作ります。クラウドのコンテナでは、CI の `cpu-application` と同じ apt パッケージを入れてビルドしました（root で動くので `sudo` は要りません）。ビルドは 4 並列で約 6 分でした。入力の `DATA.PATH` は、リポジトリーのルートから 4 階層下で実行する前提の相対パスです。

```sh
apt-get install -y --no-install-recommends gfortran clang libomp-dev openmpi-bin libopenmpi-dev \
  libscalapack-openmpi-dev libopenblas-dev libfftw3-dev autoconf automake libtool libtool-bin m4 pkg-config unzip
cmake -S . -B build/exx-cpu -DOPENMX_GPU_BACKEND=CPU -DOPENMX_CPU_COMPILER=GCC -DOPENMX_CPU_MPI=OPENMPI \
  -DOPENMX_CPU_LINALG=OPENBLAS -DOPENMX_COPY_TO_WORK=OFF -DOPENMX_BUILD_JOBS=4
cmake --build build/exx-cpu --target openmx --parallel 4
mkdir -p build/exx-m2/olp/run && cp modules/exx/tests/fixtures/overlap_reference.dat build/exx-m2/olp/run/sich.dat
(cd build/exx-m2/olp/run && mpirun --allow-run-as-root -np 1 ../../../exx-cpu/openmx sich.dat > sich.std)
python3 tools/exx_overlap_fixture.py --input build/exx-m2/olp/run/sich.dat --scfout build/exx-m2/olp/run/sich.scfout \
  --pao-dir DFT_DATA19/PAO --output modules/exx/tests/fixtures/overlap_reference.txt
```

本体の実行は 1 ランクで約 9 秒です。2026-11-23 に、この手順で作り直したファイルが、保存したフィクスチャーとバイト単位で一致することを確かめました（実行ファイルは、コンテナの一時ディレクトリーにビルドしたものを使いました）。

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
- （2026-11-23）試験用のコード `tests/energy_workspace_harness.c` と `tests/stress_grid_harness.c` にある `Set_XC_Grid` の代役に、`exx_alpha` と `exx_omega` を加えた（`19e874b9`）。代役は、どちらにも 0 が渡ることを確かめる。M4 で呼び出し側の値を変えるときは、代役も変える（指示書の 13 節の 16）。
- （2026-11-23）プッシュの後は、GitHub Actions の結果を確かめる。M1 の受け入れ試験（指示書の `ctest` の 18 件）は `host` ジョブの試験を含まず、上の 2 件の失敗を見落とした。

- （2026-11-23、M2）モジュールの部品は、指示書の 7.1 節のとおり `kernel`、`two_center`、`abf`、`lri`、`system` に分けた。`core/CMakeLists.txt` への組み込み、転送ヘッダー `source/openmx_exx.h`、C の入口は加えていない。本体から呼ぶ形が決まる M3 と M4 で加える。
- （2026-11-23、M2）CI にジョブ `exx` を加えた（`05da230`）。構成は `xc` のジョブと同じ 3 通りである。
- （2026-11-23、M2）モジュールの動径の格子は、0 からカットオフ半径までを `1DFFT.NumGridR` 等分する。逆空間の格子は、0 から `1DFFT.EnergyCutoff` の平方根までを `1DFFT.NumGridR` 等分し、球ベッセル変換は積分点で直接求める。本体の重なり積分は、PAO のメッシュの最初の点からカットオフ半径 + 0.5 bohr までを使い、`1DFFT.NumGridK` 点の表を補間する。
- （2026-11-23、M2）重なり行列の照合は 2 通りで行う。本体と同じ離散化では 1e-8、モジュールの離散化では 1e-6 で判定する。後者の差（8.2e-8）は、表の補間による。本体の同じ原子の対角成分そのものも、1 から最大 1.6e-5 ずれている。
- （2026-11-23、M2）重なり行列のフィクスチャーは、三斜晶のセルに Si（s2p2d1f1）、C（s2p2d1）、H（s2p1）を 1 個ずつ置いた入力（`modules/exx/tests/fixtures/overlap_reference.dat`）から作った。書き出すのは 45 ブロックの 6,741 要素である（同じ原子のブロック、各原子の近い 8 個、残りの 12 個に 1 個）。変換のスクリプトは `tools/exx_overlap_fixture.py` で、指示書にないファイルである。
- （2026-11-23、M2）カーネルの参照値は、`tools/exx_reference.py --kernels` が mpmath の 50 桁で作る。
- （2026-11-23、M2）主成分分析は、重なり行列 $O$ を対角化せず、積の関数そのものを片側ヤコビ法で直交化する。固有値の小さい成分も、丸め誤差の範囲で正規直交になる。しきい値は 1e-12 以上 1 未満に限った。符号は $\int r^{L+2}f\,dr$ で決め、この積分が相対 1e-10 より小さいときは、 $r$ の次数を 1 つずつ上げて決める（4 つ上まで）。
- （2026-11-23、M2）ABF の順序は、 $L$ の昇順、同じ $L$ では固有値の降順である。しきい値を大きくしたときの ABF は、小さくしたときの ABF の先頭の部分とビット単位で同じになる。
- （2026-11-23、M2）PAO を等間隔の格子に移す関数 `sample_orbitals()` は、本体と同じ補間（`modules/basis` の `RadialTableView`）を使う。メッシュの最初の点より内側では、s は最初の値、ほかは 0 になる。
- （2026-11-23、M2）同じ原子の $G_{PQ}$ と $b^P$ は、動径のクーロン積分を 2 通りの順序で求めて平均する（指示書の 13 節の 20）。
- （2026-11-23、M2）式 (7) は、行列の対角成分を 1 に揃えてからコレスキー分解で解く。枢軸が 1e-7 以下になった対は固有値分解に切り替え、固有値が 1e-7 以下の固有ベクトルを外す。2 つの値は `LriOptions` で変えられる。1e-7 は、同じ原子の動径積分と式 (12) の食い違い（既定の格子で最大 8.3e-8）から決めた。切り替えた対の数と外したベクトルの数は `LriReport` に残る（13 節の 12）。これまでに試した対では、切り替えは起きていない。
- （2026-11-23、M2）固有値の小さい方向は、特異でない限り残す。固有値 1e-5 以下の方向を外すと、短距離カーネルでの和が相対 3e-7 から 1.3e-5 変わり、格子の誤差（1e-8）より大きかった。
- （2026-11-23、M2）近傍リストは、原子自身を先頭に置き、残りを原子の番号、セルの 3 つの整数の順に並べる。距離の条件は「以下」で、本体の FNAN と同じである（`truncation.c:1487` と `1510`）。格子ベクトルは単位胞のベクトルから列挙する。
- （2026-11-23、M2）1 つの対は 1 回だけ解く。原子の番号が小さい方（同じ原子の周期像では、セルの整数が辞書式に大きい方）から見た解を、2 つの原子の T ブロックに分ける。V ブロックも 1 回だけ計算し、転置を相手に入れる。同じ原子の V ブロックは、転置との平均で対称にする。
- （2026-11-23、M2）`SystemOptions` の既定の相互作用は、HSE06 の短距離カーネル（ω = 0.11/bohr）にした。PBE0 では打ち切りクーロンのカーネルを渡す。
- （2026-11-23、M2）`prepare_system()` は MPI を使わず、単位胞の全原子を 1 回の呼び出しで処理する。ランクへの分担は M4 で加える。
- （2026-11-23、M2）例題 `lri_example` を加えた（`d4c0b44`）。`--generators` を付けると、ABF を PAO より大きい組の積から作る。`make_auxiliary_basis()` と `SpeciesTable` は、ABF を作る関数の組と PAO を別々に受け取るので、ライブラリーは変えずに済む。入力キーワードにはしていない。
- （2026-11-23、M2）指示書にない試験を 2 つ加えた（`exx-system-neighbours`、`exx-system-blocks`）。後者は、展開した積の電荷を本体の重なり行列と比べる。T ブロックの添字の順序を、モジュールが計算していない値で確かめるためである。クーロン計量の展開は電荷を保存しないので、判定は 1e-2 にした（実測は 4.4e-3）。
- （2026-11-23、M2）CPU 版の本体は、クラウドのコンテナに CI と同じ apt パッケージを入れてビルドした。M0 と M1 で使った Docker イメージは使っていない。

## 未解決の問題

- （2026-11-23）旧い厳密交換のコードが残っている。試作用入力が書いていた 4 個の変数の定義（`xmin_Rec_Coulomb`、`xmax_Rec_Coulomb`、`Ng1_Rec_Coulomb`、`Nrank_Rec_Coulomb`）、`Allocate_Arrays()` の `case 15`、`Free_Arrays.c` の 4 つの解放、ビルドに入っていない `Expand_ReCoulomb_Kernel.c`、到達しない `Band_DFT_Col_NEGF.c:743` の `if (5==XC_switch)`、`source/exx*` である。どれも動作には影響しない。削除するときは利用者に確認する（指示書の 1.3 節）。
- （2026-11-23、M1）短距離部分を含む計算は、PBE の関数の約 3.6 倍の時間がかかる。この計算機の 1 スレッドで、1 点あたり 1.5 マイクロ秒と 0.42 マイクロ秒だった（密度を 1e-8 から 1 まで対数で一様に取った点での値）。混合比 0 では PBE と同じ時間である。GPU への対応は M6 以降になる。この時間は書き直す前の計算核での値で、書き直しの後は測っていない。
- （2026-11-23、M1）`work/input_example/H2O.dat`（構造最適化）の再開ファイル `H2O.rst4gopt` は、同じ実行ファイルで 2 回計算しても 10 バイトが違う。M1 の変更とは関係がない。出力を比べるときは、このファイルを除く。
- （2026-11-23、M1）ハイブリッド汎関数を入力から選ぶ手段は、まだない（M4）。本体の中でホストのループが混合比 0.25 で動くことは、呼び出し側を書き換えた一時的なバイナリーで確かめただけである。この確認は書き直す前の定義で行った。
- （2026-11-23、M1 の書き直し）Quantum ESPRESSO のルーチンとの照合は倍精度で行った。 $w$ が 4 から 14 の領域では、差が Fortran の丸め誤差（エネルギーで 1.5e-6 まで）に埋もれる。この領域で定数が一致することは、 $w$ の小さい点で 1e-12 まで合うことと、ソースとの行ごとの照合に依っている。ルーチンを 4 倍精度でコンパイルした照合はしていない（指示書の 14 節）。
- （2026-11-23、M1 の書き直し）新しい定義の増強因子は、 $w=14$ で相対 2.6e-4 まで跳ぶ。 $\omega=0.11$ では、スピン分極のない全密度 1.6e-8 bohr⁻³ の点に当たる。SCF の収束と力への影響は確かめていない（M4、M7）。
- （2026-11-23、M1 の書き直し）交換孔の模型の出典（指示書の文献 [9]）は読んでいない。Heyd の学位論文（文献 [11]）の付録 A のコードも、全文は確認していない。定数は Quantum ESPRESSO 7.6 のソースから取った。
- （2026-11-23、M2）LRI の精度。既定の作り方の ABF では、結合距離の C–C、C–H、Si–Si で、2 原子の PAO の積の自己相互作用の和が 0.20% から 0.46% 不足する（短距離カーネルでは 0.25% から 0.56%）。主成分分析のしきい値を小さくしても減らない。角運動量の 1 つ高い関数を 1 個加えた組から ABF を作ると、0.02% から 0.06% になる。全エネルギー、ギャップ、力への影響は測っていない（M4）。ABF を増やすには、既定の作り方か入力キーワードを変えることになるので、利用者が決める（指示書の 1.3 節）。
- （2026-11-23、M2）2 原子の積の自己相互作用の厳密な値は、求めていない。比べた相手（s5p5d4f3 の積から作った ABF）も下からの近似で、g 以上の角運動量を含まない。実際の不足は、上の値以上である。
- （2026-11-23、M2）ABF の個数は、指示書の 12 節の仮定（32 個）の約 2 倍だった（C6.0-s2p2d1 と Si7.0-s2p2d1 で 69 個）。12 節の表は 32 個のままの計算値である。69 個では、T ブロックが 2.2 倍、V ブロックと $W$ の計算が 4.6 倍になる。M3 と M4 で実測する。
- （2026-11-23、M2）式 (7) の行列の条件数が大きいので（最大 6e6）、T ブロックの成分は 1e-10 程度までしか決まらない。計算機やコンパイラーが違うと、この大きさで変わると考えられる（確かめていない）。M3 で T ブロックを使うフィクスチャーを作るときは、比較の許容差をこれに合わせる。
- （2026-11-23、M2）打ち切りクーロンのカーネルで $R_c$ が大きいとき、既定の $\kappa$ の刻み（0.067/bohr）で式 (12) が収束するかは確かめていない（指示書の 13 節の 19）。M5 で確かめる。
- （2026-11-23、M2）`prepare_system()` の時間は、フィクスチャーのセル（3 原子、T ブロックの対は 112 個、V ブロックは 335 個）で 1.1 秒だった（既定の格子、1 コア、GCC 13.3.0 の `-O2`）。その前に行う元素ごとの表の作成（ABF の生成を含む）は、3 元素で 2.8 秒だった。このセルでは、T ブロックが 32 MB、V ブロックが 16 MB を占める。大きい系では測っていない。
- （2026-11-23、M2）M2 では `source/` と `core/` を変えていないので、本体の全エネルギーの比較、大域状態のゲート、GPU 版のコンパイル検査は、手元で実行していない。本体のビルドと指示書の `ctest`、GPU 版のコンパイル検査は、GitHub Actions が実行して成功している。
- （2026-11-23、M1 の書き直し）書き直しの後、本体の CPU 版のビルド、指示書の `ctest` の 18 件、ハイブリッド汎関数を指定しない計算の全エネルギー、大域状態のゲート、GPU 版のコンパイル検査を、手元では実行していない。理由は「受け入れ試験の結果」の表の下に書いた。

## 利用者への確認待ち

- （2026-11-23、M2）`exx-lri-identity` の結合距離の 2 原子の場合を、係数の成分ごとの差（最大 1.3e-10。要求の 1e-10 を超える）ではなく、展開した密度のクーロンノルム（4.3e-13）で判定してよいか。1 原子の場合（3.4e-14）と離れた 2 原子の場合（5.0e-14）は、成分ごとの差が要求を満たしている。理由は「受け入れ試験の結果」の表の下に書いた。回答を待たずに M3 へ進む。
- （2026-11-23、M2）LRI の精度への対応を、いつ決めるか。指示書の予定（14 節）のとおり、M4 で全エネルギーとギャップへの影響を測ってから相談する前提で進める。先に ABF を増やす場合は、作り方（例：角運動量の 1 つ高い関数を 1 個加える）と、既定値にするかキーワードにするかを決める必要がある。
- 2026-11-23 の 3 件（試作用入力の削除、ベースラインの更新、コンテナでのビルド）は回答を受けて対応した（「決めたこと」）。同じ日の「短距離部分を libxc の形のまま進めるか」の確認は、書き直すという回答を受けて対応した。
