# OpenMX 4.0 GPU ハイブリッド汎関数（HSE06・PBE0）実装指示書

OpenMX 4.0 GPU（`dc1394/openmx4.0_gpu`）に HSE06 と PBE0 の厳密交換を実装する作業者（Claude Code）への指示書です。設計は決まっています。作業者は 9 節のマイルストーンを番号順に実装します。

| 項目 | 内容 |
|---|---|
| 文書の改訂 | 改訂 10（2026-11-24）。M6 の実機の段階で書いた、デバイスの密行列ソルバーの経路を反映しました（付録 C）。 |
| 基準コミット | `4e0ee247438a3c9358e95adee00a6ff60a4c618b`（2026-11-22）。作業ブランチは `feature/exx` です。 |
| 行番号 | M0 が終わった時点（`feature/exx` の `50ef6b1`、2026-11-23）のものです。基準コミットと違うのは `source/Input_std.c` だけです。 |
| 付属ファイル | `exx_reference.py` は、交換行列の構築を NumPy で書いた参照実装です（10.1〜10.3 節）。`kbuild_reference.txt` は、その出力から作った単体試験用のフィクスチャーです。改訂 2 で、ωPBE の短距離交換を多倍長で評価する `wpbeh_reference.py` と、そのフィクスチャー `wpbeh_reference.txt` を加えました（10.4 節）。改訂 3 で、この 2 つを新しい定義で作り直しました。 |
| 関連文書 | `openmx_hybrid_exx_plan.md`。設計の背景です。この文書と食い違う点は、この文書を優先します（付録 B）。 |
| 置き場所 | この文書は `docs/EXX_IMPLEMENTATION_GUIDE.md`、参照実装は `tools/exx_reference.py` と `tools/wpbeh_reference.py`、フィクスチャーは `modules/exx/tests/fixtures/kbuild_reference.txt` と `modules/xc/tests/fixtures/wpbeh_reference.txt`、進捗の記録は `docs/EXX_PROGRESS.md` に置きます。 |

この文書の記述は、根拠を次の 3 種類に分けています。

| 種類 | 意味 |
|---|---|
| コードの確認 | 基準コミットと `50ef6b1` のソースを読んで確かめた事実です。場所を付録 A に示します。 |
| 数値の確認 | 2 つの参照実装を実行して確かめた式です。結果を 10 節に示します。 |
| 未確認 | 実機や実データでの確認が済んでいない事項です。本文でその旨を書き、14 節にまとめます。 |

## 1. 作業の規則

### 1.1 進める順序

1. セッションの最初に `docs/EXX_PROGRESS.md` を読みます。ファイルがなければ M0 から始め、ファイルを作ります。
2. `git log --oneline 50ef6b1..HEAD` で、行番号の基準にしたコミットからの差を確かめます。行番号がずれていたら、付録 A の検索語で場所を探します。
3. マイルストーンは番号順に進めます。受け入れ試験が全て通るまで、次の番号に進みません。
4. 1 つのコミットには 1 つの目的だけを入れます。既存コードの整形や、関係のない修正を混ぜません。
5. マイルストーンが終わるたびに `docs/EXX_PROGRESS.md` を更新します（1.4 節）。

### 1.2 守る条件

| 条件 | 理由 | 確認方法 |
|---|---|---|
| ハイブリッド汎関数を指定しない計算の結果を変えません。 | 既存の試験が、旧実装とのビット一致を確かめています（`xc-legacy` など）。 | 変更前後で `ctest` と代表的な入力（`work/input_example/Methane.dat` など）の全エネルギーを比べます。 |
| 新しい大域変数、ファイルスコープの `static` 変数、`getenv` の直接呼び出しを加えません。新しい関数は大域変数を読まず、引数で受け取ります。 | 大域状態のゲートは、関数と大域オブジェクトの組が増えると失敗します（`docs/GLOBAL_STATE_AUDIT.md`）。 | `python3 tools/audit_global_usage.py --output build/globals-usage.json --baseline docs/global-usage-baseline.json --check-no-new` を実行します。この計算機で必要な追加の引数は、`docs/EXX_PROGRESS.md` の「ビルドと実行」に書いてあるとおりです。clang と NVHPC のヘッダーがない環境では、増える組を報告に列挙します。 |
| `modules/` 以下のライブラリーは、MPI、BLAS、GPU SDK なしで単独にビルドでき、試験が通る状態を保ちます。 | CI（`.github/workflows/core.yml`）が、モジュールごとに単独でビルドして試験します。 | `cmake -S modules/exx -B build/exx && cmake --build build/exx && ctest --test-dir build/exx` |
| 演算は FP64 で実装します。FP32 と GEMMul8 は M6 で、FP64 との差を測ってから使います。 | 誤差の原因を、定式化と精度の 2 つに分けて調べられるようにするためです。 | M6 の受け入れ試験 |
| HIP 版の複製ソース（`source/backends/hip/src/`）には、M8 まで機能を入れません。共有のヘッダーや関数の引数を変えたときは、HIP 版の複製にも引数を合わせるだけの変更を入れ、コンパイルできる形を保ちます。 | HIP 版は `DFT.c` や `Band_DFT_Col.c` などを複製で置き換え、残りのファイルを `source/` と共有します（`source/backends/hip/sources.json`、`tools/prepare_build.py`）。 | `python3 tools/check_gpu_compilation.py --backend HIP`（ROCm がある環境。引数は `.github/workflows/gpu-compile.yml` に合わせます） |
| 第三者のコードを取り込むときは、ライセンスと出典をファイルに残します。 | OpenMX は GPLv3 です。短距離交換の式と定数は、GPL の Quantum ESPRESSO から書き写しました（M1）。 | レビュー |
| libxc を使いません。ビルド、リンク、実行、CI に加えず、式や定数の出典にもしません。PySCF など、libxc を含むパッケージも同じ扱いです。 | 利用者の指示です（2026-11-23）。 | `grep -rniw libxc modules source tools` が何も返さないこと |

### 1.3 作業を止めて利用者に確認する場合

- 受け入れ試験の閾値を満たせず、原因が設計（ABF の作り方、打ち切り距離、アルゴリズム）にあると判断した場合
- 既定値、キーワード名、対象範囲（2 節）を変える必要が出た場合
- 大域状態のベースライン（`docs/global-usage-baseline.json`）を更新する必要が出た場合
- 既存の数値コードの演算の順序を変える必要が出た場合
- 旧 EXX コード（`source/exx*`、`source/Expand_ReCoulomb_Kernel.c`）を削除したい場合
- 第三者コードの取り込み方に、ライセンス上の判断が要る場合
- GPU の実機が必要な試験を実行できない場合（M6）

止まるときは、試した内容、観測した数値、取り得る選択肢を報告します。

### 1.4 進捗の記録

`docs/EXX_PROGRESS.md` は次の構成にします。作業者はセッションをまたいで記憶を持たないので、このファイルを唯一の引き継ぎにします。

```markdown
# EXX 実装の進捗
## 現在の状態
- 完了したマイルストーン: M0, M1
- 作業中: M2（二中心積分まで完了、LRI の係数は未着手）
- 基準コミットからのブランチ: feature/exx
## 受け入れ試験の結果
| マイルストーン | 試験 | 結果 | 日付 | 条件（計算機、ビルド） |
## 決めたこと
- （日付）ABF の符号は、動径積分が正になる向きに固定した。
## 未解決の問題
- （日付）Si の ABF 数が 41 個になり、12 節の仮定 32 個より多い。
## 利用者への確認待ち
```

### 1.5 セッションの開始時に読ませる指示

Claude Code のセッションは毎回新しい文脈で始まり、`CLAUDE.md` は開始時に読み込まれます [14]。基準コミットのリポジトリーのルートには `CLAUDE.md` がありません。M0 で次の内容のファイルを作ります。この指示書の本文は `CLAUDE.md` に写しません。毎回読み込まれる量を小さく保つためです。

```markdown
# openmx4.0_gpu
- ハイブリッド汎関数（HSE06・PBE0）を実装しています。作業の前に `docs/EXX_IMPLEMENTATION_GUIDE.md` の 1 節と `docs/EXX_PROGRESS.md` を読んでください。
- 着手するマイルストーンの節（9 節）と、そこから参照されている節を読んでから作業してください。
- マイルストーンが終わったら `docs/EXX_PROGRESS.md` を更新してください。
```

## 2. 対象範囲

### 2.1 段階ごとの範囲

| 項目 | 第 1 版（M0〜M7） | 後続（M8） | 対象外 |
|---|---|---|---|
| 汎関数 | HSE06 | PBE0 | 長距離補正型 |
| スピン | コリニア（非分極、分極） | ノンコリニア、スピン軌道相互作用 | なし |
| 固有値ソルバー | Cluster、Band | Cluster-DIIS、Cluster2、Cluster-LNO | Recursion、DC、GDC、Krylov、DC-LNO、EGAC、NEGF |
| k 点 | Γ 点、一般の k 点 | なし | なし |
| 物理量 | 全エネルギー、固有値、力 | 応力、バンド分散、状態密度、`HS.fileout` | なし |
| 実行環境 | CPU、CUDA | HIP | なし |
| 浮動小数点 | FP64 | 混合精度 | なし |

### 2.2 入力の時点で拒否する組み合わせ

ハイブリッド汎関数を指定した入力が次のどれかに当たる場合は、理由を表示して終了します。書き方は既存のエラー処理（`source/Input_std.c:1009-1024`）と同じです。対応が済んだ項目は、表から外します。

| 条件 | キーワード | 外す時期 |
|---|---|---|
| Cluster と Band 以外のソルバー | `scf.EigenvalueSolver` | M8 以降 |
| ノンコリニア | `scf.SpinPolarization NC` | M8 |
| 構造最適化と分子動力学 | `MD.Type` が `Nomd` 以外 | M7 |
| 応力 | `scf.stress.tensor` | M8 |
| 近傍リストの形式のハミルトニアンを読む出力と、書き出す出力 | `Band.dispersion`、`Dos.fileout`、`DosGauss.fileout`、`HS.fileout`、`MO.fileout`、`Wannier.Func.Calc`、`Unfolding.Electronic.Band`、`scf.Polarization.ABC`、`CWF.Calc`、`DM.specify.energy.range`（`DM.export` と一緒に指定した場合）、`NEGF.Output_HKS` | M8 |
| 厳密交換を含まないエネルギーの分解 | `Energy.Decomposition` | M8 |
| 確認していない機能 | `scf.Hubbard.U`、`orbitalOpt.Method`、`scf.core.hole`、`ESM.switch`、`scf.Electric.Field`、`CDDF.start`、`NBO.switch`、`XANES.gs.fileout`、`XANES.calc` | 個別に確認した後 |
| HIP のビルド | なし（ビルド時の種別で判定） | M8 |

出力系を拒否する理由は 13 節の 1 番に書きます。

M4 の実装（`source/Input_std.c`）は、当たる条件を全て並べて表示してから終了します。Γ 点だけの入力で Band を指定した場合は、本体が Cluster に切り替えます。`DosGauss.fileout`、`scf.Polarization.ABC`、`CWF.Calc`、`Energy.Decomposition`、`CDDF.start`、`NBO.switch` は、初版の表になかった項目で、M4 で加えました。`DosGauss.fileout` は `Dos.fileout` と同じ出力の別の方式です。残りの 5 つを拒否してよいかは、利用者の確認を待っています（進捗ファイルの「利用者への確認待ち」）。`PBE0` は、M8 まで「実装していない」と表示して終了します。

M5 で、Band ソルバーの行を表から外しました。k 点の生成は、`scf.Generation.Kpoint` の `GCenter`（既定）と `REGULAR` で確かめています。`none`（対称性で減らした Monkhorst–Pack のメッシュ）は、ハイブリッド汎関数でない計算でも動きません（13 節の 28）。M5 では、表にキーワードを 4 つ加えました。`DM.specify.energy.range` は、SCF の後に近傍リストの形式の $H$ をもう 1 回解き、エネルギーの範囲を限った密度行列を作ります。`NEGF.Output_HKS` は、電極用のファイルに同じ形式の $H$ を書きます。どちらの $H$ も $K$ を含みません。`XANES.gs.fileout` と `XANES.calc` の処理は、k メッシュの計算でも Cluster ソルバーを呼びます。site の形式の $K$ は、この呼び出しに使えません。Cluster ソルバーの計算でも結果を確かめていないので、2 つともソルバーによらず拒否します。XANES の 2 つを拒否してよいかは、M4 の 5 つと合わせて利用者の確認を待っています。

## 3. 決定済みの設計

| 項目 | 決定 | 理由 |
|---|---|---|
| 交換行列の形 | 全交換行列 $K$ を実空間のブロック $K(\mathbf R)$ として作り、密行列の $H(\mathbf k)$ に足します。 | 非占有状態にも同じ行列が使えます。参照軌道の本数という設定が要りません。任意の k 点の $K(\mathbf k)$ が位相の和だけで得られます。 |
| 構築のアルゴリズム | 密度行列 $D(\mathbf R)$ から、原子ブロックの 4 項で作ります（5.5 節）。 | 軌道の対を添字に持つ配列を作りません。構築の演算量は単位胞の原子数に比例し、k 点数によりません（密度行列の準備を除きます）。参照実装で式を確認済みです。 |
| 四中心積分 | 局所 RI（LRI）[4] で近似します。補助基底関数（ABF）は、同じ原子の PAO の積から作ります。 | 必要な積分が全て二中心積分になります。 |
| LRI の計量 | 裸のクーロンを使います。 | ABACUS の入力説明 [12] も、LRI の係数にクーロン行列の逆を使うと記しています。 |
| 二中心積分 | 逆空間の公式（5.6 節）で計算します。 | 既存の重なり積分（`Set_OLP_Kin.c`）と同じ形の動径積分です [5]。 |
| SCF | 2 段の反復にします。内側では $K$ を固定し、外側で $K$ を作り直します。 | 構築の回数が減ります。内側は既存の混合をそのまま使えます。 |
| 最初の内側反復 | PBE だけで収束させます。 | $K$ を作るには収束に近い密度行列が要ります。 |
| 準局所部分 | 内部の `XC_switch` は 4（GGA-PBE）のままにし、混合比と遮蔽パラメーターを引数で渡します。HSE06 の交換は $E_x^{\mathrm{PBE}}-\alpha E_x^{\omega\mathrm{PBE,SR}}$ です。 | `XC_switch==4` を条件にした既存の分岐を変えずに済みます。 |
| コードの置き場所 | 状態を持たない計算は `modules/exx` と `modules/xc` に置きます。SCF との結合は `source/` に置きます。 | リポジトリーの方針です（`docs/CPP_LIBRARY_ARCHITECTURE.md`）。 |
| 行列積 | `modules/exx` は GEMM を関数の表で受け取ります。 | 単独ビルドの条件を満たしたまま、本体が BLAS、cuBLAS、GEMMul8 を選べます。 |
| 射影演算子 | 第 1 版では実装しません。 | 全交換行列で足ります。概要 [1] や ACE [2] の形が必要になったら、参照軌道の係数 $c_{\mathrm{ref}}$ から $A=Kc_{\mathrm{ref}}$ で作れます。 |

## 4. 記号と規約

| 記号 | 意味 |
|---|---|
| $\phi_\mu$ | PAO。添字は $\mu,\nu,\lambda,\eta$ を使います。 |
| $P,Q$ | ABF。動径関数 $f_P(r)$ と実球面調和関数の積です。 |
| $a,b,c$ | 単位胞の中の原子。 |
| site | 原子と格子ベクトルの組 $(a,\mathbf R)$ 。 $\mathbf R$ は整数 3 個で表します。位置は $\boldsymbol\tau_a+\mathbf R$ です。 |
| $A,B$ | site。 $A=(a,\mathbf 0)$ は ABF を持つ中心の原子、 $B$ はその相手です。 |
| $\mathrm{loc}(A)$ | $A$ の C 近傍。 $A$ の PAO と重なる PAO を持つ site の集合で、 $A$ 自身を先頭に含みます。 |
| V 近傍 | 二中心積分 $V$ を残す site の対。 |
| $u_a$ , $n_{P,a}$ , $\ell_A$ | 原子 $a$ の PAO の数、ABF の数、 $\mathrm{loc}(A)$ の PAO の総数。 |
| $D^{\sigma}$ , $K^{\sigma}$ | スピン $\sigma$ の密度行列と交換行列。 |
| $v$ , $g$ | 相互作用カーネルと、LRI の計量。 |
| $T$ , $V$ | LRI の係数のブロックと、ABF の間の二中心積分のブロック。 |

添字、位相、スピンの規約は既存のコードに合わせます。

- 単位は Hartree 原子単位です。
- 軌道の通し番号は既存の `MP[]` に従います。密行列は列優先で、要素 $(\mu,\nu)$ は `(nu)*n + mu` に置きます（`Band_DFT_Col.c:1493`）。
- 行列 $M_{\mu\nu}(\mathbf R)$ は、 $\mu$ が原点の単位胞、 $\nu$ が格子ベクトル $\mathbf R$ の単位胞にあるときの要素です。エルミート性は $M_{\mu\nu}(\mathbf R)=M^{\ast}_{\nu\mu}(-\mathbf R)$ です。
- ブロッホ和の位相は $e^{+2\pi i\mathbf k\cdot\mathbf R}$ で、 $\mathbf k$ は逆格子ベクトルを単位にした座標です。既存の $H(\mathbf k)$ の組み立て（`Band_DFT_Col.c:6862`）と同じです。
- $D^{\sigma}$ はスピン 1 成分の密度行列で、占有数は 0 から 1 です。非分極の計算（`SpinP_switch==0`）では $D^{\uparrow}=D^{\downarrow}$ として 1 成分だけを計算し、エネルギーを 2 倍します。
- コリニアでスピン軌道相互作用がない場合、 $D^{\sigma}(\mathbf R)$ と $K^{\sigma}(\mathbf R)$ は実数です。

## 5. 数式

### 5.1 汎関数

$$E_{xc}^{\mathrm{HSE06}}=E_{xc}^{\mathrm{PBE}}+\alpha\left(E_x^{\mathrm{HF,SR}}(\omega)-E_x^{\omega\mathrm{PBE,SR}}(\omega)\right),\qquad\alpha=\tfrac14,\quad\omega=0.11 \tag{1}$$

$$E_{xc}^{\mathrm{PBE0}}=E_{xc}^{\mathrm{PBE}}+\alpha\left(E_x^{\mathrm{HF}}-E_x^{\mathrm{PBE}}\right),\qquad\alpha=\tfrac14 \tag{2}$$

$\omega$ の単位は bohr⁻¹ です。 $E_x^{\mathrm{HF,SR}}$ は $\mathrm{erfc}(\omega r)/r$ による厳密交換、 $E_x^{\omega\mathrm{PBE,SR}}$ は同じ遮蔽をかけた PBE 交換です [8, 9, 10]。 $\omega=0.11$ bohr⁻¹ は文献 [10] が推奨する値で、2 つの項に同じ値を使います。既定値はコードによって違います。Quantum ESPRESSO 7.6 の `screening_parameter` の既定値は 0.106 bohr⁻¹ です [16]。VASP は HSE06 に `HFSCREEN = 0.2`（単位は Å⁻¹。約 0.1058 bohr⁻¹）を使います [15]。他のコードと比べるときは $\omega$ を揃えます。

準局所の交換は $E_x^{\mathrm{PBE}}-\alpha E_x^{\omega\mathrm{PBE,SR}}$ です。全域の部分には既存の PBE 交換を使い、Quantum ESPRESSO 7.6 も同じ形で計算します [16]。 $\omega=0$ にした短距離部分の式は、PBE の交換と一致しません。増強因子は、 $s\le5$ の範囲で PBE の増強因子と最大 0.54% 違います（M1）。全域の部分をこの式で置き換えてはいけません。短距離部分 $E_x^{\omega\mathrm{PBE,SR}}$ の定義は、Quantum ESPRESSO 7.6 の `pbexsr` と同じです（M1）。

### 5.2 交換行列とエネルギー

$\phi_{\mu\mathbf R}(\mathbf r)=\phi_\mu(\mathbf r-\boldsymbol\tau_\mu-\mathbf R)$ と書きます。交換行列の定義は次の式です。

$$K^{\sigma}_{\mu\nu}(\mathbf R)=-\alpha\sum_{\lambda\mathbf R_1}\sum_{\eta\mathbf R_2}D^{\sigma}_{\lambda\eta}(\mathbf R_2-\mathbf R_1)\left(\phi_{\mu\mathbf 0}\phi_{\lambda\mathbf R_1}\vert v\vert\phi_{\eta\mathbf R_2}\phi_{\nu\mathbf R}\right) \tag{3}$$

$$(f\vert v\vert h)=\iint f(\mathbf r)v(\lvert\mathbf r-\mathbf r'\rvert)h(\mathbf r')d^3r\thinspace d^3r'$$

単位胞あたりの厳密交換エネルギーは次の式です。

$$E_x=\frac12\sum_{\sigma}\sum_{\mathbf R}\sum_{\mu\nu}D^{\sigma}_{\mu\nu}(\mathbf R)K^{\sigma}_{\mu\nu}(\mathbf R) \tag{4}$$

### 5.3 密度行列

$$D^{\sigma}_{\mu\nu}(\mathbf R)=\frac1W\sum_{\mathbf k}w_{\mathbf k}\mathrm{Re}\left[e^{-2\pi i\mathbf k\cdot\mathbf R}\sum_jf^{\sigma}_{j\mathbf k}c^{\sigma}_{\mu j}(\mathbf k)c^{\sigma\ast}_{\nu j}(\mathbf k)\right],\qquad W=\sum_{\mathbf k}w_{\mathbf k} \tag{5}$$

$w_{\mathbf k}$ は既存の重み `T_k_op` です。 $\mathbf k$ と $-\mathbf k$ の組は片方だけを計算し、重みを 2 にしています。実部を取る操作が、省いた側の寄与にあたります。

式 (5) の位相は、必要な $\mathbf R$ ごとに実際の k 点の座標（`T_KGrids1` から `T_KGrids3`）で計算します。 $D(\mathbf R)$ を Born–von Kármán のスーパーセルの周期で折り返してはいけません。メッシュがずれていると、 $D(\mathbf R)$ はスーパーセルの並進で位相が付くためです。OpenMX の `REGULAR` メッシュは、分割数が偶数のとき半格子分ずれた点を使います（`Band_DFT_Col.c:3742`）。ずらしたメッシュでも式 (3) と式 (5) の組が k 空間の定義と一致することは、数値で確認済みです。

既存の密度行列 `CDM` は FNAN の範囲（距離が 2 つのカットオフ半径の和以下）しか持ちません。式 (3) は、それより遠い原子の対の $D$ を使います（12 節）。

### 5.4 局所 RI

$\mu$ が site $A=(a,\mathbf 0)$ に、 $\lambda$ が site $C=(c,\mathbf R)$ にあるとき、積を 2 つの site の ABF で展開します。

$$\phi_{\mu}(\mathbf r-\boldsymbol\tau_a)\phi_{\lambda}(\mathbf r-\boldsymbol\tau_c-\mathbf R)\approx\sum_{P\in A}C^{(1)P}_{\mu\lambda}(\mathbf R)P(\mathbf r-\boldsymbol\tau_a)+\sum_{P\in C}C^{(2)P}_{\mu\lambda}(\mathbf R)P(\mathbf r-\boldsymbol\tau_c-\mathbf R) \tag{6}$$

係数は、計量 $g$ で測った残差を最小にする連立方程式から決めます。 $P$ と $Q$ は 2 つの site の ABF 全体を動きます。

$$\sum_{Q\in A\cup C}G_{PQ}C^{Q}_{\mu\lambda}=b^{P}_{\mu\lambda},\qquad G_{PQ}=(P\vert g\vert Q),\qquad b^{P}_{\mu\lambda}=(\phi_\mu\phi_\lambda\vert g\vert P) \tag{7}$$

対を逆から見ると $C^{(2)P}_{\mu\lambda}(\mathbf R)=C^{(1)P}_{\lambda\mu}(-\mathbf R)$ が成り立つので、保存するのは $C^{(1)}$ だけです。保存の単位を T ブロックと呼びます。

- $T_a[i][P,\mu,\lambda]=C^{(1)P}_{\mu\lambda}$ 。 $P$ と $\mu$ は原子 $a$ に、 $\lambda$ は $a$ の $i$ 番目の C 近傍の site にあります。
- $i=0$ は $a$ 自身です。同じ原子の対は ABF が 1 組しかないので、式 (7) を $a$ の ABF だけで解きます。 $T_a[0]$ には、その解の半分を入れます。次の式 (8) の 2 つの項が同じ寄与を 1 回ずつ足すためです。

site $A$ の ABF $P$ の係数を、全ての site の PAO を添字とする行列 $C^{P}$ で書きます。 $E_A$ は site $A$ の PAO を取り出す行列、 $T_P$ は $u_a\times\ell_A$ の行列（T ブロックを $\mathrm{loc}(A)$ の順に横に並べたもの）です。

$$C^{P}=E_AT_P+T_P^{\mathsf T}E_A^{\mathsf T} \tag{8}$$

四中心積分と交換行列は次の形になります。 $V_{PQ}=(P\vert v\vert Q)$ です。

$$\left(\phi_\mu\phi_\lambda\vert v\vert\phi_\eta\phi_\nu\right)\approx\sum_{PQ}C^{P}_{\mu\lambda}V_{PQ}C^{Q}_{\eta\nu},\qquad K^{\sigma}=-\alpha\sum_{PQ}V_{PQ}C^{P}D^{\sigma}C^{Q} \tag{9}$$

### 5.5 原子ブロックの 4 項

式 (9) に式 (8) を代入すると、 $P\in A$ と $Q\in B$ の組ごとに 4 つの項が出ます。 $S_A$ を、 $A$ の全ての V 近傍 $B$ についての $\mathrm{loc}(B)$ の和集合とし、次の 2 つを定義します。

$$G_P=T_PD\left[\mathrm{loc}(A),S_A\right],\qquad W_{P,B}=\sum_{Q\in B}V_{PQ}T_Q \tag{10}$$

$G_P$ は $u_a$ 行の行列で、列は $S_A$ の PAO に対応します。 $W_{P,B}$ は $u_b\times\ell_B$ の行列です。 $D[X,Y]$ は、行を site の集合 $X$ から、列を $Y$ から取った部分行列を表します。

| 項 | 行の site | 列の site | 値 |
|---|---|---|---|
| (i) | $A$ | $B$ | $\sum_{P}G_P[\cdot,\mathrm{loc}(B)]\thinspace W_{P,B}^{\mathsf T}$ |
| (ii) | $A$ | $\mathrm{loc}(B)$ | $\sum_{P}G_P[\cdot,B]\thinspace W_{P,B}$ |
| (iii) | $\mathrm{loc}(A)$ | $B$ | $\sum_{P}T_P^{\mathsf T}\left(D[A,\mathrm{loc}(B)]\thinspace W_{P,B}^{\mathsf T}\right)$ |
| (iv) | $\mathrm{loc}(A)$ | $S_A$ | $\sum_{P}T_P^{\mathsf T}R_P$ 。 $R_P$ は $B$ ごとに $R_P[\cdot,\mathrm{loc}(B)]\mathrel{+}=D[A,B]\thinspace W_{P,B}$ で積み上げます。 |

計算は中心の原子ごとに独立で、次の 4 段からなります。

1. 単位胞の原子 $a$ を順に中心 $A=(a,\mathbf 0)$ とします。
2. $A$ の V 近傍 $B$ を全て回り、表の 4 項を計算します。 $P$ の和は $a$ の ABF について取ります。
3. 得た値を、行の site $(x,\mathbf R_x)$ と列の site $(y,\mathbf R_y)$ から決まるブロック $K_{xy}(\mathbf R_y-\mathbf R_x)$ に加算します。
4. 最後に全体を $-\alpha$ 倍します。

中心を単位胞の原子に限っても、並進対称性により全ての寄与がちょうど 1 回ずつ入ります。この手順が式 (3) に式 (9) を入れた定義と一致することは、数値で確認済みです（10 節の T1 と T3）。

### 5.6 二中心積分とカーネル

ABF を $P(\mathbf r)=f_P(r)Y_{l_Pm_P}(\hat{\mathbf r})$ と書きます。 $Y_{lm}$ は実球面調和関数です。動径部分の変換を次のように定めます。

$$\tilde f_P(\kappa)=\int_0^\infty r^2f_P(r)j_{l_P}(\kappa r)dr \tag{11}$$

$P$ の中心から $Q$ の中心へのベクトルを $\mathbf d$ 、その長さを $d$ とします。

$$(P\vert v\vert Q)_{\mathbf d}=8\sum_{L,M}i^{l_P-l_Q-L}\Theta^{LM}_{PQ}Y_{LM}(\hat{\mathbf d})\int_0^\infty\kappa^2\tilde f_P(\kappa)\tilde f_Q(\kappa)\tilde v(\kappa)j_L(\kappa d)d\kappa \tag{12}$$

$\Theta^{LM}_{PQ}=\int Y_{l_Pm_P}Y_{l_Qm_Q}Y_{LM}d\Omega$ は実球面調和関数の Gaunt 係数です。 $L$ は $\lvert l_P-l_Q\rvert$ から $l_P+l_Q$ まで 2 ずつ動くので、位相因子は 1 か $-1$ です。 $d=0$ では $L=0$ の項だけが残ります。

| カーネル | $v(r)$ | $\tilde v(\kappa)$ | $\kappa\to0$ の値 | 用途 |
|---|---|---|---|---|
| 重なり | $\delta(\mathbf r)$ | $1$ | $1$ | 三中心積分、既存の重なり行列との照合 |
| 裸のクーロン | $1/r$ | $4\pi/\kappa^2$ | 発散（ $\kappa^2$ と打ち消し合う） | 計量 $g$ |
| 短距離 | $\mathrm{erfc}(\omega r)/r$ | $4\pi\left(1-e^{-\kappa^2/4\omega^2}\right)/\kappa^2$ | $\pi/\omega^2$ | HSE06 |
| 切断クーロン | $\theta(R_c-r)/r$ | $4\pi\left(1-\cos\kappa R_c\right)/\kappa^2$ | $2\pi R_c^2$ | PBE0 |

式 (12) と 4 つのカーネルは、ガウス関数の解析値と比べて確認済みです（10 節）。PBE0 の $R_c$ は、半径 $R_c$ の球の体積が $N_k\Omega$ に等しくなるように決めます [13]。 $\Omega$ は単位胞の体積です。

### 5.7 三中心積分

式 (7) の $b^{P}_{\mu\lambda}$ は、 $P$ が 2 つの PAO のどちらかと同じ原子にある積分です。 $P$ が $\mu$ の原子にある場合を書きます。ABF $P$ が計量 $g=1/r$ のもとで作るポテンシャルを $W_P$ とします。

$$W_P(\mathbf r)=\frac{4\pi}{2l+1}\left[r^{-l-1}\int_0^rf_P(r')r'^{l+2}dr'+r^{l}\int_r^\infty f_P(r')r'^{1-l}dr'\right]Y_{lm}(\hat{\mathbf r}) \tag{13}$$

$l$ と $m$ は $P$ の角運動量です。 $b^{P}_{\mu\lambda}$ は、積 $\phi_\mu W_P$ と $\phi_\lambda$ の重なり積分になります。積 $\phi_\mu W_P$ は $\mu$ の原子を中心とする関数で、PAO のカットオフ半径の外では 0 です。角度部分は Gaunt 係数で実球面調和関数に展開できるので、式 (12) の重なりカーネルで計算できます。 $P$ が $\lambda$ の原子にある場合は、2 つの PAO の役割を入れ替えます。この方法は、s 型と p 型のガウス関数の解析値と比べて確認済みです。

同じ原子の ABF の間の $G_{PQ}$ は、 $l$ と $m$ が等しい組だけが値を持ち、 $\int_0^\infty r^2f_P(r)w_Q(r)dr$ です。 $w_Q$ は式 (13) の角括弧に $4\pi/(2l+1)$ を掛けた動径関数です。

### 5.8 k 点での行列と外側反復の判定

$$K^{\sigma}_{\mu\nu}(\mathbf k)=\sum_{\mathbf R}e^{2\pi i\mathbf k\cdot\mathbf R}K^{\sigma}_{\mu\nu}(\mathbf R) \tag{14}$$

外側反復の $n$ 回目で作った $K^{(n)}$ と、直前の内側反復で使った $K^{(n-1)}$ から、次の量を計算します。 $D^{(n)}$ は $K^{(n)}$ を作るのに使った密度行列です。

$$\delta^{(n)}=\sum_{\sigma}\sum_{\mathbf R}\sum_{\mu\nu}D^{\sigma,(n)}_{\mu\nu}(\mathbf R)\left[K^{\sigma,(n)}_{\mu\nu}(\mathbf R)-K^{\sigma,(n-1)}_{\mu\nu}(\mathbf R)\right] \tag{15}$$

$\lvert\delta^{(n)}\rvert$ が閾値より小さければ外側反復を終えます。

### 5.9 力に使う偏微分

密度行列を固定したときの $E_x$ の、T ブロックと V ブロックについての偏微分です。Γ 点だけの単位胞で、スピン 1 成分、実対称の $D$ についての式です。

$$\frac{\partial E_x}{\partial T_P[\mu,\lambda]}=-2\alpha F^{P}[\mu,\lambda],\qquad F^{P}=\left(R_P+R'_P\right)D\left[\cdot,\mathrm{loc}(A)\right] \tag{16}$$

$R_P$ は 5.5 節の (iv) と同じ行列です。 $R'_P$ は、 $B$ ごとに $R'_P[\cdot,B]\mathrel{+}=D[A,\mathrm{loc}(B)]\thinspace W_{P,B}^{\mathsf T}$ で積み上げます。

$$\frac{\partial E_x}{\partial V_{PQ}}=-\frac{\alpha}{2}\Gamma_{PQ},\qquad\Gamma_{PQ}=\mathrm{Tr}\left[DC^{P}DC^{Q}\right] \tag{17}$$

$P\in A$ と $Q\in B$ について、 $\Gamma_{PQ}$ は次の 3 つの和です。 $G_Q=T_QD[\mathrm{loc}(B),\cdot]$ は $B$ を中心とする式 (10) の行列、 $\circ$ は要素ごとの積の総和です。

$$\Gamma_{PQ}=T_Q\circ\left(D[B,A]\thinspace G_P[\cdot,\mathrm{loc}(B)]\right)+T_P\circ\left(D[A,B]\thinspace G_Q[\cdot,\mathrm{loc}(A)]\right)+2\thinspace\mathrm{Tr}\left[G_P[\cdot,B]\thinspace G_Q[\cdot,A]\right]$$

式 (16) と式 (17) は、中心差分と比べて確認済みです（10 節の T6）。site の形式への書き換えは M7 の作業です。原子の位置についての微分は連鎖律で得ます。対ごとの係数ベクトル $c$ （2 つの原子の ABF についての係数を並べたもの）の変分は、式 (7) から次の式になります。

$$\delta c=G^{-1}\left(\delta b-\delta G\thinspace c\right) \tag{18}$$

$b$ と $G$ と $V$ は二中心積分なので、微分は 2 原子の相対座標についての微分です。軌道の応答は、既存の Pulay 項（エネルギー重み付き密度行列と重なり行列の微分）が担います。

## 6. アルゴリズム

### 6.1 全体の流れ

1. 構造で決まる量を準備します（6.2 節）。
2. PBE のまま内側の SCF を収束させます。
3. 収束したハミルトニアンをもう 1 回解き、密度行列 $D(\mathbf R)$ を集めます（6.3 節）。
4. $K(\mathbf R)$ と $E_x$ と $\delta$ を計算します（6.4 節）。
5. $\lvert\delta\rvert$ が閾値より小さければ終了します。1 回目は必ず続けます。
6. 準局所部分をハイブリッド用に切り替え、 $K$ を固定して内側の SCF を収束させます。対角化のたびに $K(\mathbf k)$ を $H(\mathbf k)$ に足します（6.5 節）。
7. 手順 3 に戻ります。

### 6.2 構造ごとの準備

MD の 1 ステップにつき 1 回だけ実行します。

| 手順 | 内容 |
|---|---|
| ABF の生成 | 元素ごとに、同じ原子の PAO の動径関数の積を全て作ります。角運動量 $L$ ごとに重なり行列 $O_{kk'}=\int r^2g_kg_{k'}dr$ を対角化し、固有値が最大固有値の `scf.exx.abf.pca` 倍より大きい成分だけを残します。残した成分が ABF の動径関数で、互いに正規直交です。固有ベクトルの符号は、 $\int r^{L+2}f(r)dr$ が正になる向きに固定します。 |
| 近傍リスト | C 近傍は、距離が 2 原子の PAO のカットオフ半径の和以下の site です（FNAN と同じ条件）。V 近傍は、距離が「2 原子の ABF の半径のうち大きい方」と `scf.exx.v.cutoff` の和以下の site です。格子ベクトルは単位胞のベクトルから列挙します（13 節の 4 番）。 |
| 動径の変換 | ABF、PAO、積 $\phi_\mu W_P$ の球ベッセル変換（式 (11)）を作ります。 |
| V ブロック | V 近傍の対ごとに式 (12) で計算します。 $V_{ab}(\mathbf R)=V_{ba}(-\mathbf R)^{\mathsf T}$ なので、片方を計算して転置を入れます。 |
| T ブロック | C 近傍の対ごとに式 (7) を解きます。行列の次数は 2 原子の ABF の数の和です。解は 2 つの原子の T ブロックに分けて入れます。 |

ABF の半径は PAO のカットオフ半径と同じです。積 $g_k$ の角運動量 $L$ は、2 つの PAO の角運動量 $l_1$ と $l_2$ から $\lvert l_1-l_2\rvert\le L\le l_1+l_2$ かつ $l_1+l_2+L$ が偶数の範囲を動きます。s2p2d1 の基底では、間引く前の動径関数は原子あたり 22 個、 $m$ を数えた ABF は 78 個です（計算値）。

### 6.3 密度行列の収集

内側の SCF が収束したら、収集を有効にして固有値ソルバーをもう 1 回呼びます。FP32 の精製ステップで収束した後に FP64 で解き直す既存の流れ（`DFT.c:1463-1505`）と同じ形です。ソルバーは、スピンと k 点ごとに固有ベクトル、占有数、重みを EXX の状態に渡します。

| 場合 | 保持する量 | 計算 |
|---|---|---|
| Γ 点だけ | 密行列 $D^{\sigma}=cFc^{\mathsf T}$ | GEMM 1 回 |
| k 点 | ブロック $D^{\sigma}_{xy}(\mathbf R)$ の表 | k 点ごとに GEMM 1 回。式 (5) の位相を掛けて、表の全てのブロックに足します。 |

固有ベクトルは FP64 のものを使います。FP32 の精製ステップの結果は使いません。

k 点の計算では、 $D(x,y,\mathbf R)$ のブロックを、原子の対と格子ベクトルの組ごとに 1 回だけ計算して表にします。同じ組が、多くの site の対で繰り返し現れるためです。12 節の Si の例では、43 bohr 以内の格子ベクトルは 1253 個です（計算値）。

M5 の実装です。表に入れるブロックは、構築が読むものだけです。行の site が $\mathrm{loc}(A)$ に、列の site が $S_A$ にある組で、一覧は `modules/exx` が近傍リストから作ります（6.4 節）。結合層は、k 点ごとの $P(\mathbf k)$ を保存しません。ソルバーが k 点を 1 つ解くたびに次の行列を作り、全てのブロックに $\mathrm{Re}\left[e^{2\pi i\mathbf k\cdot\mathbf R}Q_{xy}(\mathbf k)\right]/W$ を足します。

$$Q^{\sigma}_{\mu\nu}(\mathbf k)=\sum_jw_{\mathbf k}f^{\sigma}_{j\mathbf k}c^{\sigma\ast}_{\mu j}(\mathbf k)c^{\sigma}_{\nu j}(\mathbf k)$$

式 (5) の角括弧の中身の複素共役を取った形なので、実部は同じです。k 点をランクに配る計算では、各ランクが自分の k 点の分を足します。収集の終わりに全ランクの和を取り、 $D_{xy}(\mathbf R)$ と $D_{yx}(-\mathbf R)^{\mathsf T}$ を 2 つの平均に置き換えます。

Band ソルバーには、密度行列を作る経路が 2 つあります。1 つは、1 回の対角化で得た固有ベクトルをそのまま使う経路です。もう 1 つは、化学ポテンシャルが決まった後に、k 点ごとにもう 1 回解く経路です。k 点をランクに配り、1 つのランクが複数の k 点を受け持つときに、後の経路を通ります。どちらの経路でも、ソルバーは密度行列の計算に使うのと同じ固有ベクトルと重みを結合層に渡します。

M6 で、デバイスの密行列ソルバーの経路（6.5 節）にも収集を加えました。Cluster ソルバーのこの経路は、解いた固有ベクトルを全てのランクの `EVec1` に配ります（`ClusterCol_DistributeDenseEvec`）。ホストの経路と同じ配列なので、M4 の収集の呼び出しがそのまま働きます。Band ソルバーでは、`BandCol_DenseCollectExactExchange` が、密度行列の計算と同じパネルと重みを `Exact_Exchange_Collect_K` に渡します。パネルは、最初の MaxN 個の状態の成分を基底ごとに並べた配列です。デバイスにあるパネルは、ホストに写してから渡します。この関数を呼ぶ場所は 2 か所で、上の 2 つの経路に対応します。

ハイブリッド汎関数が有効な間、2 つのソルバーは FP32 で解いて精製する方式に入りません（`ClusterCol_RefineBeginSolve`、`BandCol_RefineBeginStep`）。収集する固有ベクトルは、FP64 の固有値ソルバーの結果です。

### 6.4 交換行列の構築

擬似コードです。添字の扱いは `exx_reference.py` の `build_K_blocks` が正確な定義です。

```text
K(x, y, R) を全て 0 にする
for a in 単位胞の原子:                       # A = (a, 0)
    S    = union over B in Vnb(a) of loc(B)   # 列の site の集合（重複を除く）
    Dloc = D[loc(A), S]                       # ell_A x s_A の行列を、式 (5) で集める
    Tst  = [T_a[0] | T_a[1] | ...]            # (nP, u_a, ell_A)
    G    = Tst @ Dloc                         # (nP, u_a, s_A)        大きい GEMM
    Da   = Dloc の先頭 u_a 行                 # D[A, S]
    Rm   = zeros(nP, u_a, s_A)
    for B in Vnb(a):                          # B = (b, Rb)
        W  = einsum('pq,qul->pul', V_a[B], Tst_b)       # (nP, u_b, ell_B)
        K(A, B)        += sum_P G[:, :, loc(B)] @ W^T    # (i)
        K(A, loc(B))   += sum_P G[:, :, B] @ W           # (ii)
        K(loc(A), B)   += sum_P Tst^T (Da[:, loc(B)] @ W^T)   # (iii)
        Rm[:, :, loc(B)] += Da[:, B] @ W                 # (iv) の括弧
    K(loc(A), S) += Tst^T @ Rm                # (iv)                  大きい GEMM
全ての K を -alpha 倍する
```

`K(X, Y) +=` は、行の site と列の site の対ごとに、ブロック $K_{xy}(\mathbf R_y-\mathbf R_x)$ へ加算する操作です。

Γ 点だけの計算では、site を原子に折り畳んで密行列の上で同じ計算をします（`build_K_gamma`）。 $D$ と $K$ は $N\times N$ の実対称行列です。単位胞が近傍の範囲より小さいと、同じ原子の周期像が 1 つの近傍リストに複数回現れます。T ブロックと V ブロックは周期像ごとに別の値を持ち、行列への加算は同じ添字に重ねます。

M3 の実装（`modules/exx/src/exchange.cpp`）は、擬似コードと次の 3 点が違います。結果は丸め誤差の範囲で同じです。

- (iii) の括弧の中身を `Rm` の $B$ の列に足し、最後の `Tst^T @ Rm` で (iii) と (iv) を一度に変換します。行列積は、中心の原子 1 個につき 2 回と、V 近傍 1 個につき 5 回です。
- 係数 $-\alpha$ は、寄与を $K$ に足すときに掛けます。原子ごとの寄与を別々のランクで計算し、後から足せます。
- Γ 点の形式では、折り畳みを行列積の前に済ませます。中心の原子の寄与は、V ブロックについても T ブロックの列についても線形で、同じ原子の周期像は同じ軌道を指すためです。V ブロックは周期像について足し合わせ、T ブロックの列は同じ軌道に属するものを足し合わせます。行列積の回数は、V 近傍の site の数ではなく、V 近傍に現れる原子の数で決まります。近傍リストに周期像のない原子では、保存したブロックをそのまま使います。site の形式では、周期像ごとに行き先のブロックが違うので、まとめられません。

M5 で、site の形式の $D$ と $K$ を 1 本の配列に並べる形を加えました（`exchange.hpp` の `BlockLayout`）。一覧に入れるのは、全ての中心の原子について、行の site が $\mathrm{loc}(A)$ に、列の site が $S_A$ にあるブロックです。キー $(x,y,\mathbf R)$ の昇順に並べ、ブロックごとに配列の中の位置を決めます。一覧は近傍リストだけで決まるので、どのランクでも同じ並びになり、ランクごとの配列を要素ごとに足せます。一覧にないブロックの $K$ は 0 です。 $K_{xy}(\mathbf R)$ のキーがあれば、 $K_{yx}(-\mathbf R)$ のキーもあります。構築の後で、この 2 つのブロックの対応する要素を平均に置き換えます。平均する前の差の最大が、 $K(\mathbf k)$ がエルミートであることの目安です。

$E_x$ は式 (4) で、 $\delta$ は式 (15) で計算します。どちらも保存したブロックの上の和です。

### 6.5 ハミルトニアンへの加算

$K^{\sigma}(\mathbf k)$ は、近傍リストの形式の $H$ には入れません。 $K$ の到達距離が FNAN より長いためです。既存の関数が密行列を組み立てた直後に、式 (14) の和を加えます。

| ソルバー | 密行列の配置 | 組み立てる場所 |
|---|---|---|
| Band、デバイス上 | `ctx->d_H`（ $n\times n$ の複素行列） | `BandCol_ConstructDenseCsHs_OpenACC`（`Band_DFT_Col.c:3236`） |
| Band、ホスト上の全体行列 | `state->host_h` | `Construct_Band_CsHs` の前半（`Band_DFT_Col.c:6846-6878`） |
| Band、ブロック巡回配置 | `Hs`（ScaLAPACK の局所配列） | `Construct_Band_CsHs` の後半（`Band_DFT_Col.c:6893-6922`）。局所添字の計算は `Band_DFT_Col.c:1501-1529` と同じです。 |
| Cluster、ホスト上 | `Hs`（ブロック巡回配置の実行列） | `Hamiltonian_Cluster_Hs` の呼び出し（`Cluster_DFT_Col.c:2880`、`2884`、`3275`、`1961`）。局所添字の計算は `Hamiltonian_Cluster_Hs.c` に合わせます。 |
| Cluster、デバイス上 | `d_H`（ $n\times n$ の実行列） | `ClusterCol_BuildDeviceDenseFromPacked`（`Cluster_DFT_Col.c:7074`） |

`Construct_Band_CsHs` の呼び出しは `Band_DFT_Col.c:4439` と `4582` の 2 か所です。 $\mathbf k$ と $-\mathbf k$ の組は片方だけを解いているので、 $K(\mathbf k)$ も解く側の k 点だけで足ります。

M5 の実装は、Band のブロック巡回配置の経路だけです。`Construct_Band_CsHs` の 2 か所の呼び出しの直後で、`Exact_Exchange_Add_Band_Hs` が局所配列 `Hs` に式 (14) の和を足します。ハイブリッド汎関数の計算は、CUDA 版でも表の上の 2 つの経路（デバイス上と、ホスト上の全体行列）に入りません。Cluster と同じ扱いです（8 節）。CUDA 版での実行は確かめていません。デバイス上の経路は M6 です。

M6 の実機なしの段階では、表の経路を増やしませんでした。残りの 3 行（Band のデバイス上、Band のホスト上の全体行列、Cluster のデバイス上）は、デバイスの密行列ソルバーの経路です。実機の段階で、このうちデバイス上の 2 行に加算を書きました（`06105f2`、`1766b6f`）。実機では実行していません（9 節の M6）。

| 表の行 | M6 の実装 |
|---|---|
| Band、デバイス上 | `BandCol_DenseAddExactExchange` を、`Construct_Band_CsHs` の直後に呼びます。場所は 3 か所です。全てのランクが k 点を 1 つずつ受け持つ計算のループ、それ以外の計算の最初のループ、 $H(\mathbf k)$ を作り直す 2 番目のループです。 |
| Cluster、デバイス上 | `ClusterCol_DenseAddExactExchange` を、密行列を持つランクが `d_H` を作った直後に呼びます。場所は 2 か所です。スピンごとに 1 ランクが解く方式と、1 ランクが 2 つのスピンを順に解く方式です。 |
| Band、ホスト上の全体行列 | 書いていません。MAGMA で解く経路です。ハイブリッド汎関数の計算は、この経路に入りません。 |

2 つの関数の手順は同じです。ホストで $n\times n$ の配列を 0 にし、`Exact_Exchange_Add_Band_Hs`（Cluster は `Exact_Exchange_Add_Cluster_Hs`）で式 (14) の和を作ります。この関数には、1 つのプロセスが行列の全体を持つ配置を渡します。1×1 のグリッドで、ブロックの大きさは $n$ です。次に、配列をデバイスの作業領域 `d_tmp` に写し、OpenACC のループで `d_H` に足します。`d_tmp` は、 $H$ の組み立てと対角化の間に中身が読まれない領域です。対角化の最初の変換が、この領域を上書きします。デバイスのメモリーの使用量は増えません。

Band のデバイス上の密行列は、列を先に並べたエルミート行列の全体です。`BandCol_ConstructDenseCsHs_OpenACC` は、上三角の要素と、その複素共役を下三角に書きます。`Exact_Exchange_Add_Band_Hs` が作る配列も同じ並びです。

ハイブリッド汎関数の計算がデバイスの経路に入るのは、1 台のデバイスの容量の判定が通った場合です（`BandCol_GpuDenseFits`、`ClusterCol_GpuDiagFits`）。判定が通らない場合と、MAGMA を使う設定の場合は、CPU の固有値ソルバーで解きます。Band はブロック巡回配置の経路、Cluster はホスト上の経路です。どちらも、M4 と M5 で $K$ の加算を試験しています。既存のコードは、判定が通らない計算を、条件が合えば MAGMA の経路に回します。Cluster ソルバーには、複数のノードに分ける gpusolver2 の経路もあります。ハイブリッド汎関数の計算は、この 2 つに進みません。MAGMA の経路は、行列 1 個に 2 台から 8 台の GPU を求めます（`source/openmx_mgpu_runtime.cu` の `openmx_mgpu_host_precheck`）。GPU が 1 台の計算機では試験できないので、加算を書いていません（14 節）。

### 6.6 外側反復

- 判定は、既存の収束判定（`DFT.c:1424-1427`）と、その後の停止条件の検査（FP32 のステップで止まらないための検査、`DFT.c:1436-1505`）が全て済んだ後に置きます。`po==1` のときだけ実行します。
- 外側反復を続ける場合は `po=0` に戻し、混合を始め直します。既存のコードは `SCF_iter_shift` で混合のステップ数の起点をずらしています（`DFT.c:1623`、`1629`、`1644`）。M4 は同じ変数を使い、ずらす量を混合の方式ごとに決めました。下の「混合の始め直し」に書きます。
- 外側反復の回数が `scf.exx.outer.maxIter` を超えたら、収束していない旨を表示して既存の未収束の扱いに合わせます。
- `SCF_iter` は外側反復をまたいで増え続けます。ループの上限 `scf.maxIter`（`DFT.c:1903`）の既定値は 40 です（`Input_std.c:3215`）。M4 の実装は、上限を内側の SCF ごとに数えます。下の「`scf.maxIter` の数え方」に書きます。
- $E_x$ は最後に作った $K$ と、その構築に使った $D$ から計算した値を使います。

M4 の実装（`source/DFT.c`）の内容です。

**混合の始め直し。** $K$ を作ると、内側の SCF が解く問題が変わります。外挿する混合（DIIS）は、過去のステップの入力と残差の組を履歴に持ちます。前の $K$ で測った組は、前の $K$ で収束した密度を「残差がほぼ 0 の解」として伝えます。この組が外挿に入ると、密度が元の値へ引き戻されます。混合の関数はステップ数から動作を選ぶので、 $K$ を作ったステップで `SCF_iter_shift` を書き換え、ステップ数を戻します。戻す先は `Exact_Exchange_Mixing_Offset`（`source/Exact_Exchange.c`）が決めます。

$K$ を作った後の最初の混合は、まだ前の $K$ の残差を見ます。密度の混合（RMM-DIIS、GR-Pulay、RMM-ADIIS、RMM-DIISK）では、 $K$ を作ったステップの終わりの混合です。ハミルトニアンの混合（RMM-DIISH）とポテンシャルの混合（RMM-DIISV）では、次のステップの対角化の前の混合です。この混合に次の表のステップ数を与えると、それより後の外挿は、新しい $K$ で測った組だけを使います。`StartPulay` は `scf.Mixing.StartPulay` の値です。

| 方式 | 最初の混合のステップ数 | その後の動作 |
|---|---|---|
| RMM-DIIS、GR-Pulay、RMM-ADIIS | `StartPulay - 2` | 次が単純混合で、ステップ数が `StartPulay` になると履歴を作り直します。 |
| RMM-DIISK、RMM-DIISV（`StartPulay` が 7 以上） | `StartPulay - 2` | 次が Kerker 混合で、ステップ数 `StartPulay` の外挿は直前の 2 組を使います。 |
| RMM-DIISK、RMM-DIISV（`StartPulay` が 6 以下） | 1 | SCF の初めと同じ順序です。ステップ数が 6 以下の外挿は、ステップ数 2 以降の組を全て使います。 |
| RMM-DIISH | 1 | SCF の初めと同じ順序です。外挿は、ステップ数 2 以降の組を履歴の長さまで使います。 |
| Simple、Kerker | 変えません | 残差のベクトルを履歴に持ちません。停止条件には、この 2 つの方式だけの値があります（下）。 |

求めた値が負になるときは、ステップ数を変えません。 $K$ を作ったステップがまだ単純混合や Kerker 混合の段階にあり、後の外挿が前の $K$ の組に届かないためです。RMM-DIIS、GR-Pulay、RMM-ADIIS で `StartPulay` が 2 以下のときも変えません（RMM-DIIS は、入力の検査が 4 以上を求めます）。この表は、`Mixing_DM.c`、`DIIS_Mixing_DM.c`、`GR_Pulay_DM.c`、`ADIIS_Mixing_DM.c`、`DIIS_Mixing_Rhok.c`、`Mixing_H.c`、`Mixing_V.c` を読んで作りました。実行して確かめた範囲は、進捗ファイルに書きます。

ステップ数が 1 の混合は、既存のコードが残差ノルムを 1 に置き換えます。そのステップの `NormRD` の表示は 1 になります。

**内側の SCF の停止条件。** 既存の停止条件は、連続する 2 ステップのバンドエネルギーの差と、緩い残差の条件です（`dUele < scf.criterion` と `NormRD^2 < 10 scf.criterion`）。混合が密度をほとんど動かさないと、新しい $K$ に応答していない密度でこの条件を通ります。 $K$ がある内側の SCF には、次の条件を加えました（`Exact_Exchange_Inner_Stop_Allowed`）。

- $K$ を作ってから 2 ステップ目までは止めません。2 ステップ目の停止判定が見る残差ノルムは、新しい $K$ で測った最初の残差です。この値を基準 $R_0$ にします。
- 3 ステップ目以降は、残差ノルムが $0.1R_0$ 以下か、`scf.criterion` 以下のときだけ停止を認めます。

1 つ目の値は、密度が新しい $K$ に応答したことの判定です。2 つ目の値は、 $K$ の変化が密度を乱さなかった場合の出口です。 $R_0$ が初めから小さいと、残差を 10 分の 1 にできません。0.1 と、`scf.criterion` をそのまま使うことは、M4 の実測から決めた値です（13 節の 21）。残差ノルムの単位は混合の方式で違うので、2 つ目の値は目安です。

単純混合と Kerker 混合には、3 つ目の値 $\sqrt{10^{-9}}\approx3.2\times10^{-5}$ があります。残差ノルムがこの値以下なら、停止を認めます。この 2 つの方式は、直前の 2 ステップの残差ノルムから混合の重みを決めます（`core/src/mixing_policy.cpp` の `OPENMX_MIXING_DENSITY_RATIO`）。残差が減ったステップでは、1 つ前の残差ノルムの 2 乗を、2 乗の減った量で割った値を重みに掛けます。割る数には下限 1e-9 があります。残差ノルムの 2 乗が 1e-9 を下回ると、この比は 1 より小さくなります。残差が増えたステップでも重みは減るので、重みはステップごとに減り、`scf.Min.Mixing.Weight` に達します。その後の残差は、ほとんど減りません。ハイブリッド汎関数でない計算の SCF も、この状態で終わります。3 つ目の値がないと、 $R_0$ が既にこの水準にある内側の SCF は、 $0.1R_0$ に届きません（13 節の 25）。下限の 1e-9 は既存のコードの値で、`core/tests/test_mixing_policy.cpp` が確かめています。この値を変えるときは、`Exact_Exchange_Inner_Stop_Allowed` の値も変えます。

**未収束の扱い。** 最初の PBE の SCF が `scf.maxIter` 回で収束しない場合は、 $K$ を作らずに終わります。エネルギーは PBE の値で、その旨を警告します。 $K$ がある内側の SCF が収束しない場合と、外側反復が `scf.exx.outer.maxIter` 回で収束しない場合も、理由を表示して終わります。どの場合も `scf_convergence_flag` は 0 です。

**`scf.maxIter` の数え方。** ループの上限を「`scf.maxIter` + その内側の SCF が始まった `SCF_iter`」にしています。合計で数えると、既定値の 40 回では足りません。M4 の実測では、既定の `scf.criterion` で、合計が 36 回から 51 回でした（メタンと水分子、2 通りの基底。進捗ファイル）。この数え方は M4 の案で、利用者の確認を待っています。

**バンドエネルギー。** $K$ がある間の `Uele` は、占有数を掛けた固有値の和です。ソルバーが返すもう 1 つの値（近傍リストの形式の $H$ とのトレース）は、 $K$ を含みません。固有値の和は $\mathrm{Tr}(DK)=2E_x$ を含むので、PBE の `Uele` とは比べられません。全エネルギーは成分の和で、`Uele` を使いません。

**密度行列の照合。** $K$ を作るたびに、集めた $D$ の FNAN の範囲を、ソルバーが作った密度行列と比べます。差が 1e-8 以上なら、計算を止めます。M4 の試験の全ての実行で、差は 3e-16 以下でした。site の形式では、ソルバーの密度行列の要素のうち、一覧にあるブロックの分を比べます。M5 の試験の全ての実行では、差は 7e-16 以下でした。

### 6.7 MPI

第 1 版では次の単純な形にします。

- $D$ は全ランクが同じ内容を持ちます。k 点をランクに配る場合は、 $D$ の寄与を `MPI_Allreduce` で集めます。
- 中心の原子 $a$ をランクに配ります。各ランクは自分の原子の寄与だけを計算し、 $K$ を `MPI_Allreduce` で集めます。
- 1 回の `MPI_Allreduce` に渡す要素数は `int` の範囲に収めます。超える場合は分割します。

大きい系では、ランクごとに $D$ と $K$ の複製を持つメモリーが問題になります（12 節）。共有メモリーの利用は、実測の後に検討します。

M5 の実装は、この形です。Band ソルバーは、k 点をランクに配る場合と、1 つの k 点の行列を複数のランクで分ける場合があります。前の場合は、各ランクが自分の k 点の寄与を $D$ に足します。後の場合は、各ランクが自分の持つ状態の寄与を足します。どちらも、収集の終わりに和を取ります。構築は中心の原子をランクに配るので、ランクが原子より多いと、構築を受け持たないランクが出ます（13 節の 29）。

## 7. モジュール構成とデータ構造

### 7.1 `modules/exx` の構成

`modules/xc` と同じ形の、単独でビルドできる C++20 の静的ライブラリー（`OpenMX::exx`）にします。依存は `OpenMX::basis`（球ベッセル関数、Gaunt 係数、実球面調和関数、積分則）だけです。

```text
modules/exx/
  CMakeLists.txt  README.md  LICENSE  cmake/OpenMXExxConfig.cmake.in
  include/openmx/exx/
    kernel.hpp          # 5.6 節のカーネル
    abf.hpp             # PAO の積と主成分分析
    two_center.hpp      # 式 (11)(12)(13)
    lri.hpp             # 式 (7)、T ブロック
    system.hpp          # site、近傍リスト、ブロックの格納
    exchange.hpp        # 6.4 節、式 (4)(15)(16)(17)
    dense_backend.hpp   # 行列積の関数表
    exx.h               # C API
  src/  tests/  examples/
```

`core/CMakeLists.txt` には、`modules/xc` と同じ形で追加します（`add_subdirectory` と `$<TARGET_OBJECTS:...>`、89〜96 行と `add_library` の行）。本体からは `source/openmx_exx.h`（`source/openmx_xc.h` と同じ転送ヘッダー）で参照し、依存関係を `source/openmx_core.mk` に書きます。CI のジョブも `xc` と同じ形です。

試験の名前は `exx-<部品>-<場合>` の形にし、ラベルは `unit;exx` にします（`modules/basis/tests/CMakeLists.txt` の書き方に合わせます）。

M2 で作った例題は 2 つです。`examples/kernel` はカーネルの値を表にします。`examples/lri` は `.pao` ファイルを読み、ABF の個数と LRI の精度の目安を出します。M2 の表にない試験も 2 つ加えました。`exx-system-neighbours` は近傍リストを総当たりの探索と比べ、`exx-system-blocks` は T ブロックと V ブロックの 2 通りの見方が一致することを確かめます。重なり行列の照合は `exx-two-center-overlap` という名前です。

M3 で加えた試験は 7 つです。`exx-dense-backend`、`exx-exchange-fixture`、`exx-exchange-gamma`、`exx-exchange-gradient`、`exx-exchange-backend` の 5 つが、M3 の表の確認に当たります。表にない試験は 2 つです。`exx-exchange-system` は、PAO から作ったブロックで $K$ を別の式と比べ、C の入口の結果が C++ の入口とビット単位で同じことも確かめます。`exx-c-api` は、C のプログラムから C の入口を呼びます。この試験のために、モジュールのビルドには C のコンパイラーも要るようになりました。

LAPACK に依存できないので、小さい対称行列の固有値分解（主成分分析）とコレスキー分解（式 (7)）はモジュールの中に実装します。行列の次数は、元素あたりの ABF の数の 2 倍までです。

### 7.2 行列積の関数表

```c
typedef struct {
  /* C = alpha * op(A) * op(B) + beta * C、列優先。ホストかデバイスかは実装側が決める */
  int (*dgemm)(void *context, char transa, char transb, int m, int n, int k, double alpha,
               const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc);
  void *context;
} OpenMXExxDenseBackend;
```

表が与えられない場合に使う、素朴なループの実装をライブラリーの中に持ちます。単体試験が使うのはこの実装です。本体は CPU ビルドで BLAS、GPU ビルドで cuBLAS か GEMMul8 を渡します。

M3 の実装では、素朴なループの実装を `openmx_exx_reference_dgemm` として公開しました。引数は表の `dgemm` と同じで、DGEMM が拒否する引数には 1 を返します。C++ の側では、`DenseBackend` が表を包みます。モジュールの配列は行優先なので、`DenseBackend` は 3 通りの積（ $AB$ 、 $AB^{\mathsf T}$ 、 $A^{\mathsf T}B$ ）を、因子の順序と転置の指定を入れ替えて列優先の `dgemm` に渡します。配列の複製は作りません。次元が `int` の範囲を超えると例外を投げ、`dgemm` が 0 以外を返すと構築を止めます。

### 7.3 データ構造

| 量 | 形 | 所有者 | 寿命 |
|---|---|---|---|
| ABF | 元素ごとの動径関数の表 | EXX の状態 | 計算全体 |
| 近傍リスト | 原子ごとの site の列（原子番号と整数 3 個） | EXX の状態 | MD の 1 ステップ |
| T ブロック | 原子ごとに 1 つの配列 `t[a]`（ $n_{P,a}u_a\times\ell_a$ ）。C 近傍の順に横に並べ、`c_offset[a][i]` が $i$ 番目の近傍の最初の列です（M3 で改めました）。 | EXX の状態 | MD の 1 ステップ |
| V ブロック | `V[a][i]` は $n_{P,a}\times n_{P,b}$ | EXX の状態 | MD の 1 ステップ |
| $D$ | Γ 点では $N\times N$ 。k 点では、ブロック $D_{xy}(\mathbf R)$ を一覧の順に並べた 1 本の配列（M5 で改めました） | EXX の状態 | 外側反復 1 回 |
| $K$ | Γ 点では $N\times N$ 。k 点では、 $D$ と同じ一覧の順に並べた配列 | EXX の状態 | 新旧の 2 世代 |

EXX の状態は `OpenMXCoreContext`（`core/src/calculation_context.hpp:29`）が所有する 1 つのオブジェクトにまとめます。`DFT.c` では既存の `*_Bind_Resources(core)`（`DFT.c:484-492`）と同じ形で取り出し、関数に引数で渡します。入力キーワードの値の置き場所も、このオブジェクトです。

### 7.4 本体との契約

関数名は例です。次の 4 つの役割を分けることが設計です。

```c
/* 構造ごとの準備。MD の 1 ステップに 1 回 */
int openmx_exx_prepare(OpenMXExxState *exx, const OpenMXExxGeometry *geometry, MPI_Comm comm,
                       char *error, size_t capacity);
/* 密度行列の収集。ソルバーがスピンと k 点ごとに呼ぶ */
int openmx_exx_density_add(OpenMXExxState *exx, int spin, const double k[3], double weight,
                           int n, int states, const void *vectors, int stride, int complex_vectors,
                           const double *occupation);
/* 交換行列の構築。E_x と delta を返す */
int openmx_exx_build(OpenMXExxState *exx, MPI_Comm comm, double *energy, double *delta,
                     char *error, size_t capacity);
/* 密行列への加算。配置ごとに入口を分ける */
int openmx_exx_add_dense(const OpenMXExxState *exx, int spin, const double k[3], int n,
                         void *h, int ldh, int complex_matrix, int on_device);
```

`OpenMXExxGeometry` は、格子ベクトル、原子の位置と元素、元素ごとの PAO の動径関数とカットオフ半径を、ポインターと個数で渡す構造体です。モジュールの中から本体の大域変数は読みません。PAO の動径関数、動径の格子、実球面調和関数の変換表は、`DFT.c` がソルバーに渡している `OpenMXAtomicViews`（`core/include/openmx_core_atomic.h:27`）から取り出せます。

この 4 つは本体の側（`source/`）に置く結合層の関数です。MPI を使うのはこの層だけです。`modules/exx` の関数は、1 つのランクの中の計算（準備、原子 1 個分の構築、加算）を受け持ちます。

M3 で `modules/exx` に入れた C の入口（`exx.h`）は、結合層が呼ぶ部品です。状態の生成と破棄、構造の準備（`openmx_exx_prepare`）、大きさと式 (7) の報告の問い合わせ、原子の範囲を指定した Γ 点の形式の加算（`openmx_exx_add_gamma`）、式 (4) と (15) があります。準備は MPI を使わず、元素ごとの表を状態に残して、PAO、格子、しきい値が変わったときだけ作り直します。site の形式の入口は、k 点の計算を組み込む M5 で加えました（下）。

M4 で、結合層を `source/Exact_Exchange.c`（ヘッダーは `source/openmx_exact_exchange.h`）に作りました。4 つの役割に当たる関数は、`Exact_Exchange_Prepare`、`Exact_Exchange_Collect_Gamma`、`Exact_Exchange_Build`、`Exact_Exchange_Add_Cluster_Hs` です。外側反復の判定（`Exact_Exchange_Outer_Continue`）、混合のステップ数を戻す量（`Exact_Exchange_Mixing_Offset`）、内側の SCF の停止条件（`Exact_Exchange_Inner_Stop_Allowed`）も、この層に置きました（6.6 節）。状態は計算ごとに 1 つで、ソルバーの状態と同じ仕組み（`openmx_core_native_resources_state`）で持ちます。

- 層の関数は大域変数を読みません。入力キーワードの値は `Exact_Exchange_Configure` で、構造の配列（原子の数、元素、座標、格子ベクトル）は `Exact_Exchange_Bind_Structure` で、入力を読む関数（`Input_std`）が渡します。層は、準備のたびにその配列を読みます。`DFT` 関数は座標の配列 `Gxyz` を読んでいません。準備の呼び出しで座標を渡す形にすると、大域状態のゲートの組が 1 つ増えます。
- 密度行列と交換行列の総和は、ランク 0 に集めてから全ランクに配ります。全ランクが同じ行列を持つので、外側反復の判定がランクによって変わりません。
- 行列積は、本体の `DGEMM` を関数表で `modules/exx` に渡します。
- HIP 版のビルドには入れていません（M8）。`Input_std.c` は HIP 版と共有なので、層を呼ぶ部分を `#if !OPENMX_GPU_HIP` で囲んでいます。

M5 で加えた入口です。`modules/exx` の C の入口は 7 つあります。`openmx_exx_prepare_blocks` がブロックの一覧を作ります。`openmx_exx_block_count`、`openmx_exx_block_elements`、`openmx_exx_block`、`openmx_exx_find_block` は、一覧の内容を返します。`openmx_exx_add_blocks` は、原子の範囲を指定して $K$ の配列に寄与を足します。`openmx_exx_symmetrize_blocks` は、対になる要素を平均に置き換え、その前の差の最大を返します。式 (4) と (15) は、Γ 点の形式と同じ関数（配列の内積）で計算します。

結合層では、`Exact_Exchange_Prepare` に行列の形を選ぶ引数を加えました。`DFT` 関数は、Band ソルバーの計算で site の形式を、Cluster ソルバーの計算で Γ 点の形式を選びます。k 点の収集は `Exact_Exchange_Collect_K`、 $H(\mathbf k)$ への加算は `Exact_Exchange_Add_Band_Hs` です。`Exact_Exchange_Density_Deviation` には、近傍の原子がどのセルにあるかを表す 2 つの配列（本体の `ncn` と `atv_ijk`）を引数で渡します。`Exact_Exchange_Asymmetry` は、最後の構築で平均する前に測った $K$ の差の最大を返します。Γ 点の形式の関数は、site の形式の状態で呼ばれると計算を止めます。逆の場合も同じです。

M6 で、行列積をデバイスに送る部品を結合層の下に置きました。`source/Exact_Exchange_Device.c` の `Exact_Exchange_Device_Dgemm` は、DGEMM の引数に状態 `OpenMXExactExchangeDevice` を加えた関数です（11 節）。この状態は結合層の状態の中にあり、cuBLAS のハンドル、2 つの限度、積を数える変数を持ちます。ハンドルは、計算の終わりに `Exact_Exchange_Device_Release` が返します。関数表の `context` は、構築 1 回分の集計を指します。集計の中身は、積の数、最大の積の大きさ、デバイスで使った時間です。`Exact_Exchange_Device_Use` は、デバイスで計算した積の数などを、全てのプロセスについて合計します。中で `MPI_Allreduce` を呼ぶので、全てのプロセスが同じ場所で呼ぶ必要があります。部品の関数は大域変数を読みません。デバイスが使えるかどうかは、既存の `gpu_rank_device_usable` に問い合わせます。

M6 の実機の段階で、結合層に 3 つの関数を加えました。密行列をデバイスに持つソルバーのためのものです。`Exact_Exchange_Device_Addition` は、デバイスでの加算 1 回と、その転送と加算の秒数を記録します。秒数は、加算の時間に入ります。`Exact_Exchange_Device_Additions` は、加算の回数を全てのプロセスについて合計します。中で `MPI_Allreduce` を呼びます。`Exact_Exchange_Collecting` は、収集が開いている間だけ 1 を返します。ソルバーは、固有ベクトルをデバイスから取り出す前に、この関数で収集の有無を確かめます。3 つの関数は大域変数を読みません。

### 7.5 `modules/xc` への追加

ωPBE の短距離交換を、純粋な関数として `modules/xc` に加えます。既存の `openmx_xc_kernel_pbe`（`kernels.h:285`）の演算は変えません。加える関数は 2 つです。1 つは短距離部分だけを返す関数で、試験はこの出力をフィクスチャー（10.4 節）と比べます。もう 1 つはハイブリッド用の関数です。この関数は PBE の関数を呼び、交換の出力（`exc[0]`、`dexdd`、`dexdgd`）から短距離部分の $\alpha$ 倍を引きます。必要な特殊関数は $e^{x}E_1(x)$ と $\mathrm{erfcx}(x)=e^{x^2}\mathrm{erfc}(x)$ の 2 つで、どちらもこのモジュールの中に実装します。

PBE0（M8）では短距離部分の関数を使いません。式 (2) のとおり、PBE の交換の出力を $1-\alpha$ 倍します。 $\omega=0$ にした短距離部分の関数は PBE の交換と一致しないので、代わりには使えません（M1）。

## 8. 既存コードへの組み込み

| 作業 | 場所 | 内容 |
|---|---|---|
| 入力 | `source/Input_std.c:990-992` | `scf.XcType` に `HSE06` と `PBE0` を加えます。内部は `XC_switch=4` とし、種別と混合比と $\omega$ を EXX の状態に入れます。`HSE`（5）と `NLX`（6）の指定はエラーです（M0 で実装済み、`994-1001` 行）。 |
| 旧い厳密交換のコード | `source/Allocate_Arrays.c:770-778`、`source/Free_Arrays.c:1836-1842` | 試作用の入力（4 つのキーワードと `Allocate_Arrays(15)` の呼び出し）は、M0 で `Input_std.c` から削除しました（`bcf5222`）。変数の定義、`case 15` の確保、解放、`Expand_ReCoulomb_Kernel.c`、`source/exx*` は残っています。動作には影響しません。削除するときは利用者に確認します（1.3 節）。 |
| キーワード | `source/Input_std.c` | 下の表のキーワードを読みます。2.2 節の拒否もここに置きます。 |
| 準局所の交換 | `source/Set_XC_Grid.c:1274`、`modules/xc` | CPU の経路で、ハイブリッド用の関数を呼びます。混合比と $\omega$ は `Set_XC_Grid` の引数に加えます。呼び出しは 7 か所です（`Set_Vpot.c:97`、`Total_Energy.c:1836`、`1993`、`4444`、`Stress.c:2413`、`OutData.c:580`、`OutData_Binary.c:511`）。 |
| XC の GPU 経路 | `source/Set_XC_Grid.c:654` | ハイブリッドが有効な間は GPU の経路に入らず、ホストのループを使います。GPU への対応は M6 です。9 節の M8 の一覧にも「GPU での XC の計算」があり、時期が食い違っています（14 節）。 |
| 密行列への加算 | 6.5 節の表 | $K^{\sigma}(\mathbf k)$ を足します。 |
| 密度行列の収集 | `source/Band_DFT_Col.c:1949`、`2038`、`source/Cluster_DFT_Col.c:4121` | 密度行列の計算に渡している固有ベクトルと占有数を、EXX の状態にも渡します。 |
| 外側反復 | `source/DFT.c:1424-1505` | 6.6 節 |
| 全エネルギー | `core/include/openmx_core_energy.h`、`source/DFT.c:2822-2842`、`2869` 付近 | エネルギーの成分に厳密交換を加え、合計と出力に入れます。 |
| 力 | `source/Force.c:3055` | M7 |
| ビルド | `source/Makefile:309`、`source/openmx_core.mk`、`core/CMakeLists.txt` | 本体に新しい翻訳単位を足すときは `OBJS` と規則を追加します。CPU ビルドは `source/Makefile` から一覧を作ります（`cmake/OpenMXCPU.cmake:172-178`）。HIP 版が使うファイルの一覧は `source/backends/hip/sources.json` です。 |

M4 で組み込んだ範囲です。場所は付録 A の「M4 で加えたコード」に検索語で示します。

| 作業 | M4 の実装 |
|---|---|
| 入力 | `HSE06` を受け付け、下の表の `scf.exx.*` を読みます。`PBE0` は拒否します（M8）。2.2 節の拒否は、入力ファイルを閉じる直前に置きました。 |
| 準局所の交換 | `Set_Hamiltonian`、`Set_Vpot`、`Total_Energy` に混合比と $\omega$ の引数を加え、`DFT` 関数が EXX の状態の値を渡します。混合比は、最初の $K$ ができるまで 0 です。`Stress.c`、`OutData.c`、`OutData_Binary.c` の呼び出しと、`Total_Energy.c` の `Calc_Edc` と `Energy_Decomposition_CWF` の中の呼び出しは、0 を渡したままです。応力、Krylov ソルバー、エネルギーの分解は 2.2 節で拒否しています。`level.of.fileout 2` で書き出す `.vxc0` と `.vxc1` は、PBE のポテンシャルになります。 |
| 密行列への加算 | Cluster のホスト上の経路だけです。`Hamiltonian_Cluster_Hs` の直後で、ブロック巡回配置の `Hs` に $K^{\sigma}$ を足します。ハイブリッド汎関数の計算は、CUDA 版でもデバイス上の密行列の経路に入らず、この経路を通ります。CUDA 版での実行は確かめていません。デバイス上の経路は M6 です。 |
| 密度行列の収集 | Cluster の経路だけです。収集が有効な間、ソルバーは固有ベクトルと占有数を結合層に渡します。 |
| 外側反復 | 6.6 節 |
| 全エネルギー | 成分 `exact_exchange` を加え、合計、標準出力、`.out` の表（`Uexx.`）に入れました。ハイブリッド汎関数を指定しない計算の出力は変わりません。 |
| ビルド | `source/Exact_Exchange.c` を `OBJS` に加えました。CPU 版と CUDA 版に入り、HIP 版には入りません。 |

M5 で組み込んだ範囲です。場所は付録 A の「M5 で加えたコード」に検索語で示します。

| 作業 | M5 の実装 |
|---|---|
| 入力 | Band ソルバーの拒否を外し、2.2 節の表にキーワードを 4 つ加えました。 |
| 構造ごとの準備 | `DFT` 関数が、ソルバーの種類から行列の形を選んで `Exact_Exchange_Prepare` に渡します。 |
| 密行列への加算 | Band のホスト上のブロック巡回配置の経路だけです（6.5 節）。 |
| 密度行列の収集 | Band の密度行列を作る 2 つの経路です（6.3 節）。 |
| 外側反復 | Cluster と同じコードです。収集のための解き直しで、Band ソルバーを呼びます。 |
| ビルド | `source/Makefile` の依存関係に `Band_DFT_Col.o` を加えました。新しい翻訳単位はありません。 |

M6 で組み込んだ範囲です。実機の段階で加えた項目には、その旨を書きました。場所は付録 A の「M6 で加えたコード」に検索語で示します。

| 作業 | M6 の実装 |
|---|---|
| 行列積 | 関数表の `dgemm` が、大きい積をデバイスに送ります（11 節）。CPU 版では、全ての積が今までどおり本体の `DGEMM` に行きます。 |
| 入力 | キーワードは加えていません。`Input_std` が、既存の障害注入の環境変数 `OPENMX_GPU_CAPACITY_TEST_RECOVERY` の値を `Exact_Exchange_Configure` に渡します。 |
| 標準出力 | 計算の終わりに、デバイスで計算した積の数、そのプロセスの数、時間、容量の不足でホストに残した積の数を表示します。CPU 版と、デバイスに送る大きさの積がない計算では表示しません。`level.of.stdout 2` では、構築ごとに、ランク 0 の積の数と最大の積の大きさを表示します。 |
| 密行列への加算 | 実機なしの段階では変えていません。実機の段階で、デバイスの密行列ソルバーの 2 つの経路（Band と Cluster）に加えました（6.5 節）。MAGMA の経路と gpusolver2 の経路には加えていません。 |
| 密度行列の収集 | 実機の段階で、デバイスの密行列ソルバーの経路からも収集する形にしました（6.3 節）。 |
| 固有値ソルバーの精度 | 実機の段階で加えた条件です。ハイブリッド汎関数が有効な間、FP32 で解いて精製する方式に入りません。 |
| 標準出力（密行列の経路） | 実機の段階で加えました。計算の終わりに、デバイスで $K$ を足したハミルトニアンの数を表示します。数が 0 の計算では表示しません。容量の判定が通らずに CPU の固有値ソルバーへ切り替わるときは、その旨を 1 行で表示します（11 節）。 |
| XC の GPU 経路 | 変えていません。ハイブリッド汎関数の計算は、CUDA 版でもホストのループを使います。 |
| ビルド | `source/Exact_Exchange_Device.c` を `OBJS` に加えました。CPU 版と CUDA 版に入り、HIP 版には入りません。`tools/check_gpu_compilation.py` の CUDA の一覧にも加えています。 |
| 試験 | `tests/run_exx_device_product.py` が、GPU のライブラリーの模型で部品を試験します。CI では `gpu-contracts` ジョブが実行します。実機の試験は `tools/exx_gpu_checks.py` が実行します（9 節の M6）。 |

追加する入力キーワードの既定値は出発点で、M4 と M5 の実測の後に見直します。M5 の実測値は進捗ファイルにあります。既定値を変えるかどうかは、利用者が決めます。

| キーワード | 既定値 | 意味 |
|---|---|---|
| `scf.XcType` | 変更なし | 選択肢に `HSE06` と `PBE0` を追加 |
| `scf.exx.alpha` | 0.25 | 厳密交換の混合比 |
| `scf.exx.omega` | 0.11 | HSE06 の遮蔽パラメーター（bohr⁻¹） |
| `scf.exx.abf.pca` | 1.0e-4 | ABF の主成分分析の閾値（最大固有値に対する比） |
| `scf.exx.v.cutoff` | 8.0 | V 近傍を決める打ち切り半径（bohr） |
| `scf.exx.outer.maxIter` | 30 | 外側反復の上限 |
| `scf.exx.outer.criterion` | 1.0e-6 | 式 (15) の閾値（Hartree） |
| `scf.exx.coulomb.cutoff` | 自動 | PBE0 の切断半径（bohr）。M8 |

## 9. マイルストーン

### M0. 準備と入力の整理

| 項目 | 内容 |
|---|---|
| 目標 | 実装を始められる環境を作り、未実装の `scf.XcType` を拒否します。 |
| 前提 | なし |
| 操作 | CPU 版をビルドし（下のコマンド）、`python3 tools/exx_reference.py` を実行します。入力の変更は、`scf.XcType` の `HSE` と `NLX` をエラーにする 1 点です。最後に `CLAUDE.md`（1.5 節）と `docs/EXX_PROGRESS.md` を作ります。 |
| 成功の確認方法 | 参照実装が `self-test passed` を表示します。`scf.XcType HSE` の入力はエラーで止まります。`work/input_example/Methane.dat` の全エネルギーは、変更の前後で同じ値です。 |
| 失敗例 | CPU 版のビルドに必要なライブラリーがない場合は、不足を報告して止まります。 |

```sh
# Ubuntu 24.04 での CI と同じ構成（.github/workflows/core.yml の cpu-application）
sudo apt-get install -y --no-install-recommends gfortran libomp-dev openmpi-bin libopenmpi-dev \
  libscalapack-openmpi-dev libopenblas-dev libfftw3-dev autoconf automake libtool libtool-bin m4 pkg-config unzip
cmake -S . -B build/cpu -DOPENMX_GPU_BACKEND=CPU -DOPENMX_CPU_COMPILER=GCC \
  -DOPENMX_CPU_MPI=OPENMPI -DOPENMX_CPU_LINALG=OPENBLAS
cmake --build build/cpu --parallel 4
ctest --test-dir build/cpu --output-on-failure \
  -R '^(core-(grid-assembly|host-simd-boundary)|build-backend-isolation|cpu-build-accelerator-boundary|cpu-numerical-toolchain|environment-registry)'
# 実行例。DATA.PATH の既定値が ../DFT_DATA19 なので work/ の中で実行する
cd work && mpirun -np 1 ./openmx input_example/Methane.dat
```

ビルドが成功すると、実行ファイルは `build/cpu/openmx` にでき、`work/openmx` にも複写されます。モジュールだけの試験は `ctest --test-dir build/exx` のように、モジュールごとのビルドディレクトリーで実行します。

M0 は 2026-11-23 に完了しました。実際のビルドには Docker のコンテナと `build/exx-cpu` を使っています。手順は `docs/EXX_PROGRESS.md` の「ビルドと実行」にあり、以後はそちらを優先します。

### M1. 準局所部分

| 項目 | 内容 |
|---|---|
| 目標 | ωPBE の短距離交換を `modules/xc` に実装し、ハイブリッド用の準局所部分を CPU の経路で計算します。 |
| 前提 | M0 |
| 操作 | 表の下の 5 段階を順に進めます。 |
| 成功の確認方法 | 短距離部分の関数の出力が、フィクスチャー（10.4 節）と許容差の中で一致します。参照実装と Quantum ESPRESSO 7.6 のルーチンの照合は `comparison passed` を表示します。混合比 0 のときのハイブリッド用の関数の出力は、PBE の関数の出力とビット単位で同じです。混合比 0.25 では、PBE の出力から短距離部分の 0.25 倍を引いた値と一致します。既存の `xc-legacy` も通ります。 |
| 失敗例 | 参照実装と Quantum ESPRESSO のルーチンが合わない場合は、実装に進まずに報告して止まります。原因は、参照実装の誤りか、ソースの読み違いのどちらかです。`xc-legacy` が落ちた場合の原因は、既存の PBE の関数の演算を変えたことです。 |

1. `python3 tools/wpbeh_reference.py` を実行し、`self-test passed` の表示を確かめます。次に、`--dump` で作り直したフィクスチャーを、保存したファイルと比べます（相対 1e-15 以内）。
2. Quantum ESPRESSO 7.6 のルーチンをコンパイルして値を作り（「Quantum ESPRESSO との照合」）、`python3 tools/wpbeh_reference.py --compare-qe` で参照実装と比べます。
3. 特殊関数 $e^{x}E_1(x)$ と $\mathrm{erfcx}(x)$ を実装します。mpmath の値（`e1`、`erfc`）と、 $10^{-30}\le x\le10^{8}$ の範囲で相対 1e-14 以内に合わせます。
4. 短距離部分の関数を実装し、フィクスチャーと比べます。計算の形は、参照実装の `double_channel_energy` に合わせます。
5. ハイブリッド用の関数を作り、`Set_XC_Grid` へ引数を加えます（7.5 節、8 節）。

試験の名前は `xc-wpbeh-special`、`xc-wpbeh-fixture`、`xc-hybrid` とし、ラベルは既存の試験と同じ `unit;xc` にします（`modules/xc/tests/CMakeLists.txt`）。

M1 は 2026-11-23 に完了しました。同じ日に利用者の指示を受け、短距離部分の定義を libxc 7.1.2 の形から下の形に書き直しています（付録 C）。

#### 短距離部分の定義

定義は、Quantum ESPRESSO 7.6 の `pbexsr` と `wpbe_analy_erfc_approx_grad` に合わせます [16]。式の全体を書いたファイルが `tools/wpbeh_reference.py` です。

| 項目 | 定義 | 論文で確かめた範囲 |
|---|---|---|
| フェルミ波数 | スピンごとに $k_F=(6\pi^2\rho_s)^{1/3}$ 、 $w=\omega/k_F$ | 遮蔽した交換孔と、増強因子をその積分で決める式は、文献 [8] の式 (8) と (9) です。 |
| 還元勾配 | $s=\lvert\nabla\rho_s\rvert/(2(6\pi^2)^{1/3}\rho_s^{4/3})$ | 同上 |
| $s$ の付け替え | $s>8.3$ で $8.572844-18.796223/s^2$ | 文献 [11] の 2.2.3 節の式で、3 つの数値も一致します。増強因子を局所的な Lieb–Oxford の限界（1.804）以下に保つための付け替えです。 |
| 交換孔の係数 | $A$ から $E$ 、 $H(s)$ の係数、 $F_{c1}$ 、 $F_{c2}$ に、丸めた数値を使います（ $E=-0.051955731$ 、 $F_{c1}=6.4753871$ 、 $F_{c2}=0.47965830$ ）。 | 交換孔の模型の出典は文献 [9] です。この文献は読んでいません。数値は Quantum ESPRESSO のソースから取りました。 |
| 小さい $s$ での $EG(s)$ | $s\le0.08$ では 3 項の多項式、それより上では閉じた式 | Quantum ESPRESSO のソースで確かめました。 |
| 誤差関数 | $w\le14$ では係数 8 個の当てはめ、 $w>14$ では 1 個のガウス関数 $e^{-2x^2}$ | 文献 [11] の 2.2.2 節の当てはめです。論文は係数を小数 9 桁で載せていて、その桁まで一致します。15 桁の値は Quantum ESPRESSO のソースから取りました。 |

Quantum ESPRESSO は、スピンごとの密度を 2 倍、勾配の 2 乗を 4 倍にして非分極の式に渡し、エネルギーを半分にします [16]。上の表はこれをスピンごとの変数で書いたもので、 $\omega$ の値を換算する必要はありません。

この定義では、増強因子 $f(w,s)$ が 3 か所で不連続になります。値は多倍長で評価したものです（`--report`）。

| 場所 | 原因 | $f$ の相対的な跳び |
|---|---|---|
| $s=0.08$ | 多項式と閉じた式の切り替えです。閉じた式では $E+s^2EG(s)$ が $E$ に依存しないので、丸めた $E$ が効くのは多項式の側だけです。 | $w=0$ で 6.9e-9、 $w=1$ で 1.3e-9、 $w=3$ で 4.0e-11 |
| 付け替え前の $s=8.3$ | 付け替えた $s$ が 2.3e-9 だけ跳びます。 | 1.2e-10 以下（ $w$ が 0 から 30 の 9 点） |
| $w=14$ | 誤差関数の当てはめからガウス関数への切り替えです。 | 2.6e-4 以下（ $s$ が 0.003 から 8.5 の 5 点） |

$f(0,s\to0)$ は 1 より 4.8e-9 小さい値です。交換孔の係数が丸めた数値だからです。付け替えた $s$ の上限 8.572844 では、 $f(0,s)$ が 1.8040000 になります（1.804 との差は 3.3e-9）。

実装は、次の 3 点で Quantum ESPRESSO のルーチンと違います。

- 導関数は、付け替えを含むエネルギーの導関数です。Quantum ESPRESSO のルーチンは、 $s>8.3$ のポテンシャルに付け替えの微分（付け替え前の $s$ を $s_0$ として $2\times18.796223/s_0^3$ ）を掛けていません。
- 特殊関数 $e^{x}E_1(x)$ と $\mathrm{erfcx}(x)$ は、全ての引数で倍精度の精度まで計算します。Quantum ESPRESSO のルーチンは、引数が 700 以上になる $w>14$ で近似式を使います。
- 付け替え前の $s$ が 1e-10 より小さいときは、1e-10 で計算します。第 1 項の対数の発散が打ち消し合う点（ $w=0$ かつ $s=0$ ）を避けるための保護で、 $f$ の変化は相対 1e-19 以下です。Quantum ESPRESSO のルーチンは、この点で $f=1$ を返します。

$\omega=0$ にした短距離部分の式を、PBE の交換の代わりに使ってはいけません。増強因子が、 $s\le5$ の範囲で PBE の増強因子と −0.45% から +0.54% 違うからです（5.1 節）。

#### 許容差

フィクスチャーの各行は、スピンごとに相対の許容差 `tol` を持ちます。 $w$ で決まる値です。

| $w$ の範囲 | `tol` | 式を 16 桁で評価したときの丸め誤差 |
|---|---|---|
| $w\le4$ | $\max(10^{-12},\ 10^{-13+1.5w})$ | $w=1$ で 7e-16、 $w=2$ で 8e-14、 $w=4$ で 1e-10 |
| $4<w\le14$ | $10^{-7+0.5(w-4)}$ | $w=6$ で 5e-9、 $w=10$ で 1e-6、 $w=13.9$ で 3e-5 |
| $w>14$ | $10^{-9}$ | $w=14.1$ で 6e-14、 $w=50$ で 1e-12 |

$w$ が 4 から 14 の間では、第 1 項の中の大きい項が打ち消し合い、倍精度の桁が落ちます。 $\omega=0.11$ では、 $w=4$ は密度 7.0e-7 bohr⁻³、 $w=14$ は 1.6e-8 bohr⁻³ です（スピン分極のない全密度）。エネルギーへの寄与が小さい領域なので、許容差を広げてあります。

比較の式は 3 つです。 $\Delta$ は実装の値と参照値の差を表します。

- エネルギー密度: $\lvert\Delta e\rvert\le\mathrm{tol}_{\uparrow}\lvert e_{\uparrow}\rvert+\mathrm{tol}_{\downarrow}\lvert e_{\downarrow}\rvert$
- `dexdd`: 相対差が `tol` の 10 倍以下
- `dexdgd`: 差の絶対値が、`tol` の 10 倍に $\max(\lVert\mathrm{dexdgd}_s\rVert,\ 10^{-6}\lvert e_s\rvert/\lvert\nabla\rho_s\rvert)$ を掛けた値以下

`dexdgd` の尺度に下限があるのは、勾配についての導関数が小さい点で、相対差が意味を失うからです。 $w>14$ の点と、付け替えた $s$ がほとんど変化しない点が当たります。倍精度の試作（`--double`）は、全ての点でこの許容差を満たしました。許された差に対する実際の差の比は、最大で 0.12 です。

Quantum ESPRESSO のルーチンは倍精度なので、同じ程度の丸め誤差があります。`--compare-qe` はこの分を見込み、エネルギーで `tol` の 100 倍、導関数で 1000 倍までの差を許します。

#### 倍精度での計算の形

参照実装の `double_channel_energy` が、許容差を満たす倍精度の計算の形です。記号は Quantum ESPRESSO のルーチンの変数名に合わせてあります（ $DHs=D+s^2H$ 、 $Hsbw=s^2H+b\,w^2$ 、 $DHsbw=D+Hsbw$ 。 $b$ は当てはめの指数です）。

- 交換孔の関数 $EG(s)$ を単独では計算しません。式に現れるのは $Q=E+s^2EG(s)$ の形だけです。 $s>0.08$ では $Q=E-\left(\tfrac{3\pi}{4}+G_a(s)\right)\dfrac{16\,DHs^{7/2}}{15\sqrt{\pi}}$ として求め、 $s^2$ で割りません。 $s\le0.08$ では、多項式から $Q=E+s^2(EG_{a1}+EG_{a2}s^2+EG_{a3}s^4)$ を求めます。
- $\ln(Hsbw/DHsbw)$ は $-\mathrm{log1p}(D/Hsbw)$ として計算します。
- 導関数を求める方法の 1 つは、値と 2 つの偏微分（ $\rho_s$ と $\lvert\nabla\rho_s\rvert$ についての微分）を一緒に運ぶ演算です。式を手で微分して書く方法より、書き写しの誤りが入りにくいと考えられます。試作は、同じ丸めの性質を持つ複素数の微小な虚部で導関数を求めています。
- 密度の下限（スピンごとに 0.5e-14）と、勾配が 0 の点の扱いは、既存の PBE の関数に合わせます。付け替え前の $s$ には、「短距離部分の定義」に書いた下限 1e-10 を入れます。

#### Quantum ESPRESSO との照合

Quantum ESPRESSO 7.6 の `XClib/qe_funct_exch_gga.f90` は、モジュール `kind_l` の `DP` だけを使う 1 つのモジュールです [16]。このファイルを、`--qe-files` が書き出す 2 つの小さいファイルと一緒にコンパイルすると、`pbexsr` を単独で呼べます。Quantum ESPRESSO の全体をビルドする必要はありません。Quantum ESPRESSO のファイルは、リポジトリーに入れません。

```sh
python3 tools/wpbeh_reference.py --qe-files build/qe-check    # kind_l.f90、駆動プログラム、入力を書き出す
gfortran -O1 -J build/qe-check -o build/qe-check/wpbeh_qe_driver build/qe-check/kind_l.f90 \
  /path/to/q-e/XClib/qe_funct_exch_gga.f90 build/qe-check/wpbeh_qe_driver.f90
build/qe-check/wpbeh_qe_driver < build/qe-check/wpbeh_qe_input.txt > build/qe-check/wpbeh_qe_output.txt
python3 tools/wpbeh_reference.py --compare-qe build/qe-check/wpbeh_qe_output.txt
```

入力は、フィクスチャーの 160 点の 2 つのスピン（320 チャンネル）です。比較では、Quantum ESPRESSO のルーチンの 2 つの性質を参照値の側で再現します。1 つは、 $s>8.3$ のポテンシャルに付け替えの微分が掛かっていないことです。もう 1 つは、 $w>14$ で $e^{x}E_1(x)$ を有理式で近似することです。換算の式は、参照実装の `compare_qe` の説明にあります。残る差は、Fortran の倍精度の丸めと、指数積分の収束判定（1e-12）によるものです。

2026-11-23 の結果です（gfortran 13.3.0、`-O1`、倍精度）。320 チャンネルの全てが許容差の中にあり、ルーチンのエラーの返り値は全て 0 でした。値は相対差の最大で、`vsigma` はフィクスチャーの尺度での値です。

| `tol` の区分 | $w$ | チャンネル数 | エネルギー | `dexdd` | `vsigma` | 許された差に対する比 |
|---|---|---|---|---|---|---|
| 1.0e-12 | 0.017 から 0.36 | 160 | 7.1e-13 | 8.2e-13 | 2.2e-12 | 0.0071 |
| 1.4e-12 | 0.77 | 32 | 4.0e-13 | 2.7e-13 | 4.6e-13 | 0.0028 |
| 3.0e-11 | 1.7 | 32 | 2.4e-12 | 1.3e-11 | 1.3e-11 | 0.00081 |
| 2.2e-8 | 3.6 | 32 | 4.3e-10 | 5.9e-9 | 1.8e-8 | 0.00083 |
| 9.9e-6 | 8.0 | 32 | 1.5e-6 | 1.1e-4 | 4.3e-4 | 0.043 |
| 1.0e-9（ $w>14$ ） | 16.5 | 32 | 1.1e-12 | 1.4e-12 | 2.1e-9 | 0.0021 |

有理式の近似を換算に入れない場合、 $w>14$ の差はエネルギーで 9.7e-8、`dexdd` で 3.9e-5 でした。 $w$ が 4 から 14 の領域では、差が Fortran の丸め誤差に埋もれます。この領域で定数が一致することは、 $w$ の小さい点で 1e-12 まで合うことと、ソースとの行ごとの照合に依っています。ルーチンを 4 倍精度でコンパイルした照合はしていません（14 節）。

#### 出力の規約

短距離部分の関数とハイブリッド用の関数は、既存の PBE の関数（`kernels.h:285-420`）と同じ規約で値を返します。Quantum ESPRESSO の `pbexsr` を、スピンごとに $2\rho_s$ と $4\lvert\nabla\rho_s\rvert^2$ で呼んだときの出力との対応は、次のとおりです。

| OpenMX の出力 | 意味 | `pbexsr` の出力との対応 |
|---|---|---|
| `exc[0]` | 1 電子あたりの交換エネルギー $\varepsilon_x$ | 2 つのスピンの `sxsr` の和の半分を、全密度で割った値 |
| `dexdd[s]` | $\partial(n\varepsilon_x)/\partial\rho_s$ | そのスピンの `v1xsr` |
| `dexdgd[i][s]` | $\partial(n\varepsilon_x)/\partial(\partial_i\rho_s)$ | そのスピンの `v2xsr` の 2 倍に `gdens[i][s]` を掛けた値 |

$n$ は全密度、 $s$ はスピンです。還元勾配が 8.3 を超える点では、`v1xsr` と `v2xsr` に付け替えの微分が入っていないので、この対応は成り立ちません（「短距離部分の定義」）。OpenMX の関数は密度に下限（スピンごとに 0.5e-14）を入れるので、比較は下限より十分大きい密度の点で行い、下限の扱いは別の試験で確かめます。

表の「意味」の列は、PBE の交換について数値で確かめてあります。スピン分極を含む 4 点で、独立に書いた PBE 交換の式と `exc[0]` の相対差は 2e-15 以下でした。`dexdd` と `dexdgd` は、同じ式の中心差分と相対 1e-8 以内で一致しています。`pbexsr` との対応は、上の照合で確かめました。

#### 出典とライセンス

短距離部分の式と定数は、Quantum ESPRESSO 7.6 のソースを読んで書き写したものです [16]。Quantum ESPRESSO のファイルの表示は GNU General Public License で、OpenMX は GPLv3 です。`tools/wpbeh_reference.py` と `modules/xc/src/wpbeh_kernel.cpp` は新しく書いた実装で、冒頭に出典を書き、OpenMX と同じライセンスで配布します。

libxc の定義と、libxc を通した照合は、改訂 3 で外しました（1.2 節）。この文書はライセンスの解釈を示しません。出典の書き方やライセンスの表示を変える場合は、利用者が決めます。

### M2. ABF、二中心積分、LRI の係数

| 項目 | 内容 |
|---|---|
| 目標 | `modules/exx` の前半（`kernel`、`abf`、`two_center`、`lri`、`system`）を実装します。 |
| 前提 | M0 |
| 操作 | 6.2 節の手順を実装します。動径の格子と積分則の出発点は、既存の重なり積分と同じ設定（`1DFFT.NumGridK`、`1DFFT.EnergyCutoff`、`1DFFT.NumGridR`）です。実球面調和関数とその Gaunt 係数は、既存の重なり積分と同じ規約にします。`modules/basis` の `gaunt()` は複素球面調和関数の係数なので、`RealHarmonicTransform` で実数の係数に直す必要があります。 |
| 成功の確認方法 | 下の表の試験が全て通ります。 |
| 失敗例 | 重なり行列との照合が合わない場合は、軌道の順序、実球面調和関数の位相、 $\mathbf d$ の向きのどれかが既存のコードと違っています。PAO の積は PAO より高い波数の成分を持つので、 $\kappa$ の上限が足りないと積分が収束しません。 |

| 試験 | 内容 | 閾値 |
|---|---|---|
| `exx-kernel-analytic` | 4 つのカーネルの $\tilde v(\kappa)$ と $\kappa\to0$ の値 | 相対 1e-13 |
| `exx-two-center-gaussian` | ガウス関数の二中心積分（s と p、3 つのカーネル）。解析式は `exx_reference.py --integrals` にあります。 | s 同士は 1e-10、p を含む組は 1e-6 |
| 重なり行列の照合 | PAO に重なりカーネルを使った結果と、既存の重なり行列 `OLP[0]` の比較。本体のデータが要るので、小さい系の重なり行列をフィクスチャーに書き出すか、本体に一時的な照合の入口を作ります。 | 絶対 1e-8 |
| `exx-two-center-symmetry` | $V_{ab}(\mathbf R)=V_{ba}(-\mathbf R)^{\mathsf T}$ 、座標系の回転に対する不変性 | 1e-10 |
| `exx-three-center-gaussian` | ガウス関数の三中心積分（5.7 節） | 1e-8 |
| `exx-abf-pca` | ABF の動径関数の正規直交性と、符号の再現性。PAO の積を ABF に射影した残差の 2 乗和が、捨てた固有値の和と一致すること | 1e-12、相対 1e-8 |
| `exx-lri-identity` | ABF の線形結合で作った関数を式 (7) で展開したとき、元の係数が再現されること。結合距離の 2 原子では係数が丸め誤差で 1e-10 程度動くので（13 節の 17）、展開した密度の差をクーロンノルムで測ります。 | 1 原子と離れた 2 原子は、係数ごとに 1e-10。結合距離の 2 原子は、クーロンノルムで 1e-10、係数ごとに 1e-8 |
| `exx-lri-convergence` | 2 原子の対について、展開した密度の自己相互作用 $(\tilde\rho\vert g\vert\tilde\rho)$ が ABF を増やすと単調に増えて収束すること | 記録 |

クーロン計量での展開は、自己相互作用を下から近似します。ABF を増やしたときの増え方が、展開の誤差の目安です。元素ごとの ABF の個数と、この試験の結果を進捗ファイルに記録します。

### M3. 交換行列の構築

| 項目 | 内容 |
|---|---|
| 目標 | `modules/exx` の後半（`exchange`、`dense_backend`、C API）を実装します。 |
| 前提 | M2 の `system` |
| 操作 | 6.4 節を site の形式と Γ 点の形式で実装し、式 (4)、(15)、(16)、(17) も加えます。フィクスチャーは付属の `kbuild_reference.txt` です。同じ内容は `python3 tools/exx_reference.py --dump FILE` で作れます。 |
| 成功の確認方法 | フィクスチャーの T、V、D を読んで作った $K$ のブロックと $E_x$ が、フィクスチャーの値と絶対 1e-12 以内で一致します。Γ 点の形式の結果は、site の形式の $\sum_{\mathbf R}K(\mathbf R)$ と同じです。式 (16) と (17) は、中心差分と相対 1e-7 以内で一致します。関数表を渡さない場合と、試験用に別の実装の関数表を渡した場合の結果も同じです。 |
| 失敗例 | 同じ原子の周期像が近傍リストに複数ある場合に、加算を代入にすると寄与が消えます。同じ原子の T ブロックを半分にし忘れると、同じ原子の対の寄与が 2 倍になります。 |

### M4. Γ 点の SCF（Cluster、CPU）

| 項目 | 内容 |
|---|---|
| 目標 | Cluster ソルバーの CPU 経路で、HSE06 の全エネルギーと固有値を計算します。 |
| 前提 | M1、M2、M3 |
| 操作 | 7.4 節の 4 つの入口を `source/` に結合します。6.3 節、6.5 節、6.6 節、全エネルギーの成分を実装します。 |
| 成功の確認方法 | 下の表の試験が全て通ります。 |
| 失敗例 | 外側反復で $\delta$ が減らない場合は、 $K$ を足す行列の添字か、準局所部分の切り替えの時期が誤っています。 |

| 試験 | 内容 | 閾値 |
|---|---|---|
| 密度行列の照合 | 新しく集めた $D$ の FNAN の範囲が、既存の `CDM` と一致すること | 1e-10 |
| 混合比 0 | `scf.exx.alpha 0` の HSE06 が PBE と同じ全エネルギーを与えること | 1e-9 Hartree |
| 2 段の反復 | 外側反復が収束し、 $\delta$ が閾値より小さくなること。回数を記録します。 | 収束 |
| 並進 | 分子を実空間グリッドの間隔の整数倍だけ平行移動させたとき、全エネルギーと $E_x$ が変わらないこと | 1e-8 Hartree |
| ABF の収束 | `scf.exx.abf.pca` を 1e-3、1e-4、1e-5、1e-6 と変えたときの全エネルギー | 記録 |
| 打ち切りの収束 | `scf.exx.v.cutoff` を 6、8、12、16 と変えたときの全エネルギー | 記録 |
| スピン分極 | 閉殻の分子を `scf.SpinPolarization on` で解いた全エネルギーが、`off` の結果と一致すること。開殻の分子（三重項の O₂ など）が収束すること | 1e-8 Hartree |
| MPI | 1、2、4 ランクで全エネルギーが一致すること | 1e-9 Hartree |
| 周期系の Γ 点 | ダイヤモンド 8 原子のセルなど、セルが近傍の範囲より小さい系が動くこと | 収束 |
| 時間 | 準備、密度行列の収集、構築、加算のそれぞれの時間 | 記録 |

試験の入力は `work/input_example/Methane.dat` と `work/input_example/H2O.dat` を元に作ります。擬ポテンシャルは PBE 用（`C_PBE19`、`H_PBE19`、`O_PBE19`）です。「記録」の項目は、数値を進捗ファイルに書き、既定値を見直す材料にします。既定値を変えるかどうかは利用者が決めます。

M4 で、この表の実行と集計を `tools/exx_scf_checks.py` にまとめました。使い方は、進捗ファイルの「ビルドと実行」にあります。表にない実行（混合の方式を変えた実行と、既定の `scf.criterion` の実行）も入っています。

### M5. k 点（Band、CPU）

| 項目 | 内容 |
|---|---|
| 目標 | Band ソルバーの CPU 経路で、一般の k 点の計算をします。 |
| 前提 | M4 |
| 操作 | k 点ごとの $P(\mathbf k)$ の収集、式 (5) によるブロックの取り出し、式 (14) の加算を実装します。 |
| 成功の確認方法 | $n\times n\times n$ の k 点メッシュの計算と、 $n\times n\times n$ のスーパーセルの Γ 点の計算で、原子あたりの全エネルギーが 1e-7 Hartree 以内で一致します。 $E_x$ だけの比較では 1e-9 Hartree 以内です。 $K(\mathbf k)$ がエルミートです。 |
| 失敗例 | 位相の符号か、 $\mathbf k$ と $-\mathbf k$ の重みの扱いが誤っていると、 $K(\mathbf k)$ がエルミートになりません。 |

スーパーセルとの比較には、Γ 点を含むメッシュを使います。既定の `scf.Generation.Kpoint`（`GCenter`）は $k=i/N$ の点を使うので条件を満たします。`REGULAR` は $N$ が偶数のとき Γ 点を含みません。実空間のグリッドも 2 つの計算で揃えます。

M5 で、この節の実行と集計を `tools/exx_kpoint_checks.py` にまとめました。使い方は、進捗ファイルの「ビルドと実行」にあります。組にした 2 つの計算が同じ道筋で収束するように、入力を次の形にしています。

- 実空間のグリッドは、`scf.Ngrid` で分割数を、`scf.fixed.grid` で原点を指定します。スーパーセルの分割数は、基本単位胞の分割数の整数倍です。
- `scf.criterion` と `scf.exx.outer.criterion` には、基本単位胞の数を掛けます。本体は、この 2 つをセル全体のエネルギーの差と比べるためです（13 節の 26）。
- 基底は、1 原子に軌道が複数あるものにします（13 節の 27）。

道具は、表の組のほかに次の実行をします。1 つ目は、Band ソルバー同士の組です。同じ結晶を、単位胞と k メッシュの分け方を変えて計算します。Γ 点を含まない `REGULAR` のメッシュも、この形で比べられます。2 つ目は、M4 の表と同じ項目です（混合比 0、スピン分極、2 ランクと 4 ランク）。3 つ目は、拒否する入力と、打ち切りとメッシュを変えた記録です。

記録には、分極関数のある基底の実行も入れました（C6.0-s2p2d1 のダイヤモンドと、Si7.0-s2p2d1 の Si）。道具は、この 2 つの結晶について、メッシュの k 点の上のバンド端を表にします。`Band.dispersion` は拒否しているので（2.2 節）、ギャップは SCF のメッシュの固有値から求めます。伝導帯の底がメッシュの点にない結晶では、この値は結晶のギャップより大きくなります。

### M6. GPU（CUDA）

| 項目 | 内容 |
|---|---|
| 目標 | CUDA のビルドで、6.4 節の行列積と 6.5 節の加算を GPU で実行します。 |
| 前提 | M5。GPU の実機 |
| 操作 | 11 節 |
| 成功の確認方法 | CPU と GPU の全エネルギーが 1e-7 Hartree 以内で一致します（README の回帰比較が使っている閾値と同じです）。GPU のメモリーが足りない場合は、CPU の経路に切り替わって同じ結果を返します。段ごとの時間の記録も完了の条件です。 |
| 失敗例 | 実機がない環境では、コンパイルの検査（`tools/check_gpu_compilation.py --backend CUDA`）までを実行し、実機の試験が残っていることを報告します。 |

M6 は、GPU の実機が要るかどうかで 2 つに分けました。2026-11-24 に、実機なしで進められる部分を終えています。同じ日に利用者の計算機（RTX 5080）をつなぎ、実機の段階を始めました。この改訂の時点で、受け入れ試験の結果はありません。M6 は完了していません。

| 区分 | 内容 | 状態 |
|---|---|---|
| 行列積 | 6.4 節の行列積のうち大きいものを、FP64 の cuBLAS で計算します。容量が足りない積はホストに残します（11 節）。 | コードを書き、GPU のライブラリーの模型で試験しました。CI の CUDA のコンパイルの検査を通っています。実機の結果は、この改訂の時点でありません。 |
| 試験の道具 | `tools/exx_gpu_checks.py` が、上の表の 3 つの条件を実機で確かめます。 | CPU 版の実行ファイルを 2 つの役に当てて、道具の動作を確かめました。実機の段階で、デバイスの密行列ソルバーを通る実行を加え、49 件にしています。 |
| 6.5 節の加算 | デバイスの密行列ソルバーの経路で $K$ を足します。 | 実機の段階で、Band と Cluster のデバイス上の経路に書きました。密度行列の収集も加えています。MAGMA の経路には書いていません（6.5 節）。CI の CUDA のコンパイルの検査を通っています。実機では実行していません。 |
| GEMMul8 との比較 | 大きい積で、FP64 の cuBLAS と GEMMul8 を比べます（11 節）。 | 書いていません。切り替えの方法は利用者が決めます（14 節）。固有値ソルバーの中の行列積は、道具の `NAME-dfp64` の実行で比べます。 |
| 受け入れ試験 | 上の表の「成功の確認方法」 | 結果はありません。計算機が、最初の 18 件を実行しています。 |

6.5 節の加算を実機なしで書かなかった理由です。加算を入れる関数（`BandCol_ConstructDenseCsHs_OpenACC` など）は、ハイブリッド汎関数でない計算も通ります。この関数を変えると、GPU での結果が変わらないこと（1.2 節の最初の条件）を、実機なしでは確かめられません。行列積の部品は新しい翻訳単位で、通るのはハイブリッド汎関数の計算だけです。デバイスの密行列ソルバーの経路を使うには、その経路での密度行列の収集も要ります。実機なしの段階の収集は、ホスト上の固有ベクトルを受け取る形だけでした（6.3 節）。

計算機をつないだ後は、この理由が当てはまりません。加算と収集は、実機の最初の段階（`38992bc` の試験と測定）と並行して書きました。実機の 2 番目の段階では、先にハイブリッド汎関数でない計算を基準コミットの CUDA 版と比べます。既定のしきい値の実行と、しきい値を 1 に下げた実行の両方です。新しい経路の試験は、その後です。

実機の試験は `tools/exx_gpu_checks.py` が実行します。同じコミットからビルドした CPU 版と CUDA 版の実行ファイルを渡します。道具は、系ごとに 3 つの計算をします。

| 実行 | 実行ファイル | 確かめること |
|---|---|---|
| `NAME-cpu` | CPU 版 | 比べる相手です。 |
| `NAME-gpu` | CUDA 版 | 全エネルギーが CPU 版と 1e-7 Hartree 以内で合うこと。大きい積がデバイスで計算されること。 |
| `NAME-host` | CUDA 版。環境変数 `OPENMX_GPU_CAPACITY_TEST_RECOVERY=1` を付けます。 | 容量が足りない場合と同じ道筋で、大きい積が全てホストに残ること。全エネルギーが CPU 版と 1e-7 Hartree 以内で合うこと。 |

系は 6 つです。メタン（Cluster ソルバー）と C5.0-s1p1 のダイヤモンドには、デバイスに送る大きさの積がありません。この 2 つは、CUDA 版のホスト上の経路が CPU 版と同じ結果を出すことを確かめます。Si7.0-s2p2d1 の Si（1 ランク、スピン分極、2 ランクの 3 通り）と C6.0-s2p2d1 のダイヤモンドでは、構築 1 回、スピン 1 成分あたり 4 個の積がデバイスに行きます（11 節の実測）。`table` は、エネルギーの差、デバイスで計算した積の数、段ごとの時間を表にします。積の数が予定と違う場合も、終了コードが 1 になります。使い方は進捗ファイルの「ビルドと実行」にあります。

実機の段階で、デバイスの密行列ソルバーを通る実行を道具に加えました（`c6cdb9b`）。CUDA 版がデバイスで解くのは、行列の次数がしきい値（Band は 800、Cluster は 400）以上の計算です。道具の系の次数は 8 と 26 なので、上の 3 つの実行はホストで解きます。加えた実行は、既存の環境変数でしきい値を 1 に下げます。

| 実行 | 加える環境変数 | 確かめること |
|---|---|---|
| `NAME-dense` | `OPENMX_BAND_GPU_SWITCH_NUM=1`、`OPENMX_CLUSTER_GPU_SWITCH_NUM=1` | デバイスの密行列ソルバーで解き、 $K$ をデバイスで足すこと。全エネルギーが CPU 版と 1e-7 Hartree 以内で合うこと。 |
| `NAME-dfall` | 上の 2 つと、`OPENMX_BAND_GPU_RESERVE_MB=1000000`、`OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB=1000000` | 容量の判定が通らず、CPU の固有値ソルバーに切り替わって同じ結果を返すこと。 |
| `NAME-dpers` | 上の 2 つと、`OPENMX_BAND_GPU_PERSISTENT=1` | 固有ベクトルをデバイスに残す方式で、収集が同じ結果を返すこと。 |
| `NAME-dser` | 上の 2 つと、`OPENMX_CLUSTER_GPU_DIAG_SERIAL=2` | 1 つのプロセスが 2 つのスピンを順に解く方式で、同じ結果を返すこと。 |
| `NAME-dfp64` | 上の 2 つと、`OPENMX_GEMMUL8_DISABLE=1` | 固有値ソルバーの行列積を FP64 の cuBLAS にした実行との差（11 節）。 |

系は 6 つ加えて 12 個にしました。加えた系は、密行列の経路でランクの役割が変わる配置を作ります。Cluster ソルバーでは、密行列を持たないランクのある計算と、2 つのスピンを 1 ランクと 2 ランクで解く計算です。Band ソルバーでは、スピンごとに 1 ランクの計算、ランクごとに k 点が 1 つの計算、2 ランクが 1 つの k 点を受け持つ計算です。実行は全部で 49 件です。`table` は、密行列の経路の実行ごとに、CPU 版との全エネルギーの差と、 $K$ をデバイスで足したハミルトニアンの数を判定します。系の一覧は、進捗ファイルの「ビルドと実行」にあります。

### M7. 力

| 項目 | 内容 |
|---|---|
| 目標 | 厳密交換による力を計算します。 |
| 前提 | M5 |
| 操作 | 式 (16)、(17) を site の形式に書き換えます。次に、二中心積分と三中心積分の、原子の位置についての微分を実装します。最後の段階は、式 (18) の連鎖律を対ごとに適用する処理です。 |
| 成功の確認方法 | 密度行列を固定した $E_x$ の解析微分が、原子を動かした中心差分と相対 1e-6 以内で一致します。SCF を回した全エネルギーの数値微分と解析的な力の差は、1e-5 Hartree/bohr 以内です。厳密交換による力の総和が 0 になります。 |
| 失敗例 | 外側反復の収束が甘いと、Pulay 項が前の世代の $K$ の固有値を使うため、力に $\delta$ 程度の誤差が残ります。 |

### M8. 後続の項目

次の項目を、項目ごとに M4 から M7 と同じ形の試験で確認します。着手の順序は利用者が決めます。

- PBE0（切断クーロン）
- 近傍リストの形式で $H$ を読む出力。任意の k 点の $K(\mathbf k)$ は式 (14) で得られます。
- ノンコリニア
- HIP
- 応力
- GPU での XC の計算

## 10. 参照実装

10.1 節から 10.3 節は、交換行列の構築の参照実装 `exx_reference.py` の説明です。10.4 節は、ωPBE の短距離交換の参照実装 `wpbeh_reference.py` を扱います。

### 10.1 `exx_reference.py` の内容

| 関数 | 内容 |
|---|---|
| `Model` | 周期系の模型。真の距離で近傍リストを作り、乱数の T ブロックと V ブロックを持ちます。 |
| `build_K_blocks` | 6.4 節の site の形式。実装の手本です。 |
| `brute_force_K` | 式 (3) に式 (8) と (9) を入れた定義を、項ごとに足します。`build_K_blocks` とは別の経路です。 |
| `build_K_gamma` | Γ 点の形式 |
| `energy_gradient_gamma` | 式 (16) と (17) |
| `BlochDensity` | 式 (5) |
| `K_of_k`、`exchange_energy_blocks` | 式 (14) と (4) |
| `write_fixture` | C++ の単体試験が読むフィクスチャー |
| `integral_checks` | 式 (12) と 5.7 節の方法を、ガウス関数の解析値と比べます。SciPy が要ります。 |
| `write_kernel_fixture` | 4 つのカーネルの値を 50 桁で評価し、`exx-kernel-analytic` のフィクスチャー `kernel_reference.txt` を書きます（`--kernels FILE`）。mpmath が要ります。M2 で加えました。 |

重なり行列の照合に使うフィクスチャー `overlap_reference.txt` は、別のスクリプト `tools/exx_overlap_fixture.py` が作ります（M2 で追加）。入力は、本体が書いた `.scfout` ファイルと `.pao` ファイルです。手順はスクリプトの冒頭にあります。

### 10.2 `exx_reference.py` の実行結果

次の値は、この文書の作成時に実行した結果です（NumPy 2 系、SciPy 1 系、x86-64）。

| 試験 | 内容 | 最大の差 |
|---|---|---|
| T1 | `build_K_blocks` と `brute_force_K`（実数の $D$ 、複素エルミートの $D$ ） | 1.8e-15、3.7e-15 |
| T2 | $K_{xy}(\mathbf R)=K_{yx}(-\mathbf R)^{\dagger}$ | 8.9e-16、1.8e-15 |
| T3 | 同じ系を $2\times2\times1$ のスーパーセルとして記述した結果 | 8.9e-16 |
| T4 | `build_K_gamma` と、site の形式の $\sum_{\mathbf R}K(\mathbf R)$ | 4.4e-16 |
| T5 | $E_x$ の実空間の和（式 (4)）と、k 空間の $\frac{1}{2N_k}\sum_{\mathbf k}\mathrm{Tr}[P(\mathbf k)K(\mathbf k)]$ 。ずらさないメッシュと半格子ずらしたメッシュ | 4.4e-16、2.2e-16 |
| T5 | スーパーセルの Γ 点の $E_x$ と、k 点メッシュの $E_x$ のセル数倍 | 2.2e-15 |
| T6 | 式 (16)、(17) と中心差分（相対） | 2.4e-12、1.6e-11 |
| 積分 | 二中心積分（s 同士、3 つのカーネル） | 2e-15 以下 |
| 積分 | 二中心積分（p を含む組）。参照値は解析式の数値微分です。 | 6.3e-8 以下 |
| 積分 | 三中心積分（s と p） | 2.7e-10、5.2e-10 |

模型は 2 原子（PAO が 3 個と 2 個、ABF が 2 個と 3 個）の斜交セルです。セルが近傍の範囲より小さく、同じ原子の周期像が近傍リストに複数回現れます。

式 (3) と式 (5) の組が、ブロッホ軌道で書いた k 空間の式と一致することは、別の 1 次元の模型で確認しました（ずらさないメッシュで 4.7e-16、ずらしたメッシュで 8.9e-16）。このスクリプトは付属していません。

### 10.3 `kbuild_reference.txt` の形式

`--dump FILE` は、空白区切りのテキストを書きます。配列は行優先で、有効数字は 17 桁です。

```text
exx_fixture 1
alpha <値>
lattice <9 個>
natoms <原子数>
atom <番号> <PAO の数> <ABF の数> <分率座標 3 個>      # 原子ごと
cnb <原子> <個数>                                      # 続く各行は「原子 R1 R2 R3」
vnb <原子> <個数>
T <原子> <近傍の番号> <nP> <u_a> <u_b>                 # 次の行に値
V <原子> <近傍の番号> <nP_a> <nP_b>                    # 次の行に値
D <ブロック数>                                         # 各ブロックは「x y R1 R2 R3 行数 列数」と値
K <ブロック数>                                         # 同じ形
energy <式 (4) の値、スピン 1 成分>
```

`D` には、`build_K_blocks` が読んだブロックが全て入っています。付属の `kbuild_reference.txt` は、この形式で書いた 84 個の $K$ のブロックを持つファイルです。試験では保存したファイルを読みます。NumPy の版が変わると乱数の列が変わる可能性があるので、試験のたびに作り直す形にはしません。

### 10.4 ωPBE の参照実装 `wpbeh_reference.py`

M1 の「短距離部分の定義」の式を、mpmath で 50 桁の精度で評価します。Quantum ESPRESSO を含め、ほかの電子状態計算のコードは呼びません。

| 起動の形 | 内容 |
|---|---|
| 引数なし | 自己検査（15 項目） |
| `--report` | 定義の性質（式の切り替えでの跳び、PBE の増強因子との差、丸め誤差） |
| `--dump FILE` | フィクスチャーの作成 |
| `--point` と 8 個の数 | 1 点の値と途中の項（ $H$ 、 $F$ 、 $EG$ 、第 1 項から第 5 項） |
| `--double` | 倍精度の試作とフィクスチャーの比較（NumPy と SciPy が要ります） |
| `--qe-files DIR` | Quantum ESPRESSO の `pbexsr` を呼ぶ駆動プログラムと、その入力の書き出し（M1 の「Quantum ESPRESSO との照合」） |
| `--compare-qe FILE` | 駆動プログラムの出力と参照値の比較（NumPy が要ります） |

次の値は、改訂 3 の作成時に実行した自己検査の結果です（mpmath 1.4.1、x86-64）。

| 試験 | 内容 | 結果 |
|---|---|---|
| T1 | $EG$ の閉じた式が $s=0$ で正則になる条件の残差（論文の $E$ ）。小さい $s$ で多項式を使う理由です。 | 1.4e-8 |
| T2 | 閉じた式の $EG$ で作った $E+s^2EG$ が、 $E$ の値に依存しないこと（ $E$ を 0 にしたときとの差） | 3.3e-52 |
| T3 | $s=0.08$ での $E+s^2EG$ の跳び（閉じた式から多項式を引いた値） | −1.5e-9 |
| T4 | $s=0.08$ での $f$ の相対的な跳び（ $w$ が 0 から 3 の 5 点） | 6.9e-9 以下 |
| T5 | 1 と $f(0,s\to0)$ の差（LDA の極限） | 4.8e-9 |
| T6 | $f(w,s\to0)$ と、LDA の短距離交換の厳密な減衰関数の相対差（ $w\le5$ ） | 0.42% 以下 |
| T7 | $w=14$ の前後で切り替わる 2 つの式の相対差 | 2.2e-4 |
| T8 | $f(0,s)$ と PBE の増強因子の相対差（4 点） | 0.54% 以下 |
| T9、T10 | 導関数と中心差分の相対差（還元勾配が 0.14 と 11.9 の 2 点） | 5.4e-30、2.5e-29 |
| T11 | スピンの等しい 2 成分の和と、全密度で書いた式の相対差 | 4.0e-51 |
| T12 | 閉じた式と、交換孔の模型の数値積分の差（7 点、30 桁） | 2.1e-23 以下 |
| T13 | 還元勾配 8.3 での、付け替えた $s$ の跳び | 2.3e-9 |
| T14 | 付け替えた $s$ の上限での $f(0,s)$ （Lieb–Oxford の限界は 1.804） | 1.8040000 |
| T15 | 16 桁での丸め誤差の、許容差に対する比（フィクスチャーの密度と勾配の全ての組） | 0.011 以下 |

T6、T7、T8 の差は定義に含まれる近似（交換孔の模型と、誤差関数の当てはめ）によるもので、誤りを示す値ではありません。T12 は、閉じた式の書き写しを、定義の積分から確かめる試験です。定数と式は Quantum ESPRESSO 7.6 のソースと照合し、コンパイルしたルーチンの出力とも比べてあります（M1）。

付属の `wpbeh_reference.txt` は、 $\omega=0.11$ の 160 点のフィクスチャーです。スピンごとの密度 10 個（5 から 5e-9 bohr⁻³）と還元勾配 16 個（0.003 から 40）の組が、上向きと下向きに 1 回ずつ現れます。還元勾配には、式が切り替わる値（0.08 と 8.3）の両側の点を入れてあります。密度 5e-9 の点は $w>14$ です。1 行が 1 点で、27 列です。列の名前と比較の式は、ファイルの冒頭の注釈にあります。参照値は、ファイルに書いた倍精度の入力に対する値です。

## 11. GPU 実装の指針

行列積の形を、12 節の Si の例の大きさで示します。

| 演算 | 形 | 性質 |
|---|---|---|
| $G=T\cdot D$ | $(416\times1131)\cdot(1131\times s)$ | 原子ごとに 1 回の大きい GEMM |
| $T^{\mathsf T}R$ | $(1131\times416)\cdot(416\times s)$ | 原子ごとに 1 回の大きい GEMM |
| $W=V\cdot T$ | $(32\times32)\cdot(32\times14703)$ を V 近傍の数だけ | 中くらいの GEMM の束 |
| (i) から (iv) の括弧 | $13\times1131$ と $1131\times13$ などを、ABF と V 近傍の数だけ | 小さい GEMM の束 |

$s$ は列の数で、Γ 点の形式では $N$ 、site の形式では $S_A$ の PAO の数です。

部品はリポジトリーの既存のものを使います。

| 部品 | 場所 | 用途 |
|---|---|---|
| `OpenMXBlasDgemm` | `source/openmx_gpu_blas.h` | cuBLAS と hipBLAS に共通の GEMM |
| `openmx_gemmul8Dgemm` | `source/openmx_common.h:3747` | GEMMul8 を使う GEMM |
| `openmx_gpu_task_execute` | `source/openmx_gpu_task.h` | 容量の確認と CPU への切り替え |
| `openmx_gpu_transform` の `OPENMX_GPU_ADD` | `source/openmx_gpu_transform.h` | デバイス上の行列の加算。条件を満たさないと 0 を返すので、その場合のループを併せて書きます（`Band_DFT_NonCol.c:1154` と同じ使い方です）。 |
| `openmx_gpu_dm_entries` | `source/openmx_gpu_dm.h:17` | 任意の基底の対と位相についての密度行列 |

実行の方法は、行列積の種類ごとに分けます。

- 大きい GEMM の 2 つは、Γ 点の Si512 の例で演算量の 58% を占めます（計算値）。この 2 つに GEMMul8 と FP64 の cuBLAS の両方を試し、時間と結果の差を記録します。
- 小さい GEMM の束は FP64 でまとめて実行します。CPU の試作では、この部分が演算量の割合より長い時間を使いました（12 節）。リポジトリーにはバッチ GEMM の利用例がないので、呼び出しの形は新しく決めます。
- 密度行列の収集には `openmx_gpu_dm_entries` が使えます。Γ 点だけの大きい系では、GEMM で $cFc^{\mathsf T}$ を作る方が単純です。
- 複数のランクが 1 台の GPU を共有する構成（MPS）では、T ブロックと $D$ をランクごとにデバイスに置くとメモリーを圧迫します。デバイスに置く量と、構築を担当するランクの数を実測で決めます。
- GPU での実行時間は未測定です。この節の方針は演算の形からの判断で、実測で見直します。

M6 の実装です。実機なしで書いた範囲で、実機では実行していません（9 節の M6）。

大きい行列積をデバイスで計算する部品は、`source/Exact_Exchange_Device.c` です。結合層の `ExactExchange_Dgemm` は、`modules/exx` から来る積を 1 つずつ `Exact_Exchange_Device_Dgemm` に渡します。この関数が 1 を返した積は、デバイスで計算が済んでいます。0 を返した積は、本体の `DGEMM` が計算します。CPU 版では、全ての積で 0 を返します。

デバイスが積を引き受ける条件は 4 つです。

- プロセスに使えるデバイスがある（`gpu_rank_device_usable`）。
- 乗算の回数 $mnk$ が下限以上である。下限は 1e9 回です。
- $\beta=0$ で、行列の列の間に詰め物がない（`lda`、`ldb`、`ldc` が行数に等しい）。`modules/exx` の積は、全てこの形です（下の実測）。
- 3 つの行列が、デバイスの空きから 512 MiB を引いた範囲に入る。

1 つの積の手順は、次の順です。空きの問い合わせ（`OpenMX_GpuMemGetInfo`）、容量の判定、cuBLAS のハンドルの用意、3 つの行列の確保、2 つの因子の転送、積（`OpenMXBlasDgemm`）、同期、結果の転送、解放。積と積の間にデバイスに残るのは、ハンドルだけです。演算は FP64 です。

デバイスの呼び出しが失敗した積は、ホストが計算します。その後の扱いは 2 通りに分けました。

| 場合 | 当てはまる条件 | 扱い |
|---|---|---|
| 容量が足りない | 容量の判定で入らない。確保が `OPENMX_GPU_OUT_OF_MEMORY` を返す。ハンドルの生成か積が `OPENMX_BLAS_ALLOC_FAILED` を返す。 | デバイスを使い続けます。その積は「容量の不足でホストに残した積」に数えます。 |
| そのほかの失敗 | 上の条件に当たらない、成功でない状態コード | その計算の間、デバイスを使いません。プロセスは、失敗した段と状態コードを 1 回表示します。 |

どちらの場合も、確保した行列を解放し、ランタイムに残ったエラーを取り除きます（13 節の 32）。判定の後のメモリーの不足を 1 行目に入れた理由は、13 節の 34 に書きました。既存の GEMM の経路も、メモリーの不足とほかの失敗を分けています（`source/openmx_mgpu_gemm.cu:368-375`、`source/openmx_gpu_error.c:35-58`）。失敗の後でエラーを取り除いてホストに切り替える形は、`source/Divide_Conquer.c:596-672` と同じです。

容量が足りない場合の道筋は、既存の障害注入の環境変数 `OPENMX_GPU_CAPACITY_TEST_RECOVERY`（`docs/ENVIRONMENT_VARIABLES.md`）で試せます。`Input_std` がこの値を結合層に渡し、結合層は、残す空きを表せる最大の値にします。大きい積は全て、空きの問い合わせの後でホストに残ります。新しい環境変数は加えていません。

1 回の構築の行列積を、CPU 版で数えました（M6 の実測。基本単位胞の 2 原子、site の形式、1 ランク）。積の一覧は、どの構築でも同じでした。

| 系 | 積の数 | 最大の積（ $m\times n$ 、内側の次元） | その個数 | 乗算の回数に占める割合 | 3 つの行列の合計 | 次に大きい積 |
|---|---|---|---|---|---|---|
| C5.0-s1p1 のダイヤモンド | 2,754 | 5020×40、492 と 5020×492、40（どちらも 9.9e7 回） | 各 2 | 58% | 20.5 MiB | 1968×10、10（2.0e5 回）が 550 個 |
| Si7.0-s2p2d1 の Si | 994 | 8229×897、1131 と 8229×1131、897（どちらも 8.3e9 回） | 各 2 | 58% | 135 MiB | 14703×69、69（7.0e7 回）が 198 個 |
| C6.0-s2p2d1 のダイヤモンド | 2,934 | 20163×897、2171 と 20163×2171、897（どちらも 3.9e10 回） | 各 2 | 53% | 487 MiB | 28223×69、69（1.3e8 回）が 586 個 |

最大の積は、中心の原子ごとに 2 個あります（ $G=T\cdot D$ と $T^{\mathsf T}R$ ）。2 個は形が違い、乗算の回数と 3 つの行列の合計は同じです。表の「割合」は、この 4 個の合計です。残りは 5 種類の積で、どの種類も V 近傍の site の数だけあります。全ての積が $\beta=0$ で、行列に詰め物はありませんでした。スピン分極のある計算では、積の数が 2 倍になります（Si で 1,988 個）。下限の 1e9 回では、s2p2d1 の 2 つの系で 4 個の積がデバイスに行き、C5.0-s1p1 のダイヤモンドでは全ての積がホストに残ります。表の MiB は 2 の 20 乗バイトです。12 節の MB と、標準出力の MB も同じ単位です。

この実測から、今の部品で減らせる時間の上限が分かります。デバイスに送る積は、乗算の回数の 53% と 58% です。CPU 版の構築では、DGEMM の中の時間が 62% と 63% でした（12 節）。デバイスでの時間を 0 と仮定しても、構築の時間が減るのは 3 分の 1 程度です（計算値）。残りを減らすには、V 近傍ごとの積もデバイスで計算する必要があります。V 近傍ごとの 5 種類の積は、1 回の構築に各 198 個（Si）と各 586 個（s2p2d1 のダイヤモンド）あります。そのうち最も重い種類は、乗算の回数の 24% と 27% を占め、1 個あたり 16 MiB と 30 MiB の行列を使います。積ごとに行列を転送する今の形は、この積には向かないと考えられます。T ブロックと $D$ をデバイスに置く形（上の箇条書き）は、実機の測定で決めます。

2 つの限度（下限の 1e9 回と、残す空きの 512 MiB）は最初の値です。実機では測っていません。下限は、V 近傍ごとの積をホストに残し、最大の積だけをデバイスに送る値として選びました。

GEMMul8 は使っていません。1.2 節が、FP64 との差を測ってから使うと決めているためです。リポジトリーの GEMM の橋渡し（`openmx_gemmul8Dgemm`）は、`scf.gemmul8.enable` を書かない入力では、FP64 の弱い GPU で GEMMul8 を選びます（`source/Input_std.c:919-924` のコメント）。厳密交換の積をこの関数に通すと、差を測る前に GEMMul8 を使うことになります。2 つを比べるには切り替えが要ります。切り替えの方法は利用者が決めます（14 節）。利用者の GPU で、FP64 の cuBLAS がホストの DGEMM より速いかどうかは測っていません。リポジトリーには、RTX 5080 と 18 コアの AVX-512 のホストの組で、ホストの方が速い段があるという記述があります（`source/openmx_host_simd.h:36-43`）。

`openmx_gpu_task_execute`（上の部品の表）は使いませんでした。この関数は、項目の範囲を受け取るカーネルを、範囲を分けて実行する仕組みです。行列積 1 個は、分ける単位のない 1 回の呼び出しです。既存の GEMM の経路と同じく、容量を確かめてから BLAS を直接呼ぶ形にしました。

標準出力には、計算の終わりに次の 2 行が出ます。`N` はデバイスで計算した積の数、`P` はそのプロセスの数、`T` は 1 つのプロセスがその積に使った時間の最大、`M` は容量の不足でホストに残した積の数です。`N` と `M` は全プロセスの合計です。この 2 行が出るのは、デバイスで計算した積、容量の不足でホストに残した積、デバイスを使うのをやめたプロセスのどれかがある計算です。CPU 版と、デバイスに送る大きさの積がない計算では出ません。

```
<EXX>  GPU: N matrix products of the exchange matrix on the device, in P processes
<EXX>  (at most T s in one process); M left to the host for lack of device memory.
```

部品の試験は `tests/run_exx_device_product.py` です。`source/Exact_Exchange_Device.c` を CUDA 版の設定でコンパイルし、SDK のヘッダーの代わりに `tests/fixtures/exx_device_sdk` の模型を使います。模型は、デバイスのメモリーをホストのメモリーで表し、確保した量を数えます。どの呼び出しも失敗させられます。問い合わせと確保の間に、別のプロセスがメモリーを取る場合も作れます。試験が確かめることは 4 つです。

- デバイスで計算した積が、ホストの積と一致する。
- 引き受けない積では、配列にもデバイスにも触れない。
- 容量が足りない積の後、デバイスのメモリーが全て戻り、次の積を引き受ける。
- そのほかの失敗の後、デバイスのメモリーが全て戻り、エラーが残らず、後の積を引き受けない。失敗させる場所は 11 か所です。

この試験は、GPU の実際の動作を確かめません（13 節の 35）。

デバイスの密行列ソルバーの経路です。M6 の実機の段階で書き、実機では実行していません。加算の手順は 6.5 節、収集は 6.3 節に書きました。ここには、部品の選び方を書きます。

- 加算は、OpenACC の 1 つのループです。上の部品の表の `openmx_gpu_transform` は使いませんでした。この関数は、協調する GPU が 2 台未満だと 0 を返します（`source/openmx_gpu_transform.cu` の `w.workers.count<2`）。GPU が 1 台の計算機では、併せて書くループだけが動きます。GPU が複数ある計算機では、加算 1 回のために行列を配る転送が増えます。
- $K(\mathbf k)$ はホストで作ります。式 (14) のフーリエ和は、M5 の関数をそのまま使います。デバイスに写す配列は $n\times n$ で、加算のたびに確保して解放します。
- 固有値ソルバーは FP64 です。FP32 で解いて精製する方式には入りません（6.3 節）。
- 固有値ソルバーの中の行列積（基底の変換）は、既存の橋渡し（`openmx_gemmul8Dgemm`、`openmx_gemmul8Zgemm`）を通ります。入力に `scf.gemmul8.enable` がないとき、橋渡しはデバイスで方式を選びます。FP32 と FP64 の処理量の比が 8 未満のデバイスでは FP64 の cuBLAS、それ以外では GEMMul8 です（`source/gemmul8_bridge.cu` の `native_fp64_device_default`）。ハイブリッド汎関数のための切り替えは加えていません。1.2 節は、GEMMul8 を FP64 との差を測ってから使うと決めています。この経路では、既存のソルバーの既定の動作を変えずに、差を測る形にしました。道具の `NAME-dfp64` が、`OPENMX_GEMMUL8_DISABLE=1` を付けた実行です。差が 1e-7 Hartree を超えた場合は、ハイブリッド汎関数の計算だけを FP64 にするかどうかを決めます。

標準出力には、計算の終わりに次の行が出ます。`N` は、デバイスで $K$ を足したハミルトニアンの数で、全プロセスの合計です。SCF のステップ、スピン、k 点ごとに 1 と数えます。`N` が 0 の計算では出ません。

```
<EXX>  GPU: the exchange matrix was added to N dense Hamiltonians on the device.
```

容量の判定が通らない計算では、Band ソルバーが既存の行 `<BandCol> GPU dense memory admission declined; falling back to ELPA/ScaLAPACK.` を表示します。Cluster ソルバーは、`<EXX>  Cluster collinear n=` で始まり、`the device cannot hold the dense matrices` を含む行を表示します。MAGMA を使う設定では、どちらのソルバーも `MAGMA does not take the exchange matrix` を含む行を表示します。

## 12. 計算量とメモリーの目安

次の仮定からの計算値です。実測ではありません。

- Si のダイヤモンド構造（格子定数 10.263 bohr）、PAO は原子あたり 13 個（s2p2d1）、カットオフ半径 7 bohr
- ABF は原子あたり 32 個（仮定。実際の個数は M2 で決まります）
- C 近傍は 14 bohr 以内の 87 原子、V 近傍は 15 bohr 以内の 99 原子（ABF の半径 7 bohr と打ち切り半径 8 bohr の和）

M2 の実測では、Si7.0-s2p2d1 の ABF は既定のしきい値（1e-4）で原子あたり 69 個でした。間引く前は 78 個です。下の表は、32 個のままの計算値です。69 個にすると、T ブロックは 2.2 倍（原子あたり 8.1 MB、Si512 で 4.2 GB）、V ブロックは 4.6 倍（原子あたり 3.8 MB、Si512 で 1.9 GB）になります。演算量は、 $W$ の計算が 4.6 倍、ほかの部分が 2.2 倍です。この段落の倍率と容量も計算値です。

M6 の実測では、 $S_A$ の原子は 633 個で、 $s=8{,}229$ でした（13 節の 37）。下の表の 729 個と $s=9477$ は、29 bohr 以内の原子を数えた値のままです。

| 量 | 値 |
|---|---|
| $S_A$ の原子数（29 bohr 以内） | 729 |
| $D$ が必要になる原子間距離の上限 | 43 bohr（14 + 15 + 14）。この距離の中に 2473 原子 |
| 中心の原子 1 個あたりの積和の回数（site の形式、 $s=9477$ ） | 1.3e10 |
| 中心の原子 1 個あたりの積和の回数（Γ 点、Si512、 $s=6656$ ） | 1.1e10 |
| Si512 の Γ 点での合計 | 5.5e12（flop 数は 1.1e13） |
| 同じ系の対角化を $10N^3$ flop と仮定した値 | 2.9e12 flop |
| T ブロック | 原子あたり 3.8 MB。Si512 で 1.9 GB |
| V ブロック | 原子あたり 0.8 MB。Si512 で 0.42 GB |
| $D$ と $K$ （Si512、Γ 点） | それぞれ 354 MB |
| 中心の原子 1 個の作業領域（site の形式） | $D$ の部分行列 86 MB、 $G$ と $R$ が各 32 MB |

この表から次のことが分かります。

- site の形式の構築の演算量は、単位胞の原子数に比例します。k 点数に比例するのは、密度行列の表を作る部分だけです。Si の基本単位胞（2 原子）では 1 回の構築が 2.7e10 回の積和です。
- Si512 の Γ 点では、1 回の構築が対角化の数倍の演算量です。2 段の反復にする理由です。
- ABF の個数を $n_P$ とすると、 $W$ の計算は $n_P^2$ に、他は $n_P$ に比例します。

CPU での試作の実測値が 1 件あります。条件は、216 原子の格子模型（PAO 13 個、ABF 32 個、C 近傍 27 原子、V 近傍 33 原子）の Γ 点、NumPy と OpenBLAS、2 コアの計算機です。構築 1 回が 39 秒で、内訳は $G$ が 7 秒、 $W$ が 2 秒、4 項の小さい行列積が 25 秒、 $T^{\mathsf T}R$ が 4 秒でした。同じ計算機の DGEMM は約 60 GFLOP/s です。小さい行列積の束は、まとめ方で時間が大きく変わると考えられます。

M3 の実装の実測値です。条件は、重なり行列のフィクスチャーのセル（3 原子、PAO は 20、13、5 個、ABF は 140、69、15 個、C 近傍は合計 227 site、V 近傍は合計 335 site）、既定のしきい値と打ち切り、クラウドのコンテナの 1 コアです。行列積には OpenBLAS 0.3.26（1 スレッド）を渡しました。

| 形式 | 行列積の回数 | flop 数 | 時間 |
|---|---|---|---|
| Γ 点の形式 | 51 | 1.3e8 | 0.03 秒 |
| Γ 点の形式で、周期像を行列積の前にまとめない場合（`62ca5d8`） | 1681 | 1.2e11 | 6.9 秒 |
| site の形式（同じ密度行列を全てのセルのブロックとして渡した場合） | 1681 | 2.6e11 | 12.6 秒 |

ライブラリーの中のループでは、1 行目が 0.04 秒、2 行目が 33 秒でした。2 行目から求めた DGEMM の速度は約 18 GFLOP/s です。セルが近傍の範囲より小さい系では、Γ 点の形式の時間が周期像のまとめ方で 2 桁変わります。site の形式は周期像をまとめられないので、k 点の計算（M5）ではこの時間が出発点になります。

M4 の実装の実測値です。条件は、Γ 点の SCF（Cluster ソルバー、CPU 版）、既定のしきい値と打ち切り、クラウドのコンテナ（4 コア）です。ほかの計算を止めて、1 ランクと 2 スレッドで測りました。行列積は、本体の `DGEMM`（OpenBLAS 0.3.26）です。「全体」は、本体が表示する `DFT in total` です。

| 系 | 軌道、ABF | C 近傍、V 近傍の site | 準備 | 構築 1 回 | 構築の回数 | 全体 |
|---|---|---|---|---|---|---|
| メタン（s2p2d1、1 辺 10 Å のセル） | 33、129 | 25、25 | 1.9 秒 | 0.01 秒未満 | 6 | 13 秒 |
| 水（s2p2d1、同じセル） | 23、102 | 9、9 | 1.6 秒 | 0.01 秒未満 | 6 | 17 秒 |
| ダイヤモンド 8 原子（C5.0-s1p1） | 32、80 | 984、2,200 | 2.8 秒 | 0.01 秒未満 | 6 | 19 秒 |
| ダイヤモンド 8 原子（C6.0-s2p2d1） | 104、552 | 1,336、2,344 | 6.4 秒 | 0.26 から 0.36 秒 | 7 | 194 秒 |

s2p2d1 のダイヤモンドでは、厳密交換の時間は準備と 7 回の構築を合わせて 8.6 秒で、全体の 4% です。残りは既存のコードの時間です。力の計算が 81 秒、`Set_Hamiltonian` が 43 秒、`Set_Density_Grid` が 35 秒、`Set_ProExpn_VNA` が 17 秒で、対角化は 2.3 秒でした。密度行列の収集と $K$ の加算は、どの系でも合計 0.005 秒未満です。4 ランク（1 スレッド）では、準備が 6.6 秒、構築が 1 回 0.07 から 0.09 秒、全体が 59 秒でした（ランク 0 で測った経過時間）。準備は、全てのランクが構造の全体について行うので、ランクを増やしても短くなりません。8 原子のセルでは、上の表の仮定（Si512）の時間とメモリーは確かめられません。

M5 の実装の実測値です。条件は、k 点の SCF（Band ソルバー、CPU 版、`f2d3c9e` の実行ファイル）、既定のしきい値と打ち切り、クラウドのコンテナ（4 コア）です。ほかの計算を止めて、1 ランクと 2 スレッドで測りました。系は基本単位胞（2 原子）で、メッシュは Γ 点を含む 4×4×4（解く k 点は 36 個）と 8×8×8（260 個）です。「加算」は、 $K(\mathbf k)$ を作って `Hs` に足す時間の合計です。行列積の回数と flop 数は、数える処理を加えた一時的な実行ファイルで測りました。この段落と下の箇条書きの MB は 1,048,576 バイトです（本体と道具の表示に合わせました）。

| 系 | 軌道、ABF | C 近傍、V 近傍の site | 行列積の回数 | flop 数（構築 1 回） | メッシュ | 構築 1 回 | 構築の回数 | 加算 | 全体 | メモリーの最大 |
|---|---|---|---|---|---|---|---|---|---|---|
| ダイヤモンド（C5.0-s1p1） | 8、20 | 246、550 | 2,754 | 1.4e9 | 4×4×4 | 0.22 秒 | 7 | 0.8 秒 | 10 秒 | 137 MB |
| 同上 | 同上 | 同上 | 同上 | 同上 | 8×8×8 | 0.21 秒 | 7 | 5.7 秒 | 23 秒 | 138 MB |
| ダイヤモンド（C6.0-s2p2d1） | 26、138 | 334、586 | 2,934 | 5.9e11 | 4×4×4 | 32.3 秒 | 8 | 8.3 秒 | 324 秒 | 1,361 MB |
| 同上 | 同上 | 同上 | 同上 | 同上 | 8×8×8 | 32.3 秒 | 8 | 63.1 秒 | 423 秒 | 1,361 MB |
| Si（Si7.0-s2p2d1） | 26、138 | 174、198 | 994 | 1.2e11 | 4×4×4 | 5.5 秒 | 10 | 4.2 秒 | 101 秒 | 546 MB |
| 同上 | 同上 | 同上 | 同上 | 同上 | 8×8×8 | 5.6 秒 | 10 | 31.9 秒 | 168 秒 | 546 MB |

- 構築の時間とメモリーは、k 点の数に依りません。k 点の数に比例するのは、密度行列の収集と $K(\mathbf k)$ の加算です。
- 加算は、SCF の 1 ステップ、k 点 1 個あたり、s2p2d1 のダイヤモンドで 3.7 ミリ秒、Si で 1.6 ミリ秒でした。8×8×8 のメッシュでは、全体の 15% と 19% を占めます。加算は、SCF のステップごとに $K(\mathbf R)$ のフーリエ和を計算し直します。 $K(\mathbf k)$ は保存していません。
- Si の flop 数は、中心の原子 1 個あたり 5.8e10 です。この節の最初の表の積和の回数（1.3e10）を flop 数に直して 2 倍し、ABF の個数の違いで 2.2 倍すると 5.7e10 になります（計算値）。2 つの値は合っています。この計算値は $s=9477$ のままです。実測の $s$ は 8,229 で、 $s$ に比例する項は 13% 小さくなります（13 節の 37）。M6 で数えた乗算の回数は、中心の原子 1 個あたり 2.9e10 回で、上の flop 数の半分です（11 節）。
- 構築 1 回のうち DGEMM の中の時間は、s2p2d1 のダイヤモンドで 19.2 から 20.2 秒、Si で 3.3 から 3.5 秒でした。構築の 62% と 63% です。DGEMM の速度は、30 GFLOP/s と 34 GFLOP/s です（2 スレッド）。残りの時間の内訳は測っていません。
- 中心の原子 1 個の作業領域は、s2p2d1 のダイヤモンドで 944 MB です（M6 で数えた行列の大きさからの計算値）。 $D$ の部分行列と、最後の行列積の結果が各 334 MB、 $G$ と $R$ が各 138 MB です。行数は 2,171 と 897、列数は $s=20{,}163$ です。Si では、各 71 MB と各 56 MB で、合計 255 MB です（ $s=8{,}229$ ）。この節の最初の表の作業領域には、最後の行列積の結果が入っていません。改訂 8 までは、列数を 24,843 と 9,477 として、1,163 MB と 293 MB と書いていました。数え方の違いは 13 節の 37 にあります。
- 同じ系の PBE の実行は、メモリーの最大がダイヤモンドで 357 MB、Si で 321 MB でした。
- 2×2×2 のメッシュは、16 原子のスーパーセルの Γ 点と同じ計算です。構築 1 回の flop 数は、site の形式（基本単位胞）が Γ 点の形式（スーパーセル）の 30 倍（C5.0-s1p1）と 42 倍（C6.0-s2p2d1）でした。Γ 点の形式の値は 4.5e7 と 1.4e10 です。C6.0-s2p2d1 の構築 1 回の時間は、33.6 秒と 1.5 秒です（13 節の 30）。

## 13. 落とし穴

1. $K$ は近傍リストの形式の $H$ に入りません。この形式の $H$ から $H(\mathbf k)$ を作り直す経路（`Band_DFT_kpath.c:342` や `Band_DFT_Dosout.c:551` の `Hamiltonian_Band` など）は、 $K$ を含まない固有値を出します。どの出力が該当するかは個別に確かめていないので、第 1 版では 2.2 節の出力をまとめて拒否します。
2. HIP 版は `DFT.c`、`Band_DFT_Col.c`、`Cluster_DFT_Col.c`、`Force.c`、`Set_XC_Grid.c` などの複製を持ちます。これらのファイルへの `source/` 側の変更は、HIP 版に反映されません。一方で `openmx_common.h` や `Input_std.c` は共有なので、`Set_XC_Grid` の引数を変えると HIP 版の `Set_XC_Grid.c` も合わせる必要があります。PBE の関数も、HIP 版には `set_xc_grid_hip.cu:119` に別の実装が残っています。
3. $D(\mathbf R)$ を周期的に折り返してはいけません（5.3 節）。`Shift_K_Point`（1.0e-12）による微小なずれも、実際の座標を使えば自動的に扱えます。
4. 既存の格子ベクトルの表（`atv`、`atv_ijk`）は、近傍リストを作るのに必要な範囲で作られています（`truncation.c:225-241`）。EXX の距離を覆う保証はないので、格子ベクトルは EXX の側で列挙します。
5. Γ 点の形式では、同じ原子の周期像が近傍リストに複数回現れます。行列への書き込みは必ず加算にします。M3 の実装は、V ブロックと T ブロックの列を行列積の前に足し合わせます（6.4 節）。この和を代入にしても、周期像の寄与が消えます。
6. 同じ原子の T ブロックは半分の値を入れます（5.4 節）。
7. 主成分分析の固有ベクトルは符号が決まりません。符号を固定しないと、計算機によって T ブロックと V ブロックの符号が変わり、フィクスチャーとの比較や再開ができません。
8. 非分極の計算では、スピン 1 成分の $D$ から $K$ を作り、 $E_x$ を 2 倍します。2 倍の $D$ から作ると $K$ が 2 倍になります。
9. 最初の内側反復は PBE です。準局所部分をハイブリッド用に切り替えるのは、最初の $K$ ができた後です。切り替えを先にすると、厳密交換が補うはずの交換の一部が欠けた汎関数で収束させることになります。
10. FP32 の精製ステップや FP32 のグリッド積分で終わった状態から $K$ を作ってはいけません。既存の停止条件の検査の後に外側反復の判定を置く理由です。
11. `scf.restart` で再開した場合も、第 1 版は PBE の内側反復から始めます。 $K$ と $D$ をファイルに保存する形式は決めていません。
12. 式 (7) の行列 $G$ は、ABF の線形従属が強いと条件数が悪くなります。コレスキー分解が失敗する対があれば、固有値分解に切り替え、閾値と該当した対の数を記録します。
13. 部分内殻補正のある擬ポテンシャルでは、準局所部分は価電子密度と部分内殻密度の和で評価し、厳密交換は価電子の軌道だけで計算します。
14. エネルギーの成分を加えるときは、`OpenMXEnergyComponents` の構造体、合計の式（`DFT.c:2842`）、標準出力の 2 か所（`DFT.c:2869` 付近と `2918` 付近）を全て変えます。
15. $\omega=0$ にした短距離部分の関数を、PBE の交換の代わりに使いません。増強因子が最大 0.54% 違います（5.1 節）。全域の交換には既存の PBE の関数を使い、PBE0 では短距離部分の関数を呼びません。
16. `Set_XC_Grid` の引数や、`Total_Energy.c` と `Stress.c` の中のその呼び出しを変えたら、試験用の代役も合わせます。`tests/energy_workspace_harness.c` と `tests/stress_grid_harness.c` は、本体から切り出した関数を、代役の `Set_XC_Grid` と一緒にコンパイルします。この 2 つを使う試験（`total-energy-workspace`、`stress-grid`）は、M0 の `ctest` の 18 件に入っていません。実行するのは CI の `host` ジョブです（`build/core`、ラベル `native-state`）。M1 では代役の更新が漏れ、`host` ジョブが失敗しました。プッシュの後は、GitHub Actions の結果を確かめます。
17. 式 (7) の行列は、対角成分を 1 に揃えた後も条件数が 6e6 に達します（M2 の実測。C–C、2.3 bohr）。T ブロックの成分は、丸め誤差に条件数を掛けた 1e-10 程度までしか決まりません。計算機やコンパイラーが変わると、T ブロックはこの大きさで変わると考えられます。展開した密度のクーロンノルムは 1e-12 まで決まります。照合は $K$ やエネルギーで行うか、クーロンノルムで測ります。枢軸を選ばないコレスキー分解の枢軸は、条件数の目安になりません。実測では、最小の固有値が 1.5e-6 の行列でも、枢軸は 2e-3 以上でした。
18. クーロン計量の展開は、電荷を保存しません。展開した積の電荷と重なり積分の差は、結合距離の対で 3e-4 から 1e-2 でした（M2 の実測、既定のしきい値）。電荷に敏感な量を LRI の係数から作らないでください。
19. 打ち切りクーロンのカーネルは、 $\cos(\kappa R_c)$ で振動します。式 (12) の被積分関数は、 $\kappa$ について周期 $2\pi/(r_P+r_Q+d+R_c)$ の成分まで持ちます。 $r_P$ と $r_Q$ は 2 つの関数の半径です。既定の格子の刻みは 60/900 = 0.067/bohr で、 $2\pi$ をこの刻みで割ると 94 bohr です。 $R_c$ が大きい PBE0 の計算でこの長さを超えないかは、確かめていません（M5）。
20. 式 (13) の動径の積分は 4 次精度、台形則の動径積分は 2 次精度です。同じ原子の $G_{PQ}$ は、 $P$ と $Q$ を入れ替えると 2e-9 から 5e-9 ずれます。モジュールは 2 通りの平均を使います（同じ原子の $b^P$ も同じ）。平均しないと、ABF の張る空間にある積が正確に展開されません。
21. $K$ を作った後の内側の SCF は、2 つの形で前の $K$ の解に留まります。1 つ目は停止条件です。既存の条件は、連続する 2 ステップのバンドエネルギーの差と、緩い残差の条件です。混合が密度をほとんど動かさないと、新しい $K$ に応答していない密度でこの条件を通ります。単純混合の重みが下限（`scf.Min.Mixing.Weight`）に残っている場合と、外挿が密度を前の値に戻した場合に起きます。2 つ目は外挿の履歴です。前の $K$ で測った残差は、前の $K$ で収束した密度でほぼ 0 です。この組が履歴に残ると、外挿は密度を前の解へ引き戻します。M4 の実測です。対策のない実装では、`scf.criterion 1e-6` の Methane（RMM-DIIS）で内側の SCF が 2 ステップで止まり、全エネルギーが 3.2e-6 Hartree ずれました。ステップ数で停止を禁じるだけの実装では、`scf.criterion 1e-10` の Methane の全エネルギーが、RMM-DIISK で 1.4e-6、RMM-ADIIS で 1.7e-6 Hartree ずれました。この実装で RMM-DIIS 以外の方式を試した実行では、停止した 41 回のうち 9 回で、残差ノルムが $K$ を作った直後の値の 0.4 倍以上のままでした。RMM-DIIS の 439 回の停止では、混合比 0 の実行を除いて、直後の値の 1.5e-2 倍以下でした。今の実装は、混合を方式ごとに始め直し、残差ノルムが下がるまで停止を認めません（6.6 節）。直した後の結果は進捗ファイルにあります。
22. 分子の計算では、セルを「分子の大きさ + ABF の半径 + `scf.exx.v.cutoff`」より大きく取ります。V 近傍は周期像も数えるので、セルが小さいと、分子とその周期像の間の交換が入ります。M4 の実測では、1 辺 10 Å（18.9 bohr）のセルの水分子で、打ち切りを 8 bohr から 12、16 bohr にすると、全エネルギーが 1.5e-4、1.2e-3 Hartree 下がりました。1 辺 20 Å のセルでは、6 から 16 bohr まで全ての桁が同じです。
23. Γ 点だけの計算は、セルが短距離カーネルの到達距離より小さいと、厳密交換を大きく見積もります。Γ 点だけの密度行列は、 $D(\mathbf R)$ が全ての $\mathbf R$ で同じ形に当たり、距離で減衰しません。交換の和を打ち切るのは、カーネル $\mathrm{erfc}(\omega r)/r$ と `scf.exx.v.cutoff` だけです。ダイヤモンドの 8 原子のセル（1 辺 6.74 bohr、C5.0-s1p1）では、既定の打ち切りで $E_x$ が −4.61 Hartree になり、Γ 点のギャップが PBE の 6.5 eV から 12.5 eV に開きました（M4 の実測）。打ち切りを 6、8、12、16 bohr と変えると、 $E_x$ は −4.33、−4.61、−4.83、−4.86 Hartree と変わります。平面波の表示で $\mathbf G=0$ に当たる項 $-\alpha N_{\mathrm{occ}}\tilde v(0)/\Omega$ は、このセルで −3.4 Hartree です（計算値。 $\tilde v(0)=\pi/\omega^2$ 、 $N_{\mathrm{occ}}=16$ ）。この項は超格子を大きくしても同じ値で、原子あたりでは体積に反比例して減ります。小さいセルの固体は、k 点（M5）で計算します。
24. Γ 点だけの小さいセルでは、打ち切った V の和が半正定値でなくなります。全ての周期像について足した $\sum_{\mathbf R}V_{ab}(\mathbf R)$ は、カーネルのフーリエ変換が正なので半正定値です。V 近傍の距離で和を打ち切ると、負の固有値が現れます。ダイヤモンドの 8 原子のセルで、Γ 点の形式が使う和の固有値を測りました（M4。ABF は動径部分の 2 乗ノルムが 1 で、単位は Hartree）。C6.0-s2p2d1 の ABF（原子あたり 69 個、行列の次数 552）では、打ち切りが 6、8、12、16 bohr のとき、負の固有値が 46、36、45、37 個ありました。最小の固有値は −10.2、−17.6、−1.4、−0.47 で、最大の固有値は 2,371 から 2,986 です。C5.0-s1p1 の ABF（原子あたり 10 個）では、どの打ち切りでも負の固有値がなく、最小の固有値は 0.42 から 0.72 でした。s2p2d1 の基底のこのセルは、打ち切り 6 bohr で外側反復が発散しました。5 回目から 17 回目まで $\delta$ の符号が 1 回ごとに変わり、正の $\delta$ は 2 回ごとに約 1.4 倍になります。18 回目からは正のまま増え、27 回目の $K$ の後の内側の SCF が 100 ステップで収束しませんでした。既定の 8 bohr では 7 回で収束しましたが、 $\delta$ の比は 0.18、0.39、0.14、0.46、0.09 と 1 回おきに変わります。12 bohr と 16 bohr では 8 回で収束し、比は 0.21 から 0.25 でした。C5.0-s1p1 の基底では、どの打ち切りでも $\delta$ が毎回約 0.2 倍になります。発散の原因が負の固有値にあることは、確かめていません。外側反復は $K$ を混合せずに入れ替えるので、 $K$ の変化に対する密度の応答が元の変化より大きいと発散します。密度行列が距離で減衰する計算（大きいセル、k 点）では、打ち切られる対の寄与は小さいと考えられます。M5 で、k 点の計算を測りました。C6.0-s2p2d1 のダイヤモンドの基本単位胞は、2×2×2 のメッシュで外側反復が 12 回かかり、 $\delta$ の比は 0.07 から 0.65 でした。このメッシュは、16 原子のセルの Γ 点と同じ計算です。4×4×4 と 8×8×8 では 8 回で収束し、比は 0.09 から 0.13 です。打ち切りを変えたときのエネルギーの変化は、31 に書きました。k 点の計算では、V の和の固有値を測っていません。
25. 単純混合（Simple）と Kerker 混合は、残差ノルムが約 3e-5 を下回ると、残差をほとんど減らせません。混合の重みを決める規則が、この水準では重みを減らすだけになるためです（6.6 節）。交換行列の有無には関係しません。M4 の実測です。系は、入力例の基底のメタン、水、ダイヤモンドの 8 原子です（C5.0-s1p1 と H5.0-s1。水は O6.0-s1p1 と H6.0-s1p1）。PBE のメタン（Kerker 混合、`scf.criterion 1e-10`）は、残差ノルムが 50 ステップ目で 1.2e-5、300 ステップ目で 1.0e-5 でした。300 ステップでは収束しません。同じ条件の単純混合も、300 ステップで収束しません。この条件の HSE06 は、最初の PBE の SCF が収束しないので、 $K$ を作らずに終わります（6.6 節の「未収束の扱い」）。`scf.criterion 1e-6` の PBE の 6 件は、残差ノルムが 1.0e-5 から 1.5e-5 のステップで止まりました。全エネルギーは、RMM-DIIS（`scf.criterion 1e-10`）の値から 8e-8 から 9.3e-6 Hartree ずれます。停止条件に 3 つ目の値がない実装（`2753c7d`）では、`scf.criterion 1e-6`、`scf.maxIter 300` のメタンと水の HSE06 が、4 件とも収束しませんでした。4 回目の $K$ の後の内側の SCF が、300 ステップで止まりません。今の実装（`b8126f8`）では、ダイヤモンドを加えた 6 件が、外側反復 5 回か 6 回、合計 49 から 90 ステップで収束しました。全エネルギーのずれは 1.0e-7 から 1.7e-5 Hartree です。「PBE の SCF が止まったときの残差」を下限にする形も試しましたが、採りませんでした。水の単純混合で、 $K$ の変化が残差を 9.3e-6 から 1.6e-5 に上げ、300 ステップ後も 1.2e-5 でした。`scf.Mixing.Type` を書かない入力は、Simple になります（`Input_std.c` の選択肢の先頭）。この 2 つの方式の HSE06 は、PBE と同じく 1e-5 Hartree 程度までしか決まりません。精度が要る計算では、外挿する方式（RMM-DIIS、RMM-DIISK など）を指定します。
26. `scf.criterion` と `scf.exx.outer.criterion` は、セル全体のエネルギーの差と比べる値です。原子の数では割りません。同じ結晶をスーパーセルで計算すると、同じ値でも判定が厳しくなります。M5 の実測です。C5.0-s1p1 のダイヤモンドを、基本単位胞の 2×2×2 のメッシュと、16 原子のスーパーセルの Γ 点で計算しました。 $\lvert\delta\rvert$ は、4 回目の構築で 7.9e-7 と 6.3e-6 Hartree、5 回目で 7.3e-8 と 5.8e-7 Hartree です。既定の閾値（1e-6）なら、基本単位胞は 4 回目、スーパーセルは 5 回目で止まります。止まった回の $E_x$ は、基本単位胞あたり 7.3e-8 Hartree 違います。この差は、収束の途中で止めた回の違いによるものです。2 つの計算を比べるときは、2 つの閾値に基本単位胞の数を掛けて揃えます。`tools/exx_kpoint_checks.py` は、この形で入力を作ります。
27. 単位胞に軌道が 1 個しかない系は、 $K$ の組み込みの試験になりません。ブロッホ状態の係数が規格化だけで決まり、密度行列が占有数だけの関数になるためです。 $K$ を変えても密度行列は変わりません。M5 では、水素の鎖を H5.0-s1 で計算したときに、外側反復が 2 回で止まって気付きました。試験の系は、1 原子に軌道が 5 個ある基底（H5.0-s2p1）で作り直しました。
28. `scf.Generation.Kpoint none`（対称性で減らした Monkhorst–Pack のメッシュ）は、ハイブリッド汎関数でない計算でも異常終了します。GGA-PBE のダイヤモンド（基本単位胞、C5.0-s1p1、3×3×3、1 ランク）で、k 点を作った直後に `MPI_Comm_size` が `MPI_ERR_COMM: invalid communicator` を出し、終了コード 5 で終わりました。基準コミット（`4e0ee247`）からビルドした CPU 版でも同じです。原因は調べていません。HSE06 の試験は、`GCenter` と `REGULAR` で行いました。
29. 構築は、中心の原子をランクに配ります。ランクが単位胞の原子より多いと、構築を受け持たないランクが出ます。標準出力の構築の行は、ランク 0 が受け持った原子の範囲を表示します。基本単位胞（2 原子）のダイヤモンドを 4 ランクで計算すると、構築するのは 2 ランクです。原子 1 個の構築の中の並列は、行列積のスレッドだけです。
30. site の形式の構築は、k メッシュが粗くても軽くなりません。構築は、 $S_A$ の全ての site を密度行列の別の列として扱います。時間は k 点の数に依りません（12 節）。Γ 点を含む n×n×n のメッシュの $D(\mathbf R)$ は、基本単位胞の n 倍の周期で同じ値を繰り返します。スーパーセルの Γ 点の形式は、この周期で列をまとめてから行列積を計算します。M5 の実測です。2×2×2 のメッシュのダイヤモンド（C6.0-s2p2d1）の構築は、site の形式で 1 回 33.6 秒でした。16 原子のスーパーセルの Γ 点の形式では 1.5 秒です。構築 1 回の flop 数は、site の形式が 42 倍でした。2 つの計算の全エネルギーは、原子あたり 3e-11 Hartree まで一致します。site の形式で列をまとめる処理は、実装していません（14 節）。まとめられる列があるのは、メッシュの周期だけ離れた 2 つの site が $S_A$ に入る場合です。C6.0 のダイヤモンドの $S_A$ は半径 26 bohr の球で、8×8×8 のメッシュの周期は 38.1 bohr です（どちらも計算値）。
31. 粗い k メッシュでは、エネルギーが `scf.exx.v.cutoff` に依ります。Γ 点を含む n×n×n のメッシュの $D(\mathbf R)$ は、メッシュの周期で同じ値を繰り返し、距離で減衰しません（30）。交換が $D$ を読む距離がこの周期を超えると、遠い対の寄与を抑えるのは短距離カーネルと打ち切りだけになります。Γ 点だけの小さいセル（23）と同じ事情です。M5 の実測です。値は基本単位胞あたりです。C5.0-s1p1 のダイヤモンドでは、打ち切りを 8 bohr から 16 bohr にすると、全エネルギーが 4×4×4 のメッシュ（周期 19.1 bohr）で 2.0e-3 Hartree 下がりました。6×6×6（28.6 bohr）では 1.3e-6 Hartree、8×8×8（38.1 bohr）では 5.7e-7 Hartree です。この基底が $D$ を読む距離は、打ち切り 8 bohr で 33 bohr です（計算値）。Si7.0-s2p2d1 の Si の 8×8×8（周期 58.0 bohr）では、3.2e-7 Hartree 下がりました。6 bohr にすると、2.2e-5 Hartree 上がります。メッシュの収束を調べるときは、打ち切りを固定して比べます。粗いメッシュの値には、メッシュの誤差と打ち切りの誤差が両方入ります。
32. デバイスの呼び出しが失敗すると、ランタイムにエラーが残ります。`cudaGetLastError` は、最後のエラーを返して成功の状態に戻す関数です [19]。エラーを残したままにすると、次にこの関数を呼ぶ別の処理が、関係のない失敗を受け取ります。M6 の部品は、積を途中でやめるたびにエラーを取り除きます。既存の経路（`source/Divide_Conquer.c:605-607`）と同じ扱いです。
33. cuBLAS のハンドルは、作ったときのデバイスに結び付きます。NVIDIA の文書は、ハンドルを使う間、そのデバイスを現在のデバイスにしておくことを求めています [18]。M6 の部品は、プロセスのデバイスが変わるとハンドルを作り直します。ハンドルを返すときは、作ったデバイスに一時的に切り替えます。同じ文書は、ハンドルの生成と破棄が `cudaDeviceSynchronize` を呼ぶので、回数を減らすよう勧めています。部品は積ごとに作り直さず、計算の終わりまでハンドルを持ちます。
34. 容量を確かめた後でも、デバイスのメモリーは足りなくなります。複数のプロセスが 1 台のデバイスを共有すると、問い合わせと確保の間に、別のプロセスがメモリーを取れるためです。判定の後の確保の失敗を全て「デバイスを使うのをやめる失敗」にすると、メモリーが一時的に足りなかっただけで、残りの計算がホストだけになります。コミットする前の版が、この形でした。メモリーの不足を表す状態コードは、容量が足りない場合と同じ扱いにしています（11 節）。WSL2 などの WDDM の環境では、物理メモリーを超えた確保が失敗せず、システムのメモリーに退避して遅くなります（`source/set_cuda_default_device_from_local_rank.h:44-59` のコメント）。この場合は、確保の失敗として現れません。部品は、空きの問い合わせに、NVML の値で上限を付けた既存の関数 `OpenMX_GpuMemGetInfo` を使っています。
35. CPU 版の実行ファイルは、デバイスの経路を実行しません。CPU 版のヘッダー（`source/backends/cpu/include/openmx_cpu_runtime.h`）は GPU の呼び出しを「未対応」で返し、`gpu_rank_device_usable` は 0 です。障害注入の環境変数も、CPU 版では何も変えません。クラウドのコンテナで確かめられるのは 2 つです。CPU 版の結果が変わらないことと、模型に対する部品の動作です。
36. ルートの `CMakeLists.txt` の試験の一覧は、本体をビルドする構成でだけ登録されます。CI の `host` ジョブは `OPENMX_BUILD_APPLICATION=OFF` で構成するので、この一覧の試験を実行しません。`cpu-application` ジョブは、一覧から名前で選んだ試験だけを実行します。GPU の模型の試験は、`gpu-contracts` ジョブがスクリプトを直接実行しています。M6 の試験もこのジョブに加えました。Native GPU compilation のワークフローは、翻訳単位をコンパイルするだけです。CUDA 版のリンクと実行は、CI にはありません。
37. 構築の列の数 $s$ は、半径の和の中にある原子の数からは決まりません。 $S_A$ は、V 近傍の原子それぞれの C 近傍を合わせた集合です。M6 で行列積の大きさを数えると、 $S_A$ の原子は Si7.0-s2p2d1 の Si で 633 個、C6.0-s2p2d1 のダイヤモンドで 1,551 個でした。半径の和の中にある原子は、729 個と 1,911 個です。和の半径に近い原子は、条件を満たす V 近傍の原子がないと $S_A$ に入りません。改訂 8 までの 12 節の作業領域は、後の数で計算していました。
38. `Construct_Band_CsHs` は、密行列を持たないランクでは何も作らずに戻ります。デバイスの密行列ソルバーを使う計算で、ランクが密行列の持ち主でない場合です。このとき `state->last_construct_on_device` は 0 のままです。 $K$ の加算は、持ち主のランクだけが通る場所に置きます。M6 の `BandCol_DenseAddExactExchange` は、デバイスに $H$ がない状態で呼ばれると計算を止めます。
39. デバイスの作業領域 `d_tmp` を借りられるのは、 $H$ の組み立てと対角化の間だけです。対角化の最初の変換がこの領域に書き、解いた後は固有ベクトルが入ります。固有ベクトルをデバイスに残す方式（`OPENMX_BAND_GPU_PERSISTENT`）では、次の組み立てまで中身が読まれます。M6 の加算は、組み立ての直後にだけ `d_tmp` を使います。
40. CUDA 版の `scf.eigen.lib` の既定値は `gpusolver` です（`source/Input_std.c`）。入力にこのキーワードがなくても、次数がしきい値以上の計算はデバイスの密行列ソルバーを通ります。CPU 版は、同じ入力を ELPA2 で解きます。しきい値は環境変数で変えられます（`OPENMX_BAND_GPU_SWITCH_NUM`、`OPENMX_CLUSTER_GPU_SWITCH_NUM`）。小さい系でデバイスの経路を試験するときは、この 2 つを 1 にします。
41. Cluster ソルバーが 2 つのスピンを 1 ランクで順に解く方式に入ったことは、標準出力の `solves both spins serially on the GPU` を含む行で分かります。この行は、`level.of.stdout` が 2 以上のときだけ出ます。経路を標準出力で確かめる試験では、入力にこの指定が要ります。
42. 道具の `table` は、1 件も実行していない系を黙って飛ばしていました（`c6cdb9b` より前の版）。足りない実行として数えるのは、一部だけを実行した系でした。系を増やした版では、実行していない系の名前を表示して、終了コード 1 を返します。

## 14. 未決事項

設計の時点で確かめられていない点です。該当する試験の結果が出たら、進捗ファイルに記録して利用者に報告します。

| 項目 | 内容 | 確かめる時期 |
|---|---|---|
| LRI の精度 | OpenMX の PAO と擬ポテンシャルでの精度は未確認です。ABACUS の結果 [6, 7] は別の基底でのものです。足りない場合は、角運動量の高い ABF を加える方法を検討します。M2 の実測では、結合距離の C–C、C–H、Si–Si で、2 原子の PAO の積の自己相互作用の和が 0.2% から 0.5% 不足しました。比べた相手は、s5p5d4f3 の積から作った ABF です。ABF を作る組に角運動量の 1 つ高い関数を 1 個加えると、不足は 0.02% から 0.06% に減り、ABF は C と Si で約 2 倍になります。M4 で、全エネルギーとギャップへの影響を測りました。ABF を作る組に、角運動量の 1 つ高い動径関数を 1 個から 3 個加えた一時的な実行ファイルを使っています。s2p2d1 の基底の分子（メタン、水、三重項の酸素）では、3 個加えたときの全エネルギーの変化が −0.07 から −0.47 mHartree でした。メタンと水の HOMO と LUMO の差は、0.004 eV 以下しか変わりません。1 個加えたときの変化は、3 個のときの 59% から 79% です。入力例の基底（C5.0-s1p1 と H5.0-s1、O6.0-s1p1 と H6.0-s1p1）の分子では、全エネルギーが 0.9 から 3.3 mHartree 変わります。数値は進捗ファイルの表にあります。M5 で、同じ一時的な変更を k メッシュ（4×4×4）の計算で測りました。s2p2d1 の基底では、1 個加えたときの全エネルギーの変化が、Si で原子あたり −8.0 meV、ダイヤモンドで −6.6 meV です。メッシュ上のギャップの変化は、0.01 eV 以下でした。Si は 2 個と 3 個でも測り、1 個のときとの差は 8e-6 Hartree 以下です。C5.0-s1p1 のダイヤモンドでは、1 個から 3 個で −6.6、−8.3、−8.7 mHartree と変わり、ギャップは 0.12 eV 下がります。Γ 点だけの小さいセルで見えた単調でない変化（M4）は、k メッシュの計算では出ていません。ABF の作り方を変えるかどうかは、利用者の決定を待っています。 | M4、M5（測定は済み） |
| ABF の個数 | M2 で測りました。既定のしきい値で、H6.0-s2p1 が 15 個、C6.0-s2p2d1 と Si7.0-s2p2d1 が 69 個、Si7.0-s2p2d1f1 が 140 個です。12 節の仮定（32 個）の約 2 倍でした。 | 済み（M2） |
| ABF のしきい値 | M4 で、`scf.exx.abf.pca` を 1e-3 から 1e-6 まで変えた全エネルギーを記録しました。s2p2d1 の分子では、既定値（1e-4）と 1e-6 の差が 1.5e-5 から 2.1e-5 Hartree です。1e-3 と既定値の差は、メタンで 2.9e-4 Hartree、Γ 点だけのダイヤモンドで 9.4e-3 Hartree でした。既定値を変えるかどうかは、利用者が決めます。 | 利用者の判断 |
| V 近傍の打ち切り | M4 で Γ 点の計算を測りました。周期像が到達距離に入らない分子では、6 から 16 bohr まで全エネルギーが変わりません。Γ 点だけの小さいセル（ダイヤモンドの 8 原子）では、16 bohr でも収束しません（13 節の 23）。s2p2d1 の基底のこのセルは、6 bohr で外側反復が発散しました（13 節の 24）。M5 で、k 点の計算を測りました。8×8×8 のメッシュで打ち切りを 8 bohr から 16 bohr にすると、全エネルギーは基本単位胞あたり、C5.0-s1p1 のダイヤモンドで 5.7e-7 Hartree、Si7.0-s2p2d1 の Si で 3.2e-7 Hartree 下がります。メッシュ上のギャップの変化は、どちらも 0.001 eV 以下です。Si の 6 bohr の全エネルギーは、8 bohr より 2.2e-5 Hartree 高い値でした。この 2 つの系の密なメッシュでは、既定値の 8 bohr で足ります。文献 [6]（arXiv 版の 4.1.1 節）は、同じ役割の打ち切り半径 $R_Q$ を 24 bohr から 8 bohr に縮めたときのギャップの誤差を、1 meV 未満としています（HSE06、Si、8×8×8 のメッシュ、DZP の基底）。粗いメッシュでは、打ち切りへの依存が残ります（13 節の 31）。s2p2d1 の基底のダイヤモンドと、ギャップの小さい系は測っていません。既定値を変えるかどうかは、利用者が決めます。 | 済み（M4、M5） |
| 動径の格子 | M2 で測りました。既定の設定で足ります。動径の区間を 4 倍にしたときの 2 原子の和の変化は相対 1e-8 で、波数の上限と刻みを変えたときの変化は 1e-11 以下でした。 | 済み（M2） |
| 打ち切りクーロンと $\kappa$ の刻み | PBE0 で $R_c$ が大きいとき、既定の刻みで式 (12) が収束するかは未確認です（13 節の 19）。M5 では確かめていません。このカーネルを本体から使うのは PBE0 で、PBE0 は M8 で加えます。 | M8 |
| 外側反復の回数 | M4 で測りました。既定の閾値（1e-6 Hartree）で、分子は 4 回から 6 回、Γ 点だけのダイヤモンドの 8 原子は 6 回と 7 回でした。 $\delta$ の大きさは、1 回ごとに 0.04 倍から 0.46 倍になります（進捗ファイルの表）。M5 で、k 点の計算を測りました。 $\lvert\delta\rvert$ が 1e-6 Hartree を下回る回を数えています。基本単位胞のダイヤモンド（C5.0-s1p1）は、2×2×2 のメッシュで 4 回目、3×3×3 から 8×8×8 で 5 回目でした。s2p2d1 の基底では、ダイヤモンドが 5 回目、Si が 6 回目です（4×4×4 と 8×8×8）。この 3 つの系の $\delta$ は、1 回ごとに 0.05 倍から 0.16 倍になります。Γ 点を含まない 4×1×1 のメッシュのダイヤモンドは 10 回目で、比は 0.27 から 0.35 でした。2×2×2 のメッシュの s2p2d1 のダイヤモンドは、比が 0.07 から 0.65 の間で変わります（13 節の 24）。 | 済み（M4、M5） |
| 混合の履歴 | M4 で確かめました。`SCF_iter_shift` でステップ数を戻すと、混合を始め直せます。戻す量は混合の方式ごとに違います（6.6 節、13 節の 21）。 | 済み（M4） |
| `scf.maxIter` の数え方 | M4 は、内側の SCF ごとに数える形で実装しました（6.6 節）。利用者の確認を待っています。 | M4 |
| 短距離部分の照合の精度 | Quantum ESPRESSO のルーチンとの照合は、倍精度で行いました。 $w$ が 4 から 14 の領域では、差が Fortran の丸め誤差（エネルギーで 1.5e-6 まで）に埋もれます。ルーチンを 4 倍精度でコンパイルした照合はしていません（M1）。 | 必要になったとき |
| GPU の速度 | 未測定です。M6 で書いた部品がデバイスに送る積は、乗算の回数の 53% と 58% です（11 節の実測）。デバイスでの時間を 0 と仮定しても、構築の時間が減るのは 3 分の 1 程度です（計算値）。利用者の GPU で、FP64 の cuBLAS がホストの DGEMM より速いかどうかも測っていません。 | M6（実機） |
| 金属 | 密度行列の減衰が遅い系での k 点の収束は未確認です。第 1 版では半導体と絶縁体で検証します。M5 では、半分詰まったバンドを持つ水素の鎖（1 次元、H5.0-s2p1）を計算しました。4 点と 6 点のメッシュの結果は、スーパーセルの Γ 点の結果と一致します。メッシュを密にしたときの収束と、3 次元の金属は調べていません。 | 必要になったとき |
| 力の site 形式 | 式 (16) と (17) は Γ 点の形式でだけ確認しています。 | M7 |
| メモリー | ランクごとに $D$ と $K$ を持つ形が、大きい系で成り立つかは未確認です。M5 で測ったのは、基本単位胞（2 原子）の k 点の計算です。site の形式の $D$ と $K$ のブロックは、s2p2d1 のダイヤモンドで 9,038 個、スピン 1 成分あたり各 11.7 MB でした。実行のメモリーの最大は 1,361 MB で、中心の原子 1 個の作業領域の計算値が 944 MB です（12 節。改訂 9 で 1,163 MB から直しました）。この作業領域は、構築を受け持つランクごとに要ります。原子の多い系は測っていません。 | M6 |
| site の形式の構築の高速化 | 実装していない方法が 3 つあります。メッシュの周期で列をまとめる（13 節の 30）。小さい密度行列のブロックを省く。 $K(\mathbf k)$ を保存して、SCF のステップごとのフーリエ和を省く（12 節）。M5 の受け入れ条件には要らないので、手を付けていません。ブロックを省く方法は結果を変え、しきい値のキーワードが要ります。採るかどうかは利用者が決めます（1.3 節）。 | 必要になったとき |
| `scf.Generation.Kpoint none` | 既存のコードが、ハイブリッド汎関数でない計算でも異常終了します（13 節の 28）。原因は調べていません。直すまで、対称性で減らしたメッシュでは HSE06 を試験できません。 | 必要になったとき |
| CUDA 版での実行 | この改訂の時点で、ハイブリッド汎関数の計算の CUDA 版での結果はありません。確かめる対象は 3 つです。CUDA 版が通るホスト上の経路（Cluster と、Band のブロック巡回配置）、M6 で書いた行列積の部品、デバイスの密行列ソルバーの経路です。前の 2 つは、計算機が実行している最初の 18 件が確かめます。 | M6（実機） |
| デバイスに送る積の下限と、残す空き | 1e9 回と 512 MiB は最初の値です（11 節）。下限を変えたときの時間と、複数のプロセスが 1 台を共有したときの動作を、実機で測ります。 | M6（実機） |
| デバイスの密行列ソルバーの経路 | M6 の実機の段階で、 $K$ の加算と密度行列の収集を書きました（6.3 節、6.5 節）。実機では実行していません。しきい値を下げずにこの経路に入る大きさ（Band は次数 800 以上、Cluster は 400 以上）の HSE06 は、道具の実行に入れていません。この大きさでの加算と収集の時間も未測定です。 | M6（実機） |
| MAGMA の経路と gpusolver2 の経路 | ハイブリッド汎関数の計算は、この 2 つの経路を使わず、CPU の固有値ソルバーで解きます（6.5 節）。結果は変わらず、対角化が遅くなります。MAGMA の経路に $K$ を足すには、GPU が 2 台以上ある計算機での試験が要ります。 | 必要になったとき |
| 固有値ソルバーの中の GEMMul8 | デバイスの固有値ソルバーは、既定で GEMMul8 を使うデバイスがあります（11 節）。ハイブリッド汎関数の全エネルギーへの影響は未測定です。道具の `NAME-dfp64` の実行との差で測ります。 | M6（実機） |
| GEMMul8 の切り替え | 厳密交換の大きい積で GEMMul8 を使う方法を決めていません（11 節）。案は 2 つです。1 つは、入力に `scf.gemmul8.enable on` を書いたときだけ使う形です。もう 1 つは、厳密交換のためのキーワードを加える形です。既存のキーワードの働きか、キーワードの追加に関わるので、利用者が決めます（1.3 節）。 | 利用者の判断 |
| XC の GPU 経路の時期 | 8 節の表は M6、9 節の M8 の一覧は「GPU での XC の計算」としていて、食い違っています。9 節の M6 の目標は行列積と加算で、XC を含みません。CUDA 版のハイブリッド汎関数の計算は、今はホストのループで XC を計算します。どちらの時期に行うかは利用者が決めます。 | 利用者の判断 |
| NVIDIA の文書で確かめられなかった点 | `cublas<t>gemm` の戻り値の表、`cudaMalloc` の戻り値の一覧、`cudaErrorMemoryAllocation` の説明、`cudaMemGetInfo` の説明は、取得したページが途中で切れていて読めませんでした [18, 19]。メモリーの不足を表す状態コードの扱いは、リポジトリーの既存のコード（`source/openmx_gpu_error.c:35-58`、`source/openmx_gpu_runtime.h:126`）に合わせています。 | M6（実機） |
| 大域状態のゲート（CUDA の構成） | M6 の部品の検査は、クラウドのコンテナの clang 18 と CPU の構成で行いました。CUDA の構成での検査は、NVHPC のある計算機が要ります。 | M6（実機） |

## 付録 A. コード上の位置

`50ef6b1` での場所と、行番号がずれたときの検索語です。`source/Input_std.c` の行番号は、基準コミットの 994 行から 5018 行までが 8 行後ろに、5034 行以降が 7 行前にずれています。基準コミットの 5019 行から 5033 行は削除されました。

| 項目 | 場所 | 検索語 |
|---|---|---|
| `scf.XcType` の読み込み | `source/Input_std.c:992` | `"scf.XcType"` |
| `HSE` と `NLX` の拒否 | `source/Input_std.c:994-1001` | `is not implemented` |
| 入力エラーの書き方 | `source/Input_std.c:1009-1024` | `SpinP_switch should be OFF` |
| 旧い厳密交換のコードの残り | `source/Allocate_Arrays.c:770-778`、`source/Free_Arrays.c:1836-1842` | `case 15:`、`qgrid_Rec_Coulomb` |
| PBE の関数 | `modules/xc/include/openmx/xc/kernels.h:285` | `openmx_xc_kernel_pbe` |
| 交換の係数 `coe` | `modules/xc/include/openmx/xc/kernels.h:300` | `const double coe` |
| XC の CPU 経路 | `source/Set_XC_Grid.c:1274` | `openmx_xc_pbe(ED` |
| XC の GPU 経路の入口 | `source/Set_XC_Grid.c:654` | `Set_XC_Grid_GPU(XC_P_switch` |
| XC の `switch` | `source/Set_XC_Grid.c:1014` | `switch (XC_switch)` |
| HIP 版の PBE の実装 | `source/backends/hip/src/set_xc_grid_hip.cu:119` | `sxg_xc_pbe` |
| SCF ループ | `source/DFT.c:700-1903` | `while (po==0 && SCF_iter<SCF_MAX)` |
| 収束判定 | `source/DFT.c:1424-1427` | `dUele<SCF_Criterion` |
| FP64 での解き直し | `source/DFT.c:1463-1505` | `solved once more in FP64` |
| 状態の取り出し | `source/DFT.c:484-492` | `_Bind_Resources(core)` |
| 力と全エネルギーの呼び出し | `source/DFT.c:2778`、`2819` | `time7 += Force(`、`time8 += Total_Energy(` |
| エネルギーの合計 | `source/DFT.c:2842` | `thermo->total =` |
| Band の密行列の組み立て | `source/Band_DFT_Col.c:6828` | `void Construct_Band_CsHs` |
| Band のデバイス上の組み立て | `source/Band_DFT_Col.c:3236` | `BandCol_ConstructDenseCsHs_OpenACC` |
| 密行列の添字 | `source/Band_DFT_Col.c:1493` | `entry->index0  = (jg - 1) * n + (ig - 1)` |
| ブロック巡回配置の添字 | `source/Band_DFT_Col.c:1501-1529` | `entry->index0  = (jl - 1) * na_rows + il - 1` |
| Band の密度行列 | `source/Band_DFT_Col.c:1949`、`2038` | `BandCol_AccumulateDenseTransposedDM` |
| k 点の生成 | `source/Band_DFT_Col.c:3742`、`3890` | `Shift_K_Point` |
| Cluster のホスト上の組み立て | `source/Cluster_DFT_Col.c:2880` | `Hamiltonian_Cluster_Hs(nh[0],Hs,MP,0,0)` |
| Cluster のデバイス上の組み立て | `source/Cluster_DFT_Col.c:7074` | `ClusterCol_BuildDeviceDenseFromPacked` |
| Cluster の密度行列 | `source/Cluster_DFT_Col.c:4121` | `ClusterCol_AccumulateDM_CPU` |
| 近傍リストの条件 | `source/truncation.c:1485-1513` | `rcut = rcutA + rcutB` |
| 重なり積分の動径核 | `source/Set_OLP_Kin.c:43` | `OLPKin_Radial_Kernel` |
| 力の本体 | `source/Force.c:3055` | `static double Force_Impl` |
| エネルギーの成分 | `core/include/openmx_core_energy.h` | `double vdw;` |
| 計算コンテキスト | `core/src/calculation_context.hpp:29` | `struct OpenMXCoreContext` |
| モジュールの取り込み | `core/CMakeLists.txt:89-96` | `modules/xc` |
| 本体の翻訳単位の一覧 | `source/Makefile:309` | `OBJS    =` |
| GEMMul8 の橋渡し | `source/openmx_common.h:3747` | `openmx_gemmul8Dgemm(` |
| 密度行列の GPU 計算 | `source/openmx_gpu_dm.h:17` | `openmx_gpu_dm_entries` |
| デバイス上の行列演算 | `source/openmx_gpu_transform.h` | `OPENMX_GPU_ADD` |
| 大域状態のゲート | `docs/GLOBAL_STATE_AUDIT.md`、`tools/audit_global_usage.py` | `--check-no-new` |
| CPU 版の CI | `.github/workflows/core.yml:113-136` | `cpu-application` |

M4 で加えたコードの場所です。行番号は `b8126f8` のものです。

| 項目 | 場所 | 検索語 |
|---|---|---|
| 結合層 | `source/Exact_Exchange.c`、`source/openmx_exact_exchange.h` | `Exact_Exchange_` |
| キーワードの読み込み | `source/Input_std.c:1047` | `"scf.exx.alpha"` |
| 拒否する組み合わせ | `source/Input_std.c:5125-5148` | `conflict[conflicts++]` |
| キーワードの値と構造の配列を層に渡す | `source/Input_std.c:5176`、`5185` | `Exact_Exchange_Configure(`、`Exact_Exchange_Bind_Structure(` |
| 構造ごとの準備 | `source/DFT.c:678` | `Exact_Exchange_Prepare(exact_exchange` |
| 内側の SCF の停止条件 | `source/DFT.c:1461` | `Exact_Exchange_Inner_Stop_Allowed(` |
| 収集のための解き直しと密度行列の照合 | `source/DFT.c:1600` | `Exact_Exchange_Collect_Begin(` |
| 交換行列の構築と外側反復の判定 | `source/DFT.c:1634` | `Exact_Exchange_Build(exact_exchange` |
| 混合の始め直し | `source/DFT.c:1648` | `Exact_Exchange_Mixing_Offset(` |
| `scf.maxIter` の数え方 | `source/DFT.c:2004` | `DFTSCF_loop + exx_inner_start` |
| 外側反復の結果の表示 | `source/DFT.c:2034` | `The outer iteration converged` |
| 全エネルギーの成分 | `source/DFT.c:2987`、`source/outputfile1.c:79` | `components->exact_exchange` |
| 密行列への加算 | `source/Cluster_DFT_Col.c:2893`、`2900`、`3291` | `Exact_Exchange_Add_Cluster_Hs` |
| 固有ベクトルと占有数の受け渡し | `source/Cluster_DFT_Col.c:4297` | `Exact_Exchange_Collect_Gamma` |
| デバイス上の密行列の経路を避ける条件 | `source/Cluster_DFT_Col.c:2855`、`2870` | `exact_exchange==NULL` |
| 準局所の交換の引数 | `source/Set_Vpot.c:43`、`source/Total_Energy.c:207` | `exx_alpha` |
| M4 の試験の実行と集計 | `tools/exx_scf_checks.py` | `def table` |

M5 で加えたコードの場所です。行番号は `f2d3c9e` のものです。このコミットでは、M4 の表の行が後ろにずれています。`source/DFT.c` は 3 行から 14 行、`source/Input_std.c` の 5176 行と 5185 行は 8 行です。

| 項目 | 場所 | 検索語 |
|---|---|---|
| ブロックの一覧 | `modules/exx/include/openmx/exx/exchange.hpp:99` | `struct BlockLayout` |
| site の形式の C の入口 | `modules/exx/include/openmx/exx/exx.h:149-180` | `openmx_exx_prepare_blocks`、`openmx_exx_add_blocks`、`openmx_exx_symmetrize_blocks` |
| 拒否する組み合わせ | `source/Input_std.c:5125-5156` | `conflict[conflicts++]` |
| Band ソルバーへの状態の受け渡し | `source/DFT.c:498`、`source/Band_DFT_Col.c:518` | `Band_DFT_Col_ExactExchange` |
| 行列の形の選択 | `source/DFT.c:681` | `Exact_Exchange_Prepare(exact_exchange, atomic, Solver==3` |
| ブロックの一覧の取り込み | `source/Exact_Exchange.c:240` | `ExactExchange_List_Blocks` |
| k 点ごとの収集 | `source/Exact_Exchange.c:483`、`source/Band_DFT_Col.c:5600`、`6396` | `Exact_Exchange_Collect_K` |
| 密度行列の照合 | `source/DFT.c:1632`、`source/Exact_Exchange.c:568` | `Exact_Exchange_Density_Deviation(` |
| 構築と、対になるブロックの平均 | `source/Exact_Exchange.c:612`、`202` | `Exact_Exchange_Build(`、`ExactExchange_Symmetrize_Channel` |
| $H(\mathbf k)$ への加算 | `source/Exact_Exchange.c:797`、`source/Band_DFT_Col.c:4710`、`6196` | `Exact_Exchange_Add_Band_Hs` |
| デバイス上の密行列の経路を避ける条件 | `source/Band_DFT_Col.c:3636`、`6866` | `state->exact_exchange == NULL` |
| M5 の試験の実行と集計 | `tools/exx_kpoint_checks.py` | `def table` |

M6 で加えたコードの場所です。行番号は `4f73522` のものです。このコミットでは、M4 と M5 の表の行のうち、`source/Exact_Exchange.c` の 91 行より後ろと `source/DFT.c` の 2031 行より後ろがずれています。`source/Input_std.c` は、5184 行より後ろが 1 行ずれています。

| 項目 | 場所 | 検索語 |
|---|---|---|
| デバイスでの行列積 | `source/Exact_Exchange_Device.c:131` | `int Exact_Exchange_Device_Dgemm(` |
| 2 つの限度の最初の値 | `source/Exact_Exchange_Device.c:42-43` | `minimum_work = `、`reserve_bytes = ` |
| 容量が足りない場合と、そのほかの失敗 | `source/Exact_Exchange_Device.c:93`、`102` | `ExactExchange_Device_Decline`、`ExactExchange_Device_Fail` |
| ハンドルの用意と返却 | `source/Exact_Exchange_Device.c:115`、`63` | `ExactExchange_Device_Handle`、`Exact_Exchange_Device_Release` |
| 状態の構造体と失敗の種類 | `source/openmx_exact_exchange_device.h` | `OpenMXExactExchangeDevice` |
| 関数表の `dgemm` | `source/Exact_Exchange.c:245` | `static int ExactExchange_Dgemm` |
| 構築 1 回分の集計と表示 | `source/Exact_Exchange.c:689`、`745`、`751` | `ExactExchangeProducts products=`、`failed with status`、`matrix products of process 0` |
| 全プロセスの合計 | `source/Exact_Exchange.c:157`、`source/DFT.c:2035`、`2058` | `Exact_Exchange_Device_Use` |
| 障害注入の受け渡し | `source/Input_std.c:5184`、`source/Exact_Exchange.c:96` | `device_capacity_test` |
| ビルドの一覧 | `source/Makefile:316`、`719`、`tools/check_gpu_compilation.py:73` | `Exact_Exchange_Device` |
| 模型の試験 | `tests/run_exx_device_product.py`、`tests/exx_device_product_harness.c`、`tests/fixtures/exx_device_sdk/` | `exx device product` |
| 試験の登録 | `CMakeLists.txt:206`、`.github/workflows/core.yml:271` | `exx-device-product`、`run_exx_device_product.py` |
| 実機の試験の実行と集計 | `tools/exx_gpu_checks.py` | `def table` |

M6 の実機の段階で加えたコードの場所です。行番号は `c6cdb9b` のものです。M4 と M5 の表の「デバイス上の密行列の経路を避ける条件」は、このコミットで外しました。同じ場所の条件は、しきい値と `scf.eigen.lib` だけを見ます。

| 項目 | 場所 | 検索語 |
|---|---|---|
| Band のデバイス上の加算 | `source/Band_DFT_Col.c:3409`、呼び出しは `4547`、`4693`、`6121` | `BandCol_DenseAddExactExchange` |
| Band の密行列の経路の収集 | `source/Band_DFT_Col.c:3449`、呼び出しは `5601`、`5605`、`6237` | `BandCol_DenseCollectExactExchange` |
| Band の容量の判定と切り替え | `source/Band_DFT_Col.c:1012`、`1076` | `MAGMA does not take the exchange matrix`、`gpusolver2_flag \|\| state->exact_exchange != NULL` |
| Band の FP64 の固有値ソルバー | `source/Band_DFT_Col.c:459` | `possible=possible_kdense=possible_staged=0` |
| Cluster のデバイス上の加算 | `source/Cluster_DFT_Col.c:2200`、呼び出しは `2362`、`2414` | `ClusterCol_DenseAddExactExchange` |
| Cluster の容量の判定と切り替え | `source/Cluster_DFT_Col.c:1705` | `the device cannot hold the dense matrices` |
| Cluster の FP64 の固有値ソルバー | `source/Cluster_DFT_Col.c:399` | `empty_states_flag == 0 && state->exact_exchange == NULL` |
| デバイスでの加算の記録と合計 | `source/Exact_Exchange.c:168`、`175`、`source/DFT.c:2037`、`2069` | `Exact_Exchange_Device_Addition` |
| 収集の有無の問い合わせ | `source/Exact_Exchange.c:506` | `Exact_Exchange_Collecting` |
| 道具の密行列の経路の実行 | `tools/exx_gpu_checks.py:110-117` | `STEERED = ` |

## 付録 B. 方針文書から変えた点

`openmx_hybrid_exx_plan.md` を作った後の検討で、次の点を変えました。

| 項目 | 方針文書 | この文書 |
|---|---|---|
| ハミルトニアンに足す量 | 射影演算子 $K^{\mathrm{proj}}=AM^{-1}A^{\dagger}$ | 全交換行列 $K$ |
| 構築の入力 | 占有軌道と参照軌道の係数 | 密度行列 $D(\mathbf R)$ |
| 構築の手順 | 軌道のバッチごとの 4 段（S1 から S4） | 原子ブロックの 4 項（5.5 節） |
| k 点 | $\mathbf k$ と $\mathbf q$ の 2 重の和 | 実空間で作り、式 (14) で変換 |
| 演算量（Si512、Γ 点） | 9.4e13 回の積和 | 5.5e12 回の積和 |
| キーワード | `scf.exx.projector.extra`、`scf.exx.batch`、`scf.exx.fullK`、`scf.exx.c.threshold` | 使いません |
| 行列 $V$ の半正定値性 | コレスキー分解に必要 | 不要 |
| 力 | 軌道の形式（方針文書の式 (17)） | 密度行列の形式（式 (16)、(17)） |

準局所部分、LRI、二中心積分、カーネルの定義は変えていません。式 (8) の分解は、方針文書への補遺として受け取った提案（`openmx_hybrid_exx_acceleration.tex`）の式を採用したものです。

## 付録 C. 改訂の記録

| 改訂 | 日付 | 内容 |
|---|---|---|
| 初版 | 2026-11-23 | `d5ebd59` でリポジトリーに入った版です。 |
| 改訂 2 | 2026-11-23 | M0 の結果の反映と、M1 の書き直しです。行番号の基準を `50ef6b1` に変えました（変わったのは `source/Input_std.c` の参照だけです）。試作用の入力の削除は、8 節と付録 A に書き込み済みです。M1 では、短距離部分の定義を libxc 7.1.2 に合わせ、Quantum ESPRESSO 7.6 との違いを数値で示しました。受け入れ基準は、付属の `wpbeh_reference.py` とフィクスチャーに対する許容差に変えてあります。初版の基準（libxc と相対 1e-10）は、 $w>4$ の領域では満たせないと考えられます。式を 16 桁で評価したときの丸め誤差が、 $w=6$ で 5e-9 あるためです。規約の表は PBE の交換について数値で確かめ、進捗ファイルの例のブランチ名も `feature/exx` に合わせました。 |
| 改訂 3 | 2026-11-23 | 利用者の指示で、ωPBE の短距離交換の定義を libxc 7.1.2 の形から Quantum ESPRESSO 7.6 の形に改めました。論文の丸めた定数、 $s\le0.08$ の多項式、 $s>8.3$ の付け替えを使います。libxc と PySCF を使う手順、出力の対応表、ライセンスの選択肢を外し、Quantum ESPRESSO のルーチンをコンパイルして比べる手順に置き換えました。参照実装とフィクスチャーも作り直してあります。1.2 節には、libxc を使わない条件を加えました。変えた節は 1.2、5.1、7.5、M1、10.4、13、14 と参考文献です。文献 [11] と [15] は別の文献に差し替え、[17] は欠番にしました。同じ日に、13 節へ試験用の `Set_XC_Grid` の代役の項（16）を加えました。 |
| 改訂 4 | 2026-11-23 | M2 の結果の反映です。7.1 節に例題と追加の試験、10.1 節にカーネルと重なり行列のフィクスチャーを作る手順を書きました。12 節には、ABF の個数の実測値と、それに合わせた容量の計算値を加えています（表は 32 個のままです）。13 節には 17 から 20 の 4 項を加えました。14 節では、ABF の個数と動径の格子を「済み」にし、LRI の精度に実測値を書き、 $\kappa$ の刻みの項を加えました。 |
| 改訂 5 | 2026-11-23 | M2 の表の `exx-lri-identity` の閾値を、利用者の決定に合わせました。結合距離の 2 原子の場合は、展開した密度の差をクーロンノルムで 1e-10、係数ごとの差を 1e-8 で判定します。1 原子の場合と離れた 2 原子の場合は、係数ごとに 1e-10 のままです。 |
| 改訂 6 | 2026-11-23 | M3 の結果の反映です。6.4 節に、実装が擬似コードと違う 3 点を書きました（(iii) と (iv) をまとめた変換、係数を掛ける時期、Γ 点の形式で周期像を行列積の前にまとめること）。7.1 節に M3 の試験、7.2 節に `openmx_exx_reference_dgemm` と `DenseBackend`、7.4 節に M3 で入れた C の入口の範囲を加えています。7.3 節の T ブロックの形は、原子ごとに横に並べた 1 つの配列に改めました。12 節には構築の時間の実測値を加え、13 節の 5 に周期像をまとめる和の注意を足しました。 |
| 改訂 7 | 2026-11-23 | M4 の結果の反映です。2.2 節に、Band ソルバー（M5 まで）と、初版になかった 6 つの拒否を加えました。6.6 節には、M4 の実装を書きました。混合の始め直し、内側の SCF の停止条件、未収束の扱い、`scf.maxIter` の数え方、バンドエネルギー、密度行列の照合です。7.4 節に結合層 `source/Exact_Exchange.c` の関数、8 節に M4 で組み込んだ範囲を書いています。9 節の M4 には試験の道具 `tools/exx_scf_checks.py`、12 節には M4 の時間の実測値、13 節には 21 から 25 の 5 項、付録 A には M4 で加えたコードの場所を加えました。14 節では、混合の履歴を「済み」にし、LRI の精度、ABF のしきい値、V 近傍の打ち切り、外側反復の回数に M4 の実測値を書いています。 |
| 改訂 8 | 2026-11-24 | M5 の結果の反映です。2.2 節では、Band ソルバーの行を外し、キーワードを 4 つ加えました。6.3 節に k 点ごとの収集、6.4 節にブロックを 1 本の配列に並べる形、6.5 節に Band ソルバーでの加算を書いています。6.6 節には site の形式の密度行列の照合、6.7 節には M5 の MPI の扱いを加えました。7.3 節のデータ構造と 7.4 節の入口は、M5 の形に改めています。8 節には、M5 で組み込んだ範囲を加えました。9 節の M5 には、試験の道具 `tools/exx_kpoint_checks.py` と入力の作り方を書いています。12 節には、k 点の計算の時間、flop 数、メモリーの実測値を加えました。13 節には、24 への追記と、26 から 31 の 6 項があります。14 節では、V 近傍の打ち切りと外側反復の回数を「済み」にし、LRI の精度、金属、メモリーに M5 の実測値を書きました。打ち切りクーロンの刻みの確認は M8 に移し、site の形式の構築の高速化と `scf.Generation.Kpoint none` の 2 行を加えています。付録 A には、M5 で加えたコードの場所を加えました。 |
| 改訂 9 | 2026-11-24 | M6 のうち、GPU の実機なしで進めた部分の反映です。M6 は完了していません。6.5 節に、デバイスの密行列ソルバーの経路を変えていないことを書きました。7.4 節に行列積をデバイスに送る部品、8 節に M6 で組み込んだ範囲を加えています。9 節の M6 には、実機の要る部分と要らない部分の区分、加算を書かなかった理由、試験の道具 `tools/exx_gpu_checks.py` を書きました。11 節には、部品の条件と手順、失敗の扱い、行列積の大きさの実測、今の部品で減らせる時間の上限、GEMMul8 と `openmx_gpu_task_execute` を使わなかった理由、模型の試験を加えています。12 節の作業領域は、実測した列数で計算し直しました（1,163 MB を 944 MB に、293 MB を 255 MB に）。13 節には 32 から 37 の 6 項を加えました。14 節では、GPU の速度とメモリーの行を改め、7 行を加えています。8 節の XC の GPU 経路の行には、9 節の M8 との食い違いを書きました。付録 A に M6 で加えたコードの場所、参考文献に [18] と [19] を加えています。 |
| 改訂 10 | 2026-11-24 | M6 の実機の段階で書いた、デバイスの密行列ソルバーの経路の反映です。実機では実行していないので、M6 は完了していません。6.3 節に、この経路での収集と、FP64 の固有値ソルバーだけを使うことを書きました。6.5 節には、表の 3 行のうち 2 行（Band と Cluster のデバイス上）に加算を書いたことと、その手順を書いています。残りの 1 行（MAGMA の経路）には書かず、ハイブリッド汎関数の計算を CPU の固有値ソルバーに回す形にしました。改訂 9 は、3 行とも書くとしていました。7.4 節に結合層の 3 つの関数、8 節の M6 の表に 4 行を加えています。9 節の M6 には、状態の表の更新、実機での順序、道具に加えた実行（49 件）を書きました。11 節には、`openmx_gpu_transform` を使わなかった理由、固有値ソルバーの中の GEMMul8 の扱い、標準出力の行を加えています。13 節には 38 から 42 の 5 項を加えました。14 節では、CUDA 版での実行とデバイスの密行列ソルバーの経路の行を改め、2 行を加えています。付録 A に、実機の段階で加えたコードの場所を加えました。 |

## 参考文献

1. 尾崎泰助，「射影演算子法による Hatree-Fock 交換ポテンシャルの高速計算」（題名の表記は原文のまま），日本物理学会 第 79 回年次大会（2024 年）講演概要集，16aN101-11，p. 2521．doi:10.11316/jpsgaiyo.79.2.0_2521
2. L. Lin, J. Chem. Theory Comput. **12**, 2242 (2016).
3. S. Manzer, P. R. Horn, N. Mardirossian, and M. Head-Gordon, J. Chem. Phys. **143**, 024113 (2015).
4. A. C. Ihrig et al., New J. Phys. **17**, 093020 (2015).
5. T. Ozaki and H. Kino, Phys. Rev. B **72**, 045121 (2005).
6. P. Lin, X. Ren, and L. He, J. Chem. Theory Comput. **17**, 222 (2021); arXiv:2009.12505.
7. P. Lin, X. Ren, and L. He, J. Phys. Chem. Lett. **11**, 3082 (2020).
8. J. Heyd, G. E. Scuseria, and M. Ernzerhof, J. Chem. Phys. **118**, 8207 (2003); erratum **124**, 219906 (2006).
9. M. Ernzerhof and J. P. Perdew, J. Chem. Phys. **109**, 3313 (1998).
10. A. V. Krukau, O. A. Vydrov, A. F. Izmaylov, and G. E. Scuseria, J. Chem. Phys. **125**, 224106 (2006).
11. J. Heyd, "Screened Coulomb Hybrid Density Functionals", PhD thesis, Rice University (2004), <https://scuseria.rice.edu/preprints/J_Heyd_Thesis.pdf>（2.2.2 節の式 (2.16) と (2.17)、2.2.3 節の式 (2.20) を確認しました。付録 A の Fortran のコードは、全文を確認していません）。
12. ABACUS documentation, "Full List of INPUT Keywords", <https://abacus.deepmodeling.com/en/latest/advanced/input_files/input-main.html>（`exx_pca_threshold`、`exx_separate_loop`、`exx_cs_inv_thr` の項）。
13. J. Spencer and A. Alavi, Phys. Rev. B **77**, 193110 (2008).
14. Claude Code documentation, "How Claude remembers your project", <https://code.claude.com/docs/en/memory>（2026-11-23 に確認）。
15. VASP Wiki, "HFSCREEN", <https://vasp.at/wiki/HFSCREEN>（2026-11-23 に確認）。
16. Quantum ESPRESSO 7.6, <https://github.com/QEF/q-e>（タグ `qe-7.6`、コミット `9f93dde`。`XClib/qe_funct_exch_gga.f90` の 616〜669 行と 1661〜2255 行、`XClib/qe_drivers_gga.f90` の 174〜183 行と 610〜640 行、`PW/Doc/INPUT_PW.def` の 1467〜1478 行）。
17. （欠番。改訂 3 で外しました）
18. NVIDIA, "cuBLAS" (CUDA Toolkit Documentation), <https://docs.nvidia.com/cuda/cublas/index.html>（2026-11-24 に確認。`cublasCreate` と `cublasDestroy` の項、cuBLAS のコンテキストと現在のデバイスの項、`cublasStatus_t` の `CUBLAS_STATUS_ALLOC_FAILED` の説明を読みました。`cublas<t>gemm` の戻り値の表は読めていません）。
19. NVIDIA, "CUDA Runtime API", <https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__ERROR.html>（2026-11-24 に確認。`cudaGetLastError` の項を読みました。メモリー管理の項と状態コードの項は、ページが途中で切れて読めていません）。
