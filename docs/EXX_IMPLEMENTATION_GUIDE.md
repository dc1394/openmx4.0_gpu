# OpenMX 4.0 GPU ハイブリッド汎関数（HSE06・PBE0）実装指示書

OpenMX 4.0 GPU（`dc1394/openmx4.0_gpu`）に HSE06 と PBE0 の厳密交換を実装する作業者（Claude Code）への指示書です。設計は決まっています。作業者は 9 節のマイルストーンを番号順に実装します。

| 項目 | 内容 |
|---|---|
| 文書の改訂 | 改訂 32（2026-11-27）。PBE0 と HSE06 の MO 出力と、CPU および CUDA の MO 出力の並列化を追加しました（6.5 節）。改訂 31 では、結果を SCF の判定の範囲で変える高速化を入れました（6.6 節、9 節の「力のデバイス化とほかの高速化」）。構築の後の混合は、RMM-DIIS、GR-Pulay、RMM-ADIIS で 1 ステップ早く始め直します。既定の動作を変えるのは、この変更だけです。キーワードで、 $K$ の外挿と、内側の SCF を早めに止める判定を選べます。改訂 30 では、M8 の項目のうち、Band ソルバーのバンド分散と DOS の出力を入れました（6.5 節、9 節の M8）。SCF の後に、V 近傍をまとめない系で $K(\mathbf R)$ を 1 回作り、出力の k 点の $H(\mathbf k)$ に足します。改訂 29 からは、デバイスが呼び出しを引き受けるかを、全ての積の乗算の和で決めています（11 節）。改訂 28 では、正則な k メッシュの計算で、構築の列の site もメッシュの周期でまとめました（6.4 節）。改訂 27 では、周期の整数倍だけ離れた V 近傍を、1 つの site にまとめました。改訂 26 では、力の重みを半分の構築の形で計算し、準備と力の対のループをスレッドに分けています。改訂 25 では、力の重みのデバイスでの計算、 $K(\mathbf k)$ の保持、交換行列の半分の構築を入れました。球ベッセル関数の上向きの漸化式、準局所部分の GPU での計算、ABF の時間による分け方も、改訂 25 からです。近傍リストの形式で $H$ を読むほかの出力は、まだ拒否します（2.2 節）。 |
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
| 固有値ソルバー | Cluster、Band | DC、DC-LNO、Krylov、Cluster-DIIS、Cluster2、Cluster-LNO | Recursion、GDC、EGAC、NEGF |
| k 点 | Γ 点、一般の k 点 | なし | なし |
| 物理量 | 全エネルギー、固有値、力、MO 出力 | 応力、バンド分散、状態密度、`HS.fileout` | なし |
| 実行環境 | CPU、CUDA | HIP | なし |
| 浮動小数点 | FP64 | 混合精度 | なし |

### 2.2 入力の時点で拒否する組み合わせ

ハイブリッド汎関数を指定した入力が次のどれかに当たる場合は、理由を表示して終了します。書き方は既存のエラー処理（`source/Input_std.c:1009-1024`）と同じです。対応が済んだ項目は、表から外します。

| 条件 | キーワード | 外す時期 |
|---|---|---|
| Cluster、Band、DC、DC-LNO、Krylov 以外のソルバー | `scf.EigenvalueSolver` | M8 以降 |
| ノンコリニア | `scf.SpinPolarization NC` | M8 |
| 近傍リストの形式のハミルトニアンを読む出力と、書き出す出力 | `HS.fileout`、`Wannier.Func.Calc`、`Unfolding.Electronic.Band`、`scf.Polarization.ABC`、`CWF.Calc`、`DM.specify.energy.range`（`DM.export` と一緒に指定した場合）、`NEGF.Output_HKS` | M8 |
| バンド分散、DOS、COHP の組み合わせ | Cluster、DC、DC-LNO、Krylov の `Band.dispersion`、`LNO.flag` と一緒の `Band.dispersion`、Cluster ソルバーで `Dos.Kgrid` が 1 1 1 でない `Dos.fileout` と `DosGauss.fileout`、`COHP.calc.flag` | M8 以降 |
| 局所ソルバーの一回対角化の再開 | DC、DC-LNO、Krylov の `scf.restart pp` | EXX を含む再開形式の実装後 |
| 厳密交換を含まないエネルギーの分解 | `Energy.Decomposition` | M8 |
| 確認していない機能 | `scf.Hubbard.U`、`orbitalOpt.Method`、`scf.core.hole`、`ESM.switch`、`scf.Electric.Field`、`CDDF.start`、`NBO.switch`、`XANES.gs.fileout`、`XANES.calc` | 個別に確認した後 |
| HIP のビルド | なし（ビルド時の種別で判定） | M8 |

出力系を拒否する理由は 13 節の 1 番に書きます。

M4 の実装（`source/Input_std.c`）は、当たる条件を全て並べて表示してから終了します。Γ 点だけの入力で Band を指定した場合は、本体が Cluster に切り替えます。`DosGauss.fileout`、`scf.Polarization.ABC`、`CWF.Calc`、`Energy.Decomposition`、`CDDF.start`、`NBO.switch` は、初版の表になかった項目で、M4 で加えました。`DosGauss.fileout` は `Dos.fileout` と同じ出力の別の方式です。残りの 5 つを拒否してよいかは、利用者の確認を待っています（進捗ファイルの「利用者への確認待ち」）。`PBE0` は、M8 の `c9e26e7` から受け付けます。それまでは「実装していない」と表示して終了していました。HIP 版のビルドは、HSE06 と同じく PBE0 も拒否します。

M5 で、Band ソルバーの行を表から外しました。k 点の生成は、`scf.Generation.Kpoint` の `GCenter`（既定）と `REGULAR` で確かめています。`none`（対称性で減らした Monkhorst–Pack のメッシュ）は、ハイブリッド汎関数でない計算でも動きません（13 節の 28）。M5 では、表にキーワードを 4 つ加えました。`DM.specify.energy.range` は、SCF の後に近傍リストの形式の $H$ をもう 1 回解き、エネルギーの範囲を限った密度行列を作ります。`NEGF.Output_HKS` は、電極用のファイルに同じ形式の $H$ を書きます。どちらの $H$ も $K$ を含みません。`XANES.gs.fileout` と `XANES.calc` の処理は、k メッシュの計算でも Cluster ソルバーを呼びます。site の形式の $K$ は、この呼び出しに使えません。Cluster ソルバーの計算でも結果を確かめていないので、2 つともソルバーによらず拒否します。XANES の 2 つを拒否してよいかは、M4 の 5 つと合わせて利用者の確認を待っています。

力と応力の両方に厳密交換を含めます。固定セルの構造最適化と NVE・NVT に加えて、`OptC`・`RFC` の可変セル最適化、`scf.stress.tensor on`、NPT に対応します。応力の対ごとの寄与は、表示する体積で割る前に、変位ベクトル $d$ と交換エネルギーの変位微分 $g$ の外積 $g_i d_j$ を足します。同一原子の周期像も含みます。PBE0 の自動切断半径には、対角成分ごとに $(R_c/3)(\partial E_x/\partial R_c)$ を加えます。

M8 で、`Band.dispersion`、`Dos.fileout`、`DosGauss.fileout`、`MO.fileout` を表から外しました（9 節の M8）。Band ソルバーの計算では、出力の k 点の $H(\mathbf k)$ に $K(\mathbf k)$ を足します（6.5 節）。Cluster ソルバーの SCF は Γ 点の密行列を使います。周期系の MO 出力は最後に使った密度行列から $K(\mathbf R)$ を構築して、指定した k 点を解きます。Cluster ソルバーのバンド分散と、`Dos.Kgrid` が 1 1 1 でない DOS は引き続き入力で拒否します。`Dos.Kgrid` が 1 1 1 の DOS は、SCF と同じ関数が $K$ を足して解き直すので、拒否しません。LNO のバンドと COHP は、 $K$ のない近傍リストの形式の $H$ を読みます。COHP は DOS の関数の中で計算するので、この版で `COHP.calc.flag` を表に加えました。

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

式 (5) の位相は、必要な $\mathbf R$ ごとに実際の k 点の座標（`T_KGrids1` から `T_KGrids3`）で計算します。 $D(\mathbf R)$ を Born–von Kármán のスーパーセルの周期で折り返してはいけません。メッシュがずれていると、 $D(\mathbf R)$ はスーパーセルの並進で位相が付くためです。OpenMX の `REGULAR` メッシュは、分割数が偶数のとき半格子分ずれた点を使います（`Band_DFT_Col.c:3742`）。ずらしたメッシュでも式 (3) と式 (5) の組が k 空間の定義と一致することは、数値で確認済みです。改訂 27 で加えた V 近傍のまとめ（6.4 節）も、 $D(\mathbf R)$ は折り返しません。周期の並進で $D$ に付く符号は $K(\mathbf k)$ の位相にも付き、2 つが打ち消すことを使います。改訂 28 の列のまとめ（6.4 節）も同じで、周期だけ離れた列の $D$ は、1 つの列に符号を掛けて読みます。

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

式 (12) と 4 つのカーネルは、ガウス関数の解析値と比べて確認済みです（10 節）。PBE0 の $R_c$ は、半径 $R_c$ の球の体積が $N_k\Omega$ に等しくなるように決めます [13]。 $\Omega$ は単位胞の体積です。これより大きい値を書いた入力は、拒否します（13 節の 72）。

式 (12) の $\kappa$ の積分は台形則です。格子は、上限 $\kappa_{\max}=\sqrt{\texttt{1DFFT.EnergyCutoff}}$ を `1DFFT.NumGridR` 個の区間に分けます。既定では 60/bohr と 900 個です。切断クーロンでは、半径 $r_P$ と $r_Q$ の関数の被積分関数が、指数型 $r_P+r_Q+R_c+d$ の整関数になります。このときの台形則は、 $r_P+r_Q+R_c+d<2\pi/\Delta\kappa$ なら、上限より先の部分を除いて正確です（13 節の 19）。 $\Delta\kappa$ は刻みで、既定の格子では右辺が 94.2 bohr になります。V ブロックの $d$ は、「大きい方の ABF の半径 + $v$」まで届きます。 $v$ は V の範囲で、`scf.exx.v.cutoff` を書かない PBE0 では $R_c+r_{\max}$ です。 $r_{\max}$ は、構造の中で最大の ABF の半径です。そこで PBE0 の区間の数は、 $\max\left(\texttt{NumGridR},\lfloor\kappa_{\max}(R_c+3r_{\max}+v)/2\pi\rfloor+1\right)$ にします。この形は、利用者が 2026-11-25 に決めました。既定の $R_c$ で区間が増えるのは、Si7.0 の 9×9×9 の k 点のような計算です。

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

式 (16) と式 (17) は、中心差分と比べて確認済みです（10 節の T6）。M7 で、中心の原子ごとの形と site の形式に書き換えました（9 節の M7）。原子の位置についての微分は連鎖律で得ます。対ごとの係数ベクトル $c$ （2 つの原子の ABF についての係数を並べたもの）の変分は、式 (7) から次の式になります。

$$\delta c=G^{-1}\left(\delta b-\delta G\thinspace c\right) \tag{18}$$

$b$ と $G$ と $V$ は二中心積分なので、微分は 2 原子の相対座標についての微分です。軌道の応答は、既存の Pulay 項（エネルギー重み付き密度行列と重なり行列の微分）が担います。

本体の力は、全エネルギーの原子の位置についての微分です（`Gxyz[a][17]` から `[19]` は $\partial E/\partial\tau$ で、力はその符号を変えた値です）。厳密交換が関わる分は 2 つあります。1 つ目は、密度行列を固定した $\partial E_x/\partial\tau$ です。M7 で、結合層が計算し、`Force` が足す形にしました（7.4 節）。2 つ目は、Pulay 項の中の $K$ の寄与です。エネルギー重み付き密度行列は $H+K$ の固有値と固有ベクトルから作られるので、既存のコードのままで $K$ を含みます。1 つ目は最後の構築に使った密度行列を、2 つ目は内側の SCF の最後の密度行列を使います。外側反復が収束していないと、2 つの密度行列が食い違い、力が全エネルギーの微分からずれます（9 節の M7 の失敗例）。

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

複数のプロセスの計算では、T ブロックと V ブロックの対をプロセスに配ります（改訂 21。6.7 節）。ABF の生成、近傍リスト、動径の変換は、全てのプロセスが同じ計算をします。

ABF の半径は PAO のカットオフ半径と同じです。積 $g_k$ の角運動量 $L$ は、2 つの PAO の角運動量 $l_1$ と $l_2$ から $\lvert l_1-l_2\rvert\le L\le l_1+l_2$ かつ $l_1+l_2+L$ が偶数の範囲を動きます。s2p2d1 の基底では、間引く前の動径関数は原子あたり 22 個、 $m$ を数えた ABF は 78 個です（計算値）。

### 6.3 密度行列の収集

内側の SCF が収束したら、Cluster と Band は収集を有効にして固有値ソルバーをもう 1 回呼びます。FP32 の精製ステップで収束した後に FP64 で解き直す既存の流れ（`DFT.c:1463-1505`）と同じ形です。ソルバーは、スピンと k 点ごとに固有ベクトル、占有数、重みを EXX の状態に渡します。

| 場合 | 保持する量 | 計算 |
|---|---|---|
| Γ 点だけ | 密行列 $D^{\sigma}=cFc^{\mathsf T}$ | GEMM 1 回 |
| k 点 | ブロック $D^{\sigma}_{xy}(\mathbf R)$ の表 | k 点ごとに GEMM 1 回。式 (5) の位相を掛けて、表の全てのブロックに足します。 |
| DC、DC-LNO、Krylov | PAO 近傍の $D^{\sigma}_{xy}(\mathbf R)$ | 混合前の密度行列から各 MPI ランクが担当する原子のブロックを集めます。 |

局所ソルバーでは、PAO の第 1 近傍リストに含まれる実空間ブロックへの射影を $P$ とします。収集した $PD$ は反対向きのブロックと転置平均し、周期セルを保持します。EXX は $E_x[PD]$ とその原子位置微分を求め、配位に対応する $PK$ を作ります。CPU は AVX2 または AVX-512 と OpenMP で各ランクの $H+PK$ を組み、局所ソルバーが MPI でそのハローを交換します。CUDA の局所固有値ソルバーと EXX カーネルもこの行列の射影を使います。Krylov の埋め込み行列は、外側の反復で $K$ が更新された次のステップでも組み直します。PBE0 の切断半径の既定値には単位胞の体積を使います。

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

実装（`modules/exx/src/exchange.cpp`）は、擬似コードと次の 7 点が違います。結果は丸め誤差の範囲で同じです。

- (iii) の括弧の中身を `Rm` の $B$ の列に足し、最後の `Tst^T @ Rm` で (iii) と (iv) を一度に変換します。
- V 近傍ごとの 4 つの項は、中心の原子の ABF を 1 個ずつ計算します。M6 で、この形に書き直しました。`W` のうち ABF $P$ の部分は、 $u_b\times\ell_B$ の行列です。行列積は、この行列をそのまま、または転置して受け取ります。`G` と `Rm` は、2 つの大きい積の間だけ、 $(P,\text{列},\mu)$ の順に並べます。 $P$ 1 個の部分が 1 続きのメモリーになり、1 つの列の $u_a$ 個の要素が隣り合います。 $\mathrm{loc}(B)$ の列は、番号が続く列をまとめて写します。行列積は、中心の原子 1 個につき 2 回と、V 近傍 1 個につき $1+4n_P$ 回です。`a7fbfb8` までの実装は、全ての ABF を 1 回の積で計算し、V 近傍 1 個につき 5 回でした。その形では、(i) と (iv) が `W` の添字を入れ替えた複製を使い、`G` と `Rm` の全体から要素を 1 個ずつ読み書きします。時間の比較は 12 節にあります。
- 中心の原子の ABF は、番号が続く区間に分けて計算できます。関数表に `parallel_for` と `threads` を与えたときの形で、改訂 14 で加えました（7.2 節）。区間の数は、`threads` と $n_P$ の小さい方です。`G` と `Rm` の並べ替え、`Rm` への 0 の書き込み、V 近傍ごとの 4 つの項を、区間ごとに別のスレッドで進めます。作業用の配列は、区間ごとに持ちます。(i) と (ii) は ABF についての和です。区間ごとに和を作り、区間の番号の順に足します。`W` の積と 2 つの大きい積は、呼び出したスレッドで実行します。1 本の配列の site の形式（`BlockLayout`）と Γ 点の形式では、 $D$ の行の収集と $K$ への加算も、同じ数の区間に分けます（改訂 15）。収集は、 $\mathrm{loc}(A)$ の site（Γ 点の形式では軌道）で分けます。site の形式の加算は、行き先のブロックの一覧での番号を、区間の数で割った余りで分けます。 $\mathrm{loc}(A)$ に同じ原子の周期像が 2 つあると、2 つの site の組が $K$ の同じブロックに足すためです。どの区間も、組を 1 スレッドのときと同じ順に回ります。Γ 点の形式の加算は、軌道の範囲で分けます。 $\mathrm{loc}(A)$ の軌道は、 $K$ の別々の行だからです。どちらの形式でも、 $K$ の要素ごとに、足す値と足す順序は分ける前と同じです。map の形式（`BlockMatrix`）は、収集と加算を呼び出したスレッドで実行します。区間が 1 個のときの $K$ は、分ける前の版（`b8a1363`）とビットまで同じです。収集と加算を分けた版の $K$ は、その前の版（`a8d7910`）とビットまで同じです。区間の数（1、2、3、4、7、64）と呼び出しの順序を変えて確かめました。区間の数が同じなら、 $K$ は呼び出しの順序にも、スレッドの進み方にもよりません。条件は、`dgemm` が同じ引数に対して、どのスレッドでも同じビットを返すことです。作業用の配列は区間ごとに確保するので、配列の置き場所で結果のビットが変わる `dgemm` も、この条件から外れます（8 節）。区間の数が違うと、丸めの範囲で変わります。行列積の回数と乗算の回数は、区間の数によりません。力のための微分（`centre_derivatives`）は、区間に分けていません。
- 関数表にデバイス（7.2 節）を与え、デバイスが呼び出しを引き受けると、呼び出しの範囲の中心の原子を、全てデバイスで計算します（改訂 17）。呼び出しの間、 $D$ 、 $K$ 、全ての原子の T ブロック、値 0 を 1 個、デバイスのメモリーに置きます。中心の原子ごとの配列は、範囲の中で最も大きい原子に合わせて 1 組だけ確保し、原子ごとに使い回します。配列の並びは、転置を作らないように、ホストと変えました。`G` と括弧は $(P,\mu,\text{列})$ の順、`W` は $(P,\nu,l)$ の順で、どちらも積の結果をそのまま使います。4 つの項は、中心の原子の全ての ABF について、積 1 回ずつで計算します。(i) は、`G` の $\mathrm{loc}(B)$ の列を $(\mu,P,l)$ の順に写した行列と、`W` を $(\nu,P,l)$ の順に写した行列の積です。(ii) は、その写しの $B$ の列（ $\nu<u_b$ ）と `W` の積です。(iii) は `W` の写しと $D[A,\mathrm{loc}(B)]$ の積、(iv) は $D[A,B]$ と `W` の写しの積です。括弧には、(iii) を $B$ の列に、(iv) を $\mathrm{loc}(B)$ の列に、この順に足します。 $K$ には、(i)、(ii)、最後の積を、ホストと同じ順に足します。行列積は、中心の原子 1 個につき 2 回と、V 近傍 1 個につき 5 回です。ABF についての和を足す順序がホストと違うので、結果は丸めの範囲で違います。デバイスが断ったとき、関数が 1 個でも欠けているとき、関数が失敗したときは、変わっていない $K$ から、ホストが呼び出しの全体を計算します。 $K$ に書くのは最後の複写の後だけです。
- 係数 $-\alpha$ は、寄与を $K$ に足すときに掛けます。原子ごとの寄与を別々のランクで計算し、後から足せます。
- 構築は、単位胞の ABF の一部だけを計算できます（改訂 21）。中心の原子の寄与が、ABF についての和だからです。単位胞の ABF に、原子の順に通し番号を付けます。`add_exchange_gamma_abfs` と `add_exchange_blocks_abfs` は、番号が続く範囲の ABF の寄与を足します。範囲は、原子の途中で切れてもかまいません。途中で切れた原子では、`Tst`、`G`、`W`、`Rm` を、範囲に入る ABF の分だけ作ります。 $D$ の行の収集は、範囲ごとに原子 1 個分を行います。原子の全体を含む範囲は、原子の範囲を指定する関数と同じ関数を同じ順に呼びます。 $K$ はビットまで同じです。原子の途中で切った範囲の和は、切らない計算と丸めの範囲で違います。ABF についての和を足す順序が変わるためです。モジュールの試験での差は、 $K$ の最大の要素の 3.3e-15 倍以下でした。デバイスの呼び出しも、同じ範囲を受け取ります。プロセスへの配り方は 6.7 節にあります。
- 密度行列が対称なら、構築は V 近傍の対の半分だけを回れます（改訂 25、`80b2e08`）。中心の原子 $a$ が V 近傍 $(b,\mathbf R)$ を通る寄与は、中心 $b$ が $(a,-\mathbf R)$ を通る寄与の転置です。`prepare_system` の構造では、2 つの対の V ブロックが互いの転置になるためです。半分の構築は、2 つの対の片方だけを回って $X$ を作ります。 $K$ は $X+X^{\mathsf T}$ です。site の形式の中心が回るのは、 $\mathbf R>0$ の V 近傍と、 $\mathbf R=0$ で $b>a$ の V 近傍です。 $\mathbf R$ は成分の順に比べます。自分自身は、V ブロックの半分で回ります。 $S_A$ も、これらの近傍の site だけの集合です。そのため、2 つの大きい積も小さくなります。Γ 点の形式では、原子の周期像を 1 つの組にします。中心が回るのは、 $b>a$ の原子の組と、自分の組の半分です。単位胞が打ち切りより小さい構造でも、積が減ります。`mirror_blocks` と `mirror_gamma` が、 $X$ を $X+X^{\mathsf T}$ に変えます。結果は最後のビットまで対称です。半分の構築の関数は `add_exchange_blocks_half_abfs` と `add_exchange_gamma_half_abfs` です。ABF の範囲、区間、デバイスの引数は、全体の構築と同じ形で受け取ります。モジュールの試験（`exx-exchange-half`）では、全体の構築を対称にした $K$ との差が、最大の要素の 1.9e-15 倍以下でした。力の重みも、改訂 26 から半分の形で計算します（6.7 節）。
- Γ 点の形式では、折り畳みを行列積の前に済ませます。中心の原子の寄与は、V ブロックについても T ブロックの列についても線形で、同じ原子の周期像は同じ軌道を指すためです。V ブロックは周期像について足し合わせ、T ブロックの列は同じ軌道に属するものを足し合わせます。行列積の回数は、V 近傍の site の数ではなく、V 近傍に現れる原子の数で決まります。近傍リストに周期像のない原子のブロックは、保存した値のままです。site の形式では、周期像ごとに行き先のブロックが違うので、まとめられません。正則なメッシュの点の $K(\mathbf k)$ だけを使う計算では、V 近傍を周期でまとめられます（改訂 27、この節の終わり）。改訂 28 からは、構築の列も同じ周期でまとめます。

M5 で、site の形式の $D$ と $K$ を 1 本の配列に並べる形を加えました（`exchange.hpp` の `BlockLayout`）。一覧に入れるのは、全ての中心の原子について、行の site が $\mathrm{loc}(A)$ に、列の site が $S_A$ にあるブロックです。キー $(x,y,\mathbf R)$ の昇順に並べ、ブロックごとに配列の中の位置を決めます。一覧は近傍リストだけで決まるので、どのランクでも同じ並びになり、ランクごとの配列を要素ごとに足せます。一覧にないブロックの $K$ は 0 です。 $K_{xy}(\mathbf R)$ のキーがあれば、 $K_{yx}(-\mathbf R)$ のキーもあります。改訂 24 までは、構築の後で、この 2 つのブロックの対応する要素を平均に置き換えていました。平均する前の差の最大は、 $K(\mathbf k)$ がエルミートであることの目安でした。半分の構築の $X+X^{\mathsf T}$ は最後のビットまで対称なので、改訂 25 で平均と差の表示をやめました（`d768793`）。

改訂 27 で、正則な k メッシュの site の形式に、V 近傍をメッシュの周期でまとめる形を加えました（`1626dc4`、`fold_v_neighbours`）。対象は、 $n_1\times n_2\times n_3$ 点の `GCenter` と `REGULAR` のメッシュです。 $n\mathbf T=(n_1T_1,n_2T_2,n_3T_3)$ と書くと、このメッシュの密度行列は $D_{xy}(\mathbf R+n\mathbf T)=s_{\mathbf T}D_{xy}(\mathbf R)$ を満たします。符号 $s_{\mathbf T}=e^{2\pi i\mathbf k\cdot n\mathbf T}$ は、メッシュの全ての点で同じ値です。半点ずれた方向（ $n_i$ が偶数の `REGULAR`）では $(-1)^{T_i}$ 、ほかの方向では 1 です。中心 $a$ が V 近傍 $B=(b,\mathbf R+n\mathbf T)$ を通る 4 つの項は、列の site が $B$ か $\mathrm{loc}(B)$ にある $D$ のブロックを読みます。書き込む $K$ のブロックも、列の site が $B$ か $\mathrm{loc}(B)$ にあります。行の site は $A$ か $\mathrm{loc}(A)$ で、 $n\mathbf T$ によりません。そのため、 $D$ に付く符号と、 $K$ がメッシュの点の $K(\mathbf k)$ に入る位相の符号が打ち消します。 $K(\mathbf k)$ への寄与は、同じ V ブロックで $(b,\mathbf R)$ を通る寄与と等しくなります。寄与は、V ブロックについて線形です。

そこで、セルが周期を法として等しい V 近傍の組を、組の 1 つの site と、V ブロックの和に置き換えます。構築が回る V 近傍は、組ごとに 1 個です。組の site には、中心に最も近い要素を選び、距離が同じ要素の間では大きいセルを選びます。 $a<b$ の組が $(b,\mathbf R)$ を選ぶと、 $b$ の逆向きの組は $(a,-\mathbf R)$ を選び、V ブロックの転置を受け取ります。同じ原子の周期像の組の対も、同じ決め方です。周期を法として $\mathbf R=0$ の組は、原子自身にまとめます。そのブロックは、 $V(0)$ に、 $\mathbf R>0$ の要素ごとの $V(\mathbf R)+V(\mathbf R)^{\mathsf T}$ を足した対称な行列です。 $2\mathbf R$ が周期を法として 0 になる組は、組自身が鏡像です。この組は 2 つの site を残し、 $(a,\mathbf R^\ast)$ に $\mathbf R>0$ の要素の和を、 $(a,-\mathbf R^\ast)$ にその転置を入れます。まとめた系も、半分の構築の条件（ $V_{ab}(\mathbf R)=V_{ba}(-\mathbf R)^{\mathsf T}$ ）を満たします。要素が 1 つの組と、 $(a,\mathbf R)$ と $(a,-\mathbf R)$ だけの組は、ブロックのビットまで変わりません。一覧は、近傍リストと同じく、原子自身を先頭に原子とセルの順に並べます。

メッシュの点の $K(\mathbf k)$ 、メッシュの密度行列のエネルギー、ブロックと位置についての微分は、丸めの範囲で変わりません。変わるのは、 $K(\mathbf R)$ と、メッシュの点の外の $K(\mathbf k)$ です。M8 のバンド分散と DOS の出力は、まとめない系で構築した $K(\mathbf R)$ を使います（6.5 節）。OpenMX の k 点は、メッシュの点から `Shift_K_Point`（1e-12、`openmx_common.h:122`）の 1 倍か 2 倍だけずれています。このずれ $\boldsymbol\delta$ のため、 $s_{\mathbf T}$ に残る位相は $2\pi\boldsymbol\delta\cdot n\mathbf T$ です。 $\sum_i\lvert n_iT_i\rvert\le10$ の要素では、この位相は 1.3e-10 以下です（計算値）。`scf.Generation.Kpoint none`（Monkhorst–Pack）の点は、まとめません（13 節の 28）。

モジュールの試験（`exx-exchange-folding`）は、5 つのメッシュで、まとめた系とまとめない系を比べます。2×2×2 は、ずらさない形と、3 方向に半点ずらした形です。3×2×1 と 1×2×3 は 1 方向に半点ずらし、1×1×1 はずらしません。メッシュの点の $K(\mathbf k)$ の差は、最大の要素の 3.4e-15 倍以下でした。 $E_x$ の差は相対 9.0e-15 以下、5 つの経路の力の差は相対 1.8e-14 以下です。C の入り口の試験は、1×1×1 の周期でまとめた site の形式を、Γ 点の形式と比べます。セルについて足した $K$ 、エネルギー、2 つの経路の力が同じ値になることを確かめます。

改訂 28 で、構築の列（ $S_A$ の site）も、同じ周期でまとめる形を加えました（`51cd508`、`fold_columns`）。V 近傍をまとめた後も、 $S_A$ には、周期の整数倍だけ離れた site が残ります。中心 $a$ の構築が読む $D$ の行は $\mathrm{loc}(A)$ の site で、 $n\mathbf T$ によりません。そのため、 $C=(c,\mathbf R)$ と $C'=(c,\mathbf R+n\mathbf T)$ の列は、 $D[\mathrm{loc}(A),C']=s_{\mathbf T}D[\mathrm{loc}(A),C]$ を満たします。`G` の 2 つの列の違いも、この符号だけです。一方、 $C'$ の列の $K$ のブロックは、メッシュの点の $K(\mathbf k)$ に、 $C$ の列のブロックの $s_{\mathbf T}$ 倍の位相で入ります。 $C'$ の列への寄与を $s_{\mathbf T}$ 倍して $C$ の列に足しても、メッシュの点の $K(\mathbf k)$ は変わりません。

そこで、 $S_A$ の site を、原子が同じでセルが周期を法として等しい類に分けます。類の代表は、`column_sites` の並びで最初の site です。2 つの大きい積（`G` と最後の積）が計算するのは、代表の列だけです。V 近傍ごとの項は、ほかの site の列の `G` と $D$ を、代表の列に符号を掛けて読みます。(i) と (ii) の足し先は、これまでどおり、その site 自身のブロック $K(A,B)$ と $K(A,\mathrm{loc}(B))$ です。(iii) と (iv) の括弧は、符号を掛けて代表の列に足します。最後の積は、代表の site のブロックだけに書き込みます。符号 $s_{\mathbf T}$ を決めるのは、site と代表のセルの差 $n\mathbf T$ です。半点ずれた方向ごとに、 $T_i$ が奇数なら $-1$ を掛けます。大きい積の列の数は、 $s_A$ から類の数に減ります。

メッシュの点の $K(\mathbf k)$ 、エネルギー、微分は、丸めの範囲で変わりません。ブロックの一覧も、V 近傍だけをまとめた系と同じです。変わるのは、保存する $K(\mathbf R)$ の値です。全体の構築の $K(\mathbf R)$ は、 $K_{xy}(\mathbf R)=K_{yx}(-\mathbf R)^{\mathsf T}$ を満たさなくなります。半分の構築の $X+X^{\mathsf T}$ は対称で、結合層はこの形を使います。類の要素がどれも 1 つなら、構築と微分はビットまで変わりません。結合層では、`Input_std.c` が半点ずれた方向を決めます（7.4 節）。`REGULAR` のメッシュで点の数が偶数の方向が、それに当たります。

デバイスの写し（`move`）の係数は、1 回の呼び出しに 1 つです。符号の違う列は、別の写しに分けます。周期が 1 の方向があれば、1 つの V 近傍の $\mathrm{loc}(B)$ に、同じ類の site が 2 つ入り得ます。1 回の写しで同じ要素に 2 回足す形は、書き込みが競合するので使えません。同じ類の列は、現れた順に別の層に入れ、層と符号ごとに 1 回ずつ写します（`column_moves`）。デバイスの模型は、1 回の写しが同じ要素に 2 回書くと、契約の違反として記録します。

モジュールの試験（`exx-exchange-folding`）は、5 つのメッシュの全てで、列もまとめた系を準備の系と比べます。メッシュの点の $K(\mathbf k)$ の差は、最大の要素の 1.9e-15 倍以下でした。 $E_x$ の差は相対 9.0e-15 以下、5 つの経路の力の差は相対 9.5e-15 以下です。ずらしたメッシュでは、符号を付けずにまとめると $K(\mathbf k)$ が変わることも確かめます。半分の構築の時間の見積もりは、準備の系に対して、V 近傍だけのまとめの 0.901 倍から 0.189 倍になりました（2×2×2）。3×2×1 では 0.679 倍から 0.130 倍、1×2×3 では 0.664 倍から 0.126 倍です。周期が $S_A$ のセルの範囲の 2 倍より長い系では、類の要素がどれも 1 つで、ブロックと微分がビットまで同じです。デバイスの模型の試験には、負の符号の列と、2 つの列が同じ類になる場合が入ります（2×2×2 と 1×2×3）。C の入り口の試験は、1×1×1 の周期で列もまとめ、セルについて足した $K$ が Γ 点の形式と同じ値になることを確かめます。

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

M6 の実機なしの段階では、表の経路を増やしませんでした。残りの 3 行（Band のデバイス上、Band のホスト上の全体行列、Cluster のデバイス上）は、デバイスの密行列ソルバーの経路です。実機の段階で、このうちデバイス上の 2 行に加算を書きました（`06105f2`、`1766b6f`）。実機の 2 番目の段階で、この 2 行を通る実行を試験しました（9 節の M6）。

| 表の行 | M6 の実装 |
|---|---|
| Band、デバイス上 | `BandCol_DenseAddExactExchange` を、`Construct_Band_CsHs` の直後に呼びます。場所は 3 か所です。全てのランクが k 点を 1 つずつ受け持つ計算のループ、それ以外の計算の最初のループ、 $H(\mathbf k)$ を作り直す 2 番目のループです。 |
| Cluster、デバイス上 | `ClusterCol_DenseAddExactExchange` を、密行列を持つランクが `d_H` を作った直後に呼びます。場所は 2 か所です。スピンごとに 1 ランクが解く方式と、1 ランクが 2 つのスピンを順に解く方式です。 |
| Band、ホスト上の全体行列 | 書いていません。MAGMA で解く経路です。ハイブリッド汎関数の計算は、この経路に入りません。 |

2 つの関数の手順は同じです。ホストで $n\times n$ の配列を 0 にし、`Exact_Exchange_Add_Band_Hs`（Cluster は `Exact_Exchange_Add_Cluster_Hs`）で式 (14) の和を作ります。この関数には、1 つのプロセスが行列の全体を持つ配置を渡します。1×1 のグリッドで、ブロックの大きさは $n$ です。次に、配列をデバイスの作業領域 `d_tmp` に写し、OpenACC のループで `d_H` に足します。`d_tmp` は、 $H$ の組み立てと対角化の間に中身が読まれない領域です。対角化の最初の変換が、この領域を上書きします。デバイスのメモリーの使用量は増えません。

Band のデバイス上の密行列は、列を先に並べたエルミート行列の全体です。`BandCol_ConstructDenseCsHs_OpenACC` は、上三角の要素と、その複素共役を下三角に書きます。`Exact_Exchange_Add_Band_Hs` が作る配列も同じ並びです。

改訂 25 で、k 点の $K(\mathbf k)$ を保持する形にしました（`f5d746d`）。k 点のメッシュのソルバーは、SCF のステップごとに、全ての k 点で $H(\mathbf k)$ に $K(\mathbf k)$ を足します。改訂 24 までは、そのたびに式 (14) の和を、 $K$ の全てのブロックについて取り直していました。 $K$ が変わるのは、構築のときだけです。今は、構築の後の最初の加算が、式 (14) の和を状態の項目に入れます。次の構築までの加算は、その項目を足します。項目の鍵は、スピン、k 点、局所配列の配置の 3 つです。構築と、新しい構造の準備は、項目を捨てます。項目のバイト数の上限は、全てのスピンの交換行列 2 つ分です。置く余地がないときは、和を一時的な配列に作り、加算の後に捨てます。どちらの場合も、 $K(\mathbf k)$ の丸めは k 点とステップによりません。プロセスが局所の行も列も持たない原子の対のブロックは、和から外します。デバイスの経路には $0+K(\mathbf k)$ を渡すので、ハミルトニアンはビットまで同じです。ScaLAPACK の経路は、ブロックを 1 つずつ足す代わりに、ブロックの和を $H$ に足します。変わるのは丸めだけです。`level.of.stdout 2` の SCF は、和の数と、保持した和を足した回数を表示します（`K(k) of the k points`）。どちらも全てのプロセスの合計です。PBE0 の Si（4×4×4 の k 点、1 プロセス 1 スレッド）では、加算が 19.7 秒から 1.0 秒になりました。全エネルギーと $E_x$ は、表示の 12 桁が同じです。

ハイブリッド汎関数の計算がデバイスの経路に入るのは、1 台のデバイスの容量の判定が通った場合です（`BandCol_GpuDenseFits`、`ClusterCol_GpuDiagFits`）。判定が通らない場合と、MAGMA を使う設定の場合は、CPU の固有値ソルバーで解きます。Band はブロック巡回配置の経路、Cluster はホスト上の経路です。どちらも、M4 と M5 で $K$ の加算を試験しています。既存のコードは、判定が通らない計算を、条件が合えば MAGMA の経路に回します。Cluster ソルバーには、複数のノードに分ける gpusolver2 の経路もあります。ハイブリッド汎関数の計算は、この 2 つに進みません。MAGMA の経路は、行列 1 個に 2 台から 8 台の GPU を求めます（`source/openmx_mgpu_runtime.cu` の `openmx_mgpu_host_precheck`）。GPU が 1 台の計算機では試験できないので、加算を書いていません（14 節）。

**SCF の後の出力。** Band ソルバーのバンド分散、DOS、および周期系の MO 出力は、近傍リストの形式の $H$ から、出力する k 点の $H(\mathbf k)$ を作ります。この $H(\mathbf k)$ に $K(\mathbf k)$ を足します。V 近傍や列をまとめた系の $K(\mathbf R)$ は、メッシュの点の外で正しい $K(\mathbf k)$ を与えません（6.4 節）。そこで、SCF の後に `Exact_Exchange_Prepare_Outputs` が、まとめない系を別の状態に準備します。この系のブロックに最後の構築の $D$ を写し、半分の構築を 1 回行う形です。まとめた一覧にないブロックの $D$ は、セルが周期を法として等しいブロックの値です。半点ずれた方向（`REGULAR`）では、周期 1 つごとに符号を変えます。写した $D$ は、 $D_{xy}(\mathbf R)=D_{yx}(-\mathbf R)^{\mathsf T}$ になるように対称化します。半分の構築が、この条件を前提にするためです。

`Exact_Exchange_Add_Output_Hk` は、出力の k 点で、式 (14) の和を $H(\mathbf k)$ に足します。位相と要素の並びは、`Hamiltonian_Band` と同じです。出力の関数では、k 点ごとに 1 つのプロセスが $n\times n$ の行列の全体を持ちます。足す場所は、`Band_DFT_kpath.c` の 1 か所と、`Band_DFT_Dosout.c` の 3 か所です。後者は、ガウス関数の DOS の k 点、四面体法の DOS の k 点、四面体法がバンドの範囲を決める Γ 点の 3 つです。複数の出力を同時に指定したときは、同じ $K(\mathbf R)$ を再利用します。すべての出力が終わると、`Exact_Exchange_Release_Outputs` が $K(\mathbf R)$ を解放します。

出力の $K(\mathbf R)$ は、最後の構築の $D^{(n)}$ から作ります。`.out` の固有値は、1 つ前の $K^{(n-1)}$ で解いた値です。差は外側反復の 1 回分で、3×3×3 のダイヤモンドでは 2.1e-10 Hartree でした（9 節の M8）。V 近傍をまとめない site の形式の計算では、SCF の $K(\mathbf R)$ をそのまま使います。この枝に入るのは、`scf.Generation.Kpoint none` の計算だけです。この設定は、ハイブリッド汎関数でない計算でも動きません（13 節の 28）。

OpenMX の k 点は、メッシュの点から `Shift_K_Point` だけずれています（6.4 節）。そのため、類の別のセルから写した $D$ には、 $2\pi\boldsymbol\delta\cdot n\mathbf T$ の位相の差が残ります。3×3×3 のダイヤモンドの HSE06 では、対称化の前の非対称性が 1.0e-11 でした。メッシュの点の $K(\mathbf k)$ は、SCF の値と、最大の要素の 2.7e-12 倍まで合います。`Shift_K_Point` を 0 にした一時的な版では、非対称性が 7.9e-16、差が 3.5e-15 倍になりました（9 節の M8）。

**MO 出力。** 孤立分子の Cluster ソルバーでは、厳密交換を含む SCF 最終ステップの固有ベクトルを使います。周期系の Γ 点 SCF では、最後の構築に使った $D_\Gamma$ の原子対の小行列を各並進の $D(\mathbf R)$ とし、既存の積分から $K(\mathbf R)$ を構築します。これにより、Γ 点のみの SCF からも指定された k 点の MO を出力できます。

周期系ではバンド分散の行列梱包と並列分配を共有し、元の AO 基底に戻した固有ベクトルを求めます。固有値と全係数は報告を作るランクに送り、格子で評価する HOMO と LUMO の係数を MPI で配ります。格子投影は各ランクが担当する原子の積をまとめ、格子点を OpenMP で分けて、CPU の AVX2・AVX-512 または NVIDIA GPU で計算します。PAO の格子値は既存の FP32 配列を使い、係数との積和は FP64 で評価します。複数の軌道と実部・虚部の格子を一括で MPI 集約します。

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
| RMM-DIIS、GR-Pulay、RMM-ADIIS | `StartPulay - 1` | 次のステップ数 `StartPulay` で、新しい $K$ の最初の残差から履歴を作り直します。改訂 30 までは `StartPulay - 2` でした（下の「始め直しの位置」）。 |
| RMM-DIISK、RMM-DIISV（`StartPulay` が 7 以上） | `StartPulay - 2` | 次が Kerker 混合で、ステップ数 `StartPulay` の外挿は直前の 2 組を使います。 |
| RMM-DIISK、RMM-DIISV（`StartPulay` が 6 以下） | 1 | SCF の初めと同じ順序です。ステップ数が 6 以下の外挿は、ステップ数 2 以降の組を全て使います。 |
| RMM-DIISH | 1 | SCF の初めと同じ順序です。外挿は、ステップ数 2 以降の組を履歴の長さまで使います。 |
| Simple、Kerker | 変えません | 残差のベクトルを履歴に持ちません。停止条件には、この 2 つの方式だけの値があります（下）。 |

求めた値が負になるときは、ステップ数を変えません。 $K$ を作ったステップがまだ単純混合や Kerker 混合の段階にあり、後の外挿が前の $K$ の組に届かないためです。RMM-DIIS、GR-Pulay、RMM-ADIIS で `StartPulay` が 1 以下のときも、変えません。RMM-DIIS では、入力の検査が 4 以上を求めます。この表は、`Mixing_DM.c`、`DIIS_Mixing_DM.c`、`GR_Pulay_DM.c`、`ADIIS_Mixing_DM.c`、`DIIS_Mixing_Rhok.c`、`Mixing_H.c`、`Mixing_V.c` を読んで作りました。実行して確かめた範囲は、進捗ファイルに書きます。

ステップ数が 1 の混合は、既存のコードが残差ノルムを 1 に置き換えます。そのステップの `NormRD` の表示は 1 になります。

**始め直しの位置。** RMM-DIIS、GR-Pulay、RMM-ADIIS の最初の混合は、改訂 30 まで `StartPulay - 2` でした。新しい $K$ の最初のステップも、単純混合になる位置です。単純混合の重みは、直前の 2 ステップの残差ノルムで決まります（下の「内側の SCF の停止条件」の `OPENMX_MIXING_DENSITY_RATIO`）。 $K$ を作ったステップの残差ノルムの 2 乗は、収束した値（下限の 1e-9 より小さい値）です。そのため、このステップの混合で、重みは `scf.Min.Mixing.Weight` まで下がります。新しい $K$ の最初の残差は大きく、残差が増えたときの規則も重みを下げるので、重みは下限に残ります。新しい $K$ の最初の単純混合は、密度をほとんど動かしませんでした（13 節の 79）。改訂 31 からは、新しい $K$ の最初のステップで履歴を作り直します。外挿が使うのは新しい $K$ で測った組だけで、この点は前と同じです。既定の動作を変えるので、利用者に尋ねました（1.3 節）。利用者は 2026-11-26 の 09:22（JST）に、既定を変える形を選んでいます。RMM-DIISK と RMM-DIISV（`StartPulay` が 7 以上）の位置は、変えていません。`StartPulay - 1` にすると、ステップ数 `StartPulay` の外挿が前の $K$ の組を使うためです。RMM-DIISH と、`StartPulay` が 6 以下の RMM-DIISK と RMM-DIISV は、既に 1 から始めています。

**内側の SCF の停止条件。** 既存の停止条件は、連続する 2 ステップのバンドエネルギーの差と、緩い残差の条件です（`dUele < scf.criterion` と `NormRD^2 < 10 scf.criterion`）。混合が密度をほとんど動かさないと、新しい $K$ に応答していない密度でこの条件を通ります。 $K$ がある内側の SCF には、次の条件を加えました（`Exact_Exchange_Inner_Stop_Allowed`）。

- $K$ を作ってから 2 ステップ目までは止めません。2 ステップ目の停止判定が見る残差ノルムは、新しい $K$ で測った最初の残差です。この値を基準 $R_0$ にします。
- 3 ステップ目以降は、残差ノルムが $0.1R_0$ 以下か、`scf.criterion` 以下のときだけ停止を認めます。

1 つ目の値は、密度が新しい $K$ に応答したことの判定です。2 つ目の値は、 $K$ の変化が密度を乱さなかった場合の出口です。 $R_0$ が初めから小さいと、残差を 10 分の 1 にできません。0.1 と、`scf.criterion` をそのまま使うことは、M4 の実測から決めた値です（13 節の 21）。残差ノルムの単位は混合の方式で違うので、2 つ目の値は目安です。

単純混合と Kerker 混合には、3 つ目の値 $\sqrt{10^{-9}}\approx3.2\times10^{-5}$ があります。残差ノルムがこの値以下なら、停止を認めます。この 2 つの方式は、直前の 2 ステップの残差ノルムから混合の重みを決めます（`core/src/mixing_policy.cpp` の `OPENMX_MIXING_DENSITY_RATIO`）。残差が減ったステップでは、1 つ前の残差ノルムの 2 乗を、2 乗の減った量で割った値を重みに掛けます。割る数には下限 1e-9 があります。残差ノルムの 2 乗が 1e-9 を下回ると、この比は 1 より小さくなります。残差が増えたステップでも重みは減るので、重みはステップごとに減り、`scf.Min.Mixing.Weight` に達します。その後の残差は、ほとんど減りません。ハイブリッド汎関数でない計算の SCF も、この状態で終わります。3 つ目の値がないと、 $R_0$ が既にこの水準にある内側の SCF は、 $0.1R_0$ に届きません（13 節の 25）。下限の 1e-9 は既存のコードの値で、`core/tests/test_mixing_policy.cpp` が確かめています。この値を変えるときは、`Exact_Exchange_Inner_Stop_Allowed` の値も変えます。

**外側反復の外挿。** `scf.exx.outer.extrapolation` の値 $m$ が 1 以上なら、外側反復を続けるときに、次の内側の SCF の $K$ を Anderson 法で作ります（`Exact_Exchange_Extrapolate`）。 $n$ 回目の構築の $K$ を $X_n$ 、その構築の前の内側の SCF が使った $K$ を $Y_{n-1}$ と書きます。構築の残差は $F_n=X_n-Y_{n-1}$ です。係数 $g_j$ は $\lVert F_n-\sum_j g_j(F_j-F_{j-1})\rVert$ を最小にする値で、和は直前の $m$ 個の差についてとります。次の内側の SCF は、 $Y_n=X_n-\sum_j g_j(X_j-X_{j-1})$ を使います。ノルムは、全てのスピンの $K(\mathbf R)$ の要素の 2 乗の和です（site の形式か Γ 点の形式の配列）。最初の構築は、履歴に入れません。その前の SCF は $K$ を持たず、その残差は別の反復のものだからです。そのため、外挿は 3 回目の構築から始まります。外側反復を終える構築では外挿せず、 $E_x$ 、 $\delta$ 、力は構築した $K$ から求めます。次の構築の $\delta$ は、式 (15) のとおり、外挿した $K$ との差です。係数は、プロセス 0 が正規方程式を解いて全てのプロセスに送ります。どのプロセスも、同じビットの $K$ の全体を持つためです（6.7 節の「和の取り方」）。ピボットが対角の最大の 1e-12 倍以下なら、最も古い差を外して解き直します。外挿した後は、保持した $K(\mathbf k)$ の和を捨てて作り直します（6.5 節）。履歴のメモリーは、スピンごとに $K$ の $2(m+1)$ 個分です。履歴は、構造ごとの準備で空になります。

**内側の SCF の緩い判定。** `scf.exx.inner.criterion.ratio` の値 $c$ が 0 より大きいと、 $K$ を作った後の内側の SCF は、 $\tau=\max(\texttt{scf.criterion},\,c\lvert\delta\rvert)$ でも止まります（`Exact_Exchange_Inner_Criterion`）。 $\delta$ は、直前の構築の式 (15) の値です。既存の 2 つの条件（`dUele < scf.criterion` と `NormRD^2 < 10 scf.criterion`）を、 $\tau$ で調べ直します。`Exact_Exchange_Inner_Stop_Allowed` の応答の判定は、 $\tau$ でなく `scf.criterion` で調べます（13 節の 78）。 $\tau$ だけで止まった内側の SCF の後の構築では、 $\lvert\delta\rvert$ が閾値より小さくても外側反復を終えません。その密度は、`scf.criterion` の範囲では $K$ に応答していないためです。外側反復を終えるのは、`scf.criterion` の条件で止まった内側の SCF の後の構築だけです。そのため、 $\lvert\delta\rvert$ が `scf.exx.outer.criterion` より小さい構築の後の内側の SCF は、緩めません。次の構築で外側反復を終えられるようにするためです（13 節の 81）。最初の構築の $\delta$ は $\mathrm{Tr}(DK)$ そのもので大きく、2 回目の構築の前の内側の SCF は、ほとんど応答の判定だけで止まります。最初の PBE の SCF の判定は変わりません。

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

M5 の実装は、この形です。Band ソルバーは、k 点をランクに配る場合と、1 つの k 点の行列を複数のランクで分ける場合があります。前の場合は、各ランクが自分の k 点の寄与を $D$ に足します。後の場合は、各ランクが自分の持つ状態の寄与を足します。どちらも、収集の終わりに和を取ります。構築は中心の原子をランクに配るので、ランクが原子より多いと、構築を受け持たないランクが出ます（13 節の 29）。この配り方は、改訂 21 で改めました（下）。

改訂 21 で、厳密交換の 4 つの計算をプロセスに分けました。利用者の指示による作業です（9 節の「MPI 並列での高速化」）。改訂 20 までの版には、プロセスを増やしても短くならない箇所が 3 つありました。準備は、全てのプロセスが構造の全体について同じ計算をします。構築と力は中心の原子で配るので、原子の数より多いプロセスは働きません。大きさの違う原子が混ざる構造では、受け持ちの時間もそろいません。

`modules/exx` は、今も MPI を使いません。分け方を引数で受け取る入り口を加えました（7.4 節）。結合層（`source/Exact_Exchange.c`）が、プロセスの番号と数を渡し、結果の和を取ります。

| 計算 | 配る単位 | 配り方 | プロセスの間の和 |
|---|---|---|---|
| 準備の T ブロックと V ブロック | site の対 | 対の番号を、プロセスの数で割った余り | 原子ごとの配列。丸めはありません |
| 構築 | 単位胞の ABF | 時間の見積もりの最大が最も小さい、番号の続く範囲 | $K$ の配列。丸めがあります |
| 力の重み（ブロックの要素についての $E_x$ の微分） | 単位胞の ABF | 構築と同じ範囲（改訂 26 から） | 原子ごとの配列。改訂 26 から丸めがあります |
| 力の対（ブロックの微分と重みの積） | site の対 | 対の番号を、プロセスの数で割った余り | 原子ごとの微分。丸めがあります |

**準備。** T ブロックの対は、式 (7) を 1 回解く site の対です。V ブロックの対は、式 (12) を 1 回計算する対です。どちらも、ループの順に 0 から番号を付けます。プロセスは、番号をプロセスの数で割った余りが自分の番号に等しい対だけを計算します（`openmx_exx_prepare_part`）。同じ原子の中のブロックは元素ごとに 1 回で、プロセス 0 の受け持ちです。受け持たない対のブロックは 0 です。結合層は、原子ごとに T ブロックと V ブロックを 1 本の配列に取り出します。全てのプロセスの和を作り、全てのプロセスの状態に戻します（`ExactExchange_Add_Up_Blocks`）。どの要素も、値を持つのは 1 つのプロセスだけです。和に丸めはなく、ブロックの値は 1 プロセスの準備と同じです。1 プロセスの計算は、改訂 20 までと同じ経路を通ります。ABF の生成、動径の変換、近傍リストは、全てのプロセスが同じ計算をします。この部分は、プロセスやスレッドを増やしても短くなりません（14 節）。改訂 26 から、プロセスが受け持つ対は、`-nt` のスレッドで計算します（`openmx_exx_prepare_part_pieces`）。先に対の一覧を作り、 $k$ 番目の対を $k\bmod n$ 番目の区間に入れます。 $n$ は、スレッドの数と対の数の小さい方です。対は自分のブロックにだけ書き込み、区間ごとの局所 RI の報告（解いた系の数と最小のピボット）は最後にまとめます。ブロックと報告は、スレッドの数によらず同じビットです。

**構築。** 単位胞の ABF に、原子の順に通し番号を付けます。プロセスは、番号が続く範囲を 1 つ受け持ちます（6.4 節）。範囲を決める基準は、プロセスの時間の見積もりの最大を最も小さくすることです（改訂 25、`92f78c7`）。原子 $a$ を中心とする部分が ABF を $n$ 個含むとき、その見積もりは $n\,t_a+q_a$ です（`build_costs`）。 $t_a$ は ABF 1 個の時間を表し、 $q_a$ は部分ごとに繰り返す仕事を表します。

$$t_a=2u_as_A\ell_A+500\,u_as_A+\sum_{B\in\mathrm{Vnb}(a)}u_b\ell_B\left(1.4\,n_{P,b}+14.4\,u_a\right)$$

$$q_a=290\,\ell_As_A+\sum_{B\in\mathrm{Vnb}(a)}16\,u_b\ell_Bn_{P,b}$$

単位は、2 つの大きい積の乗算 1 回の時間です。 $t_a$ の第 1 項は 2 つの大きい積で、第 2 項は `G` と括弧の転置です。和の中では、`W` の積と 4 つの項を、乗算の速さの比で重み付けします。 $q_a$ の第 1 項は、 $D$ の行の収集と最後の結果の加算です。和の中は、`W` の積が V 近傍の T ブロックの全体を読む時間です。係数は、クラウドのコンテナで、OpenBLAS を 1 スレッドで使って測りました（12 節）。半分の構築の見積もりは、半分の構築が回る V 近傍について和を取ります（6.4 節）。Γ 点の形式では、 $s_A$ が $N$ 、 $\ell$ が C 近傍に現れる原子の軌道の数になり、和は V 近傍に現れる原子について取ります。

範囲の見積もりは、範囲が ABF を含む原子ごとの $n\,t_a+q_a$ の和です。`divide_abfs` は、番号の続く範囲への全ての分け方のうち、見積もりの最大が最も小さいものを選びます。見積もりの上限を決めると、上限を超えない限り先頭から ABF を取る貪欲な分け方が、範囲の数を最も少なくします。範囲が両端で ABF を増やすと、見積もりが減らないためです。上限は、範囲の数がプロセスの数を超えない最小の値を、二分法で探します。範囲は、全てのプロセスが同じ数から計算するので、どのプロセスでも同じです。範囲は原子の途中で切れるので、原子より多いプロセスは 1 つの原子の ABF を分け合います。範囲の数がプロセスより少ないときは、後ろのプロセスが ABF を受け持ちません。各プロセスは自分の範囲の寄与を足し、 $K$ の和を取ります。1 プロセスの範囲は全ての ABF なので、 $K$ は前の版と同じです。範囲が原子の境界で切れるときの $K$ は、原子の範囲を指定する計算とビットまで同じです。ほかの場合は、丸めの範囲で違います。

改訂 24 までの見積もりは、乗算の回数だけでした。ABF の持ち主は、その前までの和に自分の見積もりの半分を足した値で決めました。値が $[Cr/p,\,C(r+1)/p)$ に入るプロセス $r$ です（ $C$ は和の全体、 $p$ はプロセスの数）。この形では、原子の端の数個の ABF を受け持つプロセスが、部分ごとの仕事を繰り返します。Si の基本単位胞の半分の構築を 2 プロセスで分けた例です。2 番目の原子の 3 個の ABF が、ABF 1 個あたり残りの 2.6 倍の時間を使いました（13 節の 74）。

**力。** $E_x$ は、T ブロックと V ブロックを通して原子の位置に依ります（5.9 節）。微分は 2 段階です。1 段目は、ブロックの要素についての $E_x$ の微分を、密度行列から作ります。この文書での呼び名は「重み」です。重みは、中心の原子の ABF ごとに別の行に入ります。各プロセスは、構築と同じ範囲の ABF の行を計算し、原子ごとの配列の和を取ります。改訂 25 までの形では、要素ごとに、値を持つのは 1 つのプロセスだけでした。2 段目は site の対を回り、ブロックの微分と重みの積を足します（式 (18)）。対を回るのは、両側の原子の重みを使う 1 回だけです。対の番号をプロセスの数で割った余りで配り、原子ごとの微分の和を取ります。改訂 20 までの版は、中心の原子ごとに重みを作り、その原子の近傍の対を回っていました。2 つの原子の対は、中心を入れ替えて 2 回回ります。重みの計算が力の時間の大半を占める系では、原子で配ると、原子の数までしか短くなりません（13 節の 68）。新しい形の力は、改訂 20 までの版と丸めの範囲で違います。重みの配列は、全ての原子の T ブロックと V ブロックと同じ大きさです。力の計算の間、全てのプロセスがこの配列を持ちます（12 節、14 節）。改訂 26 から、プロセスが受け持つ対は、`-nt` のスレッドで計算します（`openmx_exx_add_position_gradient_pairs_pieces`）。対の分け方は、準備と同じです。対ごとのベクトル $g$ を取っておき、全ての区間が終わってから、対の順に微分に足します。微分は、スレッドの数によらず同じビットです。

改訂 26 から、重みは半分の構築の形で計算します（`2198f55`）。全体の形では、中心 $A$ が V 近傍 $B$ の V ブロックに足す重みは、 $\mathrm{factor}\cdot T_Q\circ H_P$ です。ここで $H_P=D[B,A]\thinspace G_P[\cdot,\mathrm{loc}(B)]+G_P[B,\cdot]\thinspace D[A,\mathrm{loc}(B)]$ で、 $\mathrm{factor}$ は $K$ に掛ける $-\alpha$ です。中心 $B$ が転置のブロックに足す重みと合わせると、式 (17) になります。半分の形の中心が回るのは、半分の構築と同じ V 近傍の組だけです（6.4 節）。回った V ブロックには、この重みを 2 倍にして入れます。相手の原子が持つ転置のブロックの分も、このブロックが受け持つためです。自分自身の V ブロックは半分の大きさで構築に入るので、重みは全体の形と変わりません。回らない V ブロックの重みは 0 です。T ブロックの重みのうち、相手の中心が自分の `G` を通して作っていた分は、`W` を通して作ります。回った組ごとに、 $2\cdot\mathrm{factor}\cdot V^{\mathsf T}H$ を、組の原子の T ブロックの重みに足す形です。対の段は重みについて線形なので、力は全体の形と丸めの範囲で同じです。V 近傍 1 個の `W` の大きさの積は、2 回から 3 回に増えます。回る V 近傍は、約半分になります。

重みの ABF の範囲は、半分の構築の見積もりで決めます。構築と同じ範囲です。1 つの範囲が、ほかの原子の T ブロックの重みにも足すので、プロセスの間の和に丸めが入ります。重みの計算も、構築と同じデバイスとスレッドの区間を使います（7.4 節、11 節）。改訂 25 までは、重みの範囲を全体の構築の見積もりで決めていました。重みの計算が、中心の原子の全ての V 近傍を回っていたためです。

V 近傍をまとめた系（改訂 27、6.4 節）でも、力の 2 段目は準備の site の対を回ります（`System::folding`）。重みの並びは、まとめたブロックのものです。 $E_x$ は、組の要素のブロックに、その和を通して依ります。そのため、要素のブロックの重みは、要素を含むまとめたブロックの重みと同じです。対の段は、準備の V ブロックごとに、それを含むまとめたブロックの重みを使います（`held_v_block`）。T ブロックの対は変わりません。構築と重みの見積もり（`build_costs`）は、まとめた V 近傍について和を取ります。列もまとめた系（改訂 28）でも、重みの式は同じです。 $D$ の列は、類の代表の列に符号を掛けて読みます。構築と重みの見積もりは、大きい積の列を類の数で数えます。

**和の取り方。** 配列の和は、プロセス 0 に `MPI_Reduce` で集め、`MPI_Bcast` で配ります（`ExactExchange_Sum`）。全てのプロセスが同じビットの配列を持つので、外側反復の判定がプロセスによって変わりません。1 回の呼び出しに渡す要素は $2^{27}$ 個までです。

**スレッドとの組み合わせ。** プロセスは、自分の範囲の構築を `-nt` のスレッドで計算します（6.4 節）。区間の数は、スレッドの数と、範囲に入るその原子の ABF の数の小さい方です。力の重みも、改訂 25 から同じ区間に分けます。区間の中の計算には ABF についての和がないので、重みはスレッドの数によらず同じビットです。半分の形で `W` を通す積は、範囲の全ての ABF について、呼び出したスレッドで計算します。改訂 26 から、準備の対と力の対も、同じスレッドに分けます。どちらも、スレッドの数によらず同じビットです。

**標準出力。** `level.of.stdout 2` の構築の行は、プロセス 0 の範囲を表示します（`ABFs 1 to 69 of 138 on process 0`）。複数のプロセスの計算では、次の行に、プロセスが自分の範囲に使った時間の最小と最大が出ます（`shares of the 2 processes in the build: ... s to ... s`）。2 つの値が近ければ、配り方がそろっています。力の行の次には、プロセス 0 が重みに使った時間が出ます。`local RI:` の行の数は全てのプロセスの和で、ピボットは全てのプロセスの最小です。

$D$ の収集と、 $E_x$ と $\delta$ の計算は、改訂 20 までと同じです。 $K(\mathbf k)$ の加算は、改訂 25 で和を保持する形にしました（6.5 節）。 $K$ の対称化は、半分の構築の $X+X^{\mathsf T}$ に置き換えました（6.4 節）。

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
  /* 省略できる。body(argument, 0) から body(argument, count - 1) までを 1 回ずつ呼び、全部が戻ってから 0 を返す。
     呼び出しは、別々のスレッドで同時に進めてよい */
  int (*parallel_for)(void *context, int count, void (*body)(void *argument, int index), void *argument);
  int threads;   /* ABF を分ける区間の数。2 未満なら parallel_for を呼ばない */
  /* 省略できる（NULL）。呼び出しの全体を計算するデバイス。引き受けた呼び出しには dgemm と parallel_for を使わない */
  const OpenMXExxDevice *device;
} OpenMXExxDenseBackend;
```

表が与えられない場合に使う、素朴なループの実装をライブラリーの中に持ちます。単体試験が使うのはこの実装です。本体は CPU ビルドで BLAS、GPU ビルドで cuBLAS か GEMMul8 を渡します。

M3 の実装では、素朴なループの実装を `openmx_exx_reference_dgemm` として公開しました。引数は表の `dgemm` と同じで、DGEMM が拒否する引数には 1 を返します。C++ の側では、`DenseBackend` が表を包みます。モジュールの配列は行優先なので、`DenseBackend` は 3 通りの積（ $AB$ 、 $AB^{\mathsf T}$ 、 $A^{\mathsf T}B$ ）を、因子の順序と転置の指定を入れ替えて列優先の `dgemm` に渡します。配列の複製は作りません。次元が `int` の範囲を超えると例外を投げ、`dgemm` が 0 以外を返すと構築を止めます。

表の `dgemm` は、 $\beta=0$ のとき結果の行列を読みません。BLAS の DGEMM と同じ約束です。M6 で書き直した構築は、`G` と最後の行列積の結果を、確保したままの配列で渡します（13 節の 45）。素朴なループの実装は、 $\beta=0$ のとき結果を 0 にしてから足します。

改訂 14 で、表の末尾に `parallel_for` と `threads` を加えました。どちらも省略でき、省略の値は `NULL` と 0 です。ライブラリーはスレッドを作らず、OpenMP も使いません。スレッドは、表を渡す側が持ち込みます。表は、全てのメンバーを設定して渡します（改訂 14 では 4 つ、改訂 17 からは 5 つ）。ライブラリーは全てのメンバーを読むので、設定していないメンバーは `NULL` や 0 になりません。C では、`OpenMXExxDenseBackend table = {0};` と書いてから代入します。`parallel_for` と 2 以上の `threads` を与えると、ABF が 2 個以上ある中心の原子で、`dgemm` は複数のスレッドから同時に呼ばれます。ライブラリーは、ループを、呼び出したスレッドから 1 個ずつ渡します。区間の中からは `parallel_for` を呼びません。渡す `count` は `threads` 以下で、ABF が `threads` 個より少ない原子では ABF の個数です。`dgemm` を省略して `parallel_for` だけを与えると、素朴なループの実装が区間ごとに動きます。C++ の側では、`DenseBackend::pieces` が区間の数の上限を返し、`DenseBackend::parallel_for` がループを包みます。呼び出しの中で例外が出ると、残りの呼び出しを済ませてから、添字が最も小さい呼び出しの例外を投げ直します。ループが 0 以外を返した場合、呼び出しを飛ばした場合、同じ添字を 2 回呼んだ場合、範囲の外の添字を渡した場合は、構築を止めます。区間が 1 個のとき（表がない、ループがない、`threads` が 2 未満）は、ループを呼ばず、呼び出したスレッドで順に計算します。本体が渡すループは 8 節にあります。

`modules/exx` の試験が関数表に渡すループは、`std::thread` で作ったスレッドの上で区間を実行します（`tests/test_parallel.hpp`）。各スレッドが区間の計算を始めるのは、ループの全てのスレッドができた後です。試験は、ループを呼んだ回数と各ループの `count` を記録し、試験に書いた値と比べます。区間が 2 個以上の中心の原子では、ループの回数は、map の形式で 2 + 2 × グループの数です。1 本の配列の site の形式と Γ 点の形式では、 $D$ の行の収集と $K$ への加算の 2 つが加わり、4 + 2 × グループの数です（改訂 15。6.4 節）。グループは、site の形式では V 近傍のサイト、Γ 点の形式では V 近傍にある原子です。区間に分けた結果は、区間が 1 個の結果と 1e-12 以内で合います（成分の差か、最大の成分に対する比）。区間を番号順、逆順、別々のスレッドで実行した結果は、ビット単位で同じです。約束を破るループと、1 個の区間の中だけで失敗する行列積は、構築を例外で終わらせます。導関数の計算は、ループを呼びません。スレッドを作る 3 つの試験には、ラベル `threads` があります。`-DOPENMX_EXX_THREAD_SANITIZER=ON` で構成すると、ライブラリーと試験に ThreadSanitizer が入ります。CI の `exx` のジョブが、この 3 つの試験を Clang と ThreadSanitizer で実行するようにしました（`e9b637c`）。試験が競合を見つけるかは、区間が作業用の配列を共有するライブラリーの変種で確かめています。4 コアでは、`exx-exchange-backend` と `exx-exchange-system` が 10 回とも失敗しました。1 コアでは `exx-exchange-system` だけが 10 回とも失敗し、`exx-exchange-backend` は 10 回とも合格です。ThreadSanitizer 付きのビルドは、1 コアの `exx-exchange-backend` でも、5 回とも競合を報告しました。区間の和を足す順序そのものは、試験で確かめていません。和を常に最後の区間から足す変種は、どの試験にも合格します。順序は実行によらず決まっていて、区間ごとの和を別に計算する試験がないからです。

改訂 15 で、 $D$ の行の収集と $K$ への加算を区間に分けた形（6.4 節）の試験を加えました。 $K$ の同じブロックに足す site の組があると、加算の分け方を誤ったときに結果が変わります。試験はこの組を数え、フィクスチャーの模型、狭い単位胞、周期構造の 3 つの系で 1 個以上あることを確かめます。1 本の配列の形式の $K$ は、区間の数と呼び出しの順序を変えて、map の形式の $K$ とビット単位で比べます。一覧にない $D$ のブロックは、区間に分けた収集でも例外になります。ライブラリーの変種 4 つで、試験が誤りを見つけるかを確かめました。加算を $\mathrm{loc}(A)$ の site で分ける変種、site の形式の収集で区間の最後の site を飛ばす変種、Γ 点の形式の収集と加算で区間の最後の行を飛ばす 2 つの変種です。4 つとも、2 個から 6 個の試験が失敗しました。site で分ける変種では、 $K$ の同じブロックに 2 つの区間が足します。呼び出しを番号の順に進めるループでは結果が変わらず、逆の順に進めるループで見つかりました。

改訂 17 で、表の 5 番目のメンバー `device` を加えました（`ae5c54e`）。省略の値は `NULL` です。表は、5 つのメンバーを全て設定して渡します。`device` は、デバイスの関数表 `OpenMXExxDevice`（`exx.h`）を指します。関数は 8 つです。

| 関数 | 役割 |
|---|---|
| `accept` | 呼び出しを引き受けるかを決める。引数は、呼び出しの全ての積の乗算の回数の和と、確保するメモリーの合計のバイト数。改訂 28 までは、和でなく最大の積の回数だった |
| `allocate`、`release` | デバイスのメモリーを確保する、返す |
| `upload`、`download` | ホストとデバイスの間で写す。写し終えてから戻る |
| `dgemm` | 表の `dgemm` と同じ引数の積。行列はデバイスのメモリーにある |
| `move` | 添字の表に従って、要素を写すか足す（下） |
| `finish` | 渡した仕事が全て終わるのを待つ |

`dgemm` と `move` は、仕事を終える前に戻ってかまいません。デバイスは、渡された順に仕事を進めます。`move` は、行 $r$ と列 $c$ の組ごとに、`source` の要素の `alpha` 倍を `target` の要素に写すか、足します。積を先に作り、それから足します。要素の位置は、行の位置と列の位置の和です。行の位置は `stride` の $r$ 倍か表 `rows` の値で、列の位置も同じ形です。行ごとの列の数は、表 `lengths` で変えられます。表は全てデバイスのメモリーに置きます。1 回の `move` の中で、2 つの組が同じ要素に書くことはありません。

ライブラリーが関数を呼ぶ順序は、次のとおりです。

- `accept` は、中心の原子がある呼び出しで 1 回だけ、最初に呼ぶ。
- 複写と `release` は、渡した仕事が残っていないときだけ呼ぶ。最初の `dgemm` か `move` の前か、`finish` の後である。
- 中心の原子ごとに、最後に `finish` を呼ぶ。
- 関数が失敗するか、`allocate` が `NULL` を返したら、その呼び出しではデバイスを使わない。仕事が残っていれば `finish` を呼び、確保したメモリーを全て返す。表の `dgemm` と `parallel_for` で、呼び出しの全体を計算し直す。

`k` には、最後の複写の後でだけ書きます。失敗した呼び出しの途中の値は、`k` に残りません。C++ の 2 つの関数（`add_exchange_gamma_range` と `add_exchange_blocks_range`）は、デバイスが計算したときに `true` を返します。本体が渡すデバイスは 8 節と 11 節にあります。

試験は、ホストのメモリーでデバイスを表した関数表を使います（`modules/exx/tests/test_device.hpp`）。この表は、約束を破る呼び出しを記録します。確保したメモリーの外を指す添字、1 回の `move` の中で重なる書き込み先、仕事が残っているときの複写と解放、失敗の後の呼び出し、返されないメモリーです。確保したメモリーは 0xFF で埋めます。書く前に読んだ値は NaN になり、結果の比較で見つかります。

- `exx-exchange-device`: フィクスチャーの模型と 3 原子の系で、2 つの形式を確かめる。デバイスの結果は、ホストの結果と相対 1e-12 以内で合う（実測は 6e-16 以下）。同じ呼び出しを 2 回しても、中心の原子を 2 回の呼び出しに分けても、結果はビットまで同じである。デバイスが断る場合、関数が 1 個欠けている場合、デバイスの呼び出しを 1 個ずつ順に失敗させる場合は、 $K$ がホストの結果とビットまで同じになる。
- `exx-exchange-device-system`: 同じことを、`exx-exchange-system` の周期構造で確かめる。失敗させる呼び出しは、種類ごとの最初と最後と、等間隔に選んだ約 40 個である。
- `exx-c-api`: C の入口に、ホストのメモリーで表したデバイスを渡す。結果は、ホストの結果と、最大の成分の 1e-12 倍まで合う。断る場合と失敗させる場合は、ビットまで同じになる。

ライブラリーの変種 3 つで、試験が誤りを見つけるかを確かめました。site の形式の最後の加算を 1 回の `move` にまとめる変種、`finish` を呼ばずにメモリーを返す変種、括弧を 0 にしない変種です。3 つとも、`exx-exchange-device` が失敗しました。

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

結合層では、`Exact_Exchange_Prepare` に行列の形を選ぶ引数を加えました。`DFT` 関数は、Band ソルバーの計算で site の形式を、Cluster ソルバーの計算で Γ 点の形式を選びます。k 点の収集は `Exact_Exchange_Collect_K`、 $H(\mathbf k)$ への加算は `Exact_Exchange_Add_Band_Hs` です。`Exact_Exchange_Density_Deviation` には、近傍の原子がどのセルにあるかを表す 2 つの配列（本体の `ncn` と `atv_ijk`）を引数で渡します。`Exact_Exchange_Asymmetry` は、最後の構築で平均する前に測った $K$ の差の最大を返します（改訂 25 で削除しました。下の段落）。Γ 点の形式の関数は、site の形式の状態で呼ばれると計算を止めます。逆の場合も同じです。

M6 で、行列積をデバイスに送る部品を結合層の下に置きました。`source/Exact_Exchange_Device.c` の `Exact_Exchange_Device_Dgemm` は、DGEMM の引数に状態 `OpenMXExactExchangeDevice` を加えた関数です（11 節）。この状態は結合層の状態の中にあり、cuBLAS のハンドル、2 つの限度、GEMMul8 の指定、積を数える変数を持ちます。ハンドルは、計算の終わりに `Exact_Exchange_Device_Release` が返します。関数表の `context` は、構築 1 回分の集計を指します。集計の中身は、積の数、最大の積の大きさ、デバイスで使った時間です。`Exact_Exchange_Device_Use` は、デバイスで計算した積の数などを、全てのプロセスについて合計します。中で `MPI_Allreduce` を呼ぶので、全てのプロセスが同じ場所で呼ぶ必要があります。部品の関数は大域変数を読みません。デバイスが使えるかどうかは、既存の `gpu_rank_device_usable` に問い合わせます。

M6 の実機の段階で、結合層に 3 つの関数を加えました。密行列をデバイスに持つソルバーのためのものです。`Exact_Exchange_Device_Addition` は、デバイスでの加算 1 回と、その転送と加算の秒数を記録します。秒数は、加算の時間に入ります。`Exact_Exchange_Device_Additions` は、加算の回数を全てのプロセスについて合計します。中で `MPI_Allreduce` を呼びます。`Exact_Exchange_Collecting` は、収集が開いている間だけ 1 を返します。ソルバーは、固有ベクトルをデバイスから取り出す前に、この関数で収集の有無を確かめます。3 つの関数は大域変数を読みません。

同じ段階で、GEMMul8 の指定を結合層に通しました（11 節）。`Input_std` は、入力の `scf.gemmul8.enable` が `on` のときだけ、`OpenMXExactExchangeInput` の `gemmul8` を 1 にします。`Exact_Exchange_Configure` が、この値を部品の状態に写します。`Exact_Exchange_Device_Use` が返す数は 5 つになりました。5 つ目は、GEMMul8 が計算した積の数です。`Exact_Exchange_Device_Gemmul8` は、入力の指定を返します。`DFT` 関数は、この関数で標準出力の 1 行を出すかどうかを決めます。

構築の高速化（改訂 14）で、ホストの積とループの部品を結合層の下に置きました。`source/Exact_Exchange_Host.c`（ヘッダーは `source/openmx_exact_exchange_host.h`）で、MPI を使いません。関数は 7 つです。`Exact_Exchange_Host_Pieces` は、積を分ける区間の数を返します。`Exact_Exchange_Host_Dgemm` は、積を区間に分けて計算します。`Exact_Exchange_Host_Parallel_For` は、関数表に渡すループです。`Exact_Exchange_Host_Loop_Dgemm` は、ループの中の積を数えて計算します。`Exact_Exchange_Host_Note_Product` は、ループの外の積を数えます。`Exact_Exchange_Host_Blas_Single` と `Exact_Exchange_Host_Blas_Restore` は、BLAS のスレッドの数を切り替えて戻します。集計の構造体は `OpenMXExactExchangeHostProducts` です。構築 1 回分の積の数、最大の積、ループのスレッド数、スレッドごとの集計を持ちます。結合層は、この構造体を関数表の `context` の中に置きます。積をデバイスに送るかどうかの判定と、デバイスの時間の測定は、結合層の `ExactExchange_Dgemm` に残しました。最初の形（`1219cb9`）は、ループと集計を結合層の中に持っていました。結合層は MPI と状態の全体を要するので、模型の試験が届きません。点検の後で、今の場所に移しました（`16951be`）。

構築の高速化の 3 番目の段階（改訂 17、`2e2a6da`）で、部品に 2 つの関数を加えました。`Exact_Exchange_Device_Table` は、`modules/exx` に渡すデバイスの関数表（7.2 節）を作ります。表の `context` は部品の状態で、表は状態と表が残っている間だけ使えます。CPU 版と、OpenACC なしでコンパイルした CUDA 版では、`NULL` を返します。HIP 版には、部品が入っていません。`Exact_Exchange_Device_Call_End` は、`openmx_exx_add_blocks` か `openmx_exx_add_gamma` を 1 回呼ぶたびに、その後で結合層が呼びます。cuBLAS のハンドルのストリームを元に戻し、呼び出しを数えます。デバイスの分に数えるのは、結果が戻った呼び出しだけです。容量の不足でホストに残った呼び出しは、別に数えます。結合層の `Exact_Exchange_Device_Calls` は、呼び出しの数、積の数、時間を、全てのプロセスについて集めます。中で `MPI_Allreduce` を 2 回呼ぶので、全てのプロセスが同じ場所で呼ぶ必要があります。`DFT` 関数は、この結果で標準出力の行を出します（11 節）。3 つの関数は、大域変数を読みません。

M7 で、結合層に `Exact_Exchange_Gradient` を加えました。最後の構築の密度行列を固定して、 $E_x$ の原子の位置についての微分を返します。配列は原子 1 個に 3 要素で、単位は Hartree/bohr です。中心の原子の範囲は、構築と同じ関数（`ExactExchange_Centres`）で決めます。この関数は、改訂 21 で `ExactExchange_Abfs` に置き換えました（下）。モジュールに渡す係数は、 $-\alpha$ と縮重度（スピンの成分が 1 つなら 2、2 つなら 1）の積です。Γ 点の形式は `openmx_exx_add_position_gradient_gamma`、site の形式は `openmx_exx_add_position_gradient_blocks` を呼びます。プロセスごとの寄与の和を、全てのプロセスに配ります（`ExactExchange_Sum`）。行列積は、ホストの BLAS だけで計算します。デバイスの関数表と、積をデバイスに送る部品を渡しません。デバイスの積と呼び出しの数を、構築の分だけに保つためです。交換行列がないときは `NULL` を返します。ハイブリッド汎関数でない計算と、構造を準備した後でまだ構築していないときです。

`DFT` 関数は、この戻り値を `Force` の新しい最後の引数に渡します。`Force_Impl` は、自分のプロセスの原子（`Mc_AN` が 1 から `Matomnum`）の分を、`Gxyz[Gc_AN][17]` から `[19]` に足します。`level.of.stdout` が 1 以上のとき、結合層は標準出力に 1 行を出します。厳密交換による力の最大の成分と、原子についての和です（`<EXX>  forces of the exact exchange in ... s: largest component ..., sum ...`）。和は、9 節の M7 の条件の 3 番目に当たります。HIP 版の `Force` は引数を受け取るだけで、HIP 版の `DFT` は `NULL` を渡します。この関数も大域変数を読みません。

改訂 21 で、計算をプロセスに分けるための入り口を `modules/exx` に加えました（6.7 節）。C の入り口は 11 個です。どれも MPI を使わず、分け方を引数で受け取ります。

| 入り口 | 内容 |
|---|---|
| `openmx_exx_prepare_part` | 準備のうち、対の番号の余りが `part` に等しい分を計算します。`parts` が 1 のときは `openmx_exx_prepare` と同じです。 |
| `openmx_exx_atom_table_elements`、`openmx_exx_get_atom_tables`、`openmx_exx_set_atom_tables` | 原子 1 個の T ブロックと V ブロックを 1 本の配列として扱います。要素の数、写し、置き換えです。 |
| `openmx_exx_abf_count`、`openmx_exx_divide_abfs` | 単位胞の ABF の数と、プロセスへの ABF の分け方です。`openmx_exx_divide_abfs` は、改訂 25 で `openmx_exx_abf_costs` を置き換えました（下）。 |
| `openmx_exx_add_gamma_abfs`、`openmx_exx_add_blocks_abfs` | ABF の範囲の寄与を $K$ に足します。 |
| `openmx_exx_add_block_weights_gamma_abfs`、`openmx_exx_add_block_weights_blocks_abfs` | ABF の範囲の重みを、原子ごとの配列に足します。配列の並びは `openmx_exx_get_atom_tables` と同じです。範囲が触れない原子の配列は読みません。 |
| `openmx_exx_add_position_gradient_pairs` | 全ての原子の重みを受け取り、対の番号の余りが `part` に等しい分の微分を足します。 |

複数の部分に分けた準備の後は、全ての原子の配列を置き換えるまで、構築と微分の入り口が失敗を返します。原子の範囲を指定する今までの入り口（`openmx_exx_add_gamma`、`openmx_exx_add_blocks`、`openmx_exx_add_position_gradient_gamma`、`openmx_exx_add_position_gradient_blocks`）は、残してあります。C++ の関数は、`prepare_system` の最後の引数（`PrepareShare`）、`table_elements`、`get_tables`、`set_tables`、`build_costs`、`divide_abfs`、`add_exchange_gamma_abfs`、`add_exchange_blocks_abfs`、`add_block_weights_gamma_abfs`、`add_block_weights_blocks_abfs`、`PositionGradient::add_pairs` です。

結合層では、内部の関数を 2 つ変えました。`ExactExchange_Abfs` は、プロセスが受け持つ ABF の範囲を返します。構築と力の重みが使います。前の `ExactExchange_Centres` は削除しました。`ExactExchange_Add_Up_Blocks` は、分けた準備のブロックの和を、全てのプロセスの状態に入れます。外から呼ぶ関数（`Exact_Exchange_Prepare`、`Exact_Exchange_Build`、`Exact_Exchange_Gradient`）の引数は変えていません。`source/` のほかのファイルも変えていません。2 つの関数は、大域変数を読みません。

改訂 25 で、C の入り口を 5 つ加え、`openmx_exx_abf_costs` を削除しました（9 節の「力のデバイス化とほかの高速化」）。

| 入り口 | 内容 |
|---|---|
| `openmx_exx_add_gamma_half_abfs`、`openmx_exx_add_blocks_half_abfs` | 半分の構築で、ABF の範囲の寄与を $X$ に足します（6.4 節）。引数は、全体の構築の入り口と同じ形です。 |
| `openmx_exx_mirror_gamma`、`openmx_exx_mirror_blocks` | $X$ を $X+X^{\mathsf T}$ に置き換えます。 |
| `openmx_exx_divide_abfs` | 全体か半分の構築の ABF を、指定した数のプロセスに分けます（6.7 節）。範囲の境界を、プロセスの数より 1 個多く返します。 |

C++ の関数は、`add_exchange_gamma_half_abfs`、`add_exchange_blocks_half_abfs`、`mirror_gamma`、`mirror_blocks`、`build_costs`、`divide_abfs` です。重みの 2 つの入り口（`openmx_exx_add_block_weights_gamma_abfs`、`openmx_exx_add_block_weights_blocks_abfs`）は、引数を変えずに、関数表のデバイスを使う形になりました（`d6f7db1`）。構築と同じく、呼び出し 1 回を引き受け、 $D$ と全ての原子の T ブロックをデバイスに置きます。作業用の配列は、中心の原子 1 個（またはその一部）ごとに使い回す形です。部分の重みは、2 回の転送で戻します。デバイスが失敗したときは、戻っていない部分をホストが計算します。戻った部分は、デバイスの重みのままです。

結合層では、`Exact_Exchange_Gradient` が構築のデバイスを重みに渡します（`df30625`）。M7 の段落に書いた、ホストの BLAS だけで計算する形は、改訂 25 で改めました。重みのデバイスの数は、構築の前に取っておき、重みの後で戻します。構築の行と M6 の道具の判定が、前と同じに数えるためです。`Exact_Exchange_Band_Sums` は、 $K(\mathbf k)$ の和の数と、保持した和を足した回数を返します（6.5 節）。`Exact_Exchange_Asymmetry` は、削除しました。 $K$ の平均と、平均する前の差がなくなったためです（6.4 節）。モジュールの `openmx_exx_symmetrize_blocks` は残してありますが、結合層は呼びません。

改訂 26 で、C の入り口を 4 つ加えました（9 節の「力のデバイス化とほかの高速化」）。

| 入り口 | 内容 |
|---|---|
| `openmx_exx_add_block_weights_gamma_half_abfs`、`openmx_exx_add_block_weights_blocks_half_abfs` | ABF の範囲の重みを、半分の構築の形で足します（6.7 節）。範囲が空でなければ、全ての原子の配列が要ります。 |
| `openmx_exx_prepare_part_pieces` | `openmx_exx_prepare_part` の対を、関数表のループの区間に分けます（6.7 節）。関数表の積とデバイスは使いません。 |
| `openmx_exx_add_position_gradient_pairs_pieces` | `openmx_exx_add_position_gradient_pairs` の対を、同じ形で区間に分けます。 |

C++ の関数は、`half_gradient_blocks`、`half_gradient_gamma`、`add_block_weights_gamma_half_abfs`、`add_block_weights_blocks_half_abfs` です。`prepare_system` と `PositionGradient::add_pairs` は、最後の引数で関数表を受け取ります。前からの入り口は、関数表なしでこの 2 つを呼ぶ形です。結合層の `Exact_Exchange_Gradient` は、半分の形の入り口を呼び、重みの範囲を半分の構築の見積もりで決めます（`2198f55`）。対の段には、重みと同じ関数表を渡します。`Exact_Exchange_Prepare` が準備に渡すのは、`-nt` のスレッドのループを持つ関数表です。

改訂 27 で、C の入り口を 2 つ加えました（6.4 節）。改訂 28 の `openmx_exx_fold_columns` も、同じ表に書きます。

| 入り口 | 内容 |
|---|---|
| `openmx_exx_fold_v_neighbours` | 準備した構造の V 近傍を、k メッシュの周期でまとめます。前提は、分けた準備のブロックを足し終えていることです（`openmx_exx_set_atom_tables`）。site の形式のブロックの一覧は捨てるので、`openmx_exx_prepare_blocks` で作り直します。まとめた構造と 1 未満の周期には 1 を返し、状態は変えません。新しい準備は、まとめを取り消します。 |
| `openmx_exx_prepared_v_neighbours` | 準備の V 近傍の数を返します。まとめた後の `openmx_exx_v_neighbours` が返すのは、まとめた数です。 |
| `openmx_exx_fold_columns` | V 近傍をまとめた構造で、構築の列も同じ周期でまとめます（改訂 28）。3 つの整数は、半点ずれた方向（周期ごとに $D$ の符号が変わる方向）で 0 以外の値です。方向を間違えると、 $K$ が誤ります。ブロックの一覧は変わりません。V 近傍をまとめていない構造、列をまとめた構造、NULL には 1 を返し、状態は変えません。新しい準備は、このまとめも取り消します。 |

C++ の関数は `fold_v_neighbours` で、まとめの記録は `System::folding`（`VFolding`）に入ります。力の 2 段目は、この記録で準備の対を回ります（6.7 節）。結合層で `OpenMXExactExchangeInput::fold_period` を決めるのは、`Input_std.c` です。値は、Band ソルバーで `scf.Generation.Kpoint` が `GCenter` か `REGULAR` のとき `scf.Kgrid` の 3 つの数で、ほかの場合は 0 です。`Exact_Exchange_Prepare` は、site の形式で 3 つの数が全て正のとき、ブロックの和の後、ブロックの一覧の前にまとめます。全てのプロセスが、足したブロックを同じ形でまとめます。準備の行の V sites は、準備の数のままです。まとめた計算では、次の行にまとめた数が出ます（`V sites folded by the period of the k mesh, 4 x 4 x 4 cells: 270 of 1698`）。改訂 28 からは、`Input_std.c` が `fold_antiperiodic` も決めます。`REGULAR` のメッシュで点の数が偶数の方向は 1、ほかの方向とメッシュは 0 です。`Exact_Exchange_Prepare` は、V 近傍をまとめた直後に、列もまとめます。まとめた計算では、出力に 3 つの方向の符号の行が加わります（`columns of D folded by the same period; the sign of D over one period along a1, a2, a3: - - -`）。`Exact_Exchange_Collect_K` が確かめるのは、渡された k 点が、まとめたメッシュの点かどうかです。 $k_in_i$ （半点ずれた方向では $k_in_i-1/2$ ）と整数の差が 1e-8 を超えると、計算を止めます。

改訂 30 で、結合層に 3 つの関数を加えました（6.5 節）。`modules/exx` の C の入り口は変えていません。

| 関数 | 内容 |
|---|---|
| `Exact_Exchange_Prepare_Outputs` | SCF の後に、出力の $K(\mathbf R)$ を作ります。V 近傍をまとめた計算では、まとめない系を別の状態に準備し、最後の構築の $D$ から半分の構築を 1 回行う形です。`level.of.stdout` が 2 以上のときは、メッシュの点の $K(\mathbf k)$ を SCF の値と比べて表示します。 |
| `Exact_Exchange_Add_Output_Hk` | 出力の k 点で、式 (14) の和を $n\times n$ の複素行列に足します。添字は 1 から始め、並びは `Hamiltonian_Band` と同じです。ハイブリッド汎関数でない状態と `NULL` では、何もしません。準備のない状態では、計算を止めます。 |
| `Exact_Exchange_Release_Outputs` | 出力の $K(\mathbf R)$ を捨てます。構造の準備（`Exact_Exchange_Prepare`）も、最初にこの関数を呼びます。 |

`Band_DFT_kpath` と `Band_DFT_Dosout` には、最後の引数に結合層の状態を加えました（`openmx_common.h`）。縮約した基底の経路（`Cnt_switch`）と HIP 版は、`NULL` を渡します。結合層の中では、`Exact_Exchange_Prepare` の前半を `ExactExchange_Prepare_Module` に分け、出力の準備と共有しています。`ExactExchange_Read_Blocks` は、ブロックの一覧を引数の配列に返す形にしました。`ExactExchange_Abfs` は、渡したモジュールの ABF をプロセスに分けます。

### 7.5 `modules/xc` への追加

ωPBE の短距離交換を、純粋な関数として `modules/xc` に加えます。既存の `openmx_xc_kernel_pbe`（`kernels.h:285`）の演算は変えません。加える関数は 2 つです。1 つは短距離部分だけを返す関数で、試験はこの出力をフィクスチャー（10.4 節）と比べます。もう 1 つはハイブリッド用の関数です。この関数は PBE の関数を呼び、交換の出力（`exc[0]`、`dexdd`、`dexdgd`）から短距離部分の $\alpha$ 倍を引きます。必要な特殊関数は $e^{x}E_1(x)$ と $\mathrm{erfcx}(x)=e^{x^2}\mathrm{erfc}(x)$ の 2 つで、どちらもこのモジュールの中に実装します。

PBE0（M8）では短距離部分の関数を使いません。式 (2) のとおり、PBE の交換の出力を $1-\alpha$ 倍します。 $\omega=0$ にした短距離部分の関数は PBE の交換と一致しないので、代わりには使えません（M1）。

改訂 25 で、ハイブリッド汎関数の準局所部分の演算を、C のヘッダー `modules/xc/include/openmx/xc/hybrid_kernels.h` に移しました（`1bc356f`）。ヘッダーは、短距離交換、 $e^{x}E_1(x)$ 、 $\mathrm{erfcx}(x)$ 、HSE06 と PBE0 の準局所部分を、`kernels.h` と同じ形のインライン関数で持ちます。GPU のデバイスのコードから、ライブラリーと同じ演算を呼ぶためです（8 節）。微分の前進モードは、置き換えた C++ のクラスの演算ごとに関数を 1 つ持つ C の構造体です。式の演算の順序は、`wpbeh_kernel.cpp` と同じに保ちました。ライブラリーの 6 つの関数は、このヘッダーを呼びます。`wpbeh_short_range_exchange`、`pbe_screened_hybrid`、`pbe_global_hybrid`、`wpbeh_channel_energy`、`e1_scaled`、`erfcx` です。前の版と新しい版のライブラリーは、155,838,653 個の出力がビットまで同じでした。試験 `xc-c-api` は、インラインの関数と C の入り口を、ビットまで比べます。

同じ改訂で、2 つのスピンの密度と勾配の大きさが同じとき、短距離交換の核を 1 回だけ呼ぶ形にしました（`85c1e7b`）。スピン分極のない OpenMX の計算は、2 つのスピンに同じ値を渡します。結果は、2 回呼んだときとビットまで同じです。勾配についての微分は、それぞれのスピンの成分を保ちます。試験 `xc-hybrid` は、2,436 点の短距離交換を、スピンごとに核を 1 回ずつ呼んだ値とビットまで比べます。同じスピンの点の時間は、1 点 1,401 ns から 923 ns になりました（ライブラリーの関数、400,000 点）。

## 8. 既存コードへの組み込み

| 作業 | 場所 | 内容 |
|---|---|---|
| 入力 | `source/Input_std.c:990-992` | `scf.XcType` に `HSE06` と `PBE0` を加えます。内部は `XC_switch=4` とし、種別と混合比と $\omega$ を EXX の状態に入れます。`HSE`（5）と `NLX`（6）の指定はエラーです（M0 で実装済み、`994-1001` 行）。 |
| 旧い厳密交換のコード | `source/Allocate_Arrays.c:770-778`、`source/Free_Arrays.c:1836-1842` | 試作用の入力（4 つのキーワードと `Allocate_Arrays(15)` の呼び出し）は、M0 で `Input_std.c` から削除しました（`bcf5222`）。変数の定義、`case 15` の確保、解放、`Expand_ReCoulomb_Kernel.c`、`source/exx*` は残っています。動作には影響しません。削除するときは利用者に確認します（1.3 節）。 |
| キーワード | `source/Input_std.c` | 下の表のキーワードを読みます。2.2 節の拒否もここに置きます。 |
| 準局所の交換 | `source/Set_XC_Grid.c:1274`、`modules/xc` | CPU の経路で、ハイブリッド用の関数を呼びます。混合比と $\omega$ は `Set_XC_Grid` の引数に加えます。呼び出しは 7 か所です（`Set_Vpot.c:97`、`Total_Energy.c:1836`、`1993`、`4444`、`Stress.c:2413`、`OutData.c:580`、`OutData_Binary.c:511`）。 |
| XC の GPU 経路 | `source/Set_XC_Grid.c:654` | 改訂 24 までは、ハイブリッドが有効な間は GPU の経路に入らず、ホストのループを使いました。改訂 25 から、CUDA 版はハイブリッド汎関数の準局所部分も GPU の経路で計算します（下の改訂 25 の表）。HIP 版は、ホストのループのままです。 |
| 密行列への加算 | 6.5 節の表 | $K^{\sigma}(\mathbf k)$ を足します。 |
| 密度行列の収集 | `source/Band_DFT_Col.c:1949`、`2038`、`source/Cluster_DFT_Col.c:4121` | 密度行列の計算に渡している固有ベクトルと占有数を、EXX の状態にも渡します。 |
| 外側反復 | `source/DFT.c:1424-1505` | 6.6 節 |
| 全エネルギー | `core/include/openmx_core_energy.h`、`source/DFT.c:2822-2842`、`2869` 付近 | エネルギーの成分に厳密交換を加え、合計と出力に入れます。 |
| 力 | `source/Force.c:3055`、`source/DFT.c` | M7 で組み込みました。`DFT` が結合層の `Exact_Exchange_Gradient` を呼び、戻り値を `Force` の最後の引数に渡します。`Force_Impl` が、自分のプロセスの原子の分を `Gxyz` の 17 番から 19 番に足します（7.4 節）。 |
| ビルド | `source/Makefile:309`、`source/openmx_core.mk`、`core/CMakeLists.txt` | 本体に新しい翻訳単位を足すときは `OBJS` と規則を追加します。CPU ビルドは `source/Makefile` から一覧を作ります（`cmake/OpenMXCPU.cmake:172-178`）。HIP 版が使うファイルの一覧は `source/backends/hip/sources.json` です。 |

M4 で組み込んだ範囲です。場所は付録 A の「M4 で加えたコード」に検索語で示します。

| 作業 | M4 の実装 |
|---|---|
| 入力 | `HSE06` を受け付け、下の表の `scf.exx.*` を読みます。`PBE0` は拒否します（M8）。2.2 節の拒否は、入力ファイルを閉じる直前に置きました。 |
| 準局所の交換 | `Set_Hamiltonian`、`Set_Vpot`、`Total_Energy`、`Stress` に混合比と $\omega$ の引数を加え、`DFT` 関数が EXX の状態の値を渡します。混合比は、最初の $K$ ができるまで 0 です。`OutData.c`、`OutData_Binary.c` の呼び出しと、`Total_Energy.c` の `Calc_Edc` と `Energy_Decomposition_CWF` の中の呼び出しは、0 を渡したままです。エネルギーの分解は 2.2 節で拒否しています。`level.of.fileout 2` で書き出す `.vxc0` と `.vxc1` は、PBE のポテンシャルになります。 |
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
| 入力 | キーワードは加えていません。`Input_std` が、既存の障害注入の環境変数 `OPENMX_GPU_CAPACITY_TEST_RECOVERY` の値を `Exact_Exchange_Configure` に渡します。実機の段階で、既存のキーワード `scf.gemmul8.enable` が `on` かどうかも渡す形にしました。 |
| GEMMul8 | 実機の段階で加えました。入力に `scf.gemmul8.enable on` を書いた計算だけが、デバイスに送る積を GEMM の橋渡しに通します。キーワードがない入力と `off` の入力は、FP64 の cuBLAS のままです（11 節）。 |
| 標準出力 | 計算の終わりに、デバイスで計算した積の数、そのプロセスの数、時間、容量の不足でホストに残した積の数を表示します。CPU 版と、デバイスに送る大きさの積がない計算では表示しません。`level.of.stdout 2` では、構築ごとに、ランク 0 の積の数と最大の積の大きさを表示します。 |
| 密行列への加算 | 実機なしの段階では変えていません。実機の段階で、デバイスの密行列ソルバーの 2 つの経路（Band と Cluster）に加えました（6.5 節）。MAGMA の経路と gpusolver2 の経路には加えていません。 |
| 密度行列の収集 | 実機の段階で、デバイスの密行列ソルバーの経路からも収集する形にしました（6.3 節）。 |
| 固有値ソルバーの精度 | 実機の段階で加えた条件です。ハイブリッド汎関数が有効な間、FP32 で解いて精製する方式に入りません。 |
| 標準出力（GEMMul8） | 実機の段階で加えました。`scf.gemmul8.enable on` の計算は、GEMMul8 が計算した積の数と、橋渡しが FP64 で計算した積の数を 1 行で表示します。デバイスで計算した積がない計算では表示しません。 |
| 標準出力（密行列の経路） | 実機の段階で加えました。計算の終わりに、デバイスで $K$ を足したハミルトニアンの数を表示します。数が 0 の計算では表示しません。容量の判定が通らずに CPU の固有値ソルバーへ切り替わるときは、その旨を 1 行で表示します（11 節）。 |
| XC の GPU 経路 | M6 では変えていません。ハイブリッド汎関数の計算は、CUDA 版でもホストのループを使いました。GPU の経路は、改訂 25 で加えました（下の表）。 |
| ビルド | `source/Exact_Exchange_Device.c` を `OBJS` に加えました。CPU 版と CUDA 版に入り、HIP 版には入りません。`tools/check_gpu_compilation.py` の CUDA の一覧にも加えています。 |
| 試験 | `tests/run_exx_device_product.py` が、GPU のライブラリーの模型で部品を試験します。CI では `gpu-contracts` ジョブが実行します。実機の試験は `tools/exx_gpu_checks.py` が実行します（9 節の M6）。 |

構築のスレッド並列（改訂 14）で組み込んだ範囲です。場所は付録 A の「構築のスレッド並列で加えたコード」に検索語で示します。

| 作業 | 実装 |
|---|---|
| スレッドの数 | `Exact_Exchange_Build` が、本体の OpenMP のスレッド数（`-nt`）を `omp_get_max_threads` で読み、関数表の `threads` に渡します。構築が使うスレッドは、BLAS が持つスレッドの数によらず、この数です。`-nt` が BLAS のスレッド数より小さい実行では、ループの外の大きい積が、改訂 13 までより少ないスレッドで動きます。1 スレッドのときはループを渡さず、経路は改訂 13 までと同じです。 |
| ループ | 関数表の `parallel_for` は、`Exact_Exchange_Host_Parallel_For` です（`source/Exact_Exchange_Host.c`）。OpenMP の `parallel for` で、チームのスレッド数は `threads` です。ループの中の積は、呼んだスレッドの上で本体の `DGEMM` に渡します（`Exact_Exchange_Host_Loop_Dgemm`）。デバイスには送りません。積の数と最大の積は、スレッドごとに数えて、ループの後で合計します。ループの本体からこの関数を呼ぶと、新しいチームを作らず、呼んだスレッドで順に実行します。`modules/exx` は、この呼び方をしません（7.2 節）。 |
| ループの外の積 | デバイスが引き受けなかった積は、`Exact_Exchange_Host_Dgemm` が計算します（`source/Exact_Exchange_Host.c`）。結果の行列の長い方の辺を、長さがほぼ等しい区間に分けます。区間ごとに、別のスレッドが `DGEMM` を 1 回呼びます。区間の数は、スレッドの数と辺の長さを超えず、1 回の呼び出しの乗算が平均で 2.5e5 回を下回らない数です。区間の長さの差は 1 以下なので、どの呼び出しの乗算も、その半分より多くなります。乗算が 5e5 回未満の積は分けません。チームのスレッド数は、区間が `threads` より少ない積でも `threads` です（13 節の 59）。分けない積は、構築を呼んだスレッドが、並列の領域の外で 1 回呼びます。OpenMP のスレッドで動く BLAS は、この積を自分のチームで計算できます。 |
| BLAS のスレッド | 自前のスレッドを持つ OpenBLAS（pthread 版）は、`modules/exx` の計算の間だけ 1 スレッドにして、最後の中心の原子の後で元の数に戻します（13 節の 56）。見分けには `openblas_get_parallel` の戻り値を使います。OpenBLAS の 3 つの関数の探し方は、ビルドによって 2 通りあります（13 節の 60）。CPU 版のビルドが OpenBLAS を選んだときは、部品を `OPENMX_EXX_HOST_BLAS_OPENBLAS` の定義を付けてコンパイルし、関数をそのまま参照します。`OPENMX_CPU_LINALG` が `OPENBLAS` のときと、`AUTO` が OpenBLAS を見つけたときです。ほかのビルドは、弱い参照で関数の有無を調べます。macOS では、弱い参照を使いません。OpenMP のスレッドで動く BLAS、1 スレッドの BLAS、OpenBLAS でない BLAS は変えません。 |
| BLAS への要求 | 実行時に確かめていない要求が 2 つあります。`DGEMM` を複数のスレッドから同時に呼べること。同じ引数に対して、どのスレッドでも、配列がメモリーのどこにあっても、同じビットを返すこと。2 つ目は、同じスレッド数の実行が同じ $K$ を返すための条件です。クラウドのコンテナの OpenBLAS 0.3.26 は、構築の 12 通りの形と 92 通りの配置で、同じビットを返しました（点検での実測）。MKL と AOCL では実行していません（14 節）。 |
| 入力 | キーワードは加えていません。 |
| 標準出力 | 変えていません。`level.of.stdout 2` の積の数と最大の積は、スレッドの数によりません。 |
| ビルド | `source/Exact_Exchange_Host.c` を `OBJS` に加えました。CPU 版と CUDA 版に入り、HIP 版には入りません。このファイルは、どのビルドでも位置独立のコード（`-fPIC`）でコンパイルします（13 節の 60）。CMake の CPU 版は `cmake/OpenMXCPU.cmake` が、ネイティブのビルドは `source/Makefile` の規則が、このオプションを付けます。 |
| GPU のコンパイルの検査 | `tools/check_gpu_compilation.py` の CUDA の一覧に、`Exact_Exchange_Host.c` を加えました。`Exact_Exchange.c` と `Exact_Exchange_Host.c` は、OpenMP なしと OpenMP ありの 2 通りでコンパイルします。本体のビルドは OpenMP を有効にするので、`#ifdef _OPENMP` の中の行は、OpenMP ありのコンパイルでだけ検査されます。検査は、OpenMP ありのオブジェクトが OpenMP の実行時を参照することと、OpenBLAS の 3 つの関数が弱い参照であることも確かめます。 |
| 試験 | `tests/run_exx_host_product.py` が、`DGEMM` と OpenBLAS の関数の模型で部品を試験します。プログラムは 5 つです。OpenMP あり、OpenMP なし、OpenBLAS の関数なし、`openblas_get_parallel` だけがない形（静的な OpenBLAS と同じ状態）、関数をそのまま参照する形です。1 つ目は、スレッド数の上限（`OMP_THREAD_LIMIT=2`）を付けた実行も加えます。チームが、部品の求めた数より小さくなる場合です。確かめる内容は、積の分け方と結果、`int` の範囲を超える大きさの配列での区間の位置、ループの呼び出しとスレッドの割り当て、スレッドごとの集計、集計の配列の外に書かないこと、BLAS のスレッドの切り替えです。`--thread-sanitizer` を付けると、Clang と LLVM の OpenMP の実行時で、ThreadSanitizer の実行を 1 つ行います。CI では `gpu-contracts` ジョブが、GCC と Clang で実行します。Clang では、ThreadSanitizer の実行も行います。`modules/exx` の側の試験は 7.2 節にあります。 |

構築の高速化の 3 番目の段階（改訂 17）で組み込んだ範囲です。コミットは `2e2a6da`（本体）と `09e054a`（試験の道具）です。場所は付録 A の「構築の高速化の 3 番目の段階で加えたコード」に検索語で示します。

| 作業 | 実装 |
|---|---|
| 呼び出しの全体 | CUDA 版の `Exact_Exchange_Build` は、関数表の 5 番目のメンバーに、部品のデバイスの表（`Exact_Exchange_Device_Table`）を渡します。デバイスは、積の乗算の和が 1e9 回以上で、配列が空きから 512 MiB を引いた範囲に入る呼び出しを引き受けます（11 節）。改訂 28 までは、和でなく最大の積の回数で決めていました。引き受けなかった呼び出しは今までどおりホストで計算し、その中の大きい積を 1 個ずつデバイスに送ります。CPU 版では表が `NULL` で、経路は改訂 16 と同じです。 |
| ストリーム | 積（cuBLAS）と、要素の写しと加算（OpenACC のループ）を、OpenACC のキュー 1 のストリームに順に載せます（11 節）。cuBLAS のハンドルは、呼び出しの間だけこのストリームに切り替え、`Exact_Exchange_Device_Call_End` が元に戻します。本体のほかのコードは、キュー 1 を使っていません。 |
| 写しと加算 | `move` は、`#pragma acc parallel loop collapse(2) async(1)` のループ 1 個です。ポインターは `deviceptr` で渡します。積を先に作り、それから足します。このため、`source/Makefile` は `Exact_Exchange_Device.c` を `-gpu=nofma` でコンパイルします（13 節の 65）。 |
| 入力 | キーワードは加えていません。障害注入の環境変数 `OPENMX_GPU_CAPACITY_TEST_RECOVERY` は、呼び出しにも効きます。全ての呼び出しと積が、容量の不足としてホストに残ります。 |
| 標準出力 | 計算の終わりに、デバイスで計算した呼び出しの数を 2 行で表示します。`scf.gemmul8.enable on` の計算では、GEMMul8 が計算した積の数の 1 行が加わります。`level.of.stdout 2` では、構築ごとにランク 0 の呼び出しの数を表示します（11 節）。CPU 版と、デバイスが呼び出しを引き受けない計算では、表示しません。 |
| デバイスの失敗 | 失敗を知らせる 2 行目を `<EXX>  The host computes the exchange matrix and its matrix products from here on.` に変えました。失敗の後は、呼び出しも積もホストが計算するからです。 |
| ビルド | 新しい翻訳単位はありません。`source/Makefile` に、`Exact_Exchange_Device.o` の規則を加えました（`-gpu=nofma`）。`tools/check_gpu_compilation.py` も、この単位を `-gpu=nofma` 付きでコンパイルします。HIP 版には入りません。 |
| 試験 | `tests/run_exx_device_product.py` に、呼び出しの全体の試験を加えました。ループをホストで実行する定義（`OPENMX_EXACT_EXCHANGE_DEVICE_HOST_MOVES`）を付けてコンパイルします（11 節）。CI の `gpu-contracts` は、この試験を GCC と Clang で実行します。実機の試験の道具 `tools/exx_gpu_checks.py` は、呼び出しの数も確かめます（`09e054a`）。 |

MPI 並列での高速化（改訂 21）で組み込んだ範囲です。変えたファイルは、`source/Exact_Exchange.c` と、そのヘッダーのコメントだけです。場所は付録 A の「MPI 並列での高速化で加えたコード」に検索語で示します。

| 作業 | 実装 |
|---|---|
| 構造ごとの準備 | `Exact_Exchange_Prepare` が、`openmx_exx_prepare_part` にプロセスの番号と数を渡します。複数のプロセスの計算では、続けて `ExactExchange_Add_Up_Blocks` がブロックの和を取ります（6.7 節）。 |
| 構築 | `Exact_Exchange_Build` が、`ExactExchange_Abfs` の返す範囲を、`openmx_exx_add_gamma_abfs` か `openmx_exx_add_blocks_abfs` に渡します。関数表（積、ループ、スレッドの数、デバイスの表）は、今までと同じものを渡します。 |
| 力 | `Exact_Exchange_Gradient` が、同じ範囲の重みを計算して和を取り、`openmx_exx_add_position_gradient_pairs` に対の分け方を渡します。行列積は、今までどおりホストの BLAS だけです。 |
| 入力 | キーワードは加えていません。 |
| 標準出力 | `level.of.stdout 2` の行を変えました（6.7 節）。構築の行は、中心の原子の範囲に代えて、ABF の範囲を表示します。複数のプロセスの計算では、受け持ちの時間の最小と最大の行が加わります。力の行の次に、重みの時間の行が加わります。`local RI:` の行は、全てのプロセスの集計になりました。`level.of.stdout 1` の行は変えていません。 |
| ハイブリッド汎関数でない計算 | 変えていません。結合層の関数は、今までどおり最初の行で戻ります。GGA-PBE の `.out` は、前の版と日付と時間の行だけが違います（9 節）。 |
| ビルド | 新しい翻訳単位はありません。HIP 版には入りません。 |
| 試験 | `modules/exx` の試験に、分けた準備、ABF の範囲の構築と重み、対の経路を加えました（9 節）。本体は、プロセスの数を変えた実行を前の版と比べています（9 節、12 節）。試験の道具 `tools/exx_gpu_checks.py` が読む行は、変わっていません。 |

M8 の PBE0（改訂 22）で組み込んだ範囲です。コミットは `7b77629`（`modules/xc`）、`c9e26e7`（本体）、`b8cbe29`（本体の $\kappa$ の区間）です。

| 作業 | 実装 |
|---|---|
| 入力 | `scf.XcType PBE0` を受け付け、`scf.exx.coulomb.cutoff` を読みます。HIP 版のビルドは拒否します。2.2 節の拒否は HSE06 と同じです。既定の値より大きい `scf.exx.coulomb.cutoff` も拒否します（改訂 24）。 |
| 準局所の交換 | `modules/xc` の `openmx_xc_pbe_global_hybrid` が、PBE の交換を $1-\alpha$ 倍します（7.5 節）。結合層は `Set_XC_Grid` に渡す $\omega$ を 0 にして、PBE0 を HSE06 と区別します。 |
| カーネルと V 近傍 | `Exact_Exchange_Prepare` が、切断クーロンのカーネルと $R_c$ をモジュールに渡します。 $R_c$ は準備のたびにセルから計算します（5.6 節）。`scf.exx.v.cutoff` を書かない PBE0 は、V の範囲を $R_c+r_{\max}$ にします。 |
| $\kappa$ の区間 | 同じ関数が、区間の数を $R_c+3r_{\max}+v$ から決めます（5.6 節）。`1DFFT.NumGridR` で足りる計算では、区間は変わりません。 |
| 標準出力 | `level.of.stdout 1` 以上で、 $R_c$、V の範囲、 $\kappa$ の区間の数と上限を 1 行で表示します（`<EXX>  truncated Coulomb interaction:`）。 |
| ハイブリッド汎関数でない計算 | 変えていません。HSE06 の計算も、出力は日付と時間の行だけが違います（9 節の M8）。 |
| ビルド | 新しい翻訳単位はありません。 |
| 試験 | `modules/exx` に `exx-two-center-truncated` を加えました（13 節の 19）。本体の試験は、M4、M5、M7 の 3 つの道具に PBE0 の実行を加えた形です（9 節の M8）。 |

利用者の 2 つの指示による高速化（改訂 25）で組み込んだ範囲です。9 節の「力のデバイス化とほかの高速化」に、結果があります。

| 作業 | 実装 |
|---|---|
| 力の重み | `Exact_Exchange_Gradient` が、構築のデバイスを重みの計算に渡します（`df30625`）。1 個ずつの大きい積が通るのは、`state->device` です。CUDA 版では、スピンごとの呼び出しの全体を `Exact_Exchange_Device_Table` の表で計算します。重みのデバイスの数は、構築の数に入れません。呼び出しが失敗したときは、構築と同じ警告を 1 回表示します。残りの重みと後の構築は、ホストが計算します。 |
| $K(\mathbf k)$ の加算 | Band ソルバーの加算が、構築ごと、k 点ごとに 1 回だけ作った和を足します（6.5 節、`f5d746d`）。`DFT` 関数は、和の数と、保持した和を足した回数を表示します（`level.of.stdout 2`）。 |
| 構築 | `Exact_Exchange_Build` が半分の構築を呼び、プロセスの和を $X+X^{\mathsf T}$ にします（`d768793`）。使う関数は `openmx_exx_mirror_blocks` か `openmx_exx_mirror_gamma` です。構築の後の平均と `Exact_Exchange_Asymmetry` は、削除しました。ABF の範囲は、`openmx_exx_divide_abfs` が決めます（`92f78c7`）。 |
| 準局所の交換（GPU） | `Set_XC_Grid_GPU` の OpenACC のループが、PBE0 と HSE06 も計算します（`3fafc96`）。混合比が 0 でない点では、 $\omega$ が 0 なら `openmx_xc_kernel_pbe_global_hybrid` を呼びます。0 でなければ、`openmx_xc_kernel_pbe_screened_hybrid` です。GPU の集団の仕事と、ランクの間で送るデータに、混合比と $\omega$ を加えました。GGA-PBE のループは、字下げのほかは変えていません。HIP 版は、ホストのループのままです。 |
| ビルド | `source/openmx_xc_hybrid_kernels.h` が、`modules/xc` のヘッダーを取り込みます。`source/Makefile` には、`Set_XC_Grid.o` の依存を加えました。`tools/check_gpu_compilation.py` の CUDA の一覧にも、`Set_XC_Grid.c` を加えています（`f0421a6`）。新しい翻訳単位はありません。 |
| 入力 | キーワードは加えていません。 |
| 標準出力 | `level.of.stdout 2` に、重みのデバイスの 2 行と、 $K(\mathbf k)$ の和の行が加わりました。 $K$ の非対称の行は、なくなりました。 |
| ハイブリッド汎関数でない計算 | CPU 版の GGA-PBE の `.out` は、前の版と日付と時間の行だけが違います。CUDA 版の GGA-PBE は、利用者の計算機での確認を待っています（14 節）。 |
| 試験 | `modules/exx` に、半分の構築、重みのデバイス、ABF の分け方の試験を加えました。`modules/xc` には、インラインの関数と同じスピンの短距離交換の試験があります。M6 の道具は、重みのデバイスと力も確かめます（9 節）。 |

結果を変える高速化（改訂 31）で組み込んだ範囲です。9 節の「力のデバイス化とほかの高速化」に、結果があります。場所は付録 A の「改訂 31 で変えたコード」に検索語で示します。

| 作業 | 実装 |
|---|---|
| 混合の始め直し | `Exact_Exchange_Mixing_Offset` が、RMM-DIIS、GR-Pulay、RMM-ADIIS の最初の混合のステップ数を `StartPulay - 1` にします（6.6 節）。 |
| 外挿 | `DFT` が、外側反復を続けると決めた後で `Exact_Exchange_Extrapolate` を呼びます。履歴の配列は結合層の状態にあり、構造ごとの準備で空にします。 |
| 内側の判定 | `DFT` が、`Exact_Exchange_Inner_Criterion` の値で、既存の 2 つの条件を調べ直します。外側の閾値を満たした構築の後は、この値が `scf.criterion` です。その値だけで止まったことは、`Exact_Exchange_Outer_Continue` の 2 番目の引数で伝えます。 |
| 入力 | 2 つのキーワードを加えました（下の表）。範囲の外の値は、入力の時点で止めます。 |
| 標準出力 | 外挿の係数の行と、緩い判定で止まった内側の SCF の行が加わりました（`level.of.stdout` が 1 以上）。外挿の時間は、`level.of.stdout 2` で表示します。 |
| ハイブリッド汎関数でない計算 | 3 つとも、 $K$ がないと何もしません。GGA-PBE の `.out` は、前の版と日付と時間の行だけが違います。 |
| 試験 | SCF の道具に、2 つのキーワードの 8 件を加えました（9 節）。 |

追加する入力キーワードの既定値は出発点で、M4 と M5 の実測の後に見直します。M5 の実測値は進捗ファイルにあります。既定値を変えるかどうかは、利用者が決めます。

| キーワード | 既定値 | 意味 |
|---|---|---|
| `scf.XcType` | 変更なし | 選択肢に `HSE06` と `PBE0` を追加 |
| `scf.exx.alpha` | 0.25 | 厳密交換の混合比 |
| `scf.exx.omega` | 0.11 | HSE06 の遮蔽パラメーター（bohr⁻¹） |
| `scf.exx.abf.pca` | 1.0e-4 | ABF の主成分分析の閾値（最大固有値に対する比） |
| `scf.exx.v.cutoff` | 8.0（HSE06）、自動（PBE0） | V 近傍を決める打ち切り半径（bohr）。書かない PBE0 では $R_c$ + 最大の ABF の半径 |
| `scf.exx.outer.maxIter` | 30 | 外側反復の上限 |
| `scf.exx.outer.criterion` | 1.0e-6 | 式 (15) の閾値（Hartree） |
| `scf.exx.coulomb.cutoff` | 自動 | PBE0 の切断半径（bohr）。書かないか 0 なら $(3N_k\Omega/4\pi)^{1/3}$ 。それより大きい値は拒否する（1e-6 bohr までの超過は受け付ける） |
| `scf.exx.outer.extrapolation` | 0 | 外側反復の外挿で、新しい構築と合わせる前の構築の数（0 から 8）。0 なら外挿しない（6.6 節） |
| `scf.exx.inner.criterion.ratio` | 0.0 | 構築の後の内側の SCF を、`scf.criterion` と、この比に $\lvert\delta\rvert$ を掛けた値の大きい方でも止める（0.0 から 1.0）。 $\lvert\delta\rvert$ が `scf.exx.outer.criterion` より小さい構築の後と、0.0 では、`scf.criterion` だけ（6.6 節） |

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

記録には、分極関数のある基底の実行も入れました（C6.0-s2p2d1 のダイヤモンドと、Si7.0-s2p2d1 の Si）。道具は、この 2 つの結晶について、メッシュの k 点の上のバンド端を表にします。M5 の時点では `Band.dispersion` を拒否していたので（2.2 節）、ギャップは SCF のメッシュの固有値から求めました。伝導帯の底がメッシュの点にない結晶では、この値は結晶のギャップより大きくなります。

### M6. GPU（CUDA）

| 項目 | 内容 |
|---|---|
| 目標 | CUDA のビルドで、6.4 節の行列積と 6.5 節の加算を GPU で実行します。 |
| 前提 | M5。GPU の実機 |
| 操作 | 11 節 |
| 成功の確認方法 | CPU と GPU の全エネルギーが 1e-7 Hartree 以内で一致します（README の回帰比較が使っている閾値と同じです）。GPU のメモリーが足りない場合は、CPU の経路に切り替わって同じ結果を返します。段ごとの時間の記録も完了の条件です。 |
| 失敗例 | 実機がない環境では、コンパイルの検査（`tools/check_gpu_compilation.py --backend CUDA`）までを実行し、実機の試験が残っていることを報告します。 |

M6 は、GPU の実機が要るかどうかで 2 つに分けました。2026-11-24 に、実機なしで進められる部分を終えています。同じ日に利用者の計算機（RTX 5080）をつなぎ、実機の段階を始めました。実機の最初の段階では、`38992bc` の 18 件が上の表の 3 つの条件を満たしました。2 番目の段階では、`a7fbfb8` の 53 件が同じ条件を満たしています。デバイスの密行列ソルバーの経路と GEMMul8 の切り替えを含む版です。その後に、交換行列の構築を書き直しました（`857ede9`、6.4 節）。3 番目の段階では、書き直した版の 53 件が同じ条件を満たしました。`857ede9` は、コードを変えた最後のコミットです。M6 は、この結果で完了としました（2026-11-24）。

| 区分 | 内容 | 状態 |
|---|---|---|
| 行列積 | 6.4 節の行列積のうち大きいものを、FP64 の cuBLAS で計算します。容量が足りない積はホストに残します（11 節）。 | コードを書き、GPU のライブラリーの模型で試験しました。実機の最初の段階で、CUDA 版をリンクして実行しています。4 つの系で、予定した数の積をデバイスで計算しました。2 番目と 3 番目の段階でも同じ数です（下の結果）。 |
| 試験の道具 | `tools/exx_gpu_checks.py` が、上の表の 3 つの条件を実機で確かめます。 | CPU 版の実行ファイルを 2 つの役に当てて、道具の動作を確かめました。実機の段階で、デバイスの密行列ソルバーを通る実行と、GEMMul8 を使う実行を加え、53 件にしています。同じ段階で、道具がスレッド数を本体に渡していなかったことが分かり、`a7fbfb8` で直しました（13 節の 46）。2 番目と 3 番目の段階では、本体が表示したスレッド数が、指定した 1 と 18 に合いました。 |
| 6.5 節の加算 | デバイスの密行列ソルバーの経路で $K$ を足します。 | 実機の段階で、Band と Cluster のデバイス上の経路に書きました。密度行列の収集も加えています。MAGMA の経路には書いていません（6.5 節）。CI の CUDA のコンパイルの検査を通っています。実機の 2 番目の段階では、12 個の系の 16 件が $K$ をデバイスで足しました。全エネルギーは、CPU 版と 2.1e-9 Hartree 以内で合います（下の結果）。3 番目の段階でも 16 件が足し、密行列の経路の 19 件の差は 1.6e-9 Hartree 以内でした。 |
| GEMMul8 との比較 | 大きい積で、FP64 の cuBLAS と GEMMul8 を比べます（11 節）。 | 実機の段階で、切り替えを書きました。入力に `scf.gemmul8.enable on` を書いた計算だけが、大きい積を GEMMul8 に通します。道具の `NAME-g8` が、この実行です。実機の 2 番目の段階で測りました。全エネルギーは FP64 の実行と 5e-12 Hartree 以内で合い、デバイスの積の時間は約半分でした（11 節）。書き直した版（3 番目の段階）では、3e-12 Hartree 以内で合い、時間は FP64 の 0.59 倍から 0.69 倍です。固有値ソルバーの中の行列積を FP64 にした実行（`NAME-dfp64`）も測っています（下の結果）。 |
| 受け入れ試験 | 上の表の「成功の確認方法」 | `38992bc` の 18 件、`a7fbfb8` の 53 件、`857ede9` の 53 件が、3 つの条件を満たしました（下の結果）。`857ede9` は、コードを変えた最後のコミットです。M6 は完了しました。 |

6.5 節の加算を実機なしで書かなかった理由です。加算を入れる関数（`BandCol_ConstructDenseCsHs_OpenACC` など）は、ハイブリッド汎関数でない計算も通ります。この関数を変えると、GPU での結果が変わらないこと（1.2 節の最初の条件）を、実機なしでは確かめられません。行列積の部品は新しい翻訳単位で、通るのはハイブリッド汎関数の計算だけです。デバイスの密行列ソルバーの経路を使うには、その経路での密度行列の収集も要ります。実機なしの段階の収集は、ホスト上の固有ベクトルを受け取る形だけでした（6.3 節）。

計算機をつないだ後は、この理由が当てはまりません。加算と収集は、実機の最初の段階（`38992bc` の試験と測定）と並行して書きました。実機の 2 番目の段階では、先にハイブリッド汎関数でない計算を基準コミットの CUDA 版と比べました。既定のしきい値の実行と、しきい値を 1 に下げた実行の両方です。新しい経路の試験は、その後に実行しています。比べ方は、最初の段階で分かった性質に合わせました（下の結果と 13 節の 48）。

実機の試験は `tools/exx_gpu_checks.py` が実行します。同じコミットからビルドした CPU 版と CUDA 版の実行ファイルを渡します。道具は、系ごとに 3 つの計算をします。

| 実行 | 実行ファイル | 確かめること |
|---|---|---|
| `NAME-cpu` | CPU 版 | 比べる相手です。 |
| `NAME-gpu` | CUDA 版 | 全エネルギーが CPU 版と 1e-7 Hartree 以内で合うこと。大きい積がデバイスで計算されること。 |
| `NAME-host` | CUDA 版。環境変数 `OPENMX_GPU_CAPACITY_TEST_RECOVERY=1` を付けます。 | 容量が足りない場合と同じ道筋で、大きい積が全てホストに残ること。全エネルギーが CPU 版と 1e-7 Hartree 以内で合うこと。 |

系は 6 つです。メタン（Cluster ソルバー）と C5.0-s1p1 のダイヤモンドには、デバイスに送る大きさの積がありません。この 2 つは、CUDA 版のホスト上の経路が CPU 版と同じ結果を出すことを確かめます。Si7.0-s2p2d1 の Si（1 ランク、スピン分極、2 ランクの 3 通り）と C6.0-s2p2d1 のダイヤモンドでは、構築 1 回、スピン 1 成分あたり 4 個の積がデバイスに行きます（11 節の実測）。`table` は、エネルギーの差、デバイスで計算した積の数、段ごとの時間を表にします。積の数が予定と違う場合も、終了コードが 1 になります。使い方は進捗ファイルの「ビルドと実行」にあります。

実機の最初の段階の結果です（2026-11-24、`38992bc`）。計算機は、Core i9-10980XE（18 コア）と GeForce RTX 5080（16,303 MiB）の組です。CUDA 版は、ホストの NVHPC 26.9 と CUDA 13.4 でビルドしました。CPU 版は、GCC 13.3.0 と OpenBLAS 0.3.26 のコンテナでビルドして実行しています。CUDA 版のリンクは、これが初めてです。18 件は全て収束し、`table` の行は全て合格でした。どの実行も、OpenMP は 1 スレッドです（13 節の 46）。

| 系 | CPU 版の全エネルギー（Hartree） | CUDA 版との差 | 積をホストに残した CUDA 版との差 | デバイスで計算した積 | 障害注入でホストに残した積 |
|---|---|---|---|---|---|
| `ch4` | −7.999482449274 | −4.5e-10 | +7.1e-11 | 0 | 0 |
| `dia-p222` | −11.547438022706 | +1.0e-12 | +6.0e-12 | 0 | 0 |
| `siz-p444` | −8.133954712061 | 0 | 0 | 40 | 40 |
| `siz-p444-spin` | −8.133954712060 | −1.0e-12 | 0 | 80 | 80 |
| `siz-p444-np2` | −8.133954712062 | +1.0e-12 | +2.0e-12 | 40（2 プロセス） | 40 |
| `diaz-p222` | −11.676461160568 | −1.1e-11 | −1.1e-11 | 48 | 48 |

差は、CUDA 版の値から CPU 版の値を引いたものです（Hartree）。最大はメタンの 4.5e-10 Hartree で、閾値（1e-7 Hartree）の約 220 分の 1 です。積の数は、構築 1 回、スピン 1 成分あたり 4 個の予定と合いました（構築は 10 回と 12 回）。障害注入の実行では、同じ数の積が全てホストに残っています。デバイスの呼び出しの失敗は、どの実行にもありません。1 個の積がデバイスで使ったメモリーは、最大で 135.1 MB（Si）と 486.8 MB（s2p2d1 のダイヤモンド）でした。段ごとの時間は 11 節に、実行ごとの値は進捗ファイルにあります。

同じ段階で、ハイブリッド汎関数でない計算を、基準コミット（`4e0ee24`）の CUDA 版と比べました。入力は、`work/input_example` の `Methane.dat`、`Graphite4.dat`、`Crys-MnO.dat` です。2 つの実行ファイルの全エネルギーの差は、最大で 3.6e-11 Hartree でした。基準の実行ファイルを 2 回実行したときの差は、最大で 2.3e-11 Hartree です。CUDA 版の GPU を使う経路は、同じ実行ファイルでも実行ごとに末尾の桁が変わります（13 節の 48）。このため、`.out` の差が日付と時間の行だけになることは確かめられませんでした。`OPENMX_GPU=0` を付けて全てのランクをホストのコードに残すと、2 つの実行ファイルの全エネルギーは 3 つの入力で同じになりました。このときの `.out` の差は、日付と時間の欄だけです。

実機の段階で、デバイスの密行列ソルバーを通る実行を道具に加えました（`c6cdb9b`）。CUDA 版がデバイスで解くのは、行列の次数がしきい値（Band は 800、Cluster は 400）以上の計算です。道具の系の次数は 8 と 26 なので、上の 3 つの実行はホストで解きます。加えた実行は、既存の環境変数でしきい値を 1 に下げます。

| 実行 | 加える環境変数 | 確かめること |
|---|---|---|
| `NAME-dense` | `OPENMX_BAND_GPU_SWITCH_NUM=1`、`OPENMX_CLUSTER_GPU_SWITCH_NUM=1` | デバイスの密行列ソルバーで解き、 $K$ をデバイスで足すこと。全エネルギーが CPU 版と 1e-7 Hartree 以内で合うこと。 |
| `NAME-dfall` | 上の 2 つと、`OPENMX_BAND_GPU_RESERVE_MB=1000000`、`OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB=1000000` | 容量の判定が通らず、CPU の固有値ソルバーに切り替わって同じ結果を返すこと。 |
| `NAME-dpers` | 上の 2 つと、`OPENMX_BAND_GPU_PERSISTENT=1` | 固有ベクトルをデバイスに残す方式で、収集が同じ結果を返すこと。 |
| `NAME-dser` | 上の 2 つと、`OPENMX_CLUSTER_GPU_DIAG_SERIAL=2` | 1 つのプロセスが 2 つのスピンを順に解く方式で、同じ結果を返すこと。 |
| `NAME-dfp64` | 上の 2 つと、`OPENMX_GEMMUL8_DISABLE=1` | 固有値ソルバーの行列積を FP64 の cuBLAS にした実行との差（11 節）。 |

系は 6 つ加えて 12 個にしました。加えた系は、密行列の経路でランクの役割が変わる配置を作ります。Cluster ソルバーでは、密行列を持たないランクのある計算と、2 つのスピンを 1 ランクと 2 ランクで解く計算です。Band ソルバーでは、スピンごとに 1 ランクの計算、ランクごとに k 点が 1 つの計算、2 ランクが 1 つの k 点を受け持つ計算です。この版の実行は 49 件でした。`table` は、密行列の経路の実行ごとに、CPU 版との全エネルギーの差と、 $K$ をデバイスで足したハミルトニアンの数を判定します。系の一覧は、進捗ファイルの「ビルドと実行」にあります。

同じ段階で、GEMMul8 を使う実行を加えました。デバイスに送る大きさの積がある 4 つの系（`siz-p444`、`siz-p444-spin`、`siz-p444-np2`、`diaz-p222`）に、`NAME-g8` があります。実行は全部で 53 件です。

| 実行 | 入力に足す行 | 確かめること |
|---|---|---|
| `NAME-g8` | `scf.gemmul8.enable on` | デバイスで計算した積を、全て GEMMul8 が計算すること。全エネルギーが CPU 版と 1e-7 Hartree 以内で合うこと。 |

`NAME-g8` は CUDA 版の実行で、環境変数を付けません。しきい値を下げないので、固有値問題はホストで解きます。`NAME-gpu` との違いは、デバイスに送る積の方式だけです。`table` は、`NAME-g8` と `NAME-gpu` の全エネルギーの差、厳密交換のエネルギーの差、デバイスで使った時間を表示します。この 3 つに閾値はありません。`NAME-g8` 以外の実行が GEMMul8 の行を表示すると、行が失敗になります。

M6 の受け入れは、上の表の 3 つの条件を既定の実行（`NAME-gpu`、`NAME-host`、密行列の経路の実行）で判定します。`NAME-g8` の行は、GEMMul8 との比較の記録です。この行が失敗しても、`table` の終了コードは 1 になります。その場合は、失敗した行の名前で区別します。

道具の実行は、既定で 1 スレッドです（`a7fbfb8` から）。`--threads` を付けると、その数を本体の `-nt` と BLAS に渡します。`table` は、本体が表示したスレッド数を表示し、指定と違う実行があると行を失敗にします（13 節の 46）。

実機の 2 番目の段階の結果です（2026-11-24、`a7fbfb8`）。計算機とビルドの環境は、最初の段階と同じです。CPU 版と CUDA 版は、ビルドのディレクトリーを消して最初からビルドしました。53 件は全て収束し、`table` の行は全て合格でした。どの実行も、OpenMP は 1 スレッドです。

| 系 | プロセス | CPU 版の全エネルギー（Hartree） | CUDA 版との差 | 密行列の経路との差 | デバイスで計算した積 | $K$ をデバイスで足したハミルトニアン |
|---|---|---|---|---|---|---|
| `ch4` | 1 | −7.999482449274 | +2.0e-10 | +4.0e-10 | 0 | 37 |
| `dia-p222` | 1 | −11.547438022706 | +3.0e-12 | +2.0e-12 | 0 | 360 |
| `siz-p444` | 1 | −8.133954712063 | +2.0e-12 | +2.0e-12 | 40 | 2,520 |
| `siz-p444-spin` | 1 | −8.133954712061 | 0 | −1.0e-12 | 80 | 5,112 |
| `siz-p444-np2` | 2 | −8.133954712062 | +1.0e-12 | 0 | 40（2 プロセス） | 2,556 |
| `diaz-p222` | 1 | −11.676461160540 | −4.1e-11 | −3.8e-11 | 48 | 792 |
| `ch4-np4` | 4 | −7.999482449035 | +1.5e-10 | −6.0e-10 | 0 | 42 |
| `o2` | 1 | −32.085398782773 | 0 | +2.1e-9 | 0 | 54 |
| `o2-np2` | 2 | −32.085398782773 | −1.1e-9 | 0 | 0 | 62 |
| `dia-p222-spin-np2` | 2 | −11.547438022703 | 0 | −2.0e-12 | 0 | 640 |
| `dia-k211-np2` | 2 | −12.183507308471 | +1.1e-11 | +5.0e-12 | 0 | 166 |
| `dia-k211-np4` | 4 | −12.183507308471 | +2.0e-12 | +1.0e-12 | 0 | 164 |

差は、CUDA 版の値から CPU 版の値を引いたものです（Hartree）。「CUDA 版との差」と「デバイスで計算した積」は `NAME-gpu` の値、「密行列の経路との差」と最後の列は `NAME-dense` の値です。実行ごとの値は、進捗ファイルにあります。

- 53 件の差の最大は、`o2-dense` の 2.1e-9 Hartree です。閾値（1e-7 Hartree）の約 48 分の 1 に当たります。次は `o2-np2-gpu` の 1.1e-9 Hartree で、ほかは 6.0e-10 Hartree 以下でした。
- 厳密交換のエネルギーの差は、53 件とも 2e-12 Hartree 以下です。全エネルギーの差が 1e-9 Hartree を超えた 2 件は、SCF のステップの数が CPU 版と違います（38 と 42）。差は、内側の SCF が止まる位置の違いから来ていると考えられます。確かめてはいません。
- デバイスで計算した積の数は、最初の段階と同じです。障害注入の実行（`NAME-host`）では、同じ数の積が全てホストに残りました。
- 容量の判定を通さない実行（`NAME-dfall` の 3 件）は、CPU の固有値ソルバーに切り替わり、 $K$ をデバイスで足した数が 0 でした。CPU 版との差は、5.2e-10 Hartree 以下です。
- 固有ベクトルをデバイスに残す方式（`dia-k211-np2-dpers`）と、1 つのプロセスが 2 つのスピンを順に解く方式（`o2-np2-dser`）は、CPU 版と 6e-12 Hartree 以内で合いました。
- 固有値ソルバーの行列積を FP64 にした実行（`NAME-dfp64` の 2 件）は、CPU 版との差が 4.7e-10 Hartree と 2e-12 Hartree です。既定の実行（`NAME-dense`）は、4.0e-10 Hartree と 2e-12 Hartree でした。このデバイスの固有値ソルバーは、既定で GEMMul8 を使います（FP32 と FP64 の処理量の比は 64。11 節）。
- `NAME-g8` の 4 件は、デバイスの積を全て GEMMul8 で計算しました（40、80、40、48 個）。全エネルギーは、同じ積を FP64 で計算した `NAME-gpu` と 5e-12 Hartree 以内で合います。
- デバイスの呼び出しの失敗と、`<EXX>  WARNING` の行は、どの実行にもありません。

同じ段階で、ハイブリッド汎関数でない計算を、基準コミットの CUDA 版と 3 通りの条件で比べました。入力は、最初の段階と同じ 3 つです。環境変数を付けない実行（実行ファイルごとに 2 回）では、2 つの実行ファイルの全エネルギーの差が最大 9.6e-11 Hartree でした。しきい値を 1 に下げた実行は、どの入力もデバイスの固有値ソルバーを通り、差は最大 1.6e-10 Hartree です。`Methane.dat` を実行ファイルごとに 30 回繰り返すと、全エネルギーの範囲は 5.51e-10 Hartree と 5.65e-10 Hartree でした。中央値の差は 5.0e-13 Hartree です。`OPENMX_GPU=0` を付けた実行では、3 つの入力とも、`.out` の差が日付と時間の欄だけでした。

大域状態のゲートの全体も、この段階で実行しました。M0 の後では、初めての全体の実行です。288 個の翻訳単位を解析し、ゲートが比べる 7 種類に増えた項目はありません。ベースラインは更新していません。FFTW を `build/` の下に作る木では、`FFTW3_PREFIX` を渡さないと 7 個の翻訳単位が解析されません（13 節の 52）。要約の数には、ベースラインと 1 だけ違うものがあります（13 節の 53）。

実機の 3 番目の段階の結果です（2026-11-24、`857ede9`）。交換行列の構築を書き直した版で、コードを変えた最後のコミットです。計算機とビルドの環境は、前の 2 つの段階と同じです。CPU 版と CUDA 版は、2 番目の段階のビルドのディレクトリーで、変わったファイルだけをビルドし直しました。53 件は全て収束し、`table` の行は全て合格でした。どの実行も、OpenMP は 1 スレッドです。

| 系 | プロセス | CPU 版の全エネルギー（Hartree） | CUDA 版との差 | 密行列の経路との差 | デバイスで計算した積 | $K$ をデバイスで足したハミルトニアン |
|---|---|---|---|---|---|---|
| `ch4` | 1 | −7.999482448976 | −2.8e-10 | −3.6e-10 | 0 | 37 |
| `dia-p222` | 1 | −11.547438022703 | 0 | −1.0e-12 | 0 | 320 |
| `siz-p444` | 1 | −8.133954712061 | 0 | 0 | 40 | 2,556 |
| `siz-p444-spin` | 1 | −8.133954712061 | 0 | 0 | 80 | 5,112 |
| `siz-p444-np2` | 2 | −8.133954712062 | +1.0e-12 | +1.0e-12 | 40（2 プロセス） | 2,592 |
| `diaz-p222` | 1 | −11.676461160528 | −4.9e-11 | −2.2e-11 | 48 | 744 |
| `ch4-np4` | 4 | −7.999482448967 | +4.2e-10 | −2.6e-10 | 0 | 40 |
| `o2` | 1 | −32.085398781212 | −2.3e-9 | −1.6e-9 | 0 | 62 |
| `o2-np2` | 2 | −32.085398782773 | +2.2e-9 | +1.1e-9 | 0 | 54 |
| `dia-p222-spin-np2` | 2 | −11.547438022702 | −1.0e-12 | −1.0e-12 | 0 | 640 |
| `dia-k211-np2` | 2 | −12.183507308472 | +2.0e-12 | +6.0e-12 | 0 | 164 |
| `dia-k211-np4` | 4 | −12.183507308472 | +6.0e-12 | +6.0e-12 | 0 | 164 |

列の意味は、2 番目の段階の表と同じです。実行ごとの値は、進捗ファイルにあります。

- 53 件の差の最大は、`o2-gpu` の 2.3e-9 Hartree です。閾値（1e-7 Hartree）の約 43 分の 1 に当たります。1e-9 Hartree を超えるのは `o2` と `o2-np2` の 4 件で、ほかは 4.2e-10 Hartree 以下でした。
- 厳密交換のエネルギーの差は、53 件とも 1e-12 Hartree 以下です。
- `o2` と `o2-np2` の実行は、2 つの段階を合わせて 14 件です。SCF のステップが 40 以上の 8 件は、全エネルギーが全て同じ値でした。38 ステップの 6 件が、その値から 2.2e-9 Hartree 以内でばらつきます。どちらの組にも、CPU 版と CUDA 版が入ります。差は、内側の SCF が止まる位置の違いから来ていると考えられます。止まる位置を決めている量は、調べていません。
- デバイスで計算した積の数は、前の 2 つの段階と同じです。障害注入の実行（`NAME-host`）では、同じ数の積が全てホストに残りました。
- 容量の判定を通さない実行（`NAME-dfall` の 3 件）は、CPU の固有値ソルバーに切り替わり、 $K$ をデバイスで足した数が 0 でした。CPU 版との差は、2.2e-10 Hartree 以下です。
- 固有ベクトルをデバイスに残す方式（`dia-k211-np2-dpers`）と、1 つのプロセスが 2 つのスピンを順に解く方式（`o2-np2-dser`）は、CPU 版と 2e-12 Hartree 以内で合いました。
- 固有値ソルバーの行列積を FP64 にした実行（`NAME-dfp64` の 2 件）は、CPU 版との差が 8.9e-11 Hartree と 0 です。既定の実行（`NAME-dense`）は、3.6e-10 Hartree と 0 でした。
- `NAME-g8` の 4 件は、デバイスの積を全て GEMMul8 で計算しました（40、80、40、48 個）。全エネルギーは、同じ積を FP64 で計算した `NAME-gpu` と 3e-12 Hartree 以内で合います。
- 2 番目の段階の同じ実行との全エネルギーの差は、最大 3.3e-9 Hartree です（`o2-np2-gpu`）。1e-9 Hartree を超えるのは `o2` と `o2-np2` の 4 件で、ほかは 7.5e-10 Hartree 以下でした。構築の回数は、53 件とも同じです。
- デバイスの呼び出しの失敗と、`<EXX>  WARNING` の行は、どの実行にもありません。

同じ段階で、ハイブリッド汎関数でない計算を、2 番目の段階の同じ実行と比べました。入力は同じ 3 つです。`OPENMX_GPU=0` を付けた実行の `.out` は、3 つの入力とも、2 番目の段階の `.out` と日付と時間の欄だけが違います。環境変数を付けない実行の全エネルギーは、2 番目の段階の 2 回の値と 9.8e-11 Hartree 以内で合いました。

M6 の完了は、この段階の結果で判断しました。上の表の 3 つの条件のうち、全エネルギーの一致と、容量が足りない場合の切り替えは、53 件の `table` の判定です。段ごとの時間は、11 節と進捗ファイルに記録しました。大域状態のゲートの全体は、`a7fbfb8` で実行しています。その後に変わった `source/` のファイルはありません。XC の GPU 経路は、M6 の目標（行列積と加算）に入らないので、完了の条件にしていません（14 節）。構築の速度について残した案は、14 節の「site の形式の構築の高速化」にあります。

M8 の PBE0 の確認と同じ機会に、HSE06 の表を `9e01d40` で実行し直しました（2026-11-25、利用者の計算機）。M7 の力と MPI 並列での高速化が入った版です。この 2 つの後に、CUDA 版を実機で実行したのは初めてです。表には 4 プロセスの `siz-p444-np4` を加えたので、57 件になります（MPI 並列での高速化の節）。57 件は全て収束し、`table` の終了コードは 0 でした。CUDA 版と CPU 版の全エネルギーの差は、最大 1.6e-9 Hartree です（`o2`）。厳密交換のエネルギーの差は、2e-12 Hartree 以下でした。デバイスを使う 5 つの系では、構築の呼び出しが全てデバイスで計算されました。呼び出しと大きい積の数は、設計の数と合っています。デバイスで計算する実行の交換行列の段は、実機の 6 番目の段階と 0.15 秒以内で同じでした（11 節）。DFT の全体は、M7 で加えた厳密交換の力の分だけ長くなっています（14 節の「厳密交換の力の計算の時間」）。

### 構築の高速化（M6 と M7 の間）

利用者が 2026-11-24 に、M7 より先に行う作業として選びました。この決定を、番号順に進める規則（1.1 節の 3）より優先します。M7 は、`modules/exx` の 4 つのコミット（`da77f7c` まで）で止めています。`source/` の力のルーチンと入力の拒否は、M6 のままです。

| 項目 | 内容 |
|---|---|
| 目標 | 交換行列の構築を速くします。 |
| 前提 | M6 |
| 操作 | 3 つの段階に分けます。1 番目は、中心の原子の ABF のループのスレッド並列です。2 番目は、スレッドに分けていない残りの処理です（ $D$ の行の収集、 $K$ への加算、配列の確保）。3 番目は、V 近傍ごとの計算をデバイスに移す形の設計と実装です。 |
| 成功の確認方法 | 1 スレッドの CPU 版の `.out` が、前の版と日付と時間の行だけ違います。スレッドを使う実行と CUDA 版の全エネルギーは、前の版と 1e-7 Hartree 以内で合います（M6 の閾値）。行列積の回数は変わりません。交換行列の段の時間を、同じ条件の前の版と比べて記録します。この 4 つの条件は、作業者が決めました。 |
| 失敗例 | 自前のスレッドを持つ BLAS を OpenMP のループと交互に使うと、構築が遅くなります（13 節の 56）。スレッドを使う実行の結果は、同じ実行ファイルでも末尾の桁が変わります（13 節の 58）。 |

1 番目の段階は、改訂 14 で実装しました。コミットは `a319e96`（`modules/exx`）と `1219cb9`（本体）です。実装は 6.4 節、7.2 節、8 節に、時間と結果の比較は 11 節と 12 節にあります。クラウドのコンテナの 4 スレッドでは、構築 1 回が Si で 0.61 倍、s2p2d1 のダイヤモンドで 0.55 倍になりました。利用者の計算機の 18 スレッドでは、交換行列の段が、前の版の 0.24 倍から 0.33 倍になりました（実機の 4 番目の段階。11 節）。実機の 30 件の実行は全て収束し、全エネルギーは比べた相手と 6.2e-10 Hartree 以内で合います。行列積の回数は、前の版と同じです。1 スレッドの `.out` の比較は、クラウドのコンテナで行いました（12 節）。上の表の 4 つの条件を満たしています。

この版を 6 つの観点で点検しました。結果の誤りとデータの競合の指摘は、ありません。指摘は 6 つのコミットで直しました（`b80a3c7`、`017d91a`、`e9b637c`、`16951be`、`ce92ffc`、`9888b44`）。直した点は、次の 5 つです。

- ホストの積のチームのスレッド数を、ループと同じ `threads` にしました（13 節の 59）。
- ループと、スレッドごとの積の集計を、`source/Exact_Exchange.c` から `source/Exact_Exchange_Host.c` に移しました（7.4 節）。模型の試験が届く場所です。
- OpenBLAS の関数の探し方を 2 通りにし、部品を位置独立のコードでコンパイルする形にしました（8 節、13 節の 60）。
- GPU のコンパイルの検査に、OpenMP ありのコンパイルを加えました（8 節）。
- 試験を増やしました。`modules/exx` では、区間の数、表の積と区間の組み合わせ、ThreadSanitizer です（7.2 節）。本体では、ループと集計、スレッド数の上限、OpenBLAS の関数の 3 つの形です（8 節）。`modules/exx` の古い試験の 9 か所では、C の入口の失敗を知らせる文に、呼び出しが書いたメッセージが入るようにしました。

CI の ThreadSanitizer の 2 つの手順は、最初の形が `9888b44` の実行で失敗し、`82eaa0a` で直しました（13 節の 61）。直した版は、2 番目の段階と合わせて、利用者の計算機で確かめました（実機の 5 番目の段階、`3d51128`。11 節）。

2 番目の段階は、改訂 15 で実装しました。コミットは `1b91c7b`（`modules/exx`）です。1 本の配列の site の形式と Γ 点の形式で、 $D$ の行の収集と $K$ への加算を、ABF の区間と同じ数に分けました（6.4 節）。本体は、C の入口を通してこの 2 つの形式を使います。本体のコードは変えていません。 $K$ は、前の版（`a8d7910`）とビットまで同じです。区間の数と呼び出しの順序を変えて確かめました。クラウドのコンテナの 4 スレッドでは、構築 1 回が前の版の 0.84 倍から 0.97 倍になりました（12 節）。1 スレッドの `.out` は、前の版と日付と時間の行だけが違います。行列積の回数も同じです。CUDA 版の全エネルギーと 18 スレッドの時間は、利用者の計算機で確かめました（実機の 5 番目の段階。11 節）。2 つの版の全エネルギーの差は最大 3.6e-11 Hartree で、18 スレッドの交換行列の段は前の版の 0.80 倍から 0.91 倍です。上の表の 4 つの条件を満たしています。

配列の確保は、2 番目の段階で変えていません。大きい配列を実行の間ずっと残す測定用の版で、効果の上限を測りました。4 スレッドの構築 1 回は、Si で 0.85 倍と 0.93 倍、s2p2d1 のダイヤモンドで 0.98 倍でした（12 節）。同じ版の実行の間の幅と同じ程度なので、使い回しは加えないことにしました（14 節）。

3 番目の段階の準備として、RTX 5080 で、デバイスの上の積を形ごとに測りました（`tools/exx_device_shapes.c`、`ca7925b`。11 節）。配列をデバイスに置いたままにすると、中心の原子 1 個の積の合計は、Si で 0.12 秒、s2p2d1 のダイヤモンドで 0.59 秒でした。中心の原子 2 個分は、18 スレッドのホストの構築 1 回の約半分の時間です（計算値）。3 番目の段階は、この形を FP64 で設計しました。

3 番目の段階は、改訂 17 で実装しました。コミットは `ae5c54e`（`modules/exx`）、`2e2a6da`（本体）、`09e054a`（試験の道具）です。CUDA 版は、`modules/exx` の呼び出し 1 回の中心の原子を、全てデバイスで計算します。 $D$ 、 $K$ 、全ての原子の T ブロックは、呼び出しの間デバイスに置いたままです。実装は 6.4 節、7.2 節、7.4 節、8 節、11 節にあります。CPU 版と、デバイスが引き受けない呼び出しは、前の版と同じ関数を同じ順に呼びます。

この段階では、デバイスが計算する呼び出しの行列積の回数が、設計で変わります。中心の原子 1 個につき 2 回と、V 近傍 1 個につき 5 回です。そこで、上の表の 3 つ目の条件を、この段階では次のように読み替えます（作業者の判断）。CPU 版と、ホストに残った呼び出しの行列積の回数は、前の版と変わらないこと。デバイスの呼び出しの数と、その中の大きい積の数は、設計の数と合うこと。後の条件は、`tools/exx_gpu_checks.py` が確かめます（11 節）。

クラウドのコンテナでは、GPU のライブラリーの模型で部品を試験しました（11 節）。GitHub Actions の GPU のコンパイルでは、NVHPC 26.9 の `nvc` が OpenACC のループを `-acc -gpu=nofma` でコンパイルしました。デバイスでの実行と時間は、利用者の計算機で確かめました（実機の 6 番目の段階、`09e054a`、2026-11-25。11 節）。

- 18 スレッドと 1 スレッドの 36 件の対は、全て収束しました。2 つの版の全エネルギーの差は、最大 1.9e-11 Hartree です。1 スレッドの CPU 版の `.out` は、前の版と日付と時間の行だけが違います。CPU 版とホストに残す実行の行列積の回数は、前の版と同じです。
- 受け入れ表の 53 件（1 スレッド）も全て収束し、道具の `table` の終了コードは 0 でした。CUDA 版と CPU 版の全エネルギーの差は、最大 1.6e-9 Hartree です（`o2`。閾値は 1e-7 Hartree）。デバイスを使う 4 つの系では、構築の呼び出しが全てデバイスで計算され、呼び出しの数と大きい積の数は設計の数と合いました。
- デバイスで計算する実行の交換行列の段は、18 スレッドで前の版の 0.59 倍から 0.63 倍、1 スレッドで 0.14 倍と 0.16 倍になりました。18 スレッドでも、同じ CUDA 版でホストに残す実行より 18% から 21% 短い値です（計算値）。

上の表の 4 つの条件（3 つ目は読み替えた形）を満たしたので、3 番目の段階を終えました。構築の高速化の 3 つの段階は、これで全て終わりです。次は M7 を再開します。デバイスの既定の動作は変えていません（作業者の判断。14 節）。呼び出しの全体の形は、18 スレッドのホストより短い時間で構築を終えるので、既定を変える理由がなくなったためです。

### M7. 力

| 項目 | 内容 |
|---|---|
| 目標 | 厳密交換による力を計算します。 |
| 前提 | M5 |
| 操作 | 式 (16)、(17) を site の形式に書き換えます。次に、二中心積分と三中心積分の、原子の位置についての微分を実装します。最後の段階は、式 (18) の連鎖律を対ごとに適用する処理です。 |
| 成功の確認方法 | 密度行列を固定した $E_x$ の解析微分が、原子を動かした中心差分と相対 1e-6 以内で一致します。SCF を回した全エネルギーの数値微分と解析的な力の差が、同じ構造と同じグリッドの GGA-PBE の差と、1e-5 Hartree/bohr 以内で合います（利用者の決定。改訂 20）。厳密交換による力の総和が 0 になります。 |
| 失敗例 | 外側反復の収束が甘いと、Pulay 項が前の世代の $K$ の固有値を使うため、力に $\delta$ 程度の誤差が残ります。 |

M7 の実装は、`modules/exx` の 4 つのコミット（`0d5ad4b`、`7d7d0a9`、`46951a9`、`da77f7c`）と、本体への組み込みからなります。モジュールの側は 4 段階です。二中心積分の変位についての微分、式 (18) の対ごとの連鎖律、式 (16) と (17) の中心の原子ごとの形、原子の位置についての $E_x$ の微分の順に加えました。条件の 1 番目と 3 番目は、モジュールの試験 `exx-exchange-forces` で確かめています。全ての原子を全ての座標軸に動かし、構造を準備し直した $E_x$ の中心差分と比べました。差は最大の成分の 5.4e-11 倍以内で、原子についての和は最大の成分の 5.9e-16 倍以下です。本体の側では、結合層の `Exact_Exchange_Gradient` と `Force` の引数を加えました（`f68c8b2`。5.9 節、7.4 節）。入力の拒否は、応力が要る `MD.Type` だけに狭めました（`c43bab9`。2.2 節）。試験の道具は `4a8b06b` です。

条件の 2 番目は、`tools/exx_force_checks.py` で確かめます。使い方は、進捗ファイルの「ビルドと実行」にあります。道具は、構造と、原子 1 個の座標の 1 成分を $\pm h$（ $h$ = 0.0005 Å）動かした 2 つの構造を、SCF で計算します。そして、表示された力と、全エネルギーの中心差分 $-(U(+h)-U(-h))/(2h)$ を比べます。動かした構造は、元の構造と同じグリッドで計算します。道具は元の構造を先に計算し、その原点（標準出力の `Grid_Origin`）を、動かした構造の `scf.fixed.grid` に渡します。本体の力の試験（`-forcetest`）と、`MD.Type` の 2 段目以降も、最初の構造のグリッドを使います。系は 3 つで、系ごとに 3 つの成分を動かします。

| 系 | 内容 |
|---|---|
| `ch4` | 対称な形から原子をずらしたメタン。C5.0-s1p1 と H5.0-s1、10 Å の立方体の箱、Cluster、600 Ry |
| `ch3` | 曲げたメチルラジカル。スピン分極、Cluster、600 Ry |
| `dia` | 格子定数 3.62 Å のダイヤモンドの基本単位胞で、2 番目の原子を動かした構造。C5.0-s1p1、Band、2×2×2 の k 点、40×40×40 のグリッド（原点は 0） |

`ch4` と `ch3` は、300 Ry でも同じ比較をして記録します（`ch4c300` と `ch3c300`）。`dia` の格子定数は、実験値の 3.567 Å から 1.5% 伸ばしてあります。どの構造でも、C サイトと V サイトの距離をリストの半径から 0.028 Å 以上離すためです。停止条件は、`scf.criterion` を 1e-12、`scf.exx.outer.criterion` を 1e-11 にしました。中心差分は、エネルギーの差を $2h$ = 0.0019 bohr で割るためです。GGA-PBE でも同じ比較をします。厳密交換を含まない力の差は、グリッド、刻み、SCF の停止が残す分です。この分は 600 Ry でも 1e-5 Hartree/bohr を超えることがあり、グリッドに対する原子の位置で変わります（13 節の 67）。そこで条件の 2 番目は、HSE06 の差から、同じ構造と同じグリッドの GGA-PBE の差を引いた値で判定し、限度を 1e-5 Hartree/bohr にしました。HSE06 の差そのものも記録します。この判定の方法は、利用者が 2026-11-25 に決めました（進捗ファイルの「決めたこと」）。道具は、ほかに 4 つの点を確かめます。動かした構造の近傍リストが、元の構造と同じこと。動かした構造のグリッドの原点が、元の構造と同じこと。厳密交換による力の和が、1e-12 Hartree/bohr 以下であること。2 プロセスの力が、1 プロセスの力と 1e-6 Hartree/bohr 以内で合うこと。最後の限度は、条件の 2 番目の限度の 10 分の 1 です。原子をプロセスに分ける処理を誤ると、力が厳密交換の分（この 3 つの系で 1e-2 程度）だけ変わるので、この限度で見分けられます。

最初の形の道具は、原点を原子と一緒に動かし、 $h$ を 0.002 Å、カットオフを 300 Ry、`dia` を格子定数 3.567 Å の 32×32×32 のグリッドにしていました。この形では、メタンの HSE06 の差が 7.5e-6 Hartree/bohr 以下でした。メチルラジカルでは、HSE06 で 1.7e-5、GGA-PBE で 2.0e-5 まで、`dia` の 1 成分では −8.1e-4 になりました。調べると、原点の動き、 $h^2$ に比例する誤差、近傍リストの境界の 3 つが比較に入っていました（13 節の 66）。原点を固めて $h$ を小さくすると、GGA-PBE の差自身が 300 Ry で 6.4e-5 になります（13 節の 67）。そこで道具を上の形に作り直し、カットオフを 600 Ry に上げました。

道具の 75 件を、2026-11-25 にクラウドのコンテナの CPU 版で実行しました（1 スレッド、3 件ずつ、1 時間 51 分）。全ての実行が収束し、`table` の終了コードは 0 でした。条件の 2 番目の 9 つの成分の結果です（Hartree/bohr）。「差」は、表示された力から全エネルギーの中心差分を引いた値です。

| 系 | 成分 | HSE06 の力 | HSE06 の差 | GGA-PBE の差 | 差の違い（限度 1e-5） |
|---|---|---|---|---|---|
| `ch4` | 原子 5 の x | +0.059178 | −4.12e-6 | −4.13e-6 | +9.3e-9 |
| `ch4` | 原子 1 の y | +0.030193 | −9.53e-6 | −9.69e-6 | +1.6e-7 |
| `ch4` | 原子 3 の z | −0.052143 | −1.40e-5 | −1.40e-5 | +2.0e-8 |
| `ch3` | 原子 1 の z | +0.021151 | +8.68e-6 | +8.78e-6 | −9.6e-8 |
| `ch3` | 原子 2 の x | +0.086265 | −5.62e-6 | −5.56e-6 | −6.3e-8 |
| `ch3` | 原子 3 の y | +0.067302 | −1.05e-5 | −1.06e-5 | +7.2e-8 |
| `dia` | 原子 2 の x | −0.027305 | −1.06e-5 | −1.07e-5 | +3.7e-8 |
| `dia` | 原子 2 の y | +0.040135 | −4.77e-6 | −4.49e-6 | −2.7e-7 |
| `dia` | 原子 1 の z | +0.015069 | +1.65e-5 | +1.65e-5 | −4.0e-8 |

差の違いは最大 2.7e-7 Hartree/bohr で、限度の 37 分の 1 以下です（計算値）。HSE06 の差そのものは、9 成分のうち 4 成分で 1e-5 を超えました（最大 1.65e-5）。同じ 4 成分では、GGA-PBE の差も 1e-5 を超えています。300 Ry の記録（`ch4c300` と `ch3c300`）では、差が最大 6.4e-5（HSE06 と GGA-PBE のどちらも）で、差の違いは最大 4.0e-7 でした。600 Ry のメチルラジカルで、グリッドの原点を z に間隔の半分ずらした別の実行では、C の z の差が HSE06 で 2.64e-5、GGA-PBE で 2.65e-5 でした（13 節の 67）。HSE06 の差は、カットオフと原点が変わっても、GGA-PBE の差と同じように動きます。

ほかの点の結果です。厳密交換による力の和は、HSE06 の 39 件の実行で 1.4e-17 Hartree/bohr 以下でした（限度 1e-12）。厳密交換による力の最大の成分は、`ch4` で 3.5e-2、`ch3` で 4.4e-2、`dia` で 1.0e-2 です。2 プロセスの力は、1 プロセスの力と、`ch4` と `ch3` でビットまで同じ、`dia` で 1e-12 以内でした（限度 1e-6）。動かした構造の近傍リストとグリッドの原点は、全て元の構造と同じです。既定の停止条件（`ch4-hse-default`）の力は、厳しい停止条件の力と最大 2.6e-8 違いました（GGA-PBE は 1.6e-10）。全ての原子の力の和は、600 Ry の 3 つの系で、HSE06 が最大 9.4e-6、GGA-PBE が最大 8.4e-6 です。グリッドが並進の不変性を破るためと考えられます。300 Ry では、HSE06 が最大 6.5e-5、GGA-PBE が最大 1.3e-5 でした。

厳密交換の力の計算（`Exact_Exchange_Gradient`）は、1 プロセスで、`ch4` と `ch3` が 0.11 秒から 0.29 秒、`dia` が 1.7 秒から 1.8 秒でした。2 プロセスでは、それぞれ 0.02 秒、0.04 秒、0.31 秒です。HSE06 の元の構造の実行は、`ch4` が 572 秒、`ch3` が 788 秒、`dia` が 110 秒でした。


`MD.Type` の短い実行の結果です。どれも HSE06 で、CPU 版の 1 スレッドです。

| 実行 | 内容 | 結果 |
|---|---|---|
| `NEB` | メタンの H を 1 個 0.2 Å 動かした終点と、像 2 個。120 Ry。1 プロセスと 2 プロセス、反復は 1 回と 2 回 | 像のエネルギーが、同じ座標の `Nomd` の計算と 1e-12 Hartree 以内で同じ |
| `EvsLC`、`DF` | ダイヤモンド | 2 番目の点のエネルギーが、セルを伸ばした `Nomd` の計算と 1e-12 Hartree 以内で同じ |
| `Opt`、`BFGS` | メタン、120 Ry、3 段 | エネルギーが下がり、力の最大の成分が 0.072 から 0.018 Hartree/bohr（`Opt`）、0.059 から 0.0099 Hartree/bohr（`BFGS`）に減った |
| `NVE` | メタン、120 Ry、0.5 fs を 3 段 | 全エネルギーと原子核の運動エネルギーの和が 1.5e-4 Hartree 変わった。GGA-PBE の同じ実行は 1.6e-4 Hartree |
| `NVT_NH` | スピン分極のメチルラジカル、120 Ry、3 段 | 最後まで実行した |
| `OptC1` | ダイヤモンド | 入力の時点で拒否した（2.2 節） |

本体の翻訳単位を切り出す ctest の 114 件（CPU 版のビルドの #1 から #70 と #589 から #632）は、全て合格しました。`tests/run_force_lifetime_cpu.py` は、`Force` の新しい引数を渡します。大域状態の検査は、`DFT.c`、`Force.c`、`Input_std.c`、`Exact_Exchange.c` を `openmx_globals.c` と一緒に、変更の前後で比べました（clang 18.1.3）。OpenMP の領域を外に出した関数の番号を除くと、報告は同じです。

利用者が決めた判定（差の違い）で M7 の 3 つの条件を満たしたので、M7 を終えました（2026-11-25）。次は M8 です。最初に着手する項目は、利用者が PBE0 に決めました（下の M8）。PBE0 の途中で、利用者の指示による MPI 並列での高速化を先に行いました（次の項）。

### MPI 並列での高速化（M7 と M8 の間）

利用者が 2026-11-25 に指示した作業です。指示は次の文です。「スレッド高速化しているようですが、OpenMXは基本的にはMPI並列で、ハイブリッド並列で場合によってはさらに速くなる構造です。MPI並列でも速くなるようにして下さい。」構築の高速化（上の項）が扱ったのは、スレッドとデバイスだけでした。M8 の PBE0 は、本体が入力を受け付けるコミット（`c9e26e7`）で止めて、この作業を先に行いました。

| 項目 | 内容 |
|---|---|
| 目標 | プロセスを増やすと、厳密交換の準備、構築、力が短くなる形にします。 |
| 前提 | M7 |
| 操作 | プロセスの数を変えて前の版の時間を測り、短くならない箇所を調べます。次に、その箇所をプロセスに分けます（6.7 節）。`modules/exx` は MPI を使わない形のままにします。 |
| 成功の確認方法 | 1 プロセスの全エネルギーが、前の版と全ての桁で同じです。プロセスの数を変えた全エネルギーは、1e-7 Hartree 以内で合います（M6 の閾値）。力は、1e-6 Hartree/bohr 以内で合います（M7 の道具の限度）。ハイブリッド汎関数でない計算の `.out` は、前の版と日付と時間の行だけが違います。準備、構築、力の時間を、プロセスの数ごとに、続けて測った前の版と比べて記録します。この 5 つの条件は、作業者が決めました。 |
| 失敗例 | 原子で配ると、原子の数より多いプロセスが働きません（13 節の 29）。力の重みを原子の持ち主が計算する形では、力が原子の数までしか短くなりません（13 節の 68）。時間を測る実行を並べると、同じコアに束縛されます（13 節の 69）。 |

前の版（`c9e26e7`）の時間を、クラウドのコンテナ（4 コア）で、1、2、4 プロセスで測りました。短くならない箇所は 3 つでした。

- 構築は、中心の原子をプロセスに配ります。Si の基本単位胞（2 原子）では、2 プロセスと 4 プロセスの構築が同じ時間でした（12 節の表）。
- 準備は、全てのプロセスが構造の全体について同じ計算をします。16 原子の Si では、1、2、4 プロセスのどれでも 5.1 秒から 5.4 秒でした（12 節の表）。
- 力も中心の原子で配ります。ベンゼン（12 原子）では、原子の大きさが違うので、受け持ちがそろいません。

実装は 4 つの部分です（6.7 節）。コミットは、`a1ee5ba`（`modules/exx` の入口と試験）、`a468823`（結合層）、`b265615`（`tools/exx_gpu_checks.py` の説明）です。

1. 構築を、ABF の範囲で配ります。範囲は、乗算の回数の見積もりで決めます。
2. 準備の T ブロックと V ブロックを、site の対で配ります。
3. 力の対の段階を、対を 1 回ずつ回る形に変え、対で配ります。
4. 力の重みを、構築と同じ ABF の範囲で配ります。最初の形は、重みを原子の持ち主が計算していました。Si の k 点の計算で測ると、重みが力の時間の 9 割を占めていたので、直しました（13 節の 68）。

モジュールの試験は、次の点を確かめます。ctest の 25 件は、GCC 13.3.0 のビルドと、Clang 18.1.3 にサニタイザーを付けたビルドで合格しました。サニタイザーは、アドレスと未定義動作の 2 つです。スレッドを作る 3 件（`ctest -L threads`）は、ThreadSanitizer のビルドで 3 回実行し、報告はありませんでした。

- 分けた準備（`exx-system-blocks`、`exx-exchange-system`、`exx-c-api`）。部分の数を 1、2、3、5、7 と、対の数より多い値にして、ブロックの和が分けない準備と同じ値になること。0 でない要素を持つ部分が、要素ごとに 1 つだけであること。式 (7) の報告の和が、分けない準備の報告と同じであること。和を入れる前の状態で、構築と微分が失敗を返すこと。
- ABF の範囲の構築（`exx-exchange-device`、`exx-exchange-device-system`、`exx-exchange-system`、`exx-c-api`）。原子の全体を含む範囲が、原子の範囲の関数とビットまで同じこと。原子の途中で切った範囲の和が、切らない計算と相対 1e-12 以内で合うこと。実測の差は、 $K$ の最大の要素の 3.3e-15 倍以下でした。模型のデバイスが、同じ範囲を受け取ること。
- ABF の範囲の重み（`exx-exchange-forces`、`exx-c-api`）。全ての ABF を 1 つの範囲にした重みが、中心の原子ごとの重みとビットまで同じこと。範囲を 2 つ以上に切った重みの和が、相対 1e-12 以内で合うこと。実測の差は 0 でした。範囲が触れない原子の配列を読まないこと。
- 対の経路（`exx-exchange-forces`、`exx-c-api`）。部分の数を 1、2、3、7 にして、微分の和が中心の原子ごとの経路と合うこと。中心差分との比較は、M7 と同じ限度です。実測の差は、中心の原子ごとの経路とは最大の成分の 1.2e-14 倍、中心差分とは 5.4e-11 倍以下でした。

本体では、前の版と新しい版を、同じ入力で 1 件ずつ続けて実行しました（12 節の表）。条件ごとの結果です。

| 条件 | 結果 |
|---|---|
| 1 プロセスの全エネルギーが、前の版と全ての桁で同じ | 7 つの入力の全てで同じでした（`.out` の `Utot.` の 12 桁）。1 プロセス 2 スレッドの Si でも同じです。 |
| プロセスの数を変えた全エネルギーが 1e-7 Hartree 以内 | 前の版の 1 プロセスとの差は、最大 7.7e-11 Hartree でした（Si の 16 原子）。前の版の同じ実行も、同じ差を示します。 |
| 力が 1e-6 Hartree/bohr 以内 | 前の版の 1 プロセスとの差は、最大 6.2e-11 Hartree/bohr でした（ベンゼンの 3 プロセス）。前の版どうしの差は、最大 8.0e-11 です。 |
| ハイブリッド汎関数でない計算の `.out` | GGA-PBE のベンゼンは、1 プロセスと 2 プロセスのどちらも、日付と時間の行だけが違いました。 |
| 時間の記録 | 12 節の表にあります。 |

- 44 件の実行は全て収束し、構築の回数は前の版と同じでした。入力は、HSE06 のベンゼン、Si の基本単位胞（4×4×4 の k 点）、Si の 16 原子（Γ 点）、スピン分極のメチルラジカル、PBE0 のメタンとダイヤモンド（2×2×2 の k 点）、GGA-PBE のベンゼンです。
- メチルラジカル（2 プロセスと 4 プロセス）とメタン（2 プロセス）では、プロセス 0 の範囲が C の原子の途中で終わります。C の ABF は 10 個で、プロセス 0 の範囲は、2 プロセスで 1 から 6、4 プロセスで 1 から 3 でした。
- 1 プロセスの力も、前の版と丸めの範囲で違います。新しい版は、対を 1 回ずつ回る形で和を取るためです（6.7 節）。差は、ベンゼンで 4.5e-11 Hartree/bohr でした。
- Si の基本単位胞の 1 プロセスでは、新しい版の `.out` に SCF の履歴が 2 つ入りました。前に止めた実行の一時ファイルが残っていたためです（13 節の 71）。残った履歴を除くと、力の最後の桁と時間の行だけが違います。
- 1 プロセス 2 スレッドの Si の `.out` は、SCF の 3 ステップ目から、残差の最後の桁が前の版と違いました。最初の構築より前のステップで、厳密交換はまだ計算に入っていません。全エネルギーは同じでした。

本体の翻訳単位を切り出す ctest の 114 件（CPU 版のビルドの #1 から #70 と #589 から #632）は、全て合格しました。大域状態の検査は、`Exact_Exchange.c` を `openmx_globals.c` と一緒に、変更の前後で比べました。検査の 7 つの一覧（`added`）は、どちらの報告でも空です。共通ヘッダーの大域変数を読み書きする関数も、前後とも 0 個でした。報告の違いは 2 点です。関数は、`ExactExchange_Centres` を外して `ExactExchange_Abfs` と `ExactExchange_Add_Up_Blocks` を加えたので、47 個から 48 個になりました。大域の名前の一覧には、3 つの名前が増えました。Open MPI の `MPI_LONG` と `MPI_MIN` の実体（`ompi_mpi_long`、`ompi_mpi_op_min`）と、局所変数 `report` の初期値として Clang が作る定数です（clang 18.1.3）。

5 つの条件を満たしたので、この作業を終えました（2026-11-25）。次は、止めていた M8 の PBE0 の試験です（下の M8）。4 プロセスを超える時間は、まだ測っていません（14 節）。

CUDA 版で原子の途中で切れた範囲は、M8 の PBE0 の確認と同じ機会に、利用者の計算機で確かめました（2026-11-25、`9e01d40`）。`tools/exx_gpu_checks.py` に、4 プロセスの Si の系（`siz-p444-np4` と `siz-p444-pbe0-np4`）を加えています。4 つのプロセスは、2 つの原子の 138 個の ABF を、原子の約半分ずつ受け持ちます。プロセス 0 の範囲は、HSE06 と PBE0 のどちらも 1 から 34 です。HSE06 では、10 回の構築の 40 回の呼び出しが、4 プロセスとも全てデバイスで計算されました。大きい積は、設計どおり 80 個です。全エネルギーは、CPU 版と表示の全ての桁が同じでした。PBE0 の `siz-p444-pbe0-np4` でも、11 回の構築の 44 回の呼び出しが全てデバイスで計算され、大きい積は設計どおり 88 個でした。CPU 版との差は 1e-12 Hartree です。交換行列の段は、HSE06 で CPU 版の 7.83 秒に対して 4.00 秒です。1 プロセスの 3.27 秒より長い値でした（11 節）。

### M8. 後続の項目

次の項目を、項目ごとに M4 から M7 と同じ形の試験で確認します。着手の順序は利用者が決めます。

- PBE0（切断クーロン）
- 近傍リストの形式で $H$ を読む出力。任意の k 点の $K(\mathbf k)$ は式 (14) で得られます。バンド分散と DOS は改訂 30、MO 出力は改訂 32 で入れました（6.5 節）。
- ノンコリニア
- HIP
- 応力
- GPU での XC の計算。改訂 25 で、CUDA 版に実装しました（この節の「力のデバイス化とほかの高速化」）。

利用者は 2026-11-25 に、最初に着手する項目を PBE0（切断クーロン）に決めました。PBE0 の V 近傍の範囲は、切断半径から自動で決める形を選んでいます（進捗ファイルの「決めたこと」）。PBE0 のコミットは次の 8 つです。

- `7b77629`。`modules/xc` に、PBE の交換を $1-\alpha$ 倍する関数を加えました（`pbe_global_hybrid`、`openmx_xc_pbe_global_hybrid`）。試験 `xc-hybrid` は 23,520 件です。
- `c9e26e7`。本体が `scf.XcType PBE0` を受け付けます。CPU 版と CUDA 版が対象で、HIP 版は拒否します。カーネルは $\theta(R_c-r)/r$ です。 $R_c$ は `scf.exx.coulomb.cutoff`（bohr）の値です。書かないか 0 のときは、 $(3N_k\Omega/4\pi)^{1/3}$ を準備のたびに計算します。`scf.exx.v.cutoff` を書かない PBE0 は、V 近傍を「2 原子の ABF の半径の大きい方 + $R_c$ + 構造の中で最大の ABF の半径」まで取ります。書いたときは、その値を使います。HSE06 の既定は 8 bohr のままです。
- `a3da659`。`modules/exx` に試験 `exx-two-center-truncated` を加えました。切断クーロンの二中心積分が到達範囲の外で 0 になることと、 $\kappa$ の刻みが足りる範囲を確かめます（13 節の 19）。
- `b8cbe29`。PBE0 の $\kappa$ の区間の数を、 $R_c$ と ABF の半径から決めます（5.6 節）。`1DFFT.NumGridR` で足りる計算では、区間の数も結果も変わりません。区間を増やす形は、利用者が 2026-11-25 に決めました。
- `6b53e6b`。M4、M5、M7 の試験の道具に、PBE0 の実行を加えました。使い方は、進捗ファイルの「ビルドと実行」にあります。
- `9e01d40`。M6 の試験の道具（`tools/exx_gpu_checks.py`）に、PBE0 の 13 系を加えました。HSE06 と PBE0 の両方に、4 プロセスが Si の原子を約半分ずつ受け持つ系も加えています。
- `3d810d2`。既定の値より大きい $R_c$ を書いた入力を、本体が拒否します（この節の最後の段落）。
- `6212ade`。SCF と k 点の試験の道具に、拒否を確かめる実行を加えました。

閾値と限度は、M4 から M7 と同じです。PBE0 は、4 つのマイルストーンの条件を満たしました。

| 条件 | 試験 | 結果 |
|---|---|---|
| M4 の表 | `tools/exx_scf_checks.py` の PBE0 の 98 件 | 98 件が収束し、閾値のある行は全て閾値の内でした。 |
| M5 の条件 | `tools/exx_kpoint_checks.py` の PBE0 の 8 組 | 原子あたりの全エネルギーの差は、最大 4.6e-11 Hartree でした（閾値 1e-7）。 $E_x$ の差は最大 5.7e-11 Hartree です（閾値 1e-9）。 $K(\mathbf k)$ と転置の差は、最大 2.5e-16 でした（限度 1e-12）。 |
| M6 の条件 | `tools/exx_gpu_checks.py` の PBE0 の 57 件（利用者の計算機） | CUDA 版と CPU 版の全エネルギーの差は、最大 3.0e-11 Hartree でした（閾値 1e-7）。容量が足りない場合の実行（`NAME-host`）では、全ての呼び出しがホストに残りました。CPU 版との差は、2.5e-11 Hartree 以下です。段ごとの時間は、進捗ファイルにあります。 |
| M7 の条件の 1 番目 | `exx-exchange-forces` の大きいセルの場合（ $R_c$ は 12 bohr） | M7 から試験に入っています。中心差分との差は、最大の成分の 5.5e-12 倍以下です。 |
| M7 の条件の 2 番目 | `tools/exx_force_checks.py` の PBE0 の 9 成分 | PBE0 の差から GGA-PBE の差を引いた値は、最大 3.2e-7 Hartree/bohr でした（限度 1e-5）。 |
| M7 の条件の 3 番目 | 同じ道具の PBE0 の 24 件 | 厳密交換による力の和は、3.7e-17 Hartree/bohr 以下でした（限度 1e-12）。 |

3 つの道具は、PBE0 の実行の停止条件を厳しくしています。SCF の道具では、`scf.criterion` が 1e-12、`scf.exx.outer.criterion` が 1e-9 です。HSE06 と同じ 1e-10 と 1e-6 では、メタンの 2 つの実行が合いませんでした。1 プロセスと 2 プロセスの全エネルギーが 1.2e-9 Hartree 違い、M4 の閾値（1e-9）を超えます。最後の内側の SCF が、2 つの実行で違うステップで止まったためです。k 点の道具は、`scf.criterion` を基本単位胞あたり 1e-12 にしました。外側反復の閾値は、HSE06 と同じ基本単位胞あたり 1e-9 です。力の道具は、M7 と同じ 1e-12 と 1e-11 を使います。混合比 0 の実行は、同じ停止条件の GGA-PBE の実行と比べます。

**M4 の形。** 系は、M4 の表の HSE06 の実行と同じ 8 つです。分子の箱は 1 辺 10 Å で、 $R_c$ は 11.72 bohr になります。ダイヤモンドの 8 原子のセル（Γ 点だけ）では 4.18 bohr でした。外側反復は、8 回から 10 回の構築で閾値を下回りました。HOMO と LUMO の差は、GGA-PBE より 2.5 eV から 3.2 eV 大きくなります（O₂ を除く 6 つの系。計算値）。系ごとの値は、進捗ファイルの表にあります。閾値のある行の最大の差は、次のとおりです（Hartree）。

| 行 | 閾値 | 最大の差 |
|---|---|---|
| 混合比 0 と GGA-PBE | 1e-9 | 1.0e-12 |
| 並進（全エネルギーと $E_x$ ） | 1e-8 | 2.0e-12 |
| スピン分極 `on` と `off` | 1e-8 | 2.0e-12 |
| 2 プロセスと 4 プロセス | 1e-9 | 1.8e-11 |
| `scf.exx.v.cutoff` を 24 bohr にした実行と自動の範囲 | 1e-9 | 2.0e-12 |
| 密度行列の照合 | 1e-10 | 2.2e-16 |

24 bohr の行は、HSE06 の表にない比較です。PBE0 の自動の範囲は、カーネルが届く範囲の全体です。範囲の外で足すブロックは 0 なので、全エネルギーは変わらないはずです。記録の行では、次の 4 点が分かりました。

- V 近傍を自動の範囲より短い 8 bohr か 12 bohr にすると、分子の全エネルギーが変わりました。変化は 1.1e-6 から 2.0e-5 Hartree です。PBE0 には、自動の範囲が要ります。
- 1 辺 20 Å の箱では、 $R_c$ が 23.45 bohr になります。 $E_x$ は、10 Å の箱と 3.2e-6 から 3.7e-5 Hartree 違いました。10 Å の箱では、V 近傍が分子の周期像に届きます。
- $R_c$ を入力で 16 bohr にすると、分子の全エネルギーが 3.0e-3 から 5.9e-3 Hartree 下がりました。既定の 11.72 bohr より大きい値です（13 節の 72）。8 bohr に縮めた変化は、8.5e-6 から 4.6e-5 Hartree でした。
- `scf.exx.abf.pca` の影響は、s2p2d1 の分子で HSE06 より小さく出ました。1e-6 と既定値（1e-4）の差は、メタンで −2.0e-7、水で +4.6e-7 Hartree です。HSE06 では、−1.5e-5 と −2.1e-5 Hartree でした。LRI の係数をクーロン計量で決めているためと考えられます。s2p2d1 のダイヤモンドの 8 原子は、HSE06 と同じく大きく変わります（1e-3 で −2.6e-3 Hartree）。

**M5 の形。** 組は 8 つです。5 組は、ダイヤモンドの基本単位胞（C5.0-s1p1）です。メッシュの 2×2×2、3×3×3、3×2×1 は、同じ大きさのスーパーセルの Γ 点と比べました。4×4×4 は、2×2×2 のスーパーセルの 2×2×2 のメッシュと比べ、`REGULAR` のメッシュでも同じ組を作りました。残りの 3 組は、水素の鎖の 2 つ（4×1×1）と、C6.0-s2p2d1 のダイヤモンドの 2×2×2 です。鎖の 1 つはスピン分極があり、もう 1 つはありません。 $R_c$ は、組の 2 つの計算で表示の 6 桁まで同じでした。値は 4.79 bohr から 10.54 bohr です。記録の行では、2×2×2 から 8×8×8 までメッシュを細かくすると、全エネルギーが 0.121 Hartree 下がりました。6×6×6 と 8×8×8 の差は −2.4e-4 Hartree です（ $R_c$ は 15.81 bohr と 21.07 bohr）。4×4×4 で V 近傍を 8 bohr に縮めると、全エネルギーが −9.8e-6 Hartree 変わります。12 bohr では −2.0e-8 Hartree でした（自動の範囲は 15.54 bohr）。

**M6 の形。** 系は、M6 の表の HSE06 の 13 系と同じ構造で、汎関数だけを PBE0 にしました（`9e01d40`）。実行の種類も HSE06 と同じで、57 件です。停止条件は、分子が SCF の道具、結晶が k 点の道具の PBE0 の実行と同じにしました。 $R_c$ と V 近傍は、自動の範囲です。V 近傍が HSE06 の 8 bohr より遠くまで届くので、構築の行列積が増えます。CPU 版で数えると、`siz-p444-pbe0` の構築 1 回は 470,350 個の積で、最大の積は 31057×897（内側の次元 1131）でした。HSE06 では、54,850 個と 8229×897 です。デバイスに送る大きい積は、HSE06 と同じく、中心の原子 1 個（またはその一部）ごとに 2 個です。実行は、2026-11-25 の 15:01 から 16:25（JST）に、利用者の計算機で行いました（RTX 5080、1 スレッド）。57 件は全て収束し、`table` の終了コードは 0 でした。判定のある 127 行に、不合格の行はありません。CUDA 版と CPU 版の全エネルギーの差は、最大 3.0e-11 Hartree です（`diaz-p222-pbe0-dense`）。厳密交換のエネルギーの差は、1e-12 Hartree 以下でした。デバイスを使う 5 つの系では、構築の呼び出しが全てデバイスで計算されました。呼び出しと大きい積の数は、設計の数と合っています。呼び出し 1 回のメモリーは、Si で 880.0 MB、s2p2d1 のダイヤモンドで 1246.9 MB でした。どちらも HSE06 より大きい値です。交換行列の段は、CPU 版から CUDA 版で、`siz-p444-pbe0` が 188.98 秒から 23.91 秒になりました。`diaz-p222-pbe0` では、237.50 秒から 28.47 秒です。GEMMul8 の実行は、FP64 の実行と 1e-12 Hartree 以内で合いました。CUDA 版の力は、CPU 版の力と 1.3e-11 Hartree/bohr 以内で合います（14 節の「CUDA 版の力」）。15:30 から 15:33 には、計算機で別の計算が CPU を使いました。重なった `siz-p444-pbe0-spin-gpu` の 1 件は、全エネルギーと判定の行が合格です。時間の記録には、重なったことを書いています（進捗ファイル）。

**M7 の形。** 系は、M7 の `ch4`、`ch3`、`dia` で、構造、グリッド、動かす成分も同じです。 $R_c$ は、箱の分子で 11.72 bohr、`dia`（格子定数 3.62 Å、2×2×2 の k 点）で 5.35 bohr です。V 近傍は自動の範囲で、「大きい方の ABF の半径 + $R_c$ + 5 bohr」まで届きます。どの構造でも、C サイトと V サイトの距離は、リストの半径から 0.026 Å 以上離れています。`ch4` では 0.08 Å、`ch3` では 0.16 Å 以上です。条件の 2 番目は、PBE0 の差から、同じ構造と同じグリッドの GGA-PBE の差を引いた値で判定しました。M7 で利用者が決めた方法（改訂 20）を、PBE0 に広げた形です。GGA-PBE の実行は、同じ道具でやり直しました。成分ごとの結果です（Hartree/bohr）。

| 系 | 成分 | PBE0 の力 | PBE0 の差 | GGA-PBE の差 | 差の違い（限度 1e-5） |
|---|---|---|---|---|---|
| `ch4` | 原子 5 の x | +0.059214 | −4.13e-6 | −4.13e-6 | −1.4e-9 |
| `ch4` | 原子 1 の y | +0.030193 | −9.52e-6 | −9.69e-6 | +1.8e-7 |
| `ch4` | 原子 3 の z | −0.052191 | −1.39e-5 | −1.40e-5 | +3.0e-8 |
| `ch3` | 原子 1 の z | +0.021135 | +8.67e-6 | +8.78e-6 | −1.0e-7 |
| `ch3` | 原子 2 の x | +0.086282 | −5.64e-6 | −5.56e-6 | −8.3e-8 |
| `ch3` | 原子 3 の y | +0.067316 | −1.05e-5 | −1.06e-5 | +7.0e-8 |
| `dia` | 原子 2 の x | −0.027384 | −1.06e-5 | −1.07e-5 | +4.5e-8 |
| `dia` | 原子 2 の y | +0.040260 | −4.81e-6 | −4.49e-6 | −3.2e-7 |
| `dia` | 原子 1 の z | +0.015098 | +1.65e-5 | +1.65e-5 | −5.7e-8 |

道具の 45 件は、2026-11-25 にクラウドのコンテナの CPU 版で実行しました。PBE0 が 24 件、GGA-PBE が 21 件です。実行は 1 スレッドで 2 件ずつ、ほかの 2 つの道具と並べて行い、30 分かかりました。全ての実行が収束し、`table` の終了コードは 0 でした。GGA-PBE の 21 件は、M7 の実行と、全エネルギー、力、SCF のステップ数の表示の全ての桁が同じです。差の違いは最大 3.2e-7 Hartree/bohr で、限度の 31 分の 1 以下でした（計算値）。PBE0 の差そのものは、HSE06 と同じ 4 成分で 1e-5 を超えています（最大 1.65e-5）。厳密交換による力の和は、PBE0 の 24 件で 3.7e-17 Hartree/bohr 以下でした（限度 1e-12）。厳密交換による力の最大の成分は、`ch4` で 3.9e-2、`ch3` で 4.8e-2、`dia` で 9.9e-3 です。2 プロセスの力は、1 プロセスの力と、`ch4` と `ch3` で表示の全ての桁が同じでした。`dia` では 1e-12 以内です（限度 1e-6）。動かした構造の近傍リストとグリッドの原点は、全て元の構造と同じでした。300 Ry の記録と既定の停止条件の実行は、PBE0 では行っていません。

**$\kappa$ の区間と、ハイブリッド汎関数でない計算。** `b8cbe29` の前と後の実行ファイルで、同じ入力を続けて実行しました。8 つの入力の `.out` は、日付の行だけが違います。PBE0 の入力は 5 つです。メタンは、1 辺 10 Å と 20 Å の箱の 2 つです。水は、 $R_c$ を 16 bohr にした入力と、V の範囲を 24 bohr にした入力です。5 つ目は、2×2×2 の k 点のダイヤモンドです。PBE0 でない 3 つは、HSE06 のメタンとダイヤモンド、GGA-PBE のメタンです。PBE0 の区間は、どれも 900 個のままでした。Si7.0-s1p1 の Si の 9×9×9 の k 点（ $R_c$ は 36.09 bohr）では、区間が 957 個になります。この入力は、区間が 900 個の前の版でも実行しました。全エネルギーの差は 5.1e-10 Hartree、 $E_x$ の差は 3.7e-10 Hartree です。構築の回数と V 近傍の site の数は同じです。本体の翻訳単位を切り出す ctest の 114 件は、合格しました。大域状態の検査は、`Exact_Exchange.c` を `openmx_globals.c` と一緒に、変更の前後で比べ、報告は同じでした。

**既定より大きい $R_c$ の拒否。** 利用者は 2026-11-25 の 20:42（JST）に、カードで拒否する形を選びました（14 節）。`3d810d2` から、本体は $(3N_k\Omega/4\pi)^{1/3}$ より大きい `scf.exx.coulomb.cutoff` を受け付けません。標準出力に、書いた値とこの半径を表示して終了します。1e-6 bohr までの超過を受け付けるのは、表示の 6 桁の丸めを許すためです。確かめるのは、入力の時点（`Input_std.c`）と準備のたびの 2 か所です。準備の時点では、厳密交換の層が計算を止めます。`MD.Type` の EvsLC と DF が、セルを縮めることがあるためです。受け付ける入力の計算は、前の版と全ての桁が同じでした。試験は `6212ade` で、SCF の道具の 16 bohr の 4 件を、拒否を確かめる行に変えました。既定の値を 6 桁で書いた 4 件と、EvsLC でセルを縮める 1 件も加えています。k 点の道具には、2×2×2 の k 点で拒否する 1 件を加えました。結果は、進捗ファイルにあります。

**モジュールの試験。** `exx-two-center-truncated` は 2 つの部分からなります。1 つ目は、ガウス関数の解析値との比較です。 $R_c$ は 8 bohr から 48 bohr、 $d$ は $R_c+24$ bohr までです。p 関数までの 9,984 件の差は、5.8e-15 以下でした。s 関数の微分の 1,872 件は 1.6e-15 以下です（限度はどちらも 1e-12）。2 つ目は、半径 6 bohr の関数を、区間 900 の格子と 3600 の格子で比べます。差は、細かい格子の最大のブロックに対する比です。条件の内側の 17,658 件は、3.1e-15 以下でした（限度 1e-13）。到達範囲の外の 5,022 件は 1.1e-16 以下です（限度 1e-14）。条件を外れた所の差（ $R_c$ が 36 bohr で 8.2e-3、40 bohr で 5.4e-2）が 1e-4 を超えることも確かめます。試験が、刻みの不足を見分けられることの確認です。`modules/exx` の ctest の 26 件は、GCC 13.3.0 のビルドで合格しました。Clang 18.1.3 にアドレスと未定義動作のサニタイザーを付けたビルドでも、外部の利用者の試験を除く 25 件が合格しています。ライブラリーのコードは変えていないので、ThreadSanitizer の実行は MPI 並列での高速化のときの結果のままです。

**バンド分散と DOS。** 利用者は 2026-11-25 の 20:42（JST）に、PBE0 の次の項目を「バンド分散や DOS などの出力」に決めました。この版では、Band ソルバーのバンド分散（`Band.dispersion`）と DOS（`Dos.fileout`、`DosGauss.fileout`）を入れています。設計は 6.5 節の「SCF の後の出力」、関数は 7.4 節にあります。コミットは `9f012f4`（本体）と `c52e41e`（試験の道具）です。

| 項目 | 内容 |
|---|---|
| 目標 | バンド分散と DOS の固有値に、厳密交換を含めます。 |
| 前提 | M5 と、6.4 節の V 近傍と列のまとめ |
| 操作 | SCF の後に、まとめない系の $K(\mathbf R)$ を 1 回作り、出力の k 点の $H(\mathbf k)$ に足します。 |
| 成功の確認方法 | `tools/exx_output_checks.py` の 21 件で、下の表の全ての行が限度の内にあることです。ハイブリッド汎関数でない計算の出力は、前の版とバイト列まで同じです。条件と限度は、作業者が決めました。 |
| 失敗例 | まとめた系の $K(\mathbf R)$ をそのまま使うと、メッシュの点の外の固有値が誤ります。スーパーセルの比較では、2.1e-4 Hartree の差でした（下の実験）。 |

道具の 21 件は、2026-11-26 にクラウドのコンテナの CPU 版で実行しました（1 スレッド）。全ての実行が収束し、`table` の終了コードは 0 でした。

| 確認 | 内容 | 最大の差 | 限度 |
|---|---|---|---|
| スーパーセル | スーパーセル（ダイヤモンドは 2×2×2、水素の鎖は 2×1×1）の点 $\mathbf K$ の固有値と、基本単位胞の $(\mathbf K+\mathbf m)/c$ の固有値の和集合を比べます。メッシュにない 4 点です。ダイヤモンドは GGA-PBE、HSE06、PBE0 と `REGULAR` の HSE06、水素の鎖はスピン分極のある `REGULAR` の HSE06 です。 | 2.0e-12 Hartree（GGA-PBE）。ハイブリッド汎関数では 1.5e-12 Hartree | 1e-6 Hartree |
| メッシュの点 | 3×3×3 のダイヤモンドで、メッシュの点を通る経路の固有値と、`.out` の最後の固有値を比べます。HSE06 と PBE0 の 12 点ずつです。 | 2.1e-10 Hartree | 1e-8 Hartree |
| メッシュの点の $K(\mathbf k)$ | 本体の比較（`level.of.stdout 2`）の 27 点の差を、最大の要素で割った値です。 | 5.4e-12（PBE0）、2.7e-12（HSE06） | 1e-9 |
| プロセス | 2 プロセスと 4 プロセスのバンドを、1 プロセスのバンドと比べます。 | 2.2e-12 Hartree | 1e-8 Hartree |
| DOS | `Dos.Kgrid` が 3×3×3（SCF は 4×4×4）の固有値を、同じ 27 点のバンドと比べます。`.Dos.val` は Hartree の 6 桁、`.FermiSurf0.bxsf` は eV の 5 桁です。 | 4.9e-7 Hartree、4.6e-6 eV | 2e-6 Hartree、2e-5 eV |
| Cluster ソルバーの DOS | `Dos.Kgrid` が 1 1 1 の `.Dos.val` を、`.out` の固有値と比べます。 | 4.6e-7 Hartree | 2e-6 Hartree |
| 拒否 | 2.2 節の 4 つの条件です。 | 4 件とも、表示に条件の文があります。 | なし |

スーパーセルの比較の 2 つの SCF は、同じ k 点を使います。差は収束した 2 つの SCF の差で、 $K$ のない GGA-PBE でも 2.0e-12 Hartree でした。メッシュの点の経路の差は、外側反復の 1 回分です（6.5 節）。DOS の行が確かめるのは、DOS のファイルがバンド分散と同じ $K(\mathbf k)$ を足すことです。メッシュの点の外での正しさは、スーパーセルの行が確かめます。四面体法の `.FermiSurf0.bxsf` は、化学ポテンシャルの近くのバンドだけを書きます（13 節の 76）。

**2 つの実験。** どちらも、一時的な実行ファイルで行いました（コミットしていません）。1 つ目は、`Band_DFT_Col.c` の `Shift_K_Point` を 0 にした版です。3×3×3 のダイヤモンドの $D$ の非対称性は、1.0e-11 から 7.9e-16 になりました。メッシュの点の $K(\mathbf k)$ の差は、HSE06 で 1.0e-12 から 1.3e-15 になりました。PBE0 では 2.3e-12 から 2.1e-15 です（最大の要素は 0.37 と 0.42）。限度の 1e-9 は、6.4 節の位相の上限 1.3e-10 を基に決めました。2 つ目は、出力にまとめた系の $K(\mathbf R)$ をそのまま使わせた版です。スーパーセルの比較の差は 2.1e-4 Hartree で、限度を超えました。メッシュの点の経路の差は、2.1e-10 Hartree のままでした。まとめた系でも、メッシュの点の $K(\mathbf k)$ は正しいためです。四面体法の DOS の 2 行は、この版でも合格しました。この版は、まとめない枝（`scf.Generation.Kpoint none` のための枝）の手順も通っています。

**ハイブリッド汎関数でない計算と、ほかの試験。** GGA-PBE の 5 つの入力を、`079d22c` の実行ファイルと比べました。ダイヤモンドの 4 つは、バンド分散、四面体法の DOS、ガウス関数の DOS、Cluster ソルバーの DOS です。5 つ目は、スピン分極のある水素の鎖のバンド分散と DOS です。`.out` 以外のファイルは、バイト列まで同じでした。`.out` は、日付と時間の行だけ違います。本体を切り出す ctest の 114 件（#1 から #70 と #589 から #632）は、合格しました。大域状態の検査が比べる項目に、増えたものはありません。新しい関数は、`stdout` と MPI の定数のほかに大域変数を読みません。`Input_std` は、新たに `Dos_Kgrid` を読みます。変えた 4 つのファイル（`Exact_Exchange.c`、`Band_DFT_kpath.c`、`Band_DFT_Dosout.c`、`DFT.c`）に、アドレスサニタイザーを付けた版も作りました。出力の 7 件の実行で、報告は MPI のライブラリー（Open MPI と hwloc）が解放しないメモリーだけです。出力を準備しない Cluster ソルバーの実行と、同じ報告でした。

**時間とメモリー。** 出力の準備と構築は、4×4×4 のダイヤモンドの HSE06 で 0.24 秒でした。2×2×2 のスーパーセルでは、HSE06 で 1.20 秒、PBE0 で 3.13 秒です。実行の全体（44.3 秒と 51.3 秒）の 2.7% と 6.1% に当たります（計算値）。まとめない系のブロックは、まとめた系の 1.2 倍から 2.3 倍でした。PBE0 のスーパーセルでは、スピン 1 成分あたり 5.6 MB から 12.5 MB になります。

**残りの作業。** CUDA 版では、出力の半分の構築も、SCF の構築と同じ関数表でデバイスに送ります。この経路は、まだ利用者の計算機で確かめていません（14 節）。近傍リストの形式で $H$ を読むほかの出力（`Unfolding.Electronic.Band` など）は、2.2 節の表に示します。

### 力のデバイス化とほかの高速化（M8 の項目の間）

利用者が 2026-11-25 に指示した 2 つの作業です。1 つ目は、20:43（JST）の「厳密交換の力をデバイスで計算する形にして下さい」です。2 つ目は、21:17（JST）の「他に可能な高速化でできることがあれば、すべてやってください。」です。M8 の次の項目（近傍リストの形式で $H$ を読む出力）は、この作業の後に行います。

| 項目 | 内容 |
|---|---|
| 目標 | 厳密交換を含む計算を速くします。結果は、丸めの範囲までしか変えません。 |
| 前提 | M8 の PBE0 |
| 操作 | 時間を測り、長い部分から短くします。先に考えるのは、MPI 並列で短くなる形です（利用者の指示）。結果を丸めの範囲より大きく変える方法と、キーワードが要る方法は、利用者に確認します（1.3 節）。 |
| 成功の確認方法 | ビットを変えない変更では、1 プロセス 1 スレッドの CPU 版の `.out` が、前の版と日付と時間の行だけ違います。丸めを変える変更では、全エネルギーが前の版と 1e-9 Hartree 以内、力が 1e-6 Hartree/bohr 以内で合います。CUDA 版の条件は、M6 の道具の判定に合格することです。時間は、続けて測った前の版と比べて記録します。この 4 つの条件は、作業者が決めました。 |
| 失敗例 | 球ベッセル関数の値が末尾の桁で変わると、局所 RI の Jacobi 法が収束しない行列が出ました（13 節の 73）。半分の構築では、原子の端の数個の ABF を受け持つプロセスが長くなりました（13 節の 74）。 |

コミットは 18 個です（文書のコミットは数えていません）。

| コミット | 内容 | 結果を変えるか |
|---|---|---|
| `d6f7db1`、`df30625`、`4577361` | 力の重みをデバイスで計算します（7.4 節、8 節）。`4577361` は M6 の道具です。 | 同じビット（CPU 版） |
| `f5d746d` | $K(\mathbf k)$ を、構築ごと、k 点ごとに 1 回だけ作ります（6.5 節）。 | ScaLAPACK の経路の丸め |
| `80b2e08`、`d768793` | 交換行列の半分の構築です（6.4 節）。 | 丸め |
| `1b5a063` | 局所 RI の Jacobi 法の回転を直しました（13 節の 73）。 | 分解に入る計算だけ |
| `27a952a` | M6 の道具が、力を CPU 版と比べます。原子を動かした Si の系を加えました。 | なし（道具） |
| `64352d9` | 球ベッセル関数を上向きの漸化式で計算します（13 節の 75）。 | 末尾の桁 |
| `85c1e7b` | 同じ 2 つのスピンの短距離交換を、1 回だけ計算します（7.5 節）。 | 同じビット |
| `1bc356f` | ハイブリッド汎関数の準局所部分を、C のヘッダーに移しました（7.5 節）。 | 同じビット |
| `3fafc96`、`f0421a6` | CUDA 版が、ハイブリッド汎関数の準局所部分を GPU で計算します（8 節）。`f0421a6` は CI の検査です。 | CPU 版は同じビット |
| `92f78c7` | ABF をプロセスに分ける見積もりを、時間の見積もりに変えました（6.7 節）。 | 2 プロセス以上の丸め |
| `2198f55` | 力の重みを、半分の構築の形で計算します（6.7 節）。 | 力の丸め |
| `7a79b99` | 準備の対と力の対を、`-nt` のスレッドに分けます（6.7 節）。 | 同じビット |
| `1626dc4` | 正則な k メッシュの計算で、V 近傍をメッシュの周期でまとめます（6.4 節）。 | 丸め |
| `51cd508` | V 近傍をまとめた計算で、構築の列もメッシュの周期でまとめます（6.4 節）。 | 丸め |
| `079d22c` | デバイスが呼び出しを引き受ける条件を、積の乗算の和で決めます（11 節）。 | CUDA 版の丸め（CPU 版は同じビット） |
| `6c9a592` | 構築の後の混合を 1 ステップ早く始め直します。 $K$ の外挿と、内側の SCF の緩い判定のキーワードを加えました（6.6 節）。 | SCF の判定の範囲（キーワードのない計算は、始め直しの分） |

**力の重み。** M7 の形は、力の重みの行列積をホストの BLAS だけで計算しました。M8 の PBE0 の実機の確認では、力の時間が、デバイスで計算した構築の段より長い値でした（14 節）。`d6f7db1` から、重みの 2 つの入り口が関数表のデバイスを使います。ホストでは、中心の原子の ABF のループを区間に分けます。重みには ABF についての和がないので、スレッドの数によらず同じビットです。モジュールの試験は、デバイスの模型とホストの重みを、両方の形式で相対 1e-12 以内で比べます。実測の差は 4e-16 でした。`4577361` からは、M6 の道具が重みの行を読み、`NAME-gpu` の重みがデバイスで計算されたことを確かめます。

**$K(\mathbf k)$ と半分の構築。** PBE0 の Si（4×4×4 の k 点）を、1 プロセス 1 スレッドで測りました。`f5d746d` で、加算が 19.7 秒から 1.0 秒になりました。`d768793` では、構築 1 回の行列積が 470,350 個から 235,454 個に減っています。11 回の構築の時間は、233.5 秒から 128.4 秒です。実行の全体は、319.8 秒から 219.6 秒になりました。どちらの版も、全エネルギーと $E_x$ の表示の 12 桁が前の版と同じです。2×2×2 の k 点のダイヤモンド（HSE06 と PBE0）は、`f5d746d` と 7e-12 Hartree 以内で合いました。

**Jacobi 法。** `1b5a063` の回転は、消す要素を 0 に置きます。2 つの対角要素は、 $a_{pp}-ta_{pq}$ と $a_{qq}+ta_{pq}$ です（Rutishauser の形）。OpenMX がこの分解に入るのは、Cholesky 分解のピボットが 1e-7 を下回るときだけです。このコンテナで実行したハイブリッド汎関数の 1167 件には、その場合がありませんでした。

**球ベッセル関数。** `64352d9` の値は、4 倍精度の値と比べました。 $x$ が 6000 までの誤差は、19 次までで包絡線 $1/x$ の 4.5e-15 倍以下です（29 次までは 7.3e-15 倍）。下向きの漸化式だけの誤差は、 $1/x$ の 6.7e-13 倍まで増えます。901 点の表（ $\kappa$ が 60 bohr⁻¹ まで、距離 20 bohr）は、2.1 ミリ秒から 0.035 ミリ秒になりました。1 プロセス 1 スレッドの時間です（秒、前と後）。

| 入力 | 準備 | 厳密交換の力 | 実行の全体 |
|---|---|---|---|
| ダイヤモンド 54 原子、PBE0 | 35.56、1.62 | 37.46、2.25 | 236.4、175.9 |
| ダイヤモンド 8 原子、PBE0（V 近傍 24 bohr） | 28.78、0.86 | 24.61、0.85 | 73.8、22.4 |
| メタン、PBE0 | 2.01、0.14 | 0.15、0.00 | 9.8、8.0 |
| ダイヤモンド、2×2×2 の k 点、PBE0 | 1.72、0.13 | 0.76、0.28 | 9.2、8.5 |
| ダイヤモンド、2×2×2 の k 点、HSE06 | 1.67、0.12 | 0.65、0.15 | 9.0、7.2 |
| O₂、HSE06、スピン分極 | 1.10、0.09 | 0.00、0.00 | 14.8、10.1 |

構築の回数は同じで、 $E_x$ は 1e-12 Hartree 以内で合いました。SCF が止まるステップは変わります（最大で 8 ステップ後）。全エネルギーの差は 1.4e-11 Hartree 以内です。例外は O₂ で、最後の内側の SCF が 42 ステップ目でなく 38 ステップ目で止まりました。全エネルギーの差は 7.5e-10 Hartree です。`scf.criterion` を 1e-12 にすると、2 つの版とも 54 ステップで止まり、12 桁が同じでした。

**XC の GPU 経路。** `3fafc96` は、M8 の一覧の「GPU での XC の計算」に当たります。14 節では、時期を「利用者の判断」としていた項目です。2 つ目の指示の「すべて」に入ると、作業者が判断しました。CPU 版の `.out` は、前の版と日付と時間の行だけが違います。デバイスの上の値は、利用者の計算機での確認を待っています。

**ABF の分け方。** `27a952a` の実機の確認で、半分の構築の複数プロセスの効率が下がっていました。CPU 版の `siz-p444` の交換行列の段の並列効率は、2 プロセスで 94% から 87%、4 プロセスで 78% から 66% です（計算値）。`92f78c7` の後、クラウドのコンテナ（4 コア）で、HSE06 の `siz-p444` の 10 回の構築の和を測りました。

| プロセス | 前（秒） | 後（秒） | 受け持ちの最短と最長の比（前、後） |
|---|---|---|---|
| 2 | 11.01 | 10.75 | 0.866、0.862 |
| 4 | 7.09 | 6.41 | 0.746、0.822 |

2 プロセスは、原子の境界で分かれるようになりました（前の版は 72 個と 66 個）。2 番目の原子の見積もりは、1 番目の 1.10 倍です。2 番目の原子を切ると、その部分ごとの仕事（原子の見積もりの 9%）が 1 番目のプロセスに加わります。1 プロセスの `.out` は、日付と時間の行だけが違います。2 プロセスと 4 プロセスの全エネルギーは、1 プロセスと 1e-12 Hartree 以内で合いました。4 プロセスのメタンと、2 プロセスと 4 プロセスのダイヤモンドも、前の実行と 2e-10 Hartree 以内で合います。SCF のステップ数の違いと同じ大きさの差です。

**力の重みの半分の形。** `2198f55` から、力の重みを半分の構築の形で作ります（6.7 節）。中心の原子が回るのは、半分の構築と同じ V 近傍の組だけです。組 1 つの `W` の大きさの積は 2 回から 3 回に増え、回る組は約半分になります。対の段は重みについて線形なので、力は丸めの範囲で同じです。原子を動かした Si の系を、1 プロセス 1 スレッドで、前の版と続けて測りました（秒、前と後）。

| 入力 | 力の重み | 厳密交換の力 | 実行の全体 |
|---|---|---|---|
| Si、HSE06 | 4.11、3.11 | 4.47、3.47 | 66.6、65.4 |
| Si、PBE0 | 27.62、16.58 | 28.30、17.28 | 199.5、184.4 |

1 プロセスから 4 プロセスの CPU 版の 8 件では、全エネルギーと力の表示の全ての桁が、前の版と同じでした。原子を動かさない Si の HSE06 では、プロセス 0 の重みも短くなりました。2 プロセスで 2.12 秒から 1.31 秒、4 プロセスで 1.26 秒から 0.84 秒です。

**準備と力の対のスレッド。** `7a79b99` から、プロセスが受け持つ準備の対と力の対を、`-nt` のスレッドに分けます（6.7 節）。V ブロックの配列は、呼び出したスレッドが先に確保します。スレッドの数で、ブロックの置き場所が変わらないようにするためです。1 プロセスから 4 プロセスの 1 スレッドの 8 件では、`.out` が `2198f55` と日付と時間の行だけ違いました。クラウドのコンテナ（4 コア）で、`-nt 4` の 1 プロセスの時間を、前の版と交互に 2 回ずつ測りました。表には、前の版の 2 回、後の版の 2 回の順に書いています（秒）。

| 入力 | 準備 | 厳密交換の力 | そのうちの重み |
|---|---|---|---|
| ダイヤモンド 54 原子、PBE0 | 1.43 と 1.79、0.54 と 0.58 | 2.27 と 2.60、0.96 と 0.81 | 0.33 と 0.29、0.26 と 0.29 |
| Si、HSE06（原子を動かした系） | 0.64 と 0.59、0.38 と 0.37 | 1.86 と 1.48、1.41 と 1.12 | 1.49 と 1.03、1.30 と 1.02 |

重みのコードは、この版で変えていません。Si の重みは、同じ版の 2 回の間でも 0.28 秒と 0.46 秒違いました。力から重みを引いた残りが、対の段の時間です。ダイヤモンドでは、1.94 秒と 2.31 秒から 0.70 秒と 0.52 秒になりました。Si では、0.37 秒と 0.45 秒から 0.11 秒と 0.10 秒です。

**メッシュの周期による V 近傍のまとめ。** `1626dc4` から、Band ソルバーの `GCenter` と `REGULAR` のメッシュでは、V 近傍をメッシュの周期でまとめます（6.4 節）。メッシュの点の $K(\mathbf k)$ は丸めの範囲でしか変わらず、キーワードも要りません。そのため、作業者の判断で入れました。V 近傍の site は、PBE0 の Si（4×4×4）で 1,698 個から 270 個になりました。2×2×2 のダイヤモンドでは、PBE0 の 762 個と HSE06 の 550 個が、どちらも 46 個です。HSE06 の Si（4×4×4）は、198 個のままでした。この系で要素が 2 個以上の組は、原子自身の周期像 $(a,\mathbf R)$ と $(a,-\mathbf R)$ の組だけです（計算値）。その組は、2 つの site を残します。1 プロセスから 4 プロセスの CPU 版の 8 件を、`7a79b99` と比べました。まとめた 2 件の全エネルギーの差は、2.0e-12 Hartree と 1.0e-12 Hartree です。力の差は 1e-12 Hartree/bohr 以下でした。残りの 6 件は、`.out` が日付と時間の行だけ違います。測定は、クラウドのコンテナ（4 コア）で、ほかの計算を止めて行いました。1 プロセス 1 スレッドの時間を、前の版と交互に 2 回ずつ測っています。表には、前の版の 2 回、後の版の 2 回の順に書いています（秒）。

| 入力 | 構築の和 | 厳密交換の力 | 実行の全体 |
|---|---|---|---|
| Si、PBE0（原子を動かした系、構築 11 回） | 114.73 と 116.13、25.30 と 24.60 | 17.41 と 17.20、3.87 と 3.98 | 179.7 と 181.1、76.9 と 75.9 |
| ダイヤモンド、2×2×2 の k 点、PBE0（構築 7 回） | 0.85 と 0.93、0.14 と 0.15 | 0.13 と 0.13、0.04 と 0.04 | 6.8 と 7.2、6.1 と 6.7 |
| ダイヤモンド、2×2×2 の k 点、HSE06（構築 7 回） | 0.57 と 0.55、0.14 と 0.15 | 0.10 と 0.10、0.04 と 0.06 | 6.1 と 6.3、5.7 と 7.0 |

Si の PBE0 の構築 1 回の行列積は、235,454 個から 37,676 個に減りました。力の重みは、16.72 秒と 16.51 秒から 3.16 秒と 3.29 秒です。site の形式の $D$ と $K$ のブロックは、10,166 個から 4,416 個になりました。スピン 1 成分あたりでは、13.1 MB から 5.7 MB です。同じ Si を 4 プロセスで 1 回ずつ測ると、構築の和は 38.33 秒から 7.10 秒でした。実行の全体は、71.3 秒から 32.5 秒です。ダイヤモンドの実行の全体は、厳密交換の外の時間が大半を占め、同じ版の 2 回の間でも 1.3 秒まで違いました。

**メッシュの周期による列のまとめ。** `51cd508` から、V 近傍をまとめた計算では、構築の列（ $S_A$ ）もメッシュの周期でまとめます（6.4 節）。V 近傍のまとめと同じく、丸めの範囲の変更で、キーワードも要りません。そのため、作業者の判断で入れました。プロセス 0 の最大の積の列は、PBE0 の Si（4×4×4）で 7,332 個から 1,664 個になりました。HSE06 の同じ Si では 6,162 個から 1,664 個、2×1×1 のダイヤモンドでは 652 個から 16 個です。積の数は変わりません。CPU 版の 8 件（1 プロセスから 4 プロセス）を、`1626dc4` と比べました。全エネルギーの差は 2.0e-12 Hartree 以下で、力の差は 1e-12 Hartree/bohr 以下です。O₂ とメタンの 3 件は、`.out` が日付と時間の行だけ違います。k 点を使う 5 件のメッシュは、どれも `GCenter` です。符号の付く `REGULAR` のメッシュは、2×1×1 のダイヤモンド（2 プロセス）と 4×4×4 の Si（PBE0、1 プロセス）で確かめました。まとめない `7a79b99` との差は、全エネルギーで 1.0e-12 Hartree 以下、力で 3.0e-12 Hartree/bohr 以下です。2 つの系の力の最大の成分は、5.23e-2 と 1.44e-2 Hartree/bohr でした。時間は、`1626dc4` と交互に 2 回ずつ測りました（1 プロセス 1 スレッド、ほかの計算を止めた 4 コアのコンテナ）。表には、前の版の 2 回、後の版の 2 回の順に書いています（秒）。

| 入力 | 構築の和 | 厳密交換の力 | 実行の全体 |
|---|---|---|---|
| Si、PBE0（原子を動かした系、構築 11 回） | 24.26 と 24.57、14.29 と 14.35 | 3.81 と 3.96、2.99 と 3.08 | 74.9 と 76.5、64.0 と 63.8 |
| ダイヤモンド、2×2×2 の k 点、PBE0（構築 7 回） | 0.14 と 0.14、0.02 と 0.01 | 0.04 と 0.04、0.04 と 0.03 | 6.5 と 6.1、6.4 と 5.7 |
| ダイヤモンド、2×2×2 の k 点、HSE06（構築 7 回） | 0.14 と 0.14、0.04 と 0.05 | 0.04 と 0.04、0.03 と 0.03 | 5.7 と 5.6、6.1 と 5.6 |

Si の PBE0 の力の重みは、3.07 秒と 3.27 秒から 2.31 秒と 2.33 秒になりました。重みの積も、類の代表の列だけを使うためです。同じ Si を 4 プロセスで 1 回ずつ測ると、構築の和は 7.06 秒から 4.05 秒でした。プロセスの受け持ちは 0.53〜0.58 秒から 0.33〜0.35 秒、実行の全体は 32.2 秒から 27.8 秒です。ダイヤモンドの構築は、7 回の和でも 0.05 秒以下になりました。実行の全体は、厳密交換の外の時間で決まります。

**呼び出しを引き受ける条件。** 列をまとめると、プロセスごとの最大の積が小さくなります。4 プロセスの Si と s2p2d1 のダイヤモンドでは、どの積も下限の 1e9 回に届きません。`51cd508` の実機の確認で、デバイスがこの呼び出しを断ったことが分かりました。`079d22c` で、条件を呼び出しの全ての積の乗算の和に変えています（11 節）。下限の値と、GEMMul8 を通す積は変えていません。CPU 版の結果は同じビットで、CUDA 版の結果は丸めの範囲で変わります。キーワードも要らないので、作業者の判断で入れました。M6 の道具では、この 2 つの系と PBE0 の 2 つの系が、大きい積のない呼び出しをデバイスで計算します。GEMMul8 で計算する積がないので、この 4 つの系の `NAME-g8` の実行は除きました。

**実機の確認。** 利用者の計算機（RTX 5080、1 スレッド）で、6 回確かめました。占有の確認は、6 回とも全て「空き」でした。

- `4577361`（2026-11-25 の 22:21 から 23:28、JST）。HSE06 と PBE0 の 14 件ずつは全て収束し、判定のある 84 行に不合格の行はありません。CUDA 版と CPU 版の全エネルギーの差は、最大 5.1e-11 Hartree です（`diaz-p222-host`）。力の差は 9.0e-12 Hartree/bohr 以内でした（限度 1e-6）。ただし、この 10 個の系の力は、対称性でほぼ 0 です。`gpu` の 10 件では、プロセス 0 の重みの呼び出しが全てデバイスで計算されました。力の時間は、前の確認（`9e01d40`）の 0.144 倍から 0.635 倍です（計算値）。`diaz-p222-pbe0-gpu` では 39.53 秒から 5.71 秒、`siz-p444-pbe0-gpu` では 26.56 秒から 5.30 秒になりました。重みの呼び出し 1 回のメモリーは、最大 1296.8 MB です（`diaz-p222-pbe0-gpu`）。
- `27a952a`（2026-11-25 の 23:55 から 11-26 の 01:05、JST）。HSE06 と PBE0 の 18 件ずつは全て収束し、判定のある 132 行に不合格の行はありません。CUDA 版と CPU 版の全エネルギーの差は、7.0e-12 Hartree 以下でした。力の差は 4.0e-12 Hartree/bohr 以下です。原子を動かした 2 つの系の力の最大の成分は、1.526e-2 と 1.535e-2 Hartree/bohr です。両方にある 28 件の `4577361` との差は、全エネルギーで 4.7e-11 Hartree 以下でした。力の差は 1.0e-11 Hartree/bohr 以下です。交換行列の段は、`4577361` の 0.524 倍から 0.709 倍になりました。`DFT in total` は 0.580 倍から 0.894 倍です（計算値）。デバイスの構築の呼び出し 1 回のメモリーは、`diaz-p222-pbe0-gpu` で 1246.9 MB から 938.6 MB に減りました。
- `f0421a6`（2026-11-26 の 01:15 から 02:35、JST）。HSE06 と PBE0 の 25 件ずつは全て収束し、判定のある 89 行ずつに不合格の行はありません。CUDA 版と CPU 版の全エネルギーの差は、HSE06 で最大 9.3e-10 Hartree でした（`ch4-gpu`。限度 1e-7）。PBE0 では、最大 6.0e-12 Hartree です。1e-10 を超えたのは、O₂ とメタンの HSE06 の 4 件でした。力の差は 3.6e-10 Hartree/bohr 以下です（`o2-gpu`）。準局所部分をホストで計算させると（`OPENMX_XC_GRID_GPU=0`）、`siz-p444-pbe0-gpu` の全エネルギーは同じ値でした。`o2-gpu` では 1.8e-9 Hartree 違い、SCF の止まるステップも 43 と 38 で違います。この確認と `9e01d40` の O₂ の HSE06 の 12 件では、42 ステップ以上で止まった 7 件が全て同じ値でした。37 と 38 ステップで止まった 5 件は、3.3e-9 Hartree の幅にあります（計算値）。Set_Hamiltonian の時間は、`siz-p444-pbe0-gpu` で 6.693 秒から 4.099 秒に縮みました。`o2-gpu` では、4.972 秒から 0.714 秒です。`27a952a` と比べた 36 件では、全エネルギーの差が 1.3e-11 Hartree 以下でした。力の差は 4.0e-12 Hartree/bohr 以下です。準備は 0.257 倍から 0.445 倍、`DFT in total` は 0.768 倍から 0.996 倍になりました（計算値）。ハイブリッド汎関数でない 3 つの入力は、`OPENMX_GPU=0` の `.out` が日付と時間の行だけ違います。GPU を使う実行の差は 6.4e-11 Hartree 以下で、同じ実行ファイルの 2 回の差と同じ大きさでした。
- `7fa9431`（2026-11-26 の 05:39 から 06:30、JST）。コードは `1626dc4` と同じです。HSE06 と PBE0 の 25 件ずつは全て収束し、判定のある 89 行ずつに不合格の行はありません。CUDA 版と CPU 版の全エネルギーの差は、HSE06 で最大 8.6e-10 Hartree でした（`ch4-gpu`）。PBE0 では、最大 1.0e-12 Hartree です。力の差は 3.2e-10 Hartree/bohr 以下でした（`o2-gpu`）。`f0421a6` と比べた 50 件では、全エネルギーの差が 2.9e-10 Hartree 以下でした。PBE0 の差は、6.0e-12 Hartree 以下です。力の差は、2.0e-10 Hartree/bohr 以下でした。V 近傍は、PBE0 の Si で 1,698 個から 270 個になりました。2×2×2 のダイヤモンドでは、PBE0 の 1,194 個と HSE06 の 586 個が、どちらも 46 個です。PBE0 の 18 件の交換行列の段は、`f0421a6` の 0.092 倍から 0.225 倍になりました（計算値）。2 回の平均の段は、`siz-p444-pbe0-gpu` で 13.11 秒から 2.96 秒、`diaz-p222-pbe0-gpu` で 16.02 秒から 1.92 秒です。デバイスのメモリーは、呼び出し 1 回で最大 624.3 MB から 239.9 MB に減りました（`siz-p444-pbe0-gpu`）。`diaz-p222-pbe0-gpu` では、938.6 MB から 267.9 MB です。V 近傍がまとまらない HSE06 の Si では、1 プロセスの段が 14.44 秒と 14.40 秒で、変わっていません。準備の時間も、0.93 倍から 1.06 倍の間でした（1 スレッドの実行なので、対のスレッドの効果は測っていません）。
- `51cd508`（2026-11-26 の 06:38 から 07:41、JST）。HSE06 と PBE0 の 25 件ずつは全て収束し、`<EXX>  WARNING` の行はありません。不合格の 14 行は、全てデバイスの呼び出しの行です。4 プロセスの Si と s2p2d1 のダイヤモンドでは、HSE06 と PBE0 の両方で、デバイスが呼び出しを全て断りました。原因は、この節の「呼び出しを引き受ける条件」にあります。CUDA 版と CPU 版の全エネルギーの差は、HSE06 で最大 9.0e-10 Hartree でした（`ch4-gpu`）。PBE0 では、最大 1.1e-11 Hartree です。力の差は 3.7e-10 Hartree/bohr 以下です。`7fa9431` との差は、全エネルギーで 5.5e-10 Hartree 以下（`o2-np2-gpu`）、力で 3.0e-10 Hartree/bohr 以下でした。2 回の平均の交換行列の段は、`siz-p444-pbe0-gpu` で 2.91 秒から 2.04 秒、`siz-p444-gpu` で 2.12 秒から 1.46 秒になりました。呼び出しを断った系は遅くなり、`diaz-p222-gpu` は 2.54 秒から 4.52 秒です。呼び出し 1 回のメモリーは、`siz-p444-gpu` で 208.4 MB から 101.5 MB に減りました。
- `079d22c`（2026-11-26 の 07:47 から 08:11、JST）。呼び出しを断った 4 つの系を中心に、HSE06 と PBE0 の 8 件ずつを実行しました。全て収束し、判定のある 29 行ずつに不合格の行はありません。4 つの系では、構築の呼び出しを全てデバイスが計算しました。`siz-p444-np4-gpu` は 4 プロセスで 40 回、`diaz-p222-gpu` は 12 回です。力の重みの呼び出しも、プロセス 0 のデバイスが計算しています。CUDA 版と CPU 版の全エネルギーの差は、最大 1.6e-11 Hartree でした（`diaz-p222-gpu`）。力の差は 2.0e-12 Hartree/bohr 以下です。`51cd508` との差は、全エネルギーで 1.9e-11 Hartree 以下、力で 1.0e-12 Hartree/bohr 以下でした。CPU 版の 4 件は、表示の全ての桁が `51cd508` と同じです。2 回の平均の交換行列の段は、`diaz-p222-gpu` で 4.54 秒から 0.85 秒、`diaz-p222-pbe0-gpu` で 3.46 秒から 0.66 秒になりました。4 プロセスの Si は、HSE06 で 3.28 秒から 2.08 秒、PBE0 で 4.46 秒から 2.59 秒です。`7fa9431` の値（2.54 秒、1.92 秒、2.68 秒、3.38 秒）より短くなりました。呼び出し 1 回のメモリーは、`diaz-p222-pbe0-gpu` で `7fa9431` の 267.9 MB から 109.2 MB に減っています。

`f0421a6` の確認には、`64352d9` から `f0421a6` までの 5 つのコミットが入っています。`7fa9431` の確認には、`92f78c7`、`2198f55`、`7a79b99`、`1626dc4` の 4 つのコミットが入っています。`51cd508` と `079d22c` の確認は、それぞれのコミットだけを対象にしました。

**採らなかった形。** 次の 3 つは、効果が小さいと分かったので入れていません。

- V 近傍の組ごとの積の組み替え。PBE0 の Si の大きさで、3 つの形を測りました（クラウドのコンテナ、OpenBLAS の 1 スレッド）。全ての $P$ を 1 回の積にまとめる形は、今の形の 1.20 倍から 1.46 倍の時間でした。列、 $P$ 、 $\mu$ の順の配列では、項の時間が少し縮みます。代わりに、`W` の積が 0.114〜0.128 秒から 0.155〜0.170 秒に延びました。2 つの積を 1 つにまとめる形は、12 回の中央値が今の形の 1.021 倍、最小が 0.992 倍です。
- 構築の配列を、キャッシュに収まる大きさに分ける形。PBE0 の Si では、組 1 つの `W` が 8 MB、 $P$ 1 つの `G` と括弧が 4.5 MB です。 $P$ で分けると、分けた数だけ T ブロックを読み直します。 $l$ で分けると、積の呼び出しの数が増えます。この形は測っていません。効果がないという判断は、作業者の見積もりです。
- `G` と括弧の転置を省く形。転置の時間は、原子 1 個の構築の約 2.4% でした（測定）。

**利用者の判断が要る形。** 密度行列の小さいブロックを省く形は、結果を丸めの範囲より大きく変え、しきい値のキーワードが要ります（1.3 節）。列をまとめた後は、構築より SCF のステップが実行の多くを占めます。PBE0 の Si（`51cd508`、1 プロセス）では、11 回の構築の和が 14.3 秒で、実行の全体は 64 秒でした。SCF の 107 ステップのうち、92 ステップが 10 回の内側の SCF で、1 回に 7 から 13 ステップです。外側の反復の差は、1 回ごとに 0.18 倍から 0.20 倍に縮みました（計算値）。そこで、反復を減らす形が 2 つ考えられます。早い外側の反復で、内側の SCF を早めに止める形と、外側の反復を外挿で減らす形です。どちらも、収束の判定の範囲で結果を変え、キーワードが要ります。3 つを入れるかは、2026-11-26 にカードで利用者に尋ねました。利用者の回答は、同じ日の 08:37（JST）の「3 つとも入れる」です。実装と測定は、次の「結果を変える形」に書きます。

**結果を変える形。** `6c9a592` で、利用者が選んだ形を入れました。キーワードを書かない計算で変わるのは、混合の始め直しだけです。2 つのキーワードは、既定の値では何もしません。

- 混合の始め直し（既定）。構築の後の RMM-DIIS、GR-Pulay、RMM-ADIIS は、新しい $K$ の最初のステップで履歴を作り直します（6.6 節）。利用者が、カードで既定を変える形を選びました（2026-11-26 の 09:22、JST）。
- 外挿（`scf.exx.outer.extrapolation`）。次の内側の SCF の $K$ を、前の構築と合わせて Anderson 法で作ります（6.6 節の「外側反復の外挿」）。
- 内側の判定（`scf.exx.inner.criterion.ratio`）。不動点から遠い $K$ の内側の SCF を、 $\lvert\delta\rvert$ に比例する判定でも止めます（6.6 節の「内側の SCF の緩い判定」）。
- 小さい寄与を省く形。入れていません。測った系では、省ける組がほとんどないと分かったためです（下の「省く形の測定」）。利用者が、カードで「入れずに測定を書く」を選びました（同じ日の 09:25）。

時間は、クラウドのコンテナ（4 コア）で、ほかの計算を止めて 1 件ずつ測りました。1 プロセス 1 スレッドで、停止条件は既定の値です（`scf.criterion` と `scf.exx.outer.criterion` が 1e-6）。表には、SCF のステップ数、構築の回数、実行の全体の時間（秒）を順に書きました。「前」は `9f012f4`、「既定」はキーワードを書かないこの版、「両方」は外挿 2 と比 0.1 の組です。表の時間は、緩い判定の最後の修正（13 節の 81）の前に測りました。この修正は表の計算を変えず、修正の後の `.out` は、日付と時間の行だけが違います。

| 入力 | 前 | 既定 | 比 0.1 | 外挿 2 | 両方 |
|---|---|---|---|---|---|
| Si、PBE0（4×4×4 の k 点） | 46、7、37.4 | 40、7、35.0 | 35、7、33.0 | 31、5、29.3 | 27、5、28.1 |
| Si、HSE06（4×4×4 の k 点） | 38、6、31.1 | 33、6、29.0 | 31、6、28.1 | 29、5、26.4 | 27、5、26.8 |
| ダイヤモンド、PBE0（2×2×2 の k 点） | 24、4、4.3 | 21、4、4.2 | 20、4、4.0 | 21、4、4.1 | 20、4、4.0 |
| ダイヤモンド、HSE06（2×2×2 の k 点） | 24、4、4.4 | 21、4、4.2 | 20、4、4.2 | 21、4、4.2 | 20、4、4.1 |
| メタン、HSE06 | 36、5、4.4 | 32、5、4.0 | 32、5、4.1 | 28、4、3.5 | 28、4、3.6 |
| O₂、HSE06、スピン分極 | 30、4、8.5 | 27、4、7.2 | 24、4、7.2 | 27、4、7.6 | 24、4、7.1 |

全エネルギーの「前」との差の最大は、既定で 1.2e-9 Hartree、比 0.1 で 4.9e-10 Hartree でした。外挿 2 では 9.7e-10、両方では 6.7e-10 Hartree です。最も大きい差は O₂ の「既定」で、ほかの入力の差は 2.1e-10 Hartree 以下です。両方の組の SCF のステップ数は、6 つの入力で「前」の 0.59 倍から 0.83 倍になりました。実行の全体の時間は、2 つの Si で 0.75 倍と 0.86 倍です。ほかの 4 つの入力は、0.82 倍から 0.94 倍でした。この 4 つの「前」の時間の内訳では、力の計算や VNA の射影子の行列など、SCF の回数によらない時間が 26% から 53% です。厳密交換の部分は、全体の 1% から 4% でした。

厳しい停止条件（`scf.criterion` 1e-12、`scf.exx.outer.criterion` 1e-9）の PBE0 の Si も、3 件を続けて測りました。「前」は 107 ステップ、構築 11 回、62.8 秒でした。既定は 95 ステップ、11 回、59.2 秒で、両方の組は 54 ステップ、9 回、43.1 秒です。全エネルギーの「前」との差は、既定で 0 Hartree、両方の組で 1.0e-12 Hartree でした。

厳しい `scf.criterion`（1e-12）の O₂（外側は既定の 1e-6）では、既定が 50 ステップ、構築 4 回でした。両方の組は 44 ステップ、構築 5 回で、構築が 1 回増えます。外側の閾値を満たした構築の後も内側の SCF を緩めた最初の形では、49 ステップ、構築 7 回でした（13 節の 81）。

キーワードの値は、2 つの Si で比べました。比 0.1 で外挿 1、3、4 と、外挿 2 で比 0.03、0.3 です（2 件ずつ並べて実行したので、時間は比べていません）。PBE0 の Si は構築 5 回で、27 から 29 ステップでした。HSE06 の Si は構築 5 回で、27 から 28 ステップです。外挿 1 では、PBE0 の最後の構築の $\lvert\delta\rvert$ が 7.1e-7 Hartree でした。外挿 2 の 2.8e-8 Hartree より大きい値です。外挿 3 と 4 は、外挿 2 と同じ構築の回数で止まります。そこで、表には外挿 2 と比 0.1 を載せました。

混合の始め直しは、SCF の道具の混合の 48 件を、前の版と比べて確かめました（メタン、水、8 原子のダイヤモンド。`scf.criterion` 1e-10）。RMM-DIIS の 3 件は、51、66、48 ステップから 43、61、44 ステップになりました。GR-Pulay の 3 件は、56、89、59 ステップから 53、82、55 ステップです。`StartPulay` 10 の 6 件では、5 件が 4 から 6 ステップ減り、ダイヤモンドの RMM-DIIS だけが 1 ステップ増えています。構築の回数は全て同じで、全エネルギーの差は 8.5e-10 Hartree 以下です。ほかの方式の 27 件と GGA-PBE の 9 件は、全エネルギーとステップ数が前の版と同じでした。RMM-ADIIS は道具にないので、メタンを別に実行しました（`scf.criterion` 1e-10）。512 ステップから 490 ステップになり、全エネルギーの差は 5.0e-11 Hartree です。

SCF の道具には、2 つのキーワードの 8 件を加えました（`-hse-accel` と `-hse-accel-np2`。メタン、水、8 原子のダイヤモンド、O₂）。SCF のステップ数は、既定の実行とこの組の順に、43 と 34、61 と 46、44 と 31、40 と 28 です。構築の回数は、同じ順に 5 と 5、6 と 6、6 と 5、4 と 4 でした。既定の実行との全エネルギーの差は 1.0e-9 Hartree 以下で、限度 1e-8 の内です。2 プロセスと 1 プロセスの差は、O₂ の 2.6e-9 Hartree が最大でした。最後の内側の SCF は `scf.criterion` で止まりますが、止まるステップは丸めで動きます。O₂ の 2 つの実行は、残差ノルムが 1.2e-9 と 5.4e-10 のステップで止まりました。`scf.criterion` を 1e-12 にすると、2 つの全エネルギーは小数点以下 12 桁まで同じです。そこで、この行の限度は、既定の実行との差と同じ 1e-8 にしました。

メタンの分子動力学（NVE）と構造最適化も、既定と両方の組で 3 ステップずつ実行しました（`scf.criterion` 1e-10、外側は既定の 1e-6）。SCF のステップ数は、分子動力学で 127 から 90、構造最適化で 126 から 91 に減りました。構築の回数は、どちらも各ステップ 5 回で同じです。各ステップの全エネルギーの差は、分子動力学で 3.4e-9 Hartree 以下、構造最適化で 5.8e-8 Hartree 以下でした。既定の実行は、外側反復を $\lvert\delta\rvert$ が 1.8e-7 から 2.0e-7 の構築で終えており、差はこれより小さい値です。座標は、`.md` の表示の桁（1e-5 Å）まで同じでした。

**省く形の測定。** k メッシュの計算の $D$ は、遠いブロックでも小さくなりません。Si（4×4×4、PBE0）の site の形式では、30 bohr より遠くても、全てのブロックで要素の最大が 1.7e-3 以上でした。メッシュの周期を持つ $D(\mathbf R)$ は、Born–von Kármán のスーパーセルの周期で繰り返すためです。 $K$ は速く小さくなり、要素の最大が 1e-4 未満のブロックは、PBE0 で 92.6%、HSE06 で 95.1% です。 $D$ だけのしきい値では、省けるブロックがありません。そこで、中心の原子と V 近傍の組ごとに、組が $K$ に加える 2 つの項の要素の最大を測りました（最初の 2 回の構築）。1e-6 未満の組は、Si の PBE0 で 272 組の 0 組、HSE06 で 200 組の 0 組でした。6×6×6 の Si（PBE0）でも、880 組の 0 組です（1e-5 未満は 50 組）。Γ 点のデカン（C₁₀H₂₂、PBE0）では、1056 組のうち 38 組が 1e-6 未満でした。ただし、この系の構築は実行の全体の約 2% で、省いても全体はほとんど縮みません。組を計算する前の見積もり（ $V$ 、2 つの T ブロック、 $D$ の要素の最大の積）は、実際の値と合いませんでした。実際の値と見積もりの比の中央値は、5 つの系で 2.7e-3 から 3.1e-2 です。比が 1 を超える組もあり（最大 2.6）、見積もりは上限になりません。測定には、作業者が一時的に書いたコードを使い、そのコードはリポジトリーに入れていません。

**次の作業。** バンド分散と DOS は改訂 30 で、結果を変える高速化は改訂 31 で入れました。M8 の次の項目は、利用者が選びます（M8 の一覧）。`6c9a592` と M8 の出力の CUDA 版は、利用者の計算機での確認を待っています。

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
- GPU での実行時間は、設計の時点では未測定でした。M6 の実測は、この節の後半にあります。この節の方針は演算の形からの判断で、実測で見直します。

M6 の実装です。行列積の部品は実機なしで書き、実機の最初の段階（`38992bc`）で実行しました（9 節の M6）。

大きい行列積をデバイスで計算する部品は、`source/Exact_Exchange_Device.c` です。結合層の `ExactExchange_Dgemm` は、`modules/exx` から来る積を 1 つずつ `Exact_Exchange_Device_Dgemm` に渡します。この関数が 1 を返した積は、デバイスで計算が済んでいます。0 を返した積は、本体の `DGEMM` が計算します。CPU 版では、全ての積で 0 を返します。

デバイスが積を引き受ける条件は 4 つです。

- プロセスに使えるデバイスがある（`gpu_rank_device_usable`）。
- 乗算の回数 $mnk$ が下限以上である。下限は 1e9 回です。
- $\beta=0$ で、行列の列の間に詰め物がない（`lda`、`ldb`、`ldc` が行数に等しい）。乗算が下限以上の積は、全てこの形です（下の実測）。
- 3 つの行列が、デバイスの空きから 512 MiB を引いた範囲に入る。

1 つの積の手順は、次の順です。空きの問い合わせ（`OpenMX_GpuMemGetInfo`）、容量の判定、cuBLAS のハンドルの用意、3 つの行列の確保、2 つの因子の転送、積（`OpenMXBlasDgemm`）、同期、結果の転送、解放。積と積の間にデバイスに残るのは、ハンドルだけです。演算は FP64 です。入力が GEMMul8 を指定した計算では、積の呼び出しが橋渡しの `openmx_gemmul8Dgemm` に変わります。ほかの手順は同じです。

デバイスの呼び出しが失敗した積は、ホストが計算します。その後の扱いは 2 通りに分けました。

| 場合 | 当てはまる条件 | 扱い |
|---|---|---|
| 容量が足りない | 容量の判定で入らない。確保が `OPENMX_GPU_OUT_OF_MEMORY` を返す。ハンドルの生成か積が `OPENMX_BLAS_ALLOC_FAILED` を返す。 | デバイスを使い続けます。その積は「容量の不足でホストに残した積」に数えます。 |
| そのほかの失敗 | 上の条件に当たらない、成功でない状態コード | その計算の間、デバイスを使いません。プロセスは、失敗した段と状態コードを 1 回表示します。 |

どちらの場合も、確保した行列を解放し、ランタイムに残ったエラーを取り除きます（13 節の 32）。判定の後のメモリーの不足を 1 行目に入れた理由は、13 節の 34 に書きました。既存の GEMM の経路も、メモリーの不足とほかの失敗を分けています（`source/openmx_mgpu_gemm.cu:368-375`、`source/openmx_gpu_error.c:35-58`）。失敗の後でエラーを取り除いてホストに切り替える形は、`source/Divide_Conquer.c:596-672` と同じです。

容量が足りない場合の道筋は、既存の障害注入の環境変数 `OPENMX_GPU_CAPACITY_TEST_RECOVERY`（`docs/ENVIRONMENT_VARIABLES.md`）で試せます。`Input_std` がこの値を結合層に渡し、結合層は、残す空きを表せる最大の値にします。大きい積は全て、空きの問い合わせの後でホストに残ります。新しい環境変数は加えていません。

1 回の構築の行列積を、CPU 版で数えました（M6 の実測。基本単位胞の 2 原子、site の形式、1 ランク）。積の一覧は、どの構築でも同じでした。「積の数」は、V 近傍ごとの項を ABF 1 個ずつ計算する今の形（6.4 節）での値です。括弧の中は、`a7fbfb8` までの形での値です。ほかの列は、どちらの形でも同じです。

| 系 | 積の数 | 最大の積（ $m\times n$ 、内側の次元） | その個数 | 乗算の回数に占める割合 | 3 つの行列の合計 | 次に大きい積 |
|---|---|---|---|---|---|---|
| C5.0-s1p1 のダイヤモンド | 22,554（2,754） | 5020×40、492 と 5020×492、40（どちらも 9.9e7 回） | 各 2 | 58% | 20.5 MiB | 1968×10、10（2.0e5 回）が 550 個 |
| Si7.0-s2p2d1 の Si | 54,850（994） | 8229×897、1131 と 8229×1131、897（どちらも 8.3e9 回） | 各 2 | 58% | 135 MiB | 14703×69、69（7.0e7 回）が 198 個 |
| C6.0-s2p2d1 のダイヤモンド | 162,326（2,934） | 20163×897、2171 と 20163×2171、897（どちらも 3.9e10 回） | 各 2 | 53% | 487 MiB | 28223×69、69（1.3e8 回）が 586 個 |

最大の積は、中心の原子ごとに 2 個あります（ $G=T\cdot D$ と $T^{\mathsf T}R$ ）。2 個は形が違い、乗算の回数と 3 つの行列の合計は同じです。表の「割合」は、この 4 個の合計です。次に大きい積は $W=V\cdot T$ で、V 近傍の site ごとに 1 個あります。残りは 4 つの項の積で、V 近傍の site と中心の原子の ABF の組ごとに 4 個あります。Si では 54,648 個で、1 個の乗算は 1.9e5 回です（計算値）。乗算の回数の合計は、`a7fbfb8` までの形と同じです。項ごとの 1 個の積を、ABF ごとの積に分けただけだからです。4 つの項の積のうち、 $K$ に足す 2 個は $\beta=1$ で、ほかの積は全て $\beta=0$ です。行列に詰め物はありません。`a7fbfb8` までの形では、全ての積が $\beta=0$ でした。スピン分極のある計算では、積の数が 2 倍になります。下限の 1e9 回では、s2p2d1 の 2 つの系で 4 個の積がデバイスに行き、C5.0-s1p1 のダイヤモンドでは全ての積がホストに残ります。表の MiB は 2 の 20 乗バイトです。12 節の MB と、標準出力の MB も同じ単位です。

この実測から、今の部品で減らせる時間の上限が分かります。デバイスに送る積は、乗算の回数の 53% と 58% です。`a7fbfb8` までの CPU 版の構築では、DGEMM の中の時間が 62% と 63% でした（12 節）。デバイスでの時間を 0 と仮定しても、構築の時間が減るのは 3 分の 1 程度です（計算値）。この上限は、時間が乗算の回数に比例すると仮定した値です。クラウドのコンテナで測った Si の内訳では、大きい積の時間は、`a7fbfb8` までの構築の 21% でした（1 スレッド、12 節）。書き直した構築では 35% です。4 スレッドでは 9% と 16% でした。残りを減らすには、V 近傍ごとの計算もデバイスに移す必要があります。 $W=V\cdot T$ は、乗算の回数の 24% と 27% を占め、1 個あたり 16 MiB と 30 MiB の行列を使います。積ごとに行列を転送する今の形は、この積には向かないと考えられます。4 つの項の積は 1 個が小さく、Si の構築 1 回に 54,648 個あります。デバイスでは、ABF ごとに分けず、V 近傍 1 個を数回の積と配列の写しで計算する形が向くと考えられます。その形は、T ブロック、 $D$ 、`G`、`Rm` をデバイスに置きます。実装していません（14 節）。

2 つの限度（下限の 1e9 回と、残す空きの 512 MiB）は最初の値です。下限は、V 近傍ごとの積をホストに残し、最大の積だけをデバイスに送る値として選びました。実機の 3 つの段階は、この値のまま測っています。値を変えた測定はしていません。

実機の最初の段階の時間です（2026-11-24、`38992bc`、RTX 5080 と 18 コアのホスト）。V 近傍ごとの項を書き直す前の版の値です。「交換行列」は、全ての構築の時間の合計です。「ホストに残す」は、CUDA 版に `OPENMX_GPU_CAPACITY_TEST_RECOVERY=1` を付けた実行です。1 スレッドの行は、mpirun がプロセスを 1 コアに束縛した状態で測りました（13 節の 47）。18 スレッドの行は、束縛を外し、`-nt 18` を付けた実行です。測定は 1 回ずつです。

| 系 | スレッド | 交換行列（CPU 版） | 交換行列（CUDA 版、デバイスに送る） | うちデバイスの積 | 交換行列（CUDA 版、ホストに残す） |
|---|---|---|---|---|---|
| `siz-p444`（構築 10 回、デバイスの積 40 個） | 1 | 54.89 秒 | 40.58 秒 | 1.99 秒 | 45.31 秒 |
| 同上 | 18 | 測っていません | 30.57 秒 | 1.70 秒 | 30.01 秒 |
| `diaz-p222`（構築 12 回、デバイスの積 48 個） | 1 | 347.70 秒 | 272.78 秒 | 7.88 秒 | 306.29 秒 |
| 同上 | 18 | 測っていません | 201.28 秒 | 7.98 秒 | 199.55 秒 |

- 1 スレッドでは、積をデバイスに送ると、交換行列の段が 10% と 11% 短くなりました。
- 18 スレッドでは、送っても残しても変わりません。差は 2% 以下で、送る方が長い値でした。
- デバイスの 4 個の積は、2 回目からの構築で、1 回あたり 0.16 秒（Si）と 0.65 秒（ダイヤモンド）です。最初の構築は 0.55 秒と 0.71 秒でした。構築 1 回は 4.0 秒と 22.7 秒です。
- ホストが同じ 4 個の積に使う時間は、1 スレッドで構築 1 回あたり 0.67 秒と 3.4 秒です（2 つの実行の差からの計算値）。構築の 15% と 14% に当たります。
- スレッドを 1 から 18 にしても、交換行列の段は 0.66 倍と 0.65 倍にしかなりません（ホストに残した実行）。

上の「3 分の 1 程度」は、時間が乗算の回数に比例すると仮定した上限でした。実測では、大きい積がホストで使う時間は構築の 15% と 14% で、乗算の回数の割合（58% と 53%）の約 4 分の 1 です。残りは、V 近傍ごとの積と、積のほかの処理の時間です。その内訳は、この計算機では測っていません。クラウドのコンテナで測った内訳は、12 節にあります。今の部品は、スレッドを使うホストでは構築を速くしません。表の CPU 版の列は、コンパイラーも BLAS も違うので、部品の効果を見る相手にはなりません。部品の効果は、CUDA 版の 2 つの列の差です。

実機の 2 番目の段階の時間です（2026-11-24、`a7fbfb8`）。V 近傍ごとの項を書き直す前の版の値です。スレッド数は、道具が本体の `-nt` と BLAS に渡した数で、本体の表示で確かめました。1 スレッドの行は既定の mpirun、18 スレッドの行は `mpirun --bind-to none` です。測定は 1 回ずつです。「GEMMul8」は `NAME-g8`、「密行列の経路」は `NAME-dense` の実行です。

| 系 | 実行 | スレッド | 交換行列 | うちデバイスの積 | 加算 | DFT 全体 |
|---|---|---|---|---|---|---|
| `siz-p444` | CPU 版 | 1 | 44.01 秒 | 0.00 秒 | 4.85 秒 | 87.80 秒 |
| `siz-p444` | CPU 版 | 18 | 29.11 秒 | 0.00 秒 | 5.66 秒 | 55.65 秒 |
| `siz-p444` | CUDA 版 | 1 | 39.62 秒 | 1.67 秒 | 4.31 秒 | 62.56 秒 |
| `siz-p444` | CUDA 版 | 18 | 29.66 秒 | 1.65 秒 | 5.02 秒 | 54.26 秒 |
| `siz-p444` | CUDA 版、GEMMul8 | 1 | 38.48 秒 | 0.88 秒 | 4.47 秒 | 61.98 秒 |
| `siz-p444` | CUDA 版、GEMMul8 | 18 | 28.83 秒 | 0.83 秒 | 5.22 秒 | 53.82 秒 |
| `siz-p444` | CUDA 版、密行列の経路 | 1 | 40.01 秒 | 1.67 秒 | 2.28 秒 | 57.66 秒 |
| `siz-p444` | CUDA 版、密行列の経路 | 18 | 29.99 秒 | 1.66 秒 | 2.37 秒 | 45.75 秒 |
| `diaz-p222` | CPU 版 | 1 | 293.67 秒 | 0.00 秒 | 3.17 秒 | 354.52 秒 |
| `diaz-p222` | CPU 版 | 18 | 195.78 秒 | 0.00 秒 | 3.64 秒 | 225.17 秒 |
| `diaz-p222` | CUDA 版 | 1 | 273.09 秒 | 7.89 秒 | 3.23 秒 | 290.21 秒 |
| `diaz-p222` | CUDA 版 | 18 | 197.83 秒 | 7.82 秒 | 3.67 秒 | 215.49 秒 |
| `diaz-p222` | CUDA 版、GEMMul8 | 1 | 266.29 秒 | 3.34 秒 | 3.12 秒 | 283.01 秒 |
| `diaz-p222` | CUDA 版、GEMMul8 | 18 | 193.37 秒 | 3.31 秒 | 3.66 秒 | 211.20 秒 |
| `diaz-p222` | CUDA 版、密行列の経路 | 1 | 270.47 秒 | 7.87 秒 | 1.72 秒 | 283.67 秒 |
| `diaz-p222` | CUDA 版、密行列の経路 | 18 | 197.98 秒 | 7.87 秒 | 1.66 秒 | 209.67 秒 |

- 1 スレッドの CUDA 版の交換行列の段は、積をホストに残した実行（45.24 秒と 302.96 秒）の 0.88 倍と 0.90 倍でした。最初の段階の値と同じ程度です。
- 18 スレッドでは、CUDA 版の交換行列の段が CPU 版と同じ程度になりました（29.66 秒と 29.11 秒、197.83 秒と 195.78 秒）。積をホストに残す 18 スレッドの実行は、この段階では測っていません。
- GEMMul8 では、デバイスの積の時間が FP64 の 0.53 倍（`siz-p444`）と 0.42 倍（`diaz-p222`）になりました。FP64 の実行で、デバイスの積は交換行列の段の 4% と 3% です。段の時間の差は、どちらのスレッド数でも 3% 以下でした。
- 密行列の経路では、 $K$ の加算の段が、ホストの固有値ソルバーの実行の 0.53 倍でした（2 つの系の 1 スレッド）。この実行の行列の次数は 26 です。しきい値を下げずにこの経路に入る大きさでは、測っていません（14 節）。

実機の 3 番目の段階の時間です（2026-11-24、`857ede9`）。V 近傍ごとの項を書き直した版の値です。実行の条件と列の意味は、上の表と同じです。右端は、交換行列の段の時間を、上の表の同じ行の値で割った比です。

| 系 | 実行 | スレッド | 交換行列 | うちデバイスの積 | 加算 | DFT 全体 | 書き直す前との比 |
|---|---|---|---|---|---|---|---|
| `siz-p444` | CPU 版 | 1 | 24.60 秒 | 0.00 秒 | 4.80 秒 | 68.52 秒 | 0.56 |
| `siz-p444` | CPU 版 | 18 | 16.55 秒 | 0.00 秒 | 5.55 秒 | 42.95 秒 | 0.57 |
| `siz-p444` | CUDA 版 | 1 | 20.18 秒 | 2.64 秒 | 4.52 秒 | 43.75 秒 | 0.51 |
| `siz-p444` | CUDA 版 | 18 | 18.20 秒 | 2.64 秒 | 5.07 秒 | 42.91 秒 | 0.61 |
| `siz-p444` | CUDA 版、GEMMul8 | 1 | 19.34 秒 | 1.81 秒 | 4.43 秒 | 42.73 秒 | 0.50 |
| `siz-p444` | CUDA 版、GEMMul8 | 18 | 17.41 秒 | 1.79 秒 | 5.05 秒 | 42.05 秒 | 0.60 |
| `siz-p444` | CUDA 版、密行列の経路 | 1 | 19.98 秒 | 2.64 秒 | 2.31 秒 | 37.80 秒 | 0.50 |
| `siz-p444` | CUDA 版、密行列の経路 | 18 | 18.45 秒 | 2.64 秒 | 2.33 秒 | 34.06 秒 | 0.62 |
| `diaz-p222` | CPU 版 | 1 | 160.27 秒 | 0.00 秒 | 3.19 秒 | 221.35 秒 | 0.55 |
| `diaz-p222` | CPU 版 | 18 | 110.68 秒 | 0.00 秒 | 3.69 秒 | 140.05 秒 | 0.57 |
| `diaz-p222` | CUDA 版 | 1 | 137.24 秒 | 12.30 秒 | 3.14 秒 | 154.38 秒 | 0.50 |
| `diaz-p222` | CUDA 版 | 18 | 120.51 秒 | 12.30 秒 | 3.48 秒 | 137.76 秒 | 0.61 |
| `diaz-p222` | CUDA 版、GEMMul8 | 1 | 132.42 秒 | 7.73 秒 | 3.00 秒 | 149.16 秒 | 0.50 |
| `diaz-p222` | CUDA 版、GEMMul8 | 18 | 115.43 秒 | 7.74 秒 | 3.35 秒 | 132.08 秒 | 0.60 |
| `diaz-p222` | CUDA 版、密行列の経路 | 1 | 137.47 秒 | 12.30 秒 | 1.63 秒 | 150.18 秒 | 0.51 |
| `diaz-p222` | CUDA 版、密行列の経路 | 18 | 121.99 秒 | 12.32 秒 | 1.77 秒 | 133.93 秒 | 0.62 |

- 交換行列の段は、書き直す前の 0.50 倍から 0.56 倍（1 スレッドの 22 件）と、0.57 倍から 0.62 倍（18 スレッドの 8 件）になりました。クラウドのコンテナの CPU 版で測った比（12 節の 0.57 倍から 0.65 倍）より、小さい値です。C5.0-s1p1 のダイヤモンドの 15 件は、0.74 倍から 0.81 倍でした。
- 1 スレッドの CUDA 版の交換行列の段は、積をホストに残した実行（24.96 秒と 166.01 秒）の 0.81 倍と 0.83 倍でした。書き直す前は 0.88 倍と 0.90 倍です。2 つの実行の差の秒数は、5.62 秒と 29.87 秒から、4.78 秒と 28.77 秒になりました。段の全体が短くなった分だけ、比が下がっています。
- 18 スレッドでは、CUDA 版の交換行列の段が CPU 版より長くなりました（18.20 秒と 16.55 秒、120.51 秒と 110.68 秒）。書き直す前は、同じ程度でした。差（1.65 秒と 9.83 秒）は、デバイスの積の時間（2.64 秒と 12.30 秒）より小さい値です。2 つの版は BLAS と MPI が違うので、この差を部品だけの効果とは言えません。積をホストに残す 18 スレッドの実行は、この段階では測っていません。4 番目の段階で測りました（下）。
- スレッドを 1 から 18 にすると、CPU 版の交換行列の段は 0.67 倍と 0.69 倍、CUDA 版は 0.90 倍と 0.88 倍になりました。
- デバイスの積の時間は、FP64 で 1.67 秒から 2.64 秒（`siz-p444`）、7.89 秒から 12.30 秒（`diaz-p222`）に増えました。積の数と形は同じです。GEMMul8 でも、ほぼ同じ秒数だけ増えています（0.88 秒から 1.81 秒、3.34 秒から 7.73 秒）。FP64 の実行で、デバイスの積は交換行列の段の 13% と 9% です（1 スレッド）。
- GEMMul8 では、デバイスの積の時間が FP64 の 0.69 倍（`siz-p444`）と 0.63 倍（`diaz-p222`）でした。書き直す前は 0.53 倍と 0.42 倍です。FP64 との差の秒数は、書き直す前とほぼ同じです（0.79 秒と 4.55 秒から、0.83 秒と 4.57 秒）。交換行列の段の時間の差は、どちらのスレッド数でも約 4% でした。「DFT 全体」の差は、1 スレッドで 2% と 3% です。
- 密行列の経路では、 $K$ の加算の段が、ホストの固有値ソルバーの実行の 0.51 倍と 0.52 倍でした（1 スレッド）。

デバイスの積の時間が増えた原因は、結果の配列への最初の書き込みと考えられます。書き直した構築は、 $G$ と最後の行列積の結果を、確保したままの配列に受け取ります（13 節の 45）。OS は、確保したばかりの大きい配列のページを、最初の書き込みのときに割り当てます。`a7fbfb8` までは、構築が配列を 0 で埋めていたので、この時間は部品の外にありました。今は、デバイスからホストへの転送が最初の書き込みになります。根拠は 3 つです。

- 増えた時間は、FP64 の積 1 個あたり 24 ミリ秒（`siz-p444`）と 92 ミリ秒（`diaz-p222`）です。結果の配列の大きさ（2 個の平均で 64 MiB と 236 MiB）で割ると、1 MiB あたり 0.38 ミリ秒と 0.39 ミリ秒でそろいます（計算値）。
- FP64 と GEMMul8 で、増えた秒数がほぼ同じです。積の方式に依らない処理が増えたことになります。
- クラウドのコンテナで、同じ大きさの配列への写しを測りました。確保したままの配列への最初の写しは、同じ配列への 2 度目の写しより、1 MiB あたり 0.35 ミリ秒から 0.67 ミリ秒長くなりました。実機の値と同じ桁です。

実機では、この説明を確かめていません。デバイスの時間の内訳を出す手段がないためです。ホストが積を計算する経路では、同じ最初の書き込みが BLAS の中で起きます。結果の配列を中心の原子の間で使い回すと、最初の書き込みは構築 1 回に 1 度で済むと考えられます。実装していません（14 節）。

GEMMul8 を使うのは、入力に `scf.gemmul8.enable on` を書いた計算だけです。M6 の実機の段階で、この切り替えを書きました。キーワードがない入力と `off` の入力は、どのデバイスでも FP64 の cuBLAS で計算します。1.2 節は、GEMMul8 を FP64 との差を測ってから使うと決めています。この切り替えは、その差を測る手段です。

リポジトリーの GEMM の橋渡し（`openmx_gemmul8Dgemm`）は、`scf.gemmul8.enable` を書かない入力では、FP64 の弱い GPU で GEMMul8 を選びます（`source/Input_std.c:919-924` のコメント）。厳密交換の積を無条件でこの関数に通すと、差を測る前に GEMMul8 を使うデバイスが出ます。部品は、指定がある計算でだけ橋渡しを呼びます。`on` の入力では、橋渡しはデバイスの種類によらず GEMMul8 を選びます（`source/gemmul8_bridge.cu` の `gemmul8_disabled`）。環境変数 `OPENMX_GEMMUL8_DISABLE` と `OPENMX_GEMMUL8_DISABLE_D` は、`on` の指定より優先します（README の GEMMul8 の節）。

指定は、次の順に伝わります。`Input_std` が `scf.gemmul8.enable` を読み、値が `on` のときだけ結合層の入力の `gemmul8` を 1 にします。結合層は、この値を部品の状態に写します。新しいキーワードと環境変数は加えていません。既存のキーワードの既定値も変えていません。

橋渡しは、GEMMul8 で計算できない積を FP64 の cuBLAS で計算し、成功を返します。当てはまるのは、作業領域を確保できない場合と、環境変数が GEMMul8 を止めている場合です。どちらの方式で計算したかは、戻り値からは分かりません。部品は、橋渡しが FP64 で計算した呼び出しの数（`openmx_gemmul8NativeCalls`）を、積の前後で読みます。数が増えていない積を、GEMMul8 の積と数えます。`source/eigen_refine_gpu.c` も、同じ数で経路を確かめています。

橋渡しの設定は、既存の既定のままです。バックエンドは INT8、法の数は 15 個、スケーリングは正確な方式（`fastmode` が偽）です（`source/gemmul8_bridge.cu`）。GEMMul8 の README の表では、INT8 の実数の積で、15 個の法の有効ビットは 58 です（資料の値 [20]）。README は、この種の値を精度の保証ではないとしています。厳密交換の積での差は、実機の `NAME-g8` で測りました。1 スレッドの 4 つの系では、全エネルギーが FP64 の実行と 5e-12 Hartree 以内で合います。18 スレッドの `diaz-p222` では、差が 2.5e-11 Hartree でした。この実行は、SCF のステップの数が FP64 の実行と違います（101 と 99）。

橋渡しの作業領域は、積の後もデバイスに残ります。橋渡しは、作業領域をデバイスとストリームの組ごとに持ち、次の呼び出しで使い回します。大きさの上限は、既定でプロセスあたり 256 MiB です（`OPENMX_GEMMUL8_MAX_WORKSPACE_MB`）。部品は、この領域を解放しません。本体が、SCF の終わり（`source/DFT.c` の `openmx_gemmul8ReleaseWorkspaces`）と Band ソルバーの中で解放します。部品の容量の判定は、3 つの行列と残す空きの 512 MiB を数え、この作業領域を数えません。橋渡しは、作業領域を確保した後の空きが既定の 1,536 MiB を下回るとき、その積を FP64 で計算します（`ensure_workspace`）。

部品は、結果の行列を初期化せずに積に渡します。デバイスに送る積は、全て $\beta=0$ です。GEMMul8 は、 $\beta=0$ の積で結果の行列を読まずに書きます（サブモジュールが指すコミット `f8fdf0b` の `src/oz2/undo_scaling/undo_scaling.hpp` [20]）。

`on` の計算では、標準出力に 1 行が増えます。`G` は GEMMul8 が計算した積の数、`F` は橋渡しが FP64 で計算した積の数です。どちらも全プロセスの合計で、和は下の `N` になります。デバイスで計算した積が 1 個もない計算では出ません。

```
<EXX>  GPU: GEMMul8 computed G of the matrix products on the device, FP64 the other F (scf.gemmul8.enable on).
```

橋渡しには、積を複数のデバイスに分ける経路があります（`openmx_mgpu_dispatch_dgemm`）。この経路は、FP64 の呼び出しの数を増やさずに戻ります。経路に入るのは、ソルバーが複数の GPU の範囲を開いている間だけです（`source/openmx_mgpu_runtime.cu` の `cluster_scope`）。構築はこの範囲の外で積を呼ぶので、分ける経路には入りません。GPU が 2 台以上ある計算機では確かめていません。

切り替えの形は、作業者が決めました。差を測るには切り替えが要り、既存のキーワードの `on` に従う形は、新しいキーワードも既定値の変更も要りません。`on` は、既存のコードで「GPU の密行列の GEMM に GEMMul8 を使う」という指定です（`source/Input_std.c` の `off` の表示と、README の GEMMul8 の節）。既存の入力の結果は変わりません。改訂 10 までは、切り替えの方法を利用者が決めるとしていました。利用者に残した判断は 2 つです。この形を残すか、厳密交換のためのキーワードに分けるか。FP64 の弱い GPU で、厳密交換の積も GEMMul8 を既定にするか。2026-11-24 に、実機の測定の値を添えて利用者に確認しました。この改訂の時点で、回答はありません（14 節）。

FP64 の cuBLAS の積は、利用者の計算機で測りました（上の表）。書き直す前の版では、積をデバイスに送ると、交換行列の段が、ホストが 1 スレッドのとき短くなり、18 スレッドのとき変わりませんでした。書き直した版の 18 スレッドでは、CUDA 版の段が CPU 版より長くなっています（上の 3 番目の表）。リポジトリーにも、RTX 5080 と 18 コアの AVX-512 のホストの組で、ホストの方が速い段があるという記述があります（`source/openmx_host_simd.h:36-43`）。GEMMul8 での時間は、実機の 2 番目と 3 番目の段階で測りました（上の 2 番目と 3 番目の表）。

改訂 14 で、構築をスレッド並列にしました（8 節）。デバイスが積を引き受ける条件は、変えていません。関数表のループの中から来る積は、部品に渡さず、呼んだスレッドの BLAS で計算します。この積は ABF 1 個の 4 つの項の積で、乗算が下限の 1e9 回に届きません。部品の状態（デバイスの BLAS のハンドルと積の数）は 1 つで、複数のスレッドから同時に呼ぶ形にしていません。ループの外の積（`W` と 2 つの大きい積）は、今までどおり部品に渡します。部品が引き受けなかった積は、ホストがスレッドに分けて計算します（`Exact_Exchange_Host_Dgemm`）。

実機の 4 番目の段階の時間です（2026-11-24、`1219cb9`）。構築をスレッド並列にした版の値で、交換行列の段の秒数です。計算機とビルドの環境は、前の段階と同じです。「前の版」は、同じ日の午後に `857ede9` を 18 スレッドで測り直した値です。3 番目の段階の 18 スレッドの値より、4% から 6% 長い値でした。原因は調べていません。「ホストに残す」は、同じ CUDA 版で、積をデバイスに送らない実行（`NAME-host`）です。右端は、18 スレッドの値を「前の版」で割った比です。

| 系 | 実行 | 1 スレッド | 4 スレッド | 9 スレッド | 18 スレッド | 前の版（18 スレッド） | 比 |
|---|---|---|---|---|---|---|---|
| `siz-p444` | CPU 版 | 24.67 | 8.63 | 6.18 | 5.11 | 17.57 | 0.291 |
| `siz-p444` | CUDA 版、デバイスに送る | 20.28 | 8.92 | 7.07 | 6.31 | 19.16 | 0.329 |
| `siz-p444` | CUDA 版、ホストに残す | 25.34 | 測定なし | 測定なし | 4.98 | 18.11 | 0.275 |
| `diaz-p222` | CPU 版 | 160.67 | 55.41 | 35.48 | 29.45 | 116.40 | 0.253 |
| `diaz-p222` | CUDA 版、デバイスに送る | 137.54 | 54.36 | 38.58 | 35.35 | 125.83 | 0.281 |
| `diaz-p222` | CUDA 版、ホストに残す | 166.56 | 測定なし | 測定なし | 28.57 | 118.64 | 0.241 |

- 18 スレッドの交換行列の段は、前の版の 0.24 倍から 0.33 倍になりました。CPU 版の構築 1 回は、Si で 0.47 秒から 0.54 秒、ダイヤモンドで 2.38 秒から 2.51 秒です。
- 1 スレッドの段は、3 番目の段階の 1.002 倍から 1.015 倍でした。1 スレッドでは、経路が前の版と同じです。
- スレッドを 1 から 18 にすると、CPU 版の段は 4.8 倍と 5.5 倍の速さになりました（計算値）。積をホストに残す CUDA 版は 5.1 倍と 5.8 倍、デバイスに送る CUDA 版は 3.2 倍と 3.9 倍です。前の版の CPU 版は、1.49 倍と 1.45 倍でした（3 番目の段階の表）。
- 18 スレッドでは、積をデバイスに送る実行の段が、ホストに残す実行より 1.33 秒と 6.78 秒長くなりました（27% と 24%）。2 つは同じ CUDA 版の実行です。デバイスの積の時間は、2.63 秒と 12.44 秒でした。1 スレッドでは、デバイスに送る実行の方が短いままです（0.80 倍と 0.83 倍）。
- CPU 版と比べると、デバイスに送る CUDA 版の段は、4 スレッドで 1.03 倍と 0.98 倍、9 スレッドで 1.14 倍と 1.09 倍、18 スレッドで 1.23 倍と 1.20 倍でした。2 つの版は BLAS と MPI が違います。
- Si の 18 スレッドを繰り返すと、段は CPU 版で 5.25 秒（1 回目は 5.11 秒）、CUDA 版で 6.38 秒（6.31 秒）でした。
- ほかの形の 8 件も収束しました。18 スレッドのメタン、O2、スピン分極の Si と、2 ランクに 9 スレッドずつの Si です。スピン分極の Si の段は 10.31 秒（CPU 版）と 12.49 秒（CUDA 版）、2 ランクの Si は 3.77 秒と 4.53 秒でした。
- 30 件の全エネルギーは、比べた相手と 6.2e-10 Hartree 以内で合います。相手は、「前の版」の 6 件と、3 番目の段階の 1 スレッドの実行です。差が最大の実行は、メタンの CUDA 版でした。s2p2d1 の 2 つの系の差は、5.0e-11 Hartree 以下です。行列積の回数と最大の積は、全ての実行で前の版と同じでした。
- 測定の間、openmx のほかのプロセスが使った CPU は、平均で 0.46 個分でした（論理 CPU は 36 個。20 秒ごとの記録）。
- デバイスが積を引き受ける条件と既定の動作は、変えていません。18 スレッドのホストでは、今の部品に積を送ると、構築が長くなります（14 節）。

デバイスの上の積を、形ごとに測りました（2026-11-24、`tools/exx_device_shapes.c`、`ca7925b`）。3 番目の段階（9 節）の準備です。RTX 5080 と cuBLAS 13.8.0.4 の組で、構築と同じ形の FP64 の積を、配列をデバイスに置いたまま計算します。下の表は、中心の原子 1 個分の時間です（ミリ秒）。構築を実行した時間ではなく、形ごとに測った時間に回数を掛けた和です。同じ系を 2 回ずつ実行し、合計は 1% 以内で合いました。表は 1 回目の値です。デバイスの結果とホストの結果の照合は、5 回の実行で全て通りました。

| 処理 | Si（`siz-p444` の形） | ダイヤモンド（`diaz-p222` の形） |
|---|---|---|
| 大きい 2 つの積 | 44.5 | 200.3 |
|  $W=V\cdot T$ （V 近傍 99 個と 293 個） | 33.4 | 158.7 |
| 4 つの項（全ての ABF を 1 回の積にまとめる形） | 42.9 | 222.9 |
| 4 つの項（ABF ごとのバッチの積） | 69.7 | 385.4 |
| V 近傍ごとの写しと加算 | 2.4 | 12.1 |
| 合計（4 つの項は、まとめる形） | 123.2 | 594.0 |

- 中心の原子は 2 個なので、構築 1 回分は Si で 0.25 秒、ダイヤモンドで 1.19 秒です（計算値）。18 スレッドのホストの構築 1 回は、0.48 秒から 0.53 秒と、2.34 秒から 2.48 秒でした（上の表の「ホストに残す」の実行）。デバイスの合計は、その約半分です。
- この比較は、同じ範囲の比較ではありません。ホストの値は、 $D$ のブロックの収集と $K$ への加算を含みます。デバイスの合計は、この 2 つと、ホストとの間の転送を含みません。
- 速さは、大きい積で毎秒 3.7e11 から 4.0e11 回の乗算、 $W$ で 2.1e11 から 2.5e11 回でした。まとめる形の 4 つの項は、(ii) と (iii) が 2.5e11 から 3.2e11 回（向きの速い方）、(iv) が 1.1e11 から 1.2e11 回、(i) が 6.2e10 から 6.6e10 回です。(i) は、13 × 13 の結果に長い内側の次元を持つ積で、最も遅い形でした。
- ABF ごとのバッチの積は、まとめる形の 1.6 倍から 1.7 倍の時間でした。歩幅を渡す形（`cublasDgemmStridedBatched`）と、ポインターの配列を渡す形（`cublasDgemmBatched`）の差は、1% 未満です。
- ホストからデバイスへの転送は、毎秒 8 から 10 GiB でした。ページを固定したメモリーからは、12.7 から 12.9 GiB です。デバイスの中の写しは、56 MiB 以上の配列で毎秒 373 から 380 GiB でした。`cudaMalloc` と `cudaFree` の 1 組は、0.06 から 0.8 ミリ秒です。
- 今の部品は、積ごとに配列を転送して確保します。この形のデバイスの積は、同じ計算機で、1 個あたり 66 ミリ秒（Si）と 259 ミリ秒（ダイヤモンド）でした（上の 4 番目の段階の値を、積の数の 40 と 48 で割った計算値）。配列をデバイスに置いたままの積は、22 ミリ秒と 100 ミリ秒です。差は、転送、確保、結果の配列への最初の書き込みに当たります（13 節の 55）。
- この結果から、3 番目の段階は、配列をデバイスに置いたまま V 近傍ごとの計算と大きい積を進める形を、FP64 で設計します。設計と実装は、この改訂にありません。

実機の 5 番目の段階の時間です（2026-11-25、`3d51128`）。この版は、点検の指摘を直した 6 つのコミットと、構築の高速化の 2 番目の段階を含みます。「前の版」は、4 番目の段階でビルドした `1219cb9` の実行ファイルの写しです。前の版と後の版を続けて 1 回ずつ実行し、2 組を測りました。1 組目は前の版を、2 組目は後の版を先に実行しています。値は交換行列の段の秒数で、右端は後の版を前の版で割った比です。

| 系 | 実行 | スレッド | 前の版（1 組目、2 組目） | 後の版（1 組目、2 組目） | 比（1 組目、2 組目） |
|---|---|---|---|---|---|
| `siz-p444` | CPU 版 | 18 | 5.19、5.31 | 4.41、4.59 | 0.850、0.864 |
| `siz-p444` | CUDA 版、デバイスに送る | 18 | 6.17、6.31 | 5.59、5.59 | 0.906、0.886 |
| `siz-p444` | CUDA 版、ホストに残す | 18 | 4.88、5.03 | 4.04、4.03 | 0.828、0.801 |
| `diaz-p222` | CPU 版 | 18 | 28.60、28.84 | 24.23、24.36 | 0.847、0.845 |
| `diaz-p222` | CUDA 版、デバイスに送る | 18 | 35.27、35.46 | 31.28、31.15 | 0.887、0.878 |
| `diaz-p222` | CUDA 版、ホストに残す | 18 | 28.34、28.41 | 23.58、23.78 | 0.832、0.837 |
| `siz-p444` | CPU 版 | 1 | 25.15 | 24.35 | 0.968 |
| `diaz-p222` | CPU 版 | 1 | 161.41、160.21 | 160.14、159.76 | 0.992、0.997 |

- 18 スレッドの段は、12 組の全てで後の版が短く、0.80 倍から 0.91 倍でした。2 番目の段階と、点検の指摘を直した変更を合わせた効果です。2 つを分けては測っていません。
- 積を送る実行の比（0.88 倍から 0.91 倍）は、ほかの実行（0.80 倍から 0.86 倍）より 1 に近い値でした。デバイスの積の時間が、2 つの版でほぼ同じためと考えられます。デバイスの積は、Si が 40 個で 2.66 秒から 2.73 秒、ダイヤモンドが 48 個で 12.38 秒から 12.46 秒です。
- 同じ CUDA 版の 18 スレッドでは、積を送る実行の段が、残す実行より Si で 38% と 39%、ダイヤモンドで 33% と 31% 長くなりました（計算値）。4 番目の段階では、27% と 24% でした。
- 1 スレッドの段は、後の版が 0.968 倍から 0.997 倍でした。1 スレッドの `.out` は、前の版と日付と時間の行だけが違います。
- 2 つの版の全エネルギーの差は、最大 3.6e-11 Hartree です。行列積の回数と最大の積は、全ての実行で前の版と同じでした。ほかの形の 8 件も収束し、4 番目の段階の同じ実行と 1.2e-9 Hartree 以内で合います。
- 大域状態のゲートは、289 個の翻訳単位で通りました。増えた項目はありません。
- 前の版の CUDA 版の写しは、写したライブラリーを `LD_PRELOAD` で読ませて実行しました（13 節の 63）。

`openmx_gpu_task_execute`（上の部品の表）は使いませんでした。この関数は、項目の範囲を受け取るカーネルを、範囲を分けて実行する仕組みです。行列積 1 個は、分ける単位のない 1 回の呼び出しです。既存の GEMM の経路と同じく、容量を確かめてから BLAS を直接呼ぶ形にしました。

標準出力には、計算の終わりに次の 2 行が出ます。`N` はデバイスで計算した積の数、`P` はそのプロセスの数、`T` は 1 つのプロセスがその積に使った時間の最大、`M` は容量の不足でホストに残した積の数です。`N` と `M` は全プロセスの合計です。この 2 行が出るのは、デバイスで計算した積、容量の不足でホストに残した積、デバイスを使うのをやめたプロセスのどれかがある計算です。CPU 版と、デバイスに送る大きさの積がない計算では出ません。改訂 17 からは、`N` と `M` は 1 個ずつデバイスに送った積の数です。呼び出しの全体をデバイスで計算した数は、この節の後半の行に出ます。

```
<EXX>  GPU: N matrix products of the exchange matrix on the device, in P processes
<EXX>  (at most T s in one process); M left to the host for lack of device memory.
```

部品の試験は `tests/run_exx_device_product.py` です。`source/Exact_Exchange_Device.c` を CUDA 版の設定でコンパイルし、SDK のヘッダーの代わりに `tests/fixtures/exx_device_sdk` の模型を使います。模型は、デバイスのメモリーをホストのメモリーで表し、確保した量を数えます。どの呼び出しも失敗させられます。問い合わせと確保の間に、別のプロセスがメモリーを取る場合も作れます。模型は、GEMM の橋渡しの代わりも持ちます。橋渡しの模型は、FP64 で計算した呼び出しを数えます。どの呼び出しを FP64 にするかは、試験が選びます。試験が確かめることは 5 つです。

- デバイスで計算した積が、ホストの積と一致する。
- 引き受けない積では、配列にもデバイスにも触れない。
- 容量が足りない積の後、デバイスのメモリーが全て戻り、次の積を引き受ける。
- そのほかの失敗の後、デバイスのメモリーが全て戻り、エラーが残らず、後の積を引き受けない。失敗させる場所は 11 か所です。
- GEMMul8 を指定した部品は、積を橋渡しに通し、橋渡しが FP64 で計算した積を GEMMul8 の積に数えない。橋渡しのメモリーの不足は、容量が足りない場合と同じ扱いになる。指定を外すと、積は cuBLAS に戻る。

この試験は、GPU の実際の動作を確かめません（13 節の 35）。

呼び出しの全体をデバイスで計算する形です（構築の高速化の 3 番目の段階、改訂 17、`2e2a6da`）。`modules/exx` との約束は 7.2 節に、計算の順序は 6.4 節にあります。ここには、本体の部品 `source/Exact_Exchange_Device.c` の作りを書きます。部品が呼び出しを引き受ける条件は、積の場合と同じ形の 3 つです。

- プロセスに使えるデバイスがあり、それまでにデバイスの呼び出しが失敗していない。
- 呼び出しの全ての積の乗算の回数の和が、下限の 1e9 回以上である。
- 呼び出しの配列が、デバイスの空きから 512 MiB を引いた範囲に入る。

改訂 28 までは、2 番目の条件を、呼び出しの最大の積で判定していました。列をまとめると（6.4 節）、4 プロセスの Si と s2p2d1 のダイヤモンドでは、どの積も 1e9 回に届きません。HSE06 のプロセス 0 の最大の積は、それぞれ 9.5e8 回と 4.1e8 回です（計算値）。利用者の計算機での `51cd508` の確認では、4 プロセスの Si の呼び出しを、デバイスが全て断りました。`079d22c` からは、`modules/exx` が、呼び出しの全ての積の乗算の和を `accept` に渡します。和に入るのは、 $G=T\cdot D$ 、最後の積、組ごとの 4 つの項と、`W` の大きさの積です。`W` の大きさの積は、組ごとに、構築で 1 個、微分で 2 個、半分の形の微分で 3 個あります。下限は 1e9 回のままです。GEMMul8 を通すのも、1 個で 1e9 回以上の積だけです。そのため、上の 2 つの系では、大きい積のない呼び出しをデバイスが計算します。

値は、クラウドのコンテナで、`accept` の値を表示して断る一時的な CPU 版を使って測りました。呼び出し 1 回の和は、2×2×2 のダイヤモンド（C6.0-s2p2d1）で 7.3e9 回でした。4 プロセスの Si では、プロセスごとに 4.2e9 回から 6.0e9 回です。デバイスを使わない系（C5.0-s1p1 のダイヤモンド、メタン、O₂）は、2.7e7 回以下でした。どの値も、HSE06 と PBE0 の両方で測っています。CPU 版は関数表を渡さないので、結果は変わりません。

引き受けると、cuBLAS のハンドルのストリームを、OpenACC のキュー 1 のストリーム（`acc_get_cuda_stream(1)`）に切り替えます。積と、要素の写しと加算のループ（`async(1)`）が同じストリームに載るので、ホストが間で待たなくても、渡した順に実行されます。呼び出しの終わりには、`Exact_Exchange_Device_Call_End` が、ハンドルを元のストリームに戻します。1 個ずつの積は、既定のストリームで写し、デバイスの全体を待つ作りだからです。そのほかの点は、次のとおりです。

- `upload` は、`cudaMemcpy` の後で `cudaDeviceSynchronize` を呼びます。ページ可能なメモリーからの写しは、DMA が終わる前に戻ることがあります（13 節の 64）。後に続く積とループは別のストリームで動くので、待たないと、写し終わる前の値を読むおそれがあります。ライブラリーが写しを頼むのは、仕事が残っていないときだけです。待つ時間は短いと考えられます。
- `finish` は、`acc_wait(1)` の後で `cudaDeviceSynchronize` を呼びます。後の呼び出しは、積とループのエラーを受け取るためのものです。
- `move` のループは、 $(r,c)$ の組を並列に処理します。1 回の `move` の中で同じ要素に書く組はない、という約束（7.2 節）を前提にしています。積を先に作ってから足すため、この単位は `-gpu=nofma` でコンパイルします（13 節の 65）。
- 乗算が下限以上の積は、`scf.gemmul8.enable on` の計算で、GEMM の橋渡しを通ります。橋渡しは、ハンドルのストリームで計算し、作業領域もストリームごとに持ちます（`source/gemmul8_bridge.cu` の `ensure_workspace`）。ほかの積は、FP64 の cuBLAS で計算します。
- 確保が容量の不足（`OPENMX_GPU_OUT_OF_MEMORY`）を返した呼び出しと、積が `OPENMX_BLAS_ALLOC_FAILED` を返した呼び出しは、容量の不足として数えます。ホストがその呼び出しを計算し直し、デバイスは使い続けます。ほかの失敗では、デバイスを使うのをやめ、後の呼び出しも積もホストが計算します。記録するのは最初の失敗です。失敗の種類には、`STREAM`（ストリームの取得と切り替え）と `WAIT`（`finish`）を加えました。
- 呼び出しは、`Exact_Exchange_Device_Call_End` が数えます。デバイスの分に数えるのは、引き受けた後に失敗も容量の不足もなく、結果が戻った呼び出しだけです。

デバイスが計算した呼び出しがある計算では、計算の終わりに次の 2 行が出ます。`C` はデバイスで計算した呼び出しの数、`P` はそのプロセスの数、`N` はその呼び出しの行列積の数、`L` はそのうち乗算が下限以上の積の数です。`T` は、1 つのプロセスがその呼び出しに使った時間の最大です。時間は、`modules/exx` の入口から戻るまでで測ります。`M` は、容量の不足でホストに残した呼び出しの数です。数は全プロセスの合計です。`C` と `M` のどちらかが 0 でない計算で出ます。

```
<EXX>  GPU: C calls of the exchange build on the device, in P processes, with N matrix products, L of them large
<EXX>  (at most T s in one process); M calls left to the host for lack of device memory.
```

`scf.gemmul8.enable on` の計算では、次の行が加わります。`G` は、GEMMul8 が計算した積の数です。`F` は、呼び出しの残りの積の数（ $N-G$ ）です。`level.of.stdout 2` では、構築ごとに、ランク 0 の呼び出しの数、時間、積の数、1 回の呼び出しのメモリーの最大、ホストに残した呼び出しの数が出ます（2 つ目の行）。

```
<EXX>  GPU: GEMMul8 computed G of the L large matrix products of these calls, FP64 the other F products (scf.gemmul8.enable on).
<EXX>  device calls of process 0: C in T s with N matrix products, at most B MB for one call; left to the host for lack of device memory: M
```

部品の試験（`tests/run_exx_device_product.py`）には、呼び出しの全体の場合を加えました。定義 `OPENMX_EXACT_EXCHANGE_DEVICE_HOST_MOVES` を付けてコンパイルすると、OpenACC のループと同じループがホストで動きます。模型は、キューのストリーム、`acc_wait`、ハンドルのストリームの取得と切り替えも表します。積がキュー 1 のストリームで実行されたかを記録し、仕事が残っている間の `cudaMemcpy` と `cudaFree` を誤りとして記録します。試験は、ライブラリーと同じ順に関数を呼び、次の 5 つを確かめます。

- 呼び出しの積と、写しと加算の結果が、ホストの計算とビットまで同じになる。
- 引き受けない呼び出しでは、配列にもデバイスにも触れない。
- 容量が足りない呼び出しの後、メモリーが全て戻り、ハンドルが元のストリームに戻り、次の呼び出しを引き受ける。
- ほかの失敗の後は、後の呼び出しも積も引き受けない。失敗させる場所は 16 か所で、2 か所が続けて失敗する場合に最初の失敗が残ることも確かめる。
- GEMMul8 の指定では、大きい積だけが橋渡しを通る。

部品を 1 か所ずつ壊した 11 個の変種のうち、10 個がこの試験に落ちました。落ちなかった 1 個は、加算を `target[t] = target[t]+alpha*source[s]` と書いた変種で、計算は元と同じです。この試験も、GPU の実際の動作は確かめません。ストリームとキューの関係は、利用者の計算機で確かめます（14 節）。

実機の 6 番目の段階の時間です（2026-11-25、`09e054a`）。「前の版」は、5 番目の段階でビルドした `3d51128` の実行ファイルの写しです。前の版と後の版を続けて 1 回ずつ実行し、これを 1 組としました。18 スレッドでは 2 組、GEMMul8（`scf.gemmul8.enable on`）と 1 スレッドでは 1 組を測っています。1 組目は前の版を、2 組目は後の版を先に実行しました。値は交換行列の段の秒数で、右端は後の版を前の版で割った比です。「デバイスで計算」は、前の版では大きい積を 1 個ずつ送る実行、後の版では呼び出しの全体をデバイスで計算する実行です。

| 系 | 実行 | スレッド | 前の版（1 組目、2 組目） | 後の版（1 組目、2 組目） | 比（1 組目、2 組目） |
|---|---|---|---|---|---|
| `siz-p444` | CPU 版 | 18 | 4.38、4.31 | 4.28、4.35 | 0.977、1.009 |
| `siz-p444` | CUDA 版、デバイスで計算 | 18 | 5.46、5.54 | 3.26、3.28 | 0.597、0.592 |
| `siz-p444` | CUDA 版、ホストに残す | 18 | 4.01、4.02 | 4.13、4.01 | 1.030、0.998 |
| `siz-p444` | CUDA 版、GEMMul8 | 18 | 4.60 | 2.51 | 0.546 |
| `diaz-p222` | CPU 版 | 18 | 24.24、24.21 | 24.17、24.08 | 0.997、0.995 |
| `diaz-p222` | CUDA 版、デバイスで計算 | 18 | 30.98、30.90 | 19.49、19.49 | 0.629、0.631 |
| `diaz-p222` | CUDA 版、ホストに残す | 18 | 23.74、23.63 | 23.67、23.75 | 0.997、1.005 |
| `diaz-p222` | CUDA 版、GEMMul8 | 18 | 26.40 | 15.30 | 0.580 |
| `siz-p444` | CPU 版 | 1 | 24.47 | 24.65 | 1.007 |
| `siz-p444` | CUDA 版、デバイスで計算 | 1 | 20.03 | 3.27 | 0.163 |
| `diaz-p222` | CPU 版 | 1 | 159.44 | 159.51 | 1.000 |
| `diaz-p222` | CUDA 版、デバイスで計算 | 1 | 135.02 | 19.39 | 0.144 |

- デバイスで計算する実行の段は、18 スレッドで前の版の 0.59 倍から 0.63 倍、1 スレッドで 0.14 倍と 0.16 倍になりました。CPU 版とホストに残す実行は 0.98 倍から 1.03 倍で、変わっていません。
- 18 スレッドでは、後の版のデバイスの段が、同じ組のホストに残す実行より、Si で 21% と 18%、ダイヤモンドで 2 組とも 18% 短い値でした。CPU 版と比べると、Si で 24% と 25%、ダイヤモンドで 2 組とも 19% 短い値です（どれも計算値）。前の版では、デバイスに送る実行が、残す実行より 31% から 38% 長い値でした。
- 後の版のデバイスの段は、1 スレッドと 18 スレッドでほぼ同じでした（Si で 3.26 秒から 3.28 秒、ダイヤモンドで 19.39 秒から 19.49 秒）。構築 1 回の時間のほぼ全てが、デバイスの呼び出しの時間です。呼び出し 1 回は、Si で 0.31 秒から 0.40 秒、ダイヤモンドで 1.58 秒から 1.74 秒でした。デバイスで計算するどの実行でも、最初の構築が最も長い値です。
- 呼び出し 1 回の積は、Si で 994 個、ダイヤモンドで 2,934 個です。そのうち乗算が下限以上の積は、中心の原子ごとの 2 個（呼び出しごとに 4 個）だけでした。1 回の呼び出しのメモリーは、出力の行で最大 261.3 MB と 816.9 MB です。Si の実行では、GPU のメモリーの使用量が、実行の前の 277 MiB から最大 897 MiB になりました（nvidia-smi の 1 秒ごとの記録）。
- 形ごとの測定の値から見積もった積の時間は、中心の原子 2 個分で、Si が 0.24 秒、ダイヤモンドが 1.18 秒です。2 回目からの呼び出し 1 回の時間は、その 1.3 倍から 1.4 倍でした（計算値）。差には、写しと加算のループ、複写、待ちの時間が入ると考えられます。分けては測っていません。
- GEMMul8 の実行の段は、FP64 でデバイスで計算する実行の 0.77 倍（Si）と 0.79 倍（ダイヤモンド）でした（計算値）。GEMMul8 を通ったのは、呼び出しの中の大きい積（Si で 40 個、ダイヤモンドで 48 個）だけです。
- DFT 全体の時間は、デバイスで計算する実行で、18 スレッドの Si が前の版の 0.91 倍と 0.92 倍、ダイヤモンドが 0.76 倍と 0.77 倍になりました。1 スレッドでは、Si が 0.61 倍、ダイヤモンドが 0.24 倍です。
- 2 つの版の全エネルギーの差は、最大 1.9e-11 Hartree です（`diaz-p222` の CPU 版の 2 組目）。1 スレッドの CPU 版の `.out` は、前の版と、日付と時間の行だけが違います。
- 測定の間、openmx のほかのプロセスが使った CPU は、平均で 0.37 個分でした（論理 CPU は 36 個。20 秒ごとの記録）。

同じ段階の受け入れ表（53 件、1 スレッド）から、デバイスを使う 4 つの系の交換行列の段を抜き出しました（秒）。`dense` はデバイスの密行列ソルバーの経路、`g8` は GEMMul8 の実行です。どちらも、構築の呼び出しはデバイスで計算します。

| 系 | プロセス | CPU 版 | CUDA 版、デバイスで計算 | CUDA 版、ホストに残す | `dense` | `g8` |
|---|---|---|---|---|---|---|
| `siz-p444` | 1 | 24.80 | 3.27 | 25.19 | 3.25 | 2.51 |
| `siz-p444-spin` | 1 | 49.52 | 6.46 | 50.75 | 6.44 | 4.87 |
| `siz-p444-np2` | 2 | 12.99 | 3.31 | 13.21 | 3.29 | 2.52 |
| `diaz-p222` | 1 | 161.51 | 19.39 | 165.58 | 19.53 | 15.24 |

- デバイスで計算する実行では、段の時間の 99% 以上が、デバイスの呼び出しの時間でした（Si で 3.27 秒のうち 3.23 秒、ダイヤモンドで 19.39 秒のうち 19.28 秒）。CPU 版との比は、1 プロセスの 3 つの系で 0.12 倍から 0.13 倍です（計算値）。
- スピン分極のある Si は、呼び出しが 2 倍（20 回）で、段の時間も約 2 倍でした。
- 2 プロセスが 1 台を共有する `siz-p444-np2` では、段の時間が 1 プロセスの場合とほぼ同じでした（3.31 秒と 3.27 秒）。CPU 版は 2 プロセスで約半分になるので、CPU 版との比は 0.25 倍です（計算値）。2 つのプロセスの呼び出しは、デバイスの上で並行して進んでいないと考えられます。確かめていません。
- 呼び出しの数と大きい積の数は、4 つの系とも設計の数と合いました（Si で 10 回と 40 個、スピン分極で 20 回と 80 個、2 プロセスで 20 回と 40 個、ダイヤモンドで 12 回と 48 個）。障害注入の実行では、全ての呼び出しと大きい積がホストに残りました。

M8 の実機の確認（2026-11-25、`9e01d40`）では、同じ表を実行し直し、4 プロセスの `siz-p444-np4` を加えました。上の 4 つの系のデバイスで計算する実行は、段の時間が上の表と 0.15 秒以内で同じです。4 プロセスが 1 台を共有する `siz-p444-np4` の段は 4.00 秒で、1 プロセスの 3.27 秒より長い値でした。CPU 版（7.83 秒）との比は、0.51 倍です（計算値）。呼び出し 1 回のメモリーは 186.2 MB で、1 プロセスの 261.3 MB より小さくなりました。受け持つ ABF が、原子の約半分に減るためと考えられます。PBE0 の系は、V 近傍が遠くまで届くので、積が大きくなります。呼び出し 1 回のメモリーは、Si で 880.0 MB、s2p2d1 のダイヤモンドで 1246.9 MB でした。4 プロセスの Si では、630.0 MB です。デバイスで計算する実行の段は、`siz-p444-pbe0` が 23.91 秒、`diaz-p222-pbe0` が 28.47 秒です。CPU 版の 188.98 秒と 237.50 秒に対して、0.13 倍と 0.12 倍になりました（計算値）。4 プロセスの `siz-p444-pbe0-np4` の段は 28.78 秒で、HSE06 と同じく 1 プロセスより長い値です。GEMMul8 の実行のデバイスの時間は、FP64 の実行の 0.82 倍から 0.89 倍でした（別の計算と重なった実行を除く。計算値）。

改訂 25 で、力の重みもデバイスで計算する形にしました（`d6f7db1`、`df30625`）。入り口と結合層の形は、7.4 節にあります。重みの呼び出しを引き受ける条件と確保の手順は、構築の呼び出しと同じです。デバイスに置くのは、密度行列と全ての原子の T ブロックです。中心の原子 1 個（またはその一部）の計算は、構築の配列を使い回します。V 近傍ごとに、密度行列との 2 つの積で中間の行列を作り、 $T_B$ との積で V ブロックの重みにします。T ブロックの重みは、全ての V 近傍の項を足した括弧と、密度行列の積です。部分ごとに、この 2 つを 2 回の転送でホストに戻します。改訂 26 の半分の形では、組ごとに $V^{\mathsf T}H$ の積を 1 回加えます。結果は、デバイスの上で、組の原子の T ブロックの重みに足す形です。この重みは、部分ごとの 3 回目の転送で戻ります。構築の `device_centre` も同じ補助関数で書き直しましたが、デバイスの操作の順は前と同じです。

`level.of.stdout 2` では、プロセス 0 が力の行の後に次の行を表示します。1 行目は、重みの時間です。2 行目は呼び出しの全体を、3 行目は 1 個ずつの大きい積を、デバイスが引き受けたか断ったときに出ます。最後の数は、容量が足りずにホストに残した数です。

```
<EXX>  of that, T s for the weights of the T and V blocks on process 0
<EXX>  the weights on the device of process 0: C calls in T s with N matrix products, at most B MB for one call; left to the host for lack of device memory: M
<EXX>  matrix products of the weights on the device of process 0: N in T s, at most B MB for one product; left to the host for lack of device memory: M
```

利用者の計算機の `4577361` の確認では、`gpu` の実行の重みの時間が、前の確認（`9e01d40`）の 0.106 倍から 0.567 倍になりました（計算値）。`siz-p444-pbe0-gpu` の重みは、24.56 秒から 3.30 秒です。呼び出しは 1 回で、10,192 個の積を含み、メモリーは 920.8 MB でした。`diaz-p222-pbe0-gpu` の重みは、37.82 秒から 4.01 秒になりました。呼び出し 1 回のメモリーは 1296.8 MB で、この確認で最も大きい値です。4 プロセスが 1 台を共有する `siz-p444-pbe0-np4-gpu` の比は、0.535 倍でした。重みの後に残る力の時間は、`siz-p444-pbe0-gpu` で 2.0 秒、`diaz-p222-pbe0-gpu` で 1.7 秒です（計算値）。ほとんどが、ホストで回る対の段と考えられます。この段は二中心積分の微分を使うので、`64352d9` の球ベッセル関数で短くなる見込みです（9 節）。実機では、まだ測っていません。

同じ改訂で、CUDA 版の XC の GPU 経路が、ハイブリッド汎関数の準局所部分も計算する形にしました（`3fafc96`。8 節）。前の版の CUDA 版は、ハイブリッド汎関数の計算では、この部分をホストのループで計算していました。デバイスのループは、`modules/xc` の C ヘッダー `hybrid_kernels.h` のインライン関数を呼びます。ホストのループが、ライブラリーを通して呼ぶ関数と同じです。ただし、デバイスの値はホストの値と末尾の桁で違うことがあります。`Set_XC_Grid.c` は `-gpu=nofma` なしでコンパイルされ（`source/Makefile`）、超越関数はデバイスの数学ライブラリーのものだからです。GGA-PBE のデバイスのループも、前から同じ条件でした。`-gpu=nofma` を付けるのは、`Exact_Exchange_Device.c` と `Set_Orbitals_Grid.c` です。ハイブリッド汎関数の差の大きさは、利用者の計算機での確認を待っています（14 節）。`OPENMX_XC_GRID_GPU=0` を付けると、同じ実行ファイルがホストのループで計算します。

デバイスの密行列ソルバーの経路です。M6 の実機の段階で書き、実機の 2 番目と 3 番目の段階で試験しました（9 節の M6）。加算の手順は 6.5 節、収集は 6.3 節に書きました。ここには、部品の選び方を書きます。

- 加算は、OpenACC の 1 つのループです。上の部品の表の `openmx_gpu_transform` は使いませんでした。この関数は、協調する GPU が 2 台未満だと 0 を返します（`source/openmx_gpu_transform.cu` の `w.workers.count<2`）。GPU が 1 台の計算機では、併せて書くループだけが動きます。GPU が複数ある計算機では、加算 1 回のために行列を配る転送が増えます。
- $K(\mathbf k)$ はホストで作ります。式 (14) のフーリエ和は、M5 の関数をそのまま使います。デバイスに写す配列は $n\times n$ で、加算のたびに確保して解放します。
- 固有値ソルバーは FP64 です。FP32 で解いて精製する方式には入りません（6.3 節）。
- 固有値ソルバーの中の行列積（基底の変換）は、既存の橋渡し（`openmx_gemmul8Dgemm`、`openmx_gemmul8Zgemm`）を通ります。入力に `scf.gemmul8.enable` がないとき、橋渡しはデバイスで方式を選びます。FP32 と FP64 の処理量の比が 8 未満のデバイスでは FP64 の cuBLAS、それ以外では GEMMul8 です（`source/gemmul8_bridge.cu` の `native_fp64_device_default`）。ハイブリッド汎関数のための切り替えは加えていません。1.2 節は、GEMMul8 を FP64 との差を測ってから使うと決めています。この経路では、既存のソルバーの既定の動作を変えずに、差を測る形にしました。道具の `NAME-dfp64` が、`OPENMX_GEMMUL8_DISABLE=1` を付けた実行です。実機の 2 番目の段階では、既定の実行と FP64 の実行の全エネルギーの差が、メタンで 7.2e-11 Hartree、Si で 0 でした。1e-7 Hartree より 3 桁小さいので、ハイブリッド汎関数のための切り替えは加えていません。

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

ライブラリーの中のループでは、1 行目が 0.04 秒、2 行目が 33 秒でした。2 行目から求めた DGEMM の速度は約 18 GFLOP/s です。セルが近傍の範囲より小さい系では、Γ 点の形式の時間が周期像のまとめ方で 2 桁変わります。site の形式は周期像をまとめられないので、k 点の計算（M5）ではこの時間が出発点になります。改訂 27 で、正則なメッシュの site の形式も、V 近傍をメッシュの周期でまとめるようになりました（6.4 節）。改訂 28 では、構築の列もまとめます。

M4 の実装の実測値です。条件は、Γ 点の SCF（Cluster ソルバー、CPU 版）、既定のしきい値と打ち切り、クラウドのコンテナ（4 コア）です。ほかの計算を止めて、1 ランクで測りました。OpenBLAS は 2 スレッド、OpenMP は 1 スレッドです（13 節の 46）。行列積は、本体の `DGEMM`（OpenBLAS 0.3.26）です。「全体」は、本体が表示する `DFT in total` です。

| 系 | 軌道、ABF | C 近傍、V 近傍の site | 準備 | 構築 1 回 | 構築の回数 | 全体 |
|---|---|---|---|---|---|---|
| メタン（s2p2d1、1 辺 10 Å のセル） | 33、129 | 25、25 | 1.9 秒 | 0.01 秒未満 | 6 | 13 秒 |
| 水（s2p2d1、同じセル） | 23、102 | 9、9 | 1.6 秒 | 0.01 秒未満 | 6 | 17 秒 |
| ダイヤモンド 8 原子（C5.0-s1p1） | 32、80 | 984、2,200 | 2.8 秒 | 0.01 秒未満 | 6 | 19 秒 |
| ダイヤモンド 8 原子（C6.0-s2p2d1） | 104、552 | 1,336、2,344 | 6.4 秒 | 0.26 から 0.36 秒 | 7 | 194 秒 |

s2p2d1 のダイヤモンドでは、厳密交換の時間は準備と 7 回の構築を合わせて 8.6 秒で、全体の 4% です。残りは既存のコードの時間です。力の計算が 81 秒、`Set_Hamiltonian` が 43 秒、`Set_Density_Grid` が 35 秒、`Set_ProExpn_VNA` が 17 秒で、対角化は 2.3 秒でした。密度行列の収集と $K$ の加算は、どの系でも合計 0.005 秒未満です。4 ランク（1 スレッド）では、準備が 6.6 秒、構築が 1 回 0.07 から 0.09 秒、全体が 59 秒でした（ランク 0 で測った経過時間）。準備は、全てのランクが構造の全体について行うので、ランクを増やしても短くなりません。8 原子のセルでは、上の表の仮定（Si512）の時間とメモリーは確かめられません。

M5 の実装の実測値です。条件は、k 点の SCF（Band ソルバー、CPU 版、`f2d3c9e` の実行ファイル）、既定のしきい値と打ち切り、クラウドのコンテナ（4 コア）です。ほかの計算を止めて、1 ランクで測りました。OpenBLAS は 2 スレッド、OpenMP は 1 スレッドです（13 節の 46）。系は基本単位胞（2 原子）で、メッシュは Γ 点を含む 4×4×4（解く k 点は 36 個）と 8×8×8（260 個）です。「加算」は、 $K(\mathbf k)$ を作って `Hs` に足す時間の合計です。行列積の回数と flop 数は、数える処理を加えた一時的な実行ファイルで測りました。この段落と下の箇条書きの MB は 1,048,576 バイトです（本体と道具の表示に合わせました）。

| 系 | 軌道、ABF | C 近傍、V 近傍の site | 行列積の回数 | flop 数（構築 1 回） | メッシュ | 構築 1 回 | 構築の回数 | 加算 | 全体 | メモリーの最大 |
|---|---|---|---|---|---|---|---|---|---|---|
| ダイヤモンド（C5.0-s1p1） | 8、20 | 246、550 | 2,754 | 1.4e9 | 4×4×4 | 0.22 秒 | 7 | 0.8 秒 | 10 秒 | 137 MB |
| 同上 | 同上 | 同上 | 同上 | 同上 | 8×8×8 | 0.21 秒 | 7 | 5.7 秒 | 23 秒 | 138 MB |
| ダイヤモンド（C6.0-s2p2d1） | 26、138 | 334、586 | 2,934 | 5.9e11 | 4×4×4 | 32.3 秒 | 8 | 8.3 秒 | 324 秒 | 1,361 MB |
| 同上 | 同上 | 同上 | 同上 | 同上 | 8×8×8 | 32.3 秒 | 8 | 63.1 秒 | 423 秒 | 1,361 MB |
| Si（Si7.0-s2p2d1） | 26、138 | 174、198 | 994 | 1.2e11 | 4×4×4 | 5.5 秒 | 10 | 4.2 秒 | 101 秒 | 546 MB |
| 同上 | 同上 | 同上 | 同上 | 同上 | 8×8×8 | 5.6 秒 | 10 | 31.9 秒 | 168 秒 | 546 MB |

- 構築の時間とメモリーは、k 点の数に依りません。k 点の数に比例するのは、密度行列の収集と $K(\mathbf k)$ の加算です。改訂 27 からは正則なメッシュで V 近傍をまとめ、改訂 28 からは構築の列もまとめます。そのため、粗いメッシュほど構築が軽くなります（6.4 節）。
- 加算は、SCF の 1 ステップ、k 点 1 個あたり、s2p2d1 のダイヤモンドで 3.7 ミリ秒、Si で 1.6 ミリ秒でした。8×8×8 のメッシュでは、全体の 15% と 19% を占めます。加算は、SCF のステップごとに $K(\mathbf R)$ のフーリエ和を計算し直します。 $K(\mathbf k)$ は保存していません。
- Si の flop 数は、中心の原子 1 個あたり 5.8e10 です。この節の最初の表の積和の回数（1.3e10）を flop 数に直して 2 倍し、ABF の個数の違いで 2.2 倍すると 5.7e10 になります（計算値）。2 つの値は合っています。この計算値は $s=9477$ のままです。実測の $s$ は 8,229 で、 $s$ に比例する項は 13% 小さくなります（13 節の 37）。M6 で数えた乗算の回数は、中心の原子 1 個あたり 2.9e10 回で、上の flop 数の半分です（11 節）。
- 構築 1 回のうち DGEMM の中の時間は、s2p2d1 のダイヤモンドで 19.2 から 20.2 秒、Si で 3.3 から 3.5 秒でした。構築の 62% と 63% です。DGEMM の速度は、30 GFLOP/s と 34 GFLOP/s です（2 スレッド）。残りの時間の内訳は、M6 で測りました（この節の終わり）。
- 中心の原子 1 個の作業領域は、s2p2d1 のダイヤモンドで 944 MB です（M6 で数えた行列の大きさからの計算値）。 $D$ の部分行列と、最後の行列積の結果が各 334 MB、 $G$ と $R$ が各 138 MB です。行数は 2,171 と 897、列数は $s=20{,}163$ です。Si では、各 71 MB と各 56 MB で、合計 255 MB です（ $s=8{,}229$ ）。この節の最初の表の作業領域には、最後の行列積の結果が入っていません。改訂 8 までは、列数を 24,843 と 9,477 として、1,163 MB と 293 MB と書いていました。数え方の違いは 13 節の 37 にあります。
- 同じ系の PBE の実行は、メモリーの最大がダイヤモンドで 357 MB、Si で 321 MB でした。
- 2×2×2 のメッシュは、16 原子のスーパーセルの Γ 点と同じ計算です。構築 1 回の flop 数は、site の形式（基本単位胞）が Γ 点の形式（スーパーセル）の 30 倍（C5.0-s1p1）と 42 倍（C6.0-s2p2d1）でした。Γ 点の形式の値は 4.5e7 と 1.4e10 です。C6.0-s2p2d1 の構築 1 回の時間は、33.6 秒と 1.5 秒です（13 節の 30）。

M6 で、構築の時間の内訳を測り、V 近傍ごとの項を書き直しました（6.4 節）。条件は、クラウドのコンテナ（4 コア、AVX-512）、CPU 版、1 ランクです。スレッドの数は、本体の `-nt` と OpenBLAS に同じ値を渡しました。ほかの計算を止めて、1 件ずつ測っています。「前」は `a7fbfb8` の実行ファイル、「後」は書き直した版です。「全体」は、本体が表示する `DFT in total` の時間です。この節の M5 の表の値は、書き直す前の版のものです。

| 系 | スレッド | 構築 1 回（前） | 構築 1 回（後） | 後と前の比 | 交換行列の段（前） | 交換行列の段（後） | 全体（前） | 全体（後） |
|---|---|---|---|---|---|---|---|---|
| Si（`siz-p444`、構築 10 回） | 1 | 4.41 秒 | 2.88 秒 | 0.65 | 44.1 秒 | 28.8 秒 | 87.0 秒 | 70.6 秒 |
| 同上 | 4 | 2.78 秒 | 1.76 秒 | 0.63 | 27.8 秒 | 17.6 秒 | 75.2 秒 | 65.8 秒 |
| ダイヤモンド（`diaz-p222`、C6.0-s2p2d1、構築 12 回） | 1 | 26.49 秒 | 15.07 秒 | 0.57 | 317.9 秒 | 180.9 秒 | 383.1 秒 | 244.8 秒 |
| 同上 | 4 | 16.86 秒 | 9.67 秒 | 0.57 | 202.3 秒 | 116.0 秒 | 252.3 秒 | 165.6 秒 |

構築 1 回の内訳です（秒。全ての構築の平均）。時間を測る処理を加えた一時的な実行ファイルで測りました。Si は 1 スレッドと 4 スレッド、ダイヤモンドは 1 スレッドです。

| 処理 | Si、1 スレッド（前） | Si、1 スレッド（後） | Si、4 スレッド（前） | Si、4 スレッド（後） | ダイヤモンド、1 スレッド（前） | ダイヤモンド、1 スレッド（後） |
|---|---|---|---|---|---|---|
| 大きい積（中心の原子ごとに 2 個） | 0.94 | 1.03 | 0.26 | 0.29 | 4.44 | 4.81 |
|  $W=V\cdot T$  | 0.58 | 0.55 | 0.20 | 0.19 | 3.48 | 3.33 |
| 4 つの項の積 | 1.23 | 0.69 | 0.71 | 0.68 | 8.23 | 4.00 |
| 配列を写す処理（列の収集、結果の加算、添字の入れ替え） | 1.33 | 0.45 | 1.30 | 0.39 | 8.91 | 2.72 |
| メモリーの確保と 0 の書き込み | 0.34 | 0.08 | 0.34 | 0.08 | 2.26 | 0.21 |
|  $D$ の行の収集、 $K$ への加算など | 0.13 | 0.13 | 0.13 | 0.13 | 0.68 | 0.66 |
| 合計 | 4.55 | 2.93 | 2.93 | 1.74 | 28.00 | 15.73 |

- 書き直す前の Si の 1 スレッドでは、デバイスに送る大きい積は構築の 21% でした。乗算の回数では 58% です。4 つの項の積と配列を写す処理が、合わせて 56% を占めます。ダイヤモンドでは、大きい積が 16%、4 つの項の積と配列を写す処理が 61% でした。
- 書き直す前の 4 スレッドでは、配列を写す処理とメモリーの確保が 56% を占めました。この 2 つの時間は、1 スレッドのときと変わりません。BLAS のスレッドが速くするのは、積の中だけです。
- 配列を写す処理は、Si で 1.33 秒から 0.45 秒、ダイヤモンドで 8.91 秒から 2.72 秒になりました。書き直す前は、V 近傍ごとに `W` の添字を入れ替え、`G` の全体から $\mathrm{loc}(B)$ の列を集め、(iv) の結果を `Rm` の全体に足していました。どの処理も、Si で約 100 万個、ダイヤモンドで約 195 万個の要素を 1 個ずつ動かします（計算値）。
- 4 つの項の積は、乗算の回数が同じまま、Si で 1.23 秒から 0.69 秒、ダイヤモンドで 8.23 秒から 4.00 秒になりました。
- 書き直した後の 4 つの項の積は、スレッドを 1 から 4 にしても時間が変わりません（Si で 0.69 秒と 0.68 秒）。1 個の乗算は 1.9e5 回です。
- 書き直した後に残る配列を写す処理は、(i) のための列の写しと、(iv) の結果の加算です。Si で 0.15 秒と 0.23 秒、ダイヤモンドで 0.99 秒と 1.49 秒でした。どちらも、V 近傍と ABF の組ごとに、Si で 115 KiB、ダイヤモンドで 220 KiB を読み書きします（計算値）。
- 内訳を測った実行ファイルの合計は、測る処理のない実行ファイルの値（上の表）と 6% 以内で合います。
- メモリーの最大は、Si で 546 MB から 524 MB、s2p2d1 のダイヤモンドで 1,361 MB から 1,319 MB になりました（M5 の道具の `siz-p444-hse` と `diaz-p222-hse`）。書き直す前の版は、V 近傍ごとの計算に、`W` と同じ大きさの配列をほかに 3 個使っていました。1 個は Si で 7.7 MB、ダイヤモンドで 14.9 MB です（計算値）。
- スレッドで速くならない部分を減らす方法は、実装していません（14 節）。ABF ごとの計算は互いに独立で、ABF ごとに `Rm` の別の部分へ足します。
- 利用者の計算機（18 コア）でも、書き直す前と後の版を測りました（実機の 2 番目と 3 番目の段階。11 節の表）。交換行列の段は、1 スレッドで 0.50 倍から 0.56 倍、18 スレッドで 0.57 倍から 0.62 倍になりました。CPU 版の段は、スレッドを 1 から 18 にしても 0.67 倍と 0.69 倍にとどまります。

改訂 14 で、構築をスレッド並列にしました（6.4 節、8 節）。条件は上の表と同じです。クラウドのコンテナ（4 コア）の CPU 版を、1 ランクで測りました。本体の `-nt` と OpenBLAS に同じ数を渡し、ほかの計算を止めて 1 件ずつ実行しています。「前」は `b8a1363` の実行ファイル、「後」はスレッド並列の版です。構築 1 回の時間は、全ての構築の平均です。「全体」は、`.out` の `Elapsed.Time.` です。

| 系 | スレッド | 構築 1 回（前） | 構築 1 回（後） | 後と前の比 | 構築の合計（前） | 構築の合計（後） | 全体（前） | 全体（後） |
|---|---|---|---|---|---|---|---|---|
| Si（`siz-p444`、構築 10 回） | 1 | 2.98 秒 | 3.02 秒 | 1.01 | 29.8 秒 | 30.2 秒 | 75.1 秒 | 75.5 秒 |
| 同上 | 4 | 1.90 秒 | 1.16 秒 | 0.61 | 19.0 秒 | 11.6 秒 | 71.4 秒 | 64.8 秒 |
| ダイヤモンド（`diaz-p222`、構築 12 回） | 4 | 10.74 秒 | 5.91 秒 | 0.55 | 128.9 秒 | 70.9 秒 | 178.6 秒 | 124.0 秒 |

- 1 スレッドでは、経路が前の版と同じです。全エネルギーと、構築ごとに表示する $K$ の非対称の値は、前の版と全ての桁で一致しました。
- 構築 1 回の行列積の回数（Si で 54,850 回、ダイヤモンドで 162,326 回）と最大の積は、スレッドの数によらず、前の版と同じです。
- 4 スレッドの Si の「後」は、部品に分ける前の同じ計算の版でも 2 回測っています。構築 1 回の平均は 1.02 秒と 1.04 秒でした。
- 「前」の 4 スレッドの値は、上の M6 の表の「後」（1.76 秒と 9.67 秒）と同じコードの時間です。今回の値は 8% と 11% 長く、測った時刻が違います。ダイヤモンドの比は、続けて測った「前」と「後」から出しました。Si の「前」は、「後」の 19 分前（4 スレッド）と 22 分前（1 スレッド）に測っています。
- 4 スレッドの構築 1 回は、1 スレッドの 0.38 倍（Si）です。4 コアを全て使えた場合の 0.25 倍には届きません。理由は 2 つあります（下の内訳の表）。スレッドに分けた処理が、1 スレッドの 0.30 倍から 0.35 倍にとどまること。スレッドに分けていない処理が残っていること。0.25 倍との差のうち、前者が約 3 分の 2、後者が約 3 分の 1 です（計算値）。前者の原因は、測っていません。
- スレッドを使う 9 件の実行で、前の版と結果を比べました。`tools/exx_gpu_checks.py` の `-cpu` の実行で、分子（メタン、O2）、スピン分極、複数のランクを含みます。4 スレッドの 1 ランクが 4 件、2 スレッドの 2 ランクまたは 4 ランクが 5 件です。全て収束し、構築の回数も同じでした。全エネルギーの差は最大 1.2e-9 Hartree（O2 の 2 ランク）、交換エネルギーの差は 1e-12 Hartree 以下です。分子の全エネルギーは、同じ実行ファイルでも実行ごとに変わります（13 節の 58）。1 ランクの 4 スレッドでは、幅が 2e-10 から 7e-10 Hartree でした。
- O2 の 2 ランクの実行は、3 つの実行ファイルで 4 回ずつ繰り返しました（前の版、後の版、点検の指摘を直した版）。前の版の 4 回は、1e-12 Hartree 以内でそろいました。後の版と直した版は、どちらも 1.2e-9 Hartree の幅に分かれ、値の 1 つは前の版の値です。違いは、最後の内側の SCF のステップ数にあります。前の版の 4 回は 10 または 11 ステップ、後の 2 つの版は 6 ステップ（6 回）または 10 ステップ（2 回）でした。10 ステップの 2 回の全エネルギーは、前の版と 1e-12 Hartree 以内で合います。6 ステップの 6 回は、前の版より 7.5e-10 または 1.2e-9 Hartree 高い値でした。版の間の差（1.2e-9 Hartree）は、同じ実行ファイルの実行の間の幅と同じ大きさです。

スレッド並列の版の構築 1 回の内訳です（秒。全ての構築の平均）。時間を測る処理を加えた一時的な実行ファイルで測りました。条件は上の表と同じです。ただし、同じコンテナでコードの点検の作業が動いていました。

| 処理 | Si、1 スレッド | Si、4 スレッド | ダイヤモンド、1 スレッド | ダイヤモンド、4 スレッド |
|---|---|---|---|---|
| 大きい積（中心の原子ごとに 2 個） | 1.10 | 0.37 | 5.08 | 1.66 |
|  $W=V\cdot T$  | 0.59 | 0.21 | 3.57 | 1.14 |
| ABF の区間のループ（列の写し、4 つの項の積、結果の加算） | 1.19 | 0.35 | 6.92 | 2.22 |
| 添字の入れ替えと 0 の書き込み（区間のループ） | 0.11 | 0.03 | 0.31 | 0.10 |
|  $D$ のブロックの収集（中心の原子ごとの $D[\mathrm{loc}(A),S_A]$ ） | 0.10 | 0.10 | 0.51 | 0.52 |
|  $K$ への加算 | 0.03 | 0.03 | 0.17 | 0.17 |
| そのほか（区間の和の合算、 $D$ の小さいブロックの収集、配列の解放、対称化） | 0.04 | 0.05 | 0.16 | 0.22 |
| 合計 | 3.15 | 1.16 | 16.72 | 6.04 |

- 大きい積、 $W=V\cdot T$ 、区間のループは、4 スレッドで 1 スレッドの 0.30 倍から 0.35 倍の時間になりました。
- スレッドに分けていない処理は、 $D$ のブロックの収集、 $K$ への加算、 $D$ の小さいブロックの収集、配列の解放、対称化です。合わせて、1 スレッドでは構築の 5% から 6%、4 スレッドでは 14% から 15% を占めます。
- そのうち最も長いのは、 $D$ のブロックの収集です。4 スレッドでは構築の 9% でした。中心の原子ごとに、 $\ell\times s$ の配列（Si で 71 MiB、ダイヤモンドで 334 MiB）を確保して 0 で埋めます。次に、C 近傍と $S_A$ の site の組ごとにブロックを 1 個ずつ探して写します。組の数は、Si で 55,071、ダイヤモンドで 259,017 です。Si の 1 スレッドで細かく測ると、中心の原子 1 個につき、確保と 0 の書き込みが 36 ミリ秒、ブロックを探して読む処理が 13 ミリ秒、行の写しが 7 ミリ秒でした。
- 区間のループの中では、(iv) の項が最も長く、区間の中の時間の 36% から 39% です。次が (i) の 19% から 21%、(iii) の 17% から 19%、列の写しの 13% から 15%、(ii) の 10% から 11% です。区間の中の時間の和は、スレッドを 1 から 4 にすると、Si で 1.18 秒から 1.15 秒、ダイヤモンドで 6.92 秒から 7.31 秒になりました。
- 内訳を測った実行ファイルの構築 1 回の時間は、測る処理のない実行ファイルの値（上の表の 3.02 秒、1.16 秒、5.91 秒）と 5% 以内で合います。ダイヤモンドの 1 スレッドは、測る処理のない版を今回は測っていません。M6 の表の同じコードの値（15.07 秒）より 11% 長い値です。
- この内訳は 4 コアのコンテナの値です。18 スレッドの計算機では、スレッドに分けていない処理の割合が、これより大きくなると考えられます。実機での内訳は測っていません。
- 利用者の計算機での時間は、11 節の 4 番目の段階の表にあります。18 スレッドの CPU 版の交換行列の段は、1 スレッドの 0.21 倍（Si）と 0.18 倍（ダイヤモンド）でした。

構築の高速化の 2 番目の段階（`1b91c7b`）の時間です（2026-11-25）。 $D$ の行の収集と $K$ への加算を、区間に分けた版です（6.4 節）。前の版は `a8d7910` です。計算機と入力は、上の表と同じです。前の版と後の版を続けて 1 回ずつ実行し、これを 1 組としました。3 組目は、収集、加算、中心の原子ごとの寄与の時間を表示する一時的なライブラリーで実行しています。組の中では前の版を先に実行しました。ダイヤモンドの 1 スレッドの 2 組目と 4 スレッドの 3 組目だけは、後の版が先です。

| 系 | スレッド | 1 組目（前、後、比） | 2 組目（前、後、比） | 3 組目（前、後、比） |
|---|---|---|---|---|
| Si（`siz-p444`、構築 10 回） | 1 | 3.07 秒、3.05 秒、0.99 | なし | なし |
| 同上 | 4 | 1.06 秒、0.99 秒、0.94 | 1.02 秒、0.98 秒、0.96 | 1.19 秒、1.00 秒、0.84 |
| ダイヤモンド（`diaz-p222`、構築 12 回） | 4 | 4.93 秒、4.77 秒、0.97 | 5.43 秒、4.69 秒、0.86 | 5.97 秒、5.36 秒、0.90 |
| 同上 | 1 | 15.55 秒、16.28 秒、1.05 | 15.74 秒、16.12 秒、1.02 | 15.95 秒、16.13 秒、1.01 |

値は構築 1 回の平均です。3 組目で測った、中心の原子 1 個あたりの時間です（中心の原子は 2 個で、全ての構築の平均）。

| 処理 | Si、4 スレッド（前、後） | ダイヤモンド、4 スレッド（前、後） | ダイヤモンド、1 スレッド（前、後） |
|---|---|---|---|
|  $D$ の行の収集 | 53.9、13.9 ミリ秒 | 261.8、67.7 ミリ秒 | 246.1、194.8 ミリ秒 |
|  $K$ への加算 | 13.1、6.5 ミリ秒 | 71.6、29.4 ミリ秒 | 58.8、46.3 ミリ秒 |
| 中心の原子の寄与の残り（行列積と区間のループ） | 0.52、0.47 秒 | 2.62、2.55 秒 | 7.65、7.80 秒 |

- 1 スレッドの `.out` は、Si でもダイヤモンドでも、前の版と日付と時間の行だけが違います（Si で 19 行、ダイヤモンドで 20 行）。標準出力も、時間を表示する行のほかは同じです。
- 構築 1 回の行列積の回数（Si で 54,850 回、ダイヤモンドで 162,326 回）と最大の積は、前の版と同じです。4 スレッドの実行は全て収束し、構築の回数も前の版と同じでした。全エネルギーは前の版と 2.1e-11 Hartree 以内で、交換エネルギーは 1e-12 Hartree 以内で合います。
- 4 スレッドの構築 1 回は、前の版の 0.84 倍から 0.97 倍になりました。3 組目では、収集が 0.26 倍、加算が 0.41 倍から 0.50 倍です。収集は、4 つに等しく分けた場合の 0.25 倍に近い値でした。後の版の収集は、配列を 0 で埋める処理と、ブロックを一度別の配列に写す処理を省いています。1 スレッドの収集が 0.79 倍なのは、このためと考えられます。収集と加算で短くなった時間は、4 スレッドの構築 1 回の 8% です（3 組目の Si とダイヤモンド。計算値）。
- 1 スレッドのダイヤモンドは、3 組とも後の版が 1% から 5% 長い値でした。3 組目では、収集と加算はどちらも前の版の 0.79 倍です。長くなったのは中心の原子の寄与の残りで、平均で 2% 長い値でした。この部分のソースは、引数の型のほかは同じです。中心の原子ごとの最小値は、前の版が 7.30 秒と 7.39 秒、後の版が 7.38 秒と 7.29 秒で、ほぼ同じでした。標準偏差は 0.18 秒から 0.33 秒です。差の原因は確かめていません。差は時間の幅と同じ程度なので、実行の間の幅（13 節の 62）と考えられます。利用者の計算機の 1 スレッドでは、2 組とも後の版が短い値でした（0.992 倍と 0.997 倍。11 節の 5 番目の段階）。比べた版は `1219cb9` と `3d51128` です。
- 大きい配列の使い回しの効果の上限を、測定用の版で測りました。中心の原子の 4 つの大きい配列（ $D$ の行、`G`、`Rm`、最後の結果）を、実行の間ずっと残す版です。後の版とこの版を、4 スレッドで 2 回ずつ交互に実行しました。構築 1 回は、Si で後の版の 0.93 倍と 0.85 倍、ダイヤモンドで 0.98 倍（2 回とも）でした。同じ版の 2 回の実行の間でも、構築 1 回の平均は、Si で最大 9%、ダイヤモンドで 5% 違いました。中心の原子 1 個の大きい配列は、Si で約 270 MB、ダイヤモンドで約 1 GB です（計算値）。

改訂 21 で、厳密交換の計算を MPI のプロセスに分けました（6.7 節）。前の版と新しい版の時間です（2026-11-25）。条件は次のとおりです。クラウドのコンテナ（4 コア）の CPU 版で、1 プロセスにつき 1 スレッドです（`-nt 1`。OpenBLAS も 1 スレッド）。「前」は `c9e26e7` の実行ファイルで、準備の内訳を標準エラーに表示する行だけを足してビルドしました。「後」は分けた版（`a468823`）です。ほかのプロセスが CPU を使っていないことを確かめてから、プロセスの数ごとに、前の版と後の版を 1 件ずつ続けて実行しました。「全体」は、`mpirun` の開始から終了までの時間です。準備、構築、力は、プロセス 0 が標準出力に表示した値です。単位は秒です。

ベンゼン（12 原子、C6.0-s2p2d1 と H6.0-s2p1、150 Ry、Γ 点の形式、構築 8 回）です。最大の常駐メモリーは、最も大きいプロセスの値です。

| プロセス | 全体（前、後） | 準備（前、後） | 構築の合計（前、後） | 力（前、後） | 力のうち重み（後） | 最大の常駐メモリー（前、後、MiB） |
|---|---|---|---|---|---|---|
| 1 | 48.2、47.2 | 1.61、1.54 | 0.35、0.35 | 0.61、0.36 | 0.07 | 400、400 |
| 2 | 27.8、26.7 | 1.65、1.63 | 0.21、0.19 | 0.36、0.19 | 0.03 | 265、266 |
| 3 | 24.0、24.2 | 1.55、1.53 | 0.15、0.13 | 0.29、0.13 | 0.02 | 224、224 |
| 4 | 19.9、19.6 | 1.73、1.58 | 0.11、0.11 | 0.30、0.11 | 0.02 | 191、193 |

Si の基本単位胞（2 原子、Si7.0-s2p2d1、4×4×4 の k 点、site の形式、構築 10 回）です。

| プロセス | 全体（前、後） | 準備（前、後） | 構築の合計（前、後） | 力（前、後） | 力のうち重み（後） | 最大の常駐メモリー（前、後、MiB） |
|---|---|---|---|---|---|---|
| 1 | 66.4、67.4 | 1.34、1.40 | 25.56、25.89 | 4.24、3.89 | 3.46 | 523、525 |
| 2 | 36.0、36.0 | 1.40、1.18 | 13.41、13.72 | 2.30、2.06 | 1.70 | 510、509 |
| 3 | 35.4、30.8 | 1.37、1.04 | 13.86、10.38 | 2.07、1.56 | 1.19 | 505、464 |
| 4 | 34.1、27.4 | 1.40、1.02 | 13.30、7.82 | 2.16、1.26 | 1.08 | 503、445 |

Si の 16 原子（Si7.0-s2p2d1、2×2×2 のスーパーセル、Γ 点の形式、構築 9 回）です。

| プロセス | 全体（前、後） | 準備（前、後） | 構築の合計（前、後） | 力（前、後） | 力のうち重み（後） | 最大の常駐メモリー（前、後、MiB） |
|---|---|---|---|---|---|---|
| 1 | 241.5、227.2 | 5.40、5.35 | 5.68、5.23 | 13.75、7.24 | 0.83 | 1508、1597 |
| 2 | 121.0、114.4 | 5.13、3.20 | 2.83、2.64 | 6.73、4.02 | 0.50 | 1413、1482 |
| 4 | 66.1、63.2 | 5.22、2.30 | 1.50、1.62 | 3.43、2.51 | 0.29 | 1055、1128 |

Si の基本単位胞を、1 プロセスにつき 2 スレッドで実行した値です（`-nt 2`。OpenBLAS も 2 スレッドで、`mpirun` に `--bind-to none` を付けました）。

| プロセス | 全体（前、後） | 準備（前、後） | 構築の合計（前、後） | 力（前、後） | 力のうち重み（後） | 最大の常駐メモリー（前、後、MiB） |
|---|---|---|---|---|---|---|
| 1 | 42.7、41.4 | 1.42、1.34 | 12.53、12.77 | 3.13、2.72 | 2.31 | 537、538 |
| 2 | 35.7、32.0 | 1.34、1.12 | 7.15、7.00 | 1.74、1.55 | 1.28 | 526、525 |

厳密交換の部分が小さい系です。メチルラジカルとメタンは 150 Ry、ダイヤモンドは `scf.Ngrid` が 40×40×40 で、基底は C5.0-s1p1 と H5.0-s1 です。GGA-PBE のベンゼンは、上の HSE06 のベンゼンと `scf.XcType` だけが違う入力です。

| 系 | プロセス | 全体（前、後） | 準備（前、後） | 構築の合計（前、後） | 力（前、後） |
|---|---|---|---|---|---|
| メチルラジカル（HSE06、スピン分極） | 1 | 14.6、12.9 | 1.16、1.11 | 0.00、0.00 | 0.03、0.01 |
| メチルラジカル（HSE06、スピン分極） | 2 | 7.9、7.7 | 1.17、1.14 | 0.00、0.00 | 0.02、0.00 |
| メチルラジカル（HSE06、スピン分極） | 4 | 7.2、7.0 | 1.27、1.14 | 0.00、0.00 | 0.02、0.00 |
| メタン（PBE0） | 1 | 8.2、8.4 | 1.24、1.20 | 0.00、0.00 | 0.17、0.09 |
| メタン（PBE0） | 2 | 5.0、4.9 | 1.27、1.20 | 0.00、0.00 | 0.13、0.04 |
| ダイヤモンド（PBE0、2×2×2） | 1 | 19.8、20.1 | 1.01、1.00 | 0.64、0.66 | 0.55、0.33 |
| ダイヤモンド（PBE0、2×2×2） | 2 | 11.2、11.2 | 1.02、0.81 | 0.43、0.47 | 0.29、0.18 |
| ベンゼン（GGA-PBE） | 1 | 10.9、10.7 | なし、なし | なし、なし | なし、なし |
| ベンゼン（GGA-PBE） | 2 | 6.5、6.5 | なし、なし | なし、なし | なし、なし |

- Si の基本単位胞では、前の版の構築が 2 プロセスから先で短くなりません（13.41 秒、13.86 秒、13.30 秒）。後の版は 13.72 秒、10.38 秒、7.82 秒で、1 プロセスの 1.89 倍、2.49 倍、3.31 倍の速さです（計算値）。最後の構築での受け持ちの時間は、4 プロセスで 0.68 秒から 0.75 秒でした（標準出力の `shares` の行）。
- Si の 16 原子の構築は、どのプロセスの数でも、前の版と同じ原子の分け方になります。どの原子も同じ元素で、プロセスの数が原子の数を割り切るためです。後の版の時間は前の版の 0.92 倍から 1.08 倍で、この差は実行の間の幅と考えられます（13 節の 62）。
- Si の 16 原子の準備は、前の版ではどのプロセスの数でも 5.1 秒から 5.4 秒でした。後の版は、2 プロセスで 3.20 秒、4 プロセスで 2.30 秒です。前の版が標準エラーに表示した内訳（1 プロセス）は、元素ごとの表が 0.83 秒、T ブロックと V ブロックが 4.58 秒でした。元素ごとの表は全てのプロセスが計算するので、表が 1.43 秒でブロックが 0.19 秒のベンゼンでは、準備がほとんど変わりません（14 節）。
- 厳密交換の力は、後の版では 1 プロセスでも前の版より短くなりました（Si の 16 原子で 13.75 秒から 7.24 秒）。対ごとに式 (7) を 1 回だけ分解するためと考えられます（6.7 節）。4 プロセスでは、前の版の 0.37 倍（ベンゼン）、0.58 倍（Si の基本単位胞）、0.73 倍（Si の 16 原子）でした（計算値）。
- 4 プロセスの全体の時間は、Si の基本単位胞で 34.1 秒から 27.4 秒、Si の 16 原子で 66.1 秒から 63.2 秒になりました。ベンゼンの後の版の 4 プロセスでは、準備、構築、力の和が全体の 9% でした（計算値）。
- 4 コアを 2 プロセス × 2 スレッドで使うと、後の版の構築は 7.00 秒で、4 プロセス × 1 スレッドの 7.82 秒より短い値でした。全体は 32.0 秒で、4 プロセス × 1 スレッドの 27.4 秒より長い値です。全体から準備、構築、力を引いた残りは、22.4 秒と 17.4 秒でした（計算値）。
- Si の基本単位胞の 3 プロセスと 4 プロセスでは、後の版の最大の常駐メモリーが 41 MiB と 58 MiB 小さい値でした。途中で切れた原子では、`G`、`W`、`Rm` を範囲の ABF の分だけ作るためと考えられます。Si の 16 原子では、後の版が 69 MiB から 89 MiB 大きい値でした。この系の T ブロックと V ブロックは合わせて 181 MiB で（計算値）、力の間は、全てのプロセスが同じ大きさの重みの配列を持ちます（14 節）。差の内訳は測っていません。

改訂 25 で、ABF をプロセスに分ける見積もりを、時間の見積もりに変えました（6.7 節、`92f78c7`）。係数を決めるために、構築の部分ごとの時間を測りました（2026-11-25）。条件は、クラウドのコンテナの CPU 版で、1 プロセス 1 スレッドです（OpenBLAS も 1 スレッド）。時間を測る処理を加えた一時的な実行ファイルで、半分の構築の積と加算の時間を、中心の原子ごとに標準出力へ出しました。系は、Si の基本単位胞の HSE06 と PBE0（`siz-p444`、`siz-p444-pbe0`）と、HSE06 のダイヤモンド（`diaz-p222`）です。次の表は、全ての構築の和から求めた速さです（乗算 10⁹ 回毎秒。計算値）。

| 系 | 中心の原子 | `G` | 最後の積 | `W` | 4 つの項 |
|---|---|---|---|---|---|
| `siz-p444` | 1 番目 | 30.0 | 30.1 | 22.0 | 8.4 |
| `siz-p444` | 2 番目 | 30.9 | 31.1 | 22.5 | 8.7 |
| `siz-p444-pbe0` | 1 番目 | 27.1 | 27.0 | 21.9 | 8.0 |
| `siz-p444-pbe0` | 2 番目 | 30.3 | 26.9 | 21.9 | 8.0 |
| `diaz-p222` | 1 番目 | 30.5 | 28.1 | 19.5 | 8.0 |
| `diaz-p222` | 2 番目 | 30.4 | 27.8 | 19.2 | 7.9 |

- `W` の積の速さは、2 つの大きい積の平均の 0.66 倍から 0.81 倍でした（計算値）。式の 1.4 は、この比の逆数の代表値です。
- 4 つの項は、0.27 倍から 0.30 倍の速さでした。乗算 1 回を 3.6 と数えます。式の $14.4\,u_a$ は、ABF と V 近傍ごとの乗算 $4u_au_b\ell_B$ 回に 3.6 を掛けた値です。
- 部分ごとに繰り返す仕事の 1 つは、`W` の積が V 近傍の T ブロックの全体を読む時間です。Si の HSE06 では、V 近傍 1 個あたり約 0.57 ミリ秒でした。T ブロックの要素 1 個あたり、乗算 16 回分に当たります。
- もう 1 つは、 $D$ の行の収集と最後の結果の加算です。 $\ell_As_A$ の要素 1 個あたり、乗算 290 回分と見積もりました。
- `G` の転置と、括弧の初期化と転置は、ABF ごとの仕事です。 $u_as_A$ の要素 1 個あたり乗算 500 回分で、 $t_a$ の第 2 項に入れました。

同じ実行で、中心の原子ごとの見積もりと時間を比べました。「見積もり」は $69\,t_a+q_a$ で、単位は大きい積の乗算 10⁹ 回です。「時間」は構築 1 回の平均です。「部分」は $q_a/t_a$ で、部分ごとの仕事を ABF の数で表します。

| 系 | 中心の原子 | 見積もり | 時間（秒） | 見積もり／時間 | 部分 |
|---|---|---|---|---|---|
| `siz-p444` | 1 番目 | 30.62 | 0.843 | 36.3 | 6.6 |
| `siz-p444` | 2 番目 | 33.71 | 0.910 | 37.0 | 6.5 |
| `siz-p444-pbe0` | 1 番目 | 181.36 | 5.556 | 32.6 | 5.5 |
| `siz-p444-pbe0` | 2 番目 | 195.73 | 5.939 | 33.0 | 5.4 |
| `diaz-p222` | 1 番目 | 152.19 | 4.662 | 32.6 | 6.7 |
| `diaz-p222` | 2 番目 | 170.08 | 5.286 | 32.2 | 6.6 |

- 2 つの原子の見積もりの比は、時間の比と 2% 以内で合いました。Si の HSE06 では 1.101 と 1.079、PBE0 では 1.079 と 1.069 です。ダイヤモンドは、1.118 と 1.134 でした（計算値）。
- 構築 1 回の時間のうち、この表の部分の外にあるのは、Si の HSE06 で 0.173 秒（9%）です。PBE0 では 0.714 秒（6%）、ダイヤモンドでは 0.667 秒（6%）でした。この時間は、見積もりに入れていません。
- 前の版の 2 プロセスでは、プロセス 0 が、1 番目の原子の全てと 2 番目の原子の 3 個の ABF を受け持ちました。この 3 個の部分の速さは、`G` が 19.0、最後の積が 7.5、`W` が 4.3 です（10⁹ 回毎秒）。
- その部分の時間は構築 1 回の平均で 0.122 秒、2 番目の原子の残りの 66 個は 1.044 秒でした。ABF 1 個あたりでは 2.6 倍です（計算値。13 節の 74）。2 つの時間から求めた部分ごとの仕事は、ABF の 5.3 個分です。見積もりの値は、6.5 個分でした。
- 係数は、このコンテナの CPU で測った値です。利用者の計算機では、まだ測っていません（14 節）。見積もりを変えた後の時間は、9 節にあります。

## 13. 落とし穴

1. $K$ は近傍リストの形式の $H$ に入りません。この形式の $H$ から $H(\mathbf k)$ を作り直す経路（`Band_DFT_kpath.c:342` や `Band_DFT_Dosout.c:551` の `Hamiltonian_Band` など）は、 $K$ を含まない固有値を出します。どの出力が該当するかは個別に確かめていないので、第 1 版では 2.2 節の出力をまとめて拒否します。M8 で、バンド分散と DOS の経路には $K(\mathbf k)$ を足しました（6.5 節）。
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
19. 打ち切りクーロンのカーネルは、 $\cos(\kappa R_c)$ で振動します。式 (12) の被積分関数は、 $\kappa$ について周期 $2\pi/(r_P+r_Q+d+R_c)$ の成分まで持ちます。 $r_P$ と $r_Q$ は 2 つの関数の半径です。既定の格子の刻みは 60/900 = 0.067/bohr で、 $2\pi$ をこの刻みで割ると 94 bohr です。M8 で確かめました。被積分関数は $\kappa$ について偶関数で、指数型 $r_P+r_Q+R_c+d$ の整関数です。そのフーリエ変換は、この長さの外で 0 になります（ペイリー・ウィーナーの定理）。ポアソンの和公式から、刻み $\Delta\kappa$ の台形則は、 $r_P+r_Q+R_c+d<2\pi/\Delta\kappa$ なら上限より先の部分を除いて正確です。試験 `exx-two-center-truncated` の半径 6 bohr の関数では、条件の内側の差が、最大の値の 3.1e-15 倍以下でした（区間 3600 との比較）。条件を外れた所の差は、 $R_c$ が 36 bohr で 8.2e-3 倍、40 bohr で 5.4e-2 倍です。実の ABF も同じ形でした。C6.0 の差は、34 bohr で 7.6e-15 倍、36 bohr で 1.5e-6 倍、40 bohr で 1.5e-2 倍です。Si7.0 は 33 bohr で 8.2e-15 倍、34 bohr で 2.0e-7 倍でした。PBE0 の V ブロックの $d$ は「大きい方の ABF の半径 + V の範囲」まで届くので、本体は区間の数を増やします（5.6 節）。
20. 式 (13) の動径の積分は 4 次精度、台形則の動径積分は 2 次精度です。同じ原子の $G_{PQ}$ は、 $P$ と $Q$ を入れ替えると 2e-9 から 5e-9 ずれます。モジュールは 2 通りの平均を使います（同じ原子の $b^P$ も同じ）。平均しないと、ABF の張る空間にある積が正確に展開されません。
21. $K$ を作った後の内側の SCF は、2 つの形で前の $K$ の解に留まります。1 つ目は停止条件です。既存の条件は、連続する 2 ステップのバンドエネルギーの差と、緩い残差の条件です。混合が密度をほとんど動かさないと、新しい $K$ に応答していない密度でこの条件を通ります。単純混合の重みが下限（`scf.Min.Mixing.Weight`）に残っている場合と、外挿が密度を前の値に戻した場合に起きます。2 つ目は外挿の履歴です。前の $K$ で測った残差は、前の $K$ で収束した密度でほぼ 0 です。この組が履歴に残ると、外挿は密度を前の解へ引き戻します。M4 の実測です。対策のない実装では、`scf.criterion 1e-6` の Methane（RMM-DIIS）で内側の SCF が 2 ステップで止まり、全エネルギーが 3.2e-6 Hartree ずれました。ステップ数で停止を禁じるだけの実装では、`scf.criterion 1e-10` の Methane の全エネルギーが、RMM-DIISK で 1.4e-6、RMM-ADIIS で 1.7e-6 Hartree ずれました。この実装で RMM-DIIS 以外の方式を試した実行では、停止した 41 回のうち 9 回で、残差ノルムが $K$ を作った直後の値の 0.4 倍以上のままでした。RMM-DIIS の 439 回の停止では、混合比 0 の実行を除いて、直後の値の 1.5e-2 倍以下でした。今の実装は、混合を方式ごとに始め直し、残差ノルムが下がるまで停止を認めません（6.6 節）。直した後の結果は進捗ファイルにあります。
22. 分子の計算では、セルを「分子の大きさ + ABF の半径 + `scf.exx.v.cutoff`」より大きく取ります。V 近傍は周期像も数えるので、セルが小さいと、分子とその周期像の間の交換が入ります。M4 の実測では、1 辺 10 Å（18.9 bohr）のセルの水分子で、打ち切りを 8 bohr から 12、16 bohr にすると、全エネルギーが 1.5e-4、1.2e-3 Hartree 下がりました。1 辺 20 Å のセルでは、6 から 16 bohr まで全ての桁が同じです。
23. Γ 点だけの計算は、セルが短距離カーネルの到達距離より小さいと、厳密交換を大きく見積もります。Γ 点だけの密度行列は、 $D(\mathbf R)$ が全ての $\mathbf R$ で同じ形に当たり、距離で減衰しません。交換の和を打ち切るのは、カーネル $\mathrm{erfc}(\omega r)/r$ と `scf.exx.v.cutoff` だけです。ダイヤモンドの 8 原子のセル（1 辺 6.74 bohr、C5.0-s1p1）では、既定の打ち切りで $E_x$ が −4.61 Hartree になり、Γ 点のギャップが PBE の 6.5 eV から 12.5 eV に開きました（M4 の実測）。打ち切りを 6、8、12、16 bohr と変えると、 $E_x$ は −4.33、−4.61、−4.83、−4.86 Hartree と変わります。平面波の表示で $\mathbf G=0$ に当たる項 $-\alpha N_{\mathrm{occ}}\tilde v(0)/\Omega$ は、このセルで −3.4 Hartree です（計算値。 $\tilde v(0)=\pi/\omega^2$ 、 $N_{\mathrm{occ}}=16$ ）。この項は超格子を大きくしても同じ値で、原子あたりでは体積に反比例して減ります。小さいセルの固体は、k 点（M5）で計算します。
24. Γ 点だけの小さいセルでは、打ち切った V の和が半正定値でなくなります。全ての周期像について足した $\sum_{\mathbf R}V_{ab}(\mathbf R)$ は、カーネルのフーリエ変換が正なので半正定値です。V 近傍の距離で和を打ち切ると、負の固有値が現れます。ダイヤモンドの 8 原子のセルで、Γ 点の形式が使う和の固有値を測りました（M4。ABF は動径部分の 2 乗ノルムが 1 で、単位は Hartree）。C6.0-s2p2d1 の ABF（原子あたり 69 個、行列の次数 552）では、打ち切りが 6、8、12、16 bohr のとき、負の固有値が 46、36、45、37 個ありました。最小の固有値は −10.2、−17.6、−1.4、−0.47 で、最大の固有値は 2,371 から 2,986 です。C5.0-s1p1 の ABF（原子あたり 10 個）では、どの打ち切りでも負の固有値がなく、最小の固有値は 0.42 から 0.72 でした。s2p2d1 の基底のこのセルは、打ち切り 6 bohr で外側反復が発散しました。5 回目から 17 回目まで $\delta$ の符号が 1 回ごとに変わり、正の $\delta$ は 2 回ごとに約 1.4 倍になります。18 回目からは正のまま増え、27 回目の $K$ の後の内側の SCF が 100 ステップで収束しませんでした。既定の 8 bohr では 7 回で収束しましたが、 $\delta$ の比は 0.18、0.39、0.14、0.46、0.09 と 1 回おきに変わります。12 bohr と 16 bohr では 8 回で収束し、比は 0.21 から 0.25 でした。C5.0-s1p1 の基底では、どの打ち切りでも $\delta$ が毎回約 0.2 倍になります。発散の原因が負の固有値にあることは、確かめていません。外側反復は $K$ を混合せずに入れ替えるので、 $K$ の変化に対する密度の応答が元の変化より大きいと発散します。密度行列が距離で減衰する計算（大きいセル、k 点）では、打ち切られる対の寄与は小さいと考えられます。M5 で、k 点の計算を測りました。C6.0-s2p2d1 のダイヤモンドの基本単位胞は、2×2×2 のメッシュで外側反復が 12 回かかり、 $\delta$ の比は 0.07 から 0.65 でした。このメッシュは、16 原子のセルの Γ 点と同じ計算です。4×4×4 と 8×8×8 では 8 回で収束し、比は 0.09 から 0.13 です。打ち切りを変えたときのエネルギーの変化は、31 に書きました。k 点の計算では、V の和の固有値を測っていません。
25. 単純混合（Simple）と Kerker 混合は、残差ノルムが約 3e-5 を下回ると、残差をほとんど減らせません。混合の重みを決める規則が、この水準では重みを減らすだけになるためです（6.6 節）。交換行列の有無には関係しません。M4 の実測です。系は、入力例の基底のメタン、水、ダイヤモンドの 8 原子です（C5.0-s1p1 と H5.0-s1。水は O6.0-s1p1 と H6.0-s1p1）。PBE のメタン（Kerker 混合、`scf.criterion 1e-10`）は、残差ノルムが 50 ステップ目で 1.2e-5、300 ステップ目で 1.0e-5 でした。300 ステップでは収束しません。同じ条件の単純混合も、300 ステップで収束しません。この条件の HSE06 は、最初の PBE の SCF が収束しないので、 $K$ を作らずに終わります（6.6 節の「未収束の扱い」）。`scf.criterion 1e-6` の PBE の 6 件は、残差ノルムが 1.0e-5 から 1.5e-5 のステップで止まりました。全エネルギーは、RMM-DIIS（`scf.criterion 1e-10`）の値から 8e-8 から 9.3e-6 Hartree ずれます。停止条件に 3 つ目の値がない実装（`2753c7d`）では、`scf.criterion 1e-6`、`scf.maxIter 300` のメタンと水の HSE06 が、4 件とも収束しませんでした。4 回目の $K$ の後の内側の SCF が、300 ステップで止まりません。今の実装（`b8126f8`）では、ダイヤモンドを加えた 6 件が、外側反復 5 回か 6 回、合計 49 から 90 ステップで収束しました。全エネルギーのずれは 1.0e-7 から 1.7e-5 Hartree です。「PBE の SCF が止まったときの残差」を下限にする形も試しましたが、採りませんでした。水の単純混合で、 $K$ の変化が残差を 9.3e-6 から 1.6e-5 に上げ、300 ステップ後も 1.2e-5 でした。`scf.Mixing.Type` を書かない入力は、Simple になります（`Input_std.c` の選択肢の先頭）。この 2 つの方式の HSE06 は、PBE と同じく 1e-5 Hartree 程度までしか決まりません。精度が要る計算では、外挿する方式（RMM-DIIS、RMM-DIISK など）を指定します。
26. `scf.criterion` と `scf.exx.outer.criterion` は、セル全体のエネルギーの差と比べる値です。原子の数では割りません。同じ結晶をスーパーセルで計算すると、同じ値でも判定が厳しくなります。M5 の実測です。C5.0-s1p1 のダイヤモンドを、基本単位胞の 2×2×2 のメッシュと、16 原子のスーパーセルの Γ 点で計算しました。 $\lvert\delta\rvert$ は、4 回目の構築で 7.9e-7 と 6.3e-6 Hartree、5 回目で 7.3e-8 と 5.8e-7 Hartree です。既定の閾値（1e-6）なら、基本単位胞は 4 回目、スーパーセルは 5 回目で止まります。止まった回の $E_x$ は、基本単位胞あたり 7.3e-8 Hartree 違います。この差は、収束の途中で止めた回の違いによるものです。2 つの計算を比べるときは、2 つの閾値に基本単位胞の数を掛けて揃えます。`tools/exx_kpoint_checks.py` は、この形で入力を作ります。
27. 単位胞に軌道が 1 個しかない系は、 $K$ の組み込みの試験になりません。ブロッホ状態の係数が規格化だけで決まり、密度行列が占有数だけの関数になるためです。 $K$ を変えても密度行列は変わりません。M5 では、水素の鎖を H5.0-s1 で計算したときに、外側反復が 2 回で止まって気付きました。試験の系は、1 原子に軌道が 5 個ある基底（H5.0-s2p1）で作り直しました。
28. `scf.Generation.Kpoint none`（対称性で減らした Monkhorst–Pack のメッシュ）は、ハイブリッド汎関数でない計算でも異常終了します。GGA-PBE のダイヤモンド（基本単位胞、C5.0-s1p1、3×3×3、1 ランク）で、k 点を作った直後に `MPI_Comm_size` が `MPI_ERR_COMM: invalid communicator` を出し、終了コード 5 で終わりました。基準コミット（`4e0ee247`）からビルドした CPU 版でも同じです。原因は調べていません。HSE06 の試験は、`GCenter` と `REGULAR` で行いました。
29. 改訂 20 までの構築は、中心の原子をランクに配っていました。ランクが単位胞の原子より多いと、構築を受け持たないランクが出ます。基本単位胞（2 原子）のダイヤモンドを 4 ランクで計算すると、構築するのは 2 ランクでした。準備も、全てのランクが同じ計算を重ねていました。改訂 21 で、構築を ABF の範囲で、準備を site の対で配る形に改めました（6.7 節）。原子より多いランクは、1 つの原子の ABF を分け合います。標準出力の構築の行は、ランク 0 が受け持った ABF の範囲を表示します。
30. 改訂 26 までの site の形式の構築は、k メッシュが粗くても軽くなりませんでした。構築は、 $S_A$ の全ての site を密度行列の別の列として扱います。時間は k 点の数に依りません（12 節）。Γ 点を含む n×n×n のメッシュの $D(\mathbf R)$ は、基本単位胞の n 倍の周期で同じ値を繰り返します。スーパーセルの Γ 点の形式は、この周期で列をまとめてから行列積を計算します。M5 の実測です。2×2×2 のメッシュのダイヤモンド（C6.0-s2p2d1）の構築は、site の形式で 1 回 33.6 秒でした。16 原子のスーパーセルの Γ 点の形式では 1.5 秒です。構築 1 回の flop 数は、site の形式が 42 倍でした。2 つの計算の全エネルギーは、原子あたり 3e-11 Hartree まで一致します。改訂 27 で、V 近傍をメッシュの周期でまとめる形を入れました（6.4 節）。PBE0 の Si（4×4×4）では、V 近傍の site が 1,698 個から 270 個になりました。構築の和は 0.22 倍です（9 節）。改訂 28 で、構築の列（ $S_A$ ）も同じ周期でまとめるようにしました（6.4 節）。PBE0 の Si（4×4×4）の 1 プロセスの構築の和は、V 近傍だけをまとめた版の 0.59 倍になりました（9 節）。まとめられる列があるのは、メッシュの周期だけ離れた 2 つの site が $S_A$ に入る場合です。C6.0 のダイヤモンドの $S_A$ は半径 26 bohr の球で、8×8×8 のメッシュの周期は 38.1 bohr です（どちらも計算値）。
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
43. GEMM の橋渡しは、GEMMul8 で計算できなかった積を FP64 で計算して、成功を返します。戻り値を見るだけの試験は、GEMMul8 が 1 回も動いていなくても通ります。経路は、橋渡しが数える FP64 の呼び出しの数（`openmx_gemmul8NativeCalls`）で確かめます。M6 の部品は、この数から GEMMul8 の積を数えて表示し、道具の `NAME-g8` がその行を判定します。
44. `scf.gemmul8.enable on` は、固有値ソルバーの中の行列積の方式も決めます。デバイスの密行列ソルバーを通る実行にこの行を足すと、厳密交換の積の差と、ソルバーの中の積の差が混ざります。既定で FP64 を選ぶデバイスの場合です。道具の `NAME-g8` は、しきい値を下げずに固有値問題をホストで解き、厳密交換の積だけを変えます。
45. 部品は、結果の行列を初期化せずに積に渡します。BLAS の DGEMM は、 $\beta=0$ のとき結果の行列を読みません。GEMMul8 も、サブモジュールが指すコミット（`f8fdf0b`）では読みません [20]。サブモジュールを更新するときは、 $\beta=0$ の積が結果の行列を読まないことを確かめ直します。読む版では、初期化していない値が結果に入ります。M6 で構築を書き直した後は、`modules/exx` も同じ形です。`G` と最後の行列積の結果は、確保したままの配列です（7.2 節）。
46. 本体の OpenMP のスレッド数は、起動のオプション `-nt` で決まります。既定は 1 で、本体は起動時に `omp_set_num_threads` を呼びます（`source/openmx.c:249`、`286`）。環境変数 `OMP_NUM_THREADS` を設定しても、スレッド数は変わりません。M4 から M6 の試験の道具は、環境変数だけを設定していました。`a7fbfb8` より前の道具の実行は全て、OpenMP が 1 スレッドです。文書と道具の表示の「2 スレッド」は、OpenBLAS に渡した数でした。M6 の実機の最初の段階で分かり、直しています。M4 と M5 の道具は `-nt 1` を明示し、M6 の道具は `--threads` の数を `-nt` と BLAS に渡します。本体は、標準出力の 2 行目にスレッド数を表示します。M6 の道具は、この数を記録します。`-nt 2` にすると、結果の末尾の桁が変わります。クラウドのコンテナで、メタンの HSE06 を計算しました。全エネルギーは、1 スレッドで −7.999482449274 Hartree（SCF 46 回）、2 スレッドで −7.999482448952 Hartree（49 回）です。2 スレッドの 7 回のうち 2 回は、1 スレッドと同じ値でした。原因は調べていません。1 スレッドの結果は、同じ計算機で実行するたびに全ての桁が同じです。実行ファイルどうしを桁まで比べる道具は、1 スレッドのままにしています。M6 で構築を書き直した版は、1 スレッドで −7.999482448976 Hartree（49 回）になりました。書き直す前との差は 3.0e-10 Hartree です。厳密交換のエネルギーの差は、1e-12 Hartree 以下です。書き直しは、積の分け方と足す順序を変えます。丸めの違いが SCF の止まるステップを動かし、全エネルギーの末尾の桁が変わったと考えられます。スレッド数による違いも、同じ仕組みと考えられます。どちらも確かめてはいません。
47. mpirun は、プロセスが 1 個の実行を 1 コアに束縛することがあります。利用者の計算機では、`-np 1` のプロセスが使える論理 CPU は、1 コアの 2 個だけでした（ホストの Open MPI 5.0.10rc2 と、コンテナの Open MPI 4.1.6）。この束縛のままでは、スレッドを増やしても使えるコアは増えません。スレッドを増やした時間を測るときは、`mpirun --bind-to none` を使います。
48. CUDA 版の GPU を使う経路の結果は、実行ごとに末尾の桁が変わります。利用者の計算機で、基準コミットの CUDA 版を同じ入力で 2 回実行しました。メタンの全エネルギーは 2.3e-11 Hartree 違い、SCF の回数も 13 回と 14 回でした。`.out` の差が日付と時間の行だけになることは、CUDA 版の既定の経路では確かめられません。CUDA 版どうしの比較では、全エネルギーの差を、同じ実行ファイルの 2 回の差と比べます。`OPENMX_GPU=0` を付けると、全てのランクがホストのコードに残り、結果は実行ごとに同じになります。この形で確かめられるのは、CUDA 版のホストの経路だけです。どの段が違いの元かは調べていません。
49. `OPENMX_GPU_CAPACITY_TEST_RECOVERY=1` は、厳密交換の積のほかの段も変えます。この変数は、既存の常駐の高速経路を止めます（`docs/ENVIRONMENT_VARIABLES.md`）。利用者の計算機の `siz-p444` では、`Set_Hamiltonian` の時間が 7.0 秒から 42.2 秒になりました。`NAME-host` と `NAME-gpu` で比べられる時間は、交換行列の段だけです。`DFT in total` と経過時間の差には、ほかの段の違いが入ります。
50. 構築の時間は、乗算の回数の割合では見積もれません。`a7fbfb8` までの形の Si では、デバイスに送る大きい積が、乗算の回数の 58% を占めます。時間では、構築の 21% でした（1 スレッド、12 節）。4 スレッドでは 9% です。残りの大半は、V 近傍ごとの出力の小さい積と、要素を 1 個ずつ動かす配列の写しでした。この 2 つは、乗算の回数が同じでも、配列の並べ方で時間が変わります。V 近傍ごとの項を ABF 1 個ずつ計算する形にすると、構築 1 回は Si で 0.65 倍、s2p2d1 のダイヤモンドで 0.57 倍になりました。積をデバイスに送る効果は、時間の内訳を測ってから見積もります。
51. OpenBLAS は、V 近傍ごとの小さい積をスレッドに分けません。クラウドのコンテナの OpenBLAS 0.3.26（pthread 版）で、構築と同じ形の積を 1 個ずつ測りました。ABF 1 個の 4 つの項の積（13×13 で内側が 1,131 または 2,171 の形と、その転置の形）は、4 スレッドを指定しても時間が変わりません。全ての ABF をまとめた形では、897×13（内側 1,131）の積が 4 スレッドで 3.3 倍速くなり、13×13（内側 78,039）の積は変わりません。スレッドで構築を速くするには、ABF のループを並列にする必要があります（14 節）。実機の 3 番目の段階で、利用者の計算機の 2 つの BLAS を 18 スレッドで測りました。コンテナの OpenBLAS 0.3.26 と、CUDA 版がつなぐ NVHPC 26.9 の BLAS です。後者は、OpenMP で並列化した OpenBLAS でした。ABF 1 個の 4 つの形は、どちらの BLAS でも、経過時間が 1 スレッドのときと変わりません（5 マイクロ秒から 15 マイクロ秒）。4 つのうち (i) の形（13×13、内側 1,131）だけは、OpenBLAS の 18 スレッドで、CPU 時間が経過時間の約 7 倍になりました。経過時間は、11.6 から 12.3 マイクロ秒が 12.5 から 13.1 マイクロ秒になっています。原因は調べていません。MKL などのほかの BLAS では測っていません。
52. 大域状態のゲートは、解析できなかった翻訳単位があっても、終了コードが 0 になります。ルートの CMake のビルドは、FFTW を `build/<名前>/native/source/third_party/fftw3-install` に作ります。`source/Makefile` の `FFTW3_PREFIX` の既定は `source/third_party/fftw3-install` で、この木にはありません。利用者の計算機では、`fftw3.h` を使う 7 個の翻訳単位（`Cluster_DFT_Col.c` を含みます）が解析されませんでした。増えた項目の表示は 0 でした。ゲートの結果を読むときは、標準出力の `translation units not analysed` が 0 であることを確かめます。`--make-arg FFTW3_PREFIX=...` を足すと、288 個が全て解析されます（進捗ファイルの「ビルドと実行」）。
53. ゲートの要約の数は、判定に入りません。`a7fbfb8` では、「written by some function」が 697 で、ベースラインの 696 より 1 多い値でした。増えたのは `XC_switch` です。M4 から、`Input_std` は `scf.XcType HSE06` を読んだ後に `XC_switch = 4` を代入します。ベースラインの時点の `Input_std` は、この変数のアドレスを入力の関数に渡すだけでした。道具は、その使い方を書き込みに数えません。ゲートが比べるのは関数とオブジェクトの組で、この組はベースラインにあります。増えた項目は 0 なので、ベースラインは更新していません。要約の数がベースラインと違うときは、どのオブジェクトが増えたかを報告の JSON で確かめます。
54. ビルドの道具は、`TMPDIR` を指定しても、作業のフォルダーの外にファイルを書きます。利用者の計算機では、CUDA 版のビルドのたびに、`/dev/shm` に `sm_segment.*` というファイル（16 MiB）が 1 個残りました。名前の形から、Open MPI の共有メモリーの裏付けファイルと判断しています。Open MPI 5 は、Linux でこのファイルを `/dev/shm` に置きます。置き場所は、MCA パラメーターの `btl_sm_backing_directory` で変えられます [21]。同じビルドの終わりには、`~/.nv/ComputeCache` に約 30 MB が書かれました。CUDA の JIT のキャッシュの既定の場所で、環境変数 `CUDA_CACHE_PATH` で変えられます [22]。計算の実行の時刻に書かれたファイルはありません。どのコマンドが書くかは、調べていません。作業の場所が決められている計算機では、ビルドの前後でこの 2 か所を見ます。実機の 3 番目の段階で、CUDA 版のビルドの間だけ、2 つの置き場所をフォルダーの中に向けました。2 つとも効き、外の 2 か所に増えたファイルはありません。CUDA のキャッシュは、リンクの時刻に 26 個が書かれます。共有メモリーのファイルは、依存ライブラリーを最初からビルドするときにだけできます（開始の 5 分 47 秒後）。変わったファイルだけのビルドでは、できません。
55. 確保したままの配列を結果に渡すと、最初の書き込みの時間が、その積の時間に入ります。M6 で構築を書き直した後、デバイスの積の時間が、積 1 個あたり 24 ミリ秒（Si）と 92 ミリ秒（s2p2d1 のダイヤモンド）増えました。積の数と形は同じです。書き直す前は、構築が結果の配列を 0 で埋めていて、ページの割り当てはそこで済んでいました。書き直した後は、デバイスからホストへの転送が最初の書き込みになります。増えた分は、結果の配列 1 MiB あたり 0.38 ミリ秒と 0.39 ミリ秒です（計算値。11 節）。この説明は、実機では確かめていません。段ごとの時間を版の間で比べるときは、配列の確保と最初の書き込みがどの段に入るかを見ます。
56. 自前のスレッドを持つ BLAS は、OpenMP のループと交互に使うと構築を遅くします。構築のスレッド並列の最初の形は、ABF のループだけを OpenMP にして、`W` の積と大きい積を BLAS のスレッドに残していました。クラウドのコンテナ（4 スレッド、OpenBLAS 0.3.26 の pthread 版）では、Si の構築 1 回が 1.8 秒から 2.1 秒だったものが、2.4 秒から 3.8 秒になりました。OpenMP のスレッドと OpenBLAS のスレッドは別の集まりで、V 近傍ごとに入れ替わります。Si では構築 1 回に約 400 回です（計算値）。OpenBLAS を 1 スレッドにした実行は 2.1 秒から 2.4 秒で、大きい積が 1 スレッドになる分だけ元より遅い値でした。利用者の計算機（18 スレッド）でも、同じ形の積を OpenMP のループから呼んで測っています。pthread 版の OpenBLAS を 18 スレッドのまま呼ぶと、1 スレッドにした場合と比べて、20×20（内側 3,000）の積が 15 倍、30×30（内側 5,000）の積が 8 倍遅くなりました。13×13 の積は変わりません。今の形は、構築の間この BLAS を 1 スレッドにして、大きい積を本体が OpenMP で分けます（8 節）。NVHPC 26.9 の BLAS（`libblas_lp64.a`）は、OpenMP 版の OpenBLAS 0.3.34 です。利用者の計算機で、`openblas_get_parallel()` が 2 を返し、スレッド数が `OMP_NUM_THREADS` に従うことを確かめました。この BLAS は、ループの中では 1 スレッドで動きます。切り替えの対象ではなく、変えていません。OpenBLAS の関数の探し方と、静的なライブラリーの扱いは、60 にあります。pthread 版の OpenBLAS を使う実行では、`-nt` が構築のスレッド数になります。`-nt` が 2 以上で OpenBLAS のスレッド数より小さいと、ループの外の大きい積は、改訂 13 までより少ないスレッドで動きます。
57. LLVM の OpenMP の実行時（libomp）は、チームを大きくするときに、自分で確保したメモリーの一部を指すポインターを持たなくなることがあります。LeakSanitizer は、プログラムの終わりにこれをリークとして報告します。Clang 18.1.3 と libomp 18.1.3 の組で、チームの大きさを順に変えるだけのプログラムを使って確かめました。4 コアのコンテナでは、8 スレッドの次に 9 スレッドのチームを作ると、8 個、11,520 バイトの報告が出ます。報告が出たのは、チームが「コアの数と `OMP_NUM_THREADS` の大きい方」の 2 倍を超えて大きくなったときです。境目は、既定で 9 スレッド、`OMP_NUM_THREADS=5` で 11 スレッド、`OMP_NUM_THREADS=16` で 33 スレッドでした。ブロックは 1 個 1,440 バイトで、個数は、それまでで最大のチームのスレッド数です。大きくするたびに失います。3、9、12 スレッドと変えると、3 個と 9 個で、合わせて 12 個でした。チームを小さくするだけのプログラム（8 の次に 4、64 の次に 63 と 3）と、最初から 9 スレッドのプログラムは、報告を出しません。最初の形（`1219cb9`）の試験のコメントは、原因をチームの縮小と書いていました。点検で誤りと分かり、`16951be` で直しています。`tests/run_exx_host_product.py` は、実行時の割り当て関数（`___kmp_allocate`）を通ったメモリーだけを、リークの検査から外します（`LSAN_OPTIONS` の `suppressions`）。最初の形は、ライブラリーの名前（`libomp`）で外していました。この指定では、並列の領域の中で試験や部品が確保したメモリーのリークも外れます。領域の中の割り当ては、どれも呼び出しの経路にライブラリーの関数を持つためです。今の指定がリークを報告することは、故意のリークをループの本体に入れた試験の写しで確かめました（GCC と Clang）。スレッド数の上限を付けた実行（`OMP_THREAD_LIMIT=2`）では、実行時が名前のない関数から確保したメモリーも報告されます（512 バイトが 2 個）。この実行だけは、リークの検査を止めています。GCC の実行時（libgomp）では、どの報告も出ません。
58. スレッドを使う実行の結果は、同じ実行ファイルでも、実行ごとに末尾の桁が変わります。クラウドのコンテナの CPU 版（4 スレッド、OpenBLAS の pthread 版）で、スレッド並列にする前の版（`b8a1363`）を同じ入力で 2 回実行しました。最初の構築の前に集めた密度行列の照合の値は、1.110e-16 と 5.551e-17 でした。構築ごとに表示する $K$ の非対称の値も、6.939e-17 と 7.286e-17 で違います。違いは、厳密交換の構築より前の SCF で生じています。原因は調べていません。1 スレッドでは、前の版と後の版の `.out` が、日付と時間の行を除いて一致しました。スレッドを使う実行を版の間で比べるときは、全エネルギーの差を、同じ実行ファイルの 2 回の差と比べます（48 と同じ考え方です）。Si の 4 スレッドの全エネルギーは、実行によって末尾が 1e-12 Hartree 変わりました。分子では幅が大きくなります。メタンと O2 を 4 スレッドで 4 回ずつ実行すると、全エネルギーの最大と最小の差は、前の版で 1.6e-10 と 4.7e-10 Hartree、後の版で 3.4e-10 と 6.6e-10 Hartree でした。交換エネルギーの違いは、どの実行の間でも 1e-12 Hartree 以下です。
59. GCC の OpenMP の実行時（libgomp）は、チームが小さくなると、余ったスレッドを終わらせます。次に大きいチームを作るときに、スレッドを作り直します。GCC 13.3.0 で確かめました。4 スレッドの領域を 50 回続けても、新しいスレッドはできません。2 スレッドの領域と 4 スレッドの領域を交互に 50 回ずつ実行すると、新しいスレッドが 100 個できました。LLVM の実行時（Clang 18.1.3 の libomp）では、どちらも 0 個です。構築の最初の形（`1219cb9`）は、ホストの積のチームを区間の数に、ループのチームを `threads` にしていました。区間が `threads` より少ない積があると、V 近傍ごとにスレッドを作り直します。当たるのは 2 種類の積です。乗算が 5e5 回以上で `threads` × 2.5e5 回未満の積と、 $C$ の長い方の辺が `threads` より短い積です。点検での実測（クラウドのコンテナ、4 スレッド、中央値）では、区間が 2 個の積のとき、V 近傍 1 個の時間が 108 から 132 マイクロ秒でした。積を分けない場合は 52 から 68 マイクロ秒、チームを `threads` にそろえた形は 43 から 58 マイクロ秒です。12 節で時間を測った系には、この範囲の積がありません。`16951be` で、2 つの領域のチームを `threads` にそろえました。スレッド i が区間 i を受け持つ割り当ては同じで、結果のビットも変わりません。積を分ける下限（2.5e5 回）は、そろえる前の形で決めた値です。そろえた後の形では、測り直していません（14 節）。
60. 弱い参照で関数の有無を調べる方法には、条件が 2 つあります。1 つ目は、位置独立のコードです。位置独立でないコードは、関数の番地を、リンクのときに決まる値として持ちます。リンクのときの共有ライブラリーに関数があると、その値は実行ファイルの PLT の項の番地になります。関数の番地を `NULL` と比べる式は、実行時のライブラリーにかかわらず真になります。関数を持たない同じ名前のライブラリーで実行すると、ない関数を呼んで異常終了します。模型で確かめました。同じ名前の共有ライブラリーを 2 つ作り、関数のある方でリンクして、ない方で実行します。位置独立でない実行ファイルは、GCC 13.3.0 でも Clang 18.1.3 でも、シグナル 11（終了コード 139）で終わりました。部品だけを `-fPIC` でコンパイルした実行ファイルと、PIE の実行ファイルは、関数がないと判定して正常に終わります。このため、`source/Exact_Exchange_Host.c` は、どのビルドでも `-fPIC` でコンパイルします（8 節）。2 つ目は、静的なライブラリーです。リンカーは、弱い参照のためには、アーカイブのメンバーをプログラムに入れません。OpenBLAS の `openblas_get_parallel` は、単独のメンバーにあります。静的な OpenBLAS をリンクしたプログラムでは、この関数だけが `NULL` になり、ほかの 2 つは入りました。クラウドのコンテナの `libopenblas.a`（0.3.26、pthread 版）と、利用者の計算機の CUDA 版（NVHPC 26.9 の `libblas_lp64.a`）で、この状態を確かめています。部品は、3 つの関数が全てあるときだけ OpenBLAS と判定します。静的な pthread 版の OpenBLAS は、スレッドを持ったまま残ります。CPU 版のビルドが OpenBLAS を選んだときは、関数をそのまま参照するので、静的なライブラリーでも見分けます（8 節）。点検では、リンクのオプション（`-Wl,--undefined=openblas_get_parallel`）で関数を引き込む案も出ました。採ったのは、定義を付ける方です。部品のコンパイルだけで決まり、リンクの行を変えずに済みます。macOS のリンカーは、弱い参照のシンボルも、リンクのときにライブラリーの中に見つかることを求めます [23]。リンクしたライブラリーが実行時にそのシンボルを持たないときに、番地が `NULL` になります [24, 25]。[23] は 2015 年の回答で、今のリンカーでの動作は確かめていません。macOS では弱い参照を使わず、定義を付けたビルドだけが OpenBLAS を見分けます。macOS でのリンクと実行も、確かめていません。
61. ThreadSanitizer を付けたプログラムは、アドレス空間のランダム化のビット数が大きい Linux で、起動のときに `unexpected memory mapping` を出して終わることがあります。Ubuntu のカーネル 6.5.0-25 が `vm.mmap_rnd_bits` を 28 から 32 に上げた後に、報告されました [26]。値を 28 に戻すと動く、と報告者が書いています。GitHub Actions のランナーでも、ubuntu-22.04 のイメージ 20240310.1.0 で、この値が 28 から 32 になりました。Clang の AddressSanitizer と UndefinedBehaviorSanitizer を付けたプログラムが異常終了した、という報告です [28]。報告者が書いた対処は `sudo sysctl vm.mmap_rnd_bits=28` で、同じ手順を CI に加えた例もあります [27]。管理者は、イメージの側で実行時に値を変える修正を配備した、と書いています。配備の日付は、読んだ本文からは分かりません。CI の ThreadSanitizer の 2 つの手順は、ランナーの値を出力してから、この設定をします（`.github/workflows/core.yml` の `exx` と `gpu-contracts`）。このキーを読めるのは root だけです（権限は 0600）。最初の形は値を `sudo` なしで読んだので、2 つの手順とも、設定の前に失敗しました（`9888b44` の実行）。`82eaa0a` で、読むときにも `sudo` を使う形にしています。新しい ThreadSanitizer は 30 までの値で動く、という記述が [26] にあります。新しい LLVM で直った、という記述は [28] にあります。どちらも、版の番号は書かれていません。クラウドのコンテナは、この値が 28 で、設定を変えずに実行できました。GitHub のランナーでも、設定の前の値は 28 でした（`82eaa0a` の実行 `45418534997` の 2 つの手順）。`exx` のジョブのランナーは、ubuntu-24.04 のイメージ 20261116.133.1 です。今のイメージでは、この設定は値を変えません。手順は、イメージの値が変わったときのために残しています。
62. クラウドのコンテナでは、同じ実行ファイルと同じ入力でも、構築 1 回の平均が実行ごとに最大 10% 変わります。2026-11-25 に、4 コアのコンテナで測りました。`a8d7910` で、4 スレッドのダイヤモンド（`diaz-p222`、構築 12 回）を、別の実行を 1 回挟んで 2 回実行しました。構築 1 回の平均は、4.93 秒と 5.43 秒でした。1 スレッドでも、中心の原子 1 個の寄与の時間は、同じ実行の中で標準偏差が 0.2 秒から 0.3 秒（平均の 2% から 4%）あります。版の間の数 % の差は、1 組の比較では見分けられません。前の版と後の版を何組か実行し、組の中の順序も入れ替えます。変えた処理の時間は、時間を表示する一時的なライブラリーで直接測ります（12 節の 2 番目の段階）。
63. CUDA 版の実行ファイルを別のディレクトリーに写しても、写したライブラリーは読まれません。CUDA 版の RPATH は、ビルドのディレクトリー（`build/exx-cuda/core/`）を `$ORIGIN/lib` より前に挙げます。ビルドし直すと、前の版の写しが新しい `libopenmx_core.so` を読みます。CPU 版の RPATH は `$ORIGIN/lib` が先なので、写しの `lib/` を読みます。実機の 5 番目の段階では、写した `libopenmx_core.so` と `libopenmx_host_simd.so` を `LD_PRELOAD` で先に読ませました。読んだファイルは、実行中のプロセスの `/proc/PID/maps` で確かめます。シェルのラッパーに `LD_TRACE_LOADED_OBJECTS=1` を付けると、bash のライブラリーが表示されるだけで、確かめられません。
64. ページ可能なホストのメモリーからデバイスへの `cudaMemcpy` は、写しが終わる前に戻ることがあります。CUDA の文書によると、この関数は、ページ可能なバッファーを DMA 用の領域に写した時点で戻ります。そのとき、行き先への DMA は終わっていないことがあります [29]。1 個ずつの積は、写しと積を同じ既定のストリームに載せるので、積は写しの後に動きます。呼び出しの全体では、積とループが OpenACC のキュー 1 のストリームで動きます。このストリームと既定のストリームの関係は、確かめていません（14 節）。そこで、部品の `upload` は、写しの後でデバイスの全体を待ちます（11 節）。
65. NVHPC の `nvc` は、デバイスのコードの積と和を、融合積和の命令にまとめることがあります。文書は、`-gpu` の副オプション `[no]fma` の説明に「default at -O1」と書いています [30]。`-O1` 以上では、指定しなくても融合積和を使うと考えられます。`move` は、積を丸めてから足す約束です（7.2 節）。融合積和では丸めが 1 回になるので、ホストの計算と末尾の桁が変わり得ます。部品は `-gpu=nofma` でコンパイルします。`source/Makefile` には、別の理由で `-gpu=nofma` を付ける単位（`Set_Orbitals_Grid.c`）が前からあります。複数の `-gpu` は、`-gpu=cc80` と並べても受け付けられました（GitHub Actions の GPU のコンパイル）。
66. 力を全エネルギーの中心差分と比べる試験には、落とし穴が 3 つあります。M7 の最初の形の道具は、3 つとも踏みました（9 節の M7）。1 つ目は、グリッドの原点です。`scf.fixed.grid` がないと、本体はグリッドの原点を原子の平均位置に合わせます（`truncation.c`）。原子 1 個を $h$ 動かすと、原点は $h/N$ 動きます（ $N$ は原子の数）。中心差分は、グリッドが動いた分の変化を含みます。メチルラジカルの C の z の成分（GGA-PBE、300 Ry、 $h$ = 0.002 Å）では、原点が動く形の差が 2.0e-5、原点を固めた形の差が 2.9e-5 Hartree/bohr でした。本体の力の試験（`Force_test.c`）は、動かした 2 つの計算を `MD_iter` が 2 と 3 の形で呼び、最初の構造のグリッドを使います。`MD.Type` の 2 段目以降も同じです。2 つ目は、刻みの 2 乗に比例する誤差です。同じ成分の中心差分は、 $h$ を変えると $c h^2$ の形でずれ、 $c$ は約 3.5 Hartree/bohr/Å² でした（450 Ry と 600 Ry）。 $h$ = 0.002 Å では 1.4e-5 Hartree/bohr になり、条件の限度を超えます。3 つ目は、近傍リストの境界です。最初の `dia`（格子定数 3.567 Å、2 番目の原子を (0.05, −0.03, 0.02) Å ずらした構造）では、2 番目の原子を y に −0.002 Å 動かすと、V サイトが 532 個から 530 個に減りました。原子 1 と 2 番目の原子の像の距離の 1 つが、V の半径（13.0 bohr）から 0.001 Å 以内にあったためです。リストが変わると $E_x$ は跳ぶので、この成分の差は −8.1e-4 Hartree/bohr でした。道具は、リストの大きさが変わった実行を失敗として表示します。作り直した道具は、元の構造の原点を `scf.fixed.grid` で渡し、 $h$ を 0.0005 Å にしました。`dia` は格子定数を 3.62 Å にして、どの距離もリストの半径から 0.028 Å 以上離しています。
67. 既存の GGA の力も、全エネルギーの微分から 1e-5 Hartree/bohr の水準でずれます。グリッドの原点を固め、 $h$ = 0.0005 Å で、メチルラジカルを GGA-PBE で比べました（クラウドの CPU 版）。C の z の成分の差は、300 Ry で 6.4e-5、450 Ry で 1.8e-5、600 Ry で 8.8e-6、900 Ry で 4.1e-5 Hartree/bohr です。カットオフに対して単調に減りません。600 Ry で、原点を z にグリッドの間隔の 4 分の 1 と 2 分の 1 だけずらすと、同じ成分の差は 8.3e-6 と 2.65e-5 になりました。3 つの原点の全エネルギーは、最大 2.3e-7 Hartree 違います。グリッドに対する原子の位置でエネルギーが変わる効果（SIESTA の文書の egg-box effect [32]）を、本体の力が全ては含まないためと考えられます。確かめてはいません。H（原子 3）の y の成分の差は、600 Ry と 900 Ry のどちらでも −1.06e-5 でした。こちらの原因も調べていません。どれも、厳密交換を含まない計算の値です。OpenMX のマニュアルの力の試験の例（`C2_GGA.dat`）でも、解析的な力と数値的な力の差は 7.7e-5 から 1.7e-4 Hartree/bohr です [31]。マニュアルは、差の大きさの目安を書いていません。HSE06 の差は、同じ構造と同じグリッドの GGA-PBE の差とほぼ同じです（600 Ry の C の z の成分で 8.68e-6 と 8.78e-6）。M7 の条件の 2 番目は、HSE06 の差から GGA-PBE の差を引いた値で判定します（9 節の M7）。
68. 厳密交換の力の時間の内訳は、系によって大きく変わります。力は、ブロックの重みを作る段階と、site の対を回る段階からなります（6.7 節）。site の形式の Si（基本単位胞、4×4×4 の k 点）では、1 プロセスの 3.89 秒のうち 3.46 秒が重みでした。Γ 点の形式のベンゼンでは、0.36 秒のうち 0.07 秒です（12 節の表）。MPI 並列での高速化の最初の形は、対の段階だけを細かく配り、重みは原子の持ち主が計算していました。この形の Si の力は、2 プロセスで 2.04 秒（重み 1.82 秒）で、原子の数より先では短くなりません。重みも ABF の範囲で配る形に直しました。プロセスに分ける箇所は、内訳を測ってから決めます。
69. 同時に走らせた `mpirun -np 1` の実行は、同じコアに束縛されます（クラウドのコンテナの Open MPI 4.1.6）。並べた実行は 1 つのコアを分け合い、どちらの時間も延びます。時間を測る実行は 1 件ずつにします。並べるときは `--bind-to none` を付けます。1 件の中のスレッドについては 47 番にあります。
70. クラウドのコンテナは、再起動すると同じ実行ファイルの時間が変わります。2026-11-25 の再起動の前後で、同じ版の Si の実行（`siz_p444`、1 プロセス 1 スレッド）の全体は、75.3 秒と 70.4 秒でした。版の比較は、同じ起動の中で、前の版と新しい版を続けて測ります。再起動は、実行中の計算も止めます。
71. 途中で止めた実行は、SCF の履歴の一時ファイル（`System.Name` に `.DFTSCF` を付けた名前）を残します。OpenMX は SCF の間この一時ファイルに追記し、最後に `Merge_LogFile` が `.out` に写して消します（`source/outputfile1.c:23`、`source/Merge_LogFile.c:52`、`134`）。同じディレクトリーで同じ名前の次の実行をすると、その `.out` に前の履歴も入ります。2026-11-25 の MPI の測定では、新しい版の Si の 1 プロセスの `.out` に、履歴が 2 つ入りました。1 つ目は 75 ステップで、この実行（74 ステップ）のものではありません。`.out` を比べる前に、残った一時ファイルを消します。
72. PBE0 の $R_c$ は、既定の値より大きくしないでください。k メッシュの計算の密度行列は、 $N_k$ 個の単位胞からなるスーパーセルを周期に持ちます。既定の $R_c$ は、半径 $R_c$ の球がこのスーパーセルと同じ体積になる値です [13]。それより大きい $R_c$ では、カーネルが密度行列の周期像との交換まで数えます。M8 の実測です。M7 の `dia`（2×2×2 の k 点）では、既定の 5.35 bohr の $E_x$ が −0.76 Hartree でした。 $R_c$ を 30 bohr にすると −9.23 Hartree、40 bohr では −15.87 Hartree です。どちらも、外側反復が 30 回で収束しませんでした。既定の値との差は、 $R_c$ のほぼ 2 乗で増えています（40 bohr と 30 bohr の差の比は 1.784、 $(40/30)^2$ は 1.778。計算値）。40 bohr の値は、 $\kappa$ の区間が 955 個の `b8cbe29` のものです。区間が 900 個の前の版では、同じ 30 回の構築の後の $E_x$ が 4.8e-4 Hartree 高く出ました。Cluster の計算では、セルが周期です。1 辺 10 Å の箱の分子では、既定が 11.72 bohr です。16 bohr を指定すると、全エネルギーが 3.0e-3 から 5.9e-3 Hartree 下がりました。改訂 24（`3d810d2`）から、本体はこの入力を拒否します（9 節の M8）。
73. 局所 RI の計量を対角化する巡回 Jacobi 法では、回転で消す要素を 0 に置いてください。前の版は、この要素も 2 つの行と列の回転で計算していました。この計算は、2 つの対角要素の大きい方の $\epsilon$ 倍程度の丸めの誤差を残します。反復の判定は、2 つの対角要素の相乗平均の $4\epsilon$ 倍です。対角要素が 16 倍ほど以上違うと、残る誤差が判定を上回り、回転が終わらないことがあります。`64352d9` の球ベッセル関数で、`exx-lri-gradient` の計量が末尾の桁で変わりました。そのうち 1 つの行列が、100 回の掃引で収束しませんでした。中心差分で 1 歩動かした、ABF が 280 個の対の行列です。要素は、 $a_{pp}$ = 3.3e-3、 $a_{qq}$ = 2.8e-5、 $a_{pq}$ = 1.1e-17 でした。最大の要素は、相乗平均の 170 $\epsilon$ 倍で止まっていました。`1b5a063` の回転では、同じ行列が 12 回の掃引で収束します。OpenMX がこの分解に入るのは、Cholesky 分解のピボットが 1e-7 を下回るときだけです（9 節）。
74. 1 つの原子の ABF を 2 つのプロセスで分けると、部分ごとの仕事を両方のプロセスが繰り返します。 $D$ の行の収集、最後の結果の加算、`W` の積が T ブロックを読む時間です（12 節）。改訂 24 までの決め方で、Si の基本単位胞の半分の構築を 2 プロセスで分けた例です。プロセス 0 が、2 番目の原子の 3 個の ABF も受け持ちました。この 3 個は、ABF 1 個あたり、残りの 66 個の 2.6 倍の時間を使っています。乗算の回数だけの見積もりでは、この仕事が見えません。改訂 25 の見積もりは、部分ごとの仕事を $q_a$ として数えます（6.7 節）。原子を切ると $q_a$ が加わるので、2 プロセスの範囲は原子の境界で切れるようになりました。
75. OpenMX の球ベッセル関数（`SphericalBesselSample`、下向きの漸化式）は、引数 $x$ に対して次数 $L+1.5x+10$ から始めます。1 つの値に $x$ に比例する手間がかかり、大きい $x$ では時間が延びる形です。PBE0 の V 近傍は遠くまで届き、二中心積分の表の引数は数千に達します。改訂 24 までは、この表が厳密交換の準備と力の時間の大半を占めていました（9 節の表）。`64352d9` は、 $x>2(L+1)$ で $j_0$ と $j_1$ から上向きの漸化式を使います。それ以下の $x$ では、前と同じ下向きの漸化式です。上向きの漸化式は、 $x$ が次数より小さいところで誤差が増えるので、この境界を置いています。下向きの漸化式だけの誤差は、 $x$ が 6000 までで包絡線 $1/x$ の 6.7e-13 倍まで増えました。値は末尾の桁で変わり、73 番の行列はこの変化で現れました。
76. 四面体法の DOS（`Dos.fileout`）の `.FermiSurf0.bxsf` は、化学ポテンシャルの近くのバンドだけを書きます。バンドの番号は、`BAND:` の行の値です。ガウス関数の方式（`DosGauss.fileout`）は、全てのバンドを 1 から書きます。ファイルを読む試験は、番号を行から読んでください。M8 の道具の最初の版は 1 から数え、23 eV の差を出しました。
77. 出力の $K(\mathbf k)$ を、メッシュの点で SCF の $K(\mathbf k)$ と比べる試験では、`Shift_K_Point` の位相を見込んでください（6.5 節）。類の別のセルから写した $D$ のため、差は最大の要素の 1e-12 倍ほどになります。丸めの水準の限度（1e-12）では、3×3×3 のダイヤモンドが不合格になりました。
78. 内側の SCF を緩い判定で止めるときは、応答の判定（`Exact_Exchange_Inner_Stop_Allowed`）に緩い値を渡さないでください。最初の形は、 $\tau$ をその関数の `scf.criterion` の代わりに渡していました。PBE0 の Si では、新しい $K$ の最初の残差ノルムが $\tau$ より小さく、内側の SCF が 3 ステップ目で止まりました。混合がまだ密度を動かしていないステップです。その密度で作った $K$ は前の $K$ とほとんど同じで、構築 1 回が無駄になりました。今の形は、応答の判定を `scf.criterion` のまま残し、 $\tau$ は既存の 2 つの条件だけに使います（6.6 節）。
79. 構築の後の単純混合は、重みが `scf.Min.Mixing.Weight` に落ちた状態で始まります。重みを決める規則は `OPENMX_MIXING_DENSITY_RATIO` で、残差ノルムの 2 乗が下限 1e-9 より小さいと重みが減る形です（13 節の 25）。 $K$ を作るステップの残差は収束した値なので、そのステップの混合で、重みが下限に落ちます。改訂 30 までの RMM-DIIS では、構築の後の 3 ステップの残差ノルムが、1 ステップに 0.1% しか減りませんでした。PBE0 の Si の 2 回目の構築の後では、8.977e-3、8.968e-3、8.959e-3 です。改訂 31 の始め直しで、この並びは 8.977e-3、8.968e-3 の 2 ステップになりました（6.6 節）。2 つの間の混合は、履歴を作り直すステップの単純混合で、同じ下限の重みを使います。
80. k メッシュの計算では、 $D$ の遠いブロックが小さくなりません（9 節の「省く形の測定」）。メッシュの $D(\mathbf R)$ は、スーパーセルの周期で同じ値を繰り返します。 $D$ の大きさだけで省く形は、Si（4×4×4）で 1 つのブロックも省けませんでした。 $V$ と T ブロックと $D$ の最大の積で見積もる形は、実際の値の 1 倍を超える組があり、上限になりません。
81. 緩い判定は、外側の閾値を満たした構築の後の内側の SCF には使わないでください。最初の形は、 $c\lvert\delta\rvert$ が `scf.criterion` より大きい間、内側の SCF をいつも緩めていました。緩い判定で止まった後の構築では、外側反復を終えません。`scf.criterion` が外側の閾値よりずっと小さいと、閾値を満たす構築が続いても、反復が終わりませんでした。`scf.criterion` 1e-12 の O₂ では、構築が既定の 4 回から 7 回に増えています。メタンの分子動力学（`scf.criterion` 1e-10）でも、各ステップの構築が 5 回から 6 回になりました。今の形では、O₂ の構築が 5 回、分子動力学の構築が各ステップ 5 回です（9 節）。

## 14. 未決事項

設計の時点で確かめられていない点です。該当する試験の結果が出たら、進捗ファイルに記録して利用者に報告します。

| 項目 | 内容 | 確かめる時期 |
|---|---|---|
| LRI の精度 | OpenMX の PAO と擬ポテンシャルでの精度は未確認です。ABACUS の結果 [6, 7] は別の基底でのものです。足りない場合は、角運動量の高い ABF を加える方法を検討します。M2 の実測では、結合距離の C–C、C–H、Si–Si で、2 原子の PAO の積の自己相互作用の和が 0.2% から 0.5% 不足しました。比べた相手は、s5p5d4f3 の積から作った ABF です。ABF を作る組に角運動量の 1 つ高い関数を 1 個加えると、不足は 0.02% から 0.06% に減り、ABF は C と Si で約 2 倍になります。M4 で、全エネルギーとギャップへの影響を測りました。ABF を作る組に、角運動量の 1 つ高い動径関数を 1 個から 3 個加えた一時的な実行ファイルを使っています。s2p2d1 の基底の分子（メタン、水、三重項の酸素）では、3 個加えたときの全エネルギーの変化が −0.07 から −0.47 mHartree でした。メタンと水の HOMO と LUMO の差は、0.004 eV 以下しか変わりません。1 個加えたときの変化は、3 個のときの 59% から 79% です。入力例の基底（C5.0-s1p1 と H5.0-s1、O6.0-s1p1 と H6.0-s1p1）の分子では、全エネルギーが 0.9 から 3.3 mHartree 変わります。数値は進捗ファイルの表にあります。M5 で、同じ一時的な変更を k メッシュ（4×4×4）の計算で測りました。s2p2d1 の基底では、1 個加えたときの全エネルギーの変化が、Si で原子あたり −8.0 meV、ダイヤモンドで −6.6 meV です。メッシュ上のギャップの変化は、0.01 eV 以下でした。Si は 2 個と 3 個でも測り、1 個のときとの差は 8e-6 Hartree 以下です。C5.0-s1p1 のダイヤモンドでは、1 個から 3 個で −6.6、−8.3、−8.7 mHartree と変わり、ギャップは 0.12 eV 下がります。Γ 点だけの小さいセルで見えた単調でない変化（M4）は、k メッシュの計算では出ていません。ABF の作り方を変えるかどうかは、利用者の決定を待っています。 | M4、M5（測定は済み） |
| ABF の個数 | M2 で測りました。既定のしきい値で、H6.0-s2p1 が 15 個、C6.0-s2p2d1 と Si7.0-s2p2d1 が 69 個、Si7.0-s2p2d1f1 が 140 個です。12 節の仮定（32 個）の約 2 倍でした。 | 済み（M2） |
| ABF のしきい値 | M4 で、`scf.exx.abf.pca` を 1e-3 から 1e-6 まで変えた全エネルギーを記録しました。s2p2d1 の分子では、既定値（1e-4）と 1e-6 の差が 1.5e-5 から 2.1e-5 Hartree です。1e-3 と既定値の差は、メタンで 2.9e-4 Hartree、Γ 点だけのダイヤモンドで 9.4e-3 Hartree でした。既定値を変えるかどうかは、利用者が決めます。 | 利用者の判断 |
| V 近傍の打ち切り | M4 で Γ 点の計算を測りました。周期像が到達距離に入らない分子では、6 から 16 bohr まで全エネルギーが変わりません。Γ 点だけの小さいセル（ダイヤモンドの 8 原子）では、16 bohr でも収束しません（13 節の 23）。s2p2d1 の基底のこのセルは、6 bohr で外側反復が発散しました（13 節の 24）。M5 で、k 点の計算を測りました。8×8×8 のメッシュで打ち切りを 8 bohr から 16 bohr にすると、全エネルギーは基本単位胞あたり、C5.0-s1p1 のダイヤモンドで 5.7e-7 Hartree、Si7.0-s2p2d1 の Si で 3.2e-7 Hartree 下がります。メッシュ上のギャップの変化は、どちらも 0.001 eV 以下です。Si の 6 bohr の全エネルギーは、8 bohr より 2.2e-5 Hartree 高い値でした。この 2 つの系の密なメッシュでは、既定値の 8 bohr で足ります。文献 [6]（arXiv 版の 4.1.1 節）は、同じ役割の打ち切り半径 $R_Q$ を 24 bohr から 8 bohr に縮めたときのギャップの誤差を、1 meV 未満としています（HSE06、Si、8×8×8 のメッシュ、DZP の基底）。粗いメッシュでは、打ち切りへの依存が残ります（13 節の 31）。s2p2d1 の基底のダイヤモンドと、ギャップの小さい系は測っていません。既定値を変えるかどうかは、利用者が決めます。 | 済み（M4、M5） |
| 動径の格子 | M2 で測りました。既定の設定で足ります。動径の区間を 4 倍にしたときの 2 原子の和の変化は相対 1e-8 で、波数の上限と刻みを変えたときの変化は 1e-11 以下でした。 | 済み（M2） |
| 打ち切りクーロンと $\kappa$ の刻み | M8 で確かめました。既定の刻みで式 (12) が正確なのは、 $R_c+3r_{\max}+v$ が 94 bohr より短い計算です（13 節の 19）。PBE0 は、この長さに合わせて区間の数を増やします（5.6 節）。利用者が 2026-11-25 に決めた形です。 | 済み（M8） |
| 外側反復の回数 | M4 で測りました。既定の閾値（1e-6 Hartree）で、分子は 4 回から 6 回、Γ 点だけのダイヤモンドの 8 原子は 6 回と 7 回でした。 $\delta$ の大きさは、1 回ごとに 0.04 倍から 0.46 倍になります（進捗ファイルの表）。M5 で、k 点の計算を測りました。 $\lvert\delta\rvert$ が 1e-6 Hartree を下回る回を数えています。基本単位胞のダイヤモンド（C5.0-s1p1）は、2×2×2 のメッシュで 4 回目、3×3×3 から 8×8×8 で 5 回目でした。s2p2d1 の基底では、ダイヤモンドが 5 回目、Si が 6 回目です（4×4×4 と 8×8×8）。この 3 つの系の $\delta$ は、1 回ごとに 0.05 倍から 0.16 倍になります。Γ 点を含まない 4×1×1 のメッシュのダイヤモンドは 10 回目で、比は 0.27 から 0.35 でした。2×2×2 のメッシュの s2p2d1 のダイヤモンドは、比が 0.07 から 0.65 の間で変わります（13 節の 24）。改訂 31 の外挿と緩い判定で、既定の閾値の PBE0 の Si（4×4×4）の構築は、7 回から 5 回になりました（9 節）。 | 済み（M4、M5） |
| 混合の履歴 | M4 で確かめました。`SCF_iter_shift` でステップ数を戻すと、混合を始め直せます。戻す量は混合の方式ごとに違います（6.6 節、13 節の 21）。改訂 31 で、RMM-DIIS、GR-Pulay、RMM-ADIIS の始め直しを 1 ステップ早めました（利用者の決定）。履歴を作り直すステップの単純混合は、まだ下限の重みを使います（13 節の 79）。 | 済み（M4） |
| 外挿と緩い判定の既定 | 2 つのキーワードの既定は、今の動作（0）です。既定で使う形にするかは、利用者が決めます（1.3 節）。確かめたのは、CPU 版の 6 つの入力、SCF の道具の 8 件、メタンの短い分子動力学と構造最適化です（9 節）。厳しい `scf.criterion` では、構築が 1 回増えることがあります（9 節の O₂）。 | 利用者の判断 |
| `scf.maxIter` の数え方 | M4 は、内側の SCF ごとに数える形で実装しました（6.6 節）。利用者の確認を待っています。 | M4 |
| 短距離部分の照合の精度 | Quantum ESPRESSO のルーチンとの照合は、倍精度で行いました。 $w$ が 4 から 14 の領域では、差が Fortran の丸め誤差（エネルギーで 1.5e-6 まで）に埋もれます。ルーチンを 4 倍精度でコンパイルした照合はしていません（M1）。 | 必要になったとき |
| GPU の速度 | M6 の実機の 3 つの段階で、FP64 の cuBLAS の積と GEMMul8 の積を測りました（11 節の 3 つの表）。RTX 5080 と 18 コアのホストの組です。書き直した構築（`857ede9`）では、積をデバイスに送ると、交換行列の段が 1 スレッドで 17% から 19% 短くなりました（積をホストに残す実行との比較）。18 スレッドでは、CUDA 版の段が CPU 版より 9% から 10% 長い値です。2 つの版は BLAS と MPI が違います。スレッドを使うホストで、1 個ずつの積を送る部品が構築を速くした測定はありません。GEMMul8 では、デバイスの積の時間が FP64 の 0.59 倍から 0.69 倍になります。段の時間の差は約 4% でした。デバイスの積の時間は、書き直した版で増えています（13 節の 55）。構築の残りの時間の内訳は、クラウドのコンテナで測りました（12 節）。改訂 14 で ABF のループをスレッド並列にした版は、実機の 4 番目の段階で測っています（12 節、11 節）。18 スレッドでは、積をデバイスに送る実行の段が、同じ CUDA 版で積をホストに残す実行より 24% から 27% 長い値です。1 スレッドでは、17% から 20% 短い値でした。配列をデバイスに置いたままの積は、形ごとの測定で、中心の原子 2 個分が 18 スレッドのホストの構築 1 回の約半分の時間でした（11 節）。この測定を基に、呼び出しの全体をデバイスで計算する形を改訂 17 で実装しました（9 節の 3 番目の段階）。この形の交換行列の段は、18 スレッドで、同じ CUDA 版でホストに残す実行より 18% から 21% 短い値です。1 スレッドでは、CPU 版の 0.12 倍から 0.13 倍でした（実機の 6 番目の段階。11 節）。 | 済み（M6、改訂 18）。残した案は「site の形式の構築の高速化」 |
| 金属 | 密度行列の減衰が遅い系での k 点の収束は未確認です。第 1 版では半導体と絶縁体で検証します。M5 では、半分詰まったバンドを持つ水素の鎖（1 次元、H5.0-s2p1）を計算しました。4 点と 6 点のメッシュの結果は、スーパーセルの Γ 点の結果と一致します。メッシュを密にしたときの収束と、3 次元の金属は調べていません。 | 必要になったとき |
| 力の site 形式 | M7 で、式 (16) と (17) を中心の原子ごとの形と site の形式に書き換えました。モジュールの試験は、site の形式を Γ 点の形式と、セルごとに違う密度行列では中心差分と比べます（`exx-exchange-centre-gradient`）。本体では、Band ソルバーのダイヤモンドで確かめています（9 節の M7）。 | 済み（M7） |
| メモリー | ランクごとに $D$ と $K$ を持つ形が、大きい系で成り立つかは未確認です。M5 で測ったのは、基本単位胞（2 原子）の k 点の計算です。site の形式の $D$ と $K$ のブロックは、s2p2d1 のダイヤモンドで 9,038 個、スピン 1 成分あたり各 11.7 MB でした。実行のメモリーの最大は 1,361 MB で、中心の原子 1 個の作業領域の計算値が 944 MB です（12 節。改訂 9 で 1,163 MB から直しました）。この作業領域は、構築を受け持つランクごとに要ります。M6 でも、原子の多い系は測っていません。 | 必要になったとき |
| site の形式の構築の高速化 | 実装していない方法が 3 つあります。メッシュの周期で列をまとめる（13 節の 30）。小さい密度行列のブロックを省く。 $K(\mathbf k)$ を保存して、SCF のステップごとのフーリエ和を省く（12 節）。M5 の受け入れ条件には要らないので、手を付けていません。ブロックを省く方法は結果を変え、しきい値のキーワードが要ります。採るかどうかは利用者が決めます（1.3 節）。M6 で、V 近傍ごとの項を ABF 1 個ずつ計算する形に書き直しました（6.4 節、12 節）。結果は丸め誤差の範囲で同じです。クラウドのコンテナでは、構築 1 回が Si で 0.65 倍、s2p2d1 のダイヤモンドで 0.57 倍になりました（1 スレッド）。利用者の計算機では、交換行列の段が 0.50 倍から 0.56 倍です（11 節）。ABF のループのスレッド並列は、改訂 14 で実装しました。 $D$ の行の収集と $K$ への加算のスレッド並列は、改訂 15 です（9 節の「構築の高速化」、12 節）。構築については、実装していない方法が 2 つ残っています。V 近傍ごとの計算をデバイスに移す（11 節）。大きい配列を使い回す（13 節の 55）。前者は、利用者が選んだ作業の残りです（9 節）。後者は、効果の上限を測り、加えないことにしました（12 節）。map の形式（`BlockMatrix`）の収集と加算は、呼び出したスレッドのままです。改訂 25 で、 $K(\mathbf k)$ を構築ごとに 1 回だけ作る形と、半分の構築を実装しました（`f5d746d`、`d768793`。6.4 節、6.5 節）。最初に挙げた 3 つの方法のうち、ブロックを省く方法は入れていません。改訂 31 の前に組の寄与を測り、利用者が入れない形を選びました（9 節、13 節の 80）。メッシュの周期でまとめる方法は、改訂 27 で V 近傍に入れました（`1626dc4`、6.4 節）。列（ $S_A$ ）は、改訂 28 でまとめています（`51cd508`）。 | 構築の高速化（9 節） |
| `scf.Generation.Kpoint none` | 既存のコードが、ハイブリッド汎関数でない計算でも異常終了します（13 節の 28）。原因は調べていません。直すまで、対称性で減らしたメッシュでは HSE06 を試験できません。 | 必要になったとき |
| CUDA 版での実行 | 確かめる対象は 3 つです。CUDA 版が通るホスト上の経路（Cluster と、Band のブロック巡回配置）、M6 で書いた行列積の部品、デバイスの密行列ソルバーの経路です。3 つとも、実機の 2 番目の段階の 53 件で確かめました（9 節の M6）。構築を書き直した版（`857ede9`）も、3 番目の段階の 53 件で確かめています。 | 済み（M6） |
| デバイスに送る積の下限と、残す空き | 1e9 回と 512 MiB は最初の値です（11 節）。この値での時間は、実機の 3 つの段階で測りました。2 プロセスが 1 台を共有する実行（`siz-p444-np2`）は、40 個の積を全てデバイスで計算しています。下限を変えたときの時間は測っていません。18 スレッドのホストでは、積を送る利点が見えていません。書き直した版では、CUDA 版の段が CPU 版より長くなりました（11 節）。GEMMul8 に変えても、段の時間は約 4% しか変わりません。同じ CUDA 版で積をホストに残す 18 スレッドの実行は、スレッド並列の版で測りました（実機の 4 番目の段階。11 節）。積を送る実行の段は、残す実行より 24% から 27% 長く、1 スレッドでは 17% から 20% 短い値です。2 番目の段階の版（`3d51128`）では、ホストの処理が短くなり、送る実行の段が 31% から 39% 長くなりました（実機の 5 番目の段階。11 節）。スレッドの多いホストでは、1 個ずつの積を送る形が構築を遅くします。3 番目の段階の形では、デバイスで計算する呼び出しの段が、18 スレッドのホストより短くなりました（実機の 6 番目の段階。11 節）。下限と空きは、呼び出しの全体の形でも同じ値です。改訂 29 からは、呼び出しの全体を、全ての積の乗算の和で下限と比べます（11 節）。値を変えた測定はしていません。 | 必要になったとき |
| デバイスの密行列ソルバーの経路 | M6 の実機の段階で、 $K$ の加算と密度行列の収集を書きました（6.3 節、6.5 節）。実機の 2 番目の段階で、しきい値を 1 に下げた 19 件が、CPU 版と 2.1e-9 Hartree 以内で合いました（9 節の M6）。3 番目の段階の 19 件は、1.6e-9 Hartree 以内です。行列の次数は 8 と 26 です。しきい値を下げずにこの経路に入る大きさ（Band は次数 800 以上、Cluster は 400 以上）の HSE06 は、道具の実行に入れていません。この大きさでの加算と収集の時間も未測定です。 | 必要になったとき |
| MAGMA の経路と gpusolver2 の経路 | ハイブリッド汎関数の計算は、この 2 つの経路を使わず、CPU の固有値ソルバーで解きます（6.5 節）。結果は変わらず、対角化が遅くなります。MAGMA の経路に $K$ を足すには、GPU が 2 台以上ある計算機での試験が要ります。 | 必要になったとき |
| 固有値ソルバーの中の GEMMul8 | デバイスの固有値ソルバーは、既定で GEMMul8 を使うデバイスがあります（11 節）。RTX 5080 は、その 1 つです（FP32 と FP64 の処理量の比は 64）。実機の 2 番目の段階で、既定の実行と FP64 の実行（`NAME-dfp64`）を比べました。全エネルギーの差は、メタンで 7.2e-11 Hartree、Si で 0 です。3 番目の段階では、4.5e-10 Hartree と 0 でした。メタンには、デバイスに送る積がありません。同じ計算をする 2 つの実行（`ch4-gpu` と `ch4-host`）の間にも、2.5e-10 Hartree の差がありました。行列の次数は 8 と 26 で、大きい行列では測っていません。 | 済み（M6） |
| 厳密交換の積での GEMMul8 | M6 の実機の段階で、入力に `scf.gemmul8.enable on` を書いた計算だけが GEMMul8 を使う形を書きました（11 節）。差を測るための切り替えで、形は作業者が決めています。実機の 2 番目の段階で、FP64 との差と時間を測りました。全エネルギーの差は最大 2.5e-11 Hartree、デバイスの積の時間は FP64 の 0.42 倍から 0.53 倍です。書き直した版（3 番目の段階）では、差が最大 3.1e-11 Hartree、時間が 0.59 倍から 0.69 倍でした。利用者の判断が 2 つ残っています。この形を残すか、厳密交換のためのキーワードに分けるか。FP64 の弱い GPU で、厳密交換の積も GEMMul8 を既定にするか。2026-11-24 に、測定の値を添えて確認しました。推奨は、今の形のままです。 | 利用者の判断 |
| XC の GPU 経路の時期 | 8 節の表は M6、9 節の M8 の一覧は「GPU での XC の計算」としていて、食い違っていました。9 節の M6 の目標は行列積と加算で、XC を含みません。改訂 25 で、CUDA 版の XC の GPU 経路に、ハイブリッド汎関数の準局所部分を入れました（`3fafc96`。8 節、11 節）。利用者の「すべてやってください」の指示に入ると、作業者が判断した結果です。CPU 版の結果は変わりません。デバイスの値は、FMA とデバイスの数学ライブラリーのため、ホストと末尾の桁で違うことがあります（11 節）。差の大きさは、利用者の計算機での確認を待っています。 | 実機の確認（改訂 25） |
| NVIDIA の文書で確かめられなかった点 | `cublas<t>gemm` の戻り値の表、`cudaMalloc` の戻り値の一覧、`cudaErrorMemoryAllocation` の説明、`cudaMemGetInfo` の説明は、取得したページが途中で切れていて読めませんでした [18, 19]。メモリーの不足を表す状態コードの扱いは、リポジトリーの既存のコード（`source/openmx_gpu_error.c:35-58`、`source/openmx_gpu_runtime.h:126`）に合わせています。実機の 3 つの段階では、デバイスの呼び出しの失敗が 1 回も起きませんでした。メモリーの不足は、障害注入と容量の判定でだけ試しています。実際の確保の失敗が返す状態コードは、実機で見ていません。 | 必要になったとき |
| 大域状態のゲート（CUDA の構成） | 実機の 2 番目の段階で、利用者の計算機（clang 21.1.8、NVHPC 26.9）で全体を実行しました（`a7fbfb8`）。288 個の翻訳単位を解析し、ゲートが比べる 7 種類に増えた項目はありません。`FFTW3_PREFIX` の指定が要ります（13 節の 52）。`857ede9` では実行していません。`a7fbfb8` から変わった `source/` のファイルはありません。改訂 14 のコード（`1219cb9`）は、クラウドのコンテナで、変更の前後を比べました（clang 18.1.3、CPU の構成）。対象は、変えた翻訳単位とその周りの 5 個です。ゲートが比べる 7 種類に、増えた項目はありません。実機の 4 番目の段階で、利用者の計算機で全体を実行しました（`1219cb9`）。289 個の翻訳単位を解析し、増えた項目はありません。関数は 3,508 個です（`a7fbfb8` では 3,500 個）。点検の指摘を直したコミット（`9888b44` まで）は、クラウドのコンテナで、同じ 5 個の翻訳単位を比べました。関数が 96 個から 99 個に増え、ゲートが比べる 7 種類に、増えた項目はありません。この版の利用者の計算機での全体の実行は、済んでいません。 | 済み（`1219cb9`）。修正のコミットは、実機での次の確認のとき |
| 構築のスレッド並列の実機での確認 | 最初の形（`1219cb9`）は、実機の 4 番目の段階で確かめました（11 節）。確かめたのは 3 つです。18 スレッドでの時間。CUDA 版の結果と時間。NVHPC のコンパイラー（`nvc`）が、OpenMP を有効にしたビルドで、弱い参照の宣言と 2 つのループを受け付けること。点検の指摘を直した版と 2 番目の段階は、`3d51128` で、実機の 5 番目の段階で確かめました（11 節）。18 スレッドでの時間と結果、大域状態のゲートの全体、`nvc` が `-fPIC` を付けた部品をコンパイルすることです。18 スレッドの交換行列の段は、`1219cb9` の 0.80 倍から 0.91 倍でした。 | 済み（実機の 5 番目の段階） |
| ホストの積を分ける下限 | 1 回の呼び出しの乗算の平均の下限（2.5e5 回）は、最初の値です（8 節）。クラウドのコンテナの 4 スレッドでだけ測りました。値を変えたときの時間と、18 スレッドでの時間は測っていません。時間を測った系（Si とダイヤモンド）には、この下限で区間の数が決まる積がありません。ループの外の最も小さい積は Si の $W$ （乗算 7.0e7 回）で、18 スレッドに分けても、1 回の呼び出しの乗算が下限の約 16 倍あります（計算値）。下限が効くのは、乗算が 5e5 回から `threads` × 2.5e5 回までの積です。値は、チームのスレッド数をそろえる前の形で決めました（13 節の 59）。そろえた後の形では、測っていません。 | 構築の高速化（9 節） |
| 自前のスレッドを持つ BLAS の見分け | 構築の間 1 スレッドにするのは、pthread 版の OpenBLAS です（8 節、13 節の 56 と 60）。CPU 版のビルドが OpenBLAS を選んだときは、動的なライブラリーも静的なライブラリーも見分けます。ほかのビルド（`OPENMX_CPU_LINALG` が `CUSTOM` の CPU 版、ネイティブのビルド）は、動的な OpenBLAS だけを見分けます。見分けられない静的な pthread 版の OpenBLAS と、自前のスレッドを持つほかの BLAS は、スレッドを持ったまま、OpenMP のループから呼ばれます。点検では、静的な pthread 版の OpenBLAS 0.3.26 を、4 つの OpenMP のスレッドから同時に呼びました（クラウドのコンテナ）。結果のビットは、順に呼んだ場合と同じでした（70 回の繰り返し）。時間は、1 回ずつの実行で、動的なライブラリー（1 スレッドに切り替わる）の 0.20 秒に対して 0.67 秒です。本体の実行では測っていません。このリポジトリーのビルドがつなぐ MKL と AOCL は、1 スレッドの版か OpenMP の版です（`cmake/OpenMXHostLinalg.cmake`）。この 2 つでは実行していません。配列の置き場所で結果のビットが変わる BLAS では、同じスレッド数の実行の $K$ が、実行ごとに変わり得ます（8 節）。MKL と AOCL がこれに当たるかは、確かめていません。 | 必要になったとき |
| スレッドを使う実行の再現性 | 同じ実行ファイルの結果が、実行ごとに末尾の桁で変わります（13 節の 58）。違いは、厳密交換の構築より前の SCF で生じます。スレッド並列にする前の版でも起きます。原因は調べていません。 | 必要になったとき |
| 呼び出しの全体をデバイスで計算する形 | 改訂 17 で実装し、実機の 6 番目の段階で確かめました（9 節、11 節）。受け入れ表の 53 件は、全て条件を満たしました。文書で確かめられなかった 2 点は、未確認のままです。OpenACC のキューのストリームが、既定のストリームと同期するか。`acc_get_cuda_stream` が `NULL` を返す場合があるか。部品は、写しの後でデバイスの全体を待ち、`NULL` を失敗として扱うので、結果はこの 2 点によらない作りです（13 節の 64）。複数のプロセスが 1 台を共有するときは、プロセスごとに $D$ 、 $K$ 、T ブロックをデバイスに置きます。2 プロセスの実行（`siz-p444-np2`）では、交換行列の段が 1 プロセスの場合とほぼ同じ時間でした（11 節）。 | 済み（改訂 18） |
| デバイスの既定の動作 | 変えていません（作業者の判断。9 節）。CUDA 版は、引き受けられる呼び出しと積を、全てデバイスで計算します。呼び出しの全体の形の交換行列の段は、18 スレッドでも、ホストに残す実行より 18% から 21% 短い値でした（実機の 6 番目の段階。11 節）。既定が遅くなり得る場合は、2 つ残っています。1 つ目は、呼び出しの配列がデバイスに入らないときです。このときは、前の版と同じく大きい積を 1 個ずつ送るので、18 スレッドのホストより遅くなると考えられます（実機の 5 番目の段階の 1 個ずつの形の測定から）。この場合の時間は測っていません。2 つ目は、複数のプロセスが 1 台を共有するときです。2 プロセスでは、CPU 版との比が 1 プロセスの 0.13 倍から 0.25 倍になりました。どちらの場合も、既定を変えるかどうかは、その大きさの計算で測ってから利用者に確かめます（1.3 節）。 | 済み（改訂 18）。残る 2 つの場合は、必要になったとき |
| V サイトの打ち切りでの不連続 | 対の距離が V サイトの半径（ $\max(r_A,r_B)$ に `scf.exx.v.cutoff` を足した値）をまたぐと、 $E_x$ と力が不連続に変わります。構造最適化と分子動力学では、リストが変わる前後でエネルギーが保存しません。M7 の最初の形の `dia` では、この跳びが −8.1e-4 Hartree/bohr の差として現れました（13 節の 66）。打ち切りを滑らかにする方法は、実装していません。C サイトの半径（ $r_A+r_B$ ）では、軌道の積が半径で 0 になるので、跳びは小さいと考えられます。確かめてはいません。 | 必要になったとき |
| 厳密交換の力の計算の時間 | `Exact_Exchange_Gradient` の行列積は、ホストの BLAS だけで計算します（7.4 節）。中心の原子ごとの導関数は、区間に分けていないので 1 スレッドです。改訂 21 で、重みと対をプロセスに配る形にしました（6.7 節）。M7 の 3 つの系では、力の計算が 1 プロセスで 0.11 秒から 1.8 秒でした。最も長いのは `dia`（C サイト 198 個、V サイト 478 個）です。M6 の表の系は、利用者の計算機の 1 スレッドで測りました（2026-11-25、`9e01d40`）。HSE06 の力は、`siz-p444` で 3.9 秒と 4.0 秒、`diaz-p222` で 19.6 秒と 20.2 秒です（CPU 版と CUDA 版）。`diaz-p222` の CUDA 版では、12 回の構築をデバイスで計算した段（19.5 秒）とほぼ同じ時間でした。プロセスを増やすと短くなります（`siz-p444` の 4 プロセスで 1.3 秒）。PBE0 の力は、`siz-p444-pbe0` で 26.8 秒と 26.6 秒、`diaz-p222-pbe0` で 38.5 秒と 39.5 秒でした（同じ順）。どちらも、デバイスで計算した構築の段（23.9 秒と 28.5 秒）より長い値です。HSE06 より長いのは、V 近傍が遠くまで届くためと考えられます。改訂 25 で、重みをデバイスで計算する形にしました（11 節）。利用者の計算機の `4577361` の確認では、CUDA 版の PBE0 の力が、`siz-p444-pbe0` で 5.30 秒、`diaz-p222-pbe0` で 5.71 秒です。重みの後に残る 2.0 秒と 1.7 秒は、主にホストで回る対の段と考えられます。改訂 26 で、重みを半分の構築の形で作り（`2198f55`）、対の段をスレッドに分けました（`7a79b99`）。この 2 つは、利用者の計算機では測っていません（9 節）。 | 済み（改訂 26） |
| 既存の力と全エネルギーの食い違い | GGA-PBE の力は、グリッドの原点を固めても、全エネルギーの中心差分と 1e-5 Hartree/bohr の水準で違います（13 節の 67）。メチルラジカルの C の z の成分は、600 Ry でグリッドの原点を z にずらすと、差が 8.3e-6 から 2.65e-5 まで変わりました。カットオフを 900 Ry に上げた差は 4.1e-5 です。H の y の成分の差は、600 Ry と 900 Ry のどちらでも −1.06e-5 でした。原因は調べていません。M7 の条件の 2 番目は、同じ構造と同じグリッドの GGA-PBE の差と比べて判定しました（9 節の M7）。この判定の方法は、利用者が 2026-11-25 に決めました。 | 済み（判定の方法、M7）。原因の調査は必要になったとき |
| CUDA 版の力 | `Exact_Exchange_Gradient` は、CUDA 版でも CPU 版と同じ関数です。改訂 24 までの重みの行列積は、ホストの BLAS だけで計算する形でした（7.4 節）。M7 の試験は CPU 版だけで行いました。2026-11-25 に、M6 の表の実行（`9e01d40`）の出力で、CUDA 版の力を確かめました。同じ系の CPU 版の力との差は、HSE06 の 44 件で最大 7.3e-10 Hartree/bohr でした。PBE0 の 44 件では、最大 1.3e-11 Hartree/bohr です。目安にした 1e-6 Hartree/bohr（MPI 並列での高速化の条件）より、3 桁以上小さい値でした。HSE06 の最大は `o2` の実行です。全エネルギーの差と同じく、内側の SCF が止まる位置の違いから来ていると考えられます。中心差分の試験は、CUDA 版では行っていません。改訂 25 から、重みの積は構築と同じデバイスで計算します（11 節）。その後に、利用者の計算機で 2 回確かめました（`4577361`、`27a952a`。9 節）。CUDA 版と CPU 版の力の差は、9.0e-12 と 4.0e-12 Hartree/bohr 以内です。後の確認には、原子を動かして力が 0 でない 2 つの系が入っています。 | 済み（M8 の実機の確認、改訂 25） |
| 元素ごとの表の時間 | 準備のうち、ABF の生成と動径の変換は、全てのプロセスが同じ計算をします（6.7 節）。計算は実行の間に 1 回です。時間は、1 元素の入力で 0.55 秒（C5.0-s1p1）から 0.83 秒（Si7.0-s2p2d1）でした。2 元素の入力では 1.09 秒から 1.46 秒です。12 節の測定で、前の版が標準エラーに表示した値を使いました（クラウドのコンテナ）。ベンゼン（2 元素）では、準備の 1.5 秒から 1.7 秒のうち、1.41 秒から 1.46 秒がこの部分です。この部分は、プロセスを増やしても短くなりません。分ける方法は 2 つ考えられます。動径の変換の波数の点をプロセスに配って和を取る形と、球ベッセル関数の表を元素の間で共有する形です。前の形は入り口の追加が要り、後の形は結果のビットが変わります。どちらも実装していません。改訂 25 で、球ベッセル関数を上向きの漸化式で計算する形にしました（`64352d9`。13 節の 75）。準備の全体は、メタン（2 元素）で 0.14 秒、ダイヤモンド（1 元素）で 0.12 秒と 0.13 秒になりました（9 節の表）。元素ごとの表は、その一部です。 | 済み（改訂 25） |
| プロセスに分けた計算のスレッド並列 | 構築は、プロセスの範囲の中をスレッドに分けます。改訂 25 から、力の重みも、中心の原子の ABF のループを区間に分けます（`d6f7db1`）。改訂 26 で、準備の対のループと力の対のループも、スレッドに分けました（`7a79b99`、6.7 節）。改訂 25 までは、`-nt` を増やしても、この 2 つは短くなりませんでした。 | 済み（改訂 26） |
| 力の重みのメモリー | 力の計算の間、全てのプロセスが、全ての原子の T ブロックと V ブロックと同じ大きさの配列を持ちます（6.7 節）。改訂 20 までは、原子 1 個分でした。12 節の Si512 の仮定で ABF を 69 個にすると、6.1 GB です（計算値）。大きい系では、原子を何個かずつ処理して和を取る形が要ると考えられます。実装していません。改訂 25 のデバイスの重みでは、呼び出し 1 回のデバイスのメモリーが最大 1296.8 MB でした（`diaz-p222-pbe0-gpu`、11 節）。 | 大きい系を計算するとき |
| 4 プロセスを超える MPI の時間 | クラウドのコンテナは 4 コアです。5 プロセス以上の時間と、プロセスとスレッドを組み合わせた 18 コアの時間は、測っていません。利用者の計算機での測定は、ほかのプロセスが CPU を占有していないことを確かめてから行います（利用者の指示、2026-11-25）。 | 利用者の計算機を次に使うとき |
| CUDA 版での ABF の範囲 | デバイスの呼び出しは、原子の途中で切れた範囲も受け取ります（6.4 節）。2026-11-25 に、利用者の計算機で確かめました（`9e01d40`）。`tools/exx_gpu_checks.py` の `siz-p444-np4` と `siz-p444-pbe0-np4` では、4 プロセスが Si の原子を約半分ずつ受け持ちます。HSE06 では、40 回の呼び出しが全てデバイスで計算され、全エネルギーは CPU 版と表示の全ての桁が同じでした。PBE0 でも、44 回の呼び出しが全てデバイスで計算され、全エネルギーは CPU 版と 1e-12 Hartree 以内で合いました。結果は 9 節の「MPI 並列での高速化」にあります。 | 済み（M8 の実機の確認） |
| PBE0 の CUDA 版 | CUDA 版も `scf.XcType PBE0` を受け付けます（`c9e26e7`）。カーネルが入るのは V ブロックとその微分だけで、どちらもホストが計算します。構築の経路は HSE06 と同じです。2026-11-25 に、M6 と同じ形の試験を利用者の計算機で行い、条件を満たしました（9 節の M8）。CUDA 版と CPU 版の全エネルギーの差は、最大 3.0e-11 Hartree です。 | 済み（M8） |
| PBE0 の $R_c$ の上限 | `scf.exx.coulomb.cutoff` に既定の値より大きい値を書いた入力は、拒否します（`3d810d2`）。その計算は密度行列の周期像を数え、外側反復が収束しないことがあるためです（13 節の 72）。利用者が 2026-11-25 に、拒否する形を選びました（9 節の M8）。表示は、書いた値と既定の半径です。 | 済み（M8） |
| ABF の分け方の係数 | `build_costs` の係数は、1.4、3.6、500、16、290 です。クラウドのコンテナの CPU 版で、OpenBLAS を 1 スレッドで使って測りました（12 節）。利用者の計算機と、スレッドを使う構築では、まだ測っていません。デバイスで計算する構築も、同じ範囲を使います。係数が違っても範囲は全ての ABF を含むので、結果は丸めの範囲でしか変わりません。 | 必要になったとき |
| SCF の後の出力の CUDA 版 | 出力の半分の構築は、CUDA 版ではデバイスの関数表を使います（6.5 節）。デバイスの呼び出しの数は、出力の前の値に戻します。確かめたのは CPU 版だけです。 | 次の実機の確認 |
| 出力のまとめない枝 | V 近傍をまとめない site の形式の計算では、SCF の $K(\mathbf R)$ をそのまま使います。この枝に入る `scf.Generation.Kpoint none` は、既存のコードで動きません（13 節の 28）。枝の手順は、まとめた系で枝を強制した一時的な版で通しました（9 節の M8）。 | `none` を直したとき |

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

GEMMul8 の切り替えで加えたコードの場所です。`source/` と `tests/` の行番号は `dd3a680` のものです。このコミットでは、上の 2 つの表の行のうち、`source/Exact_Exchange_Device.c` の 31 行より後ろ、`source/Exact_Exchange.c` の 97 行より後ろ、`source/Input_std.c` の 93 行より後ろ、`source/DFT.c` の 2063 行より後ろがずれています。`tools/` の行番号は `a7fbfb8` のものです。このコミットでは、上の表の `STEERED = ` が 130 行にあります。

| 項目 | 場所 | 検索語 |
|---|---|---|
| 橋渡しを通す積と、GEMMul8 の積の数え方 | `source/Exact_Exchange_Device.c:197-203`、`220` | `native_before`、`by_gemmul8` |
| 橋渡しの宣言 | `source/Exact_Exchange_Device.c:36-39` | `openmx_gemmul8Dgemm(`、`openmx_gemmul8NativeCalls(` |
| 部品の状態の指定と数 | `source/openmx_exact_exchange_device.h:53`、`55` | `int gemmul8;`、`gemmul8_products` |
| 入力の指定を読む場所 | `source/Input_std.c:934`、`5190` | `exx_gemmul8` |
| 結合層での受け渡し | `source/Exact_Exchange.c:99`、`171` | `state->device.gemmul8`、`Exact_Exchange_Device_Gemmul8` |
| 標準出力の行 | `source/DFT.c:2064-2067` | `GEMMul8 computed` |
| 模型の橋渡しと試験 | `tests/exx_device_product_harness.c:310`、`726` | `openmx_gemmul8Dgemm(`、`products_through_the_bridge` |
| 道具の `NAME-g8` | `tools/exx_gpu_checks.py:144`、`148`、`370-381` | `"g8"`、`INPUT_LINES` |
| 道具が本体に渡すスレッド数 | `tools/exx_gpu_checks.py:237`、`245`、`321-326`、`tools/exx_scf_checks.py:284`、`tools/exx_kpoint_checks.py:376` | `"-nt"`、`threads_asked` |
| 本体がスレッド数を決める場所 | `source/openmx.c:249`、`286` | `openmp_threads_num` |

構築のスレッド並列（改訂 14）で加えたコードの場所です。行番号は、点検の後の修正の最後のコミット（`82eaa0a`）のものです。改訂 14 のコミットで、上の表の行のうち、`source/Exact_Exchange.c` の 22 行より後ろ、`modules/exx/include/openmx/exx/exx.h` の 30 行より後ろ、`.github/workflows/core.yml` の 99 行より後ろ、`tools/check_gpu_compilation.py` の 62 行より後ろがずれています。

| 項目 | 場所 | 検索語 |
|---|---|---|
| 関数表のループと区間の数 | `modules/exx/include/openmx/exx/exx.h:62-63` | `(*parallel_for)`、`int threads;` |
| 区間の数とループの包み | `modules/exx/include/openmx/exx/dense_backend.hpp:48`、`61`、`modules/exx/src/dense_backend.cpp:123`、`128` | `pieces() const`、`void parallel_for`、`DenseBackend::run` |
| 例外と呼び出しの回数の記録 | `modules/exx/src/dense_backend.cpp:40` | `openmx_exx_loop_call` |
| 区間ごとの作業用の配列 | `modules/exx/src/exchange.cpp:97` | `struct PieceWork` |
| ABF を区間に分ける場所 | `modules/exx/src/exchange.cpp:150`、ループは `159`、`186`、`232`、`248` | `const std::size_t pieces = `、`backend.parallel_for` |
| 本体のループ | `source/Exact_Exchange_Host.c:169`、`source/Exact_Exchange.c:297` | `Exact_Exchange_Host_Parallel_For`、`static int ExactExchange_Parallel_For` |
| スレッドごとの積の数 | `source/openmx_exact_exchange_host.h:82`、`96`、`source/Exact_Exchange_Host.c:151`、`156` | `OpenMXExactExchangeLoopProducts`、`OpenMXExactExchangeHostProducts`、`Exact_Exchange_Host_Note_Product`、`Exact_Exchange_Host_Loop_Dgemm` |
| スレッドの数と BLAS のスレッド | `source/Exact_Exchange.c:736`、`738`、`776` | `omp_get_max_threads`、`Exact_Exchange_Host_Blas_Single`、`Exact_Exchange_Host_Blas_Restore` |
| ホストの積の分割 | `source/Exact_Exchange_Host.c:88`、`98`、`source/Exact_Exchange.c:290` | `Exact_Exchange_Host_Pieces`、`Exact_Exchange_Host_Dgemm` |
| 1 回の呼び出しの乗算の下限 | `source/Exact_Exchange_Host.c:86` | `OPENMX_EXX_HOST_PIECE_WORK` |
| OpenBLAS の関数の参照（定義と弱い参照） | `source/Exact_Exchange_Host.c:46`、`52-54`、`cmake/OpenMXCPU.cmake:216` | `OPENMX_EXX_HOST_BLAS_OPENBLAS`、`__attribute__((weak))` |
| 部品を位置独立のコードにする場所 | `cmake/OpenMXCPU.cmake:214`、`source/Makefile:724` | `CMAKE_C_COMPILE_OPTIONS_PIC`、`-fPIC -c Exact_Exchange_Host.c` |
| ビルドの一覧 | `source/Makefile:316`、`723`、`tools/check_gpu_compilation.py:74` | `Exact_Exchange_Host` |
| OpenMP ありのコンパイルの検査 | `tools/check_gpu_compilation.py:79`、`80` | `Exact_Exchange.openmp.c`、`Exact_Exchange_Host.openmp.c` |
| 模型の試験 | `tests/run_exx_host_product.py`、`tests/exx_host_product_harness.c` | `exx host product` |
| 試験の登録 | `CMakeLists.txt:209`、`.github/workflows/core.yml:286`、`300` | `exx-host-product`、`run_exx_host_product.py`、`--thread-sanitizer` |
| モジュールのスレッドのサニタイザー | `modules/exx/CMakeLists.txt:22`、`modules/exx/tests/CMakeLists.txt:79`、`.github/workflows/core.yml:105` | `OPENMX_EXX_THREAD_SANITIZER`、`LABELS threads`、`thread_sanitize` |
| 試験のループ | `modules/exx/tests/test_parallel.hpp`、`modules/exx/tests/test_dense_backend.cpp:151` | `threaded_loop`、`check_loops` |
| デバイスの上の積を形ごとに測る道具 | `tools/exx_device_shapes.c` | `checks that failed` |

構築の高速化の 2 番目の段階（改訂 15）で加えたコードの場所です。行番号は `1b91c7b` のものです。このコミットで、上の表の `modules/exx/src/exchange.cpp` の行は、8 行より後ろがずれています。

| 項目 | 場所 | 検索語 |
|---|---|---|
| 区間の数 | `modules/exx/src/exchange.cpp:105` | `piece_count` |
| ABF を区間に分ける場所 | `modules/exx/src/exchange.cpp:163`、ループは `172`、`199`、`245`、`261` | `const std::size_t pieces = `、`backend.parallel_for` |
| site の形式の $D$ の収集 | `modules/exx/src/exchange.cpp:441`、ループは `450` | `std::unique_ptr<double[]> density(` |
| site の形式の $K$ への加算 | `modules/exx/src/exchange.cpp:507`、ループは `514`、ブロックの番号は `551` | `b%pieces != piece`、`block_` |
| Γ 点の形式の $K$ への加算 | `modules/exx/src/exchange.cpp:685`、ループは `688` | `the pieces of the rows` |
| Γ 点の形式の $D$ の行の収集 | `modules/exx/src/exchange.cpp:763`、ループは `768` | `gathered_rows` |
| 収集を区間に分けて呼ぶ場所 | `modules/exx/src/exchange.cpp:953`、`996` | `piece_count(system,atom,backend)` |
| 試験の、形式ごとのループの数 | `modules/exx/tests/test_exchange.cpp:217`、`233` | `enum class Form`、`loops_of` |
| 試験の、 $K$ の同じブロックに足す組 | `modules/exx/tests/test_exchange.cpp:250` | `shared_pairs` |

構築の高速化の 3 番目の段階（改訂 17）で加えたコードの場所です。行番号は `09e054a` のものです。`ae5c54e` で、上の表の `modules/exx/src/exchange.cpp` の行は、全て 2 行以上後ろにずれています。

| 項目 | 場所 | 検索語 |
|---|---|---|
| デバイスの関数表と添字の形 | `modules/exx/include/openmx/exx/exx.h:29-90`、関数表のメンバーは `142` | `OpenMXExxDeviceIndex`、`OpenMXExxDevice`、`const OpenMXExxDevice *device;` |
| 呼び出しの全体の入口 | `modules/exx/src/exchange.cpp:1554`、`1615`、`modules/exx/src/c_api.cpp:295`、`353` | `add_exchange_blocks_range`、`add_exchange_gamma_range` |
| デバイスの仕事と配列 | `modules/exx/src/exchange.cpp:876`、`880`、`1031`、`1066` | `struct DeviceStop`、`class DeviceWork`、`device_sizes`、`struct DeviceArrays` |
| 中心の原子 1 個の計算 | `modules/exx/src/exchange.cpp:1116` | `void device_centre` |
| 呼び出しの範囲とホストへの戻し | `modules/exx/src/exchange.cpp:1240` | `bool device_range` |
| 2 つの形式の計画 | `modules/exx/src/exchange.cpp:474`、`499`、`509`、`1274`、`1326` | `device_rows`、`device_first`、`device_second`、`class DeviceBlocks`、`class DeviceGamma` |
| ホストのメモリーで表したデバイス | `modules/exx/tests/test_device.hpp` | `exchange-device` |
| 本体のデバイスの表 | `source/Exact_Exchange_Device.c:42`、`317`、`398`、`452`、`483`、`495` | `OPENMX_EXACT_EXCHANGE_DEVICE_CALLS`、`ExactExchange_Device_Accept`、`ExactExchange_Device_Upload`、`ExactExchange_Device_Move`、`ExactExchange_Device_Finish`、`Exact_Exchange_Device_Table` |
| キューとストリーム | `source/openmx_exact_exchange_device.h:51`、`source/Exact_Exchange_Device.c:270`、`352` | `OPENMX_EXACT_EXCHANGE_DEVICE_QUEUE`、`ExactExchange_Device_Stream_Back`、`acc_get_cuda_stream` |
| 呼び出しの数え方 | `source/Exact_Exchange_Device.c:282`、`source/Exact_Exchange.c:787` | `Exact_Exchange_Device_Call_End` |
| 関数表に渡す場所 | `source/Exact_Exchange.c:759` | `Exact_Exchange_Device_Table(&state->device` |
| 全プロセスの集計と標準出力 | `source/Exact_Exchange.c:176`、`833`、`source/DFT.c:2039`、`2072` | `Exact_Exchange_Device_Calls`、`device calls of process 0`、`calls of the exchange build on the device` |
| 融合積和を止める規則 | `source/Makefile:722`、`tools/check_gpu_compilation.py:74` | `-gpu=nofma -c Exact_Exchange_Device.c` |
| 模型の試験 | `tests/run_exx_device_product.py:24`、`tests/exx_device_product_harness.c:1064`、`1289`、`tests/fixtures/exx_device_sdk/openacc.h` | `OPENMX_EXACT_EXCHANGE_DEVICE_HOST_MOVES`、`calls_on_the_device`、`calls_that_fail`、`acc_get_cuda_stream` |
| CI の Clang の実行 | `.github/workflows/core.yml:299` | `CC=clang python3 tests/run_exx_device_product.py` |
| 実機の道具の呼び出しの照合 | `tools/exx_gpu_checks.py:163`、`352` | `calls=`、`calls_add_up` |

M7 で加えたコードの場所です（改訂 19）。行番号は `c43bab9` のものです。`modules/exx` の 4 つのコミット（`0d5ad4b` から `da77f7c`）の後に、構築の高速化のコミットが `modules/exx/src/exchange.cpp` を変えているので、この表の行は `c43bab9` で数え直しました。

| 項目 | 場所 | 検索語 |
|---|---|---|
| 二中心積分の変位についての微分 | `modules/exx/include/openmx/exx/two_center.hpp:195`、`215`、`236`、`modules/exx/src/two_center.cpp:393`、`515` | `class GradientPair`、`gradient_pair`、`two_center_gradient_matrix` |
| 式 (7) の分解と、計量の微分の重み | `modules/exx/include/openmx/exx/lri.hpp:128`、`160`、`modules/exx/src/lri.cpp:314` | `class MetricFactorization`、`metric_weights` |
| 式 (18) の対ごとの連鎖律 | `modules/exx/include/openmx/exx/lri.hpp:229`、`modules/exx/src/lri.cpp:640` | `pair_gradient` |
| 式 (16) と (17) の中心の原子ごとの形 | `modules/exx/include/openmx/exx/exchange.hpp:197`、`203`、`208`、`modules/exx/src/exchange.cpp:1760`、`1780` | `struct CentreGradient`、`centre_gradient_blocks`、`centre_gradient_gamma` |
| 原子の位置についての $E_x$ の微分 | `modules/exx/include/openmx/exx/forces.hpp:30`、`modules/exx/src/forces.cpp:57`、`152`、`158` | `class PositionGradient`、`add_gamma`、`add_blocks` |
| C の入口 | `modules/exx/include/openmx/exx/exx.h:307`、`314`、`modules/exx/src/c_api.cpp:359`、`383` | `openmx_exx_add_position_gradient_gamma`、`openmx_exx_add_position_gradient_blocks` |
| モジュールの試験 | `modules/exx/tests/test_two_center.cpp:406`、`modules/exx/tests/test_lri.cpp:694`、`modules/exx/tests/test_exchange.cpp:753`、`859`、`2096`、`modules/exx/tests/test_c.c:610` | `two_center_gradient`、`lri_gradient`、`exchange_gradient`、`exchange_centre_gradient`、`exchange_forces` |
| 結合層の状態と中心の原子の範囲 | `source/Exact_Exchange.c:65`、`715`、`756` | `double *gradient;`、`ExactExchange_Centres`（改訂 21 で `ExactExchange_Abfs` に置き換えました。下の表） |
| 結合層の微分 | `source/Exact_Exchange.c:875`、`source/openmx_exact_exchange.h:194` | `Exact_Exchange_Gradient`、`forces of the exact exchange` |
| `DFT` からの呼び出し | `source/DFT.c:2973` | `exx_gradient = Exact_Exchange_Gradient` |
| `Force` の引数と加算 | `source/Force.c:3024`、`4870`、`source/openmx_common.h:2979` | `exact_exchange_gradient`、`force(EXX)` |
| HIP 版 | `source/backends/hip/src/Force.c:2734`、`source/backends/hip/src/DFT.c:2891` | `(void)exact_exchange_gradient;` |
| `MD.Type` の拒否 | `source/Input_std.c:5134` | `MD.Type that needs the stress` |
| 本体から切り出した `Force` の試験 | `tests/run_force_lifetime_cpu.py:45`、`89`、`126` | `expected_gradient` |
| 試験の道具 | `tools/exx_force_checks.py` | `grid_origin`、`COMPARISONS`、`PROCESSES` |

MPI 並列での高速化で加えたコードの場所です（改訂 21）。行番号は、`a468823` の時点のものです。

| 項目 | 場所 | 検索語 |
|---|---|---|
| 準備の分け方 | `modules/exx/include/openmx/exx/system.hpp:106`、`123`、`modules/exx/src/system.cpp:144`、`154` | `struct PrepareShare`、`prepare_system`、`pair%share.parts` |
| 原子ごとのブロックの配列 | `modules/exx/include/openmx/exx/system.hpp:129`、`131`、`133`、`modules/exx/src/system.cpp:271`、`280`、`288` | `table_elements`、`get_tables`、`set_tables` |
| ABF の範囲の構築 | `modules/exx/include/openmx/exx/exchange.hpp:140`、`182`、`modules/exx/src/exchange.cpp:122`、`128`、`1613`、`1746` | `struct CentrePart`、`centre_parts`、`add_exchange_blocks_abfs`、`add_exchange_gamma_abfs` |
| 乗算の回数の見積もり | `modules/exx/include/openmx/exx/exchange.hpp:149`、`modules/exx/src/exchange.cpp:1645` | `abf_costs`（改訂 25 で `build_costs` と `divide_abfs` に置き換えました。下の表） |
| ABF の範囲の重み | `modules/exx/include/openmx/exx/exchange.hpp:253`、`256`、`modules/exx/src/exchange.cpp:1912`、`1947`、`modules/exx/include/openmx/exx/forces.hpp:54`、`57`、`modules/exx/src/forces.cpp:48`、`89`、`98` | `centre_gradient_blocks`、`centre_gradient_gamma`、`add_weight_rows`、`add_block_weights_gamma_abfs`、`add_block_weights_blocks_abfs` |
| 対を 1 回ずつ回る微分 | `modules/exx/include/openmx/exx/forces.hpp:105`、`modules/exx/src/forces.cpp:230` | `PositionGradient::add_pairs` |
| C の入口 | `modules/exx/include/openmx/exx/exx.h:214`、`224`、`225`、`226`、`268`、`269`、`280`、`334`、`389`、`392`、`405`、`modules/exx/src/c_api.cpp:290`、`300`、`311`、`326`、`402`、`411`、`424`、`497`、`559`、`572`、`586` | `openmx_exx_prepare_part`、`openmx_exx_atom_table_elements`、`openmx_exx_get_atom_tables`、`openmx_exx_set_atom_tables`、`openmx_exx_abf_count`、`openmx_exx_add_gamma_abfs`、`openmx_exx_abf_costs`（改訂 25 で `openmx_exx_divide_abfs` に置き換えました）、`openmx_exx_add_blocks_abfs`、`openmx_exx_add_block_weights_gamma_abfs`、`openmx_exx_add_block_weights_blocks_abfs`、`openmx_exx_add_position_gradient_pairs` |
| モジュールの試験 | `modules/exx/tests/test_system.cpp:411`、`modules/exx/tests/test_exchange.cpp:1617`、`1708`、`2344`、`2961`、`modules/exx/tests/test_c.c:215` | `The preparation in shares`、`compare_ranges_with_c`、`check_divided_preparation`、`The weights of ranges of ABFs`、`check_abf_ranges`、`divided_work` |
| 結合層のブロックの和 | `source/Exact_Exchange.c:387`、`539`、`560` | `ExactExchange_Add_Up_Blocks`、`openmx_exx_prepare_part` |
| 結合層の ABF の範囲 | `source/Exact_Exchange.c:798`、`844`、`1001` | `ExactExchange_Abfs` |
| 結合層の構築と力 | `source/Exact_Exchange.c:886`、`966`、`1026`、`1043`、`1063` | `openmx_exx_add_blocks_abfs`、`shares of the`、`openmx_exx_add_block_weights_blocks_abfs`、`openmx_exx_add_position_gradient_pairs`、`for the weights of the T and V blocks` |

利用者の 2 つの指示による高速化で加えたコードの場所です（改訂 25）。行番号は、`92f78c7` の時点のものです。

| 項目 | 場所 | 検索語 |
|---|---|---|
| ABF の分け方 | `modules/exx/include/openmx/exx/exchange.hpp:179`、`182`、`192`、`modules/exx/src/exchange.cpp:1964`、`1972`、`2029` | `struct BuildCosts`、`v_product_time`、`build_costs`、`divide_abfs` |
| 半分の構築 | `modules/exx/include/openmx/exx/exchange.hpp:158`、`165`、`236`、`241`、`modules/exx/src/exchange.cpp:1937`、`2084`、`2186`、`2192` | `add_exchange_blocks_half_abfs`、`mirror_blocks`、`add_exchange_gamma_half_abfs`、`mirror_gamma` |
| 重みのデバイス | `modules/exx/include/openmx/exx/exchange.hpp:332`、`335`、`modules/exx/src/exchange.cpp:1415`、`1471`、`1552`、`2375`、`2394`、`modules/exx/src/forces.cpp:101`、`115` | `device_centre`、`device_derivatives`、`device_gradient_range`、`device_gradients_blocks`、`device_gradients_gamma`、`add_block_weights_gamma_abfs` |
| C の入口 | `modules/exx/include/openmx/exx/exx.h:297`、`381`、`384`、`390`、`391`、`modules/exx/src/c_api.cpp:411`、`623`、`638`、`654`、`664` | `openmx_exx_divide_abfs`、`openmx_exx_add_gamma_half_abfs`、`openmx_exx_add_blocks_half_abfs`、`openmx_exx_mirror_gamma`、`openmx_exx_mirror_blocks` |
| 球ベッセル関数 | `modules/basis/include/openmx/basis/detail/bessel.hpp:78`、`modules/exx/src/two_center.cpp:52` | `spherical_bessel_values`、`bessel_values` |
| Jacobi 法の回転 | `modules/exx/src/lri.cpp:42`、`51` | `Rutishauser`、`diagonalize` |
| ハイブリッド汎関数の準局所部分 | `modules/xc/include/openmx/xc/hybrid_kernels.h:500`、`502`、`548`、`572`、`source/openmx_xc_hybrid_kernels.h` | `second channel takes the result of the first`、`openmx_xc_kernel_wpbeh_short_range`、`openmx_xc_kernel_pbe_screened_hybrid`、`openmx_xc_kernel_pbe_global_hybrid` |
| XC の GPU 経路 | `source/Set_XC_Grid.c:65`、`340`、`581`、`739` | `SXG_XC_PBE_HYBRID`、`Set_XC_Grid_GPU(XC_P_switch, exx_alpha` |
| 結合層の半分の構築と ABF の範囲 | `source/Exact_Exchange.c:299`、`912`、`919`、`981` | `openmx_exx_mirror_blocks`、`ExactExchange_Abfs`、`openmx_exx_divide_abfs`、`openmx_exx_add_blocks_half_abfs` |
| 結合層の $K(\mathbf k)$ | `source/Exact_Exchange.c:220`、`385`、`1381`、`source/openmx_exact_exchange.h:294`、`source/DFT.c:2041`、`2064` | `Exact_Exchange_Band_Sums`、`ExactExchange_Forget_Band_Matrices`、`Exact_Exchange_Add_Band_Hs`、`K(k) of the k points` |
| 結合層の重みのデバイス | `source/Exact_Exchange.c:1065`、`1119`、`1188`、`1192` | `Exact_Exchange_Gradient`、`Exact_Exchange_Device_Table`、`the weights on the device of process 0` |
| モジュールの試験 | `modules/exx/tests/test_exchange.cpp:1692`、`3137`、`3366`、`3556`、`modules/exx/tests/test_c.c:249`、`modules/xc/tests/test_c.c:79` | `compare_ranges_with_c`、`check_division`、`check_half_build`、`check_device_gradients`、`openmx_exx_divide_abfs`、`the inline hybrid kernels` |
| 道具 | `tools/exx_gpu_checks.py:185`、`366`、`tools/check_gpu_compilation.py:76` | `FORCE_LIMIT`、`out_forces`、`Set_XC_Grid.c` |

改訂 26 で加えたコードの場所です。行番号は、`7a79b99` の時点のものです。

| 項目 | 場所 | 検索語 |
|---|---|---|
| 力の重みの半分の形 | `modules/exx/include/openmx/exx/exchange.hpp:341`、`344`、`modules/exx/src/exchange.cpp:1688`、`1758`、`2553`、`2579`、`modules/exx/src/forces.cpp:145`、`160` | `half_gradient_blocks`、`half_gradient_gamma`、`class DeviceBlocks`、`class DeviceGamma`、`add_block_weights_gamma_half_abfs`、`add_block_weights_blocks_half_abfs` |
| 準備と力の対のスレッド | `modules/exx/include/openmx/exx/system.hpp:131`、`modules/exx/src/system.cpp:144`、`209`、`modules/exx/include/openmx/exx/forces.hpp:137`、`modules/exx/src/forces.cpp:298` | `prepare_system`、`in_pieces`、`add_pairs` |
| C の入口 | `modules/exx/include/openmx/exx/exx.h:236`、`470`、`473`、`494`、`modules/exx/src/c_api.cpp:297`、`595`、`608`、`628` | `openmx_exx_prepare_part_pieces`、`openmx_exx_add_block_weights_gamma_half_abfs`、`openmx_exx_add_block_weights_blocks_half_abfs`、`openmx_exx_add_position_gradient_pairs_pieces` |
| 結合層 | `source/Exact_Exchange.c:658`、`1146`、`1149`、`1185` | `openmx_exx_prepare_part_pieces`、`openmx_exx_add_block_weights_blocks_half_abfs`、`openmx_exx_add_position_gradient_pairs_pieces` |
| モジュールの試験 | `modules/exx/tests/test_exchange.cpp:2429`、`2532`、`3919`、`modules/exx/tests/test_system.cpp:489`、`modules/exx/tests/test_c.c:396`、`560` | `half_weight_ratio`、`The pairs in the pieces of a backend`、`check_device_half_gradients`、`The pairs of a share in the pieces`、`openmx_exx_add_block_weights_gamma_half_abfs` |

改訂 27 で加えたコードの場所です。行番号は、`1626dc4` の時点のものです。

| 項目 | 場所 | 検索語 |
|---|---|---|
| V 近傍のまとめ | `modules/exx/include/openmx/exx/system.hpp:82`、`194`、`modules/exx/src/system.cpp:324` | `struct VFolding`、`fold_v_neighbours` |
| 力の対 | `modules/exx/src/forces.cpp:29`、`37` | `pair_v_sites`、`held_v_block` |
| C の入口 | `modules/exx/include/openmx/exx/exx.h:260`、`279`、`modules/exx/src/c_api.cpp:384`、`392` | `openmx_exx_prepared_v_neighbours`、`openmx_exx_fold_v_neighbours` |
| 結合層 | `source/openmx_exact_exchange.h:46`、`source/Input_std.c:5204`、`source/Exact_Exchange.c:688`、`721` | `fold_period`、`V sites folded by the period` |
| モジュールの試験 | `modules/exx/tests/test_exchange.cpp:4386`、`modules/exx/tests/test_c.c:657` | `exchange_folding`、`folded_work` |

改訂 28 で加えたコードの場所です。行番号は、`51cd508` の時点のものです。

| 項目 | 場所 | 検索語 |
|---|---|---|
| 列のまとめ | `modules/exx/include/openmx/exx/system.hpp:85`、`222`、`modules/exx/src/system.cpp:494` | `struct VFolding`、`fold_columns` |
| 構築の列と符号 | `modules/exx/src/exchange.cpp:86`、`99`、`159`、`586`、`771` | `struct GroupColumns`、`find_runs`、`density_columns`、`folded_sites`、`const GroupColumns& columns` |
| デバイスの写し | `modules/exx/src/exchange.cpp:1340` | `column_moves` |
| C の入口 | `modules/exx/include/openmx/exx/exx.h:295`、`modules/exx/src/c_api.cpp:403` | `openmx_exx_fold_columns` |
| 結合層 | `source/openmx_exact_exchange.h:50`、`source/Input_std.c:5211`、`source/Exact_Exchange.c:697`、`732`、`828` | `fold_antiperiodic`、`columns of D folded`、`not a point of the folded mesh` |
| モジュールの試験 | `modules/exx/tests/test_exchange.cpp:4359`、`4526`、`4685`、`modules/exx/tests/test_c.c:727` | `column_cases`、`2b. A period longer`、`without their signs`、`pass < 2` |

改訂 29 で変えたコードの場所です。行番号は、`079d22c` の時点のものです。

| 項目 | 場所 | 検索語 |
|---|---|---|
| 乗算の和 | `modules/exx/src/exchange.cpp:1374`、`1419`、`1452`、`1810`、`1854` | `struct DeviceSizes`、`z.products = 2.0`、`device_multiplications` |
| モジュールの試験 | `modules/exx/tests/test_device.hpp:82`、`233`、`modules/exx/tests/test_exchange.cpp:3196` | `all_products` |
| 部品の説明 | `source/Exact_Exchange_Device.c:317`、`source/openmx_exact_exchange_device.h:33` | `multiplications is the sum`、`multiplications in all` |
| 道具 | `tools/exx_gpu_checks.py:225`、`230` | `def add`、`device=0 < calls` |

改訂 30 で変えたコードの場所です。行番号は、`c52e41e` の時点のものです。

| 項目 | 場所 | 検索語 |
|---|---|---|
| 出力の準備 | `source/Exact_Exchange.c:1762` | `Exact_Exchange_Prepare_Outputs` |
| $D$ の写し | `source/Exact_Exchange.c:1610`、`1631` | `ExactExchange_Find_Class`、`ExactExchange_Unfold_Density` |
| 加算と比較 | `source/Exact_Exchange.c:1680`、`1712`、`1912` | `ExactExchange_Output_Sum`、`ExactExchange_Check_Outputs`、`Exact_Exchange_Add_Output_Hk` |
| 解放 | `source/Exact_Exchange.c:1923` | `Exact_Exchange_Release_Outputs` |
| 呼び出し | `source/DFT.c:2718`、`2857`、`source/Band_DFT_kpath.c:360`、`source/Band_DFT_Dosout.c:572`、`3265`、`3579` | `exx_outputs`、`Exact_Exchange_Add_Output_Hk` |
| 拒否 | `source/Input_std.c:5135` から `5148` | `Band.dispersion with the cluster solver`、`COHP.calc.flag` |
| 道具 | `tools/exx_output_checks.py:238`、`331` | `def read_bxsf`、`def table` |

改訂 31 で変えたコードの場所です。行番号は、`6c9a592` の時点のものです。

| 項目 | 場所 | 検索語 |
|---|---|---|
| 混合の始め直し | `source/Exact_Exchange.c:1525` | `Exact_Exchange_Mixing_Offset` |
| 外挿 | `source/Exact_Exchange.c:1404`、`1347`、`1370`、`464` | `Exact_Exchange_Extrapolate`、`ExactExchange_Reserve_History`、`ExactExchange_Solve_Small`、`ExactExchange_Release_History` |
| 外側反復の判定 | `source/Exact_Exchange.c:1334`、`1588` | `Exact_Exchange_Outer_Continue`、`Exact_Exchange_Inner_Criterion` |
| 呼び出し | `source/DFT.c:1468`、`1679` | `exx_loose`、`Exact_Exchange_Extrapolate` |
| 入力 | `source/Input_std.c:1061` | `scf.exx.outer.extrapolation`、`scf.exx.inner.criterion.ratio` |
| 道具 | `tools/exx_scf_checks.py:330`、`592` | `def accel`、`accel minus default` |

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
| 改訂 11 | 2026-11-24 | M6 の実機の最初の段階の結果と、その間に書いたコードの反映です。M6 は完了していません。9 節の M6 に、`38992bc` の 18 件の結果と、ハイブリッド汎関数でない計算の CUDA 版どうしの比較を書きました。11 節に、段ごとの時間を書きました。18 スレッドのホストでは、積をデバイスに送る利点がありませんでした。11 節の「GEMMul8 は使っていません」の段落は、切り替えの説明に置き換えています。入力に `scf.gemmul8.enable on` を書いた計算だけが GEMMul8 を使い、既定は FP64 のままです。改訂 10 までは、切り替えの方法を利用者が決めるとしていました。この改訂では、作業者が既存のキーワードに従う形を選び、残る 2 つの判断を 14 節に書いています。試験の道具は、スレッド数を本体に渡していませんでした。12 節の M4 と M5 の「2 スレッド」を、OpenBLAS の数と書き直しています。7.4 節に結合層での受け渡し、8 節の M6 の表に 2 行、9 節の M6 に道具の `NAME-g8`（53 件）を加えました。11 節の模型の試験は、確かめることが 5 つになっています。13 節には 43 から 49 の 7 項、付録 A にはコードの場所、参考文献には [20] を加えました。14 節は、4 行を測定の結果に合わせて直しています。 |
| 改訂 12 | 2026-11-24 | M6 の実機の 2 番目の段階の結果と、交換行列の構築の書き直しの反映です。M6 は完了していません。9 節の M6 に、`a7fbfb8` の 53 件の結果を書きました。デバイスの密行列ソルバーの経路と GEMMul8 の切り替えは、この段階で初めて実機で動いています。11 節には、1 スレッドと 18 スレッドの時間、GEMMul8 でのデバイスの積の時間を加えました。構築の書き直し（`857ede9`）は、6.4 節、7.2 節、11 節、12 節に書いています。V 近傍ごとの 4 つの項を、中心の原子の ABF を 1 個ずつ計算する形です。行列積は、V 近傍 1 個につき 5 回から 1+4n 回になりました。クラウドの CPU 版では、構築 1 回が 0.57 倍から 0.65 倍の時間です。12 節に、書き直す前と後の時間の内訳を加えました。13 節には、45 と 46 への追記と、50 から 54 の 5 項があります。14 節では、GPU の速度、CUDA 版での実行、デバイスの密行列ソルバーの経路、2 つの GEMMul8 の行、大域状態のゲート、構築の高速化の行を改めました。書き直した版の実機の結果は、この改訂にありません。文献に [21] と [22] を加えています。 |
| 改訂 13 | 2026-11-24 | M6 の実機の 3 番目の段階の結果と、M6 の完了の反映です。9 節の M6 に、構築を書き直した版（`857ede9`）の 53 件の結果と、完了の判断を書きました。11 節には、書き直した版の 1 スレッドと 18 スレッドの時間の表を加えています。交換行列の段は、書き直す前の 0.50 倍から 0.62 倍でした。同じ節に、デバイスの積の時間が増えた点の説明を書いています。12 節に、利用者の計算機での比を加えました。13 節は、51 に 2 つの BLAS の 18 スレッドの結果、54 に置き場所の指定の結果を書き、55（結果の配列への最初の書き込み）を加えています。14 節は、9 つの行を実機の結果に合わせました。 |
| 改訂 14 | 2026-11-24 | 交換行列の構築のスレッド並列の反映です。利用者が、M7 より先に行う作業として、構築の高速化を選びました。9 節に、この作業の項（M6 と M7 の間）を加えています。M7 は途中で止めています。6.4 節に、中心の原子の ABF を区間に分ける形を書きました。7.2 節には、関数表の `parallel_for` と `threads` を加えています。7.4 節に、ホストの積とループの部品を書きました。8 節に、本体が渡すループ、ホストの積の分割、BLAS のスレッドの扱い、BLAS への要求、コンパイルの検査を書きました。11 節には、ループの中の積をデバイスに送らないこと、実機の 4 番目の段階の結果、デバイスの上の積を形ごとに測った結果を加えています。12 節に、クラウドのコンテナで測った時間の表と、前の版との結果の比較を書きました。4 スレッドの構築 1 回は、Si で 0.61 倍、s2p2d1 のダイヤモンドで 0.55 倍です。実機の 18 スレッドでは、交換行列の段が、前の版の 0.24 倍から 0.33 倍になりました。最初の形（`1219cb9`）を点検して受けた指摘と、その修正（`b80a3c7` から `9888b44` の 6 つのコミットと、CI の手順を直した `82eaa0a`）も、この改訂に入れています。13 節には 56 から 61 の 6 項、14 節には 4 つの項、付録 A にはコードの場所、文献には [23] から [28] を加えました。14 節の「GPU の速度」「site の形式の構築の高速化」「大域状態のゲート」「デバイスに送る積の下限と、残す空き」の 4 項も改めています。修正の後の版は、利用者の計算機で確かめていません。 |
| 改訂 15 | 2026-11-25 | 構築の高速化の 2 番目の段階（`1b91c7b`）の反映です。1 本の配列の site の形式と Γ 点の形式で、 $D$ の行の収集と $K$ への加算を、ABF の区間と同じ数に分けました。6.4 節に、加算の分け方と、 $K$ のビットが変わらない理由を書きました。7.2 節には、形式ごとのループの数と、加えた試験を書いています。9 節の「構築の高速化」に、2 番目の段階の結果と、配列の使い回しを加えない判断を書きました。12 節に、前の版と 3 組ずつ比べた時間、収集と加算の時間、使い回しの効果の上限を加えています。4 スレッドの構築 1 回は、前の版の 0.84 倍から 0.97 倍です。13 節には 62、付録 A にはコードの場所を加えました。14 節の「site の形式の構築の高速化」と「構築のスレッド並列の実機での確認」の 2 項も改めています。2 番目の段階は、利用者の計算機では実行していません。 |
| 改訂 16 | 2026-11-25 | 実機の 5 番目の段階（`3d51128`）の結果の反映です。点検の指摘を直した版と 2 番目の段階を、利用者の計算機で確かめました。9 節の「構築の高速化」に結果を書き、11 節に 18 スレッドと 1 スレッドの 2 組ずつの時間を加えました。18 スレッドの交換行列の段は、前の版（`1219cb9`）の 0.80 倍から 0.91 倍です。13 節には 63（写した CUDA 版の実行ファイルが読むライブラリー）を加えました。14 節では、「構築のスレッド並列の実機での確認」を済みにし、「デバイスに送る積の下限と、残す空き」に新しい比を書いています。3 番目の段階の `modules/exx` の部分（`ae5c54e`）は、9 節で触れるだけにしました。 |
| 改訂 17 | 2026-11-25 | 構築の高速化の 3 番目の段階の反映です。CUDA 版は、`modules/exx` の呼び出し 1 回の中心の原子を、全てデバイスで計算します。呼び出しの間、 $D$ 、 $K$ 、全ての原子の T ブロックをデバイスに置きます。V 近傍の 4 つの項は、中心の原子の全ての ABF について、積 1 回ずつで計算します。コミットは `ae5c54e`（`modules/exx`）、`2e2a6da`（本体）、`09e054a`（試験の道具）です。6.4 節に計算の順序、7.2 節に関数表の 5 番目のメンバー `device` と試験、7.4 節に本体の 2 つの関数を書きました。8 節には、組み込んだ範囲の表を加えています。9 節の「構築の高速化」には、この段階の説明と、基準の 3 つ目を読み替える判断（作業者の判断）を書きました。11 節には、部品が呼び出しを引き受ける条件、ストリームの扱い、標準出力の行、模型の試験を加えています。13 節には 64 と 65、14 節には 2 項、付録 A にはコードの場所、文献には [29] と [30] を加えました。この段階は、利用者の計算機ではまだ実行していません。 |
| 改訂 18 | 2026-11-25 | 実機の 6 番目の段階（`09e054a`）の結果の反映です。呼び出しの全体をデバイスで計算する形を、利用者の計算機で確かめました。9 節の「構築の高速化」に結果を書き、3 番目の段階と、構築の高速化の全体を終えました。次は M7 を再開します。11 節には、18 スレッドと 1 スレッドの対の時間と、受け入れ表のデバイスを使う 4 つの系の時間を加えています。デバイスで計算する実行の交換行列の段は、18 スレッドで前の版の 0.59 倍から 0.63 倍、1 スレッドで 0.14 倍と 0.16 倍です。14 節では、「GPU の速度」「デバイスに送る積の下限と、残す空き」「呼び出しの全体をデバイスで計算する形」「デバイスの既定の動作」の 4 項を改めました。デバイスの既定の動作は変えていません（作業者の判断）。 |
| 改訂 19 | 2026-11-25 | M7（力）の反映です。コミットは `f68c8b2`（結合層の `Exact_Exchange_Gradient` と `Force` の引数）、`c43bab9`（`MD.Type` の拒否）、`4a8b06b`（試験の道具 `tools/exx_force_checks.py`）です。2.2 節では、`MD.Type` の行を応力が要る種別だけに狭め、受け付ける種別を書きました。5.9 節に本体の力の 2 つの部分、7.4 節に結合層の関数と `Force` の引数、8 節の表に力の行を書いています。9 節の M7 には、試験の道具、道具を作り直した理由、結果、`MD.Type` の短い実行を書きました。条件の 2 番目は、HSE06 の差から同じグリッドの GGA-PBE の差を引いた値で判定し、9 成分で最大 2.7e-7 Hartree/bohr でした（限度 1e-5）。GGA-PBE の差そのものが 1e-5 を超えるためです。判定の方法は、利用者の確認を待っています。13 節には 66 と 67、14 節には 3 項、付録 A にはコードの場所、文献には [31] と [32] を加えました。14 節の「力の site 形式」は済みにしています。 |
| 改訂 20 | 2026-11-25 | M7 の条件の 2 番目の判定の方法を、利用者の決定に合わせました。HSE06 の力と全エネルギーの中心差分の差から、同じ構造と同じグリッドの GGA-PBE の差を引いた値を、1e-5 Hartree/bohr 以内とします。9 節の M7 では、表の「成功の確認方法」、判定の方法の段落、結びの段落を改めました。14 節の「既存の力と全エネルギーの食い違い」は、判定の方法を済みにし、「CUDA 版の力」の項を加えています。これで M7 を終え、第 1 版（2.1 節）のマイルストーンが全て終わりました。CUDA 版の力は、まだ実機で実行していません。次は M8 で、最初に着手する項目は利用者の回答を待っています。 |
| 改訂 21 | 2026-11-25 | 利用者の指示で、厳密交換の計算を MPI のプロセスに分けた作業の反映です。コミットは `a1ee5ba`（`modules/exx` の入口と試験）、`a468823`（結合層）、`b265615`（`tools/exx_gpu_checks.py` の説明）です。6.7 節に、準備、構築、力の重み、力の対の 4 つの分け方、範囲の決め方、和の取り方、標準出力を書きました。6.2 節と 6.4 節には分担の段落、7.4 節には 11 個の C の入口、8 節には結合層が組み込んだ範囲を加えています。9 節には、M7 と M8 の間に「MPI 並列での高速化」の項を置きました。5 つの条件、前の版で短くならなかった 3 か所、結果を書いています。Si の基本単位胞の 4 プロセスの構築は、前の版の 13.30 秒から 7.82 秒になりました（構築 10 回）。12 節には前の版と後の版の時間、13 節には 68 から 71、14 節には 5 項、付録 A にはコードの場所を加えました。9 節の M8 には、利用者が PBE0 を選んだことと、`7b77629` と `c9e26e7` を書いています。 |
| 改訂 22 | 2026-11-25 | M8 の PBE0（切断クーロン）の試験の反映です。コミットは `a3da659`（`modules/exx` の試験 `exx-two-center-truncated`）、`b8cbe29`（ $\kappa$ の区間）、`6b53e6b`（3 つの試験の道具）です。2 節には、本体が `PBE0` を受け付けたことを書きました。5.6 節には、 $\kappa$ の台形則が正確になる条件と、PBE0 の区間の数の決め方を加えています。区間を増やす形は、利用者が 2026-11-25 に決めました。8 節には PBE0 で組み込んだ範囲の表を加え、キーワードの表の `scf.exx.v.cutoff` と `scf.exx.coulomb.cutoff` の既定値を改めました。9 節の M8 には、5 つのコミット、M4、M5、M7 と同じ形の試験の結果、停止条件、前の版との比較、モジュールの試験を書いています。13 節では、19 に M8 の結果を書き、72（既定より大きい $R_c$ ）を加えました。14 節では、「打ち切りクーロンと $\kappa$ の刻み」を済みにし、2 項を加えています。M6 と同じ形の試験（CUDA 版）は、まだ実行していません。 |
| 改訂 23 | 2026-11-25 | M8 の PBE0 の CUDA 版の確認の反映です。コミットは `9e01d40`（M6 の試験の道具）です。利用者の計算機で、PBE0 の 57 件と、同じ版で実行し直した HSE06 の 57 件が、M6 の条件を満たしました。9 節の M8 には、6 つ目のコミット、M6 の条件の行、M6 の形の段落を加えています。9 節の M6 には、HSE06 の表を実行し直した結果を書きました。MPI 並列での高速化の節には、原子の途中で切れた範囲の結果を書いています。11 節には、4 プロセスの Si と PBE0 の系の時間とメモリーを加えました。14 節では、「CUDA 版の力」「CUDA 版での ABF の範囲」「PBE0 の CUDA 版」を済みにしました。「厳密交換の力の計算の時間」には、実機の時間を加えています。 |
| 改訂 24 | 2026-11-25 | 既定の値より大きい $R_c$ を拒否する形の反映です。利用者が 2026-11-25 に、拒否する形を選びました。コミットは `3d810d2`（本体）と `6212ade`（試験の道具）です。5.6 節と、8 節の PBE0 の表とキーワードの表に、拒否を書きました。9 節の M8 には、2 つのコミットと、拒否の段落を加えています。13 節の 72 は、本体が拒否する形に改めました。14 節の「PBE0 の $R_c$ の上限」は、済みにしています。冒頭の表には、利用者が選んだ M8 の次の項目と、その前に作る形を書きました。 |
| 改訂 25 | 2026-11-25 | 利用者の 2 つの指示（力のデバイス化と、ほかの可能な高速化の全て）による作業の反映です。コミットは 14 個で、`d6f7db1` から `92f78c7` までです（9 節の表）。6.4 節に半分の構築、6.5 節に $K(\mathbf k)$ の保持、6.7 節に時間の見積もりによる ABF の分け方を書きました。7.4 節、7.5 節、8 節には、新しい入り口と結合層の変更があります。9 節には、新しい節「力のデバイス化とほかの高速化」を加えています。11 節には重みのデバイスと XC の GPU 経路を、12 節には分け方の係数の実測を書きました。13 節に 73 から 75 を加え、14 節の 7 つの行を改めて 1 つの行を加えています。付録 A には、コードの場所の表を加えました。 |
| 改訂 26 | 2026-11-26 | 2 つ目の指示による高速化の続きです。コミットは `2198f55`（力の重みの半分の形）と `7a79b99`（準備と力の対のスレッド）です。6.4 節と 6.7 節に、半分の重みと対のスレッドを書きました。7.4 節には 4 つの C の入口を、11 節には重みのデバイスの 3 回目の転送を加えています。9 節には、2 つのコミットの時間、`f0421a6` の実機の確認、採らなかった形、利用者の判断が要る形を書きました。14 節の 3 つの行を改め、付録 A に表を加えています。改訂 25 の記録のコミットの数は、15 個から 14 個に直しました。 |
| 改訂 27 | 2026-11-26 | 2 つ目の指示による高速化の続きです。コミットは `1626dc4`（V 近傍のまとめ）です。6.4 節に、正則な k メッシュの周期で V 近傍をまとめる形を書きました。5.3 節、6.7 節、7.4 節には、その前提、力の対、C の入り口と結合層があります。9 節には、時間と 8 件の比較と、次の作業を書いています。12 節、13 節の 30、14 節の 1 つの行を改め、付録 A に表を加えました。 |
| 改訂 28 | 2026-11-26 | 2 つ目の指示による高速化の続きです。コミットは `51cd508`（列のまとめ）です。6.4 節に、V 近傍をまとめた計算で構築の列も周期でまとめる形を書きました。5.3 節、6.7 節、7.4 節には、符号の扱い、力の重み、C の入り口と結合層があります。9 節には、時間、8 件と `REGULAR` の 2 件の比較、`7fa9431` の実機の確認を書き、次の作業を改めています。12 節、13 節の 30、14 節の 1 つの行を改め、付録 A に表を加えました。 |
| 改訂 29 | 2026-11-26 | 2 つ目の指示による高速化の続きです。コミットは `079d22c`（呼び出しを引き受ける条件）です。7.2 節、8 節、11 節に、積の乗算の和で決める条件を書きました。9 節には、`51cd508` の実機の確認の途中の結果と、コンテナで測った値があります。14 節の 1 つの行を改め、付録 A に表を加えました。 |
| 改訂 30 | 2026-11-26 | M8 のバンド分散と DOS です。コミットは `9f012f4`（本体）と `c52e41e`（試験の道具）です。2.2 節の表、6.5 節、7.4 節、9 節の M8、13 節の 1、76、77、14 節の 2 行、付録 A を改めました。9 節の高速化の節には、`51cd508` と `079d22c` の実機の確認の結果を書きました。 |
| 改訂 31 | 2026-11-26 | 結果を SCF の判定の範囲で変える高速化です。コミットは `6c9a592`（本体と SCF の道具）です。6.6 節に、始め直しの位置、外挿、緩い判定を書きました。8 節には組み込みの表と 2 つのキーワードを、9 節には時間、比較、省く形の測定を加えています。13 節の 78 から 81、14 節の 3 行と 1 行、付録 A を改めました。 |

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
20. RIKEN-RCCS, "GEMMul8", <https://github.com/RIKEN-RCCS/GEMMul8>（コミット `f8fdf0b`、2026-10-14。リポジトリーのサブモジュール `source/third_party/GEMMul8` が指すコミットです。2026-11-24 に、README の「Accuracy Guidelines」の項と、`src/oz2/undo_scaling/undo_scaling.hpp`、`src/oz2/undo_scaling/scalar.hpp` を読みました）。
21. Open MPI 5.0.10 documentation, "Shared Memory", <https://docs.open-mpi.org/en/v5.0.10/_sources/tuning-apps/networking/shared-memory.rst.txt>（2026-11-24 に確認。裏付けファイルの既定の場所と、`btl_sm_backing_directory` の項を読みました。異常終了の後にファイルが残る場合についての記述は、このページにありません）。
22. NVIDIA, "CUDA Environment Variables" (CUDA Programming Guide 13.4.1), <https://docs.nvidia.com/cuda/archive/13.4.1/cuda-programming-guide/05-appendices/environment-variables.html>（2026-11-24 に確認。`CUDA_CACHE_DISABLE`、`CUDA_CACHE_PATH`、`CUDA_CACHE_MAXSIZE` の項を読みました）。
23. Apple Developer Forums, "Dynamic lookup of weak references", <https://developer.apple.com/forums/thread/17630>（2026-11-24 に確認。Apple の開発ツールの技術者による 2015 年の回答を読みました。`weak_import` のシンボルは実行時にはなくてもよいが、ビルドのときには見つかる必要がある、という内容です）。
24. Apple, "Frameworks and Weak Linking" (Framework Programming Guide), <https://developer.apple.com/library/mac/DOCUMENTATION/MacOSX/Conceptual/BPFrameworks/Concepts/WeakLinking.html>（2026-11-24 に確認。弱くリンクしたシンボルがフレームワークにないとき、番地が NULL になる、という記述を読みました。リンクのときの条件は、このページにありません）。
25. Apple Developer Forums, "Understanding Mach-O Symbols", <https://developer.apple.com/forums/thread/775650>（2026-11-24 に確認。2025 年の投稿です。弱い参照のシンボルが実行時に見つからないとき、動的リンカーは読み込みを続けて番地を NULL にする、という記述を読みました）。
26. Launchpad, Ubuntu の linux パッケージのバグ 2056762, "ThreadSanitizer: unexpected memory mapping with 6.5.0-25.25~22.04.1", <https://bugs.launchpad.net/ubuntu/+source/linux/+bug/2056762>（2026-11-24 に確認。`vm.mmap_rnd_bits` が 28 から 32 になった後の報告と、値を 28 に戻すと動くという記述を読みました）。
27. fsverity-utils, コミット `da05cca`, "ci.yml: work around ASAN binaries crashing on ubuntu-latest", <https://kernel.googlesource.com/pub/scm/fs/fsverity/fsverity-utils/+/da05cca371d246999b358138719e2f5dfd61abc4%5E%21>（2026-11-24 に確認。GitHub Actions の手順に `sudo sysctl vm.mmap_rnd_bits=28` を加える変更です。手順の名前に、actions/runner-images の issue 9491 を挙げています）。
28. actions/runner-images, issue 9491, "Crashing Clang with Asan+Ubsan builds in 20240310.1.0 Ubuntu 22.04", <https://github.com/actions/runner-images/issues/9491>（2026-11-24 に確認。イメージ 20240304.1.0 の `vm.mmap_rnd_bits = 28` と 20240310.1.0 の `vm.mmap_rnd_bits = 32`、報告者の対処の `sudo sysctl vm.mmap_rnd_bits=28`、管理者の「The fix has been deployed」で始まる書き込み、新しい LLVM で直ったという書き込み（LLVM のコミット `fb77ca05`）を読みました。読んだ本文には、書き込みの日付がありません）。
29. NVIDIA, "API synchronization behavior" (CUDA Runtime API 12.9.0), <https://docs.nvidia.com/cuda/archive/12.9.0/cuda-runtime-api/api-sync-behavior.html>（2026-11-25 に確認。Memcpy の「Synchronous」の項を読みました。ページ可能なホストのメモリーからデバイスへの写しでは、関数は DMA 用の領域への写しが終わると戻り、DMA は終わっていないことがある、という記述です。利用者の計算機の CUDA は 13.4 です。13.4 の版のこのページは、確かめていません）。
30. NVIDIA, "HPC Compilers Reference Guide" (NVIDIA HPC SDK 26.11), <https://docs.nvidia.com/hpc-sdk/compilers/hpc-compilers-ref-guide/>（2026-11-25 に確認。2.2.27 節の `-gpu` の副オプション `[no]fma` の項を読みました。「default at -O1」と、`-M[no]fma` の別名であるという記述です。利用者の計算機の NVHPC は 26.9 です。26.9 の版のこの項は、確かめていません）。
31. OpenMX 3.9 manual, "Automatic force tester", <https://openmx-square.org/openmx_man3.9/node215.html>（2026-11-25 に確認。`-forcetest` の使い方と、出力の例 `force_example/C2_GGA.dat` の解析的な力、数値的な力、差の 3 行を読みました。刻みは `ds=0.0003` と表示され、単位は書かれていません。本体のコードでは bohr です（`Force_test.c` の `step`）。グリッドの扱いと、差の大きさの目安は書かれていません。OpenMX 3.8 のマニュアルの同じ節も、同じ例です）。
32. The Siesta Group, "The real-space grid", Siesta Documentation, <https://docs.siesta-project.org/projects/siesta/en/stable/tutorials/basic/grid-convergence/index.html>（2026-11-25 に確認。「The egg-box effect」の節を読みました。原子が固定されたグリッドに対して動くだけでエネルギーが変わる、計算上の人工的な効果として説明しています。カットオフが低いと現れやすいこと、構造の全体をグリッドに対してずらして調べることが書かれています。効果の大きさの値と、力への影響の大きさは書かれていません。カットオフを上げると単調に減るとは書かれていません。版の番号と日付は表示されていません）。
