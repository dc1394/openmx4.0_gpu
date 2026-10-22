# 大域状態の利用監査（関数単位の読み書き）

[C++ / Kokkos 移行](CPP_CORE_MIGRATION.md) はサブシステムごとに所有権を `core/` へ移す
ボトムアップの作業です。この文書は、それを補う上からの視点を扱います。
「どの大域オブジェクトを、どの関数が、読む・書く・ポインタとして渡すか」を
プログラム全体で測り、書き手に基づいて所有権の行き先を分類し、
増加を止めるゲートを提供します。宣言数の棚卸し（`tools/audit_common_globals.py`）が
共通ヘッダーの供給側を数えるのに対し、ここでは消費側を数えます。

生成物は [利用一覧](GLOBAL_STATE_INVENTORY.md)（Markdown、`tools/audit_global_usage.py` が生成）と
`docs/global-usage-baseline.json`（ゲート用のベースライン）です。

## 手法

`tools/audit_global_usage.py` は、NVIDIA ビルドの Makefile が列挙する C の翻訳単位を
clang で `-O0 -emit-llvm` にコンパイルし、LLVM IR を関数ごとに走査します。
`-O0` ではすべてのアクセスが明示的な `load` / `store` になり、`static` は内部リンケージ、
関数内 `static` は `関数名.変数名` という名前で現れるため、ソースの字面に頼らずに数えられます。

| 記録 | 意味 |
| --- | --- |
| read | オブジェクト自身の値の `load` |
| write | オブジェクト自身への `store` |
| write-through | オブジェクトから読んだポインタ（配列の先頭など）を経由した `store`。配列の中身への書き込み |
| address | オブジェクトの番地そのものを関数へ渡す（`MPI_Bcast(&x)` など）。`MPI_Bcast` だけは「複製」として別に数える |
| escape | オブジェクトから読んだポインタを関数へ渡す。呼び先が中身を書く可能性がある |

ポインタの由来は関数内のフロー非依存解析で追います（SSA 値、`-O0` の `alloca`、`getelementptr` の
ベースのみ。添字や整数値は伝播しません）。呼び先の挙動は仮定しません。
`MPI_Comm` のようなハンドルも IR 上はポインタなので、`mpi_comm_level1` の escape は
実際には読み取りです。分類では escape を書き込みに数えません。

対象外: `.cu` / `.cpp` の翻訳単位、Fortran、`source/backends/hip/` の HIP 版ソース。
ただし「どの関数からも参照されない」オブジェクトの判定では、これらのソースを
文字列で検索し、参照があるものを削除候補から外します。

```sh
# 現状の一覧と JSON（Makefile の include 設定と NVHPC の omp.h/openacc.h を自動で使う）
python3 tools/audit_global_usage.py --output build/globals-usage.json \
    --markdown docs/GLOBAL_STATE_INVENTORY.md --baseline-output docs/global-usage-baseline.json

# ゲート: ベースラインに無い (関数, オブジェクト) の組、static、環境変数、
# 共通ヘッダー外の大域定義、getenv の直接呼び出し、レジストリに無い環境変数名が
# 増えていれば失敗する
python3 tools/audit_global_usage.py --output build/globals-usage.json \
    --baseline docs/global-usage-baseline.json --check-no-new
```

ゲートは「減る方向」だけを許します。所有権を移した段階でベースラインを更新してください。
CI の GitHub Actions には CUDA・MPI・NVHPC のヘッダーが無いので、ゲートはこの計算機で手動で実行します。

## 現状（2026-10-22、`refactor/cpp-kokkos-core` 84ab948c ＋ 環境変数レジストリ第 2 段階）

| 項目 | 値 |
| --- | ---: |
| 解析した翻訳単位 / 関数 | 290 / 3,007 |
| 共通ヘッダーのオブジェクト（`openmx_globals.c` の定義） | 1,051（本変更前 1,112） |
| うち、どこかの関数が参照するもの | 1,045 |
| うち、どこかの関数が書くもの（直接または中身） | 785 |
| (関数, オブジェクト) の組 | 12,912 |
| 共通ヘッダー外で外部リンケージを持つ定義 | 358（`openmx.c` 132、`DFT.c` 79、`Cluster_DFT_ON2.c` 28 …） |
| ファイルスコープ `static` | 451（`TRAN_Main_Analysis*.c` 173、`Band_DFT_*.c` 60 …、`openmx_env.c` 2） |
| 関数内 `static` | 386（第 2 段階前 416; `MD_pac.c` 58、`zero_cfrac.c` 54 …） |
| 環境変数名（全翻訳単位、id 経由の読み取りを含む） | 251 |
| `getenv` を直接呼ぶ関数（`openmx_env.c` 以外） | 0 |
| レジストリに無い環境変数名 | 0 |

最も多くの関数が触るのは `mpi_comm_level1`（604 関数、154 翻訳単位）、`WhatSpecies`（514）、
`natn`（372）、`FNAN`（359）、`M2G`（357）、`Matomnum`（353）、`atomnum`（332）、
`Spe_Total_CNO`（310）、`SpinP_switch`（274）、`List_YOUSO`（217）です。
関数側では `Input_std`（426 オブジェクト）、`Free_Arrays` の `array0`（414）、
`Allocate_Arrays_with_context`（219）、`DFT`（140）、`truncation`（108）が突出しています。

## 書き手による分類

| 分類 | 個数 | 意味と、所有権の行き先 |
| --- | ---: | --- |
| input-configuration | 298 | 入力解析（`Input_std.c` / `Inputtools.c`）だけが書く。読み取り専用の設定オブジェクトにできる |
| allocated-array/contents-fixed | 278 | 確保側（`Allocate_Arrays.c`、`truncation.c` など）だけが書き、中身も確保側と入力解析しか書かない。構造表（原子・近接・分配）の不変オブジェクト |
| allocated-array/contents-mutable | 95 | 確保は確保側だが、中身を他のモジュールが書く（`Gxyz`、`FNAN`、`time_per_atom` など）。計算状態オブジェクト |
| shared-mutable | 225 | 複数の翻訳単位が書く。個別に判断が要る。`filename` / `filepath`（60 翻訳単位から `sprintf` される作業バッファ）、`ChemP`（25 関数が書く）、`List_YOUSO`、`Matomnum` など |
| single-writer:*.c | 1 翻訳単位だけが書く。その翻訳単位の所有物にできる（`DFT.c` 28、`DFTD3vdW_init.c` 16 …） |
| read-only | 15 | 解析対象の C からは書かれない。C++/CUDA 側か定数 |
| unused | 6 | C の関数から参照されない。`T` / `W`（1 文字名で検索できず保留）、HIP 版 `DFT.c` だけが使う `UH2` / `Uxc` / `command`、`exx` 用の `g_exx` |

上位 2 分類（576 個、55%）は書き手が限定されており、消費側を「const なビューを受け取る」形に
変えるだけで共通ヘッダーから外せます。移行作業の順序としては、この 2 分類を
設定オブジェクトと構造表オブジェクトへ一括で移すのが、関数ごとの改修より少ない手数で
最大の削減になります。

## 本変更で除去した 61 オブジェクト

どの関数からも参照されず、CUDA・C++・Fortran・HIP 版ソースにも現れない共通ヘッダーの
オブジェクト 61 個の宣言（`openmx_common.h`）と定義（`openmx_globals.c`）を除去しました。
対応する説明コメントも除去しています。`BestNormRD`、`Calc_Total_Charge`、`FCR`、
`First_Moment_EC`、`Second_Moment_EC`、`Gdiis_Mixing`、`OLP_p`、`Pin`、`Ptot`、`RandomF`、
`Vol`〜`Vol3`、`Uele0`、`Uele1`、`Uvdw`、`Ucc`、`init_tv`、`gtv_FE`、`rgtv_FE`、`Ngrid*_FE`、
`Finite_Elements_Ecut`、`rcut_FEB`、`orderN_Kgrid`、`openmp_threads_eq_procs`、
`Hub_U_Enhance_OrbPol`、`VNA_reduce_ratio`、`W_OrthoNorm`、`bias_weight`、
`ref_diff_charge_filename`、`Wannier90_CWF_fileout`、`maxn_dftD`、NAO 用の 8 個
（`NAO_coefficient`、`Snd_MAN_NAO`、`Snd_GAN_NAO`、`Rcv_GAN_NAO`、`F_M2G_NAO`、`S_M2G_NAO`、
`MatomnumF_NAO`、`MatomnumS_NAO`）、格子分配用の 7 個（`My_NumGrid1`、`FNAN2_Grid`、
`Max_Num_Rcv_FNAN2_Grid`、`Max_Num_Snd_FNAN2_Grid`、`My_NGrid1_Poisson`、`My_NGrid2_Poisson`）、
ScaLAPACK 文脈の 12 個（`bhandle2_2`、`ictxt2_2`、`bhandle2_CWF1`〜`CWF5`、`ictxt2_CWF1`〜`CWF5`）です。
数値計算には影響しません。NVIDIA ビルドのリンクと標準スイートで確認しました。

## 環境変数の設定レジストリ（`source/openmx_env_registry.h`、`openmx_env.c`）

`getenv` を呼んでいた 215 か所（46 翻訳単位と 2 つのヘッダー、CUDA/C++ の単位を含む）は、
すべてレジストリ経由になりました。第 1 段階（84ab948c）が `getenv` を `openmx_env_get()` に
置き換え、第 2 段階が読み手ごとの解釈と関数内 `static` キャッシュを共通化しました。

- **表は 1 か所**: `openmx_env_registry.h` の X マクロ行 `X(id, "名前", 種類, 範囲, 読み手, 説明)`
  （名前順、260 名: ソースが読む 251 名と、ビルド・ツールだけが読む 9 名）。`openmx_env.h` が id の列から `OPENMX_ENV_<id>`（`OPENMX_BAND_PROFILE` →
  `OPENMX_ENV_BAND_PROFILE`、`OPENMX_` 以外の名前は `OPENMX_ENV_EXT_<名前>`）を作り、
  `openmx_env.c` が表を作ります。`tools/update_env_registry.py` がソースの読み手から表を
  書き直し（名前の追加、使われた読み取り関数から種類を導出、読み手の列を再生成、説明は保持）、
  `tests/run_env_registry_cpu.py` が `--check` で表とソースのずれを検出します。
- **写しと O(1) の型付き読み取り**: `openmx_env_init()`（`MPI_Init` 直後）が登録名の値を写し取り、
  種類に関係なく flag / int / uint / double として前もって解釈します。読み手は
  `openmx_env_flag_at(OPENMX_ENV_X, 既定値)`、`_long_at`、`_ulong_at`、`_double_at`、
  `_mib_bytes_at`（MiB → バイト、飽和）、`_at`（文字列）を呼ぶだけで、関数内 `static` の
  キャッシュは要りません（第 2 段階で約 30 個を除去）。解釈は 1 つです: 数値は非零で真、
  on/yes/true/y と off/no/false/n（大文字小文字不問）、`auto`・空・未設定は既定値。
  整数は文字列全体が整数のときだけ有効、前後の空白は無視します。
- **rank 0 からの配布**: 範囲 `global` の名前は、`openmx.c` が `openmx_env_pack()` /
  `openmx_env_unpack()` と `MPI_Bcast` で rank 0 の値を全ランクに写します。`mpirun -x` を
  忘れたときや、ノードごとのジョブスクリプトで一部のランクにしか渡らなかったときでも
  ランク間で設定が食い違いません。これにより `Force_collective_env_flag` などの
  「rank 0 が読んで Bcast」する読み手と `gs2_forced_mode_collective` の不一致検出は不要になり
  除去しました。範囲 `local` はランチャーが与えるランク番号・ホスト名・MPS のパスなどで、
  配布しません。`level.of.stdout 3` では各ランクが受け取った設定数を出力します。
- **診断**: 未登録の `OPENMX_*` 名（綴り違い）と、種類に合わない値（`OPENMX_BAND_PROFILE=abc`）を
  rank 0 が起動時に 1 行ずつ警告します。`level.of.stdout 2` 以上では有効な設定を一覧します。
- **解釈の変更点**: `atoi` で「1 以外は偽」や「先頭文字が 1」だった読み手も on/yes/true と
  2 以上の数値を真と解釈するようになりました。逆に、プロファイル出力など「設定されていれば
  有効」だった読み手は `=0` / `=off` で無効にできます。不正な文字列は（0 扱いではなく）
  既定値に戻り、警告されます。空文字列は未設定と同じです。
- **残る `static`**: 環境を読む関数に残る 36 個の `static` は、VRAM 探索の判定（`verdict`）、
  一度だけの通知（`announced`）、作業領域の大きさなどで、環境変数のキャッシュではありません。
- **HIP 版の複製**（`source/backends/hip/src`、固定 6 ファイルと統合 19 ファイル）は `getenv` を
  `openmx_env_get` に機械的に置き換えただけで、型付き読み取りへの移行はしていません。
  この計算機では HIP をビルドできないため未検証です。
- 検査: `tests/run_env_registry_cpu.py`（1,738 チェック＋表の同期確認、CTest
  `environment-registry`、ホスト CI）、`-runtest` 14/14、sidia333 の 2 SCF ステップで rank 0 だけに
  フラグを与えた MPMD 起動（17 ランクが 5 設定を受け取り、全ランクでプロファイル出力）。

## 次の作業単位（この監査の側）

1. 環境変数レジストリの第 3 段階: 実装文書 §10.2 の表をレジストリから生成する（既定値は
   呼び出し側の型付き定数のままなので、表には説明と種類を載せる）。HIP 版の複製の読み手を
   型付き読み取りへ移す（HIP をビルドできる計算機で）。
2. `filename` / `filepath` / `command` の作業バッファ: 各呼び出し側の局所バッファへ置き換える。
3. 共通ヘッダー外の 358 定義: `openmx.c` が定義する輸送（TRAN）の 132 変数を
   `tran_variables.h` ごと輸送の状態オブジェクトへ移す。`DFT.c` の 79 変数は
   ソルバー作業領域の所有化（移行第 20・23 段階）と合わせる。
4. 関数内 `static` の多い `MD_pac.c`（58）と `zero_cfrac.c`（54）を、状態を引数で渡す形にする。
5. この一覧とベースラインを、所有権を移した段階ごとに更新する。
