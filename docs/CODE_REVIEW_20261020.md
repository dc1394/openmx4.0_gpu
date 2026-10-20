# 2026-10-20 コードレビューへの対応

レビュー資料の対象は `04e0b45`。修正は、その後の g 軌道導関数修正を含む
`0e3de09` を起点に行った。既存の NVIDIA・AMD 最適化を維持しつつ、
実際の不具合と、Intel・Apple 対応に必要な境界を整備した。

## 不具合の修正

| 指摘 | 修正と検証 |
|---|---|
| B1・B2 | 無制限の待機マクロを共通のエラー処理へ置換。runtime・BLAS・solver のエラー番号を別々に解釈し、メモリ不足だけ最大8回再試行する。待機は合計127 msの `nanosleep`。致命的エラー・既存の未処理エラーは再試行せず、MPI使用中なら `MPI_Abort` で終了する。成功・一時的不足・恒久的不足・致命的エラー・MPI終了前後を注入して検証。 |
| B3・B7 | HIPの手書きC用型・列挙値・関数宣言を廃止し、ROCmの正式なHIP・hipBLAS・hipSOLVERヘッダーを使う。`hipblasZgeam` もSDKの宣言を参照する。`cudaDataType` 互換名は `hipDataType` と正式な定数に対応付けた。ROCm 7.2の実ヘッダーでC/C++双方をコンパイル。 |
| B4 | ワーカーの結果公開・後処理失敗を共通のMPI終了処理へ接続した。公開開始後のCPU再実行を禁止する既存の契約を維持。 |
| B5 | Force4B・HNLのホスト回復パケットで、9か所の確保結果を使用前に確認する。各確保を失敗させ、書き込みやタスク実行より前に停止することを検証。 |
| B6 | CUDAのForce3・Force4B・HNLの既定値も `OPENMX_FORCE_GPU` を継承する。各段階の明示的な設定による上書きは可能。 |
| B8 | ランタイム実装では弱宣言を無効にする。CUDA・HIPの実コンパイラで生成したオブジェクトのsolve・dispatchシンボルが `T` であることを検査。 |
| B9 | GPU占有ロックにプロセス内・プロセス間とも待機上限を設けた。ディレクトリも指定可能。競合、期限切れ、失敗後の再取得、UUID順序、解放をテスト。 |
| B10 | メモリ見積もりのコールバックへ対象デバイスを渡す。別GPU上の単一DMワーカーにも固有ベクトル全体のコピー容量を加える。実際のDMコールバックと配置計画を、P2Pを使わないテストで確認。 |
| B11 | ワーカー・GEMM・DMの入口で呼び出し元の未処理エラーを `PeekAtLastError` で検出し、開始時や辞退時に無条件で消さない。自分の確保失敗や既知の問い合わせ結果だけを回復処理で消費する。DMのCPU回復を含む試験でも、既存エラー・致命的な準備エラーでは出力を変更せず、自分のメモリ不足だけを回復することを確認。 |
| B12 | HIP Bandのk点スコープを、所有権を持つCのガードで管理する。k点の切り替え前に前スコープを閉じ、最後に呼び出し元の状態へ戻す。11回切り替えてもステージングの寿命が増えないことを検証。 |
| B13 | ホスト容量回復APIのコメントを、明示的なCPU LAPACK/BLAS回復が存在する実装に合わせた。 |
| B14 | Forceと分割統治法の到達不能な旧オフロード分岐を除去。CUDA側にも残っていた `Eigen_lapack_openacc` 参照を除去した。両コンパイラの `-O0` オブジェクトで未定義参照がないことを検証。 |

`OPENMX_GPU=0` は両バックエンドの多GPUプールを無効にする。
暗黙の関数宣言もエラーとして扱う。NVHPCでは
`--diag_error=implicit_func_decl`、ROCm clangでは
`-Werror=implicit-function-declaration` を使用する。
NVHPCではGCC形式の同名警告オプションだけではエラー化できないことを確認した。

## 将来のバックエンドを追加する境界

`core/include/openmx_accelerator.h` はSDKに依存しないCの能力記述である。
バックエンド、FP64の可否と性能区分、メモリモデル、既知の演算性能比を分けて持つ。
共有メモリであることからFP64対応を推測しない。能力が不明な場合もFP64実行を許可しない。

既存のCUDAワーカーの重ね合わせ方針は、CUDAの演算性能比に基づく。
HIPでFP64性能を調べられるようになっても、その事実だけで同じ実行方針を有効にしない。
Radeon・Instinctの既存方針を維持し、HIPの既定の直列化判定には追加のデバイス問い合わせを行わない。
AMDのアーキテクチャ分類は共通のヘルパーへ集約した。

ネイティブ層の `OPENMX_GPU_BACKEND` にHOST・CUDA・HIP・SYCL・METALの識別子を設けた。
CUDA/HIP専用アダプターを他の識別子で読み込むとコンパイルエラーになる。
従来の直接Makefile利用との互換性のため、指定なし・非HIPの既定値はCUDAのままである。
数値コアはこのネイティブ層を参照しない。

### Intel

C++コアは `OPENMX_CORE_BACKEND=SYCL` で独立にビルドできる。
KokkosのSYCLバックエンドを使用し、選択されたデバイスの `sycl::aspect::fp64` を確認する。
FP64が使えなければ動径変換をCPUで計算し、FP32への暗黙変換は行わない。
Kokkosのバックエンド設定は[公式設定ガイド](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html)も参照。

```sh
cmake -S core -B build/core-sycl \
  -DCMAKE_CXX_COMPILER=icpx -DOPENMX_CORE_BACKEND=SYCL \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/core-sycl --parallel
```

IntelLLVMには `-fp-model=precise -ffp-contract=off` を指定する。
これはインラインの数値処理や呼び出し側にも伝える。ライブラリだけに指定すると、
呼び出し側で `b+t-a` が再結合され、座標差の検査が失敗することを実際に確認した。

SYCL対応は現在はコア単独の機能である。本体の密行列ソルバー、BLAS、
データ転送・指示文オフロードのIntel用実装はまだない。
CUDA/HIP本体とSYCLコアを混ぜた構成はCMakeで拒否する。

### Apple

SDKから独立したコアをAppleClangのSERIAL実行で検査するmacOS CIを追加した。
METALは将来のアダプター識別子であり、実装済みのGPUバックエンドではない。
[Metal Shading Language仕様のスカラー型](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
には `double` がないため、現在のFP64カーネルをそのままMetalへ送る方針にはしない。
GPU対応時には、対象演算ごとに精度保証を検証した実装と、その能力を明示する必要がある。

## 本体・コアの構造

- C ABIの `openmx_core*.h` と複素数・周期セルヘッダーを `core/include/` へ移した。
  `source/` には互換インクルードを残し、コアのPUBLICインクルードから `../source` を外した。
  公開ヘッダー群はSDKなしのC11でもコンパイルできる。
- `openmx_common.h` の外部変数を `extern` 宣言にし、
  重複宣言を除く1,154変数と `g_exx` の計1,155変数の定義を `openmx_globals.c` に集約した。
  名前・型をClangのASTで照合した。明示的なゼロ初期化によりNVHPCでも強い定義にする。
  これは定義場所の一本化であり、大域状態の使用自体をなくしたわけではない。
  他ヘッダーの状態や関数内staticを含む、計算コンテキストへの移行は引き続き必要。
- CPUとKokkosの動径変換は同じ値専用Besselカーネルを使用する。
  導関数用の既存APIは維持する。
- `Context::last_fallback()` で外部Kokkos所有者、デバイス変更、FP64不足によるCPU実行を区別できる。
  `OPENMX_CORE_VERBOSE=1` では理由をコンテキストごと・理由ごとに一度表示する。
- CUDA/HIPとしてコンパイルするコアにも警告を有効にした。

## 検証

実行環境はRTX 5080、NVHPC 26.9、CUDA 13.4、GCC 14、IntelLLVM 2026.0。
ROCm 7.2のSDKとclangは作業用ビルドディレクトリに展開した。
AMD・Intel・AppleのGPU実機はこの環境にない。

| 検査 | 結果 |
|---|---|
| NVIDIA本体のビルドとCTest | ビルド成功、150件成功 |
| Kokkos CUDAコア | 実機で124件成功 |
| IntelLLVM SERIALコア | 124件成功 |
| IntelLLVM SYCLコア | コアと全テストプログラムをコンパイル。GPU初期化不要の7件成功 |
| HIP Cソース | 本体ビルド対象287本をROCm clangと正式SDKヘッダーで構文・型検査 |
| CUDA/HIPの重要11ソース | `tools/check_gpu_compilation.py` により `-O0` コンパイル、GPUカーネル生成、未定義参照とシンボル強度を検査 |
| HIPワーカー・DM・GEMM・ランタイム | ホスト上の模擬デバイスによる回帰テスト成功 |
| 本体の `-runtest` 相当の標準Sスイート | 最終バイナリをCPU/GPU各4 MPI・1 OpenMPで実行し、各14ケース完走。修正前・同梱参照値・CPU/GPU間の全Force成分とエネルギーを照合 |

Sスイートの比較許容値はエネルギー `1e-7 Ha`、Force各成分 `1e-7 Ha/bohr`。
原子座標と積分グリッドも一致することを確認した。

| 比較（14ケースの最大絶対差） | エネルギー（Ha） | Force成分（Ha/bohr） |
|---|---:|---:|
| CPU：修正前と最終版 | 0（出力桁の範囲） | 0（出力桁の範囲） |
| GPU：修正前と最終版 | 3.0e-12 | 1.0e-12 |
| CPU：同梱参照値と最終版 | 3.1e-11 | 7.8e-11 |
| GPU：同梱参照値と最終版 | 6.1e-11 | 7.8e-11 |
| 最終版のCPUとGPU | 4.8e-11 | 1.2e-11 |

スイートの時間は他の検証と同時実行しているため、性能比較には使用しない。
今回の数値検証は標準Sスイートであり、`-runtestL` / `-runtestL3` の再実行は含まない。

`work/sidia333_nc_cluster/sidia333_nc_cluster.dat` も、RTX 5080、18 MPI、
各1 OpenMP、MPS有効で修正前・最終版を比較した。他のビルドやテストを止めて測定し、
最終版の実行中にMPSサーバーへの18クライアント接続を確認した。
既存のMPSサーバーは再起動していない。

| 指標 | 修正前 `0e3de09` | 最終版 |
|---|---:|---:|
| MPI起動を含む壁時計時間（秒） | 110.925 | 111.674 |
| 本体のTotal Computational Time・最大ランク（秒） | 107.261 | 108.033 |
| Force・最大ランク（秒） | 15.841 | 16.020 |
| SCF反復数 | 25 | 25 |

エネルギー差は出力桁の範囲でゼロ、全Force成分の最大差は `1.0e-12 Ha/bohr`。
入力・座標・積分グリッドは同一である。壁時計時間の差は約+0.7%で、
修正前・最終版とも1回の計測のため、統計的な速度差は判定しない。
この計測はGeForceでの回帰確認であり、A100/H100や他社GPU、
IntelコンパイラによるOpenMX本体とのCPU性能比較は含まない。

最初のCTest実行では、環境の共有ライブラリ検索パスを継承した2件で
ASan内部の代替シグナルスタック後処理が失敗した。
`LD_LIBRARY_PATH`・`LD_PRELOAD` を除いた環境で該当テストが通り、
最終版の150件も同じ環境で全件成功した。
テスト用スタックは `ulimit -s 8192` に固定した。無制限スタックを継承した
単独起動でもASanのshadow領域競合が発生し、この設定で解消した。

`.github/workflows/gpu-compile.yml` はGPUなしのROCm/NVHPCコンパイル検査を行う。
macOSを含む新しいGitHub Actionsジョブ自体の実行結果は、まだ確認していない。
HIP本体全体のリンクとGPU実行、Intel GPU上のSYCL実行、Apple上の実行は別途確認が必要。

作業ログ・入力・ハッシュ・比較結果は `build/code-review-20261020/` に保存する。
最終版の記録は `validated-state.json`、`suite-validated-comparisons.json`、
`sidia-validated-gpu/result.json`、`sidia-mps-state.json` にある。
OpenACC/OpenMPの統合、数値カーネルのKokkos移行、SDK型を隠したメモリ・BLAS・solver API、
環境設定の一元化、残るVLA・大域状態・C ABI実装ファイルの整理は、次の移行段階で扱う。
