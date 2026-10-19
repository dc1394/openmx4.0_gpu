# リファクタリングの性能回帰検証

所有権・モジュール化・Kokkos 化は、計算精度と性能を合わせて検証します。
対象の `work/sidia333_nc_cluster/sidia333_nc_cluster.dat` は 216 原子の
非共線 Cluster 計算で、SCF 上限 25 回、MD 上限 1 回です。原子数、格子、
反復数、収束条件を変えて性能改善として扱いません。

この GeForce RTX 5080 環境での目標は、ユーザーの v3.0/v3.1 頃の GPU＋MPS 実行時間
約 105 秒です。比較条件は 18 MPI × 1 OpenMP とし、アプリケーションの
`Total Computational Time` と MPI 起動を含む経過時間を両方保存します。
CPU は Intel コンパイラで構築した `~/openmx4.0/work/openmx` を比較対象にします。

## 2026-10-19 の確認結果

GeForce RTX 5080（16 GB、sm_120）、Core i9-10980XE（18 物理コア）、
NVHPC 26.9 / CUDA 13.4 の同じ計算機で確認しました。GPU は既存の MPS サーバーを
使い、全行とも元の入力、18 MPI × 1 OpenMP、25 SCF、`140 × 140 × 140` 格子です。
CPU の固有値計算は `elpa2` に揃えています。各計測中に他の計算・ビルドは実行していません。

| 実行経路 | Total Computational Time (秒) | MPI 起動を含む経過時間 (秒) |
| --- | ---: | ---: |
| 移行前 GPU＋MPS（`ac6ebad2`、未最適化の C++ コア） | 132.026 | 135.813 |
| `v3.1` GPU＋MPS（`47556613`、同じ SDK で再構築） | 107.949 | 111.711 |
| 移行後 GPU＋MPS | 105.147 | 108.919 |
| インストールした `work/openmx` の再測定 GPU＋MPS | 104.723 | 108.418 |
| Intel コンパイラ版 CPU（`~/openmx4.0`） | 800.460 | 803.226 |
| RAII 作業領域・Release コアを使う CPU | 756.165 | 759.086 |

GPU は 2 回とも約 105 秒で、計算時間は移行前から約 20.4–20.7% 短縮し、
同環境の `v3.1` よりも短くなりました。Force は 20.886 秒から 15.013–15.161 秒です。
CPU 全体も Intel コンパイラ版に対して
約 5.5% 短縮しています。CPU 測定後に加えた VNA のバッチ制御・密度行列 gather・
Force のデバイス選択は GPU 経路だけで使い、CPU 経路の数値処理は同一です。

移行後 GPU と移行前 GPU の差は、エネルギー `1.03e-12 Ha` 以下、全 Force 成分で
`1.01e-12 Ha/bohr` 以下です。CPU と Intel 版の差は、エネルギー `1.69e-10 Ha`、
最大 Force `1.36e-10 Ha/bohr` でした。原子・格子・反復回数も照合しています。
この結果は、この入力と GeForce 環境での測定です。

入力・実行条件・バイナリとコアの SHA-256・全物理量・処理別時間は、
`build/refactor-performance-20261019/{before-gpu,v3.1-gpu,final-gpu,final-repeat-gpu,intel-cpu-configured,after-cpu}/result.json`
に保存しています。`hardware.json` は GPU・ドライバー・MPS の確認結果です。
`after-cpu/result.json` には、Intel/NVHPC 間の出力座標の丸め差を再検証した記録も残しています。
A100/H100 と AMD の実機での速度は、この結果には含みません。

再測定したインストール版の SHA-256 は次のとおりです。

```text
work/openmx:                  96e194dde008395c5aeded99c1dcb050c9161db2a3e0a996b8fa912deeef9311
work/lib/libopenmx_core.so:    0ea7ac892bea68ab7f28a4c188bfa324833a356b75dfc6ffb32139b3b3be7ece
元の sidia333 入力:            9c5466abf560c6481b83ec9a51976f4b218856a1da330e5a209b9041239decd0
```

## 同じ入力で測る

単一構成のビルドは `Release` が既定です。CMake の以前の既定である空の
`CMAKE_BUILD_TYPE` は、ネイティブ C/Fortran が最適化されても C++ コアを
最適化なしで構築していました。既存のビルドディレクトリも再構成してください。
明示的な `Debug` は計測用途には使いません。

```sh
cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=Release
cmake --build build/cuda --parallel 4
```

`tools/run_refactor_case.py` は計算専用のディレクトリを作成し、入力、実行コマンド、
環境設定、バイナリの SHA-256、原子・座標・積分格子、SCF 回数、エネルギー、
全力成分と処理別時間を保存します。既存の計算ディレクトリは上書きしません。
`DATA.PATH` を絶対パスにし、CPU 比較では `scf.eigen.lib` を `elpa2` に揃えます。

```sh
python3 tools/run_refactor_case.py \
  --binary build/cuda/openmx \
  --input work/sidia333_nc_cluster/sidia333_nc_cluster.dat \
  --output build/performance/candidate-gpu \
  --mpiexec /path/to/matching/openmpi/bin/mpirun \
  --mode gpu --ranks 18 --threads 1

python3 tools/run_refactor_case.py \
  --binary build/cuda/openmx \
  --input work/sidia333_nc_cluster/sidia333_nc_cluster.dat \
  --output build/performance/candidate-cpu \
  --mpiexec /path/to/matching/openmpi/bin/mpirun \
  --mode cpu --ranks 18 --threads 1

python3 tools/run_refactor_case.py \
  --binary "$HOME/openmx4.0/work/openmx" \
  --input work/sidia333_nc_cluster/sidia333_nc_cluster.dat \
  --output build/performance/intel-cpu \
  --mpiexec /path/to/intel/mpi/bin/mpiexec \
  --mpi intel --mode cpu --ranks 18 --threads 1 \
  --library-path /path/to/intel/compiler/lib:/path/to/intel/mpi/lib:/path/to/mkl/lib
```

GPU 比較は MPS が動いている状態で実行します。MPI ランチャーは各バイナリの
MPI ライブラリと一致させ、計測中に他の計算やビルドを並行しません。
古いバイナリを保存する場合は対応する `libopenmx_core.so` も保存し、
`--core-library /path/to/saved/libopenmx_core.so` で版を固定します。
`--reference <比較元の result.json>` を加えると、入力由来、並列数、反復数、
原子・格子を照合し、エネルギーと全力成分の差が `1e-7` を超えた実行を失敗とします。
座標は同一入力の SHA-256 と原子の対応を確認し、出力の小数点以下 5 桁の
丸めに対して 1 桁分（`1e-5`）だけ許容します。入力の座標は変更しません。
途中終了や未完了の計算は比較元として使いません。

## GPU ごとの実行方針

| 環境 | 既存の演算方針と検証対象 |
| --- | --- |
| GeForce | 弱い FP64 のデバイス向けに GEMMul8、double-float の格子演算を使う。限られた GPU メモリと MPS での複数 MPI ランクを含めて検証する。 |
| A100 | 通常の行列積と格子演算は native FP64。フル A100 の大きい実数 forward 積に限定した既存のサイズ別方針を保持する。MIG や小さい構成へ閾値を流用しない。 |
| H100 | native FP64 を基準にする。A100 の実測由来の forward 積の閾値や GeForce の測定結果を流用しない。 |
| AMD/HIP | MI300A、MI300X、Radeon の既存の実行方針を保持し、対象実機で別に測る。 |

GPU 型番だけで新しい精度低下を有効にしません。明示的な精度指定、CPU 実行、
GPU 容量不足時の復旧経路と、物理量の検証を各移行段階で維持します。

### MPS とメモリ確保の排他制御

`openmx_gpu_workers_run_prepared` は、全入力・作業領域・非公開の出力を
`PREPARE` で確保する処理だけが使います。確保と転送を完了して同期した後、
GeForce 向けには GPU の共有排他ロックを解放し、MPS による他ランクとの
並行実行を許可します。各バッファの所有権は実行元が保持し、全ワーカーの
成功を確認してから結果を反映します。準備に失敗した場合はロックを保持して
後片付けを完了し、既存の容量不足時の復旧経路に戻ります。

疎な密度行列、共通 task/vector、XC 格子を
この契約に合わせています。実行中にも確保する処理は従来の API を使います。
CUDA の FP32/FP64 性能比が 8 以上のデバイスでは自動で有効にし、A100/H100 と
HIP の既定は直列実行を維持します。異なる GPU を含むワーカー群では、全 GPU が
条件を満たす場合だけ並行実行します。比較実験では
`OPENMX_GPU_WORKER_OVERLAP=0` / `1` で明示指定でき、未指定または `auto` が
自動選択です。演算精度や加算順序を変更する設定ではありません。
VNA・Force の大きい archive バッチを GPU で実行する場合は排他実行します。

### Force の archive と CPU 経路

Force4B の完全な archive と作業領域を共有 GPU メモリに常駐できる場合は、
GPU を使います。常駐できない弱 FP64 デバイスでは、既存の融合 CPU trace を
選択し、巨大な archive の詰め替え・転送・GPU 上の直列実行を避けます。
A100/H100 は native FP64 が強いため、既存の GPU archive と容量不足時の
復旧経路を維持します。判断は archive の確保・通信の前に全 MPI ランクで
統一するので、ランクごとのデバイス状態が異なっても collective の参加が分岐しません。

`OPENMX_FORCE4B_ARCHIVE_GPU=1` は常駐できない場合も GPU archive を試し、
`0` は融合 CPU trace を選びます。常駐する GPU 経路はこの設定の対象外です。
未指定または `auto` はハードウェアに基づきます。
`OPENMX_GPU_CAPACITY_TEST_RECOVERY=1` の復旧テストでは従来どおり GPU の容量不足処理を
通します。いずれも projector の精度や Force の式を変更しません。

### VNA のバッチサイズ

GeForce では DS_VNA と VNA2/3 のバッチを、実際に同じ GPU を使うランク数と
空きメモリに合わせて縮めます。CUDA の可視デバイス番号がランクごとに異なる場合も、
PCI バス識別子で共有数を求めます。空きメモリから 256 MiB を除いた共有枠に
入力・出力・全作業領域を含めて収め、最小 1 原子対まで縮めた後は共通の容量復旧を使います。
CPU に退避したランクも同じ collective に参加します。

バッチサイズは全ランクの最小値に揃え、異なる GPU や異なる空き容量が混在しても
collective の呼び出し回数を一致させます。A100/H100 だけのジョブでは従来のバッチ上限を
維持します。GeForce でのサイズ調整を native FP64 向けの固定チューニングには使いません。
