# RTX 5080 側 測定手順書(計画 v2.6 §7.2 / 8.2 / 8.3 / 13、§18 項目 11・14 対応)

対象: 手元マシン(Core i9-10980XE 18C / RTX 5080 16 GB / PCIe 3.0)。
Pegasus 側の本測定は v2.0_thesis タグから取得済み(work/siprod_thesis_result.txt ほか)。
本書は 5080 側で残っている全測定を、同じタグ・同じ規約で取るための手順。

**大原則(計画より)**
- H100 と 5080 の絶対時間は製品比較に使わない。各 GPU 内の cuBLAS/GEMMul8 比だけを論じる(§6)。
- GEMMul8 は INT8 / `num_moduli=15` / fastmode off 固定。サイズを上げるために moduli や
  workspace policy を変えない(§7.2/13.1)。
- NC(band/cluster)の正式検証点は 64 / 128 / 216 原子。216 が確認済み上限。
  216 超は「短い preflight」のみ、安全余裕 = max(2 GiB, 総VRAMの15%) = **2.4 GiB** を
  割ったらそこが 16 GB GPU の実用上限という**結果**として報告する(§7.2)。
- col(band/cluster)は 216 + 「384/512 の preflight で決めた最大安全点を 1 点」(§7.2/8.2)。
- 測定値は Max_Time(最遅 rank)。本測定 5 反復、CV>2% なら +2 反復を 1 ラウンドだけ。
- 各ランの真正性は `<System.Name>.manifest.json` で証明する(release_tag / dense path /
  fallback=0 / MPS / VRAM ピーク)。ログの目視より manifest を正とする。

---

## 0. 事前チェックリスト

| # | 項目 | 合格条件 |
|---|---|---|
| 0-1 | OS | **ネイティブ Linux**。WSL2 では MPS が使えないため、MPS 系列(ほぼ全部)が測れない。Windows 機なら Ubuntu を直接ブートすること |
| 0-2 | ドライバ / CUDA | Blackwell (sm_120) 対応: CUDA **12.8 以上**のツールキットとそれに対応する NVIDIA ドライバ |
| 0-3 | コンパイラ | NVHPC(nvc + OpenMPI 同梱)。バージョンは自由だが、記録する(manifest に自動記録される) |
| 0-4 | MPS 動作確認 | `nvidia-cuda-mps-control -d` → `echo get_server_list \| nvidia-cuda-mps-control` が応答 |
| 0-5 | データ | DFT_DATA19 一式、ディスク 30 GB 以上 |
| 0-6 | 熱・電力 | `nvidia-smi -q -d CLOCK,POWER` を測定前後に記録。サーマルスロットリングが出た rep は破棄して取り直し(Pegasus の slow-node 規約の 5080 版) |

## 1. コード取得とビルド(タグ厳守)

```sh
git clone git@github.com:dc1394/openmx4.0_gpu.git
cd openmx4.0_gpu
git checkout v2.0_thesis          # detached HEAD で良い。branch 先端ではなくタグに固定
git submodule update --init source/third_party/GEMMul8 source/third_party/fftw3
cd source
# Makefile 冒頭の自機向け CC/FC/LIB を設定(WSL 例をベースに CUDA 12.8+/sm_120 に変更)
#   nvcc 側: GEMMUL8_GPU_ARCH=120 を必ず指定(H100 用の 90 のままでは 5080 で動かない)
make fftw3
make -j16 all GEMMUL8_GPU_ARCH=120
```

**合格ゲート G1**: `make` がエラー 0 で完走し、`manifest_buildinfo.h` が
`OPENMX_BUILD_RELEASE_TAG "v2.0_thesis"` を含むこと(タグ checkout の証明)。
`md5sum work/openmx` を記録(以後この 1 バイナリで全測定)。

**合格ゲート G2**: `cd work && mpirun -np 16 ./openmx -runtest -nt 1` が **14/14**、
diff Utot ≤ 1e-9。runtest.result を保存。

既知の罠: GEMMUL8_COMMIT を変えた再ビルド時は `third_party/GEMMul8` の手動
checkout + `src/gemmul8_openmx.o` と `lib/libgemmul8.a` の削除が必要(make は
黙って旧版を再利用する)。v2.0_thesis のピン(833e5761)のままなら不要。

## 2. ランク数と実行環境の固定

- **全測定 `-np 16 -nt 1` に固定**(flat MPI)。8 計算 k 点に 2 ranks/k で割り切れ、
  OS と MPS デーモン用に 2 コア残る。MPS off/on 比較も同一 np(§8.3 の要件)。
  ※一度決めたら 5080 の全系列で変えない。変えた場合は別系列として報告。
- CPU 参照値(elpa2)は「参考値」扱い(§8.2)。np は同じ 16 で 1 系列だけ。
- 各ラン前に GPU を占有(X/ブラウザ等の GPU プロセスを止める)。
- MPS: Pegasus の `mps_node.sh` をそのまま流用可(パイプディレクトリを /tmp 配下に
  作る方式)。ラン後は必ず quit。

## 3. デッキ(入力)一覧

すべて release/v2.0_thesis の work/ に既存、または sidia.dat から機械生成:

| 原子数 | セル | col n | NC n2 | 出どころ |
|---:|---|---:|---:|---|
| 64 | conv 2×2×2(a=10.862 Å 立方) | 832 | 1664 | `gen_siacc64.sh` の幾何生成部を流用(cluster 版は siacc64_o1 の deck そのもの) |
| 128 | **prim 4×4×4**(fcc 原始胞 ×4、等方) | 1664 | 3328 | 新規生成。変換行列 [[0,2,2],[2,0,2],[2,2,0]](conv 基準)を input repository に記録(§7.1 スーパーセル規則) |
| 216 | conv 3×3×3 | 2808 | 5616 | `sib_col216_*` / `sib_nc216_*`(band)、`sip_c*216_*`(cluster)の deck を byte 流用 |
| 384/512 | col preflight 専用 | 4992/6656 | – | `sib_col384/512` 系(H100 探索で使用した候補 deck) |

deck 共通の編集(Pegasus 本測定と同一): `scf.maxIter 25` + `scf.criterion 1.0e-15`
(25 回固定)、preflight は `scf.maxIter 3`。cluster は solver=cluster / Kgrid 1 1 1、
band は Kgrid 2 2 2。cfg は o(`scf.gemmul8.enable off`)と g(on)の 2 種。

ジョブスクリプトは NQSV ヘッダを外したローカル版を 1 つ作れば良い。必須要素は
sip 系スクリプトと同じ: MPS 起動/確認/停止、`nvidia-smi` 5 秒サンプラ、
watchdog、TIMING 抽出、`.env`(ハード情報)記録。qsub の代わりに直接実行。

## 4. 測定プログラム

### R1. サイズ / VRAM preflight(3-SCF、各 1 回)

| 系列 | 点 | 判定 |
|---|---|---|
| NC cluster | 64, 128, 216 × {o,g} | exit 0・fallback 0・VRAM 余裕 ≥ 2.4 GiB |
| NC band | 64, 128, 216 × {o,g} | 同上 |
| col band / col cluster | 216 → 384 → (384 が通れば) 512 × {o,g} | 同上。**通った最大点を 1 点だけ本測定に追加** |
| (任意) NC 240/256/288 | 短時間 preflight のみ | 上限の所在の報告用。本測定には入れない |

判定はすべて manifest で行う:
```sh
python3 - <<'EOF'
import json,glob
for f in sorted(glob.glob("*/*.manifest.json")):
    d=json.load(open(f)); g8=d["gemmul8"]; m=d["memory"]
    print(f, d["dense_solver"]["path"], "fb=",g8["d_fallbacks"]+g8["z_fallbacks"],
          "vram_used=%.1fG margin(free_min)=%.1fG"%(m["peak_device_vram_used_mb"]/1024,
                                                    m["vram_min_free_mb"]/1024))
EOF
```
**vram_min_free ≥ 2.4 GiB が安全余裕の判定値**。GEMMul8 の workspace 上限は既定
30%(16 GB では 4.8 GiB)。NC216 の Z-GEMM workspace がこれを超えて B70 fallback が
出た場合、それは「G8 の NC216 が 16 GB では workspace 制約で cuBLAS に落ちる」と
いう**正式な結果**(計画 §7.2 の報告方針)。policy は変えない。

### R2. Layer 3 本測定(25-SCF 固定、**各 5 回**、MPS on)

- NC band: 64, 128, 216 × {o, g}
- NC cluster: 64, 128, 216 × {o, g}
- col band: 216 + R1 の最大安全点 × {o, g}
- col cluster: 216 + R1 の最大安全点 × {o, g}
- (参考)CPU elpa2 np16: 216 の 4 solver × 3 回のみ

計 約 70〜80 ラン。CV>2% の点だけ +2 反復を 1 ラウンド。
評価は Pegasus と同じ: `summarize_siprod_thesis.sh` の manifest 判定 + 比率計算を
5080 用に流用(reps() と組合せ表を差し替えるだけ)。

### R3. MPS ablation(§8.3、同一 np で off/on)

- NC band 216: cuBLAS off/on × 各 3 回
- NC cluster 216: cuBLAS off/on × 各 3 回
- 代表 1 点(推奨: NC cluster 216)だけ GEMMul8 でも off/on × 各 3 回(相互作用確認)

MPS-off ランの manifest は `mps.detected` がパイプ環境変数で真になり得る
(Pegasus で確認済みの既知事項)。**off の証明は MPS サーバ無応答の記録**
(`get_server_list` が空)をジョブログに残すこと。

### R4. 精度(§13.2-A の 5080 分)

- runtest 14/14(G2 で取得済み)。
- 変位 Si216(work/siacc_dsi 系 deck 流用、np16 に合わせるだけ): CPU / cuBLAS /
  GEMMul8 の ΔE・ΔFmax、同一バックエンド 2 回で run-to-run。
  目標は Pegasus と同じ |ΔE|≤1e-10 Ha(envelope 併記)/ |ΔF|max≤1e-6。

### R5. 回収物(そのまま release/v2.0_thesis に commit する)

各ランディレクトリの `*.dat *.sh *.joblog *.out *.env *.smi(相当) *.manifest.json`、
runtest.result、集計 txt、`nvidia-smi -q` のクロック/電力記録。
持ち帰り後、Pegasus 側で `git add -f`(work/ の同パターン)→ commit → push。

## 5. 期待される見どころ(解釈の指針)

- H100 では G8 は最速でも同等〜わずかに負け(Set_Ham 干渉時 1.15×)だったが、
  5080 は native FP64 が桁違いに弱いので、**G8 正の高速化が主張の中心**(§5)。
  Diag 比で効きが最も見える。
- 64/128 は GPU 立ち上がり/overhead 支配になり得る。G8 の有効性が 216 の 1 点に
  限られる場合は RTX PRO 6000 の条件付き追加を検討(§8.4 の発動条件)。
- VRAM ピーク(manifest)対 16 GB の余裕は Fig.7(b) の実データになる。

## 6. 予算・工数目安

全プログラムでおよそ 150〜200 GPU 時間ではなく **実時間 1〜2 日**(1 ランは 3-SCF
で数十秒〜数分、25-SCF で数分〜十数分の見込み。まず R1 で実測してから R2 の
総時間を見積もり直すこと)。電気代以外のポイント消費はなし。
