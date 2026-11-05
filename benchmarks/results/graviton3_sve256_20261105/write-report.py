#!/usr/bin/env python3
"""Generate the local report only after the complete benchmark series passes."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
repo = root.parents[2]
def read(name):
    return json.loads((root / name).read_text())

result = read('analysis.json')
ablation = read('own-ablation.json')
micro = read('own-kernel-summary.json')
assert result['all_checks_passed'] and ablation['all_checks_passed']
assert result['complete_blas_audits']
assert len(result['baseline']['samples']) == len(result['optimized']['samples']) == 3
checkpoint = read('optimized-01/command.json')
final_command = read('own-final-01/command.json')
for key in ('binary_sha256', 'blas_sha256', 'environment', 'ranks', 'threads'):
    assert checkpoint[key] == final_command[key], key
comparison = result['comparisons']
de = max(x['energy_difference_Ha'] for x in comparison)
df = max(x['max_force_difference_Ha_bohr'] for x in comparison)
due = max(x['max_scf_uele_difference_Ha'] for x in comparison)

pilots = []
for folder in sorted(root.iterdir()):
    if not (folder.name.startswith('tune-') or folder.name == 'own-kernels-pilot'):
        continue
    if not (folder / 'result.json').exists():
        continue
    r = read(folder.name + '/result.json')
    c = read(folder.name + '/command.json')
    assert r['returncode'] == 0 and r['normal_finish'] and r['rank_audit_passed']
    pilots.append(dict(name=folder.name, wall_seconds=r['wall_seconds'],
                       phases=r['phases'], ranks=c['ranks'], threads=c['threads'],
                       actual_blas_threads=r['actual_blas_threads'],
                       steps=c['steps'], environment=c['environment'],
                       blas_library=c['blas_library'], blas_sha256=c['blas_sha256'],
                       numerical_energy=r['energy']))
(root / 'pilot-results.json').write_text(json.dumps(pilots, indent=2) + '\n')

lines = [
    'OpenMX 4.0 CPU / SVE256 最適化・検証結果 (2026-11-05)',
    '',
    f"指定入力の25反復、各3回の中央値: {result['baseline']['median']:.3f} 秒 -> "
    f"{result['optimized']['median']:.3f} 秒、{result['speedup']:.4f} 倍速 "
    f"({result['reduction_percent']:.2f}% 短縮)。",
    'CPU版のみ。GPU用のライブラリや実行ファイルは使用しない。',
    '',
    '測定対象と条件',
    '  AWS Graviton3 / Neoverse V1、32物理コア、SVE VL=256 bit。',
    '  元コミット: 3871b1f0fb880aa1ee3b3201e7f51d6bdf6c6e96',
    '  入力: benchmarks/inputs/sidia333_nc_cluster.dat',
    '  Si 216原子、非共線スピン、Si7.0-s2p2d1 / Si_PBE19、PBE、200 Ry。',
    '  32 MPI x 1 OpenMP、各rankを個別コアに固定、BLASも実測1スレッド。',
    '  scf.maxIter=25、SCF収束条件1e-13、MD.Opt 1回、グリッド140 x 140 x 140。',
    '  25回ではSCF収束条件に到達しない。これは固定反復数の速度比較である。',
    '  共通変更: DATA.PATHを絶対パス、scf.eigen.lib=elpa2、scf.restart=off、',
    '  level.of.fileout=1。最後の設定は全エネルギーと力の出力・比較のため。',
    '  MPI起動・出力を含む実時間を計測。各回を新規ディレクトリで実行し、',
    '  ビルドや別のベンチマークとは重ねていない。',
    '  最終集計: baseline-02/03/04 と own-final-01/02/03。',
    '  baseline-01は詳細BLAS監査を追加する前の初期値で、最終中央値から除外。',
    '  入力/実行ファイル/共有ライブラリのSHA256、zgemm_の実体、各rankの',
    '  コア割当/SVE幅/BLAS名とスレッド数を各runのJSONに記録した。',
    '',
    '25反復の実時間 (秒)',
    '  取得時のCPU版 (3回): ' + ', '.join(f'{x:.3f}' for x in result['baseline']['samples']),
    '  最終SVE256版 (3回): ' + ', '.join(f'{x:.3f}' for x in result['optimized']['samples']),
    f"  独自カーネル改良前の中間版 (1回): {ablation['baseline']['median']:.3f}",
    '  比較元はスカラー版ではなく、取得時のSVE対応CPU版。',
    '  比較元: OpenBLAS 0.3.18 (armv8)、既存のホストSIMDとELPA設定。',
    '  中間版: 最終版と同じOpenBLAS/ELPA、ScaLAPACKブロック32、',
    '          コンパイラ設定。今回の主要な独自カーネル改良を追加する直前。',
    '  中間版と最終版はアプリケーション本体とBLASのSHA256、実行環境も同一。',
    '  独自カーネルを含むCore/SIMD共有ライブラリを変更した比較である。',
    f"  中間版からの追加改善: {ablation['speedup']:.4f} 倍速 "
    f"({ablation['reduction_percent']:.2f}% 短縮)。中間版は1標本である。",
    '',
    '処理別時間 (各runのrank最大値の中央値、秒; 加算して実時間にはならない)',
    '  処理                         取得時        中間版        最終版',
]
for phase in ('Diagonalization', 'Set_Hamiltonian', 'Set_Density_Grid', 'Force',
              'Set_ProExpn_VNA', 'Total_Energy', 'OutData'):
    a = result['phases'][phase]
    b = ablation['phases'][phase]
    lines.append(f"  {phase:26s} {a['baseline_median']:10.3f} {b['baseline_median']:12.3f} "
                 f"{a['optimized_median']:12.3f}")
lines += [
    '',
    'ライブラリ外の変更',
    '  core/src/host_grid_simd.cpp:',
    '    非共線密度とHamiltonianの4成分をSVE256の4 FP64レーンに配置。',
    '    密度行列を複数の格子点で共有し、軌道データの転置を除去。',
    '    Hamiltonianを4行x4列のレジスタタイルで更新し、13軌道の端数を削減。',
    '    共線スピンの密度にも独立4行のタイルを適用。',
    '    FP32軌道のロードとFP64変換にLD1W.D/FCVTを使用。',
    '  core/src/host_force_simd.cpp:',
    '    VNA射影子を64-bitレーンの下位32 bitへ直接ロードし、積の後の',
    '    レーン並べ替えを削減。積は元どおりFP32で丸めてからFP64加算。',
    '    項数/精度/向きごとのレジスタタイルを測定で選択し、不要なpadding計算を除去。',
    '    力の密度収縮では固定4ベクトルのタイルで軌道値を共有し、動的ループを除去。',
    '  j/i/格子点の加算順序、重複格子点へのscatter順序を保持。fast-mathなし。',
    '  変更はSVE256の分岐で適用し、他のISAの既存選択を保持。',
    '',
    '代表的な単体カーネル測定 (実計算全体の倍率とは別)',
]
for case in micro['representative_cases']:
    lines.append(f"  {case['kernel']}: {case['speedup']:.3f}x ({case['shape']})")
lines += [
    '  grid-v4: 1,332条件、VNA: 変更前後それぞれ6,000条件、',
    '  力の密度収縮: 120条件で参照式とのビット単位一致を確認。',
    '  CSV、比較用ソース、コンパイラ引数、再実行用スクリプトをこのフォルダに保存。',
    '  SVE gather、汎用Hamiltonian行タイル、過大なレジスタタイルは測定で',
    '  利点が出なかったため採用していない。',
    '',
    '数値検証',
    f'  全6本の比較で最大 |ΔE| = {de:.6g} Ha。',
    f'  全648成分の最大 |ΔF| = {df:.6g} Ha/bohr。',
    f'  各SCF反復の最大 |ΔUele| = {due:.6g} Ha。',
    '  216原子の座標、グリッド、実効入力ハッシュ、25反復を照合。',
    '  カーネル試験はゼロ長/端数/転置/重複点/halo/FP32/FP64/相殺も検証。',
    '  最終ビルドのCTest: 627件中626成功。唯一の失敗は取得時から存在する',
    '  build-backend-isolationのHIPソース2ファイルとsources.jsonのハッシュ不一致。',
    '  当該HIPファイルはHEADと同一。今回のCPU変更に伴う失敗ではない。',
    '  ELPAは6/8 MPI x 1/2 OpenMP、実数/複素数、ELPA1/2、n=257、',
    '  padded leading dimension、空rank、subcommunicator、残差・直交性を検証。',
    '',
    '採用したビルド設定',
    '  GCC/G++/gfortran 14.2.1、Open MPI 4.1.2、FFTW 3.3.8。',
    '  OpenBLAS 0.3.34 (e0166008be8e466242aa76b2ff75ce3f0fbf574a)、',
    '  TARGET=NEOVERSEV1 / DYNAMIC_ARCH=0 / USE_OPENMP=1。',
    '  ELPA 2026.02.002、実数/複素数ともSVE256_BLOCK2、ScaLAPACK 2.2.3。',
    '  CPU版のブロックサイズを設定可能にし、このビルドは32を使用。',
    '  C/C++/ScaLAPACK: -mcpu=neoverse-v1 -msve-vector-bits=scalable',
    '                  -fno-fast-math -ffp-contract=off',
    '  明示的ホストSIMD: -march=armv8-a+sve -msve-vector-bits=256',
    '                    -mtune=neoverse-v1 -fno-fast-math -ffp-contract=off',
    '  ELPAの明示的SVE Cカーネルは256 bit、共通Fortran部分は-O2 -fPIC。',
    '  GCC 14の全コードへの固定256-bit指定でVPSメタデータ試験が失敗したため、',
    '  自動ベクトル化にはscalableを選び、実機の256-bit VLで実行。',
    '  ELPA共通FortranへのNeoverse/SVE指定もn=257複素固有ベクトル検証で',
    '  失敗したため採用しない。精度試験を通過した組合せを使用している。',
    '  選択したBLASが同名のシステム旧版より先にロードされるようRPATHを修正。',
    '',
    'その他の比較',
    '  32x1 / 16x2 / 8x4 / 4x8を3反復で比較し、32x1を採用。',
    '  ブロック128/64/32/16を比較。32と16は近く、32を採用。',
    '  Arm Performance Libraries 26.07 (CPU LP64)も取得・数値検証・比較。',
    '  ブロック32の3反復でArmPL 97.394秒、OpenBLAS 96.703秒であり、',
    '  OpenBLASを採用。ArmPL比較用インストールはCPUのみでGPU依存はない。',
    '  全pilotの実測値はpilot-results.json。単一の短い試行は最終倍率に含めない。',
    '  profile-32x1はperf測定用、validate-openblas-25はビルド等と重なった',
    '  精度検証用なので最終時間集計から除外。pilot-32x1は旧BLASがロードされた',
    '  初期試行で除外。各々を成功した最終速度測定として扱わない。',
    '',
    '再現方法 (このマシンで実行)',
    '  通常のopenmxとwork/openmxは ~/.local/opt/openmx-sve256/bin/openmx を指す。',
    '  元のCPU版はopenmx-cpu-baselineで利用可能。installed-links.jsonに照合結果。',
    '  cd /home/ec2-user/openmx4.0_gpu',
    '  bash build-sve256-local.sh',
    '  OPENMX_LOCAL_TEST_BUILD=build/cpu-sve256 bash test-cpu-local.sh',
    '  python3 tools/test_sve256_local.py',
    '  python3 tools/bench_sve256_local.py new-measurement \\',
    '    --binary /home/ec2-user/.local/opt/openmx-sve256/bin/openmx \\',
    '    --mode sve256 --expect-blas-core NEOVERSEV1',
    '  既存run名は上書きしない。比較時は最初のCPU版の凍結コピー',
    '  baseline-install/bin/openmxを --mode auto --expect-blas-core ARMV8 で使用。',
    '  run-own-final-series.shは今回の実行順の記録 (既存名があるので再実行は停止)。',
    '  python3 benchmarks/results/graviton3_sve256_20261105/reproduce-probes.py',
    '  で凍結した単体実験を別名に再ビルド・再実行できる。実計算測定とは重ねない。',
    '',
    '公開した記録の検証',
    '  python3 benchmarks/results/graviton3_sve256_20261105/validate.py',
    '  MPIや測定時のバイナリがなくても、SHA256SUMS、圧縮前の元ログ/出力の',
    '  ハッシュ、出力から再抽出した全648力成分、rank監査、中央値を照合できる。',
    '  各runのrun.log.gzとsidia333_nc_cluster.out.gzは元ファイルの可逆圧縮。',
    '  rank-audit.jsonはrank-00.json等をrank順にまとめたもの。',
    '  格子可視化ファイル、再開用データ、インストール済みバイナリはローカル保管。',
    '  公開アーカイブには計測結果、精度検証、ビルド設定、再現用ソースを含める。',
    '  CMakeCacheやコマンド内の絶対パスは測定時の記録。別マシンでは変更が必要。',
    '  比較元を再ビルドする場合は、元コミット3871b1f0の別checkoutと',
    '  baseline-CMakeCache.txtの構成を使用する。最終ソースでの再ビルドとは区別する。',
    '  reproduce/とscripts-sha256.jsonは計測時のスクリプトの保存版。',
    '  現在のtools/analyze_sve256_local.pyは公開用gzipの読み込みにも対応している。',
    '',
    '記録',
    '  analysis.json/.txt: 変更前後各3回の集計と数値比較。',
    '  own-ablation.json/.txt: 同一ライブラリ設定の中間版との比較。',
    '  own-build.json / own-CMakeCache.txt / own-source-changes.patch: 最終版の構成。',
    '  source-changes.patch: 独自カーネル改良前の中間版の差分。',
    '  optimized-install / own-optimized-install: ローカル保管の計測実体の凍結コピー。',
    '  build/sve256-logs/own-final-ctest.log: 全回帰試験。',
    '  build/sve256-logs/elpa-default-final.log: ELPAの精度検証。',
    '  build/sve256-logs/profile-32x1-report.txt: 最適化前のperf内訳。',
    '  build/install-logs/unchanged-hip-source-check.json: 既存HIP照合エラーの根拠。',
    '  environment.json、各run/command.json・rank-*.json・result.json・run.log・out。',
]
(root / 'README.txt').write_text('\n'.join(lines) + '\n')
print(root / 'README.txt')
