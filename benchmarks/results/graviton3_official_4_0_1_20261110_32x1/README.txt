OpenMX 4.0 + patch4.0.1, Graviton3 / Neoverse V1, 32 MPI x 1 OpenMP
2026-11-10

公式配布のOpenMX 4.0に公式patch4.0.1を適用し、既存の
graviton3_sve256_20261105と同じ数値ライブラリ・アーキテクチャ設定で
ビルドしたCPU版の測定記録。ウォームアップ1回の後に測定1回を実施した。
アプリケーションには、後述のELPA互換接続とビルド・実行環境への変更がある。

結果
  実行              MPI起動・出力込み実時間 (秒)    OpenMX内部実時間 (秒)
  warmup-01                          555.832                    553.275
  measurement-01                     555.248                    552.644

報告する測定値は555.248秒。time.monotonic()でmpirun起動から終了までを
計測し、rank監査、MPI起動・終了、結果出力を含む。OpenMX内部実時間は
出力のElapsed.Time.。処理別時間はrank最大値であり、加算して総時間には
ならない。summary.jsonには丸め前の実時間を保存した。

両実行は終了コード0、正常終了、SCF 25反復。ウォームアップは測定値から
除外した。測定標本は1個で、中央値や統計的なばらつきは算出していない。
ユーザーの依頼で中断した先行検証は、この記録・時間集計に含めない。
この2回の実行中にはビルドや別の計算を重ねていない。

測定条件
  AWS Graviton3 / ARM Neoverse V1、32コア、SVE VL=256 bit。
  Amazon Linux 2023、AArch64。OS・カーネルの版はbuild/build-info.json。
  sidia333_nc_cluster.dat、Si 216原子、非共線スピン、Cluster、GGA-PBE。
  Si7.0-s2p2d1 / Si_PBE19、200 Ry、140 x 140 x 140グリッド。
  32 MPI x 1 OpenMP、OpenMX引数 -nt 1。
  --bind-to core --map-by slot:PE=1で各rankを別コアに固定。
  各rankでSVE幅256 bit、OpenBLAS NEOVERSEV1、BLAS 1スレッド、
  zgemm_のライブラリ実体を起動直前に監査。rank-audit.jsonに保存。
  OMP_NUM_THREADS=1、OMP_DYNAMIC=FALSE、OMP_MAX_ACTIVE_LEVELS=1、
  OMP_PROC_BIND=close、OMP_PLACES=cores、OPENBLAS_NUM_THREADS=1、
  GOTO_NUM_THREADS=1。OpenMP版OpenBLASのスレッド数はOMP設定に従う。
  scf.maxIter=25、scf.criterion=1e-13、MD.Type=Opt、MD.maxIter=1。

原入力のSHA256は9c5466abf560c6481b83ec9a51976f4b218856a1da330e5a209b9041239decd0。
原入力からの変更は参照先と同じ4項目:
  DATA.PATHを今回インストールしたDFT_DATA19の絶対パスに変更
  level.of.fileout=1
  scf.eigen.lib=elpa2
  scf.restart=off
両回は同一の入力バイト列・実行バイナリを用い、それぞれ新規ディレクトリで
実行した。25反復でSCF収束条件には到達していない。固定反復数の測定である。
検証停止の依頼後、数値の回帰試験や参照先とのエネルギー・力の比較は
実施していない。生の出力は独立した解析のため圧縮保存した。

ビルド
  公式本体: https://www.openmx-square.org/openmx4.0.tar.gz
  公式パッチ: https://www.openmx-square.org/bugfixed/26May08/patch4.0.1.tar.gz
  適用方法: https://www.openmx-square.org/bugfixed/26May08/README.txt
  本体SHA256: 8d5338faf70885f276352bbd2826cdfed2ffd08f33eca58752666d79a7d0c3bf
  パッチSHA256: c5312eeee13e17e0123beeb4eb2379bcf7c7cafa1815b1dcf6846452f9620bef
  Band_DFT_Dosout.c、Mulliken_Charge.c、work/GaAs.datを公式パッチと照合。
  パッチはバージョン表記を変更しないため、実行時の表示は4.0のまま。

  GCC/G++/gfortran 14.2.1、Open MPI 4.1.2、FFTW 3.3.8。
  OpenBLAS 0.3.34、commit e0166008be8e466242aa76b2ff75ce3f0fbf574a。
  TARGET=NEOVERSEV1、DYNAMIC_ARCH=0、USE_THREAD=1、USE_OPENMP=1、
  NUM_THREADS=32、NO_AFFINITY=1。
  ScaLAPACK 2.2.3、静的リンク、OpenMXのNBLK=32。
  ELPA 2026.02.002、OpenMP、静的リンク、実数・複素数ともSVE256_BLOCK2。
  OpenMX/ScaLAPACK: -O3 -mcpu=neoverse-v1 -msve-vector-bits=scalable
                    -fno-fast-math -ffp-contract=off
  ELPA共通Fortran: -O2 -fPIC
  ELPAのCカーネル: 上記アーキテクチャ・精度設定に加え-O2 -fPIC、
                   -march=armv8-a+sve -msve-vector-bits=256
  OpenMXは-std=gnu11、-Dnosse、-fopenmp、-fcommon等を使用。
  実際の全設定はbuild/makefile.gzとbuild/build-info.jsonに保存した。

公式配布版は古いELPAを同梱しているため、参照先リポジトリの
source/openmx_elpa_cpu.c/.hを基に現行ELPAへ接続した。環境変数・stdoutへの
アクセスは公式版へ合わせた。MPI_Init_threadでMPI_THREAD_SERIALIZEDを要求し、
NBLKを32、既定DATA.PATHをインストール先へ設定した。AArch64では公式の
nosseオプションを使用する。これらの変更を、公式4.0.1からの差分として
build/platform-integration.patchに保存した。アダプタの参照コミットは
b218d3748fbc6450d1e3b40fd1cbfaf289efa8bc。

参照先のcore/src/host_grid_simd.cppとhost_force_simd.cppによる独自の
アプリケーションSIMD改良は、この公式ソースを基にしたビルドには含まれない。
従って、これは同リポジトリ最終版の再ビルドやSIMD単独効果の測定ではない。
参照先の3回中央値と本記録の1回測定は、実装とOS・カーネルの版も異なる。

記録ファイル
  summary.json/.txt、timings.csv、phase-timings.csv: 実時間と処理別時間。
  original.dat.gz: 変更前の入力。
  各run/input.dat、command.json、result.json: 実入力、コマンド・環境・結果。
  各run/rank-audit.json、linked-libraries.txt: コア・SVE・BLAS監査とリンク情報。
  各run/run.log.gz、sidia333_nc_cluster.out.gz: 生ログと出力の可逆圧縮。
  manifest.json: 圧縮前のハッシュ・サイズと保存方針。
  build/: ビルド情報、makefile、ソースファイルのハッシュ、ELPA接続等の差分。
  reproduce/: 実際の測定・ビルドスクリプトの保存版。絶対パスは測定時の記録。
  SHA256SUMS: 公開する各ファイルのハッシュ。

大きな再開・密度・可視化ファイルと実行バイナリは測定ホストに保持している。
本アーカイブの確認は、このディレクトリで sha256sum -c SHA256SUMS を実行する。
再測定する場合は入力のDATA.PATH、実行バイナリ・ライブラリ・スクリプトの
絶対パスを変更し、新規ディレクトリでウォームアップ1回、測定1回を順次実行する。

参照: ../graviton3_sve256_20261105/README.txt
https://github.com/dc1394/openmx4.0_gpu/tree/main/benchmarks/results/graviton3_sve256_20261105
