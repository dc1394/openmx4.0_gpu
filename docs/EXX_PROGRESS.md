# EXX 実装の進捗

## 現在の状態

- 完了したマイルストーン: M0
- 作業中: なし（次は M1）
- 基準コミットからのブランチ: `feature/exx`（基準 `4e0ee247438a3c9358e95adee00a6ff60a4c618b`、リモートには未プッシュ）

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
| M0 | 大域状態のゲート（`tools/audit_global_usage.py --check-no-new`） | 変更の前後で報告が同じ。M0 の変更で増えた組はない。ゲート自体は基準コミットの時点で失敗する（「未解決の問題」） | 2026-11-23 | ホスト。clang 21.1.8、NVHPC 26.9 |

試験の入力、標準出力、ログは `build/exx-m0/` にあります（git の管理外）。

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

## 決めたこと

- （2026-11-23）CPU 版は上のコンテナの中でビルドして実行する。指示書の `sudo apt-get install` は実行していない。
- （2026-11-23）ビルドディレクトリーは `build/exx-cpu` にした。指示書の `build/cpu` には 2026-10-28 の別の構成（MKL）のビルドが残っていて、構成を変えた再利用を CMake が拒否する。
- （2026-11-23）`-DOPENMX_COPY_TO_WORK=OFF` を付ける（CI と同じ指定）。この作業ツリーの `work/openmx` は GPU 版の実行ファイルなので、CPU 版で上書きしない。実行には `build/exx-cpu/openmx` を直接使う。
- （2026-11-23）試験は `build/exx-m0/` の中で実行し、`work/` に出力を作らない。
- （2026-11-23）`HSE` と `NLX` の拒否は、`scf.XcType` を読んだ直後に置いた。書き方は既存の入力エラーと同じ（ホストのランクが表示、`MPI_Finalize()`、`exit(0)`）なので、終了コードは 0 になる。
- （2026-11-23）指示書、参照実装、フィクスチャーは、指示書の冒頭の表の場所に置いた。フィクスチャーの置き場所 `modules/exx/tests/fixtures/` には、まだこのファイルしかない（モジュール本体は M2 で作る）。
- （2026-11-23）`.gitignore` の `*.md`（MD の出力用の規則）が Markdown の文書にも当たるので、新しい文書は `git add -f` で追加した。既存の `docs/*.md` と同じ扱い。

## 未解決の問題

- （2026-11-23）大域状態のゲートは、基準コミット `4e0ee247` の時点で失敗する。ベースライン `docs/global-usage-baseline.json`（最後の更新は `8161ec64`）にない組が 15 個ある。内訳は `Open_FinalCoord_Files`→`filepath` の 1 個と、`Unfolding_Bands_Col`／`Unfolding_Bands_NonCol` の OpenMP 領域の 14 個で、コミット `3266dcdd` と `4e0ee247` で入った。EXX の変更による増加ではない。ベースラインを更新するまで、ゲートは変更の前後の報告の差で確認する。
- （2026-11-23）同じゲートで、`source/elpa_cosma_bridge.c` と `source/openmx_elpa_cpu.c` の 2 つが解析されない（`elpa/elpa.h` が見つからない）。基準コミットでも同じ。
- （2026-11-23）`source/Input_std.c:5032-5040` の試作用のハイブリッド入力（`if (XC_switch==5)` の中）と、`source/Band_DFT_Col_NEGF.c:743` の `if (5==XC_switch)` は、M0 の変更で到達しなくなった。コードは残してある。
- （2026-11-23）M1 で使う libxc の値を作る手段が、まだない。ホストには libxc も `pylibxc` も `pyscf` も入っていない。`uv` はあるので、仮想環境に入れる方法を M1 の最初に試す。

## 利用者への確認待ち

- 試作用のハイブリッド入力（`scf.xmin.Rec.Coulomb`、`scf.xmax.Rec.Coulomb`、`scf.Ng1.Rec.Coulomb`、`scf.Nrank.Rec.Coulomb` と `Allocate_Arrays(15)`）を削除するか（指示書の 8 節と 1.3 節）。
- 大域状態のベースラインを、基準コミットの状態（EXX と関係のない 15 組を含む）に更新するか（1.3 節）。
- CPU 版をコンテナの中でビルドして実行する方針でよいか。ホストでビルドする場合は、上の 6 つの apt パッケージを入れる必要がある。
