# OpenMX Io

OpenMX のキーワード形式入力ファイル（`System.Name  h2o`、`scf.Kgrid 4 4 1`、`<Atoms.SpeciesAndCoordinates` …
`Atoms.SpeciesAndCoordinates>` のブロック）を読む C++20 ライブラリです。`openmx::io::KeywordFile` が
開いたストリームを所有し、スカラー（論理値・整数・実数・文字列・選択肢）、ベクトル、ブロックの
検索と、記録したエラー数・メッセージを提供します。`openmx::io::KeywordFiles` は元の OpenMX の入れ子
モデル（ファイルのスタック、先頭が応答、open ごとに 0 に戻るエラー数）です。大域変数も、プロセス終了も、
MPI もありません。本体の `Inputtools.c` はこのライブラリの薄い互換層になりました。

検索規則（各行の最初の空白区切りトークンを、大文字小文字を区別せず同じ長さで比較）、受け付ける綴り
（`on yes true .true. ok` / `off no false .false. ng`）、`sscanf` に基づく数値変換、ベクトル値の
カーソルの進め方、エラーメッセージの文面は、元の `Inputtools.c`（Hiori Kino）のものです。元のソースは
`tests/fixtures/` に凍結し、テストがライブラリとの一致を確かめます。意図的な差は次の 3 点です。

- 1 行の長さに上限がありません（元は 4096 文字で行が分割されました）。
- 文字列の格納先には容量を渡し、入り切らない値はエラーになります（元は無制限にコピーしました）。
- 実数の値が不正なとき、C++ API と C API は他の型と同じく `invalid`（-1）を返します
  （元の `input_double` は 0 を返しました。本体の互換層はその戻り値を保っています）。

CMake 3.25.2 以上と C++20 のコンパイラ・標準ライブラリでビルドできます。
このディレクトリだけを取り出して利用でき、MPI、BLAS、Kokkos、GPU SDK、
OpenMX の実行ファイル・大域変数ヘッダー、ネットワーク接続を必要としません。
コードのライセンスは元の OpenMX と同じです。

## ビルドと外部利用

リポジトリのルートで実行します。

```sh
cmake -S modules/io -B build/io -DCMAKE_BUILD_TYPE=Release
cmake --build build/io --parallel
ctest --test-dir build/io --output-on-failure
cmake --install build/io --prefix "$PWD/build/io-install"

cmake -S modules/io/examples/keywords -B build/io-example \
  -DCMAKE_PREFIX_PATH="$PWD/build/io-install"
cmake --build build/io-example
```

C++ からは `openmx/io/keyword_file.hpp`（`openmx::io::KeywordFile`、`KeywordFiles`）、C からは
`openmx/io/keyword_file.h`（`openmx_io_open`、`openmx_io_int`、`openmx_io_find` など。戻り値の規約は
元の `input_*` と同じ）を使います。

## テスト

- `io-legacy`: 凍結した元の `Inputtools.c` との一致（戻り値、値、エラー数、標準出力に出た診断の文面）。
  大文字小文字、タブ区切り、コメント、値のない行、不正な整数・実数・論理値、接尾辞付きの整数、
  ベクトルの不足、CRLF、`fscanf` / `fgets` + `ungetc` によるブロック読み、入れ子とエラー数の引き継ぎ。
- `io-semantics`: RAII（ムーブ、明示的な close、開けないファイルでの例外）、`block()`、
  メッセージの文面、診断ストリーム、スタックの容量。
- `io-invalid`: NULL や未オープンの状態、容量不足の文字列、空の選択肢。
- `io-c-api`、`io-package-consumer`: C からの利用と、インストールしたパッケージからの利用。
