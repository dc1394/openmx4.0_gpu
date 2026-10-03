#!/bin/sh
# Install autoconf/automake/libtool into a private prefix for hosts whose
# system autotools are too old for the bundled ELPA (its configure.ac needs
# autoconf >= 2.71; RHEL 8 ships 2.69).  Pass the resulting <prefix>/bin to
# build_gpusolver2_stack.sh through GPUSOLVER2_TOOL_PATH.
#
#   usage: build_host_autotools.sh <prefix> [jobs]
#
# The release tarballs are downloaded from $GNU_MIRROR (default
# https://ftp.gnu.org/gnu) into <prefix>/src and checked against the pinned
# SHA-256 sums below; tarballs already present there are reused, so the
# script also works offline once they have been copied in.
set -eu

P=${1:?usage: build_host_autotools.sh <prefix> [jobs]}
J=${2:-4}
: "${GNU_MIRROR:=https://ftp.gnu.org/gnu}"

case $P in /*) ;; *) P=$(pwd)/$P ;; esac
mkdir -p "$P/src"
PATH=$P/bin:$PATH
export PATH

build() {
  name=$1; ver=$2; sum=$3
  [ -f "$P/.$name-$ver.done" ] && return 0
  tarball=$P/src/$name-$ver.tar.xz
  [ -f "$tarball" ] || curl -L --fail "$GNU_MIRROR/$name/$name-$ver.tar.xz" -o "$tarball"
  echo "$sum  $tarball" | sha256sum -c -
  rm -rf "$P/src/$name-$ver"
  tar -xJf "$tarball" -C "$P/src"
  ( cd "$P/src/$name-$ver" && ./configure --prefix="$P" && make -j "$J" && make install ) \
    > "$P/src/$name-$ver.log" 2>&1 || { echo "host autotools: $name $ver FAILED (see $P/src/$name-$ver.log)"; exit 1; }
  rm -rf "$P/src/$name-$ver"
  touch "$P/.$name-$ver.done"
  echo "host autotools: installed $name $ver into $P"
}

# automake and libtool configure against the autoconf installed just before.
build autoconf 2.72   ba885c1319578d6c94d46e9b0dceb4014caafe2490e437a0dbca3f270a223f5a
build automake 1.16.5 f01d58cd6d9d77fbdca9eb4bbd5ead1988228fdb73d6f7a201f5f8d6b118b469
build libtool  2.4.7  4f7f217f057ce655ff22559ad221a0fd8ef84ad1fc5fcb6990cecc333aa1635d
