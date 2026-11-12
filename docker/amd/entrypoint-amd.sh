#!/bin/sh
set -e
mkdir -p /work

# Bind mounts retain this link between container starts. Leave a correct link
# untouched, including on a read-only mount. Never follow a stale directory
# symlink when replacing it, and keep real files/directories owned by the user.
work_link=/work/openmx_work
work_target=/openmx4.0_gpu/work
if [ -L "$work_link" ]; then
    if [ "$(readlink "$work_link")" != "$work_target" ]; then
        ln -sfnT "$work_target" "$work_link"
    fi
elif [ ! -e "$work_link" ]; then
    ln -sT "$work_target" "$work_link"
fi

# Flat MPI runs can exceed Docker's default 8 MB stack limit.
ulimit -s unlimited 2>/dev/null || true
exec "$@"
