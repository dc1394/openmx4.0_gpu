# AMD container entrypoint fix

The 2026-11-12 `4.0-amd` startup fix adds an entrypoint layer to the pinned
published ROCm image. This recipe preserves its OpenMX binaries and runtime;
it does not rebuild OpenMX. The resulting published manifest is
`sha256:bfd91ae6213a3c06f2c39540be7486a029480b51684dd090a76da2ca0635fe84`.

The former `ln -sf` command followed an existing `openmx_work` directory link
on the second startup and tried to write inside the root-owned installation.
The corrected script keeps a correct link, replaces stale links using GNU
`ln -T`, and preserves real user files and directories. Keep this logic when
updating the entrypoint in a full AMD image build.

Build from the repository root:

```sh
docker build -f docker/amd/Dockerfile.entrypoint-fix \
  -t openmx-entrypoint-fixed docker/amd
```

Run the regression as a non-root host user with access to Docker. No GPU is
required. The script creates and removes private bind mounts and disposable
containers, checks 15 startup cases, and writes a JSON report:

```sh
python3 docker/tests/check_entrypoint.py \
  --image openmx-entrypoint-fixed \
  --workspace build/docker-entrypoint-check \
  --report result.json
```

Use `--basic-only` to check two consecutive non-root starts. To reproduce the
failure on the old pinned image, first pull the base digest from the Dockerfile
and run the test against it with `--expect-bug`.

The [publication record](../../benchmarks/results/amd_docker_entrypoint_fix_20261112.json)
contains the image digests, unchanged binary hashes and actual test results.
