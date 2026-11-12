"""Check repeated container starts in one bind-mounted working directory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import uuid

TARGET = '/openmx4.0_gpu/work'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--expect-bug', action='store_true')
    parser.add_argument('--basic-only', action='store_true')
    args = parser.parse_args()
    A = args.workspace.resolve()
    A.mkdir(parents=True, exist_ok=True)
    uid, gid = A.stat().st_uid, A.stat().st_gid
    assert uid != 0, 'The repeated-start regression must use a non-root UID.'
    records = []

    def start(folder, name, user=None, readonly_mount=False, readonly_root=False, body=None, expected=0):
        user = user or f'{uid}:{gid}'
        container = 'openmx-entrypoint-check-' + uuid.uuid4().hex[:16]
        command = ['docker', 'run', '--rm', '--pull=never', '--network=none',
                   '--name', container, '--user', user, '--shm-size=4gb',
                   '-e', 'OPENMX_MPS=0', '-v', str(folder) + ':/work' + (':ro' if readonly_mount else '')]
        if readonly_root:
            command.append('--read-only')
        if body is None:
            body = (f'test "$(id -u)" = "{user.split(":")[0]}" && '
                    'test -L /work/openmx_work && '
                    f'test "$(readlink /work/openmx_work)" = "{TARGET}" && '
                    'test ! -L /openmx4.0_gpu/work/work && '
                    'printf "entrypoint-command-reached\\n"')
        command += [args.image, '/bin/sh', '-c', body]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            # Only the uniquely named disposable container created by this call.
            subprocess.run(['docker', 'rm', '-f', container], capture_output=True, timeout=20)
            raise
        record = dict(case=name, command=command, returncode=result.returncode,
                      expected_returncode=expected, stdout=result.stdout, stderr=result.stderr)
        records.append(record)
        (A / args.report).write_text(json.dumps(dict(image=args.image, cases=records), indent=2) + '\n')
        if expected is not None:
            assert result.returncode == expected, record
        return result

    with tempfile.TemporaryDirectory(prefix='entrypoint-mounts-', dir=A) as temporary:
        base = Path(temporary)
        repeated = base / 'repeated'
        repeated.mkdir()
        start(repeated, 'nonroot-first-start')
        link = repeated / 'openmx_work'
        before = link.lstat()
        second = start(repeated, 'nonroot-second-start', expected=None if args.expect_bug else 0)
        if args.expect_bug:
            assert second.returncode != 0 and 'entrypoint-command-reached' not in second.stdout
            assert 'Permission denied' in second.stderr or 'Read-only file system' in second.stderr, second.stderr
            status = dict(image=args.image, bug_reproduced=True, nonroot_UID=uid, cases=records)
            (A / args.report).write_text(json.dumps(status, indent=2) + '\n')
            print('REPRODUCED: first non-root startup passes; second startup fails following openmx_work.')
            return
        assert before.st_ino == link.lstat().st_ino and os.readlink(link) == TARGET
        if not args.basic_only:
            start(repeated, 'nonroot-third-start')
            assert before.st_ino == link.lstat().st_ino
            start(repeated, 'correct-link-readonly-mount', readonly_mount=True)
            start(repeated, 'correct-link-readonly-root-and-mount', readonly_mount=True, readonly_root=True)
            assert before.st_ino == link.lstat().st_ino
            start(repeated, 'command-exit-status-preserved', body='exit 37', expected=37)

            mixed = base / 'root-then-user'
            mixed.mkdir()
            start(mixed, 'root-first-start', user='0:0')
            root_link = mixed / 'openmx_work'
            assert root_link.lstat().st_uid == 0
            root_inode = root_link.lstat().st_ino
            start(mixed, 'nonroot-after-root')
            assert root_link.lstat().st_ino == root_inode

            for label, destination in [('stale-link', '/tmp/previous-install/work'),
                                       ('dangling-link', '/definitely-missing-openmx-work')]:
                folder = base / label
                folder.mkdir()
                (folder / 'openmx_work').symlink_to(destination)
                start(folder, label)
                assert os.readlink(folder / 'openmx_work') == TARGET
                start(folder, label + '-restart')

            folder = base / 'link-to-directory'
            folder.mkdir()
            other = folder / 'user-data'
            other.mkdir()
            (other / 'sentinel').write_text('preserved\n')
            (folder / 'openmx_work').symlink_to('/work/user-data')
            start(folder, 'stale-link-to-user-directory')
            assert (other / 'sentinel').read_text() == 'preserved\n'
            assert not (other / 'work').exists()

            folder = base / 'existing-directory'
            (folder / 'openmx_work').mkdir(parents=True)
            sentinel = folder / 'openmx_work/sentinel'
            sentinel.write_text('preserved\n')
            start(folder, 'existing-user-directory-preserved',
                  body='test ! -L /work/openmx_work && test -f /work/openmx_work/sentinel && test ! -e /work/openmx_work/work')
            assert sentinel.read_text() == 'preserved\n'

            folder = base / 'existing-file'
            folder.mkdir()
            existing = folder / 'openmx_work'
            existing.write_text('preserved\n')
            start(folder, 'existing-user-file-preserved', body='test ! -L /work/openmx_work && test -f /work/openmx_work')
            assert existing.read_text() == 'preserved\n'
        status = dict(image=args.image, passed=True, nonroot_UID=uid, nonroot_GID=gid,
                      correct_symlink_inode_preserved=True, cases=records)
        (A / args.report).write_text(json.dumps(status, indent=2) + '\n')
        print(f'PASS: {len(records)} container starts; non-root restarts and link state verified.')


if __name__ == '__main__':
    main()
