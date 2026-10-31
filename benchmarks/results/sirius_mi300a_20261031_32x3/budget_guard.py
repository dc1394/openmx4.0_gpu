"""Enforce this campaign's official project-balance floor."""
import datetime
import json
from pathlib import Path
import subprocess

B = Path(__file__).resolve().parent


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def check_budget():
    marker = B / 'budget-stop.json'
    if marker.exists():
        raise RuntimeError('Benchmark stopped by budget policy; explicit resume required')
    policy = json.loads((B / 'budget-policy.json').read_text())
    raw = subprocess.check_output(['rbudgetcheck'], text=True, timeout=20)
    row = next(line.split() for line in raw.splitlines()
               if line.split() and line.split()[0] == policy['project'])
    result = dict(timestamp=datetime.datetime.now().astimezone().isoformat(),
                  project=policy['project'], remaining=float(row[1]),
                  estimate=float(row[2]), initial=float(row[3]), raw=raw)
    save(B / 'budget-latest.json', result)
    if result['remaining'] >= policy['threshold']:
        return result
    result.update(threshold=policy['threshold'], triggered=True,
                  automatic_restart_allowed=False, requested_cancellation=[])
    save(marker, result)
    ids = policy.get('active_job_ids', [])
    if ids:
        report = json.loads(subprocess.check_output(
            ['qstat', '-xf', '-F', 'json', *ids], text=True, timeout=20))
        for jid, job in report['Jobs'].items():
            if job['Job_Owner'].split('@')[0] != 'dc1394' or job['Account_Name'] != policy['project']:
                raise RuntimeError('Unexpected job ownership in budget guard')
            if job['job_state'] not in ('F', 'X'):
                result['requested_cancellation'].append(jid)
        save(marker, result)
        if result['requested_cancellation']:
            proc = subprocess.run(['qdel', *result['requested_cancellation']],
                                  capture_output=True, text=True, timeout=30)
            result.update(qdel_returncode=proc.returncode, qdel_stdout=proc.stdout,
                          qdel_stderr=proc.stderr)
            save(marker, result)
    raise RuntimeError('Official remaining project points fell below the configured floor')


def check_cached_budget():
    if (B/'budget-stop.json').exists():
        raise RuntimeError('Benchmark stopped by project-budget policy')
    policy=json.loads((B/'budget-policy.json').read_text())
    result=json.loads((B/'budget-latest.json').read_text())
    checked=datetime.datetime.fromisoformat(result['timestamp'])
    age=(datetime.datetime.now(datetime.timezone.utc)-checked).total_seconds()
    if age>120 or result['remaining']<policy['threshold']:
        raise RuntimeError('Fresh login-node budget confirmation is required')
    return result


if __name__ == '__main__':
    print(json.dumps(check_budget(), ensure_ascii=False))
