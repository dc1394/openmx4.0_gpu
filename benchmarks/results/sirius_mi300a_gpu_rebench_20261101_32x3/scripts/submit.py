#!/usr/bin/env python3
"""Submit this campaign within its independent, reserved 100-point allowance."""
import datetime
import fcntl
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from budget_guard import check_budget, save

B = Path(__file__).resolve().parent


def seconds(value):
    h, m, s = map(int, value.split(':'))
    return h * 3600 + m * 60 + s


def submit(script):
    text = script.read_text()
    wall = re.search(r'^#PBS -l walltime=(\d+:\d+:\d+)$', text, re.M)
    selection = re.search(r'^#PBS -l select=(.+)$', text, re.M)
    assert wall and selection and '#PBS -A GPU2026' in text
    chunks = selection[1].split('+')
    gpus = sum(int(x.split(':')[0]) * int(re.search(r'(?:^|:)ngpus=(\d+)', x)[1]) for x in chunks)
    assert gpus in (4, 8), gpus
    reserved = math.ceil(seconds(wall[1]) * gpus / 3600 * 100) / 100
    policy = json.loads((B / 'budget-policy.json').read_text())
    allocations = json.loads((B / 'allocations.json').read_text())
    cost = 0
    for row in allocations:
        job = json.loads(subprocess.check_output(
            ['qstat', '-xf', '-F', 'json', row['jobid']], text=True))['Jobs'][row['jobid']]
        if job['job_state'] in ('F', 'X'):
            row.update(state=job['job_state'], exit_status=job.get('Exit_status'),
                       actual_wall_seconds=seconds(job.get('resources_used', {}).get('walltime', '00:00:00')))
            row['actual_points'] = math.ceil(row['actual_wall_seconds'] * row['gpus'] / 3600 * 100) / 100
            cost += row['actual_points']
        else:
            cost += row['reserved_points']
    assert cost + reserved <= policy['spending_limit_points'], (cost, reserved)
    budget = check_budget()
    assert budget['remaining'] - reserved >= policy['threshold'], budget
    jobid = subprocess.check_output(['qsub', '-h', str(script)], text=True).strip()
    assert re.fullmatch(r'\d+\.pbs', jobid), jobid
    allocations.append(dict(jobid=jobid, script=str(script), gpus=gpus,
                            reserved_points=reserved, previous_actual_or_reserved=cost,
                            envelope_points_at_submission=cost+reserved,
                            submitted=datetime.datetime.now().astimezone().isoformat(), state='H'))
    save(B / 'allocations.json', allocations)
    policy['active_job_ids'] = [r['jobid'] for r in allocations if r['state'] not in ('F', 'X')]
    save(B / 'budget-policy.json', policy)
    # Confirm the site's actual accounting coefficient before allowing execution.
    job = json.loads(subprocess.check_output(
        ['qstat', '-xf', '-F', 'json', jobid], text=True))['Jobs'][jobid]
    cache = json.loads(job['Resource_List'].get('am_job_cache', '{}'))
    if cache and (cache['queue.resources_default[coef]'] != 1 or cache['job.ngpus'] != gpus):
        subprocess.run(['qdel', jobid], check=True)
        raise RuntimeError('Unexpected site accounting coefficient or GPU count')
    check_budget()
    with (B / (jobid + '.budget-watch.log')).open('w') as log:
        monitor = subprocess.Popen([sys.executable, '-u', str(B / 'budget_watch.py')], cwd=B,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    subprocess.run(['qrls', jobid], check=True)
    print(json.dumps(dict(jobid=jobid, gpus=gpus, reserved_points=reserved,
                          previous_actual_or_reserved=cost, monitor_pid=monitor.pid)))


if __name__ == '__main__':
    with (B / 'submit.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        submit(Path(sys.argv[1]).resolve())
