"""Keep budget enforcement independent of numerical test progress."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import time
from budget_guard import check_budget

B=Path(__file__).resolve().parent
(B/'budget-watch.pid').write_text(str(os.getpid())+'\n')
while True:
    try:
        budget=check_budget()
        policy=json.loads((B/'budget-policy.json').read_text())
        ids=policy['active_job_ids']
        report=json.loads(subprocess.check_output(['qstat','-xf','-F','json',*ids],text=True,timeout=20))
        active=[j for j,x in report['Jobs'].items() if x['job_state'] not in ('F','X')]
        print(datetime.datetime.now().astimezone().isoformat(),budget['remaining'],active,flush=True)
        if not active:
            break
    except Exception as error:
        print('Budget monitor:',error,flush=True)
        if (B/'budget-stop.json').exists():
            raise SystemExit(3)
    time.sleep(15)
