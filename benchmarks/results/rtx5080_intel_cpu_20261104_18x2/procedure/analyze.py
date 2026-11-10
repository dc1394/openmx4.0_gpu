import csv,json,math
from pathlib import Path
from runtime import A,save
def main():
    runs=[json.loads(p.read_text()) for p in sorted((A/'runs').glob('*/*/result.json')) if json.loads(p.read_text()).get('status')=='completed']
    rows=[];checks=[]
    for case in ['sidia333_nc_cluster','cghelgeo']:
        pair={r['variant']:r for r in runs if r['case']==case}
        if len(pair)!=2:continue
        a,b=pair['upstream'],pair['intel']
        assert a['input_sha256']==b['input_sha256']
        assert len(a['forces'])==len(b['forces'])==3*len(a['atoms'])==3*len(b['atoms'])
        ids=all(x[:2]==y[:2] for x,y in zip(a['atoms'],b['atoms']))
        coordinates=max(abs(u-v) for x,y in zip(a['atoms'],b['atoms']) for u,v in zip(x[2:],y[2:]))
        de=abs(a['energy_hartree']-b['energy_hartree']);df=max(abs(u-v) for u,v in zip(a['forces'],b['forces']))
        check=dict(case=case,energy_difference_hartree=de,max_force_component_difference_hartree_per_bohr=df,
                   atom_ids_match=ids,FFT_grids_match=a['grids']==b['grids'],max_coordinate_difference=coordinates,
                   passed=ids and a['grids']==b['grids'] and coordinates<=1.000001e-5 and de<=1e-7 and df<=1e-7)
        checks.append(check)
        rows.append(dict(case=case,upstream_seconds=a['total_seconds'],current_Intel_seconds=b['total_seconds'],
                         upstream_over_current=a['total_seconds']/b['total_seconds'],
                         upstream_wall_seconds=a['wall_seconds'],current_Intel_wall_seconds=b['wall_seconds'],
                         upstream_SCF_steps=a['scf_steps'],current_Intel_SCF_steps=b['scf_steps'],
                         upstream_unconverged_MD=a['scf_unconverged_md'],current_Intel_unconverged_MD=b['scf_unconverged_md'],
                         upstream_diagonalization_seconds=a['phases_seconds']['Diagonalization']['max'],
                         current_Intel_diagonalization_seconds=b['phases_seconds']['Diagonalization']['max'],
                         **{k:v for k,v in check.items() if k!='case'}))
    data=dict(completed_runs=len(runs),expected_runs=4,comparison=rows,accuracy_checks=checks,
              accuracy_failures=[r for r in checks if not r['passed']],
              protocol=dict(MPI_processes=18,OMP_threads_per_process=2,MKL_threads=2,repetitions=1,
                            all_inputs_and_convergence_limits_preserved=True,
                            primary_metric='Maximum-rank Total Computational Time, seconds',
                            wall_metric='MPI-job wall time including startup/shutdown and up to one two-second polling interval'))
    save(A/'analysis.json',data)
    if rows:
        with (A/'comparison.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print(json.dumps(data,indent=2))
if __name__=='__main__':main()
