"""Compare all six outputs without hiding the full-FP64 energy-gate failures."""
import csv,itertools,json,re,sys
from pathlib import Path
from analyze import A,TOL,physical,exact_output,quantum
from run import VARIANTS,save,sha
CONTROL='gpu_native_legacy_products'
FULL_CONTROL='gpu_full_fp64_same_seed'
ALL=VARIANTS+[CONTROL,FULL_CONTROL]
def main():
    records={v:json.loads((A/'runs'/v/'result.json').read_text()) for v in ALL}
    original=(A/'original-cghelgeo.dat').read_text()
    for v,r in records.items():
        assert r['status']=='completed' and r['converged'] and r['returncode']==0
        assert r['criterion']==1e-10 and len(r['atoms'])==650 and len(r['forces_hartree_per_bohr'])==1950
        text=(A/'runs'/v/'input.dat').read_text()
        assert physical(text)==physical(original)
        assert re.search(r'^scf\.criterion\s+1e-10\s*$',text,re.M)
        assert sha(A/'runs'/v/'input.dat')==r['input_sha256']
    checks=[]
    for a,b in itertools.combinations(ALL,2):
        x,y=records[a],records[b]
        assert all(u[:2]==v[:2] for u,v in zip(x['atoms'],y['atoms'])) and x['FFT_grid']==y['FFT_grid']
        coord=max(abs(u-v) for ar,br in zip(x['atoms'],y['atoms']) for u,v in zip(ar[2:],br[2:]))
        ea,fa=exact_output(A/'runs'/a);eb,fb=exact_output(A/'runs'/b)
        de=abs(ea-eb);diffs=[abs(u-v) for u,v in zip(fa,fb)];df=max(diffs);imax=diffs.index(df)
        e_upper=de+(quantum(ea)+quantum(eb))/2
        f_upper=max(d+(quantum(u)+quantum(v))/2 for d,u,v in zip(diffs,fa,fb))
        epass=e_upper<=TOL;fpass=f_upper<=TOL
        checks.append(dict(baseline=a,candidate=b,energy_difference_hartree=float(de),
          max_force_component_difference_hartree_per_bohr=float(df),
          energy_difference_rounding_upper_bound=float(e_upper),force_difference_rounding_upper_bound=float(f_upper),
          max_force_atom=imax//3+1,max_force_axis='xyz'[imax%3],max_coordinate_difference=coord,
          energy_passed=epass,force_passed=fpass,passed=epass and fpass and coord<=1.000001e-5,
          inherited_projector_products_held_fixed=not ({a,b}&{'gpu_native_fp64',FULL_CONTROL})))
    controlled=[c for c in checks if c['inherited_projector_products_held_fixed']]
    full=[c for c in checks if not c['inherited_projector_products_held_fixed']]
    assert records[CONTROL]['restart_seed_files']==records[FULL_CONTROL]['restart_seed_files']
    control_input=(A/'runs'/CONTROL/'input.dat').read_text()
    full_input=(A/'runs'/FULL_CONTROL/'input.dat').read_text()
    assert re.sub(r'(?m)^scf\.fp32\.enable\s+on$', 'scf.fp32.enable off',control_input)==full_input
    for key,value in records[CONTROL]['environment'].items():
        if not key.startswith('CUDA_MPS_'):assert records[FULL_CONTROL]['environment'][key]==value,key
    data=dict(completed=6,criterion=1e-10,force_components_per_output=1950,pairwise_comparisons=len(checks),
      tolerance=dict(energy_hartree=float(TOL),force_component_hartree_per_bohr=float(TOL)),printed_rounding_included=True,
      all_converged=True,all_comparisons_passed=all(c['passed'] for c in checks),
      original_four_comparisons_passed=all(c['passed'] for c in checks if c['baseline'] in VARIANTS and c['candidate'] in VARIANTS),
      controlled_comparisons=6,all_controlled_comparisons_passed=len(controlled)==6 and all(c['passed'] for c in controlled),
      full_fp64_diagnostic_comparisons=len(full),full_fp64_energy_failures=sum(not c['energy_passed'] for c in full),
      matched_restart_seed_hashes_verified=True,matched_control_inputs_differ_only_in_scf_fp32_enable=True,
      matched_control_numerical_environments_identical=True,
      comparison=checks,runs=[dict(variant=v,SCF_steps=r['scf_steps'],energy_hartree=r['energy_hartree'],
        initialization=r['initialization'],preparatory_SCF_steps=r.get('preparatory_SCF_steps',0),
        seed_SCF_steps=r.get('seed_SCF_steps',0)) for v,r in records.items()])
    save(A/'control-analysis.json',data)
    with (A/'control-comparisons.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(checks[0]),lineterminator='\n');writer.writeheader();writer.writerows(checks)
    print(json.dumps(data,indent=2))
    if '--require-controlled-agreement' in sys.argv:assert data['all_controlled_comparisons_passed']
if __name__=='__main__':main()
