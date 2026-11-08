import csv,itertools,json,re,sys
from decimal import Decimal
from pathlib import Path
A=Path(__file__).resolve().parent
sys.path.insert(0,str(A))
from run import VARIANTS,save,sha,ROOT
TOL=Decimal('1e-7')
def physical(text):
    excluded={'data.path','scf.criterion','scf.eigen.lib','scf.fp32.enable','scf.gemmul8.enable','level.of.stdout','scf.restart'}
    return [line.strip() for line in text.splitlines() if line.split() and line.split()[0].lower() not in excluded]
def exact_output(folder):
    s=(folder/'cghelgeo.out').read_text()
    e=Decimal(re.search(r'^\s*Utot\.\s+(\S+)',s,re.M)[1])
    m=re.search(r'<coordinates.forces\s*\n\s*(\d+)\s*\n',s)
    forces=[Decimal(v) for line in s[m.end():].splitlines()[:int(m[1])] for v in line.split()[5:8]]
    return e,forces
def quantum(x):return Decimal(1).scaleb(x.as_tuple().exponent)
def main():
    results={v:json.loads((A/'runs'/v/'result.json').read_text()) for v in VARIANTS
             if (A/'runs'/v/'result.json').exists() and json.loads((A/'runs'/v/'result.json').read_text()).get('status')=='completed'}
    original=(A/'original-cghelgeo.dat').read_text()
    for v,r in results.items():
        assert r['converged'] and len(r['atoms'])==650 and len(r['forces_hartree_per_bohr'])==1950
        assert r['criterion']==1e-10,v
        text=(A/'runs'/v/'input.dat').read_text()
        assert physical(text)==physical(original),v
        assert re.search(r'^scf\.criterion\s+1e-10\s*$',text,re.M),v
        assert sha(A/'runs'/v/'input.dat')==r['input_sha256']
    checks=[]
    for a,b in itertools.combinations(results,2):
        x,y=results[a],results[b]
        assert all(u[:2]==v[:2] for u,v in zip(x['atoms'],y['atoms']))
        assert x['FFT_grid']==y['FFT_grid']
        coordinate=max(abs(u-v) for ar,br in zip(x['atoms'],y['atoms']) for u,v in zip(ar[2:],br[2:]))
        ea,fa=exact_output(A/'runs'/a);eb,fb=exact_output(A/'runs'/b)
        de=abs(ea-eb);diffs=[abs(u-v) for u,v in zip(fa,fb)]
        df=max(diffs);imax=diffs.index(df)
        e_upper=de+(quantum(ea)+quantum(eb))/2
        f_upper=max(d+(quantum(u)+quantum(v))/2 for d,u,v in zip(diffs,fa,fb))
        checks.append(dict(baseline=a,candidate=b,energy_difference_hartree=float(de),
          max_force_component_difference_hartree_per_bohr=float(df),
          energy_difference_rounding_upper_bound=float(e_upper),force_difference_rounding_upper_bound=float(f_upper),
          max_force_atom=imax//3+1,max_force_axis='xyz'[imax%3],max_coordinate_difference=coordinate,
          passed=e_upper<=TOL and f_upper<=TOL and coordinate<=1.000001e-5))
    data=dict(completed=len(results),expected=4,comparison=checks,
      tolerance=dict(energy_hartree=1e-7,force_component_hartree_per_bohr=1e-7),
      printed_rounding_included=True,all_complete=len(results)==4,
      all_passed=len(results)==4 and len(checks)==6 and all(x['passed'] for x in checks),
      criterion=1e-10,
      runs=[dict(variant=v,energy_hartree=r['energy_hartree'],SCF_steps=r['scf_steps'],
                 preparatory_SCF_steps=r.get('preparatory_SCF_steps',0),
                 total_seconds=r['total_seconds'],peak_RSS_gib=r['peak_RSS_gib']) for v,r in results.items()])
    save(A/'analysis.json',data)
    if checks:
        with (A/'comparisons.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(checks[0]),lineterminator='\n');w.writeheader();w.writerows(checks)
    print(json.dumps(data,indent=2))
    if '--complete' in sys.argv:assert data['all_passed'],data
if __name__=='__main__':main()
