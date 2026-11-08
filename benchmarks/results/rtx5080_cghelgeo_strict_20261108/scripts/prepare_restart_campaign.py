"""Preserve the unconverged 1e-12 probe and prepare a common 1e-10 comparison."""
import hashlib,importlib.util,json,re,shutil,sys,time
from pathlib import Path
A=Path(__file__).resolve().parent
ROOT=Path('/home/dc1394/openmx4.0_gpu')
sys.path.insert(0,str(ROOT/'tools'))
from compare_gpu_suites import read_output
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((A/'failure.json').read_text())=={'error':'upstream','completed':0}
assert not (A/'results.json').exists()
old=A/'runs/upstream';log=(old/'run.log').read_text()
assert log.count('The calculation was normally finished.')==1
assert log.count('The SCF was not fully achieved at MD=')==1
assert len(re.findall(r'MD=\s*1\s+SCF=\s*\d+',log))==200
assert not (A/'preliminary').exists()
pre=A/'preliminary';pre.mkdir()
old.rename(pre/'upstream_criterion_1e-12')
old=pre/'upstream_criterion_1e-12'
energy,atoms,forces,grid=read_output(old/'cghelgeo.out')
seed=old/'cghelgeo_rst'
seed_files=[dict(path=str(p.relative_to(seed)),sha256=sha(p),bytes=p.stat().st_size)
            for p in sorted(seed.rglob('*')) if p.is_file()]
assert len(atoms)==650 and len(forces)==1950 and len(seed_files)>650
summary=dict(status='normal_exit_unconverged',criterion=1e-12,SCF_steps=200,
  normal_finish=True,converged=False,energy_hartree=energy,
  final_Uele=re.findall(r'^<DFT>.*Uele.*dUele.*$',log,re.M)[-1],
  final_NormRD=re.findall(r'^<DFT>.*NormRD.*$',log,re.M)[-1],
  reason='The original input limit of 200 SCF steps was reached without convergence at 1e-12.',
  subsequent_criterion=1e-10,
  subsequent_initialization='Same geometry and binary; reuse this calculation\'s saved density/Hamiltonian with scf.restart on.',
  input_sha256=sha(old/'input.dat'),output_sha256=sha(old/'cghelgeo.out'),restart_files=seed_files)
save(pre/'diagnostic.json',summary)
for name in ['progress.json','failure.json','active-process.json','preflight.json','driver.log']:
    if (A/name).exists():shutil.move(A/name,pre/name)
for name in ['run.py','analyze.py','verify_runtime.py','publish.py','provenance.json']:
    shutil.copy2(A/name,pre/name)
shutil.move(A/'planned-inputs',pre/'planned-inputs')
def replace(name,old,new):
    p=A/name;s=p.read_text();assert old in s,(name,old);p.write_text(s.replace(old,new))
replace('run.py','import ctypes,hashlib,itertools,json,os,re,signal,subprocess,sys,time',
        'import ctypes,hashlib,itertools,json,os,re,shutil,signal,subprocess,sys,time')
replace('run.py',"STOP=False;MPS=False", "CRITERION='1e-10'\nSTOP=False;MPS=False")
replace('run.py',"'scf.criterion':'1e-12'", "'scf.criterion':CRITERION")
replace('run.py',"    if variant in ['upstream','cpu_avx512']:changes['scf.eigen.lib']='elpa2'",
        "    if variant in ['upstream','cpu_avx512']:changes['scf.eigen.lib']='elpa2'\n    if variant=='upstream':changes['scf.restart']='on'")
replace('run.py',"    folder=A/'runs'/variant;folder.mkdir()",
        "    folder=A/'runs'/variant;folder.mkdir()\n    if variant=='upstream':\n        seed=A/'preliminary/upstream_criterion_1e-12/cghelgeo_rst'\n        shutil.copytree(seed,folder/'cghelgeo_rst')")
replace('run.py',"start_time=time.strftime('%Y-%m-%d %H:%M:%S %z'),status='running')",
        "start_time=time.strftime('%Y-%m-%d %H:%M:%S %z'),status='running',criterion=float(CRITERION),\n      initialization=('same-geometry 200-SCF 1e-12 probe restart' if variant=='upstream' else 'fresh; no restart'),\n      preparatory_SCF_steps=(200 if variant=='upstream' else 0))")
replace('run.py','selected_name=name,criterion=1e-12','selected_name=name,criterion=float(CRITERION)')
replace('run.py','Four complete fresh cghelgeo calculations; strict convergence, no restarts.',
        'Four complete cghelgeo calculations at a common 1e-10 criterion.\nThe original CPU case is initialized from its preserved 200-SCF 1e-12 probe;\nthe three fork cases start fresh.')
replace('analyze.py',"'scf.gemmul8.enable','level.of.stdout'", "'scf.gemmul8.enable','level.of.stdout','scf.restart'")
replace('analyze.py',r"scf\.criterion\s+1e-12\s*$",r"scf\.criterion\s+1e-10\s*$")
d=json.loads((A/'provenance.json').read_text());d['criterion']=1e-10
d['initial_probe']='preliminary/diagnostic.json: original CPU, 1e-12, 200 steps, normal exit but unconverged.'
d['initialization']='Upstream reuses the same-geometry probe density; the three fork modes start fresh.'
save(A/'provenance.json',d)
active=ROOT/'build/active-cghelgeo-precision.json'
d=json.loads(active.read_text());d['criterion']=1e-10;save(active,d)
(A/'planned-inputs').mkdir()
spec=importlib.util.spec_from_file_location('campaign',A/'run.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
for v in module.VARIANTS:(A/'planned-inputs'/(v+'.dat')).write_text(module.input_text(v))
print(json.dumps(dict(preliminary=summary['status'],restart_files=len(seed_files),criterion=1e-10)))
