"""Archive every result, retaining failed full-FP64 accuracy checks explicitly."""
import csv,gzip,hashlib,json,shutil,subprocess,sys
from pathlib import Path
from run import A,ROOT,VARIANTS,sha,save
from analyze_control import CONTROL,FULL_CONTROL
def main():
    for script,args in [('analyze.py',[]),('verify_runtime.py',[]),
                        ('analyze_control.py',['--require-controlled-agreement']),
                        ('verify_control.py',[]),('verify_control.py',['--full-fp64'])]:
        subprocess.run([sys.executable,str(A/script)]+args,check=True)
    analysis=json.loads((A/'analysis.json').read_text())
    runtime=json.loads((A/'runtime-validation.json').read_text())
    controls=json.loads((A/'control-analysis.json').read_text())
    assert analysis['all_complete'] and runtime['passed'] and controls['all_converged']
    assert controls['all_controlled_comparisons_passed']
    for name in ['control-runtime-validation.json','full-control-runtime-validation.json']:
        assert json.loads((A/name).read_text())['passed']
    out=ROOT/'benchmarks/results/rtx5080_cghelgeo_strict_20261108'
    out.mkdir(exist_ok=False)
    for name in ['provenance.json','versions.json','lscpu.json','preflight.json','analysis.json',
                 'comparisons.csv','runtime-validation.json','original-cghelgeo.dat','audit-probe-results.json',
                 'audit-solver-probe-results.json','control-analysis.json','control-comparisons.csv',
                 'control-runtime-validation.json','full-control-runtime-validation.json']:
        shutil.copy2(A/name,out/name)
    # Archived capture/analysis code is machine-specific; binaries and data are
    # identified by hashes rather than redistributed here.
    scripts=out/'scripts';scripts.mkdir()
    for name in ['run.py','rank.py','analyze.py','verify_runtime.py','cusolver_audit.cpp','audit_probe.cpp','publish.py',
                 'audit_solver_probe.cpp',
                 'prepare_restart_campaign.py','run_legacy_control.py','analyze_control.py','verify_control.py']:
        shutil.copy2(A/name,scripts/name)
    preliminary=out/'preliminary';preliminary.mkdir()
    shutil.copy2(A/'preliminary/diagnostic.json',preliminary/'diagnostic.json')
    shutil.copy2(A/'preliminary/run.py',preliminary/'initial-run.py')
    source=A/'preliminary/upstream_criterion_1e-12'
    for name in ['input.dat','resources.json']:shutil.copy2(source/name,preliminary/name)
    for name in ['run.log','cghelgeo.out']:
        (preliminary/(name+'.gz')).write_bytes(gzip.compress((source/name).read_bytes(),mtime=0))
    summaries=[]
    for variant in VARIANTS+[CONTROL,FULL_CONTROL]:
        source=A/'runs'/variant;dest=out/variant;dest.mkdir()
        for name in ['input.dat','result.json','resources.json']:shutil.copy2(source/name,dest/name)
        for name in ['run.log','cghelgeo.out']:
            (dest/(name+'.gz')).write_bytes(gzip.compress((source/name).read_bytes(),mtime=0))
        ranks=[json.loads(p.read_text()) for p in sorted(source.glob('rank-??.json'))]
        threads={p.stem:json.loads(p.read_text()) for p in sorted(source.glob('rank-??-threads.json'))}
        save(dest/'rank-placement.json',dict(ranks=ranks,threads=threads))
        maps={}
        for p in sorted(source.glob('rank-??-maps.txt')):
            maps[p.stem]=sorted({line.split()[-1] for line in p.read_text().splitlines()
                if len(line.split())>5 and line.split()[-1].startswith('/') and '.so' in line.split()[-1]})
        save(dest/'runtime-libraries.json',maps)
        modes={p.name:p.read_text() for p in sorted(source.glob('rank-??-cusolver.jsonl'))}
        if modes:(dest/'cusolver-modes.json.gz').write_bytes(gzip.compress(json.dumps(modes,indent=2).encode(),mtime=0))
        r=json.loads((source/'result.json').read_text());summaries.append(dict(variant=variant,
          SCF_steps=r['scf_steps'],preparatory_SCF_steps=r.get('preparatory_SCF_steps',0),
          seed_SCF_steps=r.get('seed_SCF_steps',0),initialization=r['initialization'],
          energy_hartree=r['energy_hartree'],converged=r['converged']))
    mps=out/'MPS';mps.mkdir()
    for name in ['mps-start.json','mps-server-list.json','mps-stop.json']:shutil.copy2(A/name,mps/name)
    for variant in [CONTROL,FULL_CONTROL]:
        target=mps/variant;target.mkdir()
        for name in ['start','server','stop']:shutil.copy2(A/'control-mps'/variant/(name+'.json'),target/(name+'.json'))
    metadata=dict(source_commit=json.loads((A/'provenance.json').read_text())['source_commit'],
      input='benchmarks/inputs/cghelgeo.dat',MPI_processes=18,OpenMP_threads_per_process=2,
      BLAS_threads=2,criterion=1e-10,all_other_physical_input_parameters_preserved=True,
      MD_steps=1,SCF_iteration_limit_per_invocation=200,atom_count=650,force_components_compared=1950,
      original_campaign_runs=4,additional_matched_seed_controls=2,pairwise_comparisons=15,all_converged=True,
      original_four_accuracy_checks_passed=analysis['all_passed'],
      all_accuracy_checks_passed=controls['all_comparisons_passed'],
      controlled_legacy_product_comparisons=6,all_controlled_legacy_product_checks_passed=controls['all_controlled_comparisons_passed'],
      timings='Diagnostic accuracy runs with different initialization for reference/controls; not a performance comparison.',
      upstream='Existing Intel CPU executable from ~/openmx4.0/work/openmx, copied unchanged.',
      initialization='Upstream restarts from its same-geometry 200-SCF 1e-12 probe. Three original fork cases start fresh. Both added controls restart from exactly the same converged default-GPU files.',
      preliminary='Original CPU at 1e-12: normal exit after 200 SCF steps, not converged. Preserved in preliminary/.',
      fork='One CUDA Release executable in all five fork configurations; only snapshot RPATH order changed.',
      CPU='OPENMX_GPU=0; scf.eigen.lib elpa2; OPENMX_HOST_SIMD=avx512; other arithmetic defaults retained.',
      GPU_native='scf.fp32.enable off; scf.gemmul8.enable off; OPENMX_CUSOLVER_EMULATION=0.',
      GPU_native_legacy_products='scf.fp32.enable on; scf.gemmul8.enable off; OPENMX_CUSOLVER_EMULATION=0; OPENMX_EIGEN_REFINE=0; OPENMX_GRID_PRECISION=fp64; OPENMX_ORBS_EVAL_PRECISION=fp64. Retains inherited FP32 projector-product rounding.',
      matched_full_FP64_control='Same seed, executable and numerical environment as the legacy-product control; only scf.fp32.enable changes to off.',
      GPU_default='Default arithmetic controls unchanged: GEMMul8, FP32 refinement and double-float policies remain automatic; cuSOLVER emulation mode queried successfully.',
      GPU_MPS=True,cuSOLVER_audit='A forwarding LD_PRELOAD shim records successful math-mode changes/queries and actual Xsyevd/Xsyevdx calls, sizes and arithmetic types; it does not change arguments or results.',
      force_comparison='All components from the printed coordinates.forces block; decimal differences and rounding bounds.',
      energy_tolerance_hartree=1e-7,force_component_tolerance_hartree_per_bohr=1e-7,runs=summaries,
      raw_workspace=str(A))
    save(out/'manifest.json',metadata)
    lines=['Strict cghelgeo accuracy validation (RTX 5080, started 2026-11-08)',
      '', 'Six complete calculations at scf.criterion=1e-10, 18 MPI x 2 OpenMP: four requested modes and two matched-seed controls.',
      'Geometry, cell, PAO/VPS, k mesh, cutoff, mixing and the 200-SCF limit are preserved.',
      'The original MD.Type=Opt and MD.maxIter=1 are retained.',
      'The upstream case uses scf.restart on with density/Hamiltonian from its same-geometry probe.',
      'The three original fork cases start fresh. Both added controls use exactly the same converged default-GPU restart.',
      'All six final runs converge and complete normally. Initialization differences preclude performance comparisons.',
      'The original four-mode comparison is NOT all-pass at the unchanged 1e-7 Ha energy gate.',
      'Full-FP64 energy-gate failures are retained in analysis.json and comparisons.csv.',
      'All six comparisons with inherited projector-product rounding held fixed pass the same gates.',
      'All 15 pairwise comparisons, including failures, appear in control-comparisons.csv.',
      'The force check covers all 1950 components. Printed rounding bounds are included.',
      '', 'A preliminary upstream 1e-12 run exited normally after 200 SCF steps without convergence.',
      'Its final dUele was about 2.7e-10 Ha. The common comparison criterion was set to 1e-10,',
      '1000 times tighter than the original 1e-7 input. Energy/force comparison tolerances remain 1e-7.',
      'The failed convergence probe is preserved under preliminary/ and is not counted as converged.',
      '', 'The CPU path uses the same CUDA executable with GPU execution disabled and AVX-512 selected.',
      'Native FP64 disables FP32/DF arithmetic, GEMMul8 and cuSOLVER emulation.',
      'scf.fp32.enable off also promotes inherited FP32 projector products; this changes the numerical contract of the original code.',
      'The additional native-FP64 control preserves those products but forces FP64 dense solvers/GEMMs and grid/orbital evaluation.',
      'The matched full-FP64 control differs only in scf.fp32.enable, with identical restart-seed hashes and numerical environment.',
      'Existing float table storage is retained. The default GPU case keeps all default arithmetic policies.',
      'cuSOLVER API queries and actual Xsyevd/Xsyevdx calls verify native and emulated modes;',
      'runtime-validation.json records the matrix sizes and arithmetic types.',
      'A 3x3 Hermitian eigenproblem verifies both forwarded APIs and both modes independently.',
      'The forwarding audit shim and diagnostic logging are enabled for every GPU configuration.',
      'Every GPU run uses a private MPS service, stopped after completion.',
      '', 'See provenance.json for the measured source commit, data/library/binary hashes and RPATH snapshot change.',
      'See input.dat and result.json in each configuration directory for exact inputs, commands and environment.',
      'Full stdout and OpenMX output files are gzip-compressed; runtime-library and rank-placement records accompany them.',
      'Capture scripts require the original workstation paths, bin/lib/DFT_DATA19/runs layout and matching binaries/data.',
      'This compressed evidence archive is not a standalone runnable benchmark installation.',
      'Recorded times are diagnostic single runs and should not replace the README performance tables.',
      '', 'SCF steps and total energies:']
    lines.extend(f"{r['variant']}: {r['SCF_steps']} final SCF (+{r['preparatory_SCF_steps']} preparatory), {r['energy_hartree']:.12f} Ha; {r['initialization']}" for r in summaries)
    (out/'README.txt').write_text('\n'.join(lines)+'\n')
    files=sorted(p for p in out.rglob('*') if p.is_file())
    (out/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(out))+'\n' for p in files))
    print('ARCHIVED',out)
if __name__=='__main__':main()
