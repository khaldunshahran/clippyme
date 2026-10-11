"""PROMOTED coordinator probe (Task 19 review01).

Original: work/muse-handoff/task-19/coordinator-review01.py.
Production-call contracts for undo provenance, Hyperframes
normalization/retry, ledger serialization and strict timing.
Runs against the worktree's tests/domain/test_bulk_edit.py fixtures.
"""
"""Production-call Task19 contracts; local projects and injected failures only."""
import runpy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
from clippyme.domain import bulk_edit as b
from clippyme.domain.clip_project import project_from_json

ROOT = Path(__file__).resolve().parents[2]  # the worktree root
F = runpy.run_path(str(ROOT / 'tests/domain/test_bulk_edit.py'))

def seed(tmp_path):
    return F['_seed_job'](tmp_path, {0:F['_project_dict'](0),1:F['_project_dict'](1)})

def prepare(j, key='review', op='caption_appearance', params=None):
    return b.create_operation(job_dir=j, job_id='job-1',
        scope={'mode':'selected','clip_indices':[0]},
        intent={'op':op,'params':params or {'caption_style':'hormozi_bold'}},
        idempotency_key=key)

def test_recovered_apply_keeps_base_for_undo(tmp_path, monkeypatch):
    j=seed(tmp_path); op=prepare(j); oid=op['operation_id']
    original=b._record_item; failed=False
    def interrupt(jd, oi, ci, update):
        nonlocal failed
        if update.get('outcome')=='applied' and not failed:
            failed=True
            raise OSError('injected lost ledger acknowledgment')
        return original(jd,oi,ci,update)
    with monkeypatch.context() as m:
        m.setattr(b,'_record_item',interrupt)
        b.apply_operation(job_dir=j,operation_id=oid)
    recovered=b.retry_operation(job_dir=j,operation_id=oid)
    assert recovered['items']['0']['outcome']=='applied'
    undone=b.undo_operation(job_dir=j,operation_id=oid)
    assert undone['undo']['items']['0']['outcome']=='undone', undone['undo']

def test_undo_recovers_save_before_ledger_write(tmp_path, monkeypatch):
    j=seed(tmp_path); oid=prepare(j)['operation_id']
    b.apply_operation(job_dir=j,operation_id=oid)
    with monkeypatch.context() as m:
        m.setattr(b,'_save_undo_state',lambda *a,**k:(_ for _ in ()).throw(OSError('lost undo acknowledgment')))
        with pytest.raises(OSError): b.undo_operation(job_dir=j,operation_id=oid)
    result=b.undo_operation(job_dir=j,operation_id=oid)
    assert result['undo']['items']['0']['outcome']=='undone',result['undo']

def test_hyperframes_prepare_uses_valid_normalized_contract(tmp_path,monkeypatch):
    j=seed(tmp_path)
    monkeypatch.setattr(b,'_check_hyperframes_worker',lambda:{'available':True})
    result=prepare(j,op='hyperframes_finishing',params={'template':'clean-captions'})
    assert result['items']['0']['patch'].get('finishing')

def test_pending_worker_retry_builds_actual_finishing_patch(tmp_path,monkeypatch):
    j=seed(tmp_path); op=prepare(j); oid=op['operation_id']
    # Persist the documented pending ledger representation directly.
    ledger=b.load_bulk_ledger(j); record=ledger['operations'][oid]
    record['intent']={'op':'hyperframes_finishing','params':{'template':'clean-captions','caption_size':'medium','caption_position':'lower-third'},'on_override':'replace'}
    record['items']['0'].update(outcome='pending',patch={},changes=[],excluded={'reason':'worker down'})
    b.save_bulk_ledger(j,ledger)
    monkeypatch.setattr(b,'_check_hyperframes_worker',lambda:{'available':True})
    result=b.retry_operation(job_dir=j,operation_id=oid)
    assert result['items']['0']['outcome']=='applied'
    project=project_from_json(Path(b.get_project_path(j,0)).read_text())
    assert project.finishing and project.finishing.renderer=='hyperframes'

@pytest.mark.parametrize('same_key',[False,True])
def test_concurrent_prepare_preserves_ledger_and_idempotency(tmp_path,monkeypatch,same_key):
    j=seed(tmp_path); barrier=threading.Barrier(2); real=b.load_bulk_ledger
    local=threading.local()
    def synchronized_load(jd):
        result=real(jd)
        if not getattr(local,'seen',False):
            local.seen=True; time.sleep(0.05)
        return result
    monkeypatch.setattr(b,'load_bulk_ledger',synchronized_load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        def run(i):
            barrier.wait(timeout=5)
            return prepare(j,'same' if same_key else f'op-{i}')
        futures=[pool.submit(run,i) for i in range(2)]
        results=[]; errors=[]
        for f in futures:
            try: results.append(f.result(timeout=10))
            except Exception as e: errors.append(repr(e))
    assert not errors, errors
    ids={x['operation_id'] for x in results}
    assert len(ids)==(1 if same_key else 2)
    assert ids==set(real(j)['operations'])

def test_nonfinite_hook_intent_rejected():
    with pytest.raises(ValueError):
        b.validate_bulk_intent({'op':'hook_timing','params':{'start':0,'end':float('inf')}})
