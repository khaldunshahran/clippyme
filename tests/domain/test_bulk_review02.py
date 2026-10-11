"""Task19 remaining shared-ledger transaction integration gate."""
import runpy
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from clippyme.domain import bulk_edit as b

ROOT=Path(__file__).resolve().parents[2]  # the task19 worktree itself
F=runpy.run_path(str(ROOT/'tests/domain/test_bulk_edit.py'))

def test_apply_initial_status_write_cannot_erase_concurrent_proposal(tmp_path,monkeypatch):
    j=F['_seed_job'](tmp_path,{0:F['_project_dict'](0),1:F['_project_dict'](1)})
    def prepare(key,ci):
        return b.create_operation(job_dir=j,job_id='job-1',
            scope={'mode':'selected','clip_indices':[ci]},
            intent={'op':'caption_appearance','params':{'caption_style':'hormozi_bold'}},
            idempotency_key=key)
    first=prepare('first',0); oid=first['operation_id']
    reached=threading.Event(); release=threading.Event(); real=b.save_bulk_ledger
    blocked=False
    def stage(jd,ledger):
        nonlocal blocked
        if not blocked and ledger['operations'].get(oid,{}).get('status')=='applying':
            blocked=True;reached.set()
            if not release.wait(5): raise RuntimeError('review barrier timed out')
        return real(jd,ledger)
    monkeypatch.setattr(b,'save_bulk_ledger',stage)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(b.apply_operation,job_dir=j,operation_id=oid)
        assert reached.wait(5)
        c=pool.submit(prepare,'second',1)
        # A fixed shared transaction lock can block preparation until the
        # first transaction commits. Let it finish after the barrier.
        try:
            second=c.result(timeout=0.3)
        except TimeoutError:
            second=None
        finally: release.set()
        a.result(timeout=5)
        if second is None: second=c.result(timeout=5)
    assert {oid,second['operation_id']} <= set(b.load_bulk_ledger(j)['operations'])
def _prepare_op(job_dir, key, clip_index):
    return b.create_operation(
        job_dir=job_dir, job_id="job-1",
        scope={"mode": "selected", "clip_indices": [clip_index]},
        intent={"op": "caption_appearance",
                "params": {"caption_style": "hormozi_bold"}},
        idempotency_key=key)


"""Review02 durable controls: concurrent finalization of apply/retry/undo
must not erase a concurrently prepared operation, and concurrent applies
of different operations keep every record."""
import threading
from concurrent.futures import ThreadPoolExecutor


def _barrier_run(monkeypatch, job_dir, oid, first_fn, prepare_second,
                 block_predicate):
    """Run first_fn in a thread while its save_bulk_ledger call matching
    block_predicate is paused; prepare a second operation mid-pause; then
    release and assert both operations survive in the ledger."""
    real = b.save_bulk_ledger
    reached = threading.Event()
    release = threading.Event()
    blocked = False

    def stage(jd, ledger):
        nonlocal blocked
        op = ledger["operations"].get(oid)
        if op is not None and not blocked and block_predicate(op):
            blocked = True
            reached.set()
            if not release.wait(10):
                raise RuntimeError("review barrier timed out")
        return real(jd, ledger)

    monkeypatch.setattr(b, "save_bulk_ledger", stage)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(first_fn)
        assert reached.wait(10), "barrier was never reached"
        c = pool.submit(prepare_second)
        # A correct job transaction lock may block preparation until the
        # first transaction commits; that is the accepted serialization.
        try:
            second = c.result(timeout=0.3)
        except TimeoutError:
            second = None
        finally:
            release.set()
        a.result(timeout=15)
        if second is None:
            second = c.result(timeout=10)
    ledger = b.load_bulk_ledger(job_dir)
    assert {oid, second["operation_id"]} <= set(ledger["operations"])
    return ledger


def _prepare2(job_dir):
    return _prepare_op(j, "second", 1)


def test_apply_final_status_write_cannot_erase_concurrent_proposal(
        tmp_path, monkeypatch):
    j = F["_seed_job"](tmp_path, {0: F["_project_dict"](0),
                                  1: F["_project_dict"](1)})
    first = _prepare_op(j, "first", 0)
    oid = first["operation_id"]
    ledger = _barrier_run(
        monkeypatch, j, oid,
        lambda: b.apply_operation(job_dir=j, operation_id=oid),
        lambda: _prepare_op(j, "second", 1),
        # The finalized status write (initial/per-item writes stay
        # 'applying').
        lambda op: op.get("status") in ("applied", "partial", "failed"),
    )
    assert ledger["operations"][oid]["items"]["0"]["outcome"] == "applied"


def test_retry_reset_write_cannot_erase_concurrent_proposal(
        tmp_path, monkeypatch):
    j = F["_seed_job"](tmp_path, {0: F["_project_dict"](0),
                                  1: F["_project_dict"](1)})
    first = _prepare_op(j, "first", 0)
    oid = first["operation_id"]
    # Make the first apply fail per-item so the item is retry-eligible.
    real_save = b.save_project_version

    def boom(**kw):
        raise RuntimeError("injected save failure")

    monkeypatch.setattr(b, "save_project_version", boom)
    op = b.apply_operation(job_dir=j, operation_id=oid)
    assert op["items"]["0"]["outcome"] == "failed"
    monkeypatch.setattr(b, "save_project_version", real_save)
    ledger = _barrier_run(
        monkeypatch, j, oid,
        lambda: b.retry_operation(job_dir=j, operation_id=oid),
        lambda: _prepare_op(j, "second", 1),
        # Retry's reset transaction is the first 'applying' write; the
        # follow-on apply pass writes 'applying' again but the barrier
        # fires only once.
        lambda op: op.get("status") == "applying",
    )
    # The retry genuinely completed after the barrier lifted.
    assert ledger["operations"][oid]["items"]["0"]["outcome"] == "applied"


def test_undo_finalize_write_cannot_erase_concurrent_proposal(
        tmp_path, monkeypatch):
    j = F["_seed_job"](tmp_path, {0: F["_project_dict"](0),
                                  1: F["_project_dict"](1)})
    first = _prepare_op(j, "first", 0)
    oid = first["operation_id"]
    op = b.apply_operation(job_dir=j, operation_id=oid)
    assert op["items"]["0"]["outcome"] == "applied"
    ledger = _barrier_run(
        monkeypatch, j, oid,
        lambda: b.undo_operation(job_dir=j, operation_id=oid),
        lambda: _prepare_op(j, "second", 1),
        # Undo's finalization write (init/per-item writes stay 'pending').
        lambda op: (op.get("undo") or {}).get("status") in (
            "undone", "partially_undone", "undo_failed"),
    )
    assert ledger["operations"][oid]["undo"]["status"] == "undone"


def test_concurrent_applies_of_different_operations(tmp_path, monkeypatch):
    j = F["_seed_job"](tmp_path, {0: F["_project_dict"](0),
                                  1: F["_project_dict"](1)})
    op_a = _prepare_op(j, "keyA", 0)
    op_b = _prepare_op(j, "keyB", 1)
    # Widen the interleave window so a lost-update race would manifest.
    real = b.save_bulk_ledger
    import time as _time

    def slow(jd, ledger):
        _time.sleep(0.01)
        return real(jd, ledger)

    monkeypatch.setattr(b, "save_bulk_ledger", slow)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fa = pool.submit(b.apply_operation, job_dir=j,
                         operation_id=op_a["operation_id"])
        fb = pool.submit(b.apply_operation, job_dir=j,
                         operation_id=op_b["operation_id"])
        ra = fa.result(timeout=30)
        rb = fb.result(timeout=30)
    assert ra["status"] == "applied"
    assert rb["status"] == "applied"
    ledger = b.load_bulk_ledger(j)
    assert {op_a["operation_id"], op_b["operation_id"]} <= set(
        ledger["operations"])
    assert ledger["operations"][op_a["operation_id"]]["items"]["0"][
        "outcome"] == "applied"
    assert ledger["operations"][op_b["operation_id"]]["items"]["1"][
        "outcome"] == "applied"
