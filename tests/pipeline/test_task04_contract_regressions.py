"""Independent Task04 review contracts. No live provider or media calls."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
from clippyme.pipeline import cut_ops, gemini_request as gr, orchestrator as orch
from clippyme.pipeline.duration_policy import resolve_duration_policy

@pytest.mark.parametrize("length", [5., 10., 24.])
def test_implicit_portfolio_obeys_short_source(length):
    p = resolve_duration_policy(video_duration=length)
    assert 0 < p.min_duration <= p.max_duration <= length

def test_actual_prompt_preserves_separate_segment_summary():
    t={"text":"Hello", "language":"en", "segments":[
        {"start":0., "end":1., "speaker":7,
         "words":[{"word":"Hello", "start":0., "end":1.}]}]}
    original=copy.deepcopy(t)
    prompt, words=gr.build_viral_prompt(t, 100.)
    assert t==original
    assert "sp" not in words[0]
    # Actual serialized evidence, not an unused helper or instruction.
    assert "segment-summary" in prompt
    assert "7" in prompt.split("segment-summary", 1)[1]

def test_prompt_duration_bounds_do_not_round_up_an_explicit_max():
    t={"text":"Hello", "segments":[]}
    prompt, _=gr.build_viral_prompt(t, 100., min_duration=15.2, max_duration=30.6)
    assert "15.2s ≤ duration ≤ 30.6s" in prompt

def test_legacy_sentence_helper_retains_its_sixty_second_default():
    words=[{"word":"Before.", "start":8., "end":9.},
           {"word":"Next", "start":10., "end":10.4},
           {"word":"ending.", "start":89.5, "end":90.}]
    result=cut_ops.snap_clip_to_sentences(10.2, 89.1, words,
        word_start=10.15,word_end=89.18)
    assert result==(10.15,89.18,"word")

def test_optional_word_padding_does_not_cross_max():
    words=[{"word":"First", "start":0., "end":.4},
           {"word":"last.", "start":29.5, "end":30.}]
    clips=[{"start":0., "end":30.}]
    cut_ops.snap_clips_to_transcript(clips, words, source_duration=100., max_duration=30.)
    assert clips[0]["end"]-clips[0]["start"] <= 30.

def test_silence_polish_does_not_cross_max():
    words=[{"word":"First", "start":0., "end":.4},
           {"word":"last.", "start":29.5, "end":29.92}]
    clips=[{"start":0., "end":29.92}]
    cut_ops.snap_clips_to_transcript(clips, words, source_duration=100.,
        max_duration=30., silences=[(30.1,31.)])
    assert clips[0]["end"]-clips[0]["start"] <= 30.

def test_existing_oversized_word_complete_proposal_does_not_grow():
    words=[{"word":"First", "start":0., "end":.4},
           {"word":"last.", "start":34.5, "end":35.}]
    clips=[{"start":0., "end":35.}]
    cut_ops.snap_clips_to_transcript(clips, words, source_duration=100., max_duration=30.)
    assert clips[0]["end"]-clips[0]["start"] <= 35.

def test_unchanged_oversized_proposal_has_diagnostic():
    words=[{"word":"First", "start":0., "end":.4},
           {"word":"last.", "start":34.5, "end":34.92}]
    clips=[{"start":0., "end":35.}]
    events=cut_ops.snap_clips_to_transcript(clips, words, source_duration=100., max_duration=30.)
    # No edge moves: returning [] hides the unresolved policy conflict.
    assert events or any("duration" in key or "conflict" in key for key in clips[0])

def test_genuine_long_sentence_repair_can_expand_beyond_sixty():
    words=[{"word":"Before.", "start":8., "end":9.},
           {"word":"Next", "start":10., "end":10.4},
           {"word":"ending.", "start":99.5, "end":100.}]
    clips=[{"start":10.2, "end":99.1}]
    cut_ops.snap_clips_to_transcript(clips, words, source_duration=200., max_duration=95.)
    assert clips[0]["start"] == pytest.approx(9.95)
    assert clips[0]["end"] == pytest.approx(100.08)
    assert clips[0]["end"]-clips[0]["start"] < 95.

def test_queued_minimum_conflict_is_observable(tmp_path,monkeypatch,capsys):
    transcript={"segments":[{"words":[
        {"word":"First", "start":0., "end":.4},
        {"word":"last.", "start":29.4, "end":29.8}]}]}
    args=SimpleNamespace(min_duration=30., max_duration=60.,
        duration_mode=None, clip_type=None, reframe_mode="auto", skip_analysis=False,
        monitor=False,instructions=None,min_clips=None,max_clips=None,
        aspect="9:16", caption_style_default=None)
    class State:
        def artifact(self,*a): return None
        def completed(self,*a): return False
        def start(self,*a): pass
        def consume_steering(self): return []
        def set_clip_total(self,*a): pass
        def complete_stage(self,*a,**k): pass
    legacy=SimpleNamespace(get_viral_clips=lambda *a,**k: {"shorts":[{"start":0.,"end":30.}]})
    monkeypatch.setenv("CLIPPYME_SILENCE_SNAP","0")
    monkeypatch.setenv("CLIPPYME_MAX_CLIPS","0")
    data,_=orch._load_or_analyze(args,str(tmp_path/"source.mp4"),"source",
        str(tmp_path),100.,transcript,State(),legacy)
    clip=data["shorts"][0]
    text=capsys.readouterr().out.lower()
    assert (clip["end"]-clip["start"]>=30. or "conflict" in text or
            any("duration" in key or "conflict" in key for key in clip))

@pytest.mark.parametrize("module", ["queued", "direct"])
def test_production_repair_callers_use_the_actual_policy(tmp_path, monkeypatch, module):
    transcript={"segments":[{"words":[
        {"word":"First", "start":0., "end":.4},
        {"word":"last.", "start":29.5, "end":29.92}]}]}
    clips={"shorts":[{"start":0., "end":29.92}]}
    args=SimpleNamespace(min_duration=15., max_duration=30.,
        duration_mode="mixed", clip_type=None, reframe_mode="auto",
        skip_analysis=False, monitor=False, instructions=None,
        min_clips=None, max_clips=None, aspect="9:16", caption_style_default=None)
    calls=[]
    actual_snap=cut_ops.snap_clips_to_transcript
    def record(clips, words, **kwargs):
        calls.append(kwargs)
        return actual_snap(clips, words, **kwargs)
    monkeypatch.setattr(cut_ops,"snap_clips_to_transcript",record)
    monkeypatch.setenv("CLIPPYME_SILENCE_SNAP","0")
    monkeypatch.setenv("CLIPPYME_MAX_CLIPS","0")
    if module=="queued":
        class State:
            def artifact(self,*a): return None
            def completed(self,*a): return False
            def start(self,*a): pass
            def consume_steering(self): return []
            def set_clip_total(self,*a): pass
            def complete_stage(self,*a,**k): pass
        legacy=SimpleNamespace(get_viral_clips=lambda *a,**k: copy.deepcopy(clips))
        orch._load_or_analyze(args, str(tmp_path/"source.mp4"), "source",
            str(tmp_path), 100., transcript, State(), legacy)
    else:
        tree=ast.parse((Path(orch.__file__).parent/"main.py").read_text(encoding="utf8"))
        entry=next(n for n in tree.body if isinstance(n,ast.If) and
            "__name__" in ast.unparse(n.test))
        # Exact production contiguous repair statements, not a reimplemented wrapper.
        node=next(n for n in ast.walk(entry) if isinstance(n,ast.ImportFrom) and
            n.module=="clippyme.pipeline.cut_ops" and
            any(a.name=="snap_clips_to_transcript" for a in n.names))
        body=next(getattr(n,field) for n in ast.walk(entry)
            for field in ("body", "orelse", "finalbody")
            if isinstance(getattr(n,field,None),list) and node in getattr(n,field))
        start=body.index(node)
        end=next(i for i,n in enumerate(body[start:],start)
                 if isinstance(n,ast.For) and
                 "snap_clips_to_transcript" in ast.unparse(n.iter))
        exact=ast.Module(body=body[start:end+1],type_ignores=[])
        import os
        env={"args":args, "duration":100., "transcript":transcript,
             "clips_data":copy.deepcopy(clips), "input_video":str(tmp_path/"source.mp4"),
             "os":os}
        exec(compile(ast.fix_missing_locations(exact), "production repair block", "exec"),env)
    assert len(calls)==1
    assert calls[0]["max_duration"] == 30.
    assert calls[0]["source_duration"] == 100.


"""Task04 revision2 contracts: content-preserving fallback and actual prompt policy."""
import importlib.util
from pathlib import Path
import re
import pytest
from clippyme.pipeline import cut_ops, gemini_request as gr, gemini_parser as gp
from clippyme.pipeline.duration_policy import resolve_duration_policy

@pytest.mark.parametrize("length", [5.,10.,24.])
def test_short_source_actual_prompt_matches_resolved_minimum(length):
    p=resolve_duration_policy(video_duration=length)
    prompt,_=gr.build_viral_prompt({"segments":[]}, length)
    matches=re.findall(r"([\d.eE+-]+)s\s*≤ duration ≤\s*([\d.eE+-]+)s", prompt)
    assert matches, "actual mandatory duration constraint must be parseable"
    lo,hi=map(float,matches[0])
    assert (lo,hi)==(p.min_duration,p.max_duration)
    assert lo<=hi<=length
    # Tiers unavailable for this source cannot be a mandatory mix.
    assert "MANDATORY MIX" not in prompt

def test_padding_conflict_keeps_selected_words():
    words=[{"word":"First","start":10.,"end":10.4},
           {"word":"payoff.","start":39.5,"end":40.}]
    clips=[{"start":10.,"end":40.}]
    cut_ops.snap_clips_to_transcript(clips,words,
        source_duration=100.,max_duration=30.,min_duration=15.)
    assert clips[0]["start"]<=10.
    assert clips[0]["end"]>=40., "decline optional padding; preserve selected payoff"
    assert clips[0]["end"]-clips[0]["start"]<=30.

def test_oversized_proposal_is_not_silently_reselected():
    words=[{"word":"First","start":0.,"end":.4},
           {"word":"payoff.","start":34.5,"end":35.}]
    clips=[{"start":0.,"end":35.}]
    cut_ops.snap_clips_to_transcript(clips,words,
        source_duration=100.,max_duration=30.,min_duration=15.)
    assert clips[0]["end"]>=35., "retain selected word-safe interval and report max conflict"
    assert clips[0]["duration_conflict"]=="maximum_conflict"

def test_cap_fallback_checks_minimum_again():
    words=[{"word":"First","start":0.,"end":.4},
           {"word":"payoff.","start":34.5,"end":35.}]
    clips=[{"start":0.,"end":35.}]
    cut_ops.snap_clips_to_transcript(clips,words,
        source_duration=100.,max_duration=30.,min_duration=15.)
    duration=clips[0]["end"]-clips[0]["start"]
    assert duration>=15. or clips[0].get("duration_conflict")=="minimum_conflict"

def test_newly_selected_boundary_must_not_cut_an_overlapping_word():
    words=[{"word":"first","start":0.,"end":.4},
           {"word":"A","start":9.5,"end":10.},
           {"word":"B","start":9.7,"end":10.4}]
    clips=[{"start":0.,"end":10.4}]
    cut_ops.snap_clips_to_transcript(clips,words,
        source_duration=20.,max_duration=10.,min_duration=1.)
    edge=clips[0]["end"]
    assert not any(w["start"]<edge<w["end"] for w in words)

def test_old_conflict_is_removed_when_current_policy_is_satisfied():
    words=[{"word":"First","start":0.,"end":.4},
           {"word":"payoff.","start":19.5,"end":20.}]
    clips=[{"start":0.,"end":20.,"duration_conflict":"maximum_conflict",
            "duration_conflict_detail":"old policy"}]
    cut_ops.snap_clips_to_transcript(clips,words,
        source_duration=100.,max_duration=60.,min_duration=15.)
    assert "duration_conflict" not in clips[0]
    assert "duration_conflict_detail" not in clips[0]

@pytest.mark.parametrize("maximum",[30.1234567,99.999999])
def test_explicit_maximum_is_never_rounded_up_in_prompt(maximum):
    prompt,_=gr.build_viral_prompt({"segments":[]},100.,
        min_duration=15.,max_duration=maximum)
    matches=re.findall(r"([\d.eE+-]+)s\s*≤ duration ≤\s*([\d.eE+-]+)s", prompt)
    assert matches
    assert float(matches[0][1])==maximum

def test_committed_hook_gate_rejects_a_broken_consumer(monkeypatch):
    path=Path(gr.__file__).parents[3]/"tests"/"pipeline"/"test_task04_speaker_duration.py"
    spec=importlib.util.spec_from_file_location("task04_committed_test",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def broken(*args,**kwargs):
        raise AttributeError("Injected consumer regression")
    monkeypatch.setattr(gp,"backfill_hook_text",broken)
    with pytest.raises(AttributeError):
        module.test_hook_backfill_compatibility()

def test_real_hook_backfill_accepts_augmented_words():
    _,words=gr.build_viral_prompt({"segments":[{"words":[
        {"word":"Hello","start":0.,"end":.5,"speaker":0},
        {"word":"world","start":.5,"end":1.,"speaker":1}]}]},100.)
    clips=[{"start":0.,"end":1.,"viral_hook_text":"",
             "video_title_for_youtube_short":"The payoff is surprising"}]
    result=gp.backfill_hook_text(clips,words)
    assert result is clips
    assert clips[0]["viral_hook_text"]=="The payoff is surprising"
    assert [w["sp"] for w in words]==[0,1]


"""Task04 revision3 remaining safety gates, independent of stored examples."""
import pytest
from clippyme.pipeline import cut_ops, gemini_request as gr

def test_overlap_chain_fallback_reaches_a_genuinely_safe_edge():
    words=[{"word":"first","start":0.,"end":.4},
           {"word":"A","start":9.,"end":11.},
           {"word":"B","start":10.5,"end":12.},
           {"word":"C","start":11.5,"end":13.}]
    clips=[{"start":0.,"end":10.}]
    cut_ops.snap_clips_to_transcript(clips,words,
        source_duration=20.,max_duration=10.,min_duration=1.)
    edge=clips[0]["end"]
    assert not any(w["start"]<edge<w["end"] for w in words)
    assert clips[0].get("duration_conflict")=="maximum_conflict"

def test_in_budget_proposal_is_still_validated_for_overlapping_speech():
    words=[{"word":"first","start":0.,"end":.4},
           {"word":"A","start":9.,"end":11.},
           {"word":"B","start":10.5,"end":12.}]
    clips=[{"start":0.,"end":10.}]
    cut_ops.snap_clips_to_transcript(clips,words,
        source_duration=20.,max_duration=20.,min_duration=1.)
    edge=clips[0]["end"]
    assert not any(w["start"]<edge<w["end"] for w in words)

def test_fallback_revalidates_minimum_after_declining_padding():
    words=[{"word":"first","start":0.,"end":.4},
           {"word":"last.","start":29.5,"end":29.95}]
    clips=[{"start":0.,"end":29.95}]
    cut_ops.snap_clips_to_transcript(clips,words,
        source_duration=100.,max_duration=30.,min_duration=30.)
    assert clips[0].get("duration_conflict")=="minimum_conflict"

def test_fallback_cannot_escape_known_source_boundary():
    # ASR timestamp can drift past the actual decoded source end.
    words=[{"word":"first","start":0.,"end":.4},
           {"word":"last.","start":9.,"end":11.}]
    clips=[{"start":0.,"end":10.}]
    cut_ops.snap_clips_to_transcript(clips,words,
        source_duration=10.2,max_duration=10.,min_duration=1.)
    assert 0<=clips[0]["start"]<clips[0]["end"]<=10.2

@pytest.mark.parametrize("length",[60.,90.,119.])
def test_prompt_does_not_require_tiers_longer_than_known_source(length):
    prompt,_=gr.build_viral_prompt({"segments":[]},length)
    assert "EXTENDED DEEP DIVE (120s - 300s" not in prompt

def test_in_budget_oversized_interval_cannot_lose_selected_words():
    words=[{"word":"first","start":0.,"end":.4},
           {"word":"payoff","start":29.95,"end":30.1}]
    clips=[{"start":0.,"end":30.}]
    cut_ops.snap_clips_to_transcript(clips,words,
        source_duration=100.,max_duration=30.,min_duration=15.)
    assert clips[0]["end"]>=30.1
    assert clips[0].get("duration_conflict")=="maximum_conflict"


"""Task04 revision4 neighbor/time guard and production-test gates."""
import importlib.util
from pathlib import Path
import re
import pytest
from clippyme.pipeline import cut_ops,gemini_request as gr

def test_adjacent_clips_keep_their_selected_time_ranges():
    words=[{"word":"first","start":0.,"end":.4},
           {"word":"payoff.","start":9.5,"end":10.},
           {"word":"second","start":20.,"end":20.4},
           {"word":"last.","start":29.5,"end":30.}]
    clips=[{"start":20.,"end":30.},{"start":0.,"end":10.}]
    cut_ops.snap_clips_to_transcript(clips,words,source_duration=100.,
        min_duration=1.,max_duration=30.)
    assert clips[1]["start"]<=0.
    assert clips[1]["end"]>=10.
    assert clips[1]["end"]<=20.
    assert clips[0]["start"]>=10.
    assert clips[0]["start"]<=20.
    assert clips[0]["end"]>=30.

def test_both_neighbor_guards_do_not_create_an_interval_outside_source():
    words=[{"word":"speech","start":9.8,"end":10.5}]
    s,e,conflict=cut_ops._word_safe_interval(9.95,10.,words,
        source_duration=10.,neighbor_start=10.,neighbor_end=9.9)
    assert 9.9<=s<e<=10.
    assert conflict is not None

@pytest.mark.parametrize("length",[30.1234567,99.999999])
def test_available_tier_ranges_never_round_above_known_source(length):
    prompt,_=gr.build_viral_prompt({"segments":[]},length)
    tiers=re.findall(r"TIER \d[^\n]*?\(([\d.eE+-]+)s - ([\d.eE+-]+)s\)",prompt)
    assert tiers
    for low,high in tiers:
        assert float(low)<=float(high)<=length

def test_previous_neighbor_conflict_at_start_is_observable():
    words=[{"word":"speech","start":9.,"end":11.},
           {"word":"payoff","start":11.5,"end":12.}]
    s,e,conflict=cut_ops._word_safe_interval(10.,12.,words,
        source_duration=20.,neighbor_end=9.9)
    assert 9.9<=s<e
    assert conflict is not None and conflict["type"]=="neighbor_conflict"

def test_source_conflict_survives_optional_padding_fallback():
    words=[{"word":"first","start":2.,"end":2.4},
           {"word":"last","start":4.9,"end":6.}]
    clips=[{"start":2.,"end":5.}]
    cut_ops.snap_clips_to_transcript(clips,words,
        source_duration=5.,max_duration=3.,min_duration=1.)
    assert clips[0]["start"]==2. and clips[0]["end"]==5.
    assert clips[0].get("duration_conflict")=="source_conflict"

def test_committed_direct_gate_rejects_wrong_production_forwarding(monkeypatch):
    testpath=Path(gr.__file__).parents[3]/"tests"/"pipeline"/"test_task04_speaker_duration.py"
    spec=importlib.util.spec_from_file_location("task04_direct_gate",testpath)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original=Path.read_text
    def changed(path,*args,**kwargs):
        source=original(path,*args,**kwargs)
        if path.name=="main.py":
            source=source.replace("max_duration=_policy.max_duration","max_duration=999.0")
        return source
    monkeypatch.setattr(Path,"read_text",changed)
    # Actual AST reading gate should execute the mutated production call and fail.
    # A copied helper call never sees main.py and falsely passes.
    with pytest.raises(AssertionError):
        module.test_direct_cli_forwards_policy_values_with_spy(monkeypatch)
