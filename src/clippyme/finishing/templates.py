"""Task 18 — the three reviewed finishing templates.

Each template is a fixed, reviewed composition: the compiler injects ONLY
typed data (timings, escaped text, dimensions) through a JSON data block and
data-* attributes. No template accepts code, markup, or style from chat.

Templates:
- clean-captions   (v1): phrase-grouped captions, current word highlighted.
- emphasized-word   (v1): word-level pop (scale + accent color) on each word.
- timed-opening     (v1): grounded opening hook (top) + clean captions.

Caption timing is half-open [start, end): a cue is visible iff
start <= t < end. All seek code is a pure function of t (no Date.now(),
no Math.random), so forward/backward seeks and export agree exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TemplateSpec:
    id: str
    version: int
    title: str
    description: str
    font_family: str
    font_file: str  # file in the repo fonts/ dir, copied immutably per compile
    hook_font_family: str = ""
    hook_font_file: str = ""
    css: str = ""
    seek_js: str = ""
    # Closed parameter enums offered by this template.
    param_enums: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared CSS fragments (fixed, reviewed)
# ---------------------------------------------------------------------------

# NOTE: the CSS below uses __W__/__H__ tokens replaced by the compiler with
# the validated integer render dimensions. No other substitution happens.

_STAGE_CSS = """
html, body { margin:0; padding:0; width:__W__px; height:__H__px; background:#000; overflow:hidden; }
#stage { position:relative; width:__W__px; height:__H__px; overflow:hidden; background:#000; }
#hf-base { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; }
#hf-caps { position:absolute; left:4%; right:4%; text-align:center; pointer-events:none;
           z-index:5; }
#hf-caps.pos-lower-third { bottom:7%; }
#hf-caps.pos-middle { bottom:38%; }
.hf-phrase { display:none; font-family:'__CAPFONT__'; color:#fff;
  text-shadow:-2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000,
              0 3px 10px rgba(0,0,0,.55);
  line-height:1.18; }
.hf-phrase.on { display:block; }
.hf-phrase .w { opacity:.55; }
.hf-phrase .w.cur { opacity:1; }
.hf-phrase.sz-small { font-size:__SZ_S__px; }
.hf-phrase.sz-medium { font-size:__SZ_M__px; }
.hf-phrase.sz-large { font-size:__SZ_L__px; }
#hf-hook { display:none; position:absolute; top:5%; left:5%; right:5%; text-align:center;
  z-index:6; font-family:'__HOOKFONT__'; color:#ffe14d;
  text-shadow:-2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000,
              0 3px 12px rgba(0,0,0,.6);
  font-size:__HOOKSZ__px; line-height:1.15; }
#hf-hook.on { display:block; }
"""

_EMPH_CSS = """
html, body { margin:0; padding:0; width:__W__px; height:__H__px; background:#000; overflow:hidden; }
#stage { position:relative; width:__W__px; height:__H__px; overflow:hidden; background:#000; }
#hf-base { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; }
#hf-caps { position:absolute; left:4%; right:4%; text-align:center; pointer-events:none; z-index:5; }
#hf-caps.pos-lower-third { bottom:7%; }
#hf-caps.pos-middle { bottom:38%; }
#hf-caps .w { display:inline-block; font-family:'__CAPFONT__'; color:#fff; margin:0 .12em;
  text-shadow:-2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000,
              0 3px 10px rgba(0,0,0,.55);
  transform:scale(1); }
#hf-caps .w.sz-small { font-size:__SZ_S__px; }
#hf-caps .w.sz-medium { font-size:__SZ_M__px; }
#hf-caps .w.sz-large { font-size:__SZ_L__px; }
#hf-caps .w.on { color:#ffe14d; transform:scale(1.18); }
"""

# ---------------------------------------------------------------------------
# Shared seek implementations (fixed, reviewed). Pure functions of t.
# ---------------------------------------------------------------------------

# Phrase captions: toggle .on per half-open [start, end); mark the current
# word inside the visible phrase.
_SEEK_PHRASE_JS = """
(function(){
  var data = JSON.parse(document.getElementById('hf-data').textContent);
  var D = data.duration;
  var phrases = Array.prototype.slice.call(document.querySelectorAll('#hf-caps .hf-phrase'));
  var hook = document.getElementById('hf-hook');
  function seek(t){
    t = Math.max(0, Math.min(D, t));
    for (var i = 0; i < phrases.length; i++){
      var el = phrases[i];
      var s = parseFloat(el.getAttribute('data-start'));
      var e = parseFloat(el.getAttribute('data-end'));
      var on = (s <= t && t < e);
      el.classList.toggle('on', on);
      if (on){
        var words = el.querySelectorAll('.w');
        for (var j = 0; j < words.length; j++){
          var w = words[j];
          var ws = parseFloat(w.getAttribute('data-start'));
          var we = parseFloat(w.getAttribute('data-end'));
          w.classList.toggle('cur', ws <= t && t < we);
        }
      }
    }
    if (hook){
      var hs = parseFloat(hook.getAttribute('data-start'));
      var he = parseFloat(hook.getAttribute('data-end'));
      var hon = (hs <= t && t < he);
      hook.classList.toggle('on', hon);
      var k = hon ? Math.min(1, (t - hs) / 0.25) : 0;  /* seekable entrance */
      hook.style.opacity = String(k);
      hook.style.transform = 'translateY(' + ((1 - k) * -18).toFixed(2) + 'px)';
    }
  }
  window.__hf = {
    duration: D,
    seek: seek,
    media: [{ elementId: 'hf-base', src: 'base.mp4', startTime: 0, endTime: D, mediaOffset: 0 }]
  };
  window.__playerReady = true;
  window.__renderReady = true;
})();
"""

# Word pop: each word span toggles .on per half-open [start, end).
_SEEK_WORD_JS = """
(function(){
  var data = JSON.parse(document.getElementById('hf-data').textContent);
  var D = data.duration;
  var words = Array.prototype.slice.call(document.querySelectorAll('#hf-caps .w'));
  function seek(t){
    t = Math.max(0, Math.min(D, t));
    for (var i = 0; i < words.length; i++){
      var w = words[i];
      var s = parseFloat(w.getAttribute('data-start'));
      var e = parseFloat(w.getAttribute('data-end'));
      w.classList.toggle('on', s <= t && t < e);
    }
  }
  window.__hf = {
    duration: D,
    seek: seek,
    media: [{ elementId: 'hf-base', src: 'base.mp4', startTime: 0, endTime: D, mediaOffset: 0 }]
  };
  window.__playerReady = true;
  window.__renderReady = true;
})();
"""

_PARAM_ENUMS = {
    "caption_size": ("small", "medium", "large"),
    "caption_position": ("lower-third", "middle"),
}


def _spec(template_id: str, **kw) -> TemplateSpec:
    return TemplateSpec(id=template_id, version=1, param_enums=dict(_PARAM_ENUMS), **kw)


TEMPLATES: dict[str, TemplateSpec] = {
    "clean-captions": _spec(
        "clean-captions",
        title="Clean captions",
        description="Phrase-grouped captions in the viral style; the current word is highlighted.",
        font_family="Anton",
        font_file="Anton-Regular.ttf",
        css=_STAGE_CSS,
        seek_js=_SEEK_PHRASE_JS,
    ),
    "emphasized-word": _spec(
        "emphasized-word",
        title="Emphasized word captions",
        description="Each spoken word pops (scale + accent color) exactly on its timing.",
        font_family="Anton",
        font_file="Anton-Regular.ttf",
        css=_EMPH_CSS,
        seek_js=_SEEK_WORD_JS,
    ),
    "timed-opening": _spec(
        "timed-opening",
        title="Timed opening + captions",
        description="The grounded opening hook shows during its timed window, with clean captions below.",
        font_family="Anton",
        font_file="Anton-Regular.ttf",
        hook_font_family="Bangers",
        hook_font_file="Bangers-Regular.ttf",
        css=_STAGE_CSS,
        seek_js=_SEEK_PHRASE_JS,
    ),
}

TEMPLATE_IDS = tuple(TEMPLATES.keys())


def get_template(template_id: str) -> TemplateSpec:
    """Return the reviewed template spec, or raise KeyError for unknown ids."""
    return TEMPLATES[template_id]
