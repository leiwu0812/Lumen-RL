#!/usr/bin/env python3
"""Plot LumenRL training metrics from a log file into a self-contained HTML.

Parses lines like:
  ... lumenrl.trainer.callbacks: step=N grad_norm=X loss=Y lr=Z seq/max_len=...
       step_0_acc=A0 step_0_loss=L0 step_1_acc=A1 ... timing/step_s=T

Writes an interactive HTML using Plotly via CDN (no install needed in the
browser, just open the file).  Safe to re-run during training to refresh.

Usage:
  python tools/plot_training_log.py <log_path> [out_html_path]
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path


_STEP_RE = re.compile(r"callbacks:\s+step=(\d+)\s+(.*)$")
_KV_RE = re.compile(r"([a-zA-Z][\w/]*)=([-+]?\d+\.?\d*(?:e[-+]?\d+)?|inf|nan)")


def parse_log(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", errors="replace") as f:
        for line in f:
            m = _STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            kv = {"step": step}
            for k, v in _KV_RE.findall(m.group(2)):
                try:
                    kv[k] = float(v)
                except ValueError:
                    pass
            rows.append(kv)
    return rows


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LumenRL training trends &mdash; {title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 16px; background: #fafafa; color: #222; }}
  h1 {{ font-size: 18px; margin: 0 0 4px 0; }}
  .meta {{ color: #666; font-size: 12px; margin-bottom: 12px; }}
  .chart {{ background: #fff; border-radius: 8px; padding: 6px;
             box-shadow: 0 1px 3px rgba(0,0,0,.07); margin-bottom: 12px; }}
  code {{ background: #eee; padding: 1px 4px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>LumenRL Training Trends</h1>
<div class="meta">
  source: <code>{src}</code> &middot;
  parsed <b>{n}</b> steps (range {smin}&ndash;{smax}) &middot;
  generated {ts}
</div>

<div id="loss"     class="chart" style="height:340px;"></div>
<div id="acc"      class="chart" style="height:340px;"></div>
<div id="timing"   class="chart" style="height:260px;"></div>

<script>
const DATA = {data_json};
const step = DATA.map(r => r.step);

function col(name) {{ return DATA.map(r => r[name]); }}

Plotly.newPlot('loss', [
  {{x: step, y: col('loss'),      name: 'loss (sum across spec positions)', mode: 'lines', line:{{width:1.5}}}},
  {{x: step, y: col('grad_norm'), name: 'grad_norm', mode: 'lines', yaxis: 'y2', line:{{width:1, color:'#888'}}}},
], {{
  title: {{text: 'Loss &amp; grad_norm vs step', font:{{size:13}}}},
  margin: {{l:55, r:55, t:34, b:36}},
  yaxis:  {{title:'loss', rangemode:'tozero'}},
  yaxis2: {{title:'grad_norm', overlaying:'y', side:'right', rangemode:'tozero',
            showgrid:false}},
  legend: {{x:0.5, y:1.12, xanchor:'center', orientation:'h'}}
}}, {{responsive:true, displayModeBar:false}});

Plotly.newPlot('acc', [
  {{x: step, y: col('step_0_acc'), name: 'step_0 (first spec position)', mode: 'lines'}},
  {{x: step, y: col('step_1_acc'), name: 'step_1', mode: 'lines'}},
  {{x: step, y: col('step_2_acc'), name: 'step_2', mode: 'lines'}},
  {{x: step, y: col('step_3_acc'), name: 'step_3 (last spec position)', mode: 'lines'}},
], {{
  title: {{text: 'Per-spec-position acceptance (acc) vs step', font:{{size:13}}}},
  margin: {{l:55, r:55, t:34, b:36}},
  yaxis:  {{title:'acc', range:[0, null], rangemode:'tozero'}},
  legend: {{x:0.5, y:1.12, xanchor:'center', orientation:'h'}}
}}, {{responsive:true, displayModeBar:false}});

Plotly.newPlot('timing', [
  {{x: step, y: col('timing/step_s'),    name: 'step_s (total)', mode: 'lines'}},
  {{x: step, y: col('timing/teacher_s'), name: 'teacher_s',      mode: 'lines'}},
  {{x: step, y: col('timing/train_s'),   name: 'train_s',        mode: 'lines'}},
], {{
  title: {{text: 'Per-step timing (s)', font:{{size:13}}}},
  margin: {{l:55, r:55, t:34, b:36}},
  yaxis:  {{title:'seconds', rangemode:'tozero'}},
  legend: {{x:0.5, y:1.15, xanchor:'center', orientation:'h'}}
}}, {{responsive:true, displayModeBar:false}});
</script>
</body>
</html>
"""


def render_html(rows: list[dict], src: str, title: str = "") -> str:
    import datetime as _dt
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    smin = rows[0]["step"] if rows else 0
    smax = rows[-1]["step"] if rows else 0
    return _HTML_TEMPLATE.format(
        title=html.escape(title or Path(src).name),
        src=html.escape(src),
        n=len(rows),
        smin=smin,
        smax=smax,
        ts=ts,
        data_json=json.dumps(rows, separators=(",", ":")),
    )


def main(argv: list[str]) -> int:
    if not 2 <= len(argv) <= 3:
        print(__doc__)
        return 1
    log_path = Path(argv[1])
    out_path = (
        Path(argv[2])
        if len(argv) == 3
        else log_path.with_name(log_path.stem + ".html")
    )
    rows = parse_log(log_path)
    if not rows:
        print(f"No training steps found in {log_path}", file=sys.stderr)
        return 2
    out_path.write_text(render_html(rows, str(log_path)), encoding="utf-8")
    print(f"wrote {len(rows)} steps -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
