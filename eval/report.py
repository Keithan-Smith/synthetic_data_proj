import os
import pandas as pd
from datetime import datetime

HTML_TMPL = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Synthetic Data Report</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }
h1,h2 { margin-bottom: 6px; }
.card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin: 12px 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #e5e7eb; padding: 8px; text-align: left; }
.small { color: #6b7280; font-size: 13px; }
img { max-width: 100%; border-radius: 8px; border: 1px solid #e5e7eb; }
code { background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }
</style></head>
<body>
<h1>Synthetic Data — Run Report</h1>
<div class="small">Generated: {ts}</div>

<div class="card">
<h2>Summary</h2>
<p><b>Correlation avg abs diff:</b> {corr_diff:.4f}</p>
<p><b>Binary Utility (AUC, Brier):</b> {auc:.3f}, {brier:.3f}</p>
<p><b>Membership Inference AUC (↓ is better):</b> {mia:.3f}</p>
<p class="small">DP epsilon log (if DP enabled): <code>{epslog}</code></p>
</div>

<div class="card">
<h2>Univariate Fidelity</h2>
{fidelity_table}
</div>

<div class="card">
<h2>Visuals</h2>
<div class="grid">
  {images}
</div>
</div>

<div class="card">
<h2>Artifacts</h2>
<ul>
  {artifact_links}
</ul>
</div>

</body></html>"""

def _df_to_html_table(df: pd.DataFrame, max_rows=30) -> str:
    df2 = df.copy()
    if len(df2) > max_rows: df2 = df2.head(max_rows)
    return df2.to_html(index=False)

def build_report(output_dir: str):
    fid_path = os.path.join(output_dir, "fidelity_univariate.csv")
    sum_path = os.path.join(output_dir, "summary.txt")
    corr_diff = 0.0; auc = float('nan'); brier = float('nan'); mia = float('nan'); epslog = ""
    if os.path.exists(fid_path):
        fid = pd.read_csv(fid_path); fid_html = _df_to_html_table(fid)
    else:
        fid_html = "<p>No fidelity file found.</p>"

    if os.path.exists(sum_path):
        with open(sum_path,"r") as f: txt = f.read()
        for line in txt.splitlines():
            if "Correlation avg abs diff" in line: corr_diff = float(line.split(":")[1].strip())
            elif "AUC" in line and "synth->real" in line: auc = float(line.split(":")[1].strip())
            elif "Brier" in line: brier = float(line.split(":")[1].strip())
            elif "Membership inference AUC" in line: mia = float(line.split(":")[1].strip())
            elif "DP eps log" in line: epslog = line.split(":",1)[1].strip()

    images, artifacts = [], []
    for fname in sorted(os.listdir(output_dir)):
        if fname.endswith(".png"):
            images.append(f'<div><div class="small">{fname}</div><img src="{fname}" /></div>')
        elif fname.endswith((".csv",".txt",".json")):
            artifacts.append(f'<li><a href="{fname}">{fname}</a></li>')

    html = HTML_TMPL.format(
        ts=datetime.utcnow().isoformat(), corr_diff=corr_diff, auc=auc, brier=brier, mia=mia, epslog=epslog,
        fidelity_table=fid_html, images="\n".join(images), artifact_links="\n".join(artifacts)
    )
    out_path = os.path.join(output_dir, "index.html")
    with open(out_path, "w") as f: f.write(html)
    return out_path
