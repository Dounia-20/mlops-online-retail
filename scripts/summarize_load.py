import glob
import re

import pandas as pd

rows = []
for f in sorted(glob.glob("docs/charge/w*_u*_stats.csv")):
    m = re.search(r"w(\d+)_u(\d+)_stats", f)
    df = pd.read_csv(f)
    r = df[df["Name"] == "/predict"].iloc[0]
    rows.append({
        "workers": int(m.group(1)), "utilisateurs": int(m.group(2)),
        "requetes": int(r["Request Count"]), "echecs": int(r["Failure Count"]),
        "req/s": round(r["Requests/s"], 1),
        "mediane (ms)": int(r["50%"]), "p95 (ms)": int(r["95%"]), "p99 (ms)": int(r["99%"]),
    })
out = pd.DataFrame(rows).sort_values(["workers", "utilisateurs"])
out["taux d'echec (%)"] = (100 * out["echecs"] / out["requetes"]).round(2)
table = out.to_markdown(index=False)
open("docs/charge/resultats.md", "w", encoding="utf-8").write(table + "\n")
print(table)
