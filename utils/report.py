
import os
import pandas as pd

def coverage_tables(obs_df: pd.DataFrame, env_key: str, out_md: str):
    lines = []
    lines.append("# Coverage Report\n")
    lines.append(f"- Total cells: {len(obs_df):,}\n")
    lines.append(f"- Env key: `{env_key}`\n")
    if "is_control" in obs_df:
        n_ctrl = int(obs_df["is_control"].sum())
        lines.append(f"- Controls: {n_ctrl:,} ({n_ctrl/len(obs_df):.1%})\n")
    if "cell_type" in obs_df and env_key in obs_df:
        tbl = (obs_df
               .groupby(["cell_type", env_key])
               .size()
               .reset_index(name="n_cells")
               .sort_values(["cell_type","n_cells"], ascending=[True, False]))
        lines.append("\n## Cells per cell_type × env\n\n")
        lines.append(tbl.to_markdown(index=False))
        lines.append("\n")
    if "target_gene" in obs_df:
        tg = (obs_df.query("target_gene.notnull() and target_gene != '' and is_control == False")
              .groupby(["cell_type","target_gene"]).size()
              .reset_index(name="n").sort_values("n", ascending=False).head(50))
        if len(tg) > 0:
            lines.append("\n## Top targets (non-control)\n\n")
            lines.append(tg.to_markdown(index=False))
            lines.append("\n")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    return out_md
