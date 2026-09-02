"""合并并行调参的两份日志与参数 (problem2_tuning_r30.txt / _r50.txt)."""
import json, os, re

ROOT = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(ROOT, "results", "tables")

merged = ["# 问题二调参结果 (Lena 512, v2: β 网格扩展至 640, 选优优先收敛解)"]
params = {}
for r in (0.3, 0.5):
    fn = os.path.join(TAB, f"problem2_tuning_r{int(r * 100)}.txt")
    txt = open(fn, encoding="utf-8").read()
    merged.append("")
    merged.append(f"== r={r:.0%} ==")
    for line in txt.splitlines():
        if line.startswith("=="):
            merged.append("== r={:.0f}% ==".format(r * 100))
            continue
        if line.startswith("#") or not line.strip():
            continue
        merged.append(line)
    for line in txt.splitlines():
        m = re.match(r"\s*>> (\w+): alpha\*=([0-9.eE+-]+), beta\*=([0-9.eE+-]+), PSNR\*=([0-9.]+)",
                     line)
        if m:
            params.setdefault(m.group(1), {})[
                "0.3" if abs(r - 0.3) < 1e-9 else "0.5"] = [
                float(m.group(2)), float(m.group(3))]
with open(os.path.join(TAB, "problem2_tuning.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(merged))
with open(os.path.join(TAB, "problem2_params.json"), "w", encoding="utf-8") as f:
    json.dump(params, f, indent=2)
print(json.dumps(params, indent=2))
