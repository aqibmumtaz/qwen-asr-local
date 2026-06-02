import json

d = json.loads(open("benchmark_results/cll_confidence_raw.json").read())
words = d["words"]
total = len(words)
low = sum(1 for w in words if w["min_conf"] <= 0.65)
print("=== CLL Audio Confidence Stats ===")
print(f"Duration: 141s (8kHz) | Inference: {d['elapsed']:.0f}s CPU")
print(f"Total words: {total} | LOW: {low} ({low/total*100:.0f}%)")
print()
b = [0, 0, 0, 0]
for w in words:
    c = w["min_conf"]
    if c <= 0.50:
        b[0] += 1
    elif c <= 0.65:
        b[1] += 1
    elif c <= 0.80:
        b[2] += 1
    else:
        b[3] += 1
for label, n in zip(["0.00-0.50", "0.50-0.65", "0.65-0.80", "0.80-1.00"], b):
    print(f"  {label}: {n:3d} ({n*100//total}%)")
print()
print("Top 15 lowest confidence (annotation priority):")
sw = sorted(words, key=lambda w: w["min_conf"])
for i, w in enumerate(sw[:15], 1):
    print(f"  {i:2d}. {w['text']:<12} min={w['min_conf']:.3f} geo={w['geo_conf']:.3f}")
