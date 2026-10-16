import sys, numpy as np
r = np.load(sys.argv[1]); s, t, T, a, b, fc, rc, eos = r.T
fr = eos < 0
pre = fr & (t < a); post = fr & (t >= b); ref = fr & (t >= a) & (t < b)
def row(name, m):
    print(f"{name:34s} n={m.sum():7d}  first={fc[m].mean():.3f}  resid={np.nanmean(rc[m]):.3f}")
print("== regions (25 frames = 2 s)")
row("pre-reference [0,a)", pre); row("reference (masked) [a,b)", ref); row("post-reference [b,T)", post); row("EOS", eos > 0)
print("== absolute position from BOS, pre-reference frames only")
for lo, hi in [(0,1),(1,2),(2,3),(3,5),(5,10),(10,25),(25,50),(50,100),(100,10**6)]:
    row(f"t in [{lo},{hi})", pre & (t >= lo) & (t < hi))
print("== same, restricted to samples with a >= 50 (pre region >= 4 s)")
long = a >= 50
for lo, hi in [(0,1),(1,3),(3,10),(10,25),(25,50)]:
    row(f"t in [{lo},{hi})", pre & long & (t >= lo) & (t < hi))
print("== right after the masked reference, frames t-b")
for lo, hi in [(0,1),(1,3),(3,10),(10,25),(25,10**6)]:
    row(f"t-b in [{lo},{hi})", post & (t - b >= lo) & (t - b < hi))
print("== paired per sample (a >= 25): first 3 frames minus frames [10,a)")
d1, d2 = [], []
for k in np.unique(s[fr & (a >= 25)]):
    m = fr & (s == k)
    on = m & (t < 3); rest = m & (t >= 10) & (t < a)
    if on.sum() and rest.sum():
        d1.append(fc[on].mean() - fc[rest].mean()); d2.append(rc[on].mean() - rc[rest].mean())
d1, d2 = np.array(d1), np.array(d2)
for name, d in (("first", d1), ("resid", d2)):
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"{name}: n={len(d)} mean diff={d.mean():+.3f} (95% CI {d.mean()-1.96*se:+.3f}..{d.mean()+1.96*se:+.3f}), share>0={np.mean(d>0):.2f}")
print("== relative position deciles (supervised frames, pre and post separately)")
for name, m in (("pre", pre), ("post", post)):
    rel = t / T
    print(name, " ".join(f"{fc[m & (rel>=q/10) & (rel<(q+1)/10)].mean():.2f}" if (m & (rel>=q/10) & (rel<(q+1)/10)).sum() else "  - " for q in range(10)))
print("== paired per sample: pre-reference mean minus post-reference mean (>= 5 frames each)")
for label, extra in (("all pre frames", pre), ("pre frames t >= 10", pre & (t >= 10))):
    d1, d2 = [], []
    for k in np.unique(s):
        m = s == k; p = extra & m; q = post & m
        if p.sum() >= 5 and q.sum() >= 5:
            d1.append(fc[p].mean() - fc[q].mean()); d2.append(rc[p].mean() - rc[q].mean())
    for name, d in (("first", np.array(d1)), ("resid", np.array(d2))):
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"{label} {name}: n={len(d)} diff={d.mean():+.3f} (95% CI {d.mean()-1.96*se:+.3f}..{d.mean()+1.96*se:+.3f}), share(pre>post)={np.mean(d>0):.2f}")
