"""Dataset-paper statistics for the MS MARCO supervision set (gemma-4-31B-it, XML,
constrained, reasoning): split sizes, query/passage/span lengths, spans per pair,
span coverage, relative span start position and empty rate, for train and
validation. Prints a text summary and writes figs/msmarco_supervision/*.png.

Run from the repo root:  python eval/msmarco_stats.py
"""
import json, numpy as np, collections, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B="data/extracted_relevancy/ms-marco-xml-constrained-reasoning/google~gemma-4-31B-it_"
W=lambda s: len(s.split())
res={}
os.makedirs("figs/msmarco_supervision", exist_ok=True)
for split in ("train","validation"):
    qlen=[];plen=[];nsp=[];splen=[];cov=[];pos=[];empty=0;n=0;notsub=0
    qids=set();pids=set();qdup=collections.Counter()
    for l in open(B+split+".jsonl"):
        d=json.loads(l); n+=1; q=d["query"]; p=d["passage"]; sp=d.get("selected_spans") or []
        qids.add(d["qid"]); pids.add(d["doc_id"]); qdup[d["qid"]]+=1
        qlen.append(W(q)); plen.append(W(p)); nsp.append(len(sp))
        if not sp: empty+=1; continue
        tot=0
        for s in sp:
            splen.append(W(s)); i=p.find(s)
            if i<0: notsub+=1
            else: pos.append(i/max(1,len(p)-len(s)) if len(p)>len(s) else 0.0)
            tot+=len(s)
        cov.append(min(1.0,tot/len(p)))
    pct=lambda a:(np.mean(a),np.median(a),np.percentile(a,90),np.max(a))
    res[split]=dict(n=n,uq=len(qids),up=len(pids),multi=sum(1 for v in qdup.values() if v>1),
        empty=empty,notsub=notsub,qlen=pct(qlen),plen=pct(plen),nsp=collections.Counter(nsp),
        nsp_nonempty_mean=np.mean([x for x in nsp if x]),splen=pct(splen),cov=pct(cov),
        cov_hist=np.histogram(cov,bins=20,range=(0,1))[0],pos_hist=np.histogram(pos,bins=10,range=(0,1))[0],
        splen_hist=np.histogram(np.clip(splen,0,60),bins=30,range=(0,60))[0],total_spans=len(splen),
        single_frac=np.mean([x==1 for x in nsp if x]))
    res[split]["_arrays"]=(cov,pos,splen,nsp,qlen,plen)
for s,r in res.items():
    print(f"\n== {s}: pairs={r['n']:,} unique queries={r['uq']:,} unique passages={r['up']:,} queries with >1 passage={r['multi']:,}")
    print(f"empty={r['empty']:,} ({100*r['empty']/r['n']:.1f}%)  spans not verbatim substrings={r['notsub']}  total spans={r['total_spans']:,}")
    for k in ("qlen","plen","splen","cov"):
        m,md,p90,mx=r[k]; f=(lambda x:f"{x:.3f}") if k=="cov" else (lambda x:f"{x:.1f}")
        print(f"{k:6s} mean={f(m)} median={f(md)} p90={f(p90)} max={f(mx)}")
    c=r["nsp"]; tot=sum(c.values())
    print("spans/pair dist:", {k:f"{100*v/tot:.1f}%" for k,v in sorted(c.items())[:7]}, f"| mean (non-empty)={r['nsp_nonempty_mean']:.2f} single-span={100*r['single_frac']:.1f}%")
# figs
from matplotlib.ticker import PercentFormatter
def add_pct_axis(a, total):
    r=a.twinx(); lo,hi=a.get_ylim(); r.set_ylim(lo/total*100, hi/total*100)
    r.yaxis.set_major_formatter(PercentFormatter(decimals=0)); r.set_ylabel("% of "+a.get_ylabel())
    return r
for split in ("train","validation"):
    cov,pos,splen,nsp,qlen,plen=res[split]["_arrays"]; n=res[split]["n"]
    fig,ax=plt.subplots(1,4,figsize=(18,3.6))
    ax[0].hist(cov,bins=25,range=(0,1),color="#4C72B0"); ax[0].set_title("Span coverage of passage (chars)"); ax[0].set_xlabel("fraction of passage covered")
    ax[1].hist(np.clip(splen,0,60),bins=30,range=(0,60),color="#55A868"); ax[1].set_title("Span length (words)"); ax[1].set_xlabel("words (clipped at 60)")
    c=collections.Counter(np.clip(nsp,0,6)); ax[2].bar([str(k) if k<6 else "6+" for k in sorted(c)],[c[k] for k in sorted(c)],color="#C44E52"); ax[2].set_title("Spans per pair"); ax[2].set_xlabel("#spans (0 = empty)")
    ax[3].hist(pos,bins=20,range=(0,1),color="#8172B2"); ax[3].set_title("Relative span start position"); ax[3].set_xlabel("0 = passage start, 1 = end")
    for a,lab in zip(ax,("pairs","spans","pairs","spans")): a.set_ylabel(lab)
    for a,tot in zip(ax,(len(cov),len(splen),len(nsp),len(pos))): add_pct_axis(a,tot)
    fig.suptitle(f"MS MARCO {split} supervision, gemma-4-31B-it (XML, constrained, reasoning), n={n:,} pairs",y=1.03)
    fig.tight_layout(); fig.savefig(f"figs/msmarco_supervision/{split}_span_stats.png",dpi=130,bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3.4))
    ax[0].hist(qlen,bins=range(0,30),color="#4C72B0"); ax[0].set_title("Query length (words)"); ax[0].set_xlabel("words")
    ax[1].hist(np.clip(plen,0,200),bins=40,color="#55A868"); ax[1].set_title("Passage length (words)"); ax[1].set_xlabel("words (clipped at 200)")
    for a in ax: a.set_ylabel("pairs"); add_pct_axis(a,n)
    fig.suptitle(f"MS MARCO {split}, n={n:,} pairs",y=1.03)
    fig.tight_layout(); fig.savefig(f"figs/msmarco_supervision/{split}_length_stats.png",dpi=130,bbox_inches="tight"); plt.close(fig)
print("\nfigs written")
