"""
15_build_dashboard_v4.py  --  HarvestSight dashboard, "coffee-house glass box" edition, compact side-by-side layout (supersedes v1-v3)

Reads kharif_suitability.json (script 13) and kharif_year_robustness.json (script 14), checks them against
known results, and writes ONE self-contained harvestsight_dashboard.html (data embedded, no server, no libraries).
Computes nothing new. The only numbers not read from those two files are the model-comparison constants in
STATIC below (from scripts 11 and 12, as printed in their outputs), including the per-epoch training curves of
script 12 (5 runs x 20 epochs) used by the "Inside the model" replay.
"""
import json
from pathlib import Path

AUTHOR = "[Your name]"
KNOWN_TEST_RATES = {"Low": 0.177, "Medium": 0.289, "High": 0.349}
STATIC = {"rnn_pooled": 0.529, "rnn_kharif": 0.526, "rnn_kharif_ci": [0.472, 0.579],
          "index_test": 0.595, "index_all": 0.608,   # scripts 07 v3, 12, 11
          "train": [{'loss': [0.6487, 0.6418, 0.6384, 0.6359, 0.6322, 0.629, 0.6255, 0.621, 0.6199, 0.6163, 0.6133, 0.6087, 0.6054, 0.6048, 0.598, 0.5964, 0.5902, 0.5885, 0.5856, 0.5822], 'val': [0.605, 0.585, 0.621, 0.59, 0.596, 0.605, 0.659, 0.607, 0.689, 0.656, 0.67, 0.609, 0.677, 0.661, 0.592, 0.636, 0.644, 0.662, 0.651, 0.628], 'ckpt': 9, 'test': 0.52, 'lift': 1.34, 'ref': 0.513}, {'loss': [0.6495, 0.6421, 0.6373, 0.6346, 0.6333, 0.6275, 0.6253, 0.6209, 0.6178, 0.6139, 0.6118, 0.6051, 0.6024, 0.5986, 0.5963, 0.5929, 0.5881, 0.5841, 0.5831, 0.5797], 'val': [0.578, 0.572, 0.57, 0.593, 0.584, 0.601, 0.569, 0.568, 0.598, 0.599, 0.601, 0.606, 0.576, 0.595, 0.603, 0.589, 0.572, 0.592, 0.614, 0.561], 'ckpt': 19, 'test': 0.525, 'lift': 0.99, 'ref': 0.501}, {'loss': [0.6464, 0.6409, 0.6382, 0.6369, 0.6325, 0.6297, 0.6254, 0.6214, 0.6145, 0.6141, 0.6126, 0.6077, 0.6037, 0.5982, 0.5954, 0.5903, 0.5837, 0.5828, 0.5778, 0.5771], 'val': [0.626, 0.602, 0.611, 0.589, 0.604, 0.596, 0.569, 0.531, 0.611, 0.524, 0.461, 0.548, 0.519, 0.507, 0.437, 0.504, 0.515, 0.483, 0.519, 0.419], 'ckpt': 1, 'test': 0.54, 'lift': 1.46, 'ref': 0.49}, {'loss': [0.6494, 0.6401, 0.6355, 0.6332, 0.63, 0.6244, 0.6192, 0.6161, 0.6099, 0.6072, 0.6022, 0.599, 0.5943, 0.5909, 0.5851, 0.5844, 0.5806, 0.5801, 0.5779, 0.5738], 'val': [0.57, 0.609, 0.546, 0.535, 0.554, 0.564, 0.603, 0.622, 0.596, 0.62, 0.586, 0.604, 0.575, 0.633, 0.538, 0.542, 0.511, 0.501, 0.515, 0.483], 'ckpt': 14, 'test': 0.493, 'lift': 0.7, 'ref': 0.459}, {'loss': [0.6455, 0.6399, 0.637, 0.6353, 0.633, 0.6303, 0.6276, 0.6217, 0.6202, 0.6177, 0.6138, 0.6099, 0.6081, 0.6008, 0.5974, 0.5942, 0.5924, 0.5881, 0.585, 0.5814], 'val': [0.611, 0.583, 0.618, 0.524, 0.599, 0.616, 0.595, 0.64, 0.655, 0.595, 0.552, 0.614, 0.538, 0.481, 0.518, 0.513, 0.491, 0.494, 0.481, 0.47], 'ckpt': 9, 'test': 0.55, 'lift': 1.17, 'ref': 0.555}]}   # script 12 console output, per epoch


def find_input(name):
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for p in (here.parent / "cleaned_data" / name, here / name, Path.cwd() / name, Path("/content") / name):
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {name}. Put it next to this script, in ../cleaned_data, or /content.")


def check(s, y):
    ok = True
    for t, v in KNOWN_TEST_RATES.items():
        r = s["tier_reports"]["test"]["tiers"][t]["rate"]
        good = r is not None and abs(r - v) < 0.002
        ok &= good
        print(f"  test At-Risk rate, {t:<6} tier {r:.3f}  expected {v:.3f}  {'MATCH' if good else 'DIFFERS'}")
    for label, got, exp in (("eligible states", sum(1 for r in s["states"] if r["eligible"]), 24), ("years", len(y["years"]), 23)):
        good = got == exp
        ok &= good
        print(f"  {label}: {got}  expected {exp}  {'MATCH' if good else 'DIFFERS'}")
    print("  verdicts:", s.get("tier_verdict_test"), "/", y.get("verdict"))
    print("  " + ("Data matches the known results." if ok else "WARNING: data differs from known results; check the input files."))


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HarvestSight: Kharif moisture-risk tiers</title>
<style>
:root{--bg:#f2e8d5;--fg:#3b2a1f;--mut:#7b6452;--card:#fbf5e8;--line:#dac9aa;--wet:#3b7a86;--mid:#cdb792;--dry:#b4472a;--acc:#3b2a1f;--gold:#c08a33}
*{box-sizing:border-box}body{margin:0;color:var(--fg);font:15px/1.55 Georgia,'Iowan Old Style','Palatino Linotype',serif;background:var(--bg) radial-gradient(rgba(110,70,35,.08) 1px,transparent 1.6px) 0 0/15px 15px}
.w{max-width:1000px;margin:0 auto;padding:0 16px}.spec{height:10px;background:repeating-linear-gradient(90deg,var(--fg) 0 18px,var(--gold) 18px 36px,var(--dry) 36px 54px,var(--mid) 54px 72px)}
.mast{text-align:center;padding:22px 0 6px;border-bottom:3px double var(--fg)}.kick{font:700 11px/1 ui-monospace,Courier New,monospace;letter-spacing:.3em;text-transform:uppercase;color:var(--dry)}
h1{font:700 34px/1.1 Georgia,serif;margin:8px 0 4px;letter-spacing:.02em}h2{font:700 21px Georgia,serif;margin:14px 0 6px}h3{font:700 12px ui-monospace,Courier New,monospace;letter-spacing:.06em;text-transform:uppercase;margin:12px 0 4px;color:var(--mut)}
.mut{color:var(--mut)}.banner{border:2px dashed var(--mut);padding:8px 12px;margin:14px 0;color:var(--mut);background:rgba(255,255,255,.35)}
nav{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0;position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:2}
button{font:inherit}nav button,.chip{background:var(--card);color:var(--fg);border:2px solid var(--fg);border-radius:7px;padding:5px 13px;cursor:pointer;box-shadow:0 3px 0 var(--fg);transition:transform .12s,box-shadow .12s,background .2s}
nav button:hover,.chip:hover{transform:translateY(-2px);box-shadow:0 5px 0 var(--fg)}nav button:active,.chip:active{transform:translateY(3px);box-shadow:0 0 0 var(--fg)}
nav button[aria-selected=true],.chip[aria-pressed=true]{background:var(--fg);color:var(--card);box-shadow:0 3px 0 var(--gold)}
button:focus-visible,[tabindex]:focus-visible,input:focus-visible,summary:focus-visible,select:focus-visible{outline:3px solid var(--gold);outline-offset:2px}
.card{background:var(--card);border:2px solid var(--fg);border-radius:6px;padding:14px;margin:12px 0;box-shadow:5px 5px 0 rgba(59,42,31,.16)}.detail{border-left:8px solid var(--gold)}
.big{font:700 21px/1.35 Georgia,serif}.num{font:700 40px/1 Georgia,serif}.row{display:flex;gap:12px;flex-wrap:wrap}.row>div{flex:1;min-width:210px}.scroll{overflow-x:auto}
.chips{display:flex;gap:9px;flex-wrap:wrap;margin:9px 0 12px;align-items:center}
.v{display:inline-block;padding:2px 10px;border:2px solid var(--dry);color:var(--dry);border-radius:4px;font:700 12px ui-monospace,Courier New,monospace;letter-spacing:.06em;text-transform:uppercase;transform:rotate(-2deg);margin-right:10px}
.sw{display:inline-block;width:13px;height:13px;margin:0 5px 0 12px;vertical-align:-1px;border:1.5px solid var(--fg);border-radius:3px}
code{display:block;background:rgba(59,42,31,.07);border:1px dashed var(--mut);padding:8px 10px;font:13px ui-monospace,Courier New,monospace;margin:8px 0}
details.expl{margin:6px 0;color:var(--mut)}details.expl summary{cursor:pointer;color:var(--dry);font-weight:700}
svg{width:100%;height:auto;display:block}svg text{fill:var(--fg);font:11px ui-monospace,Courier New,monospace}svg .ax{fill:var(--mut)}svg .grid{stroke:var(--line)}svg [tabindex],svg rect[role],svg circle[role]{cursor:pointer}
.t-Low{fill:var(--wet)}.t-Medium{fill:var(--mid)}.t-High{fill:var(--dry)}.dim{opacity:.18}.sel{stroke:var(--fg);stroke-width:3}
table.g{border-collapse:collapse;font-size:11px}table.g td,table.g th{width:22px;height:20px;padding:0;text-align:center;border:1px solid var(--card)}
table.g th.l{width:auto;text-align:left;padding:0 8px 0 0;white-space:nowrap;font-weight:400;cursor:pointer}table.g th.y,table.g td{cursor:pointer}
td.c-Low{background:var(--wet)}td.c-Medium{background:var(--mid)}td.c-High{background:var(--dry)}td.c-none{background:repeating-linear-gradient(45deg,var(--line) 0 3px,transparent 3px 6px);cursor:default}
td:hover:not(.c-none){outline:2px solid var(--fg)}th.s-val,th.s-test{border-bottom:3px solid var(--fg)}th.s-test{border-bottom-style:double}tr.hl th.l{font-weight:700;text-decoration:underline}
table.d{border-collapse:collapse;width:100%}table.d td,table.d th{padding:3px 8px;border-bottom:1px solid var(--line);text-align:right}table.d th:first-child,table.d td:first-child{text-align:left}
input[type=range]{flex:1;min-width:160px;accent-color:var(--dry)}select{font:inherit;background:var(--card);color:var(--fg);border:2px solid var(--fg);border-radius:5px;padding:3px}
.dots{display:flex;gap:7px;margin:10px 0}.dots i{width:11px;height:11px;border-radius:50%;border:2px solid var(--fg)}.dots i.on{background:var(--dry)}
footer{border-top:3px double var(--fg);margin-top:8px;padding-top:6px}
@media(prefers-reduced-motion:no-preference){#app>*{animation:rise .45s ease both}svg rect[class^=t-]{transform-box:fill-box;transform-origin:50% 100%;animation:grow .7s cubic-bezier(.2,.8,.2,1) both}.dots i{transition:background .25s}.card{transition:transform .2s}.card:hover{transform:translateY(-1px)}
@keyframes rise{from{opacity:0;transform:translateY(10px)}}@keyframes grow{from{transform:scaleY(0)}}}
.split{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(300px,1fr);gap:14px;align-items:start}.side{position:sticky;top:56px;max-height:calc(100vh - 66px);overflow:auto;padding:0 6px 6px 0}
.mast{padding:10px 0 4px}h1{font-size:26px;margin:4px 0 2px}.banner{margin:8px 0;padding:5px 10px;font-size:13px}nav{margin:4px 0;padding:6px 0;gap:6px}nav button{padding:3px 11px}
.card{padding:10px;margin:8px 0}h2{font-size:18px;margin:6px 0 4px}.big{font-size:18px;margin:6px 0}.num{font-size:34px}.stack h3{margin:6px 0 2px}.chips{margin:6px 0 8px}
table.g td,table.g th{height:15px;width:20px;font-size:10px}table.g th.l{font-size:10.5px}footer p{font-size:13px}
@media(max-width:860px){.split{grid-template-columns:1fr}.side{position:static;max-height:none;overflow:visible;padding:0}}
.mast{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap;padding:8px 0 4px}h1{font-size:22px;margin:0}.kick{font-size:10px}
.banner{padding:3px 10px;margin:6px 0;font-size:12.5px}nav{margin:2px 0;padding:5px 0}.card>input[type=range]{width:100%}.big{font-size:17px}.step p{margin:4px 0}
</style></head><body><div class="spec"></div>
<div class="w"><div class="mast"><h1>Kharif Moisture-Risk Tiers</h1><div class="kick">HarvestSight &middot; India 1997&ndash;2019 &middot; SDG 2 &amp; 13 &middot; a glass-box study</div></div>
<div class="banner" role="note"><b>Historical pattern, not a forecast.</b> Rates are observed shares, not probabilities. Press anything pressable.</div>
<nav role="tablist" id="tabs"></nav><main id="app"></main>
<footer><details class="expl"><summary>Notes and limits</summary><p class="mut">Built from kharif_suitability.json (script 13) and kharif_year_robustness.json (script 14); model-comparison figures from scripts 11 and 12. The index uses season-mean rainfall, root-zone wetness and humidity anomalies against each state&rsquo;s training-year climatology; nothing is fitted to labels. The features were chosen by an earlier diagnostic that saw all splits, so this is not a clean out-of-sample confirmation. State results are exploratory. Irrigation as an explanation for flat states is a hypothesis, not a finding. One weather point per state; yields are state-level. Author: __AUTHOR__</p></details></footer></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const NS="http://www.w3.org/2000/svg",$=s=>document.querySelector(s);
const h=(t,a,...c)=>{const e=document.createElement(t);for(const k in a||{})k==="class"?e.className=a[k]:e.setAttribute(k,a[k]);c.flat().forEach(x=>e.append(x));return e};
const sv=(t,a,...c)=>{const e=document.createElementNS(NS,t);for(const k in a||{})e.setAttribute(k,a[k]);c.flat().forEach(x=>e.append(x));return e};
const pct=x=>x==null?"n/a":(100*x).toFixed(1)+"%",f3=x=>x==null?"n/a":x.toFixed(3),sg=x=>x==null?"n/a":(x>=0?"+":"")+x.toFixed(3);
const TN={Low:"Low risk (wet)",Medium:"Medium",High:"High risk (dry)"},TIERS=["Low","Medium","High"],SPL=["train","val","test"],CL={Low:"wet",Medium:"mid",High:"dry"},SN={train:"Train 1997–2014",val:"Validation 2015–16",test:"Test 2017–19"};
let D,K,tab="Tour";
const key=()=>h("div",{},TIERS.flatMap(k=>[h("span",{class:"sw",style:`background:var(--${CL[k]})`}),TN[k]]));
const chips=(o,cur,cb)=>h("div",{class:"chips"},o.map(([v,l])=>{const b=h("button",{class:"chip","aria-pressed":String(v===cur)},l);b.onclick=()=>cb(v);return b}));
const expl=t=>h("details",{class:"expl"},h("summary",{},"What am I looking at?"),h("p",{},t));
const split=(l,r)=>h("div",{class:"split"},h("div",{},l),h("div",{class:"side"},r));
const act=(g,f)=>{g.onclick=f;g.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();f()}}};
function load(){const e=$("#data").textContent.trim();if(e&&e!=="null"){D=JSON.parse(e);K=D.k;start();return}
 const app=$("#app");app.append(h("div",{class:"card"},h("p",{},"No data is embedded. Choose both result files:"),[["s","kharif_suitability.json"],["y","kharif_year_robustness.json"]].map(([k,n])=>h("p",{},n+" ",Object.assign(h("input",{type:"file",accept:".json"}),{onchange:async ev=>{window["_"+k]=JSON.parse(await ev.target.files[0].text());if(window._s&&window._y){D={s:window._s,y:window._y,k:{rnn_pooled:.529,rnn_kharif:.526,rnn_kharif_ci:[.472,.579],index_test:.595,index_all:.608,train:null}};K=D.k;app.replaceChildren();start()}}})))))}
const V={Tour:story,"Inside the model":inside,Tiers:overview,Map:grid,States:states,Years:years};
function start(){const nav=$("#tabs");Object.keys(V).forEach(n=>{const b=h("button",{role:"tab","data-n":n},n);b.onclick=()=>show(n);nav.append(b)});show("Tour")}
function show(n){clearInterval(window._tm);window._tm=null;tab=n;[...$("#tabs").children].forEach(b=>b.setAttribute("aria-selected",String(b.dataset.n===n)));$("#app").replaceChildren(V[n]())}

function gauge(items,focus){const w=640,H=104,x=v=>30+(w-60)*(v-.4)/.3,s=sv("svg",{viewBox:`0 0 ${w} ${H}`,role:"img","aria-label":"Held-out AUC on a scale from 0.4 to 0.7"});
 s.append(sv("line",{x1:x(.4),x2:x(.7),y1:58,y2:58,stroke:"var(--line)","stroke-width":6}),sv("line",{x1:x(.5),x2:x(.5),y1:40,y2:76,stroke:"currentColor","stroke-dasharray":"3 3"}),sv("text",{x:x(.5),y:14,"text-anchor":"middle",class:"ax"},"chance = 0.5"));
 items.forEach((it,i)=>{const up=i%2==0;if(it.ci)s.append(sv("line",{x1:x(it.ci[0]),x2:x(it.ci[1]),y1:58,y2:58,stroke:it.c,"stroke-width":4,opacity:.45}));
  s.append(sv("circle",{cx:x(it.v),cy:58,r:8,fill:it.c,class:focus===i?"sel":""},sv("title",{},it.l+" "+f3(it.v))),sv("text",{x:x(it.v),y:up?38:84,"text-anchor":"middle"},it.l+" "+f3(it.v)))});return s}
const GA=()=>[{v:K.rnn_pooled,l:"RNN, all seasons",c:"var(--mid)"},{v:K.rnn_kharif,l:"RNN, Kharif only",c:"var(--mid)",ci:K.rnn_kharif_ci}];
const GB=()=>GA().concat([{v:K.index_test,l:"Moisture rule",c:"var(--dry)"}]);

function story(){const T=D.s.tier_reports.test.tiers,S=D.y.sign_test.all,B=D.y.bootstrap.pooled_high_minus_low;let i=0;const box=h("div",{});
 const SL=[
  ()=>[h("h2",{},"The question"),h("p",{class:"big"},"Can weekly weather tell us which Kharif seasons put crops at risk?"),h("p",{},"We used 23 years of data for Indian states. A crop is At-Risk when its yield falls well below its own trend. We first tried a learned sequence model (an RNN), then a simple rule fixed in advance."),h("p",{class:"mut"},"Use Next, or jump to any tab.")],
  ()=>[h("h2",{},"The learned model did not beat chance"),h("p",{},"Scale: 0.5 is a coin flip, 1.0 is perfect. Both RNNs stay close to 0.5 on years they had never seen."),gauge(GA()),h("p",{class:"mut"},"Even restricted to Kharif and averaged over five random starts, the model overfits: it looks good on validation years and falls back on test.")],
  ()=>[h("h2",{},"A simple rule did better"),h("p",{},"The rule: how much wetter or drier than usual was the season, using rainfall, soil wetness and humidity, compared with each state’s own history? Nothing in it is fitted to the labels."),gauge(GB(),2),h("p",{class:"mut"},"Held-out Kharif AUC "+f3(K.index_test)+"; over all years "+f3(K.index_all)+". Modest, but clearly above chance and fully explainable.")],
  ()=>[h("h2",{},"Three tiers anyone can read"),h("p",{},"Seasons are split into wet, medium and dry by cut points set on training years. In the test years:"),h("div",{class:"row"},TIERS.map(k=>h("div",{class:"card"},h("div",{class:"num","data-v":String(T[k].rate),style:`color:var(--${CL[k]})`},pct(T[k].rate)),h("div",{},TN[k]),h("div",{class:"mut"},T[k].n+" crop records")))),h("p",{class:"mut"},"Share of crop records that were At-Risk. Open the Tiers tab to click into each bar.")],
  ()=>[h("h2",{},"It holds year after year"),h("p",{},`The dry tier had the higher At-Risk rate in ${S.high_above_low} of ${S.usable_years} usable years (sign test p = ${S.sign_p_one_sided.toFixed(3)}). Pooled difference ${sg(B.point)} [${sg(B.ci95[0])}, ${sg(B.ci95[1])}], resampling whole years.`),h("p",{class:"mut"},"Five years ran the other way, so this is a tendency, not a law. The Years tab shows each one.")],
  ()=>[h("h2",{},"What this does not show"),h("ul",{},h("li",{},"It is not a forecast, and the rates are not probabilities."),h("li",{},"The three weather features were picked by an earlier look at all the data, so this is not a clean out-of-sample test."),h("li",{},"It works in Kharif; the relationship is weak in Rabi and reverses in Summer."),h("li",{},"It is uneven across states, and why is untested. Irrigation is a hypothesis."),h("li",{},"One weather point per state and state-level yields limit the detail."))]];
 const draw=()=>{const b=h("button",{class:"chip"},"Back"),n=h("button",{class:"chip"},i<SL.length-1?"Next":"Start over");b.onclick=()=>{i=Math.max(0,i-1);draw()};n.onclick=()=>{i=i<SL.length-1?i+1:0;draw()};
  box.replaceChildren(h("div",{class:"card"},SL[i](),h("div",{class:"dots","aria-label":`Step ${i+1} of ${SL.length}`},SL.map((_,j)=>h("i",{class:j===i?"on":""}))),h("div",{class:"chips"},b,n)));animateNums(box)};draw();return box}

function overview(){const T=D.s.tier_reports,C=D.s.tier_cuts,w=640,H=222,m={l:42,r:8,t:12,b:46};let pick=["test","High"];const out=h("div",{}),dc=h("div",{});
 const top=Math.max(.6,...SPL.flatMap(s=>TIERS.map(k=>(T[s].tiers[k].ci95||[0,T[s].tiers[k].rate])[1])))*1.05,y=v=>m.t+(H-m.t-m.b)*(1-v/top);
 const info=()=>{const[s,k]=pick,d=T[s].tiers[k],ci=d.ci95;dc.replaceChildren(h("div",{class:"card detail"},h("h3",{},`${SN[s]}, ${TN[k]}`),h("p",{},`${d.n_pos} of ${d.n} crop records were At-Risk: `,h("b",{},pct(d.rate)),ci?` (95% interval ${pct(ci[0])} to ${pct(ci[1])}).`:"."),h("p",{class:"mut"},`That is ${d.lift_vs_base==null?"n/a":d.lift_vs_base.toFixed(2)} times the overall rate in these years (${pct(T[s].base_rate)}). Tier share of rows: ${Math.round(d.share*100)}%.`)))};
 const svg=sv("svg",{viewBox:`0 0 ${w} ${H}`,role:"group","aria-label":"Observed At-Risk rate by moisture tier. Select a bar for details."});
 for(let v=0;v<=top;v+=.1)svg.append(sv("line",{x1:m.l,x2:w-m.r,y1:y(v),y2:y(v),class:"grid"}),sv("text",{x:m.l-5,y:y(v)+4,class:"ax","text-anchor":"end"},Math.round(v*100)+"%"));
 const gw=(w-m.l-m.r)/3,bw=gw/4.2,rs={};
 SPL.forEach((s,i)=>{const g0=m.l+i*gw+(gw-3*bw)/2;TIERS.forEach((k,j)=>{const d=T[s].tiers[k],x=g0+j*bw,r=sv("rect",{x:x+2,y:y(d.rate),width:bw-4,height:y(0)-y(d.rate),class:"t-"+k,tabindex:"0",role:"button","aria-label":`${SN[s]}, ${TN[k]}: ${pct(d.rate)}`},sv("title",{},`${SN[s]}, ${TN[k]}: ${pct(d.rate)}`));
  rs[s+k]=r;act(r,()=>{pick=[s,k];Object.values(rs).forEach(q=>q.classList.remove("sel"));r.classList.add("sel");info()});svg.append(r);
  if(d.ci95)svg.append(sv("line",{x1:x+bw/2,x2:x+bw/2,y1:y(d.ci95[0]),y2:y(d.ci95[1]),stroke:"currentColor","stroke-width":1.5}));
  svg.append(sv("text",{x:x+bw/2,y:y(d.ci95?d.ci95[1]:d.rate)-4,"text-anchor":"middle"},Math.round(d.rate*100)+"%"))});
  svg.append(sv("text",{x:m.l+i*gw+gw/2,y:H-24,"text-anchor":"middle"},SN[s]),sv("text",{x:m.l+i*gw+gw/2,y:H-10,"text-anchor":"middle",class:"ax"},T[s].n+" rows"))});
 rs.testHigh.classList.add("sel");
 const t=T.test.tiers,dd=T.test.diff_ci95;
 const rd=h("div",{}),sl=h("input",{type:"range",min:"-2",max:"2",step:"0.01",value:"0","aria-label":"Index value"});
 const upd=()=>{const v=+sl.value,k=v<=C.lower?"Low":v<=C.upper?"Medium":"High";rd.replaceChildren(h("p",{},`Index value ${sg(v)} falls in the `,h("b",{style:`color:var(--${CL[k]})`},TN[k]),` tier. In past seasons of this tier, the share of crop records At-Risk was `,SPL.map((s,i)=>h("span",{},(i?", ":"")+pct(T[s].tiers[k].rate)+" ("+s+")")),". "),h("p",{class:"mut"},"This reads the historical table; it does not predict a season."))};sl.oninput=upd;upd();
 out.append(h("p",{class:"big"},`In the test years, dry Kharif seasons went with At-Risk crops ${pct(t.High.rate)} of the time, against ${pct(t.Low.rate)} in wet seasons (about ${(t.High.rate/t.Low.rate).toFixed(1)} times as often). `,h("span",{class:"v"},"Tiers: "+D.s.tier_verdict_test),h("span",{class:"v"},"Years: "+D.y.verdict)),
  split(h("div",{class:"card"},h("h2",{},"At-Risk rate by tier (press a bar)"),svg,key(),h("p",{class:"mut"},`Whiskers: 95% intervals. Test dry minus wet: ${sg(T.test.diff_high_low)}`+(dd?` [${sg(dd[0])}, ${sg(dd[1])}]`:"")+`. Test years ran wetter than training: ${Math.round(T.test.tiers.Low.share*100)}% of test rows in the wet tier vs ${Math.round(T.train.tiers.Low.share*100)}%.`)),
   h("div",{},dc,h("div",{class:"card"},h("h2",{},"Try the cut points"),h("p",{class:"mut"},`Cut points: ${f3(C.lower)} and ${f3(C.upper)} (higher = drier).`),sl,rd),expl("Every crop record in a Kharif season gets a wet, medium or dry label from one number: how far the state’s rainfall, soil wetness and humidity ran from its own usual. Bars show how often records in each group were At-Risk."))));info();return out}

function stateDetail(r){const rows=D.s.state_year.filter(q=>q.state===r.state).sort((a,b)=>a.year-b.year),w=380,H=130,m={l:36,r:6,t:10,b:22},mx=Math.max(.3,...rows.map(q=>q.at_risk_share))*1.1,y=v=>m.t+(H-m.t-m.b)*(1-v/mx),bw=(w-m.l-m.r)/Math.max(1,rows.length);
 const s=sv("svg",{viewBox:`0 0 ${w} ${H}`,role:"img","aria-label":`${r.state}: At-Risk share by year, colored by moisture tier`});
 [0,mx/2,mx].forEach(v=>s.append(sv("line",{x1:m.l,x2:w-m.r,y1:y(v),y2:y(v),class:"grid"}),sv("text",{x:m.l-4,y:y(v)+4,"text-anchor":"end",class:"ax"},Math.round(v*100)+"%")));
 rows.forEach((q,i)=>{const x=m.l+i*bw;s.append(sv("rect",{x:x+1,y:y(q.at_risk_share),width:bw-2,height:y(0)-y(q.at_risk_share),class:"t-"+q.tier},sv("title",{},`${q.year} (${q.split}): ${TN[q.tier]}, ${pct(q.at_risk_share)} At-Risk of ${q.n_records}`)));if(i%2==0)s.append(sv("text",{x:x+bw/2,y:H-8,"text-anchor":"middle",class:"ax"},String(q.year).slice(2)))});
 return h("div",{class:"card detail"},h("h2",{},r.state),h("p",{},r.eligible?`Moisture index AUC ${f3(r.auc)} [${f3(r.ci_lo)}, ${f3(r.ci_hi)}] over ${r.n_years} years and ${r.n} crop records (${pct(r.base_rate)} At-Risk): ${r.flag==="SENSITIVE"?"the index tracks risk here.":r.flag==="REVERSED"?"the relationship runs the opposite way (few records).":"no clear relationship."}`:"Too little Kharif data for a reliable estimate."),s,key(),
  h("p",{class:"mut"},"Bars: At-Risk share of crop records that year; color: the state’s moisture tier. If dry (orange) bars tend to be taller, the index works for this state."),
  h("details",{class:"expl"},h("summary",{},"Show table"),h("div",{class:"scroll"},h("table",{class:"d"},h("tr",{},["Year","Split","Tier","Index (higher = drier)","At-Risk"].map(t=>h("th",{},t))),rows.map(q=>h("tr",{},h("td",{},String(q.year)),h("td",{},q.split),h("td",{},TN[q.tier]),h("td",{},sg(q.index_risk_score)),h("td",{},pct(q.at_risk_share))))))))}
function yearDetail(r){const d=r.diff;return h("div",{class:"card detail"},h("h3",{},`${r.year} (${r.split})`),h("p",{},`National mean index ${sg(r.national_index)} (higher = drier); ${pct(r.national_at_risk)} of ${r.n} Kharif crop records At-Risk.`),
  h("p",{},d==null?"One tier had too few rows this year to compare.":`Dry tier ${pct(r.rate_high)} (${r.n_high} rows) vs wet tier ${pct(r.rate_low)} (${r.n_low} rows): ${sg(d)}, ${d>0?"the dry tier was riskier.":"the wet tier was riskier."}`))}

function grid(){const sy=D.s.state_year,ys=[...new Set(sy.map(r=>r.year))].sort(),ss=[...new Set(sy.map(r=>r.state))].sort(),M={},sp={},st={f:"all",t:new Set(TIERS),row:null};sy.forEach(r=>{M[r.state+"|"+r.year]=r;sp[r.year]=r.split});
 const out=h("div",{}),dc=h("div",{}),host=h("div",{class:"card scroll"}),ctl=h("div",{});
 const draw=()=>{const tb=h("table",{class:"g"},h("tr",{},h("th",{class:"l"},""),ys.map(y=>{const t=h("th",{class:"y s-"+sp[y],tabindex:"0",title:y+" ("+sp[y]+")"},String(y).slice(2));act(t,()=>dc.replaceChildren(yearDetail(D.y.years.find(q=>q.year===y))));return t})));
  ss.forEach(s=>{const th=h("th",{class:"l",scope:"row",tabindex:"0"},s);act(th,()=>{st.row=s;draw();const r=D.s.states.find(q=>q.state===s);dc.replaceChildren(stateDetail(r))});
   tb.append(h("tr",{class:st.row===s?"hl":""},th,ys.map(y=>{const r=M[s+"|"+y];if(!r)return h("td",{class:"c-none",title:`${s} ${y}: no Kharif data`});
    const on=(st.f==="all"||r.split===st.f)&&st.t.has(r.tier),c=h("td",{class:"c-"+r.tier+(on?"":" dim"),tabindex:"0",title:`${s} ${y} (${r.split}): ${TN[r.tier]}, ${pct(r.at_risk_share)} of ${r.n_records} crop records At-Risk`});
    act(c,()=>dc.replaceChildren(h("div",{class:"card detail"},h("h3",{},`${s}, ${y} (${r.split})`),h("p",{},`${TN[r.tier]} season (index ${sg(r.index_risk_score)}, higher = drier). ${pct(r.at_risk_share)} of ${r.n_records} crop records were At-Risk.`))));return c}))) });
  host.replaceChildren(tb);
  ctl.replaceChildren(chips([["all","All years"],["train","Train"],["val","Validation"],["test","Test"]],st.f,v=>{st.f=v;draw()}),
   h("div",{class:"chips"},TIERS.map(k=>{const b=h("button",{class:"chip","aria-pressed":String(st.t.has(k))},TN[k]);b.onclick=()=>{st.t.has(k)?st.t.delete(k):st.t.add(k);if(!st.t.size)TIERS.forEach(q=>st.t.add(q));draw()};return b})))};
 dc.append(h("div",{class:"card"},h("p",{class:"mut"},"Press a square, a year heading or a state name.")));
 out.append(split(h("div",{},h("h2",{},"Season moisture tier, by state and year"),host),h("div",{},ctl,dc,expl("Each square is one state in one year, colored by how wet or dry that Kharif season was for the state. Press a square for its numbers, a year heading for the dry-vs-wet comparison, a state name for its history. Hatched = no data; underlined years are validation, double-underlined are test."))));draw();return out}

function states(){const st={f:"all",s:"auc",sel:null},out=h("div",{}),host=h("div",{class:"card"}),dc=h("div",{}),ctl=h("div",{});
 const col=f=>f==="SENSITIVE"?"var(--wet)":f==="REVERSED"?"var(--dry)":"var(--mid)",S=D.s.state_summary,all=D.s.states.filter(r=>r.eligible);
 const draw=()=>{let el=all.filter(r=>st.f==="all"||r.flag===st.f);el=st.s==="auc"?el.sort((a,b)=>b.auc-a.auc):el.sort((a,b)=>a.state.localeCompare(b.state));
  const rh=17,w=640,m={l:130,r:14,t:20,b:8},H=m.t+m.b+rh*Math.max(1,el.length),x=v=>m.l+(w-m.l-m.r)*(v-.1)/.9,svg=sv("svg",{viewBox:`0 0 ${w} ${H}`,role:"group","aria-label":"AUC of the moisture index by state with 95% intervals. Select a state for its history."});
  for(let v=.2;v<=1;v+=.2)svg.append(sv("line",{x1:x(v),x2:x(v),y1:m.t-6,y2:H-m.b,class:"grid"}),sv("text",{x:x(v),y:12,"text-anchor":"middle",class:"ax"},v.toFixed(1)));
  svg.append(sv("line",{x1:x(.5),x2:x(.5),y1:m.t-6,y2:H-m.b,stroke:"currentColor","stroke-dasharray":"4 3"}));
  el.forEach((r,i)=>{const y=m.t+i*rh+rh/2,g=sv("g",{tabindex:"0",role:"button","aria-label":`${r.state}: AUC ${f3(r.auc)}, ${r.flag}`,opacity:st.sel&&st.sel!==r.state?.45:1},sv("rect",{x:0,y:y-rh/2,width:w,height:rh,fill:"transparent"}),
   sv("text",{x:m.l-8,y:y+4,"text-anchor":"end"},r.state),sv("line",{x1:x(r.ci_lo),x2:x(r.ci_hi),y1:y,y2:y,stroke:col(r.flag),"stroke-width":2}),sv("circle",{cx:x(r.auc),cy:y,r:5,fill:col(r.flag)}),sv("title",{},`${r.state}: AUC ${f3(r.auc)} [${f3(r.ci_lo)}, ${f3(r.ci_hi)}], ${r.n} rows, ${r.n_years} years`));
   act(g,()=>{st.sel=r.state;draw();dc.replaceChildren(stateDetail(r))});svg.append(g)});
  host.replaceChildren(svg,h("p",{class:"mut"},"Dot = all-years AUC, line = 95% interval over years, dashed = chance (0.5). Blue sensitive, orange reversed, gray unclear."));
  ctl.replaceChildren(chips([["all","All"],["SENSITIVE","Sensitive"],["-","Unclear"],["REVERSED","Reversed"]],st.f,v=>{st.f=v;draw()}),h("label",{class:"mut"},"Sort by ",(()=>{const s=h("select",{"aria-label":"Sort states"},h("option",{value:"auc"},"AUC"),h("option",{value:"name"},"Name"));s.value=st.s;s.onchange=()=>{st.s=s.value;draw()};return s})()))};
 const thin=D.s.states.filter(r=>!r.eligible).map(r=>r.state).join(", ");
 dc.append(h("div",{class:"card"},h("p",{class:"mut"},"Press a state to see its history.")));
 out.append(split(h("div",{},h("h2",{},"Where the moisture index tracks risk"),host),h("div",{},h("p",{class:"mut"},`Exploratory. ${S.auc_above_half} of ${S.eligible} states are above 0.5 (a coin flip); ${S.sensitive} flagged sensitive (about ${S.expected_false_positives.toFixed(1)} expected by chance), ${S.reversed} reversed.`+(thin?` Too little data: ${thin}.`:"")),ctl,dc,expl("AUC asks: pick one At-Risk and one other record at random; how often does the index rank the At-Risk one as drier? A flat state means the index says little there, not that weather does not matter. Top-ranked states are biased upward, picked from many estimates."))));draw();return out}

function years(){const Y=D.y.years,S=D.y.sign_test.all,B=D.y.bootstrap.pooled_high_minus_low,R=D.y.national_rank_correlation,w=640,H=210,m={l:42,r:8,t:12,b:30},out=h("div",{}),dc=h("div",{});
 const ds=Y.map(r=>r.diff).filter(v=>v!=null),lo=Math.min(-.1,...ds)-.03,hi=Math.max(.1,...ds)+.03,y=v=>m.t+(H-m.t-m.b)*(1-(v-lo)/(hi-lo)),bw=(w-m.l-m.r)/Y.length,svg=sv("svg",{viewBox:`0 0 ${w} ${H}`,role:"group","aria-label":"Dry-tier minus wet-tier At-Risk rate for each year. Select a bar for details."});
 for(let v=Math.ceil(lo*10)/10;v<=hi;v+=.1)svg.append(sv("line",{x1:m.l,x2:w-m.r,y1:y(v),y2:y(v),class:"grid"}),sv("text",{x:m.l-5,y:y(v)+4,"text-anchor":"end",class:"ax"},(v>=0?"+":"")+v.toFixed(1)));
 svg.append(sv("line",{x1:m.l,x2:w-m.r,y1:y(0),y2:y(0),stroke:"currentColor"}));
 Y.forEach((r,i)=>{const x=m.l+i*bw,e=r.diff==null?sv("text",{x:x+bw/2,y:y(0)-4,"text-anchor":"middle",class:"ax",tabindex:"0",role:"button","aria-label":r.year+": not usable"},"n/u"):sv("rect",{x:x+1.5,y:Math.min(y(0),y(r.diff)),width:bw-3,height:Math.max(1,Math.abs(y(r.diff)-y(0))),class:r.diff>0?"t-High":"t-Low",tabindex:"0",role:"button","aria-label":`${r.year}: ${sg(r.diff)}`},sv("title",{},`${r.year}: ${sg(r.diff)}`));
  act(e,()=>dc.replaceChildren(yearDetail(r)));svg.append(e);if(i%2==0)svg.append(sv("text",{x:x+bw/2,y:H-12,"text-anchor":"middle",class:"ax"},String(r.year).slice(2)))});
 const sw=380,sh=150,sm={l:46,r:10,t:10,b:36},ax=Y.map(r=>r.national_index),ay=Y.map(r=>r.national_at_risk),x0=Math.min(...ax)-.1,x1=Math.max(...ax)+.1,y0=Math.min(...ay)-.03,y1=Math.max(...ay)+.03,X=v=>sm.l+(sw-sm.l-sm.r)*(v-x0)/(x1-x0),Y2=v=>sm.t+(sh-sm.t-sm.b)*(1-(v-y0)/(y1-y0)),sc=sv("svg",{viewBox:`0 0 ${sw} ${sh}`,role:"group","aria-label":"National dryness against national At-Risk share, one point per year. Select a point for details."});
 sc.append(sv("line",{x1:sm.l,x2:sw-sm.r,y1:sh-sm.b,y2:sh-sm.b,class:"grid"}),sv("line",{x1:sm.l,x2:sm.l,y1:sm.t,y2:sh-sm.b,class:"grid"}),sv("text",{x:sw/2,y:sh-8,"text-anchor":"middle",class:"ax"},"National mean index (drier →)"),sv("text",{x:12,y:sh/2,class:"ax",transform:`rotate(-90 12 ${sh/2})`,"text-anchor":"middle"},"National At-Risk share"));
 Y.forEach(r=>{const c=sv("circle",{cx:X(r.national_index),cy:Y2(r.national_at_risk),r:6,class:r.split==="test"?"t-High":"t-Low",stroke:"var(--fg)","stroke-width":r.split==="test"?1.5:0,opacity:.85,tabindex:"0",role:"button","aria-label":`${r.year}: index ${sg(r.national_index)}`},sv("title",{},`${r.year} (${r.split}): index ${sg(r.national_index)}, At-Risk ${pct(r.national_at_risk)}`));act(c,()=>dc.replaceChildren(yearDetail(r)));sc.append(c)});
 dc.append(yearDetail(Y[Y.length-1]));
 out.append(h("h2",{},"Does the pattern hold year by year?"),split(h("div",{class:"card"},svg,h("p",{class:"mut"},"One bar per year: dry-tier minus wet-tier At-Risk rate, cut points fixed from training years. Orange = dry tier riskier, blue = wet tier riskier, n/u = a tier had under "+D.y.min_tier_rows+" rows. Press a bar or point for that year; years are not fully independent.")),h("div",{},h("p",{},`The dry tier had the higher At-Risk rate in ${S.high_above_low} of ${S.usable_years} usable years (sign test p = ${S.sign_p_one_sided.toFixed(3)}). Pooled dry-minus-wet ${sg(B.point)} [${sg(B.ci95[0])}, ${sg(B.ci95[1])}], resampling whole years.`),dc,h("div",{class:"card"},sc,h("p",{class:"mut"},`Each point is a year (test outlined): national dryness vs At-Risk share. Spearman rho ${sg(R.spearman_rho)}, permutation p = ${R.perm_p_one_sided.toFixed(3)}.`)))));return out}
function animateNums(root){if(!root.querySelectorAll||matchMedia("(prefers-reduced-motion:reduce)").matches)return;root.querySelectorAll(".num[data-v]").forEach(el=>{const v=+el.dataset.v,t0=performance.now();const f=t=>{const p=Math.min(1,(t-t0)/900);el.textContent=pct(v*(1-Math.pow(1-p,3)));if(p<1)requestAnimationFrame(f)};requestAnimationFrame(f)})}
function replay(){const S=K.train;if(!S)return h("div",{class:"card"},h("p",{class:"mut"},"Training curves are not embedded in this file."));let si=0,e=1;
 const box=h("div",{class:"card"}),ca=h("div",{}),ro=h("div",{}),sc=h("div",{}),sl=h("input",{type:"range",min:"1",max:"20",value:"1","aria-label":"Training epoch"}),pb=h("button",{class:"chip"},"Play"),w=640,H=112,m={l:40,r:8,t:8,b:20};
 const stop=()=>{clearInterval(window._tm);window._tm=null};
 const chart=(a,lo,hi,col,lab,ref,mark)=>{const x=i=>m.l+(w-m.l-m.r)*i/19,y=v=>m.t+(H-m.t-m.b)*(1-(v-lo)/(hi-lo)),s=sv("svg",{viewBox:`0 0 ${w} ${H}`,role:"img","aria-label":lab});
  [lo,(lo+hi)/2,hi].forEach(v=>s.append(sv("line",{x1:m.l,x2:w-m.r,y1:y(v),y2:y(v),class:"grid"}),sv("text",{x:m.l-4,y:y(v)+4,"text-anchor":"end",class:"ax"},v.toFixed(2))));
  for(let i=0;i<20;i+=5)s.append(sv("text",{x:x(i),y:H-6,"text-anchor":"middle",class:"ax"},String(i+1)));
  if(ref!=null)s.append(sv("line",{x1:m.l,x2:w-m.r,y1:y(ref),y2:y(ref),stroke:"currentColor","stroke-dasharray":"4 3"}),sv("text",{x:w-m.r,y:y(ref)-3,"text-anchor":"end",class:"ax"},"chance"));
  s.append(sv("polyline",{points:a.slice(0,e).map((v,i)=>x(i)+","+y(v)).join(" "),fill:"none",stroke:col,"stroke-width":2.5}),sv("circle",{cx:x(e-1),cy:y(a[e-1]),r:5,fill:col}));
  if(mark!=null&&mark<e)s.append(sv("circle",{cx:x(mark),cy:y(a[mark]),r:9,fill:"none",stroke:"var(--gold)","stroke-width":3}));return s};
 const upd=()=>{const d=S[si];const b=d.val.slice(0,e).indexOf(Math.max(...d.val.slice(0,e))),done=e===20;
  ca.replaceChildren(h("div",{class:"stack"},h("div",{},h("h3",{},"Training loss: fit to training seasons"),chart(d.loss,.57,.66,"var(--wet)","Training loss by epoch")),h("div",{},h("h3",{},"Validation AUC: does it carry over?"),chart(d.val,.4,.72,"var(--dry)","Validation AUC by epoch",.5,b))));
  ro.replaceChildren(h("p",{class:"big"},`Epoch ${e}: loss ${d.loss[e-1].toFixed(4)}, validation AUC ${d.val[e-1].toFixed(3)}`),
   h("p",{},done?`Saved model: epoch ${d.ckpt} (validation ${d.val[d.ckpt-1].toFixed(3)}). On test years it never saw it scored AUC ${d.test.toFixed(3)}, lift ×${d.lift.toFixed(2)}. Validation promised more than test delivered.`:`Best so far: ${d.val[b].toFixed(3)} at epoch ${b+1} (gold ring), the version that would be saved.`+(e>=10?" Loss keeps falling but validation does not follow: it is fitting the training seasons, not learning a rule that carries over.":"")));
  sl.value=String(e);pb.textContent=window._tm?"Pause":"Play";pb.setAttribute("aria-pressed",String(!!window._tm))};
 const seeds=()=>sc.replaceChildren(chips(S.map((_,i)=>[i,"Run "+(i+1)]),si,v=>{si=v;e=1;stop();seeds();upd()}));
 sl.oninput=()=>{stop();e=+sl.value;upd()};pb.onclick=()=>{if(window._tm)stop();else{if(e>=20)e=1;window._tm=setInterval(()=>{e++;if(e>=20)stop();upd()},500)}upd()};
 box.append(h("h2",{},"Watch the black box train (5 runs, Kharif only)"),sc,ca,h("div",{class:"chips"},pb,sl),ro);seeds();upd();return box}
function inside(){const out=h("div",{}),C=D.s.tier_cuts,
 P=[["1 Weather","Weekly rainfall, soil wetness and humidity for each state through the 26-week Kharif season.","raw weekly values, one weather point per state"],["2 Anomaly","Compare the season with that state’s own usual. Wetter than usual gives a positive number.","anomaly = (this season − usual) ÷ typical spread"],["3 Index","Average the three anomalies, then flip the sign so higher means drier. No labels are used, so there is nothing to memorize.","index = −mean(rain, wetness, humidity anomalies)"],["4 Tier","Cut points from training years sort seasons into wet, medium and dry.","wet ≤ lower cut < medium ≤ upper cut < dry"]],
 SU=[["Too little data","Likely, not proven","About 4,200 training rows for a model with 16 hidden units reading 30 inputs a week. It overfit: training loss fell while validation AUC faded."],["Mixed seasons","Tested: not the cause","The Kharif-only RNN scored 0.526, the same as the all-season RNN (0.529)."],["Soil and crop shortcuts","Tested: not the cause","A model given only soil and crop identity scored 0.506, so it found no hidden shortcut."],["Climate shifted","Measured","Test years ran wetter than training (soil wetness +0.65 standard deviations), so the model met inputs unlike those it learned from."]];
 let p=0,q=0;const flow=h("div",{class:"card"}),sus=h("div",{class:"card"});
 const df=()=>flow.replaceChildren(h("h2",{},"The glass box: 4 steps"),chips(P.map((x,i)=>[i,x[0]]),p,v=>{p=v;df()}),h("p",{class:"big"},P[p][1]),h("code",{},P[p][2]),p===3?h("p",{class:"mut"},`Cut points: ${f3(C.lower)} and ${f3(C.upper)}. Try them on the Tiers tab.`):"");
 const dq=()=>sus.replaceChildren(h("h2",{},"Why did the black box fail?"),chips(SU.map((x,i)=>[i,x[0]]),q,v=>{q=v;dq()}),h("p",{},h("span",{class:"v"},SU[q][1]),SU[q][2]));
 df();dq();out.append(split(replay(),h("div",{},flow,sus,expl("The RNN learns its own rules from data and cannot say why. The moisture rule has four steps you can verify by hand. Press Play to watch the RNN train; open each step of the rule; or question why the RNN fell short. The side-by-side scale of every run against the rule is on the Tour tab."))));return out}
load();
</script></body></html>
"""


def main():
    sp, yp = find_input("kharif_suitability.json"), find_input("kharif_year_robustness.json")
    print(f"Reading {sp}\n        {yp}")
    s, y = json.load(open(sp, encoding="utf-8")), json.load(open(yp, encoding="utf-8"))
    check(s, y)
    data = json.dumps({"s": s, "y": y, "k": STATIC}, separators=(",", ":")).replace("<", "\\u003c")
    html = TEMPLATE.replace("__AUTHOR__", AUTHOR).replace("__DATA__", data)
    out = sp.parent / "harvestsight_dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({len(html) / 1024:.0f} KB). Open it in any browser; no server needed.")


if __name__ == "__main__":
    main()