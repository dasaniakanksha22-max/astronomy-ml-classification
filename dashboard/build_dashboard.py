"""Build a self-contained interactive dashboard from the Phase 3 exports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs


CLASSES = ["GALAXY", "STAR", "QSO"]


def build_dashboard(project_root: Path) -> Path:
    project_root = Path(project_root)
    dashboard_dir = project_root / "dashboard"
    full_summary = pd.read_csv(dashboard_dir / "looker_class_summary.csv")
    band_summary = pd.read_csv(dashboard_dir / "looker_band_summary.csv")
    hist = pd.read_csv(dashboard_dir / "looker_magnitude_bins.csv")
    sample = pd.read_csv(dashboard_dir / "looker_scatter_sample.csv")

    # Round sample coordinates for a lighter HTML document; the source CSVs
    # retain the original numerical precision.
    payload = {
        "classes": full_summary.to_dict(orient="records"),
        "bands": band_summary.round(4).to_dict(orient="records"),
        "hist": hist.round(3).to_dict(orient="records"),
        "sample": sample[["class", "u", "g", "r", "i", "z", "u_g", "g_r", "r_i", "i_z", "alpha", "delta", "redshift"]]
        .round(4).to_dict(orient="records"),
    }
    json_payload = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")

    template = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SDSS | Astronomical Data Explorer</title>
<style>
:root{--bg:#080e1b;--pane:#111c2d;--pane2:#17243a;--stroke:#26354a;--txt:#edf3fc;--muted:#93a8c1;--mint:#64d8b1;--blue:#72a7f3;--pink:#d49cea;--amber:#f5bc72}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.5 system-ui,-apple-system,Segoe UI,Arial,sans-serif}button,select{font:inherit}
.layout{display:flex;min-height:100vh}.sidebar{width:244px;flex-shrink:0;background:#0b1424;border-right:1px solid var(--stroke);padding:27px 17px;position:sticky;top:0;height:100vh}.brand{font-size:14px;letter-spacing:2.1px;font-weight:800;color:#e6f6fa}.brand small{display:block;letter-spacing:.6px;font-size:10px;color:var(--mint);margin-top:8px;font-weight:700}.divider{border-top:1px solid var(--stroke);margin:25px 0 17px}.side-caption{font-size:11px;text-transform:uppercase;color:var(--muted);font-weight:700;letter-spacing:1.6px;margin:0 10px 12px}.nav{display:block;width:100%;text-align:left;border:0;background:transparent;color:#adc0d4;padding:13px 14px;border-radius:9px;margin:3px 0;cursor:pointer}.nav.active{background:#173349;color:#c1ffe9;box-shadow:inset 3px 0 0 var(--mint)}.nav:hover:not(.active){background:#142438}.sidebar-foot{font-size:11px;color:var(--muted);line-height:1.8;margin-top:30px;padding:12px}.main{min-width:0;flex:1;max-width:1600px;margin:auto;padding:30px 36px 70px}.eyebrow{color:var(--mint);font-size:11px;font-weight:800;letter-spacing:1.9px;text-transform:uppercase}.header{display:flex;justify-content:space-between;gap:20px;align-items:end;flex-wrap:wrap}.header h1{font-size:31px;line-height:1.18;letter-spacing:-1px;margin:7px 0}.lead{color:var(--muted);max-width:650px;margin:7px 0 0}.pill{border:1px solid #2f6f6c;color:#8ce9cb;border-radius:50px;padding:8px 13px;font-size:12px;background:#11302f;white-space:nowrap}.filters{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:24px 0}.filters b{margin-right:4px;font-size:12px;color:var(--muted)}.filter{border:1px solid var(--stroke);background:var(--pane);padding:8px 14px;border-radius:30px;color:var(--txt);cursor:pointer}.filter.active{border-color:#5fdab6;background:#194436;color:#a9ffe0}.cards{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:14px}.stat{border:1px solid var(--stroke);background:linear-gradient(150deg,var(--pane),#0d1727);padding:19px 21px;border-radius:14px}.stat .tag{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;font-weight:700}.stat .val{font-size:29px;font-weight:800;margin-top:6px;letter-spacing:-.9px}.stat .sub{font-size:11px;color:var(--muted);margin-top:4px}.two{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:18px}.chart-card{border:1px solid var(--stroke);background:var(--pane);padding:19px 20px 13px;border-radius:15px;min-width:0}.chart-card h2{font-size:15px;font-weight:750;margin:0}.chart-card .desc{font-size:12px;color:var(--muted);margin:5px 0 5px}.chart{width:100%;height:335px}.chart.large{height:440px}.wide{margin-top:18px}.note{border-left:3px solid #4db795;padding:12px 17px;background:#102b2b;color:#b5d9cf;border-radius:0 8px 8px 0;font-size:12px;margin-top:19px}.controls-line{display:flex;align-items:center;gap:11px;flex-wrap:wrap;margin-top:15px}.controls-line label{color:var(--muted);font-size:12px}.select{background:#172b41;border:1px solid #38526a;border-radius:7px;color:#e8f3fa;padding:7px 12px}.page{display:none}.page.active{display:block}.foot{border-top:1px solid var(--stroke);margin-top:27px;padding-top:18px;color:var(--muted);font-size:12px}.legend{display:flex;gap:17px;flex-wrap:wrap;margin:10px 0 0;color:var(--muted);font-size:12px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:12px;border-bottom:1px solid var(--stroke);text-align:right}th:first-child,td:first-child{text-align:left}th{color:#a4c1d4;font-weight:700}.empty{margin:28px 0;border:1px dashed #3b5b75;border-radius:14px;padding:28px;color:#bdd2e0}.empty b{color:#e8f4fd;display:block;font-size:17px;margin-bottom:9px}
@media(max-width:950px){.sidebar{width:75px;padding:20px 9px}.brand{font-size:11px;letter-spacing:0}.brand small,.side-caption,.sidebar-foot{display:none}.nav{font-size:0;padding:13px 10px}.nav:before{content:'◦';font-size:24px}.main{padding:22px 17px 45px}.two{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,minmax(120px,1fr))}}
@media(max-width:520px){.sidebar{display:none}.main{padding:15px}.cards{grid-template-columns:1fr 1fr}.stat .val{font-size:22px}.header h1{font-size:25px}}
</style>
<script>/*PLOTLY_JS*/</script>
</head>
<body>
<div class="layout">
<aside class="sidebar">
  <div class="brand">SDSS / EXPLORE<small>ASTRONOMICAL OBJECTS</small></div>
  <div class="divider"></div><div class="side-caption">Dashboard sections</div>
  <button class="nav active" data-page="overview">01&nbsp; Overview</button>
  <button class="nav" data-page="photometry">02&nbsp; Photometry</button>
  <button class="nav" data-page="colours">03&nbsp; Colour-space</button>
  <button class="nav" data-page="sky">04&nbsp; Sky & redshift</button>
  <button class="nav" data-page="models">05&nbsp; Model results</button>
  <div class="divider"></div><div class="sidebar-foot">Dataset: SDSS17<br/>Source: cleaned survey CSV<br/>99,999 observations<br/><br/>Photometry-only classification<br/>Target: STAR / GALAXY / QSO</div>
</aside>
<main class="main">
  <header class="header"><div><div class="eyebrow">ML PROJECT · PHASE 03</div><h1>Astronomical Data Explorer</h1><p class="lead">An interactive overview of SDSS stellar classifications, five-band photometry and astronomical colour indices.</p></div><div class="pill">● Dataset validated · 99,999 rows</div></header>
  <div class="filters"><b>Filter class</b><button class="filter active" data-class="ALL">All objects</button><button class="filter" data-class="GALAXY">GALAXY</button><button class="filter" data-class="STAR">STAR</button><button class="filter" data-class="QSO">QSO</button></div>
  <section class="page active" id="overview"><div class="cards" id="kpis"></div><div class="two"><div class="chart-card"><h2>Observed class distribution</h2><p class="desc">Full population · counts, not sample estimates</p><div class="chart" id="class-chart"></div></div><div class="chart-card"><h2>Median magnitude profiles</h2><p class="desc">Five photometric bands; smaller magnitudes are brighter</p><div class="chart" id="medians-chart"></div></div></div><div class="note">Class distribution uses the complete cleaned dataset. All plots in this dashboard are descriptive; they do not show model predictions or classification accuracy.</div></section>
  <section class="page" id="photometry"><div class="chart-card"><h2>Magnitude frequency by class</h2><p class="desc">Counts from the full cleaned dataset, aggregated into 0.5-magnitude bins</p><div class="controls-line"><label for="band">Photometric filter</label><select class="select" id="band"><option value="u">u · near ultraviolet</option><option value="g">g · blue-green</option><option value="r" selected>r · red</option><option value="i">i · far red</option><option value="z">z · near infrared</option></select></div><div class="chart large" id="hist-chart"></div></div><div class="wide chart-card"><h2>Median observed magnitudes</h2><p class="desc">Complete dataset: filter × spectroscopic class</p><div class="chart" id="band-bars"></div></div></section>
  <section class="page" id="colours"><div class="chart-card"><h2>Colour-colour diagrams</h2><p class="desc">Stratified sample: 3,000 randomly selected observations per class (9,000 total); sampling is for visualization only</p><div class="controls-line"><label for="xcolor">X-axis</label><select class="select" id="xcolor"><option value="g_r" selected>g − r</option><option value="u_g">u − g</option><option value="r_i">r − i</option><option value="i_z">i − z</option></select><label for="ycolor">Y-axis</label><select class="select" id="ycolor"><option value="u_g" selected>u − g</option><option value="g_r">g − r</option><option value="r_i">r − i</option><option value="i_z">i − z</option></select></div><div class="chart large" id="colour-chart"></div></div><div class="note">Balanced samples are deliberately not representative of real class proportions. Never use this sample to compute dataset-wide class percentages.</div></section>
  <section class="page" id="sky"><div class="two"><div class="chart-card"><h2>Sky coverage</h2><p class="desc">Right ascension α versus declination δ; visualization sample</p><div class="chart" id="sky-chart"></div></div><div class="chart-card"><h2>Redshift distributions</h2><p class="desc">Spectroscopic redshift; visualization sample</p><div class="chart" id="z-chart"></div></div></div><div class="note">Redshift is explored here for scientific context but is excluded from the primary photometry-only classifier. Sky coordinates describe the survey footprint, not intrinsic object type.</div></section>
  <section class="page" id="models"><div class="empty"><b>Model performance will be added in Phase 6</b>No accuracy, macro F1, confusion matrices, feature importances or prediction probabilities are shown yet. This prevents reporting untrained or invented model results. When Phase 6 is complete, import validated evaluation metrics here.</div></section>
  <footer class="foot">Data: SDSS17 Stellar Classification dataset · Original 100,000 observations · One invalid photometric row removed · Figures derived from 99,999 cleaned records · Sampling seed: 42</footer>
</main></div>
<script>
const D=/*DASHBOARD_JSON*/;
const COLORS={GALAXY:'#65c8d1',STAR:'#f1be74',QSO:'#c19bf7'};
let active='ALL', currentPage='overview';
const visibleClasses=()=>active==='ALL'?['GALAXY','STAR','QSO']:[active];
const sum=a=>a.reduce((s,v)=>s+v,0);
const fmt=n=>Math.round(n).toLocaleString('en-US');
const plotLayout=(extra={})=>({paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{family:'Arial, sans-serif',color:'#b3c5d9',size:12},margin:{l:59,r:20,t:18,b:56},xaxis:{gridcolor:'#26364a',zerolinecolor:'#26364a'},yaxis:{gridcolor:'#26364a',zerolinecolor:'#26364a'},legend:{orientation:'h',y:-.24,x:0},hovermode:'closest',...extra});
const plot=(id,traces,layout={})=>Plotly.react(id,traces,plotLayout(layout),{responsive:true,displayModeBar:false,scrollZoom:false});
function renderKpis(){
 const C=D.classes.filter(x=>visibleClasses().includes(x.class)),n=sum(C.map(x=>x.observations));
 const all=sum(D.classes.map(x=>x.observations));
 const stars=C.find(x=>x.class==='STAR')?.observations||0;
 const galaxies=C.find(x=>x.class==='GALAXY')?.observations||0;
 const qsos=C.find(x=>x.class==='QSO')?.observations||0;
 const items=[['Observations',fmt(n),`${(100*n/all).toFixed(2)}% of cleaned dataset`],['Galaxies',fmt(galaxies),'Spectroscopic GALAXY'],['Stars',fmt(stars),'Spectroscopic STAR'],['Quasars',fmt(qsos),'Spectroscopic QSO']];
 document.getElementById('kpis').innerHTML=items.map(([t,v,s])=>`<div class="stat"><div class="tag">${t}</div><div class="val">${v}</div><div class="sub">${s}</div></div>`).join('');
}
function renderOverview(){renderKpis();
 const c=D.classes.filter(x=>visibleClasses().includes(x.class));
 plot('class-chart',[{type:'bar',x:c.map(x=>x.class),y:c.map(x=>x.observations),marker:{color:c.map(x=>COLORS[x.class]),line:{width:0}},text:c.map(x=>fmt(x.observations)),textposition:'outside',hovertemplate:'%{x}: %{y:,}<extra></extra>'}],{showlegend:false,yaxis:{gridcolor:'#26364a',title:'Observations'},xaxis:{showgrid:false},margin:{l:67,r:22,t:32,b:50}});
 const b=['u','g','r','i','z'];
 plot('medians-chart',visibleClasses().map(c=>({type:'scatter',mode:'lines+markers',name:c,x:b,y:b.map(k=>D.bands.find(x=>x.class===c&&x.band===k).median),line:{color:COLORS[c],width:3},marker:{size:8},hovertemplate:'%{x}: %{y:.3f}<extra>'+c+'</extra>'})),{xaxis:{title:'SDSS filter',showgrid:false},yaxis:{title:'Median magnitude',gridcolor:'#26364a'},margin:{l:65,r:25,t:17,b:65}});
}
function renderPhotometry(){const b=document.getElementById('band').value;
 const records=D.hist.filter(x=>x.band===b&&visibleClasses().includes(x.class));
 const traces=visibleClasses().map(c=>{const r=records.filter(x=>x.class===c).sort((a,b)=>a.bin_center-b.bin_center);return {type:'bar',name:c,x:r.map(x=>x.bin_center),y:r.map(x=>x.count),marker:{color:COLORS[c]},opacity:.76,hovertemplate:'Magnitude %{x:.1f}<br>Count %{y:,}<extra>'+c+'</extra>'};});
 plot('hist-chart',traces,{barmode:'overlay',xaxis:{title:`${b}-band magnitude`,gridcolor:'#26364a'},yaxis:{title:'Observations',gridcolor:'#26364a'},legend:{orientation:'h',y:-.25},margin:{l:63,r:20,t:12,b:66}});
 const bands=['u','g','r','i','z'];
 plot('band-bars',visibleClasses().map(c=>({type:'bar',name:c,x:bands,y:bands.map(x=>D.bands.find(t=>t.class===c&&t.band===x).median),marker:{color:COLORS[c]},hovertemplate:'%{x}: %{y:.3f}<extra>'+c+'</extra>'})),{barmode:'group',xaxis:{title:'Filter',showgrid:false},yaxis:{title:'Median magnitude',gridcolor:'#26364a'},margin:{l:64,r:22,t:18,b:65}});
}
function filteredSample(c){return D.sample.filter(x=>x.class===c);}
function renderColours(){const x=document.getElementById('xcolor').value,y=document.getElementById('ycolor').value;
 const traces=visibleClasses().map(c=>{const r=filteredSample(c);return{type:'scattergl',mode:'markers',name:c,x:r.map(o=>o[x]),y:r.map(o=>o[y]),marker:{color:COLORS[c],size:5,opacity:.46},hovertemplate:`${x}: %{x:.3f}<br>${y}: %{y:.3f}<extra>${c}</extra>`};});
 const lim={u_g:[-2.5,6],g_r:[-2.3,4.5],r_i:[-2,3],i_z:[-2.1,3.2]};
 plot('colour-chart',traces,{xaxis:{title:x.replace('_',' − '),gridcolor:'#26364a',range:lim[x]},yaxis:{title:y.replace('_',' − '),gridcolor:'#26364a',range:lim[y]},legend:{orientation:'h',y:-.2},margin:{l:66,r:25,t:12,b:66}});
}
function renderSky(){const classes=visibleClasses();
 plot('sky-chart',classes.map(c=>{const r=filteredSample(c);return {type:'scattergl',mode:'markers',name:c,x:r.map(o=>o.alpha),y:r.map(o=>o.delta),marker:{size:3,color:COLORS[c],opacity:.32},hovertemplate:'RA %{x:.1f}°<br>Dec %{y:.1f}°<extra>'+c+'</extra>'};}),{xaxis:{title:'Right ascension (°)',gridcolor:'#26364a',range:[0,360]},yaxis:{title:'Declination (°)',gridcolor:'#26364a',range:[-22,90]},margin:{l:65,r:20,t:12,b:65}});
 plot('z-chart',classes.map(c=>{const r=filteredSample(c);return {type:'histogram',name:c,x:r.map(o=>o.redshift),xbins:{start:-.2,end:7.4,size:.15},marker:{color:COLORS[c]},opacity:.70,hovertemplate:'z = %{x:.2f}<br>Sample count %{y}<extra>'+c+'</extra>'};}),{barmode:'overlay',xaxis:{title:'Spectroscopic redshift z',gridcolor:'#26364a',range:[-.2,7.4]},yaxis:{title:'Sample frequency',gridcolor:'#26364a'},margin:{l:65,r:20,t:12,b:65}});
}
function render(){if(currentPage==='overview')renderOverview();else if(currentPage==='photometry')renderPhotometry();else if(currentPage==='colours')renderColours();else if(currentPage==='sky')renderSky();}
document.querySelectorAll('[data-page]').forEach(e=>e.addEventListener('click',()=>{currentPage=e.dataset.page;document.querySelectorAll('.nav').forEach(n=>n.classList.toggle('active',n===e));document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id===currentPage));render();}));
document.querySelectorAll('[data-class]').forEach(e=>e.addEventListener('click',()=>{active=e.dataset.class;document.querySelectorAll('.filter').forEach(n=>n.classList.toggle('active',n===e));render();}));
['band','xcolor','ycolor'].forEach(id=>document.getElementById(id).addEventListener('change',render));
render();
</script>
</body></html>'''
    html = template.replace("/*PLOTLY_JS*/", get_plotlyjs()).replace("/*DASHBOARD_JSON*/", json_payload)
    output = dashboard_dir / "SDSS_Interactive_Dashboard.html"
    output.write_text(html, encoding="utf-8")
    return output


if __name__ == "__main__":
    print(build_dashboard(Path(__file__).resolve().parents[1]))
