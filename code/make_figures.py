"""Reproduce proposal figures from FPA FOD v6. No model training.
Run: python make_figures.py --database PATH_TO_SQLITE
Requires numpy, pandas, reportlab; pypdfium2 optionally exports PNGs.
"""
from pathlib import Path
import argparse, ast, sqlite3, json, calendar
import numpy as np
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color, white
parser=argparse.ArgumentParser()
parser.add_argument("--database",type=Path,required=True)
parser.add_argument("--output",type=Path,default=Path("figures"))
args=parser.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=True)
# Reuse the audited loader without importing its optional model dependencies.
src=Path(__file__).with_name("baseline.py").read_text(encoding="utf-8")
fn=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=="load_data")
exec(compile(ast.Module(body=[fn],type_ignores=[]),"load_data","exec"))
df,audit=load_data(args.database)
tr=df[df.FIRE_YEAR<=2014].copy()
assert len(df)==213463 and len(tr)==180656, "Unexpected dataset version or cleaning results"
monthly=pd.crosstab(tr.month,tr.target).reindex(range(1,13),fill_value=0)
monthly.columns=["Natural","Human"]
monthly.to_csv(out/"monthly_training_counts.csv",index_label="Month")
normalized=monthly.div(monthly.sum()).mul(100)
normalized.to_csv(out/"monthly_training_percentages.csv",index_label="Month")
stats=[]
for label,mask in [("Train 1992-2014",df.FIRE_YEAR<=2014),("Validation 2015-2017",df.FIRE_YEAR.between(2015,2017)),("Test 2018-2020",df.FIRE_YEAR.between(2018,2020))]:
    p=df[mask]
    stats.append({"Split":label,"Records":len(p),"Human":int(p.target.sum()),"Natural":int((p.target==0).sum()),"Human_pct":p.target.mean()*100})
pd.DataFrame(stats).to_csv(out/"split_statistics.csv",index=False)
(out/"data_audit.json").write_text(json.dumps(audit,default=lambda x:int(x),indent=2))
colors={"Human":HexColor("#B05A18"),"Natural":HexColor("#176C9C")}
ink=HexColor("#26343E"); grey=HexColor("#DCE2E5")
def text(c,x,y,t,size=10,font="Helvetica",align="left",color=ink):
    c.setFillColor(color);c.setFont(font,size)
    getattr(c,{"left":"drawString","center":"drawCentredString","right":"drawRightString"}[align])(x,y,str(t))
def axis(c,x,y,w,h,xmin,xmax,ymin,ymax,xticks,yticks,xlabel,ylabel):
    fx=lambda v:x+(v-xmin)/(xmax-xmin)*w
    fy=lambda v:y+(v-ymin)/(ymax-ymin)*h
    for v,label in yticks:
        c.setStrokeColor(grey);c.setLineWidth(.5);c.line(x,fy(v),x+w,fy(v))
        text(c,x-8,fy(v)-4,label,14,align="right")
    c.setStrokeColor(ink);c.setLineWidth(.7)
    c.line(x,y,x+w,y);c.line(x,y,x,y+h)
    for v,label in xticks:
        c.line(fx(v),y,fx(v),y-3);text(c,fx(v),y-19,label,14,align="center")
    text(c,x+w/2,y-40,xlabel,15,align="center")
    c.saveState();c.translate(x-49,y+h/2);c.rotate(90)
    text(c,0,0,ylabel,15,align="center");c.restoreState()
    return fx,fy
def header(c,title,sub):
    text(c,45,389,title,20,"Helvetica-Bold")
    text(c,45,365,sub,14)
def footer(c,note):
    text(c,45,33,note,11)
    text(c,45,16,"Source: Short (2022), FPA FOD sixth edition, DOI: 10.2737/RDS-2013-0009.6.",11)
c=canvas.Canvas(str(out/"01_seasonality.pdf"),pagesize=(950,415))
header(c,"Seasonal patterns of recorded California wildfires","Training records only, 1992-2014 | Human: 154,858; Natural: 25,798")
for x,title in [(75,"A  Recorded fires by month"),(535,"B  Monthly share within each cause class")]:
    text(c,x,337,title,15,"Helvetica-Bold")
for j,(vals,ymax,label) in enumerate([(monthly,35000,"Recorded fires (count)"),(normalized,35,"Within-class share (%)")]):
    x=75+j*460
    ticks=[(v,f"{int(v/1000)}k" if j==0 and v else str(v)) for v in (range(0,35001,5000) if j==0 else range(0,36,5))]
    fx,fy=axis(c,x,109,350,205,1,12,0,ymax,[(i,calendar.month_abbr[i]) for i in range(1,13)],ticks,"Discovery month",label)
    for name,dash in [("Human",[]),("Natural",[4,3])]:
        c.setStrokeColor(colors[name]);c.setLineWidth(1.8);c.setDash(dash)
        p=c.beginPath()
        for i,v in vals[name].items():
            if i==1:p.moveTo(fx(i),fy(v))
            else:p.lineTo(fx(i),fy(v))
        c.drawPath(p);c.setDash()
        for i,v in vals[name].items():
            c.setFillColor(colors[name] if name=="Human" else white)
            c.circle(fx(i),fy(v),2.4,stroke=1,fill=1)
    for k,name in enumerate(["Human","Natural"]):
        c.setStrokeColor(colors[name]);c.setLineWidth(1.8);c.setDash([] if name=="Human" else [4,3])
        c.line(x+k*110,56,x+20+k*110,56);c.setDash()
        text(c,x+26+k*110,51,name,14)
footer(c,"Panel B uses each class's full training count as its denominator; percentages sum to 100% within each class.")
c.save()
c=canvas.Canvas(str(out/"02_locations.pdf"),pagesize=(950,415))
header(c,"Recorded wildfire locations by reported cause","California training records, 1992-2014 | Equal-size random samples; seed = 42")
samples=[]
for j,(target,name) in enumerate([(1,"Human"),(0,"Natural")]):
    part=tr[tr.target==target].sample(n=5000,random_state=42)
    samples.append(part[["FOD_ID","LATITUDE","LONGITUDE","target"]])
    x=86+j*465
    text(c,x,337,f"{name}: 5,000 of {int((tr.target==target).sum()):,} records",15,"Helvetica-Bold")
    fx,fy=axis(c,x,107,310,208,-125,-113,32,43,[(-124,"-124"),(-122,"-122"),(-120,"-120"),(-118,"-118"),(-116,"-116"),(-114,"-114")],[(v,str(v)) for v in [32,34,36,38,40,42]],"Longitude (degrees)","Latitude (degrees)")
    c.setFillColor(colors[name]);c.setFillAlpha(.30)
    for lon,lat in zip(part.LONGITUDE,part.LATITUDE):c.circle(fx(lon),fy(lat),.85,stroke=0,fill=1)
    c.setFillAlpha(1)
footer(c,"Coordinate scatterplots, not projected maps. Identical axes; equal samples compare spatial patterns, not class prevalence.")
c.save()
pd.concat(samples).to_csv(out/"sampled_training_locations.csv",index=False)
summary={"training_n":len(tr),"human_peak_month":int(monthly.Human.idxmax()),"natural_peak_month":int(monthly.Natural.idxmax()),
"human_july_august_pct":float(normalized.loc[[7,8],"Human"].sum()),"natural_july_august_pct":float(normalized.loc[[7,8],"Natural"].sum())}
(out/"findings.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
try:
    import pypdfium2 as pdfium
    for f in out.glob("0*.pdf"):
        doc=pdfium.PdfDocument(f);doc[0].render(scale=2).to_pil().save(f.with_suffix(".png"));doc.close()
except ImportError:
    print("PNG export skipped; install pypdfium2 if needed. Vector PDFs were created.")

