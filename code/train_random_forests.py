import argparse,time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from baseline import load_data
from train_numpy_baselines import metrics, ranking_metrics

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    data,_=load_data(args.database)
    tr=data[data.FIRE_YEAR.between(1992,2014)]
    va=data[data.FIRE_YEAR.between(2015,2017)]
    assert (len(data),len(tr),len(va))==(213463,180656,17559)
    season=va.month.map(dict(zip(range(1,13),['Winter']*2+['Spring']*3+['Summer']*3+['Fall']*3+['Winter'])))
    region=pd.Series(np.where(va.LATITUDE>=37,'37N and above','Below 37N'),index=va.index)
    rows=[]; groups=[]
    for location in (False,True):
        name='Calendar + location' if location else 'Calendar'
        cols=['month','weekday']
        prep=ColumnTransformer([('calendar',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cols)],remainder='passthrough')
        if location: cols=cols+['LATITUDE','LONGITUDE']
        X=prep.fit_transform(tr[cols]); V=prep.transform(va[cols])
        for weight in (None,'balanced'):
            start=time.perf_counter()
            forest=RandomForestClassifier(n_estimators=200,max_depth=15,min_samples_leaf=20,n_jobs=4,random_state=42,class_weight=weight)
            forest.fit(X,tr.target)
            elapsed=time.perf_counter()-start
            p=forest.predict_proba(V)[:,0]
            rank=ranking_metrics(va.target,p)
            for threshold in (.3,.4,.5,.6,.7):
                pred=(p<threshold).astype(int)
                rows.append(dict(features=name,weight=weight or 'unweighted',natural_threshold=threshold,fit_seconds=elapsed,**metrics(va.target.to_numpy(),pred),**rank))
                for grouping,labels in [('season',season),('latitude band',region)]:
                    for group in labels.unique():
                        mask=(labels==group).to_numpy()
                        y=va.target.to_numpy()[mask]; yp=pred[mask]
                        groups.append(dict(features=name,weight=weight or 'unweighted',natural_threshold=threshold,grouping=grouping,group=group,n=len(y),natural_n=int((y==0).sum()),**metrics(y,yp)))
            print(name,weight,round(elapsed,2),flush=True)
    pd.DataFrame(rows).to_csv(args.output/'forest_validation.csv',index=False)
    pd.DataFrame(groups).to_csv(args.output/'forest_validation_errors.csv',index=False)

if __name__=='__main__':
    main()
