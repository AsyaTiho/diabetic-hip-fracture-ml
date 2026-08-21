import os
import pandas as pd, numpy as np, ast
DIR=os.environ.get('HIPFX_BEHRT_DIR', 'data/behrt')

print('loading df_with_dobyr (35GB)...', flush=True)
df = pd.read_pickle(f'{DIR}/df_with_dobyr.pkl')
print('cols:', list(df.columns), flush=True)
print('rows:', len(df), flush=True)
print(df.head(3).to_string(), flush=True)

# detect intday column
intcol = 'all_intdays' if 'all_intdays' in df.columns else ('first_intday' if 'first_intday' in df.columns else None)
print('intday column =', intcol, flush=True)

DIAB = df['phenotype'].astype(str).str.contains('diabetes', case=False, na=False)
HIPSET = {'MS_745.11','MS_745.9'}
if 'phecode' in df.columns:
    HIP = df['phecode'].isin(HIPSET)
else:
    HIP = df['phenotype'].isin({'Fracture of femur','Pathological fracture'})
print(f'diabetes rows={DIAB.sum():,}  hip rows={HIP.sum():,}', flush=True)

def explode(sub):
    sub = sub[['enrolid','dobyr',intcol]].copy()
    def tolist(x):
        if isinstance(x,(list,tuple,np.ndarray)): return list(x)
        if isinstance(x,str):
            try: return ast.literal_eval(x)
            except: return []
        return [x]
    sub['days']=sub[intcol].map(tolist)
    sub=sub.explode('days').dropna(subset=['days'])
    sub['days']=sub['days'].astype(float)
    sub['age']=(2003 + sub['days']/365.25 - sub['dobyr']).astype(int)
    return sub

dia=explode(df[DIAB]); hip=explode(df[HIP])
print('exploded diab,hip:', len(dia), len(hip), flush=True)

# OLD = min over ALL days ; NEW = min over days with age>=20
def first(sub, adult):
    s = sub[sub['age']>=20] if adult else sub
    return s.groupby('enrolid')['days'].min()
old_dm=first(dia,False); new_dm=first(dia,True)
old_hf=first(hip,False); new_hf=first(hip,True)

pat=pd.DataFrame(index=df['enrolid'].drop_duplicates())
pat['old_dm']=old_dm; pat['new_dm']=new_dm; pat['old_hf']=old_hf; pat['new_hf']=new_hf

def rc(dm,hf):
    out=np.zeros(len(dm),dtype=np.int8)
    dt=hf-dm
    out=np.where(dm.isna()|hf.isna(),0,np.where(dt<=0,0,np.where(dt<=365,2,np.where(dt<=1825,1,np.where(dt<=3650,3,0))))).astype(np.int8)
    return out
pat['rc_old']=rc(pat['old_dm'],pat['old_hf'])
pat['rc_new']=rc(pat['new_dm'],pat['new_hf'])

# validate OLD against hip_frac_labels
lab=pd.read_pickle(f'{DIR}/hip_frac_labels.pkl').set_index('enrolid')
j=pat.join(lab[['risk_class']],how='right')
mism=(j['rc_old'].fillna(0).astype(int)!=j['risk_class'].astype(int)).sum()
print(f'\nVALIDATION: my rc_old vs hip_frac_labels mismatches = {mism:,} / {len(j):,}', flush=True)

sem={0:'Control',2:'<1y',1:'1-5y',3:'5-10y'}
j['old']=j['rc_old'].fillna(0).astype(int).map(sem); j['new']=j['rc_new'].fillna(0).astype(int).map(sem)
chg=j['old']!=j['new']
print(f'\nLABELS THAT CHANGE under age>=20 filter: {chg.sum():,} / {len(j):,} ({100*chg.mean():.4f}%)', flush=True)
print('\nAmong FRACTURE-class patients (old label != Control):', flush=True)
frac=j[j['old']!='Control']
print(f'  total fracture-class (old) = {len(frac):,}; of these changed = {(frac["old"]!=frac["new"]).sum():,}', flush=True)
print('\nCross-tab old->new for changed patients:', flush=True)
order=['Control','<1y','1-5y','5-10y']
print(pd.crosstab(j.loc[chg,'old'],j.loc[chg,'new']).reindex(index=order,columns=order).fillna(0).astype(int).to_string(), flush=True)
print('\nNew class counts vs old:', flush=True)
print(pd.DataFrame({'old':j['old'].value_counts().reindex(order),'new':j['new'].value_counts().reindex(order)}).to_string(), flush=True)
print('\nDONE', flush=True)
