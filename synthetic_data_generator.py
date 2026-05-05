import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

CLASS_LABELS = {
    1: "Acute Cardiac Event",
    2: "Septic/Infectious Shock",
    3: "Neurological Emergency",
    4: "Metabolic Emergency",
    5: "Drug/Medication Induced Collapse",
    6: "Hemorrhagic/Internal Bleeding Shock",
}

CLASS_COUNTS = {1: 5500, 2: 5500, 3: 4500, 4: 4500, 5: 5000, 6: 5000}

NUMERIC_SPECS = {
    1: dict(age=(63,10,40,88), sbp=(102,18,72,155), dbp=(64,10,40,98), hr=(118,16,85,165), rr=(24,4,16,34), spo2=(92,4,82,99), temp=(98.7,0.8,97,101), glucose=(135,40,70,260), gcs=(11,3,5,15), hb=(13.1,1.2,9,16), wbc=(9.5,2,4,15), lactate=(2.3,0.8,0.8,5), creatinine=(1.1,0.3,0.5,2.2), troponin=(5.5,2,1.2,12), sodium=(138,4,128,147), potassium=(4.3,0.5,3.1,5.8)),
    2: dict(age=(58,14,25,90), sbp=(82,14,55,118), dbp=(48,9,28,75), hr=(126,15,95,170), rr=(30,5,20,42), spo2=(89,5,78,97), temp=(101.9,1.1,99,104.5), glucose=(128,45,60,280), gcs=(9,3,3,15), hb=(12,1.5,8,16), wbc=(16,4,9,28), lactate=(4.8,1.5,2,9), creatinine=(1.6,0.5,0.7,3.5), troponin=(0.25,0.2,0,1.2), sodium=(136,5,124,148), potassium=(4.5,0.7,3,6.5)),
    3: dict(age=(66,12,38,92), sbp=(138,22,88,190), dbp=(82,14,45,118), hr=(98,18,60,145), rr=(22,4,14,34), spo2=(94,3,84,99), temp=(98.8,0.7,97,101), glucose=(118,35,55,240), gcs=(7,3,3,14), hb=(12.8,1.3,8,16), wbc=(9,2.5,4,16), lactate=(2,0.7,0.7,4.5), creatinine=(1.1,0.3,0.5,2.2), troponin=(0.15,0.15,0,0.9), sodium=(137,5,124,148), potassium=(4.1,0.5,3,5.8)),
    4: dict(age=(52,16,18,85), sbp=(96,18,60,140), dbp=(58,12,30,90), hr=(112,18,70,160), rr=(26,5,16,38), spo2=(93,4,82,99), temp=(98.4,0.9,96,101), gcs=(9,3,3,15), hb=(12.7,1.4,8,16), wbc=(9,3,4,18), lactate=(2.7,1,0.8,6), creatinine=(1.4,0.5,0.5,3), troponin=(0.1,0.1,0,0.8), sodium=(131,8,110,150), potassium=(5.1,1,2.2,7)),
    5: dict(age=(49,17,18,88), sbp=(92,22,50,145), dbp=(54,13,25,92), hr=(104,26,40,165), rr=(20,5,8,34), spo2=(91,5,78,99), temp=(98.3,0.8,96,101), glucose=(122,65,35,300), gcs=(8,4,3,15), hb=(12.9,1.4,8,16), wbc=(8.5,2.5,3,16), lactate=(2.4,1,0.6,6), creatinine=(1.2,0.4,0.5,3), troponin=(0.35,0.4,0,2.5), sodium=(136,6,120,148), potassium=(4.4,0.9,2.5,6.5)),
    6: dict(age=(57,15,20,90), sbp=(74,13,45,102), dbp=(42,9,20,65), hr=(132,16,100,175), rr=(28,5,18,40), spo2=(88,5,78,97), temp=(98.1,0.9,96,101), glucose=(116,35,60,240), gcs=(8,3,3,14), hb=(8.1,1.5,4.5,11), wbc=(10.5,3,4,18), lactate=(4.1,1.4,1.5,8), creatinine=(1.3,0.4,0.5,3), troponin=(0.2,0.15,0,1), sodium=(136,5,122,148), potassium=(4.3,0.7,2.8,6)),
}

BINARY_FIELDS = [
    "chest_pain","breathlessness","sweating","seizure_activity","unconsciousness","fever",
    "dizziness_confusion","palpitations","visible_bleeding_signs","diabetes_history","hypertension_history",
    "cardiac_history","stroke_history","recent_surgery","antihypertensive_use","insulin_or_diabetic_drug_use",
    "anticoagulant_use","psychiatric_drug_use","cardiac_drug_use","ecg_abnormality_present"
]

# explicit class Bernoulli probabilities
BINARY_PROBS = {
    1: {"chest_pain":0.82,"breathlessness":0.73,"sweating":0.66,"seizure_activity":0.03,"unconsciousness":0.12,"fever":0.07,"dizziness_confusion":0.28,"palpitations":0.62,"visible_bleeding_signs":0.03,"diabetes_history":0.44,"hypertension_history":0.67,"cardiac_history":0.71,"stroke_history":0.12,"recent_surgery":0.06,"antihypertensive_use":0.59,"insulin_or_diabetic_drug_use":0.31,"anticoagulant_use":0.24,"psychiatric_drug_use":0.08,"cardiac_drug_use":0.68,"ecg_abnormality_present":0.78},
    2: {"chest_pain":0.12,"breathlessness":0.58,"sweating":0.34,"seizure_activity":0.08,"unconsciousness":0.37,"fever":0.84,"dizziness_confusion":0.63,"palpitations":0.18,"visible_bleeding_signs":0.06,"diabetes_history":0.32,"hypertension_history":0.39,"cardiac_history":0.21,"stroke_history":0.10,"recent_surgery":0.18,"antihypertensive_use":0.30,"insulin_or_diabetic_drug_use":0.26,"anticoagulant_use":0.11,"psychiatric_drug_use":0.09,"cardiac_drug_use":0.22,"ecg_abnormality_present":0.22},
    3: {"chest_pain":0.10,"breathlessness":0.24,"sweating":0.19,"seizure_activity":0.42,"unconsciousness":0.61,"fever":0.10,"dizziness_confusion":0.72,"palpitations":0.12,"visible_bleeding_signs":0.03,"diabetes_history":0.29,"hypertension_history":0.55,"cardiac_history":0.24,"stroke_history":0.58,"recent_surgery":0.07,"antihypertensive_use":0.46,"insulin_or_diabetic_drug_use":0.22,"anticoagulant_use":0.19,"psychiatric_drug_use":0.17,"cardiac_drug_use":0.19,"ecg_abnormality_present":0.15},
    4: {"chest_pain":0.16,"breathlessness":0.32,"sweating":0.41,"seizure_activity":0.17,"unconsciousness":0.33,"fever":0.11,"dizziness_confusion":0.64,"palpitations":0.27,"visible_bleeding_signs":0.04,"diabetes_history":0.71,"hypertension_history":0.40,"cardiac_history":0.21,"stroke_history":0.09,"recent_surgery":0.08,"antihypertensive_use":0.34,"insulin_or_diabetic_drug_use":0.67,"anticoagulant_use":0.09,"psychiatric_drug_use":0.11,"cardiac_drug_use":0.18,"ecg_abnormality_present":0.17},
    5: {"chest_pain":0.15,"breathlessness":0.29,"sweating":0.39,"seizure_activity":0.26,"unconsciousness":0.53,"fever":0.06,"dizziness_confusion":0.57,"palpitations":0.31,"visible_bleeding_signs":0.04,"diabetes_history":0.27,"hypertension_history":0.30,"cardiac_history":0.19,"stroke_history":0.11,"recent_surgery":0.07,"antihypertensive_use":0.33,"insulin_or_diabetic_drug_use":0.30,"anticoagulant_use":0.13,"psychiatric_drug_use":0.48,"cardiac_drug_use":0.20,"ecg_abnormality_present":0.20},
    6: {"chest_pain":0.18,"breathlessness":0.62,"sweating":0.52,"seizure_activity":0.09,"unconsciousness":0.46,"fever":0.10,"dizziness_confusion":0.59,"palpitations":0.29,"visible_bleeding_signs":0.67,"diabetes_history":0.23,"hypertension_history":0.33,"cardiac_history":0.20,"stroke_history":0.10,"recent_surgery":0.34,"antihypertensive_use":0.27,"insulin_or_diabetic_drug_use":0.18,"anticoagulant_use":0.41,"psychiatric_drug_use":0.10,"cardiac_drug_use":0.16,"ecg_abnormality_present":0.18},
}

TEST_MAP = {
    1: "ECG, Troponin, CK-MB, Echocardiogram",
    2: "Blood Culture, Lactate, Procalcitonin, ABG",
    3: "CT Brain, MRI Brain, EEG, Coagulation Panel",
    4: "Glucose Panel, ABG, Serum Ketones, Electrolytes",
    5: "Toxicology Screen, ABG, ECG, Drug Levels",
    6: "CBC, Coagulation Profile, FAST Ultrasound, CT Abdomen",
}

AMBIG = {1:(5,6),2:(6,4),3:(4,5),4:(5,2),5:(1,4),6:(2,1)}

REASON_TEMPLATES = {
    1:["Chest pain with ECG/troponin abnormalities and hemodynamic stress.","Cardiac symptom cluster with elevated troponin and tachycardia."],
    2:["Infection-shock pattern: fever, hypotension, leukocytosis, and high lactate.","Septic physiology with low BP and elevated inflammatory markers."],
    3:["Neurologic compromise with low consciousness and focal neurologic indicators.","Seizure/confusion pattern with depressed GCS and stroke risk profile."],
    4:["Severe metabolic derangement in glucose/electrolytes with altered sensorium.","Endocrine-metabolic instability with marked lab abnormalities."],
    5:["Medication/toxic exposure pattern with CNS depression and unstable vitals.","Drug-induced collapse pattern supported by medication history and low GCS."],
    6:["Hemorrhagic shock profile with hypotension, tachycardia, low hemoglobin, and bleeding risk.","Probable internal/external bleed with perfusion failure and anemia."],
}

def tnorm(n,m,s,lo,hi):
    return np.clip(RNG.normal(m,s,n),lo,hi)

def bern(n,p):
    return (RNG.random(n)<p).astype(int)

def urgency(row):
    bp = max(0,(100-row.systolic_bp)*0.45) + max(0,(60-row.diastolic_bp)*0.25)
    gcs = max(0,(15-row.consciousness_score))*3.0
    hr = max(0,abs(row.heart_rate-85)-15)*0.35
    sp = max(0,(95-row.spo2))*1.8
    lab = 0
    lab += 8 if row.lactate>=4 else (4 if row.lactate>=2.5 else 0)
    lab += 6 if row.troponin>=1.2 else 0
    lab += 5 if row.hemoglobin<=8.5 else 0
    lab += 4 if (row.sodium<125 or row.sodium>150 or row.potassium<2.8 or row.potassium>6.2) else 0
    return min(100, round(bp+gcs+hr+sp+lab,2))

def build_class(c,n):
    s=NUMERIC_SPECS[c]
    d={
        "patient_age":tnorm(n,*s["age"]),"systolic_bp":tnorm(n,*s["sbp"]),"diastolic_bp":tnorm(n,*s["dbp"]),
        "heart_rate":tnorm(n,*s["hr"]),"respiratory_rate":tnorm(n,*s["rr"]),"spo2":tnorm(n,*s["spo2"]),
        "body_temperature":tnorm(n,*s["temp"]),"consciousness_score":tnorm(n,*s["gcs"]),"hemoglobin":tnorm(n,*s["hb"]),
        "wbc_count":tnorm(n,*s["wbc"]),"lactate":tnorm(n,*s["lactate"]),"creatinine":tnorm(n,*s["creatinine"]),
        "troponin":tnorm(n,*s["troponin"]),"sodium":tnorm(n,*s["sodium"]),"potassium":tnorm(n,*s["potassium"]),
        "gender":RNG.integers(0,2,n)
    }
    if c==4:
        mask=RNG.random(n)<0.5
        g=np.empty(n)
        g[mask]=RNG.uniform(30,65,mask.sum())
        g[~mask]=RNG.uniform(250,450,(~mask).sum())
        d["blood_glucose"]=g
    else:
        d["blood_glucose"]=tnorm(n,*s["glucose"])

    for b in BINARY_FIELDS:
        d[b]=bern(n,BINARY_PROBS[c][b])
    df=pd.DataFrame(d)

    # soft dependencies
    cp = df["chest_pain"]==1
    df.loc[cp & (RNG.random(n)<0.35),"palpitations"]=1
    dm = df["diabetes_history"]==1
    df.loc[dm & (RNG.random(n)<0.45),"insulin_or_diabetic_drug_use"]=1
    ch = df["cardiac_history"]==1
    df.loc[ch & (RNG.random(n)<0.5),"cardiac_drug_use"]=1
    un = df["unconsciousness"]==1
    df.loc[un,"consciousness_score"] = np.minimum(df.loc[un,"consciousness_score"], tnorm(un.sum(),7,2,3,11))
    fe = df["fever"]==1
    df.loc[fe,"body_temperature"] = np.maximum(df.loc[fe,"body_temperature"],100.0)
    if c==6:
        rs = df["recent_surgery"]==1
        df.loc[rs,"hemoglobin"] = np.clip(df.loc[rs,"hemoglobin"]-RNG.uniform(0.2,0.8,rs.sum()),4.5,11)

    # hard dependencies
    if c==5:
        meds=["antihypertensive_use","insulin_or_diabetic_drug_use","anticoagulant_use","psychiatric_drug_use","cardiac_drug_use"]
        m=df[meds].sum(axis=1)==0
        for idx in df.index[m]:
            df.at[idx,RNG.choice(meds)]=1
    if c==6:
        m=(df[["visible_bleeding_signs","recent_surgery","anticoagulant_use"]].sum(axis=1)==0)
        flip=RNG.choice(["visible_bleeding_signs","recent_surgery","anticoagulant_use"],m.sum())
        for i,col in zip(df.index[m],flip): df.at[i,col]=1
    if c==1:
        strong=(df["chest_pain"]+df["palpitations"]+df["ecg_abnormality_present"]+df["cardiac_history"]+(df["troponin"]>1.2).astype(int))
        low=strong<2
        for idx in df.index[low]:
            opts=["chest_pain","palpitations","ecg_abnormality_present","cardiac_history"]
            RNG.shuffle(opts)
            for k in opts[:2]: df.at[idx,k]=1
    if c==2:
        septic=(df["fever"]+(df["wbc_count"]>12).astype(int)+(df["lactate"]>2.5).astype(int)+(df["systolic_bp"]<95).astype(int)+(df["heart_rate"]>110).astype(int))
        low=septic<3
        df.loc[low,"fever"]=1; df.loc[low,"wbc_count"]=np.maximum(df.loc[low,"wbc_count"],13); df.loc[low,"lactate"]=np.maximum(df.loc[low,"lactate"],2.6)
    if c==4:
        severe=((df["blood_glucose"]<70)|(df["blood_glucose"]>250)).astype(int)
        metab=(df["diabetes_history"]+df["insulin_or_diabetic_drug_use"]+severe+((df["sodium"]<125)|(df["sodium"]>145)).astype(int)+((df["potassium"]<3.0)|(df["potassium"]>5.8)).astype(int))
        low=metab<2
        df.loc[low,"diabetes_history"]=1; df.loc[low,"insulin_or_diabetic_drug_use"]=1
    if c==3:
        neuro=(df["unconsciousness"]+df["seizure_activity"]+df["dizziness_confusion"]+(df["consciousness_score"]<10).astype(int)+df["stroke_history"])
        low=neuro<2
        df.loc[low,"dizziness_confusion"]=1; df.loc[low,"consciousness_score"]=np.minimum(df.loc[low,"consciousness_score"],9)

    # contradiction filter
    mask = (df.systolic_bp>df.diastolic_bp) & ~((df.unconsciousness==1)&(df.consciousness_score>13))
    if c==6: mask &= (df[["visible_bleeding_signs","recent_surgery","anticoagulant_use"]].sum(axis=1)>=1)
    if c==5: mask &= (df[["antihypertensive_use","insulin_or_diabetic_drug_use","anticoagulant_use","psychiatric_drug_use","cardiac_drug_use"]].sum(axis=1)>=1)
    if c==1: mask &= ((df["chest_pain"]+df["palpitations"]+df["ecg_abnormality_present"]+df["cardiac_history"]+(df["troponin"]>1.2).astype(int))>=2)
    if c==2: mask &= ((df["fever"]+(df["wbc_count"]>12).astype(int)+(df["lactate"]>2.5).astype(int)+(df["systolic_bp"]<95).astype(int)+(df["heart_rate"]>110).astype(int))>=3)
    if c==4:
        sev=((df["blood_glucose"]<70)|(df["blood_glucose"]>250)).astype(int)
        mask &= ((df["diabetes_history"]+df["insulin_or_diabetic_drug_use"]+sev+((df["sodium"]<125)|(df["sodium"]>145)).astype(int)+((df["potassium"]<3.0)|(df["potassium"]>5.8)).astype(int))>=2)

    df=df[mask].copy()
    while len(df)<n:
        extra=build_class(c,n-len(df))
        df=pd.concat([df,extra],ignore_index=True)
    return df.iloc[:n].copy()

def add_outputs(df,c):
    l1,l2=AMBIG[c]
    p1=RNG.uniform(0.62,0.84,len(df))
    p2=RNG.uniform(0.10,0.25,len(df))
    p3=np.clip(1-p1-p2,RNG.uniform(0.03,0.12,len(df)),0.2)
    p1=1-p2-p3
    df["primary_emergency_class"]=CLASS_LABELS[c]
    df["top3_predictions"]=df.apply(lambda _: f"{CLASS_LABELS[c]}|{CLASS_LABELS[l1]}|{CLASS_LABELS[l2]}",axis=1)
    df["top3_probabilities"]= [f"{a:.2f}|{b:.2f}|{d:.2f}" for a,b,d in zip(p1,p2,p3)]
    df["urgency_score"]=df.apply(urgency,axis=1)
    df["recommended_priority_tests"]=TEST_MAP[c]
    temp=REASON_TEMPLATES[c]
    df["rule_trigger_reason"]=[temp[i] for i in RNG.integers(0,len(temp),len(df))]
    return df

def main():
    all_df=[]
    for c,n in CLASS_COUNTS.items():
        cls=build_class(c,n)
        cls=add_outputs(cls,c)
        all_df.append(cls)
    df=pd.concat(all_df,ignore_index=True)
    df=df.drop_duplicates().reset_index(drop=True)
    while len(df)<30000:
        c=RNG.choice(list(CLASS_COUNTS.keys()))
        ex=add_outputs(build_class(c,200),c)
        df=pd.concat([df,ex],ignore_index=True).drop_duplicates().reset_index(drop=True)
    df=df.iloc[:30000].copy()

    int_cols=["gender"]+BINARY_FIELDS
    for col in int_cols:
        df[col]=df[col].astype(int)
    round_cols=[c for c in df.columns if c not in int_cols+["primary_emergency_class","top3_predictions","top3_probabilities","recommended_priority_tests","rule_trigger_reason"]]
    df[round_cols]=df[round_cols].round(2)

    order=["patient_age","gender","systolic_bp","diastolic_bp","heart_rate","respiratory_rate","spo2","body_temperature","blood_glucose","consciousness_score","chest_pain","breathlessness","sweating","seizure_activity","unconsciousness","fever","dizziness_confusion","palpitations","visible_bleeding_signs","diabetes_history","hypertension_history","cardiac_history","stroke_history","recent_surgery","antihypertensive_use","insulin_or_diabetic_drug_use","anticoagulant_use","psychiatric_drug_use","cardiac_drug_use","hemoglobin","wbc_count","lactate","creatinine","troponin","sodium","potassium","ecg_abnormality_present","primary_emergency_class","top3_predictions","top3_probabilities","urgency_score","recommended_priority_tests","rule_trigger_reason"]
    df=df[order]
    df.to_csv("emergency_forecasting_dataset.csv",index=False)

    print("Class counts:")
    print(df["primary_emergency_class"].value_counts())
    print("\nNull counts:")
    print(df.isnull().sum())
    print("\nSample rows:")
    print(df.sample(5, random_state=42))

if __name__=="__main__":
    main()
