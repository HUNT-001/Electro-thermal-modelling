import json
import re
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.titlesize": 13})

DATA_DIR   = Path(r"E:\IML\measurement")
OUTPUT_DIR = Path(r"E:\IML\processed_data_v8\tripB_only_final_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR  = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE       = 42
MIN_ROWS_PER_FILE  = 200
TARGET_CLIP_LOW    = 0.0
TARGET_CLIP_HIGH   = 7000.0
ACTIVE_THRESHOLD_W = 120.0
ROUND_STEP_W       = 40.0
LEAKAGE_PATTERNS   = ["requested heating power","heating power can",
                      "heating power lin","heater voltage","heater current","heater signal"]

def log(msg): print(msg, flush=True)

def clean_col(x):
    x = str(x).replace("\ufeff"," ").strip().lower()
    x = re.sub(r"([a-z])([A-Z])",r"\1 \2",x)
    x = re.sub(r"([A-Z]+)([A-Z][a-z])",r"\1 \2",x)
    x = x.replace("°"," ")
    x = re.sub(r"[\[\]\(\){}]"," ",x)
    for ch in "/\\-_.,;:": x = x.replace(ch," ")
    x = x.replace("soc"," state of charge ")
    return re.sub(r"\s+"," ",x).strip()

def smape(y_true, y_pred):
    y_true,y_pred = np.asarray(y_true,float),np.asarray(y_pred,float)
    d = np.abs(y_true)+np.abs(y_pred); r = np.zeros_like(y_true)
    m = d!=0; r[m] = 2*np.abs(y_pred[m]-y_true[m])/d[m]
    return float(r.mean()*100)

def round_step(x): return np.round(np.asarray(x,float)/ROUND_STEP_W)*ROUND_STEP_W

def find_target(columns):
    canon=[clean_col(c) for c in columns]
    for key in ["requested heating power w","requested heating power","heating power lin w",
                "heating power lin","heating power can kw","heating power can w",
                "heating power can","heating power"]:
        for i,c in enumerate(canon):
            if c==key: return i,columns[i],c
    for i,c in enumerate(canon):
        if "heating" in c and "power" in c: return i,columns[i],c
    raise ValueError("No heating power target found")

def unique_cols(columns):
    seen,out={},[]
    for c in columns:
        base=clean_col(c); n=seen.get(base,0)+1; seen[base]=n
        out.append(base if n==1 else f"{base}__dup{n}")
    return out

def read_trip(path):
    for sep,enc in [(";","cp1252"),(";","latin1"),(";","utf-8-sig"),(",","cp1252"),(",","latin1")]:
        try:
            df=pd.read_csv(path,sep=sep,encoding=enc,engine="python")
            if df.shape[1]<=1: continue
            raw=[str(c).strip() for c in df.columns]
            tidx,traw,tcanon=find_target(raw)
            uc=unique_cols(raw); df.columns=uc
            return df,uc[tidx],traw,tcanon,sep,enc
        except Exception: pass
    raise ValueError(f"Cannot load {path.name}")

def load_tripB():
    files=sorted(DATA_DIR.glob("TripB*.csv"))
    if not files: raise FileNotFoundError(DATA_DIR)
    log("="*70); log("TRIPB-ONLY FINAL PIPELINE  —  Full Graphs Edition"); log("="*70)
    frames,report=[],[]
    for p in files:
        try:
            df,tstd,traw,tcanon,sep,enc=read_trip(p)
            df=df.apply(pd.to_numeric,errors="coerce").dropna(how="all").reset_index(drop=True)
            if len(df)<MIN_ROWS_PER_FILE: raise ValueError("too few rows")
            ts=pd.to_numeric(df.iloc[:,df.columns.get_loc(tstd)],errors="coerce")
            if "kw" in tcanon: ts=ts*1000.0
            df["trip_name"]=p.stem; df["row_id"]=np.arange(len(df)); df["target_w"]=ts
            frames.append(df); report.append({"file":p.name,"rows":len(df),"sep":sep,"enc":enc})
            log(f"  Loaded {p.name:<28} rows={len(df):>7}")
        except Exception as e:
            report.append({"file":p.name,"error":str(e)}); log(f"  WARNING {p.name}: {e}")
    pd.DataFrame(report).to_csv(OUTPUT_DIR/"load_report.csv",index=False)
    if not frames: raise RuntimeError("No TripB files loaded")
    return pd.concat(frames,ignore_index=True)

def is_leaky(col): return any(p in clean_col(col) for p in LEAKAGE_PATTERNS)

def engineer_features(df):
    meta={"trip_name","row_id","target_w"}
    cols=[c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c]) and not is_leaky(c)]
    X=df[cols].copy(); h=lambda c: c in X.columns
    if h("battery voltage v") and h("battery current a"):
        X["f_battery_power_w"]=X["battery voltage v"]*X["battery current a"]
    if h("ambient temperature c") and h("cabin temperature sensor c"):
        X["f_cabin_ambient_dt"]=X["cabin temperature sensor c"]-X["ambient temperature c"]
    if h("requested coolant temperature c") and h("coolant temperature heatercore c"):
        X["f_coolant_req_dt"]=X["requested coolant temperature c"]-X["coolant temperature heatercore c"]
    if h("coolant temperature inlet c") and h("heat exchanger temperature c"):
        X["f_hx_inlet_dt"]=X["heat exchanger temperature c"]-X["coolant temperature inlet c"]
    if h("battery temperature c") and h("ambient temperature c"):
        X["f_batt_ambient_dt"]=X["battery temperature c"]-X["ambient temperature c"]
    if h("velocity km h"):
        X["f_velocity_sq"]=X["velocity km h"]**2
        X["f_is_moving"]=(X["velocity km h"]>1.0).astype(int)
    if h("cabin temperature sensor c") and h("requested coolant temperature c"):
        X["f_cabin_coolant_dt"]=X["requested coolant temperature c"]-X["cabin temperature sensor c"]
    if h("ambient temperature c"):
        X["f_warmup_needed"]=np.maximum(0,20.0-X["ambient temperature c"])
    if h("battery temperature c"):
        X["f_battery_cold"]=np.maximum(0,15.0-X["battery temperature c"])
    return X.replace([np.inf,-np.inf],np.nan)

def preprocess(raw):
    df=raw.drop_duplicates().reset_index(drop=True)
    df["target_w"]=pd.to_numeric(df["target_w"],errors="coerce").clip(TARGET_CLIP_LOW,TARGET_CLIP_HIGH)
    df.loc[df["target_w"]<=ACTIVE_THRESHOLD_W,"target_w"]=0.0
    df=df.dropna(subset=["target_w"]).reset_index(drop=True)
    X=engineer_features(df)
    ok=X.notna().mean(axis=1)>0.05
    df=df.loc[ok].reset_index(drop=True); X=X.loc[ok].reset_index(drop=True)
    y=df["target_w"].reset_index(drop=True)
    good=X.notna().mean(); X=X[good[good>=0.10].index].copy()
    return df,X,y

def trip_split(df):
    trips=df[["trip_name"]].drop_duplicates().reset_index(drop=True); g=trips["trip_name"].values
    tr_idx,tmp=next(GroupShuffleSplit(1,test_size=0.30,random_state=RANDOM_STATE).split(trips,groups=g))
    train_t=set(trips.iloc[tr_idx]["trip_name"]); tmp_df=trips.iloc[tmp].reset_index(drop=True)
    v_idx,te_idx=next(GroupShuffleSplit(1,test_size=0.50,random_state=RANDOM_STATE).split(tmp_df,groups=tmp_df["trip_name"].values))
    val_t=set(tmp_df.iloc[v_idx]["trip_name"]); test_t=set(tmp_df.iloc[te_idx]["trip_name"])
    return df["trip_name"].isin(train_t),df["trip_name"].isin(val_t),df["trip_name"].isin(test_t)

def select_features(X_tr,y_tr,X_va,X_te,top_n=40,max_corr=0.993):
    nz=X_tr.nunique(dropna=True); X_tr=X_tr[nz[nz>1].index]
    X_va=X_va[X_tr.columns]; X_te=X_te[X_tr.columns]
    cr=X_tr.apply(lambda s:s.corr(y_tr)).abs().replace([np.inf,-np.inf],np.nan).fillna(0)
    sel=cr.sort_values(ascending=False).head(min(top_n,len(cr))).index.tolist()
    X_tr,X_va,X_te=X_tr[sel],X_va[sel],X_te[sel]
    upper=X_tr.corr().abs().where(np.triu(np.ones((len(sel),len(sel))),k=1).astype(bool))
    drop=[c for c in upper.columns if any(upper[c]>max_corr)]
    return X_tr.drop(columns=drop),X_va.drop(columns=drop),X_te.drop(columns=drop)

def metrics_dict(y_true,y_pred,split):
    yt,yp=np.asarray(y_true),np.asarray(y_pred)
    mse=mean_squared_error(yt,yp); active=yt>ACTIVE_THRESHOLD_W
    r2_act=r2_score(yt[active],yp[active]) if active.sum()>1 else float("nan")
    mae_act=mean_absolute_error(yt[active],yp[active]) if active.sum()>1 else float("nan")
    idle_acc=accuracy_score(yt<=ACTIVE_THRESHOLD_W,yp<=ACTIVE_THRESHOLD_W)
    return {"split":split,"rows":int(len(yt)),"active_rows":int(active.sum()),
            "mae":float(mean_absolute_error(yt,yp)),"mse":float(mse),"rmse":float(np.sqrt(mse)),
            "r2":float(r2_score(yt,yp)),"smape_pct":float(smape(yt,yp)),
            "mae_active_only":float(mae_act),"r2_active_only":float(r2_act),
            "smape_active_only_pct":float(smape(yt[active],yp[active]) if active.sum()>1 else float("nan")),
            "idle_detection_accuracy":float(idle_acc)}

def two_stage_predict(clf,reg,X_df,thr=0.50):
    prob=clf.predict_proba(X_df)[:,1]; is_active=prob>=thr
    pred=np.zeros(len(X_df))
    if is_active.any():
        pred[is_active]=np.expm1(reg.predict(X_df.iloc[np.where(is_active)[0]]))
    return np.clip(round_step(pred),TARGET_CLIP_LOW,TARGET_CLIP_HIGH)

def save_preds(meta,y_true,y_pred,name):
    out=meta[["trip_name","row_id"]].copy()
    out["actual_w"]=np.asarray(y_true); out["predicted_w"]=np.asarray(y_pred)
    out["abs_error_w"]=np.abs(out["actual_w"]-out["predicted_w"])
    out["residual_w"]=out["actual_w"]-out["predicted_w"]
    out["state"]=np.where(out["actual_w"]>ACTIVE_THRESHOLD_W,"active","idle")
    out.to_csv(OUTPUT_DIR/f"{name}_predictions.csv",index=False)
    return out

# ═══════════════════════════════════════════════════════════════
#  ALL GRAPH FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def graph_target_distribution(y_all, y_tr, y_va, y_te):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, title, color in zip(
        axes,
        [y_all, y_all[y_all > ACTIVE_THRESHOLD_W], None],
        ["Full Target Distribution (all rows)", "Active Rows Only (target > 120W)", "Idle vs Active Split"],
        ["steelblue","tomato",None]
    ):
        if title == "Idle vs Active Split":
            idle_pct  = (y_all <= ACTIVE_THRESHOLD_W).mean() * 100
            active_pct = 100 - idle_pct
            ax.bar(["Idle (0W)","Active (>120W)"], [idle_pct, active_pct], color=["steelblue","tomato"])
            ax.set_ylabel("Percentage of rows (%)"); ax.set_title(title)
            for i,(v,l) in enumerate(zip([idle_pct,active_pct],["Idle","Active"])):
                ax.text(i, v+0.5, f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")
        else:
            ax.hist(data, bins=60, color=color, edgecolor="white", alpha=0.85)
            ax.set_xlabel("Heating Power (W)"); ax.set_ylabel("Row count"); ax.set_title(title)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"01_target_distribution.png", dpi=200, bbox_inches="tight"); plt.close()
    log("  Saved: 01_target_distribution.png")


def graph_per_trip_stats(df_all, y_all):
    df_all = df_all.copy(); df_all["target_w"] = y_all.values
    stats = df_all.groupby("trip_name")["target_w"].agg(["mean","std","median"]).sort_values("mean")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, col, title, color in zip(
        axes, ["mean","std","median"],
        ["Mean Heating Power per Trip (W)","Std Dev of Heating Power per Trip (W)","Median Heating Power per Trip (W)"],
        ["steelblue","orange","seagreen"]
    ):
        ax.barh(stats.index, stats[col], color=color, edgecolor="white")
        ax.set_xlabel("Watts"); ax.set_title(title); ax.axvline(stats[col].mean(), color="red", ls="--", lw=1.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"02_per_trip_target_stats.png", dpi=200, bbox_inches="tight"); plt.close()
    log("  Saved: 02_per_trip_target_stats.png")


def graph_train_val_test_split(df_tr, df_va, df_te, y_tr, y_va, y_te):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    labels=["Train","Validation","Test"]
    counts=[len(y_tr),len(y_va),len(y_te)]
    colors=["steelblue","orange","tomato"]
    axes[0].bar(labels, counts, color=colors, edgecolor="white")
    axes[0].set_title("Row Count per Split"); axes[0].set_ylabel("Rows")
    for i,(v,c) in enumerate(zip(counts,colors)): axes[0].text(i,v+500,f"{v:,}",ha="center",fontweight="bold")

    for arr, label, color in zip([y_tr,y_va,y_te], labels, colors):
        active = np.asarray(arr)[np.asarray(arr)>ACTIVE_THRESHOLD_W]
        axes[1].hist(active, bins=50, alpha=0.55, label=label, color=color, edgecolor="white")
    axes[1].set_xlabel("Heating Power (W)"); axes[1].set_ylabel("Row count")
    axes[1].set_title("Active Power Distribution across Splits"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"03_split_overview.png", dpi=200, bbox_inches="tight"); plt.close()
    log("  Saved: 03_split_overview.png")


def graph_actual_vs_predicted(y_true, y_pred, split_name):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    active = yt > ACTIVE_THRESHOLD_W
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # All rows
    axes[0].scatter(yt, yp, s=5, alpha=0.25, color="steelblue")
    lo,hi = min(yt.min(),yp.min()), max(yt.max(),yp.max())
    axes[0].plot([lo,hi],[lo,hi],"r--",lw=2,label="Perfect prediction")
    axes[0].set_xlabel("Actual Power (W)"); axes[0].set_ylabel("Predicted Power (W)")
    axes[0].set_title(f"{split_name} — All Rows\nR²={r2_score(yt,yp):.4f}  MAE={mean_absolute_error(yt,yp):.1f}W")
    axes[0].legend()

    # Active only
    axes[1].scatter(yt[active], yp[active], s=5, alpha=0.25, color="tomato")
    lo2,hi2 = yt[active].min(), yt[active].max()
    axes[1].plot([lo2,hi2],[lo2,hi2],"b--",lw=2,label="Perfect prediction")
    axes[1].set_xlabel("Actual Power (W)"); axes[1].set_ylabel("Predicted Power (W)")
    r2_a = r2_score(yt[active],yp[active])
    mae_a = mean_absolute_error(yt[active],yp[active])
    axes[1].set_title(f"{split_name} — Active Rows Only\nR²={r2_a:.4f}  MAE={mae_a:.1f}W")
    axes[1].legend()
    plt.tight_layout()
    name = split_name.lower().replace(" ","_")
    plt.savefig(PLOTS_DIR/f"04_actual_vs_pred_{name}.png", dpi=200, bbox_inches="tight"); plt.close()
    log(f"  Saved: 04_actual_vs_pred_{name}.png")


def graph_residuals(y_true, y_pred, split_name):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    res = yt - yp
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(yp, res, s=5, alpha=0.25, color="steelblue")
    axes[0].axhline(0, color="red", lw=2, ls="--")
    axes[0].set_xlabel("Predicted Power (W)"); axes[0].set_ylabel("Residual (W)")
    axes[0].set_title(f"{split_name} — Residual vs Predicted")

    axes[1].hist(res, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", lw=2, ls="--")
    axes[1].axvline(res.mean(), color="orange", lw=2, ls="--", label=f"Mean={res.mean():.1f}W")
    axes[1].set_xlabel("Residual (W)"); axes[1].set_ylabel("Count")
    axes[1].set_title(f"{split_name} — Residual Distribution"); axes[1].legend()
    plt.tight_layout()
    name = split_name.lower().replace(" ","_")
    plt.savefig(PLOTS_DIR/f"05_residuals_{name}.png", dpi=200, bbox_inches="tight"); plt.close()
    log(f"  Saved: 05_residuals_{name}.png")


def graph_metrics_comparison(metrics_df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metric_info = [
        ("r2",           "R² Score (all rows)",            "higher is better"),
        ("r2_active_only","R² Score (active rows only)",   "higher is better"),
        ("rmse",         "RMSE (W)",                       "lower is better"),
        ("mae",          "MAE (W)",                        "lower is better"),
        ("mae_active_only","MAE — Active rows only (W)",   "lower is better"),
        ("idle_detection_accuracy","Idle Detection Accuracy","higher is better"),
    ]
    colors = ["steelblue","orange","tomato"]
    for ax, (col, title, note) in zip(axes.flat, metric_info):
        vals = metrics_df[col].values
        bars = ax.bar(metrics_df["split"], vals, color=colors, edgecolor="white")
        ax.set_title(f"{title}\n({note})", fontsize=11)
        ax.set_ylim(0, max(vals)*1.15 if max(vals)>0 else 1)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.02,
                    f"{v:.3f}" if v<10 else f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
    plt.suptitle("Model Performance Metrics — TripB Winter Trips", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"06_metrics_comparison.png", dpi=200, bbox_inches="tight"); plt.close()
    log("  Saved: 06_metrics_comparison.png")


def graph_per_trip_error(preds_va, preds_te):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, pdf, title in zip(axes, [preds_va, preds_te], ["Validation", "Test"]):
        active = pdf[pdf["state"]=="active"]
        trip_err = active.groupby("trip_name")["abs_error_w"].agg(["mean","median","std"]).sort_values("mean")
        x = np.arange(len(trip_err))
        ax.bar(x, trip_err["mean"], color="steelblue", alpha=0.8, label="Mean Error")
        ax.bar(x, trip_err["median"], color="tomato", alpha=0.6, label="Median Error", width=0.4)
        ax.errorbar(x, trip_err["mean"], yerr=trip_err["std"], fmt="none", color="black", capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(trip_err.index, rotation=45, ha="right")
        ax.set_xlabel("Trip"); ax.set_ylabel("Abs Error (W)")
        ax.set_title(f"{title} — Per-Trip MAE (active rows only)"); ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"07_per_trip_error.png", dpi=200, bbox_inches="tight"); plt.close()
    log("  Saved: 07_per_trip_error.png")


def graph_idle_active_matrix(y_true, y_pred, split_name):
    a = np.where(np.asarray(y_true)<=ACTIVE_THRESHOLD_W,"Idle","Active")
    p = np.where(np.asarray(y_pred)<=ACTIVE_THRESHOLD_W,"Idle","Active")
    mat = pd.crosstab(pd.Series(a,name="Actual"),pd.Series(p,name="Predicted"))
    mat = mat.reindex(index=["Idle","Active"],columns=["Idle","Active"],fill_value=0)
    mat.to_csv(OUTPUT_DIR/f"{split_name.lower()}_idle_active_matrix.csv")
    plt.figure(figsize=(6,5))
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues",
                annot_kws={"size":14,"weight":"bold"})
    total = mat.values.sum()
    correct = mat.values.diagonal().sum()
    plt.title(f"{split_name} — Idle/Active Detection\nAccuracy={correct/total:.4f}  ({correct:,}/{total:,} correct)")
    plt.tight_layout()
    name = split_name.lower().replace(" ","_")
    plt.savefig(PLOTS_DIR/f"08_idle_active_matrix_{name}.png", dpi=200, bbox_inches="tight"); plt.close()
    log(f"  Saved: 08_idle_active_matrix_{name}.png")


def graph_feature_importance(clf, reg, feature_names):
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    for ax, model, title in zip(axes, [clf, reg], ["Stage 1: Classifier (Idle vs Active)","Stage 2: Regressor (Active Power)"]):
        imp = pd.DataFrame({"feature":feature_names,"importance":model.feature_importances_})\
                .sort_values("importance",ascending=False).head(20).iloc[::-1]
        imp.to_csv(OUTPUT_DIR/f"{title[:10].strip().lower().replace(' ','_')}_importance.csv",index=False)
        bars = ax.barh(imp["feature"], imp["importance"], color="steelblue", edgecolor="white")
        for bar, v in zip(bars, imp["importance"]):
            ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=9)
        ax.set_xlabel("Feature Importance Score"); ax.set_title(title, fontweight="bold")
    plt.suptitle("Feature Importance — Top 20 Features per Stage", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"09_feature_importance.png", dpi=200, bbox_inches="tight"); plt.close()
    log("  Saved: 09_feature_importance.png")


def graph_feature_correlation(X_tr):
    cols = (X_tr.corr().abs().mean().sort_values(ascending=False).head(15).index.tolist()
            if X_tr.shape[1]>15 else X_tr.columns.tolist())
    corr = X_tr[cols].corr()
    corr.to_csv(OUTPUT_DIR/"train_corr_matrix.csv")
    plt.figure(figsize=(12,10))
    mask = np.triu(np.ones_like(corr,dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                annot_kws={"size":8}, square=True, linewidths=0.5)
    plt.title("Train Feature Correlation Matrix (Top 15)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"10_feature_correlation.png", dpi=200, bbox_inches="tight"); plt.close()
    log("  Saved: 10_feature_correlation.png")


def graph_sample_trip_timeseries(preds_te, n_trips=3):
    trips = preds_te["trip_name"].unique()[:n_trips]
    fig, axes = plt.subplots(len(trips), 1, figsize=(18, 5*len(trips)))
    if len(trips)==1: axes=[axes]
    for ax, trip in zip(axes, trips):
        sub = preds_te[preds_te["trip_name"]==trip].reset_index(drop=True)
        sample = sub if len(sub)<=3000 else sub.iloc[::len(sub)//3000]
        ax.plot(sample.index, sample["actual_w"], lw=1.2, color="steelblue", label="Actual", alpha=0.9)
        ax.plot(sample.index, sample["predicted_w"], lw=1.2, color="tomato", label="Predicted", alpha=0.8, ls="--")
        ax.fill_between(sample.index, sample["actual_w"], sample["predicted_w"], alpha=0.15, color="orange")
        ax.set_xlabel("Row (time steps)"); ax.set_ylabel("Heating Power (W)")
        ax.set_title(f"{trip} — Actual vs Predicted over Time"); ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"11_sample_trip_timeseries.png", dpi=200, bbox_inches="tight"); plt.close()
    log("  Saved: 11_sample_trip_timeseries.png")


def graph_error_distribution_by_split(preds_tr, preds_va, preds_te):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, pdf, split, color in zip(axes,
                                      [preds_tr,preds_va,preds_te],
                                      ["Train","Validation","Test"],
                                      ["steelblue","orange","tomato"]):
        active = pdf[pdf["state"]=="active"]["abs_error_w"]
        ax.hist(active, bins=60, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(active.mean(), color="black", lw=2, ls="--", label=f"Mean={active.mean():.1f}W")
        ax.axvline(active.median(), color="red", lw=2, ls=":", label=f"Median={active.median():.1f}W")
        ax.set_xlabel("Absolute Error (W)"); ax.set_ylabel("Count")
        ax.set_title(f"{split} — Error Distribution (active rows)"); ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"12_error_distribution.png", dpi=200, bbox_inches="tight"); plt.close()
    log("  Saved: 12_error_distribution.png")


def graph_feature_distributions(X_tr, feature_names, top_n=12):
    top_feats = feature_names[:min(top_n, len(feature_names))]
    rows = (len(top_feats)+3)//4
    fig, axes = plt.subplots(rows, 4, figsize=(20, 4*rows))
    for ax, feat in zip(axes.flat, top_feats):
        data = X_tr[feat].dropna()
        ax.hist(data, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_title(feat[:35], fontsize=9); ax.set_xlabel("Value"); ax.set_ylabel("Count")
    for ax in axes.flat[len(top_feats):]: ax.set_visible(False)
    plt.suptitle("Top Feature Distributions (Training Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"13_feature_distributions.png", dpi=200, bbox_inches="tight"); plt.close()
    log("  Saved: 13_feature_distributions.png")


def graph_summary_dashboard(metrics_df):
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    metric_pairs = [
        ("r2",         "R² Score",       "gs[0,0]"),
        ("r2_active_only","R² Active",   "gs[0,1]"),
        ("rmse",       "RMSE (W)",        "gs[0,2]"),
        ("mae_active_only","MAE Active(W)","gs[0,3]"),
    ]
    colors_bar=["steelblue","orange","tomato"]
    for i, (col, title, _) in enumerate(metric_pairs):
        ax = fig.add_subplot(gs[0, i])
        vals = metrics_df[col].values
        bars = ax.bar(metrics_df["split"], vals, color=colors_bar, edgecolor="white")
        ax.set_title(title, fontweight="bold"); ax.set_ylim(0, max(vals)*1.2 if max(vals)>0 else 1)
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.03,
                    f"{v:.3f}" if v<10 else f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")

    ax_tbl = fig.add_subplot(gs[1, :])
    ax_tbl.axis("off")
    show_cols=["split","rows","active_rows","mae","rmse","r2","r2_active_only","mae_active_only","idle_detection_accuracy"]
    tbl_data = metrics_df[show_cols].copy()
    tbl_data.columns = ["Split","Rows","Active Rows","MAE","RMSE","R²","R² Active","MAE Active","Idle Acc"]
    for col in ["MAE","RMSE","MAE Active"]: tbl_data[col]=tbl_data[col].map("{:.1f}".format)
    for col in ["R²","R² Active","Idle Acc"]: tbl_data[col]=tbl_data[col].map("{:.4f}".format)
    tbl = ax_tbl.table(cellText=tbl_data.values, colLabels=tbl_data.columns,
                       cellLoc="center", loc="center", bbox=[0,0.3,1,0.65])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    for j in range(len(tbl_data.columns)):
        tbl[0,j].set_facecolor("#2c3e50"); tbl[0,j].set_text_props(color="white",fontweight="bold")
    for i in range(1,len(tbl_data)+1):
        bg = "#eaf0fb" if i%2==0 else "white"
        for j in range(len(tbl_data.columns)): tbl[i,j].set_facecolor(bg)
    ax_tbl.set_title("Final Model Results Summary — TripB Winter Trips", fontsize=13, fontweight="bold", pad=10)
    plt.suptitle("EV Heating Power Prediction — 2-Stage XGBoost Model", fontsize=15, fontweight="bold", y=1.01)
    plt.savefig(PLOTS_DIR/"00_summary_dashboard.png", dpi=200, bbox_inches="tight"); plt.close()
    log("  Saved: 00_summary_dashboard.png")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    raw = load_tripB()
    df, X, y = preprocess(raw)
    tr_mask,va_mask,te_mask = trip_split(df)

    df_tr=df[tr_mask].reset_index(drop=True); df_va=df[va_mask].reset_index(drop=True); df_te=df[te_mask].reset_index(drop=True)
    X_tr=X[tr_mask].reset_index(drop=True);  X_va=X[va_mask].reset_index(drop=True);  X_te=X[te_mask].reset_index(drop=True)
    y_tr=y[tr_mask].reset_index(drop=True);  y_va=y[va_mask].reset_index(drop=True);  y_te=y[te_mask].reset_index(drop=True)

    log(f"\nTrain:{len(X_tr)} rows / Val:{len(X_va)} rows / Test:{len(X_te)} rows")

    imp=SimpleImputer(strategy="median")
    X_tr_i=pd.DataFrame(imp.fit_transform(X_tr),columns=X_tr.columns)
    X_va_i=pd.DataFrame(imp.transform(X_va),    columns=X_va.columns)
    X_te_i=pd.DataFrame(imp.transform(X_te),    columns=X_te.columns)
    X_tr_i,X_va_i,X_te_i=select_features(X_tr_i,y_tr,X_va_i,X_te_i)
    log(f"Features selected: {X_tr_i.shape[1]}")

    y_cls_tr=(y_tr>ACTIVE_THRESHOLD_W).astype(int); y_cls_va=(y_va>ACTIVE_THRESHOLD_W).astype(int)
    clf=XGBClassifier(n_estimators=500,max_depth=4,learning_rate=0.05,min_child_weight=5,
                      subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=1.0,
                      use_label_encoder=False,eval_metric="logloss",random_state=RANDOM_STATE,n_jobs=-1)
    try:
        clf.fit(X_tr_i,y_cls_tr,eval_set=[(X_va_i,y_cls_va)],verbose=False,early_stopping_rounds=40)
    except TypeError:
        clf.set_params(early_stopping_rounds=40); clf.fit(X_tr_i,y_cls_tr,eval_set=[(X_va_i,y_cls_va)],verbose=False)

    active_tr=(y_tr>ACTIVE_THRESHOLD_W).values; active_va=(y_va>ACTIVE_THRESHOLD_W).values
    y_reg_tr=np.log1p(y_tr.values[active_tr]); y_reg_va=np.log1p(y_va.values[active_va])
    reg=XGBRegressor(objective="reg:squarederror",n_estimators=1000,max_depth=5,learning_rate=0.03,
                     min_child_weight=5,subsample=0.8,colsample_bytree=0.8,gamma=0.1,
                     reg_alpha=0.1,reg_lambda=1.5,random_state=RANDOM_STATE,n_jobs=-1,eval_metric="rmse")
    try:
        reg.fit(X_tr_i.iloc[active_tr],y_reg_tr,eval_set=[(X_va_i.iloc[active_va],y_reg_va)],verbose=False,early_stopping_rounds=50)
    except TypeError:
        reg.set_params(early_stopping_rounds=50); reg.fit(X_tr_i.iloc[active_tr],y_reg_tr,eval_set=[(X_va_i.iloc[active_va],y_reg_va)],verbose=False)

    pred_tr=two_stage_predict(clf,reg,X_tr_i)
    pred_va=two_stage_predict(clf,reg,X_va_i)
    pred_te=two_stage_predict(clf,reg,X_te_i)

    metrics=pd.DataFrame([metrics_dict(y_tr.values,pred_tr,"Train"),
                           metrics_dict(y_va.values,pred_va,"Validation"),
                           metrics_dict(y_te.values,pred_te,"Test")])
    metrics.to_csv(OUTPUT_DIR/"final_metrics.csv",index=False)

    preds_tr=save_preds(df_tr,y_tr,pred_tr,"train")
    preds_va=save_preds(df_va,y_va,pred_va,"validation")
    preds_te=save_preds(df_te,y_te,pred_te,"test")

    joblib.dump({"clf":clf,"reg":reg,"imputer":imp,"features":X_tr_i.columns.tolist(),
                 "active_threshold":ACTIVE_THRESHOLD_W},OUTPUT_DIR/"tripB_2stage_model.joblib")

    log("\nGenerating all graphs...")
    graph_target_distribution(y, y_tr, y_va, y_te)
    graph_per_trip_stats(df, y)
    graph_train_val_test_split(df_tr,df_va,df_te,y_tr,y_va,y_te)
    for nm,yt,yp in [("Train",y_tr,pred_tr),("Validation",y_va,pred_va),("Test",y_te,pred_te)]:
        graph_actual_vs_predicted(yt,yp,nm)
        graph_residuals(yt,yp,nm)
        graph_idle_active_matrix(yt,yp,nm)
    graph_metrics_comparison(metrics)
    graph_per_trip_error(preds_va,preds_te)
    graph_feature_importance(clf,reg,X_tr_i.columns.tolist())
    graph_feature_correlation(X_tr_i)
    graph_sample_trip_timeseries(preds_te, n_trips=3)
    graph_error_distribution_by_split(preds_tr,preds_va,preds_te)
    graph_feature_distributions(X_tr_i, X_tr_i.columns.tolist())
    graph_summary_dashboard(metrics)

    log("\n"+"="*70)
    log("FINAL METRICS  (TripB winter trips only)")
    log("="*70)
    log(metrics[["split","rows","active_rows","mae","rmse","r2","r2_active_only",
                  "mae_active_only","smape_active_only_pct","idle_detection_accuracy"]].to_string(index=False))
    log(f"\nAll graphs saved to: {PLOTS_DIR}")
    log(f"All data  saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
