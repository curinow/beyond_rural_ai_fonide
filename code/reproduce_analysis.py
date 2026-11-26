"""
Reproduces all tables, figures and statistics reported in

  Beyond Territorial Gaps: Attitudes, Skills, and Artificial Intelligence Adoption Profiles
  among Rural Primary Teachers in Chile

Usage
-----
    python code/reproduce_analysis.py --data data/ai_rural_teachers_chile_full.csv --out output

The script also runs on the data set (without gender and age). In that case the
models that adjust for gender and age are estimated without these covariates and the results differ
slightly from those reported in the article (see README). 

The dataset is available from the authors upon reasonable request.

Outputs (in --out); numbers follow the article (Table 1 and 2 are not produced by this script)
------------------
    table3_descriptives_by_region.csv
    table4_variance_decomposition.csv
    table5_ols_models.csv
    table6_profiles.csv
    table7_firth_models.csv
    school_units.csv
    figure1_variance_decomposition.png
    figure2_ai_use_by_school.png
    results.json                       (every statistic cited in the text)
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.power import TTestIndPower

import kmedoids
from firthmodels import FirthLogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, pairwise_distances, roc_auc_score, silhouette_score

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------------------
# Scales
# ----------------------------------------------------------------------------------------------
PREFIX = {
    "use": "use_",            # AI use (3 items)
    "pu": "pu_",              # perceived usefulness (3)
    "peou": "peou_",          # perceived ease of use (3)
    "int": "int_",            # intention to use (3)
    "imp": "imp_",            # perceived importance of AI for education (3)
    "att": "att_",            # general attitude towards AI (3)
    "info": "info_",          # information skills (4)
    "ict": "ict_",            # basic ICT skills (7)
    "acc": "acc_",            # perceived access (3) - not a reliable scale, excluded from analyses
}
LABEL = {
    "use": "AI use", "pu": "Perceived usefulness", "peou": "Perceived ease of use", "int": "Intention to use",
    "imp": "Perceived importance", "att": "General attitude", "info": "Information skills",
    "ict": "Basic ICT skills", "att15": "Attitudes (composite)", "sk11": "Digital skills (composite)",
}
MEASURES = ["use", "pu", "peou", "int", "imp", "att", "info", "ict", "att15", "sk11"]

N_BOOT = 9999        # wild cluster bootstrap replications
SEED_BOOT = 1
N_PAM_STARTS = 50    # random starts for PAM


def cronbach_alpha(x: pd.DataFrame) -> float:
    x = x.dropna()
    k = x.shape[1]
    return k / (k - 1) * (1 - x.var(ddof=1).sum() / x.sum(axis=1).var(ddof=1))


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def load(path: str):
    d = pd.read_csv(path)
    items = {k: [c for c in d.columns if c.startswith(p)] for k, p in PREFIX.items()}
    items["att15"] = items["int"] + items["peou"] + items["pu"] + items["imp"] + items["att"]
    items["sk11"] = items["info"] + items["ict"]
    for k, cols in items.items():
        d["I_" + k] = d[cols].mean(axis=1)          # mean of available items, 1-5 metric
        d["z" + k] = zscore(d["I_" + k])
    d["araucania"] = (d.region == "Araucania").astype(int)
    has_demo = {"gender", "age"}.issubset(d.columns)
    if has_demo:
        d["female"] = (d.gender == "female").astype(int)
        d["zage"] = zscore(d.age)
    d["school"] = d.school_unit
    return d, items, has_demo


# ----------------------------------------------------------------------------------------------
# Wild cluster restricted bootstrap (Webb six-point weights)
# ----------------------------------------------------------------------------------------------
def _cluster_t(X, y, idx, G, j):
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    S = np.zeros((X.shape[1], X.shape[1]))
    for g in range(G):
        m = idx == g
        s = X[m].T @ e[m]
        S += np.outer(s, s)
    n, k = X.shape
    V = G / (G - 1) * (n - 1) / (n - k) * XtXi @ S @ XtXi
    return b[j] / np.sqrt(V[j, j])


def wild_cluster_bootstrap_p(formula, data, var, B=N_BOOT, seed=SEED_BOOT):
    yv, X = patsy.dmatrices(formula, data, return_type="dataframe")
    cl = data.loc[X.index, "school"].values
    uc = np.unique(cl)
    idx = np.searchsorted(uc, cl)
    G = len(uc)
    j = list(X.columns).index(var)
    Xa, y = X.values, yv.values.ravel()
    t0 = _cluster_t(Xa, y, idx, G, j)
    Xr = np.delete(Xa, j, axis=1)                       # restricted model (H0: beta_j = 0)
    yhat = Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
    e = y - yhat
    w6 = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
    rng = np.random.default_rng(seed)
    count = sum(abs(_cluster_t(Xa, yhat + e * rng.choice(w6, G)[idx], idx, G, j)) >= abs(t0) for _ in range(B))
    return (count + 1) / (B + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="output")
    ap.add_argument("--boot", type=int, default=N_BOOT, help="wild cluster bootstrap replications")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d, items, has_demo = load(args.data)
    demo = ["female", "zage"] if has_demo else []
    if not has_demo:
        print("NOTE: gender and age not in data set; models are estimated without these covariates.")
    R = {"data_file": os.path.basename(args.data), "demographics_included": has_demo}

    # ------------------------------------------------------------------ sample & reliability
    R["n"] = len(d)
    R["n_by_region"] = d.region.value_counts().to_dict()
    R["teachers_per_school_unit"] = d.groupby("school").size().to_dict()
    if has_demo:
        R["women"] = int(d.female.sum())
        R["age"] = {"mean": round(d.age.mean(), 1), "sd": round(d.age.std(), 1),
                    "min": int(d.age.min()), "max": int(d.age.max()),
                    "by_region": d.groupby("region").age.agg(["mean", "std"]).round(1).to_dict()}
    R["alpha"] = {k: round(cronbach_alpha(d[v]), 3) for k, v in items.items()}
    item_cols = sum((items[k] for k in PREFIX), [])
    R["respondents_with_missing_items"] = int(d[item_cols].isna().any(axis=1).sum())

    # ------------------------------------------------------------------ Table 2
    nA, nM = int((d.araucania == 1).sum()), int((d.araucania == 0).sum())
    pw = TTestIndPower()
    R["power"] = {"mde_80": round(pw.solve_power(nobs1=nM, ratio=nA / nM, alpha=.05, power=.8), 2),
                  "power_d03": round(pw.power(.3, nM, .05, ratio=nA / nM), 2),
                  "power_d05": round(pw.power(.5, nM, .05, ratio=nA / nM), 2)}
    rows = []
    for k in MEASURES:
        v = "I_" + k
        x, y = d.loc[d.araucania == 1, v].dropna(), d.loc[d.araucania == 0, v].dropna()
        t = stats.ttest_ind(x, y, equal_var=False)
        sp = np.sqrt((x.var() + y.var()) / 2)
        diff = x.mean() - y.mean()
        dcoh = diff / sp
        se_d = np.sqrt(1 / len(x) + 1 / len(y) + dcoh ** 2 / (2 * (len(x) + len(y))))
        se = np.sqrt(x.var() / len(x) + y.var() / len(y))
        dfw = se ** 4 / ((x.var() / len(x)) ** 2 / (len(x) - 1) + (y.var() / len(y)) ** 2 / (len(y) - 1))
        tost = {}
        for b in (.3, .5):
            B = b * sp
            tost[b] = max(1 - stats.t.cdf((diff + B) / se, dfw), stats.t.cdf((diff - B) / se, dfw))
        ci90 = (diff - stats.t.ppf(.95, dfw) * se) / sp, (diff + stats.t.ppf(.95, dfw) * se) / sp
        rows.append({"measure": LABEL[k], "items": len(items[k]), "alpha": round(R["alpha"][k], 2),
                     "araucania_M": round(x.mean(), 2), "araucania_SD": round(x.std(), 2),
                     "metropolitan_M": round(y.mean(), 2), "metropolitan_SD": round(y.std(), 2),
                     "total_M": round(d[v].mean(), 2), "total_SD": round(d[v].std(), 2),
                     "welch_t": round(t.statistic, 2), "df": round(dfw, 1), "p": round(t.pvalue, 3),
                     "mann_whitney_p": round(stats.mannwhitneyu(x, y).pvalue, 3),
                     "d": round(dcoh, 2), "d_ci95_lo": round(dcoh - 1.96 * se_d, 2), "d_ci95_hi": round(dcoh + 1.96 * se_d, 2),
                     "d_ci90_lo": round(ci90[0], 2), "d_ci90_hi": round(ci90[1], 2),
                     "tost_p_0.3SD": round(tost[.3], 3), "tost_p_0.5SD": round(tost[.5], 3)})
    t1 = pd.DataFrame(rows)
    t1.to_csv(f"{args.out}/table3_descriptives_by_region.csv", index=False)

    # ------------------------------------------------------------------ Table 3 (variance decomposition)
    rows = []
    for k in MEASURES:
        v = "I_" + k
        dd = d.dropna(subset=[v])
        mr = smf.ols(f"{v} ~ araucania", dd).fit()
        ms = smf.ols(f"{v} ~ C(school)", dd).fit()
        comp = sm.stats.anova_lm(mr, ms)
        one = sm.stats.anova_lm(ms)
        nk = dd.groupby("school").size()
        n0 = (len(dd) - (nk ** 2).sum() / len(dd)) / (len(nk) - 1)
        msb, msw = one.mean_sq.iloc[0], one.mean_sq.iloc[1]
        icc = (msb - msw) / (msb + (n0 - 1) * msw)
        rows.append({"measure": LABEL[k], "n": len(dd),
                     "pct_region": round(mr.rsquared * 100, 1),
                     "pct_schools_within_region": round((ms.rsquared - mr.rsquared) * 100, 1),
                     "pct_teachers_within_schools": round((1 - ms.rsquared) * 100, 1),
                     "F": round(comp.F.iloc[1], 2), "df1": int(comp.df_diff.iloc[1]), "df2": int(comp.df_resid.iloc[1]),
                     "p": round(comp["Pr(>F)"].iloc[1], 3), "icc1": round(icc, 2)})
    t2 = pd.DataFrame(rows)
    t2.to_csv(f"{args.out}/table4_variance_decomposition.csv", index=False)

    # ------------------------------------------------------------------ Table 4 (OLS)
    base_m2 = ["zpu", "zpeou", "zimp", "zatt", "zinfo", "zict", "araucania"] + demo
    need = ["zuse", "zatt15", "zsk11", "zint"] + base_m2
    dm = d.dropna(subset=need).copy()
    models = {"Model 1": "zuse ~ " + " + ".join(["zatt15", "zsk11", "araucania"] + demo),
              "Model 2": "zuse ~ " + " + ".join(base_m2),
              "Model 2 + intention": "zuse ~ " + " + ".join(["zint"] + base_m2)}
    rows, R["ols"] = [], {}
    for name, f in models.items():
        m = smf.ols(f, dm).fit()
        mh = smf.ols(f, dm).fit(cov_type="HC3")
        X = sm.add_constant(dm[list(m.params.index[1:])])
        cooks = m.get_influence().cooks_distance[0]
        keep = cooks <= 4 / len(dm)
        key = "zpu" if "zpu" in f else "zatt15"
        mk = smf.ols(f, dm[keep]).fit(cov_type="HC3")
        R["ols"][name] = {"n": int(m.nobs), "R2": round(m.rsquared, 3), "adj_R2": round(m.rsquared_adj, 3),
                          "breusch_pagan_p": round(het_breuschpagan(m.resid, m.model.exog)[1], 3),
                          "vif_max": round(max(variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])), 2),
                          "cooks_excluded": int((~keep).sum()), f"{key}_b_without_influential": round(mk.params[key], 3),
                          f"{key}_p_without_influential": float(mk.pvalues[key])}
        for v in m.params.index[1:]:
            rows.append({"model": name, "predictor": v, "beta": round(m.params[v], 3), "se_hc3": round(mh.bse[v], 3),
                         "p_hc3": round(mh.pvalues[v], 3),
                         "p_wild_cluster_bootstrap": (round(wild_cluster_bootstrap_p(f, dm, v, B=args.boot), 3)
                                                      if name != "Model 2 + intention" else np.nan)})
    pd.DataFrame(rows).to_csv(f"{args.out}/table5_ols_models.csv", index=False)

    # ------------------------------------------------------------------ Profiles (PAM on use items only)
    U = d[items["use"]]
    keep = U.dropna().index
    Zu = ((U - U.mean()) / U.std()).loc[keep]
    D = pairwise_distances(Zu.values)
    sil, labels = {}, {}
    for k in (2, 3, 4):
        best = min((kmedoids.fasterpam(D, k, random_state=s) for s in range(N_PAM_STARTS)), key=lambda q: q.loss)
        labels[k], sil[k] = best.labels, round(silhouette_score(D, best.labels, metric="precomputed"), 3)
    d.loc[keep, "cluster"] = labels[2]
    low = d.loc[keep].groupby("cluster").I_use.mean().idxmin()
    d.loc[keep, "nonadopter"] = (d.loc[keep, "cluster"] == low).astype(int)
    km = KMeans(2, n_init=50, random_state=0).fit_predict(Zu.values)
    ct = pd.crosstab(d.region, d.nonadopter)
    R["profiles"] = {"n": len(keep), "silhouette": sil, "ari_pam_vs_kmeans": round(adjusted_rand_score(labels[2], km), 2),
                     "nonadopters": int(d.nonadopter.sum()), "adopters": int((d.nonadopter == 0).sum()),
                     "nonadopters_by_region": {r: f"{int(ct.loc[r, 1.0])} of {int(ct.loc[r].sum())}" for r in ct.index},
                     "fisher_p_region": round(stats.fisher_exact(ct.values)[1], 3)}
    comp_vars = items["use"] + ["I_use", "I_pu", "I_peou", "I_imp", "I_att", "I_info", "I_ict"] + (["age"] if has_demo else [])
    rows = []
    for v in comp_vars:
        a_, n_ = d.loc[d.nonadopter == 0, v].dropna(), d.loc[d.nonadopter == 1, v].dropna()
        rows.append({"variable": v, "adopters_M": a_.mean(), "adopters_SD": a_.std(),
                     "nonadopters_M": n_.mean(), "nonadopters_SD": n_.std(),
                     "mann_whitney_p": (np.nan if v.startswith("use_") or v == "I_use" else stats.mannwhitneyu(a_, n_).pvalue)})
    t4 = pd.DataFrame(rows)
    # round half up (APA) – avoids binary floating-point artefacts such as 40.15 -> 40.1
    for c in ["adopters_M", "adopters_SD", "nonadopters_M", "nonadopters_SD"]:
        t4[c] = t4[c].apply(lambda x: float(np.floor(x * 100 + 0.5 + 1e-9) / 100))
    t4["mann_whitney_p"] = t4["mann_whitney_p"].round(6)   # 6 decimals: avoids double rounding (e.g., .0155009 -> .016)
    t4.to_csv(f"{args.out}/table6_profiles.csv", index=False)

    # school units
    su = d.groupby(["region", "school"]).agg(n=("I_use", "size"), use_mean=("I_use", "mean"),
                                             nonadopters=("nonadopter", "sum"),
                                             n_complete_use=("nonadopter", "count")).reset_index()
    su["pct_nonadopters"] = (100 * su.nonadopters / su.n_complete_use).round(0)
    su.round(2).to_csv(f"{args.out}/school_units.csv", index=False)

    # ------------------------------------------------------------------ Table 6 (Firth)
    rows, R["firth"] = [], {}
    for name, fe in {"Model A": ["zpu", "zinfo", "zict", "araucania"] + demo,
                     "Model B": ["zatt15", "zsk11", "araucania"] + demo}.items():
        dd = d.dropna(subset=fe + ["nonadopter"])
        y = dd.nonadopter.values.astype(int)
        m = FirthLogisticRegression().fit(dd[fe], y)
        m.lrt()
        ci = np.asarray(m.conf_int(method="pl"))[:len(fe)]     # feature rows first, intercept last
        p = np.asarray(m.lrt_pvalues_)[:len(fe)]
        ml = smf.logit("nonadopter ~ " + " + ".join(fe), dd).fit(disp=0)
        R["firth"][name] = {"n": len(y), "events": int(y.sum()), "mcfadden_R2": round(ml.prsquared, 3),
                            "auc": round(roc_auc_score(y, m.predict_proba(dd[fe])[:, 1]), 2)}
        for i, v in enumerate(fe):
            rows.append({"model": name, "predictor": v, "OR": round(float(np.exp(m.coef_[i])), 2),
                         "ci95_lo": round(float(np.exp(ci[i, 0])), 2), "ci95_hi": round(float(np.exp(ci[i, 1])), 2),
                         "p_penalised_LR": round(float(p[i]), 3)})
    pd.DataFrame(rows).to_csv(f"{args.out}/table7_firth_models.csv", index=False)

    # ------------------------------------------------------------------ Robustness: predictor-based profiles
    Pv = ["zpu", "zpeou", "zimp", "zatt", "zsk11"]
    kp = d.dropna(subset=Pv).index
    D2 = pairwise_distances(d.loc[kp, Pv].values)
    best = min((kmedoids.fasterpam(D2, 2, random_state=s) for s in range(N_PAM_STARTS)), key=lambda q: q.loss)
    d.loc[kp, "pred_profile"] = best.labels
    g = [d.loc[d.pred_profile == c, "I_use"].dropna() for c in (0, 1)]
    R["predictor_profiles"] = {"silhouette": round(silhouette_score(D2, best.labels, metric="precomputed"), 3),
                               "sizes": [len(x) for x in g], "use_means": [round(x.mean(), 2) for x in g],
                               "mann_whitney_p": float(stats.mannwhitneyu(*g).pvalue),
                               "fisher_p_region": round(stats.fisher_exact(pd.crosstab(d.pred_profile, d.region).values)[1], 3)}

    # ------------------------------------------------------------------ Figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"

    vd = t2.set_index("measure")
    show = [LABEL["use"], LABEL["pu"], LABEL["att15"], LABEL["sk11"]]
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    for i, lab in enumerate(show[::-1]):
        left = 0
        for part, col in [("pct_region", BLUE), ("pct_schools_within_region", ORANGE), ("pct_teachers_within_schools", AQUA)]:
            w = max(vd.loc[lab, part], 0)
            ax.barh(i, w, left=left, color=col, edgecolor="white", linewidth=2, height=.6)
            if w >= 4:
                ax.text(left + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=8,
                        color="#0b0b0b" if col == AQUA else "white")
            left += w
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(show[::-1])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of total variance (%)")
    ax.legend(handles=[Patch(color=BLUE, label="Between regions"), Patch(color=ORANGE, label="Between schools within region"),
                       Patch(color=AQUA, label="Between teachers within school")],
              ncol=3, loc="lower center", bbox_to_anchor=(.4, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out}/figure1_variance_decomposition.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    y, ticks, labs = 0, [], []
    for reg, col in [("Araucania", ORANGE), ("Metropolitan", BLUE)]:
        sub = su[su.region == reg].sort_values("use_mean", ascending=False)
        for _, r in sub.iterrows():
            x = d.loc[d.school == r.school, "I_use"].dropna()
            h = stats.t.ppf(.975, len(x) - 1) * x.std() / np.sqrt(len(x))
            ax.plot([max(1, x.mean() - h), min(5, x.mean() + h)], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot(x.mean(), y, "o", ms=8, color=col, mec="white", mew=2)
            ax.text(5.08, y, f"n = {len(x)};  non-adopters {int(r.nonadopters)}/{int(r.n_complete_use)} ({r.pct_nonadopters:.0f}%)",
                    va="center", fontsize=7.5, color="#52514e")
            ticks.append(y)
            labs.append(r.school)
            y += 1
        y += .6
    ax.axvline(d.I_use.mean(), color=GRAY, lw=1, ls="--")
    ax.text(d.I_use.mean() + .04, -0.75, "Overall mean", ha="left", fontsize=7.5, color="#52514e")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labs)
    ax.invert_yaxis()
    ax.set_xlim(1, 5)
    ax.set_xlabel("AI use index (1–5), school mean with 95% CI")
    ax.legend(handles=[Patch(color=ORANGE, label="La Araucanía"), Patch(color=BLUE, label="Metropolitan Region")],
              loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.08, right=.68, bottom=.15, top=.88)
    fig.savefig(f"{args.out}/figure2_ai_use_by_school.png", dpi=300)
    plt.close(fig)

    with open(f"{args.out}/results.json", "w") as fh:
        json.dump(R, fh, indent=1, default=float)
    print(json.dumps(R, indent=1, default=float))


if __name__ == "__main__":
    main()
