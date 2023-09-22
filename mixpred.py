import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import Dict, List
import tempfile
import subprocess
import fire
import os


def fit(
    pred_mat: pd.DataFrame,
    mix_mat: pd.DataFrame,
    y: pd.Series,
    covar_mat: pd.DataFrame = None,
):
    """fit model to model variable prediction slope as a function of covariates

    Parameters
    ----------
    pred_mat : pd.DataFrame
        base predictors, PGSs from different GWASs, constant column will be added
    mix_mat : pd.DataFrame
        covariates used to estimate the mixing weights
    covar_mat: pd.DataFrame
        covariates to control for background
    y : pd.Series
        phenotype of interest

    Returns
    -------
    sm.OLS
        fitted model
    """
    index = pred_mat.index
    assert np.all(index == mix_mat.index) and np.all(index == y.index)
    mix_mat = sm.add_constant(mix_mat)
    if covar_mat is not None:
        covar_mat = sm.add_constant(covar_mat)
        assert np.all(index == covar_mat.index)
    else:
        covar_mat = pd.DataFrame(1, index=index, columns=["const"])
    pred_cols, mix_cols, covar_cols = (
        pred_mat.columns,
        mix_mat.columns,
        covar_mat.columns,
    )
    pred_mat, mix_mat = pred_mat.values, mix_mat.values

    n = pred_mat.shape[0]
    assert (mix_mat.shape[0] == n) and (len(y) == n)
    # columns are grouped by pred
    # | pred1 x mix1, ..., pred1 x mixN | pred2 x mix1, ..., pred2 x mixN | ...
    exog = np.hstack([pred_mat[:, [i]] * mix_mat for i in range(pred_mat.shape[1])])
    columns = [
        f"{pred_col}@{mix_col}" for pred_col in pred_cols for mix_col in mix_cols
    ]
    exog = pd.DataFrame(exog, columns=columns, index=index)
    exog = pd.concat([covar_mat, exog], axis=1)

    model = sm.OLS(endog=y.values, exog=exog, missing="drop").fit()
    mix_params_df = pd.DataFrame(0, index=pred_cols, columns=mix_cols, dtype=float)
    for pred_col in pred_cols:
        for mix_col in mix_cols:
            mix_params_df.loc[pred_col, mix_col] = model.params.loc[
                f"{pred_col}@{mix_col}"
            ]
    model.mix_params_df = mix_params_df
    model.pred_cols, model.mix_cols, model.covar_cols = pred_cols, mix_cols, covar_cols
    return model


def predict(
    pred_mat: pd.DataFrame, mix_mat: pd.DataFrame, model, covar_mat: pd.DataFrame = None
):
    index = pred_mat.index
    assert np.all(index == mix_mat.index)
    mix_mat = sm.add_constant(mix_mat)
    if covar_mat is not None:
        assert np.all(index == covar_mat.index)
        covar_mat = sm.add_constant(covar_mat)
    else:
        covar_mat = pd.DataFrame(1, index=index, columns=["const"])
    pred_cols, mix_cols, covar_cols = model.pred_cols, model.mix_cols, model.covar_cols
    assert np.all(pred_cols == model.pred_cols)
    assert np.all(mix_cols == model.mix_cols)
    assert np.all(covar_cols == model.covar_cols)
    assert np.all(mix_cols == model.mix_params_df.columns)

    alpha_dict = dict()
    for pred_col in model.mix_params_df.index:
        alpha_dict[pred_col] = mix_mat.dot(model.mix_params_df.loc[pred_col, :])

    # calculate weighted score
    exog = np.hstack(
        [pred_mat.values[:, [i]] * mix_mat.values for i in range(pred_mat.shape[1])]
    )
    columns = [
        f"{pred_col}@{mix_col}" for pred_col in pred_cols for mix_col in mix_cols
    ]
    exog = pd.concat(
        [covar_mat, pd.DataFrame(exog, columns=columns, index=index)], axis=1
    )
    score = exog.dot(model.params)
    return score, alpha_dict


def calc_inc_r2(df, y_col, pred_cols, covar_cols=[]):
    model0 = sm.OLS(
        endog=df[y_col], exog=sm.add_constant(df[covar_cols]), missing="drop"
    ).fit()

    model1 = sm.OLS(
        endog=df[y_col],
        exog=sm.add_constant(df[covar_cols + pred_cols]),
        missing="drop",
    ).fit()

    return model1.rsquared - model0.rsquared


def simulte_genotype_single_pop(freq: pd.Series, size: int) -> pd.DataFrame:
    """
    Simulate genotype for a single population.

    Parameters
    ----------
    freq : (n_snp,) vector
        Allele frequency of each SNP.
    size : int
        Number of individuals to simulate.

    Returns
    -------
    (n_indiv, n_snp) matrix
    """
    geno = pd.DataFrame(
        np.random.binomial(n=2, p=freq, size=(size, len(freq))), columns=freq.index
    )
    return geno


def simulate_genotype_admix(prop: pd.DataFrame, freq: pd.DataFrame) -> pd.DataFrame:
    """Simulate genotype for admixed population.

    Parameters
    ----------
    prop : pd.DataFrame
        ancestry proportion of each individual.
    freq : pd.DataFrame
        allele frequency of each SNP in each ancestral population.

    Returns
    -------
    pd.DataFrame
        genotype matrix.
    """
    af = prop.dot(freq.T)
    geno = pd.DataFrame(
        np.random.binomial(n=2, p=af), index=af.index, columns=af.columns
    )
    return geno


def simulate_cor_beta(
    df_snp: pd.DataFrame,
    cor_mat: pd.DataFrame,
    varb: float,
    n_sim: int = 10,
    causal_prop: float = 1,
):
    """Simulate beta with correlation.

    Parameters
    ----------
    df_snp : pd.DataFrame
        to provide SNP index
    cor_mat : pd.DataFrame
        correlation across populations
    varb : float
        variance of the genetic component (b ~ N(0, varb / n_snp))
    n_sim : int, optional
        number of simulation, by default 30
    causal_prop : float, optional
        proportion of causal variants, by default 1
    """
    n_snp = df_snp.shape[0]
    pop_list = cor_mat.index
    n_pop = len(pop_list)

    beta_dict = {pop: np.zeros((n_snp, n_sim)) for pop in pop_list}
    for i in range(n_sim):
        causal = np.random.choice(n_snp, size=int(n_snp * causal_prop), replace=False)
        beta = np.random.multivariate_normal(
            mean=np.zeros(n_pop), cov=cor_mat.values, size=len(causal)
        )
        for j, pop in enumerate(pop_list):
            beta_dict[pop][causal, i] = beta[:, j] * np.sqrt(varb / len(causal))
    beta_dict = {
        pop: pd.DataFrame(beta_dict[pop], index=df_snp.index) for pop in beta_dict
    }
    return beta_dict


def simulate_phenotype(
    geno_dict: Dict[str, pd.DataFrame],
    beta_dict: Dict[str, pd.DataFrame],
    vare: float,
    admix_geno: pd.DataFrame = None,
    admix_prop: pd.DataFrame = None,
):
    """Simulate phenotype for multiple populations.

    Parameters
    ----------
    geno_dict : Dict[str, pd.DataFrame]
        pop -> genotype matrix (n_indiv, n_snp)
    beta_dict : Dict[str, pd.DataFrame]
        pop -> beta matrix (n_snp, n_sim)
    vare : float
        variance of the environmental component (e ~ N(0, vare))
        where variance of the genetic component (b ~ N(0, varb / n_snp))
    admix_geno : pd.DataFrame, optional
        genotype matrix of admixed population (n_indiv, n_snp), by default None
    admix_prop : pd.DataFrame, optional
        ancestry proportion of admixed population (n_indiv, n_prop), by default None

    Returns
    -------
    y_dict : Dict[str, pd.DataFrame]
        pop -> phenotype matrix (n_indiv, n_sim)
    """

    assert isinstance(geno_dict, dict)
    assert isinstance(beta_dict, dict)

    pop_list = list(set(geno_dict.keys()) & set(beta_dict.keys()))
    # pop_list = [pop for pop in geno_dict if pop in beta_dict]
    print(f"Reading {len(pop_list)} populations: {','.join(pop_list)}")
    n_pop = len(pop_list)
    _, n_snp = geno_dict[pop_list[0]].shape
    n_sim = beta_dict[pop_list[0]].shape[1]

    # all beta in beta_dict have the same shape of n_snp and n_sim
    assert all(beta_dict[pop].shape[0] == n_snp for pop in pop_list)
    assert all(beta_dict[pop].shape[1] == n_sim for pop in pop_list)
    assert all(geno_dict[pop].shape[1] == n_snp for pop in pop_list)

    y_dict = dict()
    for pop in pop_list:
        geno, beta = geno_dict[pop], beta_dict[pop]
        n_indiv = geno.shape[0]
        g = geno.dot(beta)
        e = np.random.normal(size=(n_indiv, n_sim)) * np.sqrt(vare)
        snr = g.var(axis=0).mean() / e.var(axis=0).mean()
        print(f"{pop}: ratio of signal:noise = {snr:.2f}")
        y = g + e
        y_dict[pop] = y

    # optionally simulate admixed population
    assert (admix_geno is None) == (
        admix_prop is None
    ), "both or none for admix_geno and admix_prop"
    if (admix_geno is not None) and (admix_prop is not None):
        # TODO: allow for two or more admixed populations
        admix_g = np.zeros((admix_geno.shape[0], n_sim))
        for i in tqdm(range(n_sim), desc="Simulating admixed phenotype"):
            beta = pd.concat(
                [beta_dict[pop].iloc[:, i] for pop in pop_list], axis=1
            ).values  # (n_snp, n_pop)
            admix_beta = admix_prop[pop_list].dot(beta.T).values  # (n_indiv, n_snp)
            admix_g[:, i] = np.sum(admix_geno * admix_beta, axis=1)  # (n_indiv, )

        admix_e = np.random.normal(size=admix_g.shape) * np.sqrt(vare)
        admix_y = admix_g + admix_e
        snr = admix_g.var(axis=0).mean() / admix_e.var(axis=0).mean()
        print(f"ADMIX: ratio of signal:noise = {snr:.2f}")
        y_dict["ADMIX"] = pd.DataFrame(admix_y, index=admix_geno.index)

    return y_dict


def gwas(pfile: str, pheno: pd.DataFrame):
    # create temporary folder tempfile

    tmp_dir = tempfile.TemporaryDirectory()
    out = os.path.join(tmp_dir.name, "tmp")

    pheno_path = out + ".plink2_tmp_pheno"
    pheno.index.name = "#IID"
    pheno.to_csv(pheno_path, sep="\t", na_rep="NA")

    cmds = [
        "plink2",
        f"--pfile {pfile}",
        f"--pheno {pheno_path} --no-psam-pheno",
        f"--out {out}",
        "--linear hide-covar omit-ref allow-no-covars",
    ]
    subprocess.call(
        " ".join(cmds), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    # list directory of tmp_dir
    assoc_dict = dict()
    for trait_col in pheno.columns:
        df_assoc = pd.read_csv(f"{out}.{trait_col}.glm.linear", sep="\t")
        assoc_dict[trait_col] = df_assoc[["ID", "BETA", "SE", "P"]].set_index("ID")

    tmp_dir.cleanup()

    return assoc_dict


def write_plink2(out_prefix: str, geno: pd.DataFrame, df_pvar: pd.DataFrame):
    import dapgen

    dapgen.write_pgen(out_prefix + ".pgen", np.ascontiguousarray(geno.T.values))
    dapgen.write_pvar(out_prefix + ".pvar", df_pvar.loc[geno.columns.values, :])
    df_psam = pd.DataFrame({"#IID": geno.index})
    df_psam.to_csv(out_prefix + ".psam", index=False)


def train_cli(df, y_col, pred_cols, mix_cols, out, covar_cols=None):
    """Train MixPred model

    Parameters
    ----------
    df : str
        path to dataframe, delimited by tab, index is at first column
    y_col : str
        column name of phenotype
    pred_cols : str
        column names of variables as predictors, deliminated by ','
    mix_cols : str
        column names of variables as mixture proportions, delimited by ','
    covar_cols : str
        column names of covariates, delimited by ','

    Examples
    --------
        python mixpred.py train-cli --df data.tsv --y-col y --pred-cols EURPRS,AFRPRS --mix-cols PC1,PC2 --out out

    Notes
    -----
    The first variable in `pred_cols` will be taken as baseline, and other variables are subtracted from the baseline.
        [v1 | v2 | v3] -> [v1 | v2 - v1 | v3 - v1]. Then a2, a3 are fitted for v2 - v1 and v3 - v1, respectively.
        a1 can be recovered by 1 - a2 - a3.
    """
    print(
        f"Received parameters: \ntrain-cli\n  "
        + "\n  ".join(f"--{k}={v}" for k, v in locals().items())
    )

    if isinstance(df, str):
        df = pd.read_csv(df, sep="\t", index_col=0)

    if isinstance(pred_cols, str):
        pred_cols = [pred_cols]
    elif isinstance(pred_cols, tuple):
        pred_cols = list(pred_cols)
    else:
        raise ValueError(f"pred_cols must be str or tuple, got {type(pred_cols)}")

    if isinstance(mix_cols, str):
        mix_cols = [mix_cols]
    elif isinstance(mix_cols, tuple):
        mix_cols = list(mix_cols)
    else:
        raise ValueError(f"mix_cols must be str or tuple, got {type(mix_cols)}")

    if covar_cols is not None:
        if isinstance(covar_cols, str):
            covar_cols = [covar_cols]
        elif isinstance(covar_cols, tuple):
            covar_cols = list(covar_cols)
        else:
            raise ValueError(f"covar_cols must be str or tuple, got {type(covar_cols)}")

    model = fit(
        pred_mat=df[pred_cols],
        mix_mat=df[mix_cols],
        y=df[y_col],
        covar_mat=df[covar_cols] if covar_cols else None,
    )
    print(model.summary())
    model.save(out)


def predict_cli(df: str, model, out: str):
    """Apply a trained MixPred model to a new dataset

    Parameters
    ----------
    df : str
        path to dataframe, delimited by tab, index is at first column
    model : str
        path to trained model, `pred_cols`, `mix_cols`, `covar_cols` will be extracted from the model
        `df` will be checked to see if the same set of columns are present
    out : str
        path to output file
        - mixpred: mixed prediction
        - [pred_cols].alpha: the mixing weights of each predictors
    """
    print(
        f"Received parameters: \npredict-cli\n  "
        + "\n  ".join(f"--{k}={v}" for k, v in locals().items())
    )

    if isinstance(df, str):
        df = pd.read_csv(df, sep="\t", index_col=0)

    model = sm.load(model)
    pred_cols, mix_cols, covar_cols = model.pred_cols, model.mix_cols, model.covar_cols

    # remove `const` column, `const` column will be added back in `predict2`
    pred_cols = [col for col in pred_cols if col != "const"]
    mix_cols = [col for col in mix_cols if col != "const"]
    covar_cols = [col for col in covar_cols if col != "const"]

    score, alpha_dict = predict(
        pred_mat=df[pred_cols],
        mix_mat=df[mix_cols],
        model=model,
        covar_mat=df[covar_cols] if covar_cols else None,
    )
    df_out: pd.DataFrame = dict()
    df_out["mixpred"] = score.values
    for col in alpha_dict:
        df_out[col] = alpha_dict[col]
    df_out = pd.DataFrame(df_out, index=df.index)
    df_out.columns = ["mixpred"] + [f"{col}.alpha" for col in pred_cols]
    df_out.to_csv(out, sep="\t")
    print(f"Predictions written to {out}")


def plot_alpha(
    df, alpha_cols, xcol="PC1", ycol="PC2", width=2.5, height=2.4, vmin=None, vmax=None
):
    if vmin is None:
        vmin = df[alpha_cols].values.min()
    if vmax is None:
        vmax = df[alpha_cols].values.max()
    fig, axes = plt.subplots(
        figsize=(len(alpha_cols) * width, height),
        dpi=150,
        ncols=len(alpha_cols),
        sharey=True,
    )
    for ax, col in zip(axes, alpha_cols):
        sc = ax.scatter(
            x=df[xcol],
            y=df[ycol],
            s=0.5,
            c=df[col],
            cmap="Reds",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xticklabels([])
        ax.set_xlabel(xcol)
        ax.set_title(col)

        # add basic stats
        ax.text(
            0.63,
            0.94,
            f"Avg={round(df[col].mean() * 100)}%, "
            f"Std={round(df[col].std() * 100)}%",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
        )
        ax.text(
            0.63,
            0.84,
            f"Range={round(df[col].min() * 100)}%" f"-{round(df[col].max()* 100)}%",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    axes[0].set_ylabel(ycol)
    axes[0].set_yticklabels([])
    fig.colorbar(sc, ax=axes.ravel().tolist(), pad=0.02)

    return fig, axes


if __name__ == "__main__":
    fire.Fire()
