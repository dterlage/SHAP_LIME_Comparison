import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

import shap
from lime import lime_tabular

import DataLoader as dl
import On_repeat as OnR


def make_predict_proba_with_feature_names(model, feature_names):
    """
    Creates a predict_proba wrapper that preserves feature names.

    :param model: Random Forest model
    :param feature_names: The names of the features which needs to be ranked.
    """
    def predict_proba(x):
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x, columns=feature_names)
        return model.predict_proba(x)
    return predict_proba


def get_class_names(model):
    """
    Returns class labels as strings.

    :param model: Random Forest model
    """
    return [str(c) for c in getattr(model, 'classes_', [0, 1])]


def compute_shap_importance(X_test, model):
    """"
    Computes the global SHAP importance feature rankings.

    :param X_test: pandas DataFrame of testing data
    :param model: Random Forest model
    """
    expl = shap.TreeExplainer(model)
    out = expl(X_test)
    vals = out.values
    if vals.ndim == 3:
        mean_abs = np.abs(vals).mean(axis=0)
        per_feature = mean_abs.mean(axis=1)
    else:
        per_feature = np.abs(vals).mean(axis=0)
    return per_feature


def compute_lime_importance(X_train, X_test, model, feature_names, lime_random_state):
    """
    Computes global feature importance ranking of LIME by computing and averaging 
    the feature importance rankings of all instances.

    :param X_train: pandas DataFrame of training data 
    :param X_test: pandas DataFrame of testing data
    :param model: Random Forest model
    :param feature_names: The names of the features which needs to be ranked.
    :param lime_random_state: random state for reproducibility (specifically for LIME)
    """
    explainer = lime_tabular.LimeTabularExplainer(training_data=X_train.values, 
        feature_names=list(feature_names), class_names=get_class_names(model),
        mode='classification',  discretize_continuous=False, random_state=lime_random_state)

    n_instances = len(X_test)
    idxs = list(range(n_instances))

    n_features = X_train.shape[1]
    accum = np.zeros(n_features, dtype=float)

    predict_proba = make_predict_proba_with_feature_names(model, X_train.columns)
    for i in idxs:
        row = X_test.iloc[i].values
        exp = explainer.explain_instance(row, predict_proba, num_features=n_features)
        for label, feats in exp.local_exp.items():
            for feat_idx, weight in feats:
                accum[feat_idx] += abs(weight)

    mean_abs_lime = accum / n_instances
    return mean_abs_lime


def compare_shap_lime_global(df,target, noise, random_state, n):
    """
    Runs experiment over multiple random states and computes global Spearman correlations and
    Kendall's Tau between SHAP and LIME explanations.

    :param df: pandas DataFrame
    :param target: name of the target column for classification
    :param noise: percentage of labels to corrupt (0 to 1)
    :param random_state: random state for reproducibility
    :param n: number of repeats
    """
    all_runs = []
    accuracy = 0

    for i in range(n):

        current_random_state = random_state + i

        print(f'Run {i + 1}/{n} with noise {noise*100}% and random_state={current_random_state}')

        X_train, X_test, model, rf_accuracy = OnR.data_preperation(df, target, noise,  
                                                                   current_random_state)
        accuracy += rf_accuracy

        feature_names = X_train.columns.to_list()

        # Computing SHAP importances
        shap_imp = compute_shap_importance(X_test, model)

        # Computing LIME importances using the training set for explainer and 
        # the test sest for explanations
        lime_imp = compute_lime_importance(X_train, X_test, model, feature_names, 
                                           lime_random_state=current_random_state)

        run_df = pd.DataFrame({
            'feature': feature_names,
            'shap': shap_imp,
            'lime': lime_imp,
            'run': i
        })

        # rank INSIDE each run
        run_df['shap_rank'] = run_df['shap'].rank(ascending=False, method='average')
        run_df['lime_rank'] = run_df['lime'].rank(ascending=False, method='average')

        all_runs.append(run_df)

    accuracy /= n   # Average accuracy over the runs

    all_runs_df = pd.concat(all_runs, ignore_index=True)

    # Aggregate over runs
    summary_df = (all_runs_df.groupby('feature').agg({
            'shap_rank': ['median'],
            'lime_rank': ['median']}))


    summary_df.columns = ['_'.join(col).strip()for col in summary_df.columns.values]
    summary_df = summary_df.reset_index()

    # Spearman on average ranks (median)
    corr, pval = spearmanr(summary_df['shap_rank_median'], summary_df['lime_rank_median'])

    # Kendall's Tau on average ranks (median)
    tau, pval_tau = kendalltau(summary_df['shap_rank_median'], summary_df['lime_rank_median'])

    # Saving to csv
    summary_df.to_csv(f'dataframes/global_ranks_n{noise*100}_v2.csv', index=False)
    print(f'Saved {noise*100}% correctly to csv!')

    # Dot plot of median rankings of SHAP and LIME
    plt.figure(figsize=(7, 7))

    plt.scatter(summary_df['shap_rank_median'], summary_df['lime_rank_median'])

    for _, r in summary_df.iterrows():
        plt.text(
            r['shap_rank_median'],
            r['lime_rank_median'],
            r['feature'],
            fontsize=8)

    plt.xlabel('Median SHAP rank')
    plt.ylabel('Median LIME rank')

    plt.title( f'SHAP vs LIME Median Ranks with noise {noise * 100}% \n'
        f"Spearman's ρ={corr:.3f}, Kendall's τ ={tau:.3f}, Average Accuracy={accuracy:.3f}")
    
    plt.yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    plt.ylim(ymax=12)  
    plt.xlim(xmax=12)

    plt.tight_layout()
    plt.savefig(f'graphs/global_ranks_n{noise*100}.png')
    print(f'Succesfully saved plot for noise {noise*100}%!')


    return summary_df

if __name__ == '__main__':
    dataset_configs = [
        (dl.data_fraud, 'is_fraud', 0, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.1, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.2, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.3, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.4, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.5, 1, 100)]

    print('Start!🎬')
    for df, target, noise, random_state, n in dataset_configs:
        df = compare_shap_lime_global(df, target, noise, random_state, n)
        print(f' Done! plot saved for noise: {noise*100}%.')

    print('All done!🎉')



