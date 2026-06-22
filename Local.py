import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

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


def rbo_score(list1, list2, p=0.9, k=5):
    """
    Compute rank-biased overlap (RBO) for two ranked lists of feature importance rankings.

    :param list1: ranked list (most important first)
    :param list2: ranked list (most important first)
    :param p: persistence parameter
    :param k: the ranking depth
    """
    if k is None:
        k = max(len(list1), len(list2))

    set1 = set()
    set2 = set()
    score = 0.0

    for d in range(1, k + 1):
        if d <= len(list1):
            set1.add(list1[d - 1])
        if d <= len(list2):
            set2.add(list2[d - 1])

        overlap = len(set1.intersection(set2))
        score += (overlap / d) * (p ** (d - 1))

    # Normalizing the RBO score so the results are from 0 (no overlap) to 1 (perfect overlap)
    # instead of 0 (no overlap) to 0.40951 (perfect overlap)
    max_score = 1 - p**k
    return ((1-p) * score) / max_score


def compute_local_similarity(X_train, X_test, model, lime_random_state=1):
    """
    Computes Spearman correlation between SHAP and LIME explanations for 
    each individual test instance.

    :param X_train: pandas DataFrame of training data 
    :param X_test: pandas DataFrame of testing data
    :param model: Random Forest model
    :param lime_random_state: random state for reproducibility (specifically for LIME)
    """

    # SHAP
    shap_explainer = shap.TreeExplainer(model)
    shap_output = shap_explainer(X_test)
    shap_values = shap_output.values


    # LIME
    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        class_names=get_class_names(model),
        mode='classification',
        discretize_continuous=False,
        random_state=lime_random_state
    )

    predict_proba = make_predict_proba_with_feature_names(model, X_train.columns)

    n_instances = len(X_test)
    probs = model.predict_proba(X_test)
    preds = probs.argmax(axis=1)

    records = []

    for i in range(n_instances):
        # SHAP vector
        if shap_values.ndim == 3:
            shap_vec = shap_values[i, :, preds[i]]
        else:
            shap_vec = shap_values[i, :]

        shap_vec = np.abs(shap_vec)

        # LIME vector
        row = X_test.iloc[i].values
        exp = lime_explainer.explain_instance(row, predict_proba, num_features=X_train.shape[1])
        lime_vec = np.zeros(X_train.shape[1], dtype=float)
        label = preds[i]

        
        local_exp = None # In case of error

        if label in exp.local_exp:
            local_exp = exp.local_exp[label]
        else:
            keys = list(exp.local_exp.keys())
            local_exp = exp.local_exp[keys[0]]

        for feat_idx, weight in local_exp:
            lime_vec[feat_idx] = abs(weight)

        # Spearman correlation
        try:
            corr, _ = spearmanr(shap_vec, lime_vec)
            if np.isnan(corr):
                corr = 0.0
        except Exception:
            corr = 0.0

        # Rank-based overlap (RBO)
        if np.all(shap_vec == 0) or np.all(lime_vec == 0):
            rank_overlap = 0.0
        else:
            # produce ranked lists of feature indices (most important first)
            shap_rank = list(np.argsort(-shap_vec))
            lime_rank = list(np.argsort(-lime_vec))
            rank_overlap = rbo_score(shap_rank, lime_rank, p=0.9, k=5)

        records.append({
            'instance': i,
            'pred_class': int(label),
            'spearman': corr,
            'rank_overlap': rank_overlap
        })

    return pd.DataFrame(records)



def compare_shap_lime_local(df, target, noise, random_state, n):
    """
    Runs experiment over multiple random states and computes local Spearman correlations and
    Rank-Bias overlap between SHAP and LIME explanations.

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

        # Prepare data + model
        X_train, X_test, model, rf_accuracy = OnR.data_preperation(df, target, noise, 
                                                                   current_random_state)

        accuracy += rf_accuracy

        # Compute local Spearman correlations and Rank-Bias Overlap
        run_df = compute_local_similarity(
            X_train=X_train,
            X_test=X_test,
            model=model,
            lime_random_state = current_random_state
        )

        run_df['run'] = i
        run_df['random_state'] = current_random_state

        all_runs.append(run_df)

    accuracy /= n   # Average accuracy over the runs

    all_runs_df = pd.concat(all_runs, ignore_index=True)    # Combine all runs

    # Saving to csv
    all_runs_df.to_csv(f'dataframes/local_noise_{int(noise * 100)}.csv', index=False)
    print(f'Saved {noise*100}% correctly to csv!')

    # Info dataframe for Table in report
    info = {
        'noise' : noise,
        'mean_spearman': all_runs_df['spearman'].mean(),
        'std_spearman': all_runs_df['spearman'].std(),
        'median_spearman': all_runs_df['spearman'].median(),
        'skew_spearman': all_runs_df['spearman'].skew(),

        'mean_rank_overlap': all_runs_df['rank_overlap'].mean(),
        'std_rank_overlap': all_runs_df['rank_overlap'].std(),
        'median_rank_overlap': all_runs_df['rank_overlap'].median(),
        'skew_rank_overlap': all_runs_df['rank_overlap'].skew(),

        'average_accuracy': accuracy}

    info_df = pd.DataFrame([info])
    info_df.to_csv(f'dataframes/local_noise_{int(noise * 100)}_summary.csv', index=False)


    # Save Spearman correlation histogram
    plt.figure(figsize=(8, 6))

    plt.hist(all_runs_df['spearman'], bins=25)

    plt.xlabel('Spearman correlation')
    plt.ylabel('Frequency')

    plt.title(f'SHAP vs LIME Spearman Correlation with noise {noise * 100:.0f}%  \n'
        f'Average Accuracy = {accuracy:.3f}')

    plt.tight_layout()
    plt.xlim(xmin = -1, xmax=1)
    plt.ylim(ymax=10000)  

    plt.savefig(f'graphs/local_spearman_hist_noise_{int(noise * 100)}.png')


    # Save rank-overlap histogram
    plt.figure(figsize=(8, 6))

    plt.hist(all_runs_df['rank_overlap'], bins=25)

    plt.xlabel('Rank-based overlap (RBO)')
    plt.ylabel('Frequency')

    plt.title( f'SHAP vs LIME Rank-based overlap with noise {noise * 100:.0f}%  \n'
        f'Average Accuracy = {accuracy:.3f}')

    plt.tight_layout()
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) 
    plt.xlim(xmax=1)
    plt.ylim(ymax=10000)  


    plt.savefig(f'graphs/local_RBO_hist_noise_top5{int(noise * 100)}.png')

    return all_runs_df


if __name__ == '__main__':
    dataset_configs = [
        (dl.data_fraud, 'is_fraud', 0.0, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.1, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.2, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.3, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.4, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.5, 1, 100)]

    print('Start!🎬')
    for df, target, noise, random_state, n in dataset_configs:
        results_df = (compare_shap_lime_local(df=df, target=target, noise=noise, 
                                              random_state=random_state, n=n,))

        print(f' Done! plots saved for noise: {noise*100}%.')
    print('All done!🎉')
