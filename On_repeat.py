
#imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import DataLoader as dl



def make_random_forest(random_state):
    """
    Creates a Random Forest model.
    :param random_state: random state for reproducibility
    """
    return RandomForestClassifier(random_state=random_state, criterion="entropy", n_estimators=10, max_depth=20)

def data_preperation(df, target_value, noise_percent, random_state):
    """

    :param df: pandas DataFrame
    :param target_value: name of the target column  (string)
    :param noise_percent: percentage of labels to corrupt (0 to 1)
    :param random_state: random state for reproducibility
    """
    df = dl.corrupt_labels(df, target_value, noise_percent, random_state)

    # split dataset 75/25 with all sets having equal density of the target (stratified split)
    # use random state for reproducibility
    df_train, df_test = train_test_split(df, test_size=0.3, stratify=df.is_fraud, random_state=random_state)

    # We are predicting is_fraud (0=no, 1=yes)
    X_train = df_train.drop(target_value, axis=1).copy()
    y_train = df_train[target_value].copy()

    # Test data
    X_test = df_test.drop(target_value, axis=1).copy()
    y_test = df_test[target_value].copy()

    # Fitting the Random Forest model
    rf = make_random_forest(random_state).fit(X_train, y_train)
    rf_accuracy = round(rf.score(X_test, y_test), 4)

    return X_train, X_test, rf, rf_accuracy





