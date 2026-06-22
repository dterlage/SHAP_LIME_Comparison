Repository for the experiments described in "Decision trees for extroversion and introversion classification" for the interm assignment for JBC090 Language and AI.

## 📜 Overview

- [🔎 Paper Details]()
    - [♻️ Reproduction]()
    - [🚀 Dependencies]()
- [⭐ Experimental manipulation]()


## 🔎 Paper Details

NEED TO ADD ABSTRACT



### ♻️ Reproduction

To reproduce the global results simply run `Global.py`.  For the local results simply run `Loca.py`.

> \* The code was tested with Python 3.13.3 on Windows.

### 🚀 Dependencies

The code was tested using these libraries and versions:

```

lime==0.2.0.1
matplotlib==3.11.0
numpy==2.4.6
pandas==3.0.3
scikit-learn==1.9.0
scipy==1.17.1
shap==0.52.0


```

## ⭐ Experimental manipulation

> `Dataloader.py` -- line 7
Update the file path:
```python
    data_fraud = pd.read_csv(r'data/fraud.csv')
```

> `On_Repeat.py` -- lines 15
Modify the Random Forest parameters:
```python
    return RandomForestClassifier(random_state=random_state, criterion="entropy", n_estimators=10, max_depth=20)
```

> `Global.py` -- lines 188-194
Modify the dataframe, target column name, noise percentage, random state and number of repeats.
```python
    dataset_configs = [
        (dl.data_fraud, 'is_fraud', 0, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.1, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.2, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.3, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.4, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.5, 1, 100)]
```

> `Local.py` -- lines 37
Modify the Rank-Based Overlap persistence parameter and ranking depth.
```python
def rbo_score(list1, list2, p=0.9, k=5):
```

> `Local.py` -- lines 266-272
Modify the dataframe, target column name, noise percentage, random state and number of repeats.
```python
    dataset_configs = [
        (dl.data_fraud, 'is_fraud', 0, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.1, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.2, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.3, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.4, 1, 100),
        (dl.data_fraud, 'is_fraud', 0.5, 1, 100)]
```
