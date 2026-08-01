# Datasets

The six benchmark datasets are not redistributed in this repository. They are
publicly available from the sources below and must be downloaded and placed in
`data/raw/` under the file names expected by `src/preprocessing/data_loader.py`.

| Dataset | Expected file name | Source |
|---|---|---|
| Parkinson's Disease | `parkinsons.csv` | https://archive.ics.uci.edu/dataset/174/parkinsons |
| PIMA Indians Diabetes | `diabetes.csv` | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |
| Heart Disease (Cleveland) | `heart_disease.csv` | https://archive.ics.uci.edu/dataset/45/heart+disease |
| Breast Cancer Wisconsin | `breast-cancer.csv` | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| Cervical Cancer (Risk Factors) | `kag_risk_factors_cervical_cancer.csv` | https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors |
| Alzheimer's Disease | `alzheimers.csv` | https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset |

## Preprocessing applied by the loaders

- **Heart Disease.** Non-numeric entries (the question marks of the original
  file) are coerced to missing values and the corresponding rows are removed,
  leaving 297 of the 303 instances. The multi-class target is binarised, giving
  137 positive and 160 negative instances.
- **Cervical Cancer.** Non-numeric entries are coerced to missing values and
  imputed with the column mean. `Biopsy` is used as the target.
- **Alzheimer's Disease.** `PatientID` and `DoctorInCharge` are removed and
  missing values are imputed with the column mean. `Diagnosis` is the target.
- **Breast Cancer.** The `id` column is removed and `diagnosis` is mapped to
  1 for malignant and 0 for benign.

Verify the shapes after loading, for example:

```python
from src.preprocessing.data_loader import DataLoader
X, y, names = DataLoader.load_heart()
print(X.shape, int((y == 1).sum()), int((y == 0).sum()))
# (297, 13) 137 160
```
