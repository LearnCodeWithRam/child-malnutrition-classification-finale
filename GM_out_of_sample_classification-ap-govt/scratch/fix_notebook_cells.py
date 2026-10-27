import json

path = r'GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/SAM_MAM_ONC_Out_of_Sample_Classification.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 1 imports to include f_classif
cell1_source = [
    "import os\n",
    "import warnings\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.transforms as transforms\n",
    "from matplotlib.patches import Ellipse\n",
    "import seaborn as sns\n",
    "from scipy.linalg import svd\n",
    "from scipy.spatial.distance import cdist\n",
    "\n",
    "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n",
    "from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
    "from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.feature_selection import f_classif\n",
    "from sklearn.metrics import (\n",
    "    classification_report, confusion_matrix, accuracy_score, f1_score,\n",
    "    precision_score, recall_score, roc_auc_score, roc_curve, auc\n",
    ")\n",
    "from sklearn.decomposition import PCA\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Style configuration\n",
    "np.random.seed(42)\n",
    "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
    "plt.rcParams['figure.figsize'] = (12, 8)\n",
    "plt.rcParams['font.size'] = 11\n",
    "\n",
    'print("Environment successfully configured and dependencies loaded!", flush=True)\n',
    'print(f"NumPy: {np.__version__} | Pandas: {pd.__version__}", flush=True)\n'
]

# Update Cell 23 to include X_scaled definition
cell23_source = [
    "# Filter dataset for SAM and ONC classes only\n",
    "df_2c = df[df['Class_Type'].isin(['sam', 'onc'])].copy().reset_index(drop=True)\n",
    "n_2c = len(df_2c)\n",
    "labels_2c = df_2c['Class_Type'].values\n",
    "y_binary = np.array([1 if l == 'sam' else 0 for l in labels_2c])\n",
    "\n",
    "# Extract raw feature matrix and scaled feature matrix (38 landmark coordinates) for SAM vs ONC 2-class analysis\n",
    "X_raw = df_2c[x_cols + y_cols].values\n",
    "scaler_2c = StandardScaler()\n",
    "X_scaled = scaler_2c.fit_transform(X_raw)\n",
    "feature_names = [f'X{i}' for i in landmark_ids] + [f'Y{i}' for i in landmark_ids]\n",
    "\n",
    "print('='*60, flush=True)\n",
    "print('2-CLASS DATASET OVERVIEW (SAM vs ONC)', flush=True)\n",
    "print('='*60, flush=True)\n",
    "print(f'Total Samples (N): {n_2c}', flush=True)\n",
    "print(f'Class Distribution: SAM = {sum(y_binary==1)}, ONC = {sum(y_binary==0)}', flush=True)\n",
    "print(f'Total Feature Coordinates: {len(x_cols) + len(y_cols)} (19 landmarks x 2 dimensions)', flush=True)\n"
]

for idx, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "import numpy as np" in src:
            cell["source"] = cell1_source
            cell["outputs"] = []
            cell["execution_count"] = None
        elif "2-CLASS DATASET OVERVIEW" in src:
            cell["source"] = cell23_source
            cell["outputs"] = []
            cell["execution_count"] = None

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Updated Cell 1 and Cell 23 successfully!")
