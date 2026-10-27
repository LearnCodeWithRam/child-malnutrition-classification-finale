import json

path = r'GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/SAM_MAM_ONC_Out_of_Sample_Classification.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

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

# Locate cell 1
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["source"] = cell1_source
        cell["outputs"] = []
        cell["execution_count"] = None
        print("Updated Cell 1 imports successfully!")
        break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Saved updated notebook.")
