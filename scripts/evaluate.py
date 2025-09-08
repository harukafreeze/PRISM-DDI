# scripts/evaluate.py (或者一个新的评估脚本 a_detailed_evaluation.py)
import os
import sys
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer
# --- 1. Project Setup and Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import configs.default_config as config
from prism.dataloader import PrismDdiDataloader
from prism.model import PRISM_DDI
# --- 2. 参数设置 ---
# !!! 关键：将这里替换成您训练结束后保存的最佳模型文件名 !!!
BEST_MODEL_FILENAME = "PRISM_DDI_scheduler_run_20250905-025330_best.h5" 
MODEL_PATH = os.path.join(config.MODEL_SAVE_PATH, BEST_MODEL_FILENAME)
# --- 3. 加载模型 ---
print("--- Loading Pre-trained Model ---")
model = PRISM_DDI(config)
model.load_weights(MODEL_PATH)
print(f"Model loaded successfully from {MODEL_PATH}")
# --- 4. 加载测试数据 ---
print("\n--- Loading Test Dataset ---")
dataloader = PrismDdiDataloader(config)
# 重要：设置batch_size为一个合理的值，可以更大一些以加速预测
# 这里我们将整个测试集加载到一个大的batch中，如果内存不足可以分批
test_dataset = dataloader.get_dataset(mode='test')
# --- 5. 进行预测 ---
print("\n--- Predicting on Test Set ---")
all_labels = []
all_predictions = []
# 遍历测试集以获取所有真实标签和预测概率
for inputs, labels in test_dataset:
    predictions = model.predict(inputs)
    all_labels.append(labels.numpy())
    all_predictions.append(predictions)
# 将列表合并成一个大的Numpy数组
y_true = np.concatenate(all_labels, axis=0) # 真实标签 (整数形式, e.g., 0, 1, 2, 3)
y_pred_probs = np.concatenate(all_predictions, axis=0) # 预测概率 (形状: [N, 4])
# --- 6. 计算评估指标 (使用scikit-learn) ---
print("\n--- Calculating Performance Metrics ---")
# 6.1 Accuracy
y_pred_labels = np.argmax(y_pred_probs, axis=1) # 获取预测概率最高的类别
acc = accuracy_score(y_true, y_pred_labels)
# 6.2 多分类 AUROC 和 AUPR
# 首先，需要将整数标签转换为one-hot编码形式以用于计算
lb = LabelBinarizer()
y_true_one_hot = lb.fit_transform(y_true)
if config.NUM_CLASSES == 2: # 二分类特殊处理
     y_true_one_hot = np.hstack((1 - y_true_one_hot, y_true_one_hot))
# OvR (One-vs-Rest) AUROC
# 'macro'平均：对每个类的AUROC取平均值，平等对待每个类
# 'weighted'平均：按每个类的样本数量加权平均
auroc_macro = roc_auc_score(y_true_one_hot, y_pred_probs, average='macro', multi_class='ovr')
auroc_weighted = roc_auc_score(y_true_one_hot, y_pred_probs, average='weighted', multi_class='ovr')
# OvR AUPR (也称为 Average Precision Score)
aupr_macro = average_precision_score(y_true_one_hot, y_pred_probs, average='macro')
aupr_weighted = average_precision_score(y_true_one_hot, y_pred_probs, average='weighted')
# 6.3 其他指标 (F1, Precision, Recall)
f1_macro = f1_score(y_true, y_pred_labels, average='macro')
precision_macro = precision_score(y_true, y_pred_labels, average='macro')
recall_macro = recall_score(y_true, y_pred_labels, average='macro')
# --- 7. 打印最终结果 ---
print("\n--- Final Test Set Performance Report ---")
print(f"  - Accuracy:           {acc * 100:.2f}%")
print("---------------------------------------------")
print(f"  - Macro AUROC:          {auroc_macro:.4f}")
print(f"  - Weighted AUROC:       {auroc_weighted:.4f}")
print("---------------------------------------------")
print(f"  - Macro AUPR (AP):      {aupr_macro:.4f}")
print(f"  - Weighted AUPR (AP):   {aupr_weighted:.4f}")
print("---------------------------------------------")
print(f"  - Macro F1-Score:       {f1_macro:.4f}")
print(f"  - Macro Precision:      {precision_macro:.4f}")
print(f"  - Macro Recall:         {recall_macro:.4f}")
print("---------------------------------------------")
