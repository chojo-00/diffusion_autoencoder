import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from dataset.dataset import CLASS_LIST
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc


def compute_AUCs(overall_gts, overall_logits, num_classes):
    labels_one_hot = label_binarize(overall_gts, classes=[i for i in range(num_classes)])
    # print(len(labels_one_hot), len(overall_logits))
    auc = roc_auc_score(labels_one_hot, overall_logits, average="macro", multi_class="ovr")
            
    return auc # mean_auc

def plot_multiclass_roc(opt, log, overall_gts, overall_logits, num_classes):
    overall_gts = np.array(overall_gts)
    overall_logits = np.array(overall_logits)
    labels_one_hot = label_binarize(overall_gts, classes=[i for i in range(num_classes)])

    # plot ROC curve
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}

    # print(overall_gts, len(overall_gts))
    # print(overall_logits, np.array(overall_logits).shape)

    for i in range(num_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(labels_one_hot[:, i], overall_logits[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        plt.plot(fpr_dict[i], tpr_dict[i], color=colors[i], lw=2, 
                 label=f"Class {CLASS_LIST[i]} (AUROC = {roc_auc_dict[i]:.2f})")
        
    # Micro-average ROC Curve
    fpr_micro, tpr_micro, _ = roc_curve(labels_one_hot.ravel(), overall_logits.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, color='deeppink', linestyle=':', linewidth=4,
             label=f"Micro-average (AUROC = {roc_auc_micro:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("Multi-class ROC Curve", fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)

    plt.savefig(str(opt.load).replace('.pth.tar', '') + '_auroc_curve.png', bbox_inches='tight', dpi=300)
    # plt.savefig(str(opt.load).replace('.pth.tar', '') + '_external_auroc_curve.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    log.info("AUCOR Curve Figure saved!")

def save_confusion_matrix(opt, log, labels, predictions):
    conf_matrix = confusion_matrix(labels, predictions)

    # For external testset (if not predict a specific class at all)
    # conf_matrix_extend = np.zeros((9,9), dtype=int)
    # conf_matrix_extend[:8, :8] = conf_matrix

    # Confusion Matrix Plot
    plt.figure(figsize=(10, 8))
    plt.tight_layout()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_LIST, yticklabels=CLASS_LIST)
    plt.title('Confusion Matrix for Multi-class Classification')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(str(opt.load).replace('.pth.tar', '') + '_conf_matrix.png', bbox_inches="tight", dpi=300)
    # plt.savefig(str(opt.load).replace('.pth.tar', '') + '_external_conf_matrix.png', bbox_inches="tight", dpi=300)
    log.info("Confusion Matrix Figure saved!")
    return conf_matrix

def show_metrics(opt, log, conf_matrix):
    metrics = calculate_metrics(conf_matrix, opt.num_classes)
    for idx, cls in enumerate(CLASS_LIST):
        log.info("=======================================================")
        log.info(f"                     Class: {cls}                     ")
        log.info("=======================================================")
        log.info(f"PPV (Precision): {metrics['PPV'][idx]}")
        log.info(f"Sensitivity (Recall): {metrics['Sensitivity'][idx]}")
        log.info(f"Specificity: {metrics['Specificity'][idx]}")
        log.info(f"F1 Score: {metrics['F1 Score'][idx]}")
        log.info(f"NPV: {metrics['NPV'][idx]}")
        log.info(f"Accuracy: {metrics['Accuracy'][idx]}")

def calculate_metrics(conf_matrix, num_classes):
    sensitivity = []
    specificity = []
    ppv = []  # Positive Predictive Value
    npv = []  # Negative Predictive Value
    acc = []
    f1 = []
    
    # 전체 샘플 수
    total_samples = np.sum(conf_matrix)
    
    for i in range(num_classes):
        # True Positives (TP)
        TP = conf_matrix[i, i]
        
        # False Positives (FP)
        FP = np.sum(conf_matrix[:, i]) - TP
        
        # False Negatives (FN)
        FN = np.sum(conf_matrix[i, :]) - TP
        
        # True Negatives (TN)
        TN = total_samples - (TP + FP + FN)
        
        # Specificity
        specificity_value = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificity.append(np.round(specificity_value, 3))
        
        # Sensitivity (Recall)
        sensitivity_value = TP / (TP + FN) if (TP + FN) > 0 else 0
        sensitivity.append(np.round(sensitivity_value, 3))
        
        # Positive Predictive Value (Precision)
        ppv_value = TP / (TP + FP) if (TP + FP) > 0 else 0
        ppv.append(np.round(ppv_value, 3))
        
        # Negative Predictive Value
        npv_value = TN / (TN + FN) if (TN + FN) > 0 else 0
        npv.append(np.round(npv_value, 3))

        # Accuracy
        acc_value = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
        acc.append(np.round(acc_value, 3))
        
        # F1 Score
        f1_value = 2 * (ppv_value * sensitivity_value) / (ppv_value + sensitivity_value) if (ppv_value + sensitivity_value) > 0 else 0
        f1.append(np.round(f1_value, 3))
    
    return {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "Accuracy": acc,
        "F1 Score": f1
    }