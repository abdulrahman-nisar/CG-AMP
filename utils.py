import numpy as np
from Bio import SeqIO
from sklearn.metrics import roc_auc_score
import torch
import os


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def normalise_feature(feature):
    # ESM embeddings can be variable-length per sequence.
    # NumPy 2.x raises on ragged arrays unless dtype=object is explicit.
    return np.asarray(list(feature), dtype=object)

def get_sequences_and_max_sequence_length(file_path):
    sequences = []
    # seqs = []
    max_len = 0
    for record in SeqIO.parse(file_path, "fasta"):
        seq_len = len(record.seq)
        if seq_len > max_len:
            max_len = seq_len
        sequences.append(str(record.seq))
    return np.array(sequences), max_len#, seqs


def get_metrics(real_score, predict_score):
    real_score, predict_score = real_score.flatten(), predict_score.flatten()
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[
        (sorted_predict_score_num * np.arange(1, 1000) / 1000).astype(np.int32)
    ]
    thresholds_num = thresholds.shape[0]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = predict_score_matrix < thresholds[:, None]
    positive_index = predict_score_matrix >= thresholds[:, None]
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    # Confusion matrix values for each threshold
    TP = predict_score_matrix @ real_score
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score) - TP - FP - FN

    # Metrics over thresholds
    recall_list = np.divide(TP, (TP + FN), out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
    precision_list = np.divide(TP, (TP + FP), out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
    accuracy_list = (TP + TN) / len(real_score)

    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc_list = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)

    # Standard F1
    f1_score_list = np.divide(
        2 * TP,
        (2 * TP + FP + FN),
        out=np.zeros_like(TP, dtype=float),
        where=(2 * TP + FP + FN) != 0,
    )

    max_index = int(np.nanargmax(f1_score_list))
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    mcc = mcc_list[max_index]
    roc_auc = roc_auc_score(real_score, predict_score)
    a = [f1_score, accuracy, recall, precision, mcc, roc_auc]  # auc[0, 0], aupr[0, 0],specificity,
    res = [f"{num:.4f}" for num in a]
    return res

def original_feature(feature, truncated_len):
    feature_truncated = []

    for i in range(len(truncated_len)):
        feature1 = feature[i]
        feature2 = feature1.reshape(feature1.shape[1], feature1.shape[2])
        # feature_truncated.append(feature2[1:truncated_len[i] + 1, :])
        feature_truncated.append(feature2[1:-1, :])
    return np.array(feature_truncated)