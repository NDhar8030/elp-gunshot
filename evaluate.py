import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from load_data import compute_spectrogram_tf_nolabel, load_config
from train import ensure_length
import numpy as np
import time
from scipy.interpolate import interp1d
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, log_loss
import seaborn as sns
import pandas as pd
import sklearn

config = load_config()

def get_slices_and_labels(partition):
    if partition == 'train':
        data = np.load("elp_slices.npz", allow_pickle=True)
        train_pos = data['train_pos']
        train_neg = data['train_neg']
        slices = [ensure_length(slice[0]) for slice in train_pos] + [ensure_length(slice[0]) for slice in train_neg]
        labels = [slice[1] for slice in train_pos] + [slice[1] for slice in train_neg]
        return slices, labels
    elif partition == 'test':
        data = np.load("elp_slices.npz", allow_pickle=True)
        test_pos = data['test_pos']
        test_neg = data['test_neg']
        slices = [ensure_length(slice[0]) for slice in test_pos] + [ensure_length(slice[0]) for slice in test_neg]
        labels = [slice[1] for slice in test_pos] + [slice[1] for slice in test_neg]
        return slices, labels
    elif partition == 'val':
        data = np.load("elp_slices.npz", allow_pickle=True)
        train_pos = data['train_pos']
        train_neg = data['train_neg']
        trimmed_train_pos_slices = [(ensure_length(slice[0]), slice[1]) for slice in train_pos]
        trimmed_train_neg_slices = [(ensure_length(slice[0]), slice[1]) for slice in train_neg]
        train_full = trimmed_train_pos_slices + trimmed_train_neg_slices
        val = train_full[int(0.8*len(train_full)):]
        slices = [ensure_length(slice[0]) for slice in val]
        labels = [slice[1] for slice in val]
        return slices, labels
    else:
        raise ValueError(f"Invalid partition: {partition}")

def get_preds(partition,model):
    slices, labels = get_slices_and_labels(partition)

    start = time.time()
    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.map(compute_spectrogram_tf_nolabel, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(config['data']['batch_size'])
    ds = ds.prefetch(tf.data.AUTOTUNE)

    start = time.time()
    for _ in ds.take(1):
        pass
    print(f'Time to build and fetch one batch: {time.time() - start:.2f}s')

    start = time.time()
    y_preds = model.predict(ds)
    print(f'Time to process files and perform inference: {time.time()-start}')
    return y_preds

def get_metrics_custom(partition, preds, threshold, make_csv=bool, make_specs=bool, num_specs=int, model_name=str):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    misclassified_files = []

    slices, labels = get_slices_and_labels(partition)
    if len(preds) != len(labels):
        raise ValueError(f"Number of predictions and labels must be the same, got {len(preds)} predictions and {len(labels)} labels")
        
    for i in range(len(labels)):
        pred = ((preds[i])[0])
        label = labels[i]
        index = i
        if pred < threshold and label==0:
            true_negatives += 1
        elif pred < threshold and label==1:
            false_negatives += 1
            prediction_type = 'FN'
            distance = abs(label - pred)
            misclassified_files.append([index, label, pred, prediction_type, distance, threshold])
            if make_specs and false_negatives + false_positives < num_specs:
                print(label, pred, prediction_type, index)
                spec = compute_spectrogram_tf_nolabel(slices[i])
                plt.figure()
                plt.imshow(tf.transpose(spec))
                plt.show()
        elif pred >= threshold and label==0:
            false_positives += 1
            prediction_type = 'FP'
            distance = abs(label - pred)
            misclassified_files.append([index, label, pred, prediction_type, distance, threshold])
            if make_specs and false_negatives + false_positives < num_specs:
                print(label, pred, prediction_type, index)
     
                plt.figure()
                plt.imshow(tf.transpose(spec))
                plt.show()
        elif pred >= threshold and label==1:
            true_positives += 1

    precision = true_positives / (true_positives + false_positives + 1e-5)
    recall = true_positives / (true_positives + false_negatives + 1e-5)
    f1 = 2*(precision*recall) / (precision + recall + 1e-5)
    log_loss = log_loss(labels, preds)
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    print("Evaluation Metrics:")
    print(f"True Positives: {true_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Log loss: {log_loss:.4f}")

    if make_csv:
        misclassified_files = sorted(misclassified_files, key=lambda x: x[4], reverse=True)
        metrics_row = ['', '', '', '', '', '', '', 'Prediction Type', f'True Positives: {true_positives:.2f}', f'False Positives: {false_positives:.2f}', f'True Negatives: {true_negatives:.2f}', f'False Negatives: {false_negatives:.2f}', f'Accuracy: {accuracy:.2f}', f'Recall: {recall:.2f}', f'Precision: {precision:.2f}', f'Log loss: {log_loss:.2f}']
        misclassified_files.insert(0, metrics_row)
        now = datetime.now()
        date = now.strftime("%m-%d-%Y,%H-%M-%S")
        with open(f'outputs//preds//{partition}//{model_name}_misclassified_files_{date}.csv','w', newline='') as out:
            csv_out=csv.writer(out)
            csv_out.writerow(['Index', 'Label', 'Prediction', 'Prediction Type', 'Distance', 'Threshold'])
            for row in misclassified_files:
                newline=''
                csv_out.writerow(row)

    return

def get_pr_curve(partition, y_pred_probs, save_figs=bool, model_name=str):
    y_true, _ = get_slices_and_labels(partition)
    y_pred_probs = [pred for pred in y_pred_probs]
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probs)
    area = auc(recall,precision)

    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Find the best F1 score and its associated precision, recall, threshold
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_f1_precision = precision[best_f1_idx]
    best_f1_recall = recall[best_f1_idx]
    best_f1_threshold = thresholds[best_f1_idx if best_f1_idx < len(thresholds) else best_f1_idx-1]

    # Interpolate precision as a function of recall
    # Note: Recall values decrease in order for precision-recall curve, so we reverse them
    recall_for_interp = recall[::-1]
    precision_for_interp = precision[::-1]

    # Create interpolator
    precision_interp_func = interp1d(recall_for_interp, precision_for_interp, kind='linear', bounds_error=False, fill_value="extrapolate")

    # Get interpolated precision at recall = 0.95
    recall_target = 0.95
    precision_at_95_exact = float(precision_interp_func(recall_target))

    # Plot metrics
    plt.figure(figsize=(10,6))
    plt.plot(recall, precision, label=f'AUC = {area:.3f}', color='blue')
    plt.scatter(best_f1_recall, best_f1_precision, color='red', label=f'Best F1 (F1={best_f1:.3f}, Th={best_f1_threshold:.3f})')
    plt.scatter(best_f1_recall, best_f1_precision, color='green', label=f'Precision of Best F1: {best_f1_precision:.3f}, Recall of Best F1={best_f1_recall:.3f}')

    # Plot the No Skill classifier
    no_skill = len([label for label in y_true if label==1]) / len(y_true)
    plt.plot([0,1], [no_skill, no_skill], ls='--', color='black', label='No-Skill Classifier')

    # Plot the exact point on PR curve
    plt.scatter(0.95, precision_at_95_exact, color='orange', label=f'Interpolated Precision at Recall=0.95: {precision_at_95_exact:.3f}')
    plt.axvline(0.95, color='orange', linestyle='--')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.legend()
    if save_figs:
        now = datetime.now()
        date = now.strftime("%m-%d-%Y,%H-%M-%S")
        plt.savefig(f'outputs//pr//{partition}//{model_name}_PR_curve_{date}.png', bbox_inches='tight')
    plt.show()
    return best_f1_threshold

def get_evaluations(loaded_model, model_name, hist):
    preds = get_preds('val', loaded_model)
    best_f1_threshold = get_pr_curve('val', preds, save_figs=config['evaluation']['show_pr_curve'], model_name=model_name)
    get_metrics_custom('val', preds, best_f1_threshold, make_csv=config['evaluation']['make_metrics_csv'], make_specs=False, num_specs=0, model_name=model_name)

    if config['evaluation']['show_hist_curves']:
        plt.figure(figsize=(10,6))
        plt.plot(hist.history['val_auc_pr'], label='Validation AUC-PR', style='--', color='blue')
        plt.plot(hist.history['train_auc_pr'], label='Training AUC-PR', color='blue')
        plt.plot(hist.history['val_precision'], label='Validation Precision', style='--', color='green')
        plt.plot(hist.history['train_precision'], label='Training Precision', color='green')
        plt.plot(hist.history['val_recall'], label='Validation Recall', style='--', color='red')
        plt.plot(hist.history['train_recall'], label='Training Recall', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training History')
        plt.legend()
        plt.savefig(f'outputs//curves//{model_name}_hist_curves.png', bbox_inches='tight')

    if config['evaluation']['show_confusion_matrix']:
        _, labels = get_slices_and_labels('val')
        cm = sklearn.metrics.confusion_matrix(labels, (preds > best_f1_threshold).astype(int))
        df_cm = pd.DataFrame(cm, index=["Other", "Gunshot"], columns=["Other", "Gunshot"])
        plt.figure(figsize=(8,6))
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        now = datetime.now()
        date = now.strftime("%m-%d-%Y,%H-%M-%S")
        plt.savefig(f'outputs//pr//val//{model_name}_confusion_matrix_{date}.png', bbox_inches='tight')
    
    if config['evaluation']['show_train_metrics']:
        preds = get_preds('train', loaded_model)
        best_f1_threshold = get_pr_curve('train', preds, save_figs=config['evaluation']['show_pr_curve'], model_name=model_name)
        get_metrics_custom('train', preds, best_f1_threshold, make_csv=config['evaluation']['make_metrics_csv'], make_specs=False, num_specs=0, model_name=model_name)
        if config['evaluation']['show_confusion_matrix']:
            _, labels = get_slices_and_labels('train')
            cm = sklearn.metrics.confusion_matrix(labels, (preds > best_f1_threshold).astype(int))
            df_cm = pd.DataFrame(cm, index=["Other", "Gunshot"], columns=["Other", "Gunshot"])
            plt.figure(figsize=(8,6))
            sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("True")
            now = datetime.now()
            date = now.strftime("%m-%d-%Y,%H-%M-%S")
            plt.savefig(f'outputs//pr//train//{model_name}_confusion_matrix_{date}.png', bbox_inches='tight')
        

    if config['evaluation']['show_test_metrics']:
        preds = get_preds('test', loaded_model)
        best_f1_threshold = get_pr_curve('test', preds, save_figs=config['evaluation']['show_pr_curve'], model_name=model_name)
        get_metrics_custom('test', preds, best_f1_threshold, make_csv=config['evaluation']['make_metrics_csv'], make_specs=False, num_specs=0, model_name=model_name)
        if config['evaluation']['show_confusion_matrix']:
            _, labels = get_slices_and_labels('test')
            cm = sklearn.metrics.confusion_matrix(labels, (preds > best_f1_threshold).astype(int))
            df_cm = pd.DataFrame(cm, index=["Other", "Gunshot"], columns=["Other", "Gunshot"])
            plt.figure(figsize=(8,6))
            sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("True")
            now = datetime.now()
            date = now.strftime("%m-%d-%Y,%H-%M-%S")
            plt.savefig(f'outputs//pr//test//{model_name}_confusion_matrix_{date}.png', bbox_inches='tight')

