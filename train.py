import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from load_data import compute_spectrogram_tf, load_config, set_seeds
from model import create_model
import yaml
import argparse
import os
from evaluate import load_group,ensure_length, get_preds, get_metrics_custom, get_pr_curve, get_evaluations, get_slices_and_labels
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import sklearn
import h5py
import random

config = load_config()

set_seeds()

def elp_dataset_pipeline(train_pos_slices, train_pos_labels, train_neg_slices, train_neg_labels, test_pos_slices, test_pos_labels, test_neg_slices, test_neg_labels):
    trimmed_train_pos_slices = [ensure_length(slice) for slice in train_pos_slices]
    trimmed_train_neg_slices = [ensure_length(slice) for slice in train_neg_slices]

    trimmed_test_pos_slices = [ensure_length(slice) for slice in test_pos_slices]
    trimmed_test_neg_slices = [ensure_length(slice) for slice in test_neg_slices]

    train_full = [[slice, label] for slice, label in zip(trimmed_train_pos_slices, train_pos_labels)] + [[slice, label] for slice, label in zip(trimmed_train_neg_slices, train_neg_labels)]
    random.seed(42)
    random.shuffle(train_full)
    train = train_full[:int(0.8*len(train_full))]
    val = train_full[int(0.8*len(train_full)):]

    train_wavs = [item[0] for item in train]
    train_labels = [item[1] for item in train]
    
    val_wavs = [item[0] for item in val]
    val_labels = [item[1] for item in val]

    test_full = [[slice, label] for slice, label in zip(trimmed_test_pos_slices, test_pos_labels)] + [[slice, label] for slice, label in zip(trimmed_test_neg_slices, test_neg_labels)]

    test_wavs = [item[0] for item in test_full]
    test_labels = [item[1] for item in test_full]

    train = tf.data.Dataset.from_tensor_slices((train_wavs, train_labels)).cache()
    train = train.shuffle(buffer_size=(len(train))//2, reshuffle_each_iteration=True)
    if config['training']['augment']:
        train = train.map(augment_waveform_tf, num_parallel_calls=tf.data.AUTOTUNE)
    train = train.map(compute_spectrogram_tf, num_parallel_calls=tf.data.AUTOTUNE)
    train = train.batch(batch_size=config['training']['batch_size'])
    train = train.prefetch(tf.data.AUTOTUNE)

    val = tf.data.Dataset.from_tensor_slices((val_wavs, val_labels)).cache()
    val = val.shuffle(buffer_size=(len(val)//2), reshuffle_each_iteration=False)
    val = val.map(compute_spectrogram_tf, num_parallel_calls=tf.data.AUTOTUNE)
    val = val.batch(batch_size=config['training']['batch_size'])
    val = val.prefetch(tf.data.AUTOTUNE)

    test = tf.data.Dataset.from_tensor_slices((test_wavs, test_labels)).cache()
    test = test.shuffle(buffer_size=(len(test)//2), reshuffle_each_iteration=False)
    test = test.map(compute_spectrogram_tf, num_parallel_calls=tf.data.AUTOTUNE)
    test = test.batch(batch_size=config['training']['batch_size'])
    test = test.prefetch(tf.data.AUTOTUNE)

    return train, val, test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model for gunshot detection on the ELP dataset"
    )
    parser.add_argument("--learning_rate", type=float, default=config['training']['learning_rate'], help="Learning rate for the optimizer")
    parser.add_argument("--loss", type=str, default=config['training']['loss'], help="Loss function to use")
    parser.add_argument("--epochs", type=int, default=config['training']['epochs'], help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=config['training']['batch_size'], help="Batch size for training")
    parser.add_argument("--model_dir", type=str, default=config['training']['model_dir'], help="Directory to save the trained model")
    parser.add_argument("--model_name", type=str, default=config['training']['model_name'], help="Name of the model")
    parser.add_argument("--early_stopping_patience", type=int, default=config['training']['early_stopping_patience'], help="Number of epochs with no improvement after which training will be stopped")
    parser.add_argument("--save_best_only", type=bool, default=config['training']['save_best_only'], help="Whether to save only the best model")
    parser.add_argument("--save_weights_only", type=bool, default=config['training']['save_weights_only'], help="Whether to save only the model weights")
    parser.add_argument("--verbose", type=int, default=config['training']['verbose'], help="Verbosity mode")
    parser.add_argument("--augment", type=bool, default=config['training']['augment'], help="Whether to augment the data or not")
    parser.add_argument("--seed", type=int, default=config['training']['seed'], help="Seed for the random number generator")
    parser.add_argument("--show_hist_curves", type=bool, default=config['evaluation']['show_hist_curves'], help="Whether to show the history curves or not")
    parser.add_argument("--make_metrics_csv", type=bool, default=config['evaluation']['make_metrics_csv'], help="Whether to make the metrics csv or not")
    parser.add_argument("--show_pr_curve", type=bool, default=config['evaluation']['show_pr_curve'], help="Whether to show the PR curve or not")
    parser.add_argument("--show_confusion_matrix", type=bool, default=config['evaluation']['show_confusion_matrix'], help="Whether to show the confusion matrix or not")
    parser.add_argument("--show_train_metrics", type=bool, default=config['evaluation']['show_train_metrics'], help="Whether to show the training metrics or not")
    parser.add_argument("--show_test_metrics", type=bool, default=config['evaluation']['show_test_metrics'], help="Whether to show the test metrics or not")
    args = parser.parse_args()

    LEARNING_RATE = args.learning_rate
    LOSS = args.loss
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    MODEL_DIR = args.model_dir
    MODEL_NAME = args.model_name
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    SAVE_BEST_ONLY = args.save_best_only
    SAVE_WEIGHTS_ONLY = args.save_weights_only
    VERBOSE = args.verbose
    AUGMENT = args.augment
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

    print("Loading data from HDF5...")

    with h5py.File(f"elp_slices_{config['data']['balance_data']}_{config['data']['ratio']}_{config['data']['snr_filter']}_{config['data']['duration_filter']}_{config['data']['snr_cutoff']}_{config['data']['duration_cutoff']}_{config['data']['clip']}_{config['data']['positive_slice_seconds']}_{config['data']['negative_slice_seconds']}_{config['data']['nfft']}_{config['data']['window_size']}_{config['data']['window_stride']}_{config['data']['mels']}_{config['data']['fmin']}_{config['data']['fmax']}_{config['data']['sample_rate']}_{config['data']['top_db']}.h5", "r") as f:
        train_pos_slices, train_pos_labels = load_group("train_pos")
        train_neg_slices, train_neg_labels = load_group("train_neg")
        test_pos_slices,  test_pos_labels  = load_group("test_pos")
        test_neg_slices,  test_neg_labels  = load_group("test_neg")

        print(f"  • train_pos: {len(train_pos_slices)} slices")
        print(f"  • train_neg: {len(train_neg_slices)} slices")
        print(f"  •  test_pos: {len(test_pos_slices)} slices")
        print(f"  •  test_neg: {len(test_neg_slices)} slices")
    '''print("Loading data from pickle...")
    data = np.load(f"elp_slices_{config['data']['balance_data']}_{config['data']['ratio']}_{config['data']['snr_filter']}_{config['data']['duration_filter']}_{config['data']['snr_cutoff']}_{config['data']['duration_cutoff']}_{config['data']['clip']}_{config['data']['positive_slice_seconds']}_{config['data']['negative_slice_seconds']}_{config['data']['nfft']}_{config['data']['window_size']}_{config['data']['window_stride']}_{config['data']['mels']}_{config['data']['fmin']}_{config['data']['fmax']}_{config['data']['sample_rate']}_{config['data']['top_db']}.npz", allow_pickle=True)
    train_pos_slices = [tf.constant(arr, dtype=tf.float32) for arr in data['train_pos_slices']]
    train_pos_labels = [arr for arr in data['train_pos_labels']]
    train_neg_slices = [tf.constant(arr, dtype=tf.float32) for arr in data['train_neg_slices']]
    train_neg_labels = [arr for arr in data['train_neg_labels']]
    test_pos_slices = [tf.constant(arr, dtype=tf.float32) for arr in data['test_pos_slices']]
    test_pos_labels = [arr for arr in data['test_pos_labels']]
    test_neg_slices = [tf.constant(arr, dtype=tf.float32) for arr in data['test_neg_slices']]
    test_neg_labels = [arr for arr in data['test_neg_labels']]'''



    print("Data loaded, creating dataset pipeline...")

    train, val, test = elp_dataset_pipeline(train_pos_slices, train_pos_labels, train_neg_slices, train_neg_labels, test_pos_slices, test_pos_labels, test_neg_slices, test_neg_labels)

    print("Dataset pipeline created, compiling model...")
    input_shape = train.element_spec[0].shape
    model = create_model(input_shape[1:])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=LOSS,
        metrics=[
            tf.keras.metrics.AUC(curve='PR', name='auc_pr'),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    model.summary()

    print("Model compiled, training...")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            save_best_only=SAVE_BEST_ONLY,
            monitor='val_auc_pr',
            mode='max',
            verbose=VERBOSE,
            save_weights_only=SAVE_WEIGHTS_ONLY
        )
    ]

    if EARLY_STOPPING_PATIENCE > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc_pr',
                patience=EARLY_STOPPING_PATIENCE,
                mode='max',
                verbose=VERBOSE
            )
        )

    hist = model.fit(train, validation_data=val, epochs=EPOCHS, callbacks=callbacks)

    print("Training completed, saving history...")

    if os.path.exists("model_hists.pkl"):
        with open("model_hists.pkl", "rb") as f:
            all_hists = pickle.load(f)
    else:
        all_hists = {}

    all_hists[MODEL_NAME] = hist.history
    with open("model_hists.pkl", "wb") as f:
        pickle.dump(all_hists, f)

    print("Saved history, evaluating model...")

    loaded_model = tf.keras.models.load_model(MODEL_PATH)

    get_evaluations(loaded_model, MODEL_NAME, hist)

    print("Evaluation complete, exiting...")
    print(f"Recap: {MODEL_NAME} trained for {EPOCHS} epochs (augmentation={AUGMENT}) with a learning rate of {LEARNING_RATE} and a batch size of {BATCH_SIZE} with {LOSS} loss.")
    print("All done!")

