import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
import yaml
import h5py
import random


def load_config(config_path='config.yaml'):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return None
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return None
    
config = load_config()

def set_seeds():
    print(f"Setting seeds (seed={config['training']['seed']}) for reproducibility...")

    if config and 'training' in config and 'seed' in config['training']:
        seed = config['training']['seed']
        # Set Python built-in random seed
        random.seed(seed)
        # Set NumPy random seed
        np.random.seed(seed)
        # Set TensorFlow random seed
        tf.random.set_seed(seed)
        # Set PYTHONHASHSEED environment variable for reproducibility
        os.environ['PYTHONHASHSEED'] = str(seed)
    else:
        print("Warning: Could not set seeds because 'training.seed' not found in config.")
    print("Seeds set.")

def get_elp_slices(path_to_sounds, path_to_meta, balance_data=False, ratio=int,
                     snr_filter=False, duration_filter=False, snr_cutoff=int, duration_cutoff=int,
                     clip=True, positive_slice_seconds=int, negative_slice_seconds=int,
                     nfft=256, window_size=256, window_stride=128, mels=32, fmin=5, fmax=2000, 
                     sample_rate=8000, top_db=80):
    """
    Create dataset with positive and negative audio slices for ELP (Environmental sound Localization and Prediction).
    
    Args:
        path_to_sounds: Path to audio files
        path_to_meta: Path to metadata CSV file
        balance_data: Whether to balance positive/negative samples
        ratio: Ratio of negative to positive samples
        snr_filter: Whether to filter by Signal-to-Noise Ratio
        duration_filter: Whether to filter by duration
        snr_cutoff: SNR threshold for filtering
        duration_cutoff: Duration threshold for filtering
        clip: Whether to clip audio to fixed length
        positive_slice_seconds: Length of positive slices in seconds
        negative_slice_seconds: Length of negative slices in seconds
        nfft, window_size, window_stride, mels, fmin, fmax, sample_rate, top_db: Spectrogram parameters
    
    Returns:
        tuple: (positive_slices, negative_slices) where each slice is a tuple of (audio_tensor, label)
               positive_slices contains tuples of (audio_tensor, 1)
               negative_slices contains tuples of (audio_tensor, 0)
    """
    
    # Load and filter metadata
    df = pd.read_csv(path_to_meta, sep='\t')
    
    if duration_filter:
        print(f"Filtering out instances with durations greater than {duration_cutoff} seconds")
        durations = df.iloc[:,4].values - df.iloc[:,3].values
        df = df[durations < duration_cutoff].reset_index(drop=True)
    
    # Extract key data
    files = df['Begin File'].values
    durations = df.iloc[:,4].values - df.iloc[:,3].values
    begins = df['File Offset (s)'].values
    
    # Audio processing parameters
    SAMPLE_RATE = sample_rate
    POS_LEN = np.int32(positive_slice_seconds * SAMPLE_RATE)
    NEG_LEN = np.int32(negative_slice_seconds * SAMPLE_RATE)
    MARGIN = np.int32(0.5 * SAMPLE_RATE)
    
    # Extract positive slices
    pos_slices = _extract_positive_slices(
        path_to_sounds, files, begins, durations, 
        POS_LEN, SAMPLE_RATE, clip, snr_filter, snr_cutoff,
        nfft, window_size, window_stride, mels, fmin, fmax, top_db
    )
    
    # Extract negative slices
    neg_slices = _extract_negative_slices(
        path_to_sounds, df, pos_slices, balance_data, ratio,
        NEG_LEN, MARGIN, SAMPLE_RATE
    )
    
    return pos_slices, neg_slices


def _extract_positive_slices(path_to_sounds, files, begins, durations, pos_len, sample_rate, 
                           clip, snr_filter, snr_cutoff, nfft, window_size, window_stride, 
                           mels, fmin, fmax, top_db):
    """Extract positive audio slices from sound files.
    
    Returns:
        list: List of tuples (audio_slice, 1) where 1 is the positive label
    """
    
    start = time.time()
    print(f"Starting to extract positive slices: snr_filter={snr_filter}")
    
    pos_slices = []
    
    # Special handling for pnnn_dep1-7 dataset
    if path_to_sounds.endswith("pnnn_dep1-7//Sound_clips"):
        pos_slices = _extract_pnnn_slices(path_to_sounds, pos_len, clip, snr_filter, snr_cutoff,
                                         nfft, window_size, window_stride, mels, fmin, fmax, top_db, sample_rate)
    else:
        # Standard processing for other datasets
        pos_slices = _extract_standard_positive_slices(
            path_to_sounds, files, begins, durations, pos_len, sample_rate, clip, 
            snr_filter, snr_cutoff, nfft, window_size, window_stride, mels, fmin, fmax, top_db
        )
    
    print(f"Time taken to extract positive slices: {time.time() - start} seconds")
    print(f"Total positive slices extracted: {len(pos_slices)}")
    
    return pos_slices


def _extract_pnnn_slices(path_to_sounds, pos_len, clip, snr_filter, snr_cutoff,
                        nfft, window_size, window_stride, mels, fmin, fmax, top_db, sample_rate):
    """Extract slices from pnnn_dep1-7 dataset.
    
    Returns:
        list: List of tuples (audio_slice, 1) where 1 is the positive label
    """
    
    audio_slices = []
    base_path = "D://naveens documents//elp_data//gunshot//Training//pnnn_dep1-7//Sound_clips"
    
    for filename in os.listdir(base_path):
        if not filename.endswith('.wav'):
            continue
            
        file_path = os.path.join(base_path, filename)
        audio = tf.io.read_file(file_path)
        audio, sr = tf.audio.decode_wav(audio)
        
        if clip:
            clipped_end = np.int32(np.ceil(pos_len))
            slice_tensor = tf.cast(tf.squeeze(audio[0:clipped_end]), dtype=tf.float32)
        else:
            slice_tensor = tf.cast(tf.squeeze(audio), dtype=tf.float32)
        
        if snr_filter:
            spec = _compute_spectrogram(slice_tensor, nfft, window_size, window_stride, 
                                      sample_rate, mels, fmin, fmax, top_db)
            if np.max(spec) - np.median(spec) >= snr_cutoff:
                audio_slices.append((slice_tensor, 1))  # Add positive label
        else:
            audio_slices.append((slice_tensor, 1))  # Add positive label
    
    return audio_slices


def _extract_standard_positive_slices(path_to_sounds, files, begins, durations, pos_len, sample_rate, 
                                     clip, snr_filter, snr_cutoff, nfft, window_size, window_stride, 
                                     mels, fmin, fmax, top_db):
    """Extract positive slices from standard dataset format.
    
    Returns:
        list: List of tuples (audio_slice, 1) where 1 is the positive label
    """
    
    pos_slices = []
    unique_files = np.unique(files)
    
    for filename in unique_files:
        file_mask = files == filename
        file_begins = begins[file_mask]
        file_durations = durations[file_mask]
        
        audio = tfio.audio.AudioIOTensor(os.path.join(path_to_sounds, filename))
        
        # Extract all slices for this file
        audio_slices = []
        begin_samples = np.floor(sample_rate * file_begins).astype(np.int64)
        end_samples = np.ceil(begin_samples + sample_rate * file_durations).astype(np.int64)
        
        for begin_sample, end_sample in zip(begin_samples, end_samples):
            if clip:
                clipped_end = np.int32(np.ceil(begin_sample + pos_len))
                slice_tensor = tf.cast(tf.squeeze(audio[begin_sample:clipped_end]), dtype=tf.float32)
            else:
                slice_tensor = tf.cast(tf.squeeze(audio[begin_sample:end_sample]), dtype=tf.float32)
            audio_slices.append(slice_tensor)
        
        # Apply SNR filtering if enabled and add positive labels
        if snr_filter and audio_slices:
            filtered_slices = _apply_snr_filter(audio_slices, snr_cutoff, nfft, window_size, 
                                              window_stride, sample_rate, mels, fmin, fmax, top_db)
            # Add positive labels to filtered slices
            pos_slices.extend([(slice_tensor, 1) for slice_tensor in filtered_slices])
        else:
            # Add positive labels to all slices
            pos_slices.extend([(slice_tensor, 1) for slice_tensor in audio_slices])
    
    return pos_slices


def _apply_snr_filter(audio_slices, snr_cutoff, nfft, window_size, window_stride, 
                     sample_rate, mels, fmin, fmax, top_db):
    """Apply SNR filtering to audio slices."""
    
    filtered_slices = []
    
    try:
        # Check if all slices have the same length for batch processing
        slice_lengths = [len(s) for s in audio_slices]
        if len(set(slice_lengths)) == 1:  # All same length
            stacked_slices = tf.stack(audio_slices)
            batch_specs = _compute_spectrogram(stacked_slices, nfft, window_size, window_stride, 
                                             sample_rate, mels, fmin, fmax, top_db)
            
            for i, spec in enumerate(tf.unstack(batch_specs)):
                if np.max(spec) - np.median(spec) >= snr_cutoff:
                    filtered_slices.append(audio_slices[i])
        else:
            # Process individually
            for audio_slice in audio_slices:
                spec = _compute_spectrogram(audio_slice, nfft, window_size, window_stride, 
                                          sample_rate, mels, fmin, fmax, top_db)
                if np.max(spec) - np.median(spec) >= snr_cutoff:
                    filtered_slices.append(audio_slice)
    
    except Exception as e:
        print(f"Batch processing failed, using individual processing: {e}")
        # Fallback to individual processing
        for audio_slice in audio_slices:
            spec = _compute_spectrogram(audio_slice, nfft, window_size, window_stride, 
                                      sample_rate, mels, fmin, fmax, top_db)
            if np.max(spec) - np.median(spec) >= snr_cutoff:
                filtered_slices.append(audio_slice)
    
    return filtered_slices


def _compute_spectrogram(audio_slice, nfft, window_size, window_stride, sample_rate, mels, fmin, fmax, top_db):
    """Compute log-mel spectrogram."""
    
    spec = tfio.audio.dbscale(
        tfio.audio.melscale(
            tfio.audio.spectrogram(
                input=audio_slice,
                nfft=nfft,
                window=window_size,
                stride=window_stride
            ),
            rate=sample_rate,
            mels=mels,
            fmin=fmin,
            fmax=fmax
        ),
        top_db=top_db
    )
    return spec

def compute_spectrogram_tf_nolabel(wav):
    """Compute log-mel spectrogram."""
    
    spec = tfio.audio.dbscale(
        tfio.audio.melscale(
            tfio.audio.spectrogram(
                input=wav,
                nfft=config['data']['nfft'],
                window=config['data']['window_size'],
                stride=config['data']['window_stride']
            ),
            rate=config['data']['sample_rate'],
            mels=config['data']['mels'],
            fmin=config['data']['fmin'],
            fmax=config['data']['fmax']
        ),
        top_db=config['data']['top_db']
    )
    return spec

def compute_spectrogram_tf(wav, label):
    """Compute log-mel spectrogram."""
    
    spec = tfio.audio.dbscale(
        tfio.audio.melscale(
            tfio.audio.spectrogram(
                input=wav,
                nfft=config['data']['nfft'],
                window=config['data']['window_size'],
                stride=config['data']['window_stride']
            ),
            rate=config['data']['sample_rate'],
            mels=config['data']['mels'],
            fmin=config['data']['fmin'],
            fmax=config['data']['fmax']
        ),
        top_db=config['data']['top_db']
    )
    return spec, label


def _extract_negative_slices(path_to_sounds, df, pos_slices, balance_data, ratio, 
                           neg_len, margin, sample_rate):
    """Extract negative audio slices.
    
    Returns:
        list: List of tuples (audio_slice, 0) where 0 is the negative label
    """
    
    # Skip negative extraction for pnnn_dep1-7 dataset
    if path_to_sounds.endswith("pnnn_dep1-7//Sound_clips"):
        print(f"No negative slices to extract (pnnn_dep1-7 dataset)")
        return []
    
    start = time.time()
    print(f"Starting to extract negative slices (balance_data={balance_data})")
    
    neg_slices = []
    
    if balance_data:
        neg_slices = _extract_balanced_negatives(df, path_to_sounds, pos_slices, 
                                               neg_len, margin, sample_rate)
    else:
        neg_slices = _extract_ratio_negatives(df, path_to_sounds, len(pos_slices), 
                                            ratio, neg_len, margin, sample_rate)
    
    print(f"Total negative slices extracted: {len(neg_slices)}")
    print(f"Time taken to extract negative slices: {time.time() - start} seconds")
    
    return neg_slices


def _extract_balanced_negatives(df, path_to_sounds, pos_slices, neg_len, margin, sample_rate):
    """Extract one negative slice for each positive slice.
    
    Returns:
        list: List of tuples (audio_slice, 0) where 0 is the negative label
    """
    
    neg_slices = []
    files = df['Begin File'].values
    begins = df['File Offset (s)'].values
    durations = df.iloc[:,4].values - df.iloc[:,3].values
    
    for i in range(len(pos_slices)):
        audio = tfio.audio.AudioIOTensor(os.path.join(path_to_sounds, files[i]))
        total_samples = tf.cast(audio.shape[0], tf.int32)
        
        begin_sample = tf.cast(np.floor(sample_rate * begins[i]), dtype=tf.int32)
        end_sample = tf.cast(np.ceil(begin_sample + sample_rate * durations[i]), dtype=tf.int32)
        
        # Define candidate regions for negative sampling
        pre_event_end = tf.cast((begin_sample - margin - neg_len), tf.int32)
        post_event_start = tf.cast((end_sample + margin), tf.int32)
        
        candidate_regions = []
        if pre_event_end > 0:
            candidate_regions.append((0, pre_event_end))
        if post_event_start + neg_len < total_samples:
            candidate_regions.append((post_event_start, total_samples - neg_len))
        
        # Randomly sample one negative segment
        if candidate_regions:
            start_min, start_max = candidate_regions[np.random.choice(len(candidate_regions))]
            neg_start = np.random.randint(start_min, start_max + 1)
            neg_end = neg_start + neg_len
            negative_slice = tf.cast(tf.squeeze(audio[neg_start:neg_end]), dtype=tf.float32)
            neg_slices.append((negative_slice, 0))  # Add negative label
    
    return neg_slices


def _extract_ratio_negatives(df, path_to_sounds, num_pos_slices, ratio, neg_len, margin, sample_rate):
    """Extract negative slices based on specified ratio.
    
    Returns:
        list: List of tuples (audio_slice, 0) where 0 is the negative label
    """
    
    neg_slices = []
    files = df['Begin File'].values
    unique_files = np.unique(files)
    
    for filename in unique_files:
        audio = tfio.audio.AudioIOTensor(os.path.join(path_to_sounds, filename))
        desired_negatives = (num_pos_slices * ratio) // len(unique_files)
        
        # Get exclusion zones (areas with positive events)
        file_events = df[df['Begin File'] == filename]
        exclusion_zones = []
        for _, row in file_events.iterrows():
            begin_sample = np.int32(np.floor(row['File Offset (s)'] * sample_rate))
            duration_samples = np.int32(np.ceil(row.iloc[4] - row.iloc[3]) * sample_rate)
            end_sample = begin_sample + duration_samples
            exclusion_zones.append((begin_sample, end_sample))
        
        # Extract negative slices avoiding exclusion zones
        total_samples = np.int32(audio.shape[0])
        file_neg_slices = []
        neg_starts = set()
        
        while len(file_neg_slices) < desired_negatives:
            idx = np.random.randint(0, total_samples - neg_len)
            
            # Check if slice overlaps with any exclusion zone
            slice_end = idx + neg_len
            valid_slice = True
            
            for zone_start, zone_end in exclusion_zones:
                if not (slice_end + margin <= zone_start or idx - margin >= zone_end):
                    valid_slice = False
                    break
            
            if valid_slice and idx not in neg_starts:
                negative_slice = tf.cast(tf.squeeze(audio[idx:idx + neg_len]), tf.float32)
                file_neg_slices.append((negative_slice, 0))  # Add negative label
                neg_starts.add(idx)
        
        neg_slices.extend(file_neg_slices)
    
    return neg_slices

def make_full_elp_dataset(training_dataset_paths, testing_dataset_paths):
    train_pos_slices = []
    train_neg_slices = []
    for data_path in training_dataset_paths:
        pos_slices, neg_slices = get_elp_slices(
            data_path[0],
            data_path[1],
            balance_data=config['data']['balance_data'],
            ratio=config['data']['ratio'],
            snr_filter=config['data']['snr_filter'],
            duration_filter=config['data']['duration_filter'],
            snr_cutoff=config['data']['snr_cutoff'],
            duration_cutoff=config['data']['duration_cutoff'],
            clip=config['data']['clip'],
            positive_slice_seconds=config['data']['positive_slice_seconds'],
            negative_slice_seconds=config['data']['negative_slice_seconds'],
            nfft=config['data']['nfft'],
            window_size=config['data']['window_size'],
            window_stride=config['data']['window_stride'],
            mels=config['data']['mels'],
            fmin=config['data']['fmin'],
            fmax=config['data']['fmax'],
            sample_rate=config['data']['sample_rate'],
            top_db=config['data']['top_db']
        )
        train_pos_slices.extend(pos_slices)
        train_neg_slices.extend(neg_slices)
    print(f"(TRAIN) Length of positives: {len(train_pos_slices)}\n Length of negatives: {len(train_neg_slices)}")
    print(f"(TRAIN) Imbalance: {len(train_pos_slices)/len(train_neg_slices)}, ≈ 1:{len(train_neg_slices)//len(train_pos_slices)}")

    test_pos_slices = []
    test_neg_slices = []
    for data_path in testing_dataset_paths:
        pos_slices, neg_slices = get_elp_slices(
            data_path[0],
            data_path[1],
            balance_data=config['data']['balance_data'],
            ratio=config['data']['ratio'],
            snr_filter=config['data']['snr_filter'],
            duration_filter=config['data']['duration_filter'],
            snr_cutoff=config['data']['snr_cutoff'],
            duration_cutoff=config['data']['duration_cutoff'],
            clip=config['data']['clip'],
            positive_slice_seconds=config['data']['positive_slice_seconds'],
            negative_slice_seconds=config['data']['negative_slice_seconds'],
            nfft=config['data']['nfft'],
            window_size=config['data']['window_size'],
            window_stride=config['data']['window_stride'],
            mels=config['data']['mels'],
            fmin=config['data']['fmin'],
            fmax=config['data']['fmax'],
            sample_rate=config['data']['sample_rate'],
            top_db=config['data']['top_db']
        )
        test_pos_slices.extend(pos_slices)
        test_neg_slices.extend(neg_slices)
    print(f"(TEST) Length of positives: {len(test_pos_slices)}\n Length of negatives: {len(test_neg_slices)}")
    print(f"(TEST) Imbalance: {len(test_pos_slices)/len(test_neg_slices)}, ≈ 1:{len(test_neg_slices)//len(test_pos_slices)}")

    return train_pos_slices, train_neg_slices, test_pos_slices, test_neg_slices



if __name__ == "__main__":
    if config:
        data_cfg = config.get('data')
        training_cfg = config.get('training')

        balance_data = data_cfg.get('balance_data', False)
        ratio = data_cfg.get('ratio', 50)
        snr_filter = data_cfg.get('snr_filter', True)
        duration_filter = data_cfg.get('duration_filter', True)
        snr_cutoff = data_cfg.get('snr_cutoff', 30)
        duration_cutoff = data_cfg.get('duration_cutoff', 7)
        clip = data_cfg.get('clip', True)
        positive_slice_seconds = data_cfg.get('positive_slice_seconds', 3)
        negative_slice_seconds = data_cfg.get('negative_slice_seconds', 3)
        nfft = data_cfg.get('nfft', 256)
        window_size = data_cfg.get('window_size', 256)
        window_stride = data_cfg.get('window_stride', 128)
        mels = data_cfg.get('mels', 50)
        fmin = data_cfg.get('fmin', 50)
        fmax = data_cfg.get('fmax', 2000)
        sample_rate = data_cfg.get('sample_rate', 8000)
        top_db = data_cfg.get('top_db', 80)

        seed = training_cfg.get('seed', 42)
        n_epochs = training_cfg.get('n_epochs', 20)
        batch_size = training_cfg.get('batch_size', 32)

        # Print config summary for verification
        print("Loaded config variables:")
        print(f"  balance_data: {balance_data}")
        print(f"  ratio: {ratio}")
        print(f"  snr_filter: {snr_filter}")
        print(f"  duration_filter: {duration_filter}")
        print(f"  snr_cutoff: {snr_cutoff}")
        print(f"  duration_cutoff: {duration_cutoff}")
        print(f"  clip: {clip}")
        print(f"  positive_slice_seconds: {positive_slice_seconds}")
        print(f"  negative_slice_seconds: {negative_slice_seconds}")
        print(f"  nfft: {nfft}")
        print(f"  window_size: {window_size}")
        print(f"  window_stride: {window_stride}")
        print(f"  mels: {mels}")
        print(f"  fmin: {fmin}")
        print(f"  fmax: {fmax}")
        print(f"  sample_rate: {sample_rate}")
        print(f"  top_db: {top_db}")
        print(f"  seed: {seed}")
        print(f"  n_epochs: {n_epochs}")
        print(f"  batch_size: {batch_size}")

        training_dataset_paths = [
            [
                "D://naveens documents//elp_data//gunshot//Training//pnnn_dep1-7//Sound_clips",
                "D://naveens documents//elp_data//gunshot//Training//pnnn_dep1-7//nn_Grid50_guns_dep1-7_Guns_Training.txt"
            ],
            [
                "D://naveens documents//elp_data//gunshot//Training//ecoguns//sounds",
                "D://naveens documents//elp_data//gunshot//Training//ecoguns//Guns_Training_ecoGuns_SST.txt"
            ]
        ]

        testing_dataset_paths = [
            [
                "D://naveens documents//elp_data//gunshot//Testing//PNNN//Sounds",
                "D://naveens documents//elp_data//gunshot//Testing//PNNN//nn201710_hbguns_10days_gunsOnly.txt"
            ]
        ]

        print("Extracting slices...")

        train_pos_slices, train_neg_slices, test_pos_slices, test_neg_slices = make_full_elp_dataset(training_dataset_paths, testing_dataset_paths)

        print("Finished extracting slices, saving to HDF5...")

        with h5py.File(f"elp_slices_{config['data']['balance_data']}_{config['data']['ratio']}_{config['data']['snr_filter']}_{config['data']['duration_filter']}_{config['data']['snr_cutoff']}_{config['data']['duration_cutoff']}_{config['data']['clip']}_{config['data']['positive_slice_seconds']}_{config['data']['negative_slice_seconds']}_{config['data']['nfft']}_{config['data']['window_size']}_{config['data']['window_stride']}_{config['data']['mels']}_{config['data']['fmin']}_{config['data']['fmax']}_{config['data']['sample_rate']}_{config['data']['top_db']}.h5", "w") as f:
            # Define a variable‑length float32 dtype for audio arrays
            dt_vlen = h5py.special_dtype(vlen=np.dtype("float32"))

            # Utility to create a pair of extendable datasets for one group
            def make_group(name):
                grp = f.create_group(name)
                grp.create_dataset(
                    "audio",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=dt_vlen,
                    chunks=True
                )
                grp.create_dataset(
                    "label",
                    shape=(0,),
                    maxshape=(None,),
                    dtype="int32",
                    chunks=True
                )
                return grp

            # Create groups
            gp_train_pos = make_group("train_pos")
            gp_train_neg = make_group("train_neg")
            gp_test_pos  = make_group("test_pos")
            gp_test_neg  = make_group("test_neg")

            # Function to append a list of (tensor, label) to a group
            def append_to_group(grp, slice_list):
                aud_ds = grp["audio"]
                lbl_ds = grp["label"]
                for i, (audio_tensor, label) in enumerate(slice_list, start=aud_ds.shape[0]):
                    arr = audio_tensor.numpy()  # 1D float32 array
                    # Resize to make room for one more element
                    aud_ds.resize((i + 1,))
                    lbl_ds.resize((i + 1,))
                    # Write data
                    aud_ds[i] = arr
                    lbl_ds[i] = label

            # Append all four lists
            append_to_group(gp_train_pos, train_pos_slices)
            append_to_group(gp_train_neg, train_neg_slices)
            append_to_group(gp_test_pos,  test_pos_slices)
            append_to_group(gp_test_neg,  test_neg_slices)

        '''print("Finished extracting slices, pickling...")

        train_pos_slices = np.array([t[0].numpy() for t in train_pos_slices], dtype=object)
        train_pos_labels = np.array([t[1] for t in train_pos_slices], dtype=np.int32)
        train_neg_slices = np.array([t[0].numpy() for t in train_neg_slices], dtype=object)
        train_neg_labels = np.array([t[1] for t in train_neg_slices], dtype=np.int32)

        test_pos_slices = np.array([t[0].numpy() for t in test_pos_slices], dtype=object)
        test_pos_labels = np.array([t[1] for t in test_pos_slices], dtype=np.int32)
        test_neg_slices = np.array([t[0].numpy() for t in test_neg_slices], dtype=object)
        test_neg_labels = np.array([t[1] for t in test_neg_slices], dtype=np.int32)

        np.savez_compressed(
            f"elp_slices_{config['data']['balance_data']}_{config['data']['ratio']}_{config['data']['snr_filter']}_{config['data']['duration_filter']}_{config['data']['snr_cutoff']}_{config['data']['duration_cutoff']}_{config['data']['clip']}_{config['data']['positive_slice_seconds']}_{config['data']['negative_slice_seconds']}_{config['data']['nfft']}_{config['data']['window_size']}_{config['data']['window_stride']}_{config['data']['mels']}_{config['data']['fmin']}_{config['data']['fmax']}_{config['data']['sample_rate']}_{config['data']['top_db']}.npz",
            train_pos_slices=train_pos_slices,
            train_pos_labels=train_pos_labels,
            train_neg_slices=train_neg_slices,
            train_neg_labels=train_neg_labels,
            test_pos_slices=test_pos_slices,
            test_pos_labels=test_pos_labels,
            test_neg_slices=test_neg_slices,
            test_neg_labels=test_neg_labels
        )

        print("Pickled slices, all done!")'''

        print("Saved slices to HDF5, all done!")