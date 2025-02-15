import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pkl
import random
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader

sns.set()
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


def cyclic_shift_tensor(tensor, window_index, window_length):
    """
    Performs a cyclic shift in the tensor based on the given window index. When the frame_ind is 0, than no cyclic is actually made

    :param tensor: A torch tensor of shape [batch, seq_len, features].
    :param window_index: The index of the window to start the sequence from.
    :param window_length: The length of each window.
    :return: A torch tensor with the cyclic shift applied.
    """
    batch_size, seq_len, num_features = tensor.shape
    total_windows = seq_len // window_length

    # Validating the window index
    if window_index < 0 or window_index >= total_windows:
        raise ValueError("Invalid window index. It should be between 0 and total_windows-1.")

    # Calculating the start index for slicing
    start_idx = window_index * window_length

    # Slicing and rearranging the tensor
    first_part = tensor[:, start_idx:, :]  # From the window index to the end
    second_part = tensor[:, :start_idx, :] # From the beginning to the window index

    # Concatenating the two parts to get the cyclic shift
    shifted_tensor = torch.cat((first_part, second_part), dim=1)

    return shifted_tensor

def simulation_data_generate(n_samples, t_len, noise=0.1):
    """"Function to generate the simulated dataset

    Args:
        n_samples: The number of samples in the dataset
        t_len: Length of each sample
        noise: The variance of Gaussian noise on the signal
    """
    zgs = [0, 1]
    zls = [0, 1]
    dataset, globals, locals = [], [], []
    t = np.array(np.arange(0, 10, 0.1))
    for _ in range(n_samples):
        zg = np.random.choice(zgs)
        zl = np.random.choice(zls)
        g_noise = np.random.randn(t_len) * noise
        if zg == 0:
            trend = t * 0.05
            c = np.ones((t_len,)) * -1.5
        elif zg == 1:
            trend = t * -0.05
            c = np.ones((t_len,)) * 1.5
        if zl == 0:
            seasonality = np.sin(40 * t / (2 * np.math.pi)) * 1.8
            ratio = 0.5
        elif zl == 1:
            seasonality = np.sin(20 * t / (2 * np.math.pi)) * 1.2
            ratio = 0.8
        dataset.append(ratio * (trend + c + seasonality) + g_noise)
        globals.append(zg)
        locals.append(zl)
    sample1_ind = globals.index(0)
    sample2_ind = globals.index(1)
    dataset = np.array(dataset)
    globals = np.array(globals)
    locals = np.array(locals)
    print('Number in class 1: ', sum(np.logical_and(globals == 1, locals == 1)))
    print('Number in class 2: ', sum(np.logical_and(globals == 1, locals == 0)))
    print('Number in class 3: ', sum(np.logical_and(globals == 0, locals == 1)))
    print('Number in class 4: ', sum(np.logical_and(globals == 0, locals == 0)))
    # plt.figure(figsize=(10, 3))
    # plt.plot(dataset[sample1_ind], label='Global: %d' % globals[sample1_ind])
    # plt.plot(dataset[sample2_ind], label='Global: %d' % globals[sample2_ind])
    # plt.legend()
    # plt.show()
    with open('./data/simulation/sim_data.pkl', 'wb') as f:
        pkl.dump(dataset, f)
    with open('./data/simulation/sim_data_globals.pkl', 'wb') as f:
        pkl.dump(globals, f)
    with open('./data/simulation/sim_data_locals.pkl', 'wb') as f:
        pkl.dump(locals, f)


def simulation_loader(normalize="none", mask_threshold=0.3, recreate=False):
    """Function to load the simulated dataset into torch dataloaders objects

    The data is loaded, normalized, padded, and a mask channel is generated to indicate missing observations

    Args:
        normalize: The type of data normalization to perform ["none", "mean_zero", "min_max"]
        mask_threshold: The percentage of measurements to mask in order to induce missingness in the simulated dataset
        recreate: If True, regenerate the dataset from scratch, otherwise, load the existing file
    """
    if recreate:
        simulation_data_generate(n_samples=500, t_len=100)
    with open('./data/simulation/sim_data.pkl', 'rb') as f:
        full_signals = pkl.load(f)
    with open('./data/simulation/sim_data_globals.pkl', 'rb') as f:
        globals = pkl.load(f)
    with open('./data/simulation/sim_data_locals.pkl', 'rb') as f:
        locals = pkl.load(f)

    # Padding the sequences
    # Note: PyTorch doesn't have a direct function like pad_sequences in TensorFlow.
    # We'll pad the sequences manually using torch functionalities.
    max_len = max([len(signal) for signal in full_signals])
    padded_signals = [torch.tensor(np.pad(signal, (0, max_len - len(signal))), dtype=torch.float32) for signal in
                      full_signals]
    padded_signals = torch.stack(padded_signals).unsqueeze(-1)

    # Generating masks
    masks = torch.tensor(np.random.binomial(n=1, p=mask_threshold, size=padded_signals.shape), dtype=torch.float32)
    padded_masks = torch.where(masks == 1, torch.ones_like(padded_signals), torch.zeros_like(padded_signals))

    # Splitting dataset
    n_train = int(0.85 * len(padded_signals))
    train_inds = list(np.arange(0, n_train))
    test_inds = list(np.arange(n_train, len(padded_signals)))

    random.shuffle(train_inds)
    valid_inds = train_inds[:int(0.2 * len(train_inds))]
    train_inds = train_inds[int(0.2 * len(train_inds)):]

    # Assuming normalize_signals() function works for both TensorFlow and PyTorch without changes
    train_signals, valid_signals, test_signals, normalization_specs = normalize_signals(padded_masks,
                                                                                        (train_inds, valid_inds,
                                                                                         test_inds),
                                                                                        normalize)

    # Creating custom PyTorch datasets
    class SimulatedDataset(Dataset):
        def __init__(self, signals, masks, signals_len, globals, locals):
            self.signals = signals
            self.masks = masks
            self.signals_len = signals_len
            self.globals = globals
            self.locals = locals

        def __len__(self):
            return len(self.signals)

        def __getitem__(self, idx):
            return self.signals[idx], self.masks[idx], self.signals_len[idx], self.globals[idx], self.locals[idx]

    # Creating PyTorch dataloaders
    trainset = DataLoader(
        SimulatedDataset(train_signals, padded_masks[train_inds], globals[train_inds], locals[train_inds]),
        batch_size=20)
    validset = DataLoader(
        SimulatedDataset(valid_signals, padded_masks[valid_inds], globals[valid_inds], locals[valid_inds]),
        batch_size=20)
    testset = DataLoader(SimulatedDataset(test_signals, padded_masks[test_inds], globals[test_inds], locals[test_inds]),
                         batch_size=20)

    return trainset, validset, testset, normalization_specs, [train_inds, valid_inds, test_inds]


def normalize_signals(signals, masks, inds, normalize):
    """Function to normalize the time series sample
    Args:
        signals: Batch of time series samples
        masks: Batch of tensor with similar size to signals, indicating the missing measurements in x
                (mask==1: missing measurement, mask==0: observed measurement)
        inds: A tuple of 3 sets of indices for the train/validation/test set.
                Note that the normalization specs are based on only the training data
        normalize: The type of data normalization
                "none": No normalization
                "min-max": Normalize all measurements to be between 0-1 using the min and max values from training set
                "mean-zero": Z-score normalization
    Returns:
        train_signals: Normalized training samples
        valid_signals: Normalized validation samples
        test_signals: Normalized test samples
        normalization_specs: The extracted normalization parameters from the training set
                            (for instance mean, variance, min, max)
    """
    train_inds, valid_inds, test_inds = inds
    if normalize == 'min_max':
        min_vals, max_vals = [], []
        for feat in range(signals.shape[-1]):
            max_vals.append(torch.max(signals[train_inds, :, feat][masks[train_inds, :, feat] == 0]))
            min_vals.append(torch.min(signals[train_inds, :, feat][masks[train_inds, :, feat] == 0]))
        max_vals = torch.tensor(max_vals)
        min_vals = torch.tensor(min_vals)
        train_signals = torch.where(masks[train_inds] == 0,
                                    (signals[train_inds] - min_vals) / (max_vals - min_vals),
                                    signals[train_inds])
        valid_signals = torch.where(masks[valid_inds] == 0,
                                    (signals[valid_inds] - min_vals) / (max_vals - min_vals),
                                    signals[valid_inds])
        test_signals = torch.where(masks[test_inds] == 0,
                                   (signals[test_inds] - min_vals) / (max_vals - min_vals),
                                   signals[test_inds])
        normalization_specs = {'max': max_vals, 'min': min_vals}
    elif normalize == 'mean_zero':
        mean_vals, std_vals = [], []
        for feat in range(signals.shape[-1]):
            mean_vals.append(torch.mean(signals[train_inds, :, feat][masks[train_inds, :, feat] == 0]))
            std_vals.append(torch.std(signals[train_inds, :, feat][masks[train_inds, :, feat] == 0]))
        mean_vals = torch.tensor(mean_vals)
        std_vals = torch.tensor(std_vals)
        std_vals = torch.where(std_vals == 0, torch.tensor(1.0), std_vals)
        train_signals = torch.where(masks[train_inds] == 0,
                                    (signals[train_inds] - mean_vals) / std_vals,
                                    signals[train_inds])
        valid_signals = torch.where(masks[valid_inds] == 0,
                                    (signals[valid_inds] - mean_vals) / std_vals,
                                    signals[valid_inds])
        test_signals = torch.where(masks[test_inds] == 0, (signals[test_inds] - mean_vals) / std_vals,
                                   signals[test_inds])
        normalization_specs = {'mean': mean_vals, 'std': std_vals}
    elif normalize == 'none':
        train_signals = signals[train_inds]
        valid_signals = signals[valid_inds]
        test_signals = signals[test_inds]
        normalization_specs = None
    else:
        raise RuntimeError('Normalization strategy not implemented')
    return train_signals, valid_signals, test_signals, normalization_specs


def har_data_loader(normalize="none"):
    """Function to load the Human Activity Recognition (HAR) dataset into TF dataset objects

        The data is loaded, normalized, padded, and a mask channel is generated to indicate missing observations

        Args:
            normalize: The type of data normalizatino to perform ["none", "mean_zero", "min_max"]
        """
    trainX = pd.read_csv('./data/HAR_data/train/X_train.txt', delim_whitespace=True, header=None)
    trainy = pd.read_csv('./data/HAR_data/train/y_train.txt', delim_whitespace=True, header=None)
    train_subj = pd.read_csv('./data/HAR_data/train/subject_train.txt', delim_whitespace=True, header=None)
    testX = pd.read_csv('./data/HAR_data/test/X_test.txt', delim_whitespace=True, header=None)
    testy = pd.read_csv('./data/HAR_data/test/y_test.txt', delim_whitespace=True, header=None)
    test_subj = pd.read_csv('./data/HAR_data/test/subject_test.txt', delim_whitespace=True, header=None)

    train_ids = np.unique(train_subj)
    x_train, locals_train, global_train, map_train = [], [], [], []
    lens_train = []
    for i, ids in enumerate(train_ids):
        inds = np.where(train_subj == ids)[0]
        ts = np.take(trainX, inds, 0).to_numpy()
        ts_labels = np.take(trainy, inds, 0).to_numpy().reshape(-1, )
        # Split each individual into 2 samples
        rnd_split = np.random.randint(int(0.4 * len(ts)), int(0.6 * len(ts)))
        lens_train.extend([rnd_split, len(ts) - rnd_split])
        x_train.append(ts[:rnd_split, :])
        x_train.append(ts[rnd_split:, :])
        map_train.append(np.zeros_like(ts[:rnd_split, :]))
        map_train.append(np.zeros_like(ts[rnd_split:, :]))
        locals_train.append(ts_labels[:rnd_split])
        locals_train.append(ts_labels[rnd_split:])
        global_train.extend([ids, ids])

    test_ids = np.unique(test_subj)
    x_test, locals_test, global_test, map_test = [], [], [], []
    lens_test = []
    for i, ids in enumerate(test_ids):
        inds = np.where(test_subj == ids)[0]
        ts = np.take(testX, inds, 0).to_numpy()
        ts_labels = np.take(testy, inds, 0).to_numpy().reshape(-1, )
        # Split each individual into 2 samples
        rnd_split = np.random.randint(int(0.4 * len(ts)), int(0.6 * len(ts)))
        lens_test.extend([rnd_split, len(ts) - rnd_split])
        x_test.append(ts[:rnd_split, :])
        x_test.append(ts[rnd_split:, :])
        map_test.append(np.zeros_like(ts[:rnd_split, :]))
        map_test.append(np.zeros_like(ts[rnd_split:, :]))
        locals_test.append(ts_labels[:rnd_split])
        locals_test.append(ts_labels[rnd_split:])
        global_test.extend([ids, ids])

    # Padding the sequences
    # Note: PyTorch doesn't have a direct function like pad_sequences in TensorFlow.
    # We'll pad the sequences manually using torch functionalities.
    max_len = 180  # From the TensorFlow code provided
    padded_signals = [torch.tensor(np.pad(signal, (0, max_len - len(signal))), dtype=torch.float32) for signal in (x_train + x_test)]
    padded_signals = torch.stack(padded_signals).unsqueeze(-1)

    padded_locals = [torch.tensor(np.pad(signal, (0, max_len - len(signal))), dtype=torch.float32) for signal in (locals_train + locals_test)]
    padded_locals = torch.stack(padded_locals).unsqueeze(-1)

    padded_maps = [torch.tensor(np.pad(signal, (0, max_len - len(signal))), dtype=torch.float32) for signal in (map_train + map_test)]
    padded_maps = torch.stack(padded_maps).unsqueeze(-1)

    train_inds = list(range(len(x_train)))
    test_inds = list(range(len(x_train), len(x_train) + len(x_test)))
    valid_inds = test_inds

    train_signals, valid_signals, test_signals, normalization_specs = normalize_signals(padded_signals, padded_maps,
                                                                                        (train_inds, valid_inds, test_inds),
                                                                                        normalize)

    # Creating custom PyTorch datasets
    class HAR_Dataset(Dataset):
        def __init__(self, signals, maps, lens, locals, globals):
            self.signals = signals
            self.maps = maps
            self.lens = lens
            self.locals = locals
            self.globals = globals

        def __len__(self):
            return len(self.signals)

        def __getitem__(self, idx):
            return self.signals[idx], self.maps[idx], self.lens[idx], self.locals[idx], self.globals[idx]

    # Creating PyTorch dataloaders
    trainset = DataLoader(HAR_Dataset(train_signals, padded_maps[train_inds], lens_train, padded_locals[train_inds], global_train), batch_size=10)
    validset = DataLoader(HAR_Dataset(valid_signals, padded_maps[valid_inds], lens_test, padded_locals[valid_inds], global_test), batch_size=5)
    testset = DataLoader(HAR_Dataset(test_signals, padded_maps[test_inds], lens_test, padded_locals[test_inds], global_test), batch_size=5)

    return trainset, validset, testset, normalization_specs


def etth_data_loader(dataset_path, window_size=24, frame_ind=0, normalize="none"):
    if dataset_path is None:
        raise ValueError("Dataset Path path must be provided by the user.")
    all_files = glob.glob(dataset_path)
    column_list = ["year", "month", "day", "hour", "HUFL", "HULL",	"MUFL",	"MULL",	"LUFL",	"LULL",	"OT"]
    feature_list = ["HUFL", "HULL",	"MUFL",	"MULL",	"LUFL",	"LULL"]
    sample_len = 24 * 28 * 1  # 2 months worth of data

    all_stations = []
    for file_names in all_files:
        station_data = pd.read_csv(file_names)[column_list]
        all_stations.append(station_data)
    all_stations = pd.concat(all_stations, axis=0, ignore_index=True)

    df_sampled = all_stations[column_list].groupby(['year', 'month'])

    signals, signal_maps = [], []
    inds, valid_inds, test_inds = [], [], []
    z_ls, z_gs = [], []
    for i, sample in enumerate(df_sampled):
        if len(sample[1]) < sample_len:
            continue
        # Determine training indices for different years
        if sample[0][0] in [2017, 2018]:
            inds.extend([i])
        elif sample[0][0] in [2016]:  # data from 2016 is used for testing, because we have fewer recordings for the final year
            test_inds.extend([i])
        x = sample[1][feature_list][:sample_len].astype('float32')
        sample_map = x.isna().astype('float32')
        z_l = sample[1][['day', 'OT']][:sample_len]
        x = x.fillna(0)
        z_g = np.array(sample[0])
        signals.append(torch.tensor(x.values))
        signal_maps.append(torch.tensor(sample_map.values))
        z_ls.append(torch.tensor(z_l.values))
        z_gs.append(torch.tensor(z_g))
    signals_len = torch.zeros(len(signals)) + sample_len
    signals = torch.stack(signals)
    signal_maps = torch.stack(signal_maps)
    z_ls = torch.stack(z_ls)
    z_gs = torch.stack(z_gs)

    random.shuffle(inds)
    train_inds = inds[:int(len(inds) * 0.85)]
    valid_inds = inds[int(len(inds) * 0.85):]

    train_signals, valid_signals, test_signals, normalization_specs = normalize_signals(signals, signal_maps,
                                                                                        (train_inds, valid_inds,
                                                                                         test_inds),
                                                                                        normalize)

    train_signals = cyclic_shift_tensor(train_signals, frame_ind, window_size)
    valid_signals = cyclic_shift_tensor(valid_signals, frame_ind, window_size)
    test_signals = cyclic_shift_tensor(test_signals, frame_ind, window_size)

    # plot a random sample
    ind = np.random.randint(0, len(train_inds))
    f, axs = plt.subplots(nrows=train_signals.shape[-1], ncols=1, figsize=(18, 14))
    for i, ax in enumerate(axs):
        ax.plot(train_signals[ind, :, i])
        ax.set_title(feature_list[i])
    plt.tight_layout()
    plt.show()

    # Creating custom PyTorch datasets
    class ETTH_DATASET(Dataset):
        def __init__(self, signals, maps, lens, locals, globals):
            self.signals = signals
            self.maps = maps
            self.lens = lens
            self.locals = locals
            self.globals = globals

        def __len__(self):
            return len(self.signals)

        def __getitem__(self, idx):
            return self.signals[idx], self.maps[idx], self.lens[idx], self.locals[idx], self.globals[idx]

    # Creating PyTorch dataloaders
    trainset = DataLoader(ETTH_DATASET(train_signals, signal_maps[train_inds], signals_len[train_inds],
                                       z_ls[train_inds], z_gs[train_inds]), batch_size=10, shuffle=True)
    validset = DataLoader(ETTH_DATASET(valid_signals, signal_maps[valid_inds], signals_len[valid_inds],
                                       z_ls[valid_inds], z_gs[valid_inds]), batch_size=10)
    testset = DataLoader(ETTH_DATASET(test_signals, signal_maps[test_inds], signals_len[test_inds],
                                      z_ls[test_inds], z_gs[test_inds]), batch_size=10)

    return trainset, validset, testset, normalization_specs

def airq_data_loader(dataset_path, window_size=24, frame_ind=0, normalize="none"):
    """Function to load the Air Quality dataset into TF dataset objects

        The data is loaded, normalized, padded, and a mask channel is generated to indicate missing observations
        The raw csv files can be downloaded from:
        https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data

        Args:
            normalize: The type of data normalizatino to perform ["none", "mean_zero", "min_max"]
        """
    if dataset_path is None:
        raise ValueError("Paths path must be provided by the user.")
    all_files = glob.glob(dataset_path)
    column_list = ["year", "month", "day", "hour", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP",
                   "RAIN", "WSPM", "station"]
    feature_list = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "WSPM"]
    sample_len = 24 * 28 * 1  # 2 months worth of data

    all_stations = []
    for file_names in all_files:
        station_data = pd.read_csv(file_names)[column_list]
        all_stations.append(station_data)
    all_stations = pd.concat(all_stations, axis=0, ignore_index=True)
    # Convert station names to unique integers
    station_mapping = {name: i for i, name in enumerate(all_stations['station'].unique())}
    all_stations['station'] = all_stations['station'].map(station_mapping)

    df_sampled = all_stations[column_list].groupby(['year', 'month', 'station'])

    signals, signal_maps = [], []
    inds, valid_inds, test_inds = [], [], []
    z_ls, z_gs = [], []
    for i, sample in enumerate(df_sampled):
        if len(sample[1]) < sample_len:
            continue
        # Determine training indices for different years
        if sample[0][0] in [2013, 2014, 2015, 2017]:
            inds.extend([i])
        elif sample[0][0] in [2016]:  # data from 2016 is used for testing, because we have fewer recordings for the final year
            test_inds.extend([i])
        x = sample[1][feature_list][:sample_len].astype('float32')
        sample_map = x.isna().astype('float32')
        z_l = sample[1][['day', 'RAIN']][:sample_len]
        x = x.fillna(0)
        z_g = np.array(sample[0])
        signals.append(torch.tensor(x.values))
        signal_maps.append(torch.tensor(sample_map.values))
        z_ls.append(torch.tensor(z_l.values))
        z_gs.append(torch.tensor(z_g))
    signals_len = torch.zeros(len(signals)) + sample_len
    signals = torch.stack(signals)
    signal_maps = torch.stack(signal_maps)
    z_ls = torch.stack(z_ls)
    z_gs = torch.stack(z_gs)

    random.shuffle(inds)
    train_inds = inds[:int(len(inds) * 0.85)]
    valid_inds = inds[int(len(inds) * 0.85):]

    train_signals, valid_signals, test_signals, normalization_specs = normalize_signals(signals, signal_maps,
                                                                                        (train_inds, valid_inds,
                                                                                         test_inds),
                                                                                        normalize)

    train_signals = cyclic_shift_tensor(train_signals, frame_ind, window_size)
    valid_signals = cyclic_shift_tensor(valid_signals, frame_ind, window_size)
    test_signals = cyclic_shift_tensor(test_signals, frame_ind, window_size)

    # plot a random sample
    ind = np.random.randint(0, len(train_inds))
    f, axs = plt.subplots(nrows=train_signals.shape[-1], ncols=1, figsize=(18, 14))
    for i, ax in enumerate(axs):
        ax.plot(train_signals[ind, :, i])
        ax.set_title(feature_list[i])
    plt.tight_layout()
    plt.show()

    # Creating custom PyTorch datasets
    class AirQ_Dataset(Dataset):
        def __init__(self, signals, maps, lens, locals, globals):
            self.signals = signals
            self.maps = maps
            self.lens = lens
            self.locals = locals
            self.globals = globals

        def __len__(self):
            return len(self.signals)

        def __getitem__(self, idx):
            return self.signals[idx], self.maps[idx], self.lens[idx], self.locals[idx], self.globals[idx]

    # Creating PyTorch dataloaders
    trainset = DataLoader(AirQ_Dataset(train_signals, signal_maps[train_inds], signals_len[train_inds],
                                       z_ls[train_inds], z_gs[train_inds]), batch_size=10, shuffle=True)
    validset = DataLoader(AirQ_Dataset(valid_signals, signal_maps[valid_inds], signals_len[valid_inds],
                                       z_ls[valid_inds], z_gs[valid_inds]), batch_size=10)
    testset = DataLoader(AirQ_Dataset(test_signals, signal_maps[test_inds], signals_len[test_inds],
                                      z_ls[test_inds], z_gs[test_inds]), batch_size=10)

    return trainset, validset, testset, normalization_specs


def physionet_data_loader(data_dir, window_size=4, frame_ind=0, normalize='none', dataset='set-a'):
    feature_map = {'Albumin': 'Serum Albumin (g/dL)',
                   'ALP': 'Alkaline phosphatase (IU/L)',
                   'ALT': 'Alanine transaminase (IU/L)',
                   'AST': 'Aspartate transaminase (IU/L)',
                   'Bilirubin': 'Bilirubin (mg/dL)',
                   'BUN': 'Blood urea nitrogen (mg/dL)',
                   'Cholesterol': 'Cholesterol (mg/dL)',
                   'Creatinine': 'Serum creatinine (mg/dL)',
                   'DiasABP': 'Invasive diastolic arterial blood pressure (mmHg)',
                   'FiO2': 'Fractional inspired O2 (0-1)',
                   'GCS': 'Glasgow Coma Score (3-15)',
                   'Glucose': 'Serum glucose (mg/dL)',
                   'HCO3': 'Serum bicarbonate (mmol/L)',
                   'HCT': 'Hematocrit (%)',
                   'HR': 'Heart rate (bpm)',
                   'K': 'Serum potassium (mEq/L)',
                   'Lactate': 'Lactate (mmol/L)',
                   'Mg': 'Serum magnesium (mmol/L)',
                   'MAP': 'Invasive mean arterial blood pressure (mmHg)',
                   'Na': 'Serum sodium (mEq/L)',
                   'NIDiasABP': 'Non-invasive diastolic arterial blood pressure (mmHg)',
                   'NIMAP': 'Non-invasive mean arterial blood pressure (mmHg)',
                   'NISysABP': 'Non-invasive systolic arterial blood pressure (mmHg)',
                   'PaCO2': 'partial pressure of arterial CO2 (mmHg)',
                   'PaO2': 'Partial pressure of arterial O2 (mmHg)',
                   'pH': 'Arterial pH (0-14)',
                   'Platelets': 'Platelets (cells/nL)',
                   'RespRate': 'Respiration rate (bpm)',
                   'SaO2': 'O2 saturation in hemoglobin (%)',
                   'SysABP': 'Invasive systolic arterial blood pressure (mmHg)',
                   'Temp': 'Temperature (°C)',
                   'TroponinI': 'Troponin-I (μg/L)',
                   'TroponinT': 'Troponin-T (μg/L)',
                   'Urine': 'Urine output (mL)',
                   'WBC': 'White blood cell count (cells/nL)'
                   }
    feature_list = list(feature_map.keys())
    local_list = ['MechVent', 'Weight']
    static_vars = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']
    if data_dir is None:
        raise ValueError("Dataset Path path must be provided by the user.")
    physionet_dataset_path = os.path.join(data_dir, 'processed_df.csv')
    physionet_static_dataset_path = os.path.join(data_dir, 'processed_static_df.csv')
    if os.path.exists(data_dir):
        df_full = pd.read_csv(physionet_dataset_path)
        df_static = pd.read_csv(physionet_static_dataset_path)
    else:
        txt_all = list()
        for f in os.listdir(os.path.join(data_dir, dataset)):
            with open(os.path.join(data_dir, dataset, f), 'r') as fp:
                txt = fp.readlines()
            # get recordid to add as a column
            recordid = txt[1].rstrip('\n').split(',')[-1]
            try:
                txt = [t.rstrip('\n').split(',') + [int(recordid)] for t in txt]
                txt_all.extend(txt[1:])
            except:
                continue

        # convert to pandas dataframe
        df = pd.DataFrame(txt_all, columns=['time', 'parameter', 'value', 'recordid'])

        # extract static variables into a separate dataframe
        df_static = df.loc[df['time'] == '00:00', :].copy()

        df_static = df_static.loc[df['parameter'].isin(static_vars)]

        # remove these from original df
        idxDrop = df_static.index
        df = df.loc[~df.index.isin(idxDrop), :]

        # pivot on parameter so there is one column per parameter
        df_static = df_static.pivot(index='recordid', columns='parameter', values='value')

        # some conversions on columns for convenience
        df['value'] = pd.to_numeric(df['value'], errors='raise')
        df['time'] = df['time'].map(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

        df.head()
        # convert static into numeric
        for c in df_static.columns:
            df_static[c] = pd.to_numeric(df_static[c])

        # preprocess
        for c in df_static.columns:
            x = df_static[c]
            if c == 'Age':
                # replace anon ages with 91.4
                idx = x > 130
                df_static.loc[idx, c] = 91.4
            elif c == 'Gender':
                idx = x < 0
                df_static.loc[idx, c] = np.nan
            elif c == 'Height':
                idx = x < 0
                df_static.loc[idx, c] = np.nan

                # fix incorrectly recorded heights

                # 1.8 -> 180
                idx = x < 10
                df_static.loc[idx, c] = df_static.loc[idx, c] * 100

                # 18 -> 180
                idx = x < 25
                df_static.loc[idx, c] = df_static.loc[idx, c] * 10

                # 81.8 -> 180 (inch -> cm)
                idx = x < 100
                df_static.loc[idx, c] = df_static.loc[idx, c] * 2.2

                # 1800 -> 180
                idx = x > 1000
                df_static.loc[idx, c] = df_static.loc[idx, c] * 0.1

                # 400 -> 157
                idx = x > 250
                df_static.loc[idx, c] = df_static.loc[idx, c] * 0.3937

            elif c == 'Weight':
                idx = x < 35
                df_static.loc[idx, c] = np.nan

                idx = x > 299
                df_static.loc[idx, c] = np.nan

        df = delete_value(df, 'DiasABP', -1)
        df = replace_value(df, 'DiasABP', value=np.nan, below=1)
        df = replace_value(df, 'DiasABP', value=np.nan, above=200)
        df = replace_value(df, 'SysABP', value=np.nan, below=1)
        df = replace_value(df, 'MAP', value=np.nan, below=1)

        df = replace_value(df, 'NIDiasABP', value=np.nan, below=1)
        df = replace_value(df, 'NISysABP', value=np.nan, below=1)
        df = replace_value(df, 'NIMAP', value=np.nan, below=1)

        df = replace_value(df, 'HR', value=np.nan, below=1)
        df = replace_value(df, 'HR', value=np.nan, above=299)

        df = replace_value(df, 'PaCO2', value=np.nan, below=1)
        df = replace_value(df, 'PaCO2', value=lambda x: x * 10, below=10)

        df = replace_value(df, 'PaO2', value=np.nan, below=1)
        df = replace_value(df, 'PaO2', value=lambda x: x * 10, below=20)

        # the order of these steps matters
        df = replace_value(df, 'pH', value=lambda x: x * 10, below=0.8, above=0.65)
        df = replace_value(df, 'pH', value=lambda x: x * 0.1, below=80, above=65)
        df = replace_value(df, 'pH', value=lambda x: x * 0.01, below=800, above=650)
        df = replace_value(df, 'pH', value=np.nan, below=6.5)
        df = replace_value(df, 'pH', value=np.nan, above=8.0)

        # convert to farenheit
        df = replace_value(df, 'Temp', value=lambda x: x * 9 / 5 + 32, below=10, above=1)
        df = replace_value(df, 'Temp', value=lambda x: (x - 32) * 5 / 9, below=113, above=95)

        df = replace_value(df, 'Temp', value=np.nan, below=25)
        df = replace_value(df, 'Temp', value=np.nan, above=45)

        df = replace_value(df, 'RespRate', value=np.nan, below=1)
        df = replace_value(df, 'WBC', value=np.nan, below=1)

        df = replace_value(df, 'Weight', value=np.nan, below=35)
        df = replace_value(df, 'Weight', value=np.nan, above=299)

        df_full = pd.DataFrame(columns=['time', 'recordid'] + feature_list + local_list)
        df_sampled = df.groupby(['recordid'])  # , 'parameter'])
        for i, sample in enumerate(df_sampled):
            id = sample[0]
            df_signal = sample[1].groupby(['parameter'])
            signal_df = pd.DataFrame(columns=['time', 'recordid'])
            for j, signal_sample in enumerate(df_signal):
                param = signal_sample[0]
                sub_df = pd.DataFrame(columns=['time', 'recordid'] + [param])
                sub_df[param] = signal_sample[1]['value']
                sub_df['recordid'] = id
                sub_df['time'] = signal_sample[1]['time']
                signal_df = signal_df.merge(sub_df, how='outer', on=['recordid', 'time'], sort=True,
                                            suffixes=[None, None])
            # Bin the values
            bins = pd.cut(signal_df.time, np.arange(signal_df['time'].iloc[0], signal_df['time'].iloc[-1], 60))
            col_list = list(signal_df.columns[2:])  # - ['recordid', 'time']
            signal_df_binned = pd.DataFrame(columns=signal_df.columns)
            signal_df_binned['time'] = np.arange(signal_df['time'].iloc[0], signal_df['time'].iloc[-1], 60)[:-1]
            signal_df_binned['recordid'] = id
            signal_df_binned[col_list] = signal_df.groupby(bins).agg(
                dict(zip(col_list, ["mean"] * len(col_list)))).to_numpy()  # {"Temperature": "mean"})
            df_full = pd.concat([signal_df_binned, df_full])
        df_full.to_csv(physionet_dataset_path)
        df_static.to_csv(physionet_static_dataset_path)

    selected_features = ['DiasABP', 'GCS', 'HCT', 'MAP', 'NIDiasABP', 'NIMAP', 'NISysABP', 'RespRate', 'SysABP', 'Temp']

    # load in outcomes
    if dataset == 'set-a':
        y = pd.read_csv(os.path.join(data_dir, 'Outcomes-a.txt'))
    elif dataset == 'set-b':
        y = pd.read_csv(os.path.join(data_dir, 'Outcomes-.txt'))
    label_list = ['SAPS-I', 'SOFA', 'In-hospital_death']

    df_sampled = df_full.groupby(['recordid'])
    max_len = 80
    signals, signal_maps, signal_lens = [], [], []
    z_ls, z_gs = [], []
    for i, sample in enumerate(df_sampled):
        id = sample[0]
        x = sample[1][selected_features]
        if np.array(x.isna()).mean() > 0.6 or len(x) < 0.5 * max_len:
            continue
        sample_map = x.isna().astype('float32')
        labels = y[y['RecordID'] == id][label_list]
        z_l = sample[1][['MechVent']]
        x = x.fillna(0.0)
        z_g = df_static[df_static['RecordID'] == id][['Age', 'Gender', 'Height', 'ICUType', 'Weight']]
        signals.append(np.array(x))
        signal_maps.append(np.array(sample_map))
        z_ls.append(np.array(z_l))
        z_gs.append(np.concatenate([np.array(z_g), np.array(labels)], axis=-1).reshape(-1, ))
        signal_lens.append(min(max_len, len(x)))

    def pad_sequences(sequences, max_len, value=0.0):
        return torch.stack([torch.cat([s, torch.full((max_len - len(s), *s.shape[1:]), value)]) for s in sequences])

    signals = pad_sequences([torch.tensor(s, dtype=torch.float32) for s in signals], max_len)
    locals = pad_sequences([torch.tensor(l, dtype=torch.float32) for l in z_ls], max_len)
    maps = pad_sequences([torch.tensor(m, dtype=torch.float32) for m in signal_maps], max_len, value=1.0)
    z_gs = torch.tensor(z_gs, dtype=torch.float32)
    signal_lens = torch.tensor(signal_lens, dtype=torch.long)

    test_inds = list(range(int(0.2 * len(signals))))
    inds = list(range(int(0.2 * len(signals)), len(signals)))
    random.shuffle(inds)
    train_inds = inds[:int(0.8 * len(inds))]
    valid_inds = inds[int(0.8 * len(inds)):]

    # plot a random sample
    ind = np.random.randint(0, len(train_inds))
    f, axs = plt.subplots(nrows=signals.shape[-1], ncols=1, figsize=(18, 14))
    for i, ax in enumerate(axs):
        ax.plot(signals[ind, :, i])
        ax.set_title(feature_list[i])
    plt.tight_layout()
    plt.show()

    train_signals, valid_signals, test_signals, normalization_specs = normalize_signals(signals, maps,
                                                                                        (train_inds, valid_inds,
                                                                                         test_inds),
                                                                                        normalize)

    train_signals = cyclic_shift_tensor(train_signals, frame_ind, window_size)
    valid_signals = cyclic_shift_tensor(valid_signals, frame_ind, window_size)
    test_signals = cyclic_shift_tensor(test_signals, frame_ind, window_size)

    class PhysionetDataset(Dataset):
        def __init__(self, signals, maps, lens, locals, z_gs):
            self.signals = signals
            self.maps = maps
            self.lens = lens
            self.locals = locals
            self.z_gs = z_gs

        def __len__(self):
            return len(self.signals)

        def __getitem__(self, idx):
            return self.signals[idx], self.maps[idx], self.lens[idx], self.locals[idx], self.z_gs[idx]

    train_dataset = PhysionetDataset(train_signals, maps[train_inds], signal_lens[train_inds], locals[train_inds], z_gs[train_inds])
    valid_dataset = PhysionetDataset(valid_signals, maps[valid_inds], signal_lens[valid_inds], locals[valid_inds], z_gs[valid_inds])
    test_dataset = PhysionetDataset(test_signals, maps[test_inds], signal_lens[test_inds], locals[test_inds], z_gs[test_inds])

    train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)


    return train_loader, valid_loader, test_loader, normalization_specs

def delete_value(df, c, value=0):
    """Helper function for processing the Physionet dataset"""
    idx = df['parameter'] == c
    idx = idx & (df['value'] == value)
    df.loc[idx, 'value'] = np.nan
    return df


def replace_value(df, c, value=np.nan, below=None, above=None):
    """Helper function for processing the Physionet dataset"""
    idx = df['parameter'] == c
    if below is not None:
        idx = idx & (df['value'] < below)
    if above is not None:
        idx = idx & (df['value'] > above)
    if 'function' in str(type(value)):
        # value replacement is a function of the input
        df.loc[idx, 'value'] = df.loc[idx, 'value'].apply(value)
    else:
        df.loc[idx, 'value'] = value
    return df
