import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional


SAMPLING_INTERVAL_MIN = 5
INPUT_WINDOW = 24
PREDICTION_HORIZON = 6
MAX_INTERP_GAP_MIN = 20
MAX_MASK_GAP_MIN = 120
MAX_ROC = 4.0
N_FEATURES = 6

@dataclass
class PatientStats:
    glucose_mean : float
    glucose_std  : float
    patient_id   : str


@dataclass
class TokenSequence:
    features       : np.ndarray
    target         : float
    target_raw     : float
    attention_mask : np.ndarray
    start_time     : pd.Timestamp


def build_cgm_grid(glucose_df: pd.DataFrame) -> pd.DataFrame:
    # Build the complete regular grid
    start = glucose_df['ts'].min().floor('5min')
    end   = glucose_df['ts'].max().ceil('5min')
    grid  = pd.date_range(start=start, end=end, freq='5min')
    grid_df = pd.DataFrame({'ts': grid})

    glucose_sorted = glucose_df.sort_values('ts').copy()
    glucose_sorted = glucose_sorted.rename(columns={'value': 'glucose_raw'})

    grid_df = pd.merge_asof(
        grid_df,
        glucose_sorted,
        on='ts',
        tolerance=pd.Timedelta('2min30s'),
        direction='nearest'
    )

    is_missing = grid_df['glucose_raw'].isna()
    gap_group  = (is_missing != is_missing.shift()).cumsum()
    gap_lengths = pd.Series(0.0, index=grid_df.index)
    for group_id, group_df in grid_df[is_missing].groupby(gap_group[is_missing]):
        gap_len_min = len(group_df) * SAMPLING_INTERVAL_MIN
        gap_lengths.loc[group_df.index] = gap_len_min

    grid_df['gap_minutes'] = gap_lengths

    return grid_df


def handle_gaps(grid_df: pd.DataFrame) -> pd.DataFrame:
    df = grid_df.copy()

    short_gap  = (df['gap_minutes'] > 0) & (df['gap_minutes'] <= MAX_INTERP_GAP_MIN)
    medium_gap = (df['gap_minutes'] > MAX_INTERP_GAP_MIN) & (df['gap_minutes'] <= MAX_MASK_GAP_MIN)
    long_gap   = df['gap_minutes'] > MAX_MASK_GAP_MIN

    df['glucose_filled']  = df['glucose_raw'].copy()
    df['is_interpolated'] = 0
    df['attention_valid'] = 1
    df['is_long_gap']     = 0

    # linear interp for short gaps
    df['glucose_filled'] = df['glucose_filled'].interpolate(
        method='linear', limit_area='inside'
    )
    df.loc[short_gap, 'is_interpolated'] = 1

    # forward fill for medium gaps
    df['glucose_filled'] = df['glucose_filled'].ffill()
    df.loc[medium_gap, 'is_interpolated'] = 1
    df.loc[medium_gap, 'attention_valid'] = 0
    df.loc[long_gap, 'is_interpolated'] = 1
    df.loc[long_gap, 'attention_valid'] = 0
    df.loc[long_gap, 'is_long_gap']     = 1

    df['glucose_filled'] = df['glucose_filled'].bfill()

    return df


def compute_features(
    grid_df    : pd.DataFrame,
    meal_df    : pd.DataFrame,
    stats      : PatientStats,
) -> pd.DataFrame:
    df = grid_df.copy()

    # z-score normalization
    df['f0_glucose'] = (
        (df['glucose_filled'] - stats.glucose_mean) / stats.glucose_std
    )

    roc_raw = df['glucose_filled'].diff() / SAMPLING_INTERVAL_MIN
    df['f1_roc'] = roc_raw.clip(-MAX_ROC, MAX_ROC).fillna(0.0)

    hours = df['ts'].dt.hour + df['ts'].dt.minute / 60.0
    df['f2_time_sin'] = np.sin(2 * np.pi * hours / 24.0)
    df['f3_time_cos'] = np.cos(2 * np.pi * hours / 24.0)

    df['f4_carbs'] = 0.0

    if len(meal_df) > 0:
        meal_aligned = pd.merge_asof(
            meal_df.sort_values('ts')[['ts', 'carbs']],
            df[['ts']].reset_index(),
            on='ts',
            tolerance=pd.Timedelta('2min30s'),
            direction='nearest'
        )
        valid_matches = meal_aligned.dropna(subset=['index'])
        for _, row in valid_matches.iterrows():
            grid_idx = int(row['index'])
            df.loc[grid_idx, 'f4_carbs'] = row['carbs'] / 100.0

    df['f5_interp'] = df['is_interpolated'].astype(float)

    return df

def compute_patient_stats(glucose_df: pd.DataFrame, patient_id: str) -> PatientStats:
    values = glucose_df['value'].dropna().values
    return PatientStats(
        glucose_mean = float(np.mean(values)),
        glucose_std  = float(np.std(values)),
        patient_id   = patient_id,
    )

def extract_sequences(feature_df: pd.DataFrame) -> List[TokenSequence]:
    feature_cols = ['f0_glucose', 'f1_roc', 'f2_time_sin',
                    'f3_time_cos', 'f4_carbs', 'f5_interp']

    features_arr   = feature_df[feature_cols].values.astype(np.float32)
    attn_valid_arr = feature_df['attention_valid'].values
    long_gap_arr   = feature_df['is_long_gap'].values
    glucose_raw    = feature_df['glucose_filled'].values
    timestamps     = feature_df['ts'].values

    sequences = []

    total = len(feature_df)
    first_valid = INPUT_WINDOW
    last_valid  = total - PREDICTION_HORIZON

    for t in range(first_valid, last_valid):
        window_start = t - INPUT_WINDOW
        window_end   = t
        target_idx   = t + PREDICTION_HORIZON - 1

        if long_gap_arr[window_start:window_end].any():
            continue

        if not attn_valid_arr[window_end - 1]:
            continue

        if long_gap_arr[target_idx] or np.isnan(glucose_raw[target_idx]):
            continue

        if long_gap_arr[window_end:target_idx + 1].any():
            continue

        window_features = features_arr[window_start:window_end].copy()
        window_attn     = attn_valid_arr[window_start:window_end].astype(bool)
        target_normalized = features_arr[target_idx, 0]
        target_raw = float(glucose_raw[target_idx])

        sequences.append(TokenSequence(
            features       = window_features,
            target         = float(target_normalized),
            target_raw     = target_raw,
            attention_mask = window_attn,
            start_time     = pd.Timestamp(timestamps[window_start]),
        ))

    return sequences


def tokenize_patient(
    data       : Dict[str, pd.DataFrame],
    patient_id : str,
    stats      : Optional[PatientStats] = None,
) -> Tuple[List[TokenSequence], PatientStats]:

    required = ['glucose_level']
    for stream in required:
        if stream not in data:
            raise KeyError(
                f"Patient {patient_id}: required stream '{stream}' missing "
                f"from XML. Available streams: {list(data.keys())}"
            )

    glucose_df = data['glucose_level']

    if 'meal' in data and len(data['meal']) > 0:
        meal_df = data['meal']
    else:
        if 'meal' not in data:
            print(f"  Patient {patient_id}: 'meal' stream absent "
                  f"— no carb features for this file")
        meal_df = pd.DataFrame(columns=['ts', 'carbs'])

    glucose_df = data['glucose_level']

    if 'meal' in data and len(data['meal']) > 0:
        meal_df = data['meal']
    else:
        meal_df = pd.DataFrame(columns=['ts', 'carbs'])

    grid_df = build_cgm_grid(glucose_df)
    grid_df = handle_gaps(grid_df)

    if stats is None:
        stats = compute_patient_stats(glucose_df, patient_id)

    feature_df = compute_features(grid_df, meal_df, stats)
    sequences = extract_sequences(feature_df)

    print(f"Patient {patient_id}: {len(sequences)} valid sequences "
          f"from {len(grid_df)} grid timesteps "
          f"(glucose mean={stats.glucose_mean:.1f}, "
          f"std={stats.glucose_std:.1f})")

    return sequences, stats
def verify_tokenization(sequences: List[TokenSequence]):
    assert len(sequences) > 0, "No valid sequences extracted"

    shapes_ok   = all(s.features.shape == (INPUT_WINDOW, N_FEATURES)
                      for s in sequences)
    masks_ok    = all(s.attention_mask.shape == (INPUT_WINDOW,)
                      for s in sequences)
    no_nan      = all(not np.isnan(s.features).any() for s in sequences)
    no_inf      = all(not np.isinf(s.features).any() for s in sequences)
    targets_ok  = all(not np.isnan(s.target) for s in sequences)

    print(f"\nVerification results:")
    print(f"  Total sequences : {len(sequences)}")
    print(f"  Shape correct   : {shapes_ok}")
    print(f"  Masks correct   : {masks_ok}")
    print(f"  No NaN in feats : {no_nan}")
    print(f"  No Inf in feats : {no_inf}")
    print(f"  Targets valid   : {targets_ok}")

    features_stacked = np.stack([s.features for s in sequences])
    print(f"\n  Feature ranges (min, max, mean):")
    feature_names = ['glucose', 'roc', 'time_sin', 'time_cos', 'carbs', 'interp']
    for i, name in enumerate(feature_names):
        col = features_stacked[:, :, i].flatten()
        print(f"    f{i} {name:10s}: "
              f"[{col.min():7.3f}, {col.max():7.3f}], "
              f"mean={col.mean():7.3f}")

    carb_nonzero = (features_stacked[:, :, 4] > 0).sum()
    print(f"\n  Non-zero carb tokens: {carb_nonzero} "
          f"({carb_nonzero / features_stacked[:, :, 4].size * 100:.2f}% of all tokens)")

    masks_stacked = np.stack([s.attention_mask for s in sequences])
    masked_fraction = (~masks_stacked).mean()
    print(f"  Masked token fraction: {masked_fraction:.4f}")

    all_ok = shapes_ok and masks_ok and no_nan and no_inf and targets_ok
    print(f"\n  {'ALL CHECKS PASSED' if all_ok else 'CHECKS FAILED — review above'}")
    return all_ok