import torch
import numpy as np


def extract_snr_from_dataset(dataset, batch_indices=None):
    if batch_indices is None:
        batch_indices = range(len(dataset))
    
    p_snr_values = []
    s_snr_values = []
    
    for idx in batch_indices:
        try:
            event, _ = dataset._load_event_data(idx)
            
            if 'snr' in event and len(event['snr']) >= 3:
                # DiTing SNR：[Z_P_power_snr, N_S_power_snr, E_S_power_snr]
                z_p_snr = event['snr'][0]
                n_s_snr = event['snr'][1]
                e_s_snr = event['snr'][2]
                
                p_snr = z_p_snr
                s_snr = (n_s_snr + e_s_snr) / 2
                
                p_snr_values.append(p_snr)
                s_snr_values.append(s_snr)
            else:
                p_snr_values.append(None)
                s_snr_values.append(None)
        except Exception as e:
            p_snr_values.append(None)
            s_snr_values.append(None)
    
    return p_snr_values, s_snr_values


def update_model_with_dataset_info(model, dataset):    
    if hasattr(dataset, '_sampling_rate'):
        model.sampling_rate = dataset._sampling_rate
    
    sample_size = min(100, len(dataset))
    sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
    p_snr_values, s_snr_values = extract_snr_from_dataset(dataset, sample_indices)
    
    p_snr_filtered = [x for x in p_snr_values if x is not None]
    s_snr_filtered = [x for x in s_snr_values if x is not None]
    
    if p_snr_filtered:
        model.p_snr_cache = np.mean(p_snr_filtered)
    else:
        model.p_snr_cache = np.nan  
        logger.warning("No valid P-SNR values found in the sample. Setting p_snr_cache to the default value NaN.")

    if s_snr_filtered:
        model.s_snr_cache = np.mean(s_snr_filtered)
    else:
        model.s_snr_cache = np.nan
        logger.warning("No valid S-SNR values found in the sample. Setting s_snr_cache to the default value NaN.")
    
    print(f"p_snr_cache: {model.p_snr_cache}")
    print(f"s_snr_cache: {model.s_snr_cache}")
    return model


def extract_metadata_from_batch(batch):
    metadata = {}
    
    # (x, loss_targets, metrics_targets, metadata)
    if isinstance(batch, tuple) and len(batch) >= 4:
        batch_metadata = batch[3]
        if batch_metadata is not None:
            metadata = batch_metadata
    
    return metadata