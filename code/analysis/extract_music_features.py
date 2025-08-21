#!/usr/bin/env python3
"""
extract_music_features.py

Extracts multiple audio features for music preference analysis:
1. Gammatone-filtered amplitude envelope (128 channels, 60-6000 Hz)
2. Full-band Hilbert envelope (alternative/fallback)
3. Half-wave rectified first derivative of amplitude envelope
4. Spectral novelty/flux following Müller (2015)

All features are downsampled to 128 Hz and z-score normalized.

References:
- Daube et al., 2019; Di Liberto et al., 2020 for onset-sensitive envelope derivative
- Müller, 2015 for spectral flux computation
"""

import numpy as np
import scipy.signal
import librosa
import matplotlib.pyplot as plt
import argparse
import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Try to import gammatone package, fallback gracefully
try:
    from gammatone.filters import centre_freqs, erb_filterbank, make_erb_filters
    GAMMATONE_AVAILABLE = True
except ImportError:
    GAMMATONE_AVAILABLE = False
    warnings.warn("Gammatone package not available. Will use full-band Hilbert envelope as primary feature.")


def load_audio(audio_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Load audio file as mono and optionally resample."""
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True, dtype=np.float32)
    return audio, sr


def compute_gammatone_envelope(audio: np.ndarray, sr: int, n_channels: int = 128, 
                               freq_low: float = 60, freq_high: float = 6000) -> np.ndarray:
    """
    Compute gammatone-filtered amplitude envelope by averaging subband envelopes.
    
    Args:
        audio: mono audio signal
        sr: sampling rate
        n_channels: number of gammatone channels
        freq_low: lowest center frequency (Hz)
        freq_high: highest center frequency (Hz)
    
    Returns:
        envelope: amplitude envelope time series
    """
    if not GAMMATONE_AVAILABLE:
        raise ImportError("Gammatone package required for subband envelope computation")
    
    # Generate linearly spaced center frequencies
    center_freqs = np.linspace(freq_low, freq_high, n_channels)
    
    # Create ERB filter coefficients
    filter_coefs = make_erb_filters(sr, center_freqs)
    
    # Apply filterbank to get filtered signals (one per channel)
    filtered_signals = erb_filterbank(audio, filter_coefs)
    
    # Compute subband envelopes
    subband_envelopes = np.zeros((n_channels, len(audio)), dtype=np.float32)
    
    for i in range(n_channels):
        # Compute analytic signal and take magnitude
        analytic = scipy.signal.hilbert(filtered_signals[i])
        subband_envelopes[i] = np.abs(analytic).astype(np.float32)
    
    # Average across all subbands to get full-band envelope
    envelope = np.mean(subband_envelopes, axis=0)
    
    return envelope


def compute_hilbert_envelope(audio: np.ndarray) -> np.ndarray:
    """Compute full-band Hilbert envelope (magnitude of analytic signal)."""
    analytic = scipy.signal.hilbert(audio)
    envelope = np.abs(analytic).astype(np.float32)
    return envelope


def compute_envelope_derivative(envelope: np.ndarray) -> np.ndarray:
    """
    Compute half-wave rectified first derivative of amplitude envelope.
    Emphasizes onsets by keeping only positive changes.
    """
    # First-order temporal derivative (discrete difference)
    derivative = np.diff(envelope, prepend=envelope[0])
    
    # Half-wave rectification (set negative values to 0)
    derivative_hwrect = np.maximum(derivative, 0).astype(np.float32)
    
    return derivative_hwrect


def compute_spectral_flux(audio: np.ndarray, sr: int, frame_length: int = 374, 
                         hop_length: int = 187) -> np.ndarray:
    """
    Compute spectral novelty (spectral flux) following Müller (2015).
    
    Args:
        audio: mono audio signal
        sr: sampling rate
        frame_length: STFT frame length in samples
        hop_length: STFT hop length in samples
    
    Returns:
        spectral_flux: 1D novelty curve
    """
    # Compute STFT
    stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length, window='hann')
    magnitude_spec = np.abs(stft)
    
    # Convert to logarithmic amplitude spectrogram
    log_magnitude = np.log1p(magnitude_spec)
    
    # Compute spectral flux as positive differences between consecutive frames
    spectral_diff = np.diff(log_magnitude, axis=1, prepend=log_magnitude[:, [0]])
    spectral_flux = np.sum(np.maximum(spectral_diff, 0), axis=0).astype(np.float32)
    
    return spectral_flux


def resample_to_target_rate(signal: np.ndarray, orig_rate: float, target_rate: float = 128.0) -> np.ndarray:
    """Resample signal to target rate using scipy."""
    if orig_rate == target_rate:
        return signal
    
    # Calculate resampling ratio
    num_samples_target = int(len(signal) * target_rate / orig_rate)
    
    # Use scipy's resample for clean resampling
    resampled = scipy.signal.resample(signal, num_samples_target).astype(np.float32)
    
    return resampled


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    """Z-score normalize signal (mean=0, std=1)."""
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    
    if std_val == 0:
        warnings.warn("Signal has zero standard deviation. Returning zero-centered signal.")
        return signal - mean_val
    
    normalized = (signal - mean_val) / std_val
    
    # Check for NaNs/Infs
    if np.any(~np.isfinite(normalized)):
        raise ValueError("NaN or Inf values found after z-score normalization")
    
    return normalized.astype(np.float32)


def truncate_to_shortest(*arrays) -> Tuple[np.ndarray, ...]:
    """Truncate all arrays to the length of the shortest one."""
    min_length = min(len(arr) for arr in arrays)
    return tuple(arr[:min_length] for arr in arrays)


def create_time_axis(length: int, sample_rate: float = 128.0) -> np.ndarray:
    """Create time axis in seconds for given length and sample rate."""
    return np.arange(length, dtype=np.float32) / sample_rate


def save_features(features_dict: Dict[str, np.ndarray], metadata: Dict[str, Any], 
                 output_prefix: str) -> None:
    """Save features to both CSV and NPZ formats."""
    import pandas as pd
    
    # Save as CSV
    df = pd.DataFrame(features_dict)
    csv_path = f"{output_prefix}_features.csv"
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Features saved to: {csv_path}")
    
    # Save as NPZ with metadata
    npz_path = f"{output_prefix}_features.npz"
    np.savez_compressed(npz_path, **features_dict, **metadata)
    print(f"Features and metadata saved to: {npz_path}")


def plot_features(features_dict: Dict[str, np.ndarray], output_prefix: str) -> None:
    """Create QC plot showing all four feature curves."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Extracted Music Features', fontsize=14)
    
    time_s = features_dict['time_s']
    
    # Plot each feature
    features_to_plot = [
        ('amp_env_gammatone', 'Gammatone Amplitude Envelope (Primary)', 'blue'),
        ('amp_env_hilbert_fullband', 'Full-band Hilbert Envelope (Alternative)', 'green'),
        ('amp_env_deriv_hwrect', 'Half-wave Rectified Envelope Derivative', 'red'),
        ('spectral_flux', 'Spectral Flux (Novelty)', 'purple')
    ]
    
    for i, (key, title, color) in enumerate(features_to_plot):
        if key in features_dict:
            axes[i].plot(time_s, features_dict[key], color=color, linewidth=0.8)
            axes[i].set_ylabel('Z-score')
            axes[i].set_title(title)
            axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    
    # Save plot
    plot_path = f"{output_prefix}_features.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Feature plot saved to: {plot_path}")
    plt.close()


def extract_music_features(audio_path: str, target_sr: Optional[int] = None, 
                          output_prefix: Optional[str] = None, 
                          use_hilbert_only: bool = False) -> Dict[str, np.ndarray]:
    """
    Main function to extract all music features.
    
    Args:
        audio_path: path to audio file
        target_sr: optional target sampling rate for audio
        output_prefix: output file prefix (default: audio stem)
        use_hilbert_only: skip gammatone and use only Hilbert envelope
    
    Returns:
        Dictionary containing all extracted features and metadata
    """
    print(f"Processing audio: {audio_path}")
    
    # Load audio
    audio, sr = load_audio(audio_path, target_sr)
    print(f"Loaded audio: {len(audio)} samples at {sr} Hz ({len(audio)/sr:.2f} seconds)")
    
    # Set output prefix if not provided
    if output_prefix is None:
        output_prefix = Path(audio_path).stem
    
    # Feature extraction
    features = {}
    
    # 1. Amplitude envelopes
    if GAMMATONE_AVAILABLE and not use_hilbert_only:
        print("Computing gammatone-filtered amplitude envelope...")
        primary_envelope = compute_gammatone_envelope(audio, sr)
        features['primary_envelope_raw'] = primary_envelope
        envelope_type = 'gammatone'
    else:
        print("Computing full-band Hilbert envelope as primary...")
        primary_envelope = compute_hilbert_envelope(audio)
        features['primary_envelope_raw'] = primary_envelope
        envelope_type = 'hilbert'
    
    # Alternative envelope (always compute full-band Hilbert)
    print("Computing full-band Hilbert envelope...")
    hilbert_envelope = compute_hilbert_envelope(audio)
    features['hilbert_envelope_raw'] = hilbert_envelope
    
    # 2. Half-wave rectified derivative of primary envelope
    print("Computing half-wave rectified envelope derivative...")
    envelope_deriv = compute_envelope_derivative(primary_envelope)
    features['envelope_deriv_raw'] = envelope_deriv
    
    # 3. Spectral flux
    print("Computing spectral flux...")
    frame_length = 374
    hop_length = 187
    spectral_flux = compute_spectral_flux(audio, sr, frame_length, hop_length)
    features['spectral_flux_raw'] = spectral_flux
    
    # Original sample rates for resampling
    audio_rate = float(sr)
    spectral_flux_rate = float(sr) / hop_length
    
    print(f"Original rates - Audio: {audio_rate} Hz, Spectral flux: {spectral_flux_rate:.1f} Hz")
    
    # 4. Downsample all to 128 Hz
    target_rate = 128.0
    print(f"Downsampling all features to {target_rate} Hz...")
    
    primary_env_128 = resample_to_target_rate(primary_envelope, audio_rate, target_rate)
    hilbert_env_128 = resample_to_target_rate(hilbert_envelope, audio_rate, target_rate)
    deriv_env_128 = resample_to_target_rate(envelope_deriv, audio_rate, target_rate)
    flux_128 = resample_to_target_rate(spectral_flux, spectral_flux_rate, target_rate)
    
    # 5. Truncate to shortest length
    print("Truncating features to matching length...")
    primary_env_128, hilbert_env_128, deriv_env_128, flux_128 = truncate_to_shortest(
        primary_env_128, hilbert_env_128, deriv_env_128, flux_128
    )
    
    final_length = len(primary_env_128)
    print(f"Final feature length: {final_length} samples ({final_length/target_rate:.2f} seconds)")
    
    # 6. Z-score normalize
    print("Applying z-score normalization...")
    primary_env_norm = zscore_normalize(primary_env_128)
    hilbert_env_norm = zscore_normalize(hilbert_env_128)
    deriv_env_norm = zscore_normalize(deriv_env_128)
    flux_norm = zscore_normalize(flux_128)
    
    # 7. Create time axis
    time_axis = create_time_axis(final_length, target_rate)
    
    # Prepare final feature dictionary
    features_final = {
        'time_s': time_axis,
        'amp_env_gammatone' if envelope_type == 'gammatone' else 'amp_env_hilbert_primary': primary_env_norm,
        'amp_env_hilbert_fullband': hilbert_env_norm,
        'amp_env_deriv_hwrect': deriv_env_norm,
        'spectral_flux': flux_norm
    }
    
    # Standardize naming for consistent output
    if envelope_type == 'hilbert':
        features_final['amp_env_gammatone'] = primary_env_norm  # Use same name for consistency
    
    # Metadata
    metadata = {
        'original_sr': sr,
        'target_sr': int(target_rate),
        'frame_length': frame_length,
        'hop_length': hop_length,
        'envelope_type': envelope_type,
        'n_samples_original': len(audio),
        'n_samples_features': final_length,
        'duration_seconds': float(final_length / target_rate)
    }
    
    # Save outputs
    print("Saving features...")
    save_features(features_final, metadata, output_prefix)
    
    # Create plot
    print("Creating visualization...")
    plot_features(features_final, output_prefix)
    
    print("Feature extraction completed successfully!")
    
    return {**features_final, **metadata}


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Extract music features: gammatone envelope, Hilbert envelope, "
                   "envelope derivative, and spectral flux",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--audio', '-a', required=True, type=str,
                       help='Path to input audio file')
    parser.add_argument('--target-sr', type=int, default=None,
                       help='Target sampling rate for audio loading (optional)')
    parser.add_argument('--out-prefix', '-o', type=str, default=None,
                       help='Output filename prefix (default: audio filename stem)')
    parser.add_argument('--use-hilbert-only', action='store_true',
                       help='Skip gammatone filtering and use only full-band Hilbert envelope')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.audio).exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    
    # Extract features
    try:
        extract_music_features(
            audio_path=args.audio,
            target_sr=args.target_sr,
            output_prefix=args.out_prefix,
            use_hilbert_only=args.use_hilbert_only
        )
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        raise


if __name__ == '__main__':
    main()