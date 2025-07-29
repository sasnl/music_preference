import argparse
import logging
import os
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import librosa
import scipy.signal as sig
import shutil
import sys
import subprocess
import tempfile

def parse_args():
    parser = argparse.ArgumentParser(description="Batch music preprocessing: mp3/wav to mono, flatten, normalize, resample, trim, with directory structure and logging.")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory (recursively process all mp3/wav files)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory (preserve structure)')
    parser.add_argument('--trim_length', type=int, default=60, help='Trim length in seconds (default: 60)')
    parser.add_argument('--lowpass_fc', type=float, default=0.1, help='Low-pass envelope cutoff frequency in Hz (default: 0.1)')
    parser.add_argument('--target_rms', type=float, default=0.01, help='Target RMS for normalization (default: 0.01)')
    parser.add_argument('--log_file', type=str, default=None, help='Log file path (default: output_dir/process_log.txt)')
    parser.add_argument('--no_trim', action='store_true', help='Do not trim audio, preserve original length (default: False)')
    return parser.parse_args()

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def collect_audio_files(input_dir):
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(('.mp3', '.wav')):
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, input_dir)
                audio_files.append((abs_path, rel_path))
    return audio_files

def stereo2mono(audio):
    if audio.ndim == 2:
        return np.mean(audio, axis=1)
    return audio

def flatten_envelope(audio, fs, fc=0.1):
    b, a = sig.butter(1, fc / (fs / 2.))
    env = sig.filtfilt(b, a, np.abs(audio))
    env += env.std() * 0.1  # Prevent division by zero
    return audio / env

def normalize_rms(audio, target_rms=0.01):
    current_rms = np.sqrt(np.mean(audio**2))
    return audio / current_rms * target_rms

def process_and_save(input_path, output_path, trim_length, lowpass_fc, target_rms, no_trim, logger):
    try:
        # Read audio file
        if input_path.lower().endswith('.mp3'):
            try:
                # Try pydub first
                audio_segment = AudioSegment.from_mp3(input_path)
                audio = np.array(audio_segment.get_array_of_samples())
                if audio_segment.channels == 2:
                    audio = audio.reshape((-1, 2))
                fs = audio_segment.frame_rate
            except Exception as e:
                # If pydub fails, try ffmpeg re-encoding
                logger.warning(f"pydub failed to decode {input_path}, trying ffmpeg re-encoding...")
                try:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    # Use ffmpeg to re-encode the problematic mp3/DASH file
                    # For YouTube DASH files, we need to handle them differently
                    # First try to extract audio stream specifically
                    cmd = ['ffmpeg', '-i', input_path, '-map', '0:a:0', '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', temp_path, '-y']
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        # Try alternative approach for DASH files
                        logger.warning(f"First ffmpeg attempt failed, trying alternative approach...")
                        cmd2 = ['ffmpeg', '-i', input_path, '-map', '0:a:0', '-vn', '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '2', temp_path, '-y']
                        result2 = subprocess.run(cmd2, capture_output=True, text=True)
                        
                        if result2.returncode != 0:
                            # Try without map parameter
                            logger.warning(f"Second ffmpeg attempt failed, trying without map parameter...")
                            cmd3 = ['ffmpeg', '-i', input_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '2', temp_path, '-y']
                            result3 = subprocess.run(cmd3, capture_output=True, text=True)
                            
                            if result3.returncode != 0:
                                raise Exception(f"ffmpeg re-encoding failed: {result.stderr}, second attempt: {result2.stderr}, third attempt: {result3.stderr}")
                    
                    # Read the re-encoded file (moved outside the if-else blocks)
                    audio, fs = sf.read(temp_path)
                    
                    # Clean up temporary file
                    os.unlink(temp_path)
                    
                    logger.info(f"Successfully re-encoded {input_path} using ffmpeg")
                    
                except Exception as ffmpeg_error:
                    raise Exception(f"Both pydub and ffmpeg failed: {str(e)}, ffmpeg error: {str(ffmpeg_error)}")
        else:
            # For wav files, use soundfile
            audio, fs = sf.read(input_path)
        
        # Convert to mono if stereo
        audio = stereo2mono(audio)
        
        # Flatten envelope
        audio = flatten_envelope(audio, fs, lowpass_fc)
        
        # Normalize RMS
        audio = normalize_rms(audio, target_rms)
        
        # Resample to 48kHz
        if fs != 48000:
            audio = librosa.resample(audio, orig_sr=fs, target_sr=48000)
            fs = 48000
        
        # Trim to desired length (only if not no_trim)
        if not no_trim:
            target_samples = trim_length * fs
            if len(audio) > target_samples:
                audio = audio[:target_samples]
                logger.info(f"Trimmed audio to {trim_length} seconds")
            elif len(audio) < target_samples:
                logger.warning(f"Audio {input_path} is shorter than {trim_length}s, padding with zeros")
                # Pad with zeros if too short
                padding = np.zeros(target_samples - len(audio))
                audio = np.concatenate([audio, padding])
        else:
            logger.info(f"Preserved original audio length: {len(audio)/fs:.2f} seconds")
        
        # Save processed audio
        sf.write(output_path, audio, fs)
        
        logger.info(f"Processed: {input_path} -> {output_path} | Success")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {str(e)}")
        return False

def main():
    args = parse_args()
    
    # Check ffmpeg availability
    if not check_ffmpeg():
        logging.error("ffmpeg is not available. Please install ffmpeg first.")
        sys.exit(1)
    
    log_file = args.log_file or os.path.join(args.output_dir, 'process_log.txt')
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(log_file)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Batch music preprocessing started. Input: {args.input_dir}, Output: {args.output_dir}")
    logger.info(f"Trim mode: {'Disabled (preserve original length)' if args.no_trim else f'Enabled (trim to {args.trim_length}s)'}")
    
    # Collect all audio files
    audio_files = collect_audio_files(args.input_dir)
    logger.info(f"Found {len(audio_files)} audio files to process.")
    
    # Process each file
    for input_path, rel_path in audio_files:
        # Create output directory structure
        output_dir = os.path.join(args.output_dir, os.path.dirname(rel_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(rel_path))[0]
        output_base = f"{base_name}_proc"
        
        # Process and save
        process_and_save(input_path, os.path.join(output_dir, f"{output_base}.wav"), 
                         args.trim_length, args.lowpass_fc, args.target_rms, args.no_trim, logger)
    
    logger.info("Batch processing completed.")

if __name__ == "__main__":
    main() 