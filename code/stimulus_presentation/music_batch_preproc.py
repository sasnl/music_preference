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
import json

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

def detect_audio_format(file_path):
    """Detect audio file format using ffprobe"""
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        format_name = data.get('format', {}).get('format_name', '').lower()
        codec_name = data.get('streams', [{}])[0].get('codec_name', '').lower() if data.get('streams') else ''
        
        # Check for DASH format
        if 'dash' in format_name or 'dash' in str(data.get('format', {}).get('tags', {})):
            return 'dash'
        elif codec_name == 'aac' and format_name in ['mov,mp4,m4a,3gp,3g2,mj2']:
            return 'aac_container'
        elif file_path.lower().endswith('.mp3'):
            return 'mp3'
        elif file_path.lower().endswith('.wav'):
            return 'wav'
        else:
            return 'unknown'
    except Exception as e:
        logging.warning(f"Could not detect format for {file_path}: {e}")
        return 'unknown'

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

def validate_processed_audio(audio, fs, original_duration, logger):
    """Validate that processed audio has reasonable properties"""
    if len(audio) == 0:
        logger.error("Processed audio is empty")
        return False
    
    processed_duration = len(audio) / fs
    if processed_duration < 0.1:  # Less than 100ms
        logger.error(f"Processed audio too short: {processed_duration:.3f}s (expected ~{original_duration:.1f}s)")
        return False
    
    if processed_duration > original_duration * 1.1:  # More than 10% longer
        logger.warning(f"Processed audio longer than original: {processed_duration:.3f}s vs {original_duration:.3f}s")
    
    # Check for silent audio
    if np.max(np.abs(audio)) < 1e-6:
        logger.error("Processed audio is silent")
        return False
    
    return True

def process_and_save(input_path, output_path, trim_length, lowpass_fc, target_rms, no_trim, logger):
    try:
        # Detect file format
        file_format = detect_audio_format(input_path)
        logger.info(f"Detected format for {input_path}: {file_format}")
        
        # Get original duration for validation
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', input_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            original_duration = float(result.stdout.strip())
        except:
            original_duration = 300  # Default 5 minutes if we can't get duration
        
        # Read audio file based on format
        if file_format in ['dash', 'aac_container']:
            # Handle DASH/AAC files with specialized ffmpeg parameters
            logger.info(f"Processing DASH/AAC file: {input_path}")
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Use specialized ffmpeg parameters for DASH files
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '48000',  # 48kHz sample rate
                '-ac', '2',  # Stereo (will be converted to mono later)
                '-y',  # Overwrite output
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"ffmpeg failed for DASH file {input_path}: {result.stderr}")
                raise Exception(f"ffmpeg processing failed: {result.stderr}")
            
            # Read the converted file
            audio, fs = sf.read(temp_path)
            os.unlink(temp_path)  # Clean up temp file
            
        elif input_path.lower().endswith('.mp3'):
            try:
                # Try pydub first for regular MP3 files
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
                    
                    # Use ffmpeg to re-encode the problematic mp3 file
                    cmd = ['ffmpeg', '-i', input_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '2', temp_path, '-y']
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        raise Exception(f"ffmpeg re-encoding failed: {result.stderr}")
                    
                    # Read the re-encoded file
                    audio, fs = sf.read(temp_path)
                    os.unlink(temp_path)  # Clean up temp file
                    
                    logger.info(f"Successfully re-encoded {input_path} using ffmpeg")
                    
                except Exception as ffmpeg_error:
                    raise Exception(f"Both pydub and ffmpeg failed: {str(e)}, ffmpeg error: {str(ffmpeg_error)}")
        else:
            # For wav files, use soundfile
            audio, fs = sf.read(input_path)
        
        # Validate the loaded audio
        if not validate_processed_audio(audio, fs, original_duration, logger):
            raise Exception("Audio validation failed")
        
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
        
        # Final validation
        if not validate_processed_audio(audio, fs, original_duration, logger):
            raise Exception("Final audio validation failed")
        
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