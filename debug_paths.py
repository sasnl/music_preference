#!/usr/bin/env python3
from pathlib import Path
import os

# Test the path calculation from the script
script_path = Path('code/analysis/derive_music_ABR/generate_music_anm_regressors.py')
project_root = script_path.parent.parent.parent
music_dir = project_root / 'music_stim' / 'preprocesed'

print(f"Script path: {script_path}")
print(f"Project root: {project_root}")
print(f"Music dir: {music_dir}")
print(f"Music dir exists: {music_dir.exists()}")

# Test the correct path
correct_music_dir = Path('music_stim/preprocesed')
print(f"Correct music dir: {correct_music_dir}")
print(f"Correct music dir exists: {correct_music_dir.exists()}")

# Test file discovery
if correct_music_dir.exists():
    for participant_dir in sorted(correct_music_dir.glob('[1-5]')):
        participant_id = participant_dir.name
        wav_files = sorted(list(participant_dir.glob('*_proc.wav')))
        print(f"Participant {participant_id}: {len(wav_files)} files")
        for wav_file in wav_files:
            print(f"  {wav_file.name}")

