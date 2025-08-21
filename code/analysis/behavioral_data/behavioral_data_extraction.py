filename = "paste.txt"  # Name of the data file

results = []                   # stores parsed results from each trial
trial_num = 1                  # starts the trial index at 1
curr_song_id = None            #  stores the most recent song ID (proc.wav)

with open(filename, "r", encoding="utf8") as file:
    for line in file:
        # update the song ID whenever a 'trial_id' line appears
        if "trial_id" in line and "proc.wav" in line:
            # Try to extract what's after 'trial_id\t'
            parts = line.split()
            for part in parts:
                if part.endswith('.proc.wav') or part.endswith('_proc.wav') or part.endswith('-proc.wav') or part.endswith('proc.wav'):
                    curr_song_id = part  # found song file name
            # or do a more robust parse if format varies
            continue

        # everytime a responses line is seen look through the responses and log with song ID
        if "responses" in line and curr_song_id is not None:
            start = line.find("{")
            end = line.find("}")
            if start != -1 and end != -1:
                text = line[start+1:end]
                items = text.split()
                values = []
                i = 0
                while i < len(items):
                    if ":" in items[i] and (i + 1) < len(items):
                        value_str = items[i+1].strip().replace(",", "")
                        value_digits = ''.join(filter(lambda c: c.isdigit(), value_str))
                        try:
                            number = int(value_digits)
                        except ValueError:
                            number = value_digits
                        values.append(number)
                        i += 2
                    else:
                        i += 1
                if len(values) == 4:
                    # make sure we append the curr_song_id as part of result
                    results.append((trial_num, curr_song_id, values[0], values[1], values[2], values[3]))
                    trial_num += 1

# print the result keeping the format, including the original song id
for res in results:
    print(f"Trial {res[0]}:")
    print(f"  song_id: {res[1]}")
    print(f"  preference: {res[2]}")
    print(f"  pleasantness: {res[3]}")
    print(f"  valence_arousal: {res[4]}")
    print(f"  chills: {res[5]}")
    print()