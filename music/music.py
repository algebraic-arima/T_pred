import mido
import numpy as np
import pandas as pd

def parse_midi_to_sequence(midi_path, time_step=100, num_tracks=4):
    # 加载MIDI文件
    midi = mido.MidiFile(midi_path)
    
    current_notes = np.zeros(num_tracks, dtype=int)
    
    sequence = []
    current_time = 0
    
    events = []
    for i, track in enumerate(midi.tracks[:num_tracks]):
        time = 0
        for msg in track:
            time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                events.append((time, i, 'note_on', msg.note))
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                events.append((time, i, 'note_off', msg.note))
    
    events.sort(key=lambda x: x[0])
    
    event_ptr = 0
    max_time = events[-1][0] if events else 0
    while current_time <= max_time:
        # 处理所有发生在当前时间之前的事件
        while event_ptr < len(events) and events[event_ptr][0] <= current_time:
            time, track_idx, action, note = events[event_ptr]
            if action == 'note_on':
                current_notes[track_idx] = note
            elif action == 'note_off' and current_notes[track_idx] == note:
                current_notes[track_idx] = 0 
            event_ptr += 1
        sequence.append(current_notes.copy())
        current_time += time_step
    
    return np.array(sequence)

# 示例使用
midi_path = "./music/BWV1080.mid"
time_series = parse_midi_to_sequence(midi_path)
print(time_series.shape)  # (时间步数, 4)

root_path = "./music/"
file_name = "bach.csv"
df = pd.DataFrame(time_series)
df.to_csv(root_path + file_name, index=False, header=['s', 'a', 't', 'b'])


