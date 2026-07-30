import numpy as np
import sounddevice as sd

# 导入琵琶模块
from pipa_synth import OvertoneSynth, render_melody, PLUCK_TIMBRE, SAMPLING_RATE
# 导入二胡模块
from erhu_synth import ErhuSynth, render_erhu

# ==========================================
# 1. 实例化两件乐器
# ==========================================

pipa_synth = OvertoneSynth(
    partial_ratios=PLUCK_TIMBRE,
    attack=0.001, decay=1.2, sustain=0.8, release=0.5, damping=0.1,
    pitch_drop_cents=10.0, pitch_drop_time=0.1, noise_amp=0.01
)

# 二胡稍微调快一点起音，适应快节奏的断奏
erhu_synth = ErhuSynth() 

# ==========================================
# 辅助函数
# ==========================================

def add_pipa_chord(track, notes, total_duration, vel, spread=0.04):
    """琵琶扫弦"""
    num_notes = len(notes)
    for i, note in enumerate(notes):
        step_time = total_duration - (spread * (num_notes - 1)) if i == num_notes - 1 else spread
        track.append((note, total_duration - i*spread, vel, step_time))

def run_16th_notes(track, notes, vel=0.85):
    """添加16分音符快速跑句 (每音0.25拍)"""
    for note in notes:
        track.append((note, 0.25, vel, 0.25))

# ==========================================
# 2. 编写双声部快板协奏曲 (D Harmonic Minor)
# BPM 提升至 100，增强跑动感
# ==========================================

pipa_track = []
erhu_track = []

# ---------------------------------------------------------
# 乐章一：短促起势 (抛弃冗长引子，直接切入节奏)
# ---------------------------------------------------------
# 琵琶：强有力的切分和弦，干脆利落
for _ in range(2):
    add_pipa_chord(pipa_track, ["D3", "A3", "D4"], 0.5, 1.0, spread=0.02)
    pipa_track.extend([("REST", 0.5, 0), ("D4", 0.5, 0.8, 0.5), ("F4", 0.5, 0.8, 0.5)])
    add_pipa_chord(pipa_track, ["A2", "E3", "C#4"], 1.0, 0.9, spread=0.02)
    pipa_track.extend([("REST", 1.0, 0)])

# 二胡：顿弓回应，带有挑衅感
for _ in range(2):
    erhu_track.extend([
        ("REST", 2.0, 0), 
        ("A4", 0.25, 0.9, 0.25), ("Bb4", 0.25, 0.9, 0.25), 
        ("A4", 0.25, 0.8, 0.25), ("G4", 0.25, 0.8, 0.25),
        ("F4", 0.5, 0.9, 0.5), ("E4", 0.5, 0.9, 0.5)
    ])


# ---------------------------------------------------------
# 乐章二：激烈对话 (Call and Response)
# 特色：琵琶与二胡的十六分音符短句交替互咬
# ---------------------------------------------------------
for loop in range(2):
    # 问 1：琵琶极速上行
    run_16th_notes(pipa_track, ["D4", "E4", "F4", "G4", "A4", "Bb4", "A4", "G4"])
    pipa_track.extend([("F4", 0.5, 0.9, 0.5), ("D4", 1.5, 0.9, 1.5)]) # 琵琶收尾留白
    
    # 答 1：二胡等待，然后极速下行回应
    erhu_track.extend([("REST", 2.0, 0)])
    erhu_track.extend([("A5", 0.25, 0.9, 0.25), ("G5", 0.25, 0.9, 0.25), ("F5", 0.25, 0.8, 0.25), ("E5", 0.25, 0.8, 0.25)])
    erhu_track.extend([("D5", 0.25, 0.9, 0.25), ("C#5", 0.25, 0.9, 0.25), ("D5", 0.5, 0.9, 0.5)])
    erhu_track.extend([("A4", 1.0, 0.8, 1.0)])

    # 问 2：琵琶低音区扫弦挑衅
    add_pipa_chord(pipa_track, ["A2", "E3", "A3"], 0.5, 0.9, 0.02)
    run_16th_notes(pipa_track, ["C#4", "D4", "E4", "F4", "G4", "F4", "E4", "C#4"])
    pipa_track.extend([("D4", 1.5, 0.9, 1.5)])
    
    # 答 2：二胡滑音拉扯后接入快速连奏
    erhu_track.extend([("REST", 0.5, 0)])
    erhu_track.extend([("E5", 0.5, 0.9, 0.5, {"slide_from": "C#5", "slide_time": 0.1})])
    erhu_track.extend([("A4", 0.25, 0.8, 0.25), ("Bb4", 0.25, 0.8, 0.25), ("C#5", 0.25, 0.9, 0.25), ("D5", 0.25, 0.9, 0.25)])
    erhu_track.extend([("E5", 0.25, 0.9, 0.25), ("F5", 0.25, 0.9, 0.25), ("D5", 1.5, 0.9, 1.5)])


# ---------------------------------------------------------
# 乐章三：双轨齐驱 (复合节奏并行)
# 特色：琵琶持续分解和弦律动，二胡拉奏切分主旋律，不再互相等待
# ---------------------------------------------------------
for _ in range(4):
    # 琵琶：连绵不绝的8分音符律动 (每音0.5拍)
    pipa_track.extend([
        ("D4", 0.5, 0.8, 0.5), ("A3", 0.5, 0.7, 0.5), ("F4", 0.5, 0.8, 0.5), ("A3", 0.5, 0.7, 0.5),
        ("E4", 0.5, 0.8, 0.5), ("A3", 0.5, 0.7, 0.5), ("G4", 0.5, 0.8, 0.5), ("A3", 0.5, 0.7, 0.5),
        ("F4", 0.5, 0.8, 0.5), ("D4", 0.5, 0.7, 0.5), ("Bb3", 0.5, 0.8, 0.5), ("D4", 0.5, 0.7, 0.5),
        ("C#4", 0.5, 0.9, 0.5), ("A3", 0.5, 0.7, 0.5), ("E4", 0.5, 0.8, 0.5), ("A3", 0.5, 0.7, 0.5)
    ])

    # 二胡：充满跃动感的切分音主旋律
    erhu_track.extend([
        ("D5", 0.75, 0.9, 0.75), ("F5", 0.25, 0.8, 0.25), ("E5", 0.5, 0.9, 0.5), ("C#5", 0.5, 0.8, 0.5),
        ("D5", 0.5, 0.9, 0.5), ("REST", 0.5, 0), ("A4", 1.0, 0.9, 1.0, {"vib_depth": 15}),
        ("Bb4", 0.75, 0.9, 0.75), ("G4", 0.25, 0.8, 0.25), ("A4", 0.5, 0.9, 0.5), ("F4", 0.5, 0.8, 0.5),
        ("E4", 1.0, 0.9, 1.0), ("A4", 1.0, 0.9, 1.0)
    ])


# ---------------------------------------------------------
# 乐章四：高潮爆发，音阶对飙
# 特色：琵琶和二胡同时进行极速16分音符跑动，最后汇聚
# ---------------------------------------------------------
# 前两拍：琵琶上行，二胡下行
run_16th_notes(pipa_track, ["D3", "E3", "F3", "G3", "A3", "Bb3", "C#4", "D4"])
run_16th_notes(erhu_track, ["D6", "C#6", "Bb5", "A5", "G5", "F5", "E5", "D5"])

# 后两拍：琵琶下行，二胡上行
run_16th_notes(pipa_track, ["F4", "E4", "D4", "C#4", "Bb3", "A3", "G3", "F3"])
run_16th_notes(erhu_track, ["A4", "Bb4", "C#5", "D5", "E5", "F5", "G5", "A5"])

# 再来一轮更密集的对飙
run_16th_notes(pipa_track, ["D4", "E4", "F4", "G4", "A4", "Bb4", "C#5", "D5"])
run_16th_notes(erhu_track, ["F5", "E5", "D5", "C#5", "Bb4", "A4", "G4", "F4"])

run_16th_notes(pipa_track, ["A4", "G4", "F4", "E4", "D4", "C#4", "Bb3", "A3"])
run_16th_notes(erhu_track, ["D4", "E4", "F4", "G4", "A4", "Bb4", "C#5", "E5"])


# ---------------------------------------------------------
# 尾声：戛然而止的强收
# 特色：不拖泥带水，3次强力顿奏直接结束
# ---------------------------------------------------------
for _ in range(2):
    add_pipa_chord(pipa_track, ["D3", "A3", "D4", "F4"], 0.5, 1.0, spread=0.01)
    pipa_track.extend([("REST", 0.5, 0)])
    
    erhu_track.extend([("D5", 0.5, 1.0, 0.5), ("REST", 0.5, 0)])

# 最后一击！极短、极爆
add_pipa_chord(pipa_track, ["D2", "A2", "D3", "F3", "A3", "D4"], 2.0, 1.0, spread=0.02)
erhu_track.extend([("D6", 2.0, 1.0, 2.0, {"slide_from": "D5", "slide_time": 0.1})])


# ==========================================
# 3. 渲染与混音 (BPM 设为 100，突出快板节奏)
# ==========================================
print("正在渲染琵琶音轨 (快板跑动与干脆和弦)...")
pipa_audio = render_melody(pipa_synth, pipa_track, bpm=100)

print("正在渲染二胡音轨 (短促顿弓与极速音阶)...")
erhu_audio = render_erhu(erhu_synth, erhu_track, bpm=100)

max_frames = max(len(pipa_audio), len(erhu_audio))
mixed_audio = np.zeros(max_frames)

mixed_audio[:len(pipa_audio)] += pipa_audio * 0.5 
mixed_audio[:len(erhu_audio)] += erhu_audio * 0.5 

# 压限防止削波
mixed_audio = np.tanh(mixed_audio * 1.2)

print("开始播放快板竞奏 (BPM=100)...")
sd.play(mixed_audio, SAMPLING_RATE)
sd.wait()
print("播放结束。")