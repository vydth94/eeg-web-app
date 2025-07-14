from flask import Flask, render_template, request
import torch
import torch.nn as nn
import pickle
import numpy as np
import os

# --- Firebase Admin SDK setup ---
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()   # Nếu bạn dùng Firestore

# ==== Mapping code và cảm xúc ====
EMOTION_MAP = {
    "HAHV": "Happy",
    "LAHV": "Calm",
    "HALV": "Angry",
    "LALV": "Sad",
}
EMO_ICON = {
    "Happy": "happy.png",
    "Calm": "calm.png",
    "Angry": "angry.png",
    "Sad": "sad.png"
}
CLASS_NAMES = list(EMOTION_MAP.keys())
EMO_LABELS = ["Happy", "Calm", "Angry", "Sad"]

# --- Model và PositionalEncoding như bạn đã train ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class EEG_ST_TCNN(nn.Module):
    def __init__(self, input_channels=32, seq_len=384, num_classes=4, n_heads=8, depth=3, dropout=0.3):
        super().__init__()
        self.pos_enc_temporal = PositionalEncoding(d_model=seq_len, max_len=input_channels)
        encoder_tm = nn.TransformerEncoderLayer(d_model=seq_len, nhead=n_heads, batch_first=True, dropout=dropout, activation='relu')
        self.transformer_tm = nn.TransformerEncoder(encoder_tm, num_layers=depth)
        self.pos_enc_spatial = PositionalEncoding(d_model=input_channels, max_len=seq_len)
        encoder_sp = nn.TransformerEncoderLayer(d_model=input_channels, nhead=n_heads, batch_first=True, dropout=dropout, activation='relu')
        self.transformer_sp = nn.TransformerEncoder(encoder_sp, num_layers=depth)
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * (input_channels // 2) * (seq_len // 2), num_classes)
    def forward(self, x):
        tm_in = self.pos_enc_temporal(x)
        tm_out = self.transformer_tm(tm_in)
        sp_in = x.permute(0, 2, 1)
        sp_in = self.pos_enc_spatial(sp_in)
        sp_out = self.transformer_sp(sp_in)
        sp_out = sp_out.permute(0, 2, 1)
        features = torch.stack([tm_out, sp_out], dim=1)
        out = self.cnn(features)
        out = out.flatten(1)
        out = self.fc(out)
        return out

# ==== Load model ====
device = torch.device("cpu")
model = EEG_ST_TCNN()
model.load_state_dict(torch.load("best_model_global.pth", map_location=device))
model.eval()

# ==== Tiền xử lý file .dat ====
def preprocess_eeg_dat(file_path):
    with open(file_path, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
        signals = d['data'][:, :32, :]
        fs, seg_len = 128, 3 * 128
        segments = []
        for signal in signals:
            filtered = np.array([signal[ch] for ch in range(32)])
            for start in range(0, filtered.shape[1] - seg_len + 1, seg_len):
                seg = filtered[:, start:start + seg_len]
                segments.append(seg)
        return segments

# ==== Flask app ====
app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        subject_id = request.form.get('subject_id')
        gender = request.form.get('gender')
        file = request.files['eeg_file']
        save_path = 'uploaded.dat'
        file.save(save_path)
        X = preprocess_eeg_dat(save_path)
        os.remove(save_path)
        preds, probs, emotions, emo_idxs = [], [], [], []
        with torch.no_grad():
            for seg in X:
                x_ = (seg - np.mean(seg)) / (np.std(seg) + 1e-5)
                x_tensor = torch.tensor(x_, dtype=torch.float32).unsqueeze(0)
                out = model(x_tensor)
                softmax = torch.softmax(out, dim=1).cpu().numpy().flatten()
                pred_idx = int(np.argmax(softmax))
                label = CLASS_NAMES[pred_idx]
                preds.append(label)
                emotions.append(EMOTION_MAP[label])
                emo_idxs.append(EMO_LABELS.index(EMOTION_MAP[label]))
                probs.append([float(v) for v in softmax])  # Sửa lại: giữ là float!
        # Tính % mỗi cảm xúc
        from collections import Counter
        total = len(emotions)
        cnt = Counter(emotions)
        emotion_percent = {emo: int(100*cnt.get(emo,0)/total) for emo in EMO_LABELS}
        return render_template(
            'result.html',
            subject_id=subject_id,
            gender=gender,
            results=[{"segment": i+1, "label": preds[i], "emotion": emotions[i], "probs": probs[i]} for i in range(total)],
            class_names=CLASS_NAMES,
            emotion_map=EMOTION_MAP,
            emotion_percent=emotion_percent,
            emo_icon=EMO_ICON,
            emotion_labels=emo_idxs
        )
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
