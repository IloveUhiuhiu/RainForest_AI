import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Chọn backend không GUI
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
import cv2

def create_spectrogram(audio_data, image_file = 'static/spectrogram.png'):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Đọc dữ liệu từ audio_data (FileStorage) thành BytesIO
    audio_stream = BytesIO(audio_data.read())

    # Load âm thanh từ stream
    y, sr = librosa.load(audio_stream, sr=None)

    # Tính mel-spectrogram
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)

    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)
