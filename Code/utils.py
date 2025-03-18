import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import librosa
import os
import heapq
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas
from tensorflow.keras.models import model_from_json
from scipy import stats
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import soundfile as sf


def nb_audios(file_path):
    print(f'number of audios in "{file_path}" are ' + str(len(os.listdir(file_path))))

def keep_n_files(directory, n):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    files.sort()
    num_files_to_delete = len(files) - n

    for i in range(num_files_to_delete):
        os.remove(os.path.join(directory, files[i]))

def split_Audios(dir_path, frame_length, hop_length):
  frame_list = []
  print(f'number of audios in "{dir_path}" are ' + str(len(os.listdir(dir_path))))
  for audio in os.listdir(dir_path):
    if audio.split('.')[-1] != 'wav':
      continue
    audio_path = os.path.join(dir_path, audio)
    print(audio_path)
    y, sr = librosa.load(audio_path, sr=8000)
    for i in range(0, y.shape[0] - frame_length + 1, hop_length):
      frame_list.append(y[i : i + frame_length])
      print(i)

  print(f'number of frames in "{dir_path}" are ' + str(len(frame_list)))
  
  frame_list = np.vstack(frame_list)
  print(frame_list.shape)
  return frame_list

def plot_spectrogram(spectrogram, sr, title="Spectrogram"):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log', cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()

def spectrogram(frame_list, sr=8000, save=False, plot=False):
    spectrograms = []
    phases = []

    plt.figure(figsize=(10, len(frame_list) * 3))  # Adjust figure size for multiple spectrograms

    for i, frame in enumerate(frame_list):
        # Compute STFT
        D = librosa.stft(frame, n_fft=255, hop_length=63) # 255 and 63 as with 256 and 64 the shape will be 129,127. We need 128,128
        print("D.shape in spec")
        print(D.shape)
        magnitude, phase = librosa.magphase(D)
        print('magnitude and ohase')
        print(magnitude.shape, phase.shape)
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)  # Convert to dB scale

        spectrograms.append(magnitude_db)
        phases.append(phase)


        if plot:
          # Plot each spectrogram
          plt.subplot(len(frame_list), 1, i + 1)
          librosa.display.specshow(magnitude_db, sr=sr, x_axis='time', y_axis='log', cmap='inferno')
          plt.colorbar(format='%+2.0f dB')
          plt.title(f'Spectrogram {i+1}')

          # Save if needed
          if save:
              plt.savefig(f'spectrogram_{i+1}.png', bbox_inches='tight', pad_inches=0)

    plt.tight_layout()
    plt.show()

    return spectrograms, phases  # Return all spectrograms & phases

def normalize_spectrogram(spectrograms, print=False):
    """Normalize spectrogram to range [-1, 1] (for tanh activation)."""
    if isinstance(spectrograms, list):
        normalized_list = []
        for i, spec in enumerate(spectrograms):
            norm_spec = 2 * (spec - spec.min()) / (spec.max() - spec.min()) - 1
            normalized_list.append(norm_spec)
            if print:
              print(f"Spectrogram {i+1}: min={norm_spec.min()}, max={norm_spec.max()}")

        return normalized_list

    # Normalize a single spectrogram
    return 2 * (spectrograms - spectrograms.min()) / (spectrograms.max() - spectrograms.min()) - 1

def inverse_min_max_normalization(normalized_audio, min_value, max_value):
    return ((normalized_audio + 1) * (max_value - min_value)) / 2 + min_value

def spectrogram_to_audio(magnitude_db_list, phase_list, hop_length=63):
    reconstructed_audio = []

    for magnitude_db, phase in zip(magnitude_db_list, phase_list):
        # Convert from dB scale to amplitude
        magnitude = librosa.db_to_amplitude(magnitude_db)
        # print("magnitude.shape")
        # print(magnitude.shape)

        # Reconstruct the complex spectrogram
        D = magnitude * phase  # Element-wise multiplication

        # print("D.shape")
        # print(D.shape)
        # Inverse STFT to get back the audio signal
        audio = librosa.istft(D,n_fft=255, hop_length=hop_length)
        reconstructed_audio.append(audio)

    return np.array(reconstructed_audio)

hop_length = 8064

def predict_audio(audio_dir):
    frame_length = 8064
    # load json and create model
    json_file = open('model_unet_new.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model/
    loaded_model.load_weights('model_unet_best_new.h5')
    print("Loaded model from disk")

    # for x in os.listdir(audio_dir):
    #   y, sr = librosa.load(os.path.join(audio_dir, x), sr=sample_rate)
    #   print(y.shape)
    noisy_frame_list = split_Audios(audio_dir, frame_length, hop_length)  # It should take dir and return framelist of all the audios
    mag_n, pha_n = spectrogram(noisy_frame_list)

    min_value = np.min(mag_n)
    max_value = np.max(mag_n)

    mag_n = np.array(mag_n)
    pha_n = np.array(pha_n)
    X_input = normalize_spectrogram(mag_n)
    X_input = np.array(X_input)
    X_input = X_input.reshape(X_input.shape[0],X_input.shape[1],X_input.shape[2],1)
    print('X_input.shape')
    print(X_input.shape)
    X_pred = loaded_model.predict(X_input)
    X_pred = np.squeeze(X_pred, axis=-1)
    x_reconstructed = inverse_min_max_normalization(X_pred, min_value, max_value)
    print(x_reconstructed.shape)
    print(pha_n.shape)
    print(frame_length)
    # print(hop_length_fft)

    reconstructed_audio = spectrogram_to_audio(x_reconstructed, pha_n)
    if reconstructed_audio.shape[1] < frame_length:
        # Padding the reconstructed audio if it's shorter than expected
        padding = frame_length - reconstructed_audio.shape[1]
        reconstructed_audio = np.pad(reconstructed_audio, ((0, 0), (0, padding)), mode='constant')
    nb_samples = reconstructed_audio.shape[0]
    #denoise_long = reconstructed_audio.reshape(1, nb_samples * frame_length)*10
    print(reconstructed_audio.shape)
    expected_size = nb_samples * frame_length
    actual_size = reconstructed_audio.size
    print(f"Expected size: {expected_size}, Actual size: {actual_size}")

    x = reconstructed_audio.reshape(1, nb_samples * frame_length)*10

    return sf.write(f'{audio_dir}.wav', x[0, :], 8000, 'PCM_24')

