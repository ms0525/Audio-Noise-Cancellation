# Audio Noise Cancellation
Built a deep learning pipeline to enhance audio quality by reducing background noise.

Built a deep learning pipeline to enhance audio quality by reducing background noise.

## Getting Started

### Clone the Repository

To get started, clone the repository:

```bash
git clone https://github.com/ms0525/Audio-Noise-Cancellation.git
cd Audio-Noise-Cancellation
```

### Install Dependencies

Install all required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Running the Jupyter Notebook

The Noise_cancellation folder contains `Noise_cancellation.ipynb` an interactive Jupyter Notebook.\
To run it:

1. Upload the notebook to Google Colab.
2. Open and run the notebook.

## Running the Streamlit App

The Code folder contains `app.py` file that runs the Streamlit-based user interface. To launch the app:

```bash
streamlit run Code/app.py
```

### Features:

- Upload a pre-recorded audio file or record one using your device.
- Apply noise cancellation to enhance the audio quality.

## 1 - Introduction
A model can be trained for noise reduction using convolution neuratl network (CNN). Most of the applications of CNN are image related, such as image segmentation and image classification. But can we use CNN to reduce background noise from an audio? The answer is yes, We can get visual representation of audios by ploting their spectrograms. Let's get started.

## 2 - Preparing Dataset
We will prepare noisy audio dataset by combining 2 different datasets.
    
1. LibriSpeech (clean speech)
2. ECS-50 (sounds from different environments)

## 3 - Preprocessing Audio Files
Following are the steps to preprocess audios:

1. Split audios of frame length which is 8064
2. Convert the list of frames into 2D array or in matrix form
3. Convert audio frames to spectrograms using STFT
4. Convert spectrograms to matrix spectrograms
5. Normalize the matrix values between -1 and 1 for tanh activation function


### 3.1 - Splitting Audios into Frames
We will be splitting audios into small frames of same length. It is important as Convolutional Neural Network (CNN) requies contsistant shape and size of inputs. We have frames of 8064 length and the hop length will be of 8064. The hop length and frame length are equal which means no overlapping. The function split_Audios() is used to split the audios.

### 3.2 - Create Spectrograms
We will be making spectrograms of all the frames. We will use STFT function from librosa library. We will also create a function to plot these spectrograms. We will use n_fft of 255 size and hop_size of 63. The fuction is defined as spectrogram(). It returns magniture and phase of the audio. We also converted mangnitude to DB as we need to represent spectrograms in matrix form.

### 3.3 - Normalizing the Spectrogram
Normalizing or scalling is very important for the training of models. We used mix,max normalization and we normalized it in range [-1,1].

# 4 - Training CNN Model
U-Net is a deep learning model. It has a U-shaped structure with three parts: an encoder, a bottleneck, and a decoder. The encoder extracts important features from the input using convolutional layers and reduces the size with max pooling. The bottleneck acts as a bridge, capturing deeper features. The decoder then reconstructs the output using transposed convolutions, with skip connections helping to retain details lost during encoding. This structure makes U-Net great for processing spectrograms, as it learns both fine and broad patterns in speech while preserving important details. Its ability to work well with limited data makes it a strong choice for noise reduction in audio processing.

# 5 - Results
To check the results you can play the noisy and cleaned audio in the Audios directory. We trained our model with limited resourses and the dataset was really small and we trained it for only 20 epochs, but still the model performed very well. We can get better results by increasing the dataset size and trained it on super computer.
