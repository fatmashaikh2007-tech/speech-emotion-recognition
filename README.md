# speech-emotion-recognition
A Machine Learning project that identifies human emotions from speech audio using audio feature extraction and classification techniques.

# 🎙️ Speech Emotion Recognition Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)
![Dataset](https://img.shields.io/badge/Dataset-RAVDESS-green)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow?logo=googlecolab)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

Speech Emotion Recognition (SER) is an Artificial Intelligence application that identifies human emotions from speech audio recordings.

This project implements a Deep Learning-based Speech Emotion Recognition system using the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

The system processes audio recordings, extracts Mel-Frequency Cepstral Coefficients (MFCC) as audio features, and uses a Neural Network to classify speech into eight emotional categories.

The project demonstrates how audio signal processing and Deep Learning can be combined to recognize emotional expressions in human speech.

## 🎯 Objectives

- Understand the fundamentals of Speech Emotion Recognition.
- Work with an emotional speech audio dataset.
- Load and process audio recordings using Python.
- Extract MFCC features from speech signals.
- Prepare audio features for Deep Learning.
- Develop and train a Neural Network for emotion classification.
- Evaluate the model using test accuracy, classification report, and confusion matrix.

## 📂 Dataset

**Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

**Dataset Source:** RAVDESS dataset available through Kaggle.

**Total Audio Recordings Used:** 1,440

The RAVDESS dataset contains recorded speech and song samples representing different emotional expressions.

This project uses the speech recordings to train an emotion classification model.

### Emotions Recognized

The model classifies audio recordings into the following eight emotional categories:

| Label | Emotion |
|---|---|
| 01 | Neutral |
| 02 | Calm |
| 03 | Happy |
| 04 | Sad |
| 05 | Angry |
| 06 | Fearful |
| 07 | Disgust |
| 08 | Surprised |

## 🛠️ Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Google Colab | Development environment |
| Librosa | Audio loading and feature extraction |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Matplotlib | Data visualization |
| Seaborn | Confusion matrix visualization |
| Scikit-learn | Data preprocessing and evaluation |
| TensorFlow | Deep Learning framework |
| Keras | Neural Network implementation |
| SoundFile | Audio file support |

## ⚙️ Methodology

### 1. Dataset Loading

The RAVDESS audio dataset is loaded from Google Drive into the Google Colab environment.

The project processes the audio files stored in the dataset directory.

### 2. Audio Loading and Visualization

The Librosa library is used to load audio recordings while preserving their original sample rate.

The project also:
- Plays an audio sample.
- Displays its speech waveform.
- Examines the audio signal.

### 3. MFCC Feature Extraction

Mel-Frequency Cepstral Coefficients (MFCC) are extracted from each audio recording using Librosa.

MFCC features represent important characteristics of speech signals and are commonly used in audio processing applications.

The project extracts 40 MFCC coefficients from each audio recording.

The mean of the MFCC values across time is calculated to obtain a fixed-length feature vector containing 40 values per recording.

### 4. Dataset Preparation

The extracted audio features are converted into NumPy arrays.

The emotion labels are converted into numerical values using Scikit-learn's LabelEncoder.

The resulting dataset contains:

- 1,440 audio samples.
- 40 features per sample.
- 1,440 corresponding emotion labels.

### 5. Train-Test Splitting

The dataset is divided into training and testing sets using an 80:20 split.

| Dataset | Samples |
|---|---:|
| Training | 1,152 |
| Testing | 288 |
| Total | 1,440 |

The random state is set to 42 for reproducibility.

### 6. Data Normalization

StandardScaler is applied to normalize the extracted MFCC features.

The scaler is fitted on the training data and then used to transform both the training and testing data.

### 7. Neural Network Architecture

A Sequential Neural Network is developed using TensorFlow and Keras.

The architecture consists of:

| Layer | Configuration |
|---|---|
| Input | 40 MFCC features |
| Dense Layer 1 | 128 neurons, ReLU |
| Dropout | 0.3 |
| Dense Layer 2 | 64 neurons, ReLU |
| Dropout | 0.3 |
| Output Layer | 8 neurons, Softmax |

The model uses:
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Evaluation Metric:** Accuracy

The Softmax output layer predicts the probabilities for the eight emotion categories.

### 8. Model Training

The model is trained using the following configuration:

- Epochs: 30
- Batch Size: 32
- Validation Split: 20%

Training and validation accuracy are monitored during the training process.

### 9. Model Evaluation

The trained model is evaluated using the testing dataset.

The project generates:
- Test accuracy
- Classification report
- Precision
- Recall
- F1-score
- Confusion matrix
- Training and validation accuracy graph

### 10. Model Saving

The trained model is saved in Keras format as:

`emotion_recognition_model.keras`

## 🧠 System Architecture

```text
       RAVDESS Audio Dataset
                 |
                 v
          Audio Loading
                 |
                 v
        Audio Visualization
                 |
                 v
       MFCC Feature Extraction
           (40 Features)
                 |
                 v
        Mean Feature Calculation
                 |
                 v
          Dataset Preparation
                 |
                 v
          Label Encoding
                 |
                 v
          Train-Test Split
                 |
                 v
          Data Normalization
                 |
                 v
        Dense Neural Network
                 |
                 v
           Model Training
                 |
                 v
          Model Evaluation
                 |
                 v
        Emotion Classification
                 |
                 v
          Saved Keras Model
```

## 📊 Results and Performance

The model was evaluated using 288 testing samples.

### Accuracy Results

| Evaluation | Accuracy |
|---|---:|
| Training Accuracy | Approximately 68% |
| Validation Accuracy | Approximately 61% |
| Test Accuracy | 55.21% |

The test accuracy was obtained directly from the model evaluation output.

### Classification Report

The following results were obtained from the test dataset:

| Emotion | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Angry | 0.77 | 0.73 | 0.75 |
| Calm | 0.61 | 0.68 | 0.64 |
| Disgust | 0.63 | 0.71 | 0.67 |
| Fearful | 0.50 | 0.47 | 0.49 |
| Happy | 0.42 | 0.39 | 0.41 |
| Neutral | 0.45 | 0.28 | 0.34 |
| Sad | 0.44 | 0.50 | 0.47 |
| Surprised | 0.50 | 0.47 | 0.48 |

The model achieved an overall test accuracy of approximately 55.21%.

## 💻 Project Notebook

The complete project implementation is available in Google Colab.

**[Open Speech Emotion Recognition Notebook](https://colab.research.google.com/drive/1bW58UY0F0oQZBmxOD5x_eQaPZzfND1sS)**

The notebook includes:
- Audio loading and playback
- Speech waveform visualization
- MFCC feature extraction
- Dataset preparation
- Label encoding
- Train-test splitting
- Feature normalization
- Neural Network development
- Model training
- Performance evaluation
- Confusion matrix
- Accuracy visualization
- Model saving

## 🚀 How to Run the Project

1. Open the Google Colab notebook using the link above.
2. Sign in to your Google account.
3. Make a copy of the notebook in your Google Drive if necessary.
4. Upload or connect the RAVDESS dataset to Google Drive.
5. Set the dataset directory path according to the notebook.
6. Install the required audio libraries.
7. Execute the notebook cells sequentially.
8. Observe the training, evaluation, and classification outputs.

### Required Libraries

```bash
pip install librosa soundfile numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## 🔮 Future Enhancements

- Improve classification accuracy through feature engineering.
- Experiment with CNN, LSTM, and other advanced architectures.
- Explore additional audio features such as Chroma and Mel Spectrograms.
- Apply hyperparameter tuning to improve model performance.
- Develop a real-time speech emotion recognition system.
- Integrate the trained model into a web application.
- Explore emotion recognition across different languages and speaking styles.

## 🏁 Conclusion

The Speech Emotion Recognition project demonstrates the application of Deep Learning and audio signal processing in identifying human emotions from speech recordings.

Using the RAVDESS dataset, 40 MFCC features were extracted from 1,440 audio recordings and used to train a Dense Neural Network with eight output classes.

The model achieved a test accuracy of approximately 55.21%. The project provided practical experience in audio preprocessing, MFCC feature extraction, data normalization, Neural Network development, model training, and performance evaluation.

This work highlights the potential of speech emotion recognition in intelligent virtual assistants, human-computer interaction, and emotion-aware applications.

## 👩‍💻 Author

**Fatma Sameer Shaikh**

Computer Science and Engineering  
Artificial Intelligence and Machine Learning

St. Joseph's Institute of Technology

**GitHub:** [fatmashaikh2007-tech](https://github.com/fatmashaikh2007-tech)

---

⭐ If you find this project interesting, feel free to explore the repository and the implementation.
