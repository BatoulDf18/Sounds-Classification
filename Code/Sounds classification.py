# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 12:19:19 2023

@author: SMZ
"""

import os
import librosa
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.preprocessing import LabelEncoder

# Fonction pour charger les fichiers audio
def load_audio_files(folder_path):
    audio_data = []
    labels = []
    
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        
        if os.path.isdir(class_path):
            class_audio_data = []
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                y, sr = librosa.load(file_path, sr=None)
                class_audio_data.append(y)
                labels.append(class_folder)
            audio_data.append(class_audio_data)
    
    return audio_data, labels

# Fonction pour normaliser le volume d'une séquence audio
def normalize_volume(audio_sequence, target_db=-20.0):
    rms = np.sqrt(np.mean(audio_sequence**2))
    target_rms = 10**(target_db / 20.0)
    normalization_factor = target_rms / rms
    normalized_audio = audio_sequence * normalization_factor
    return normalized_audio

# Fonction pour appliquer un filtre passe-bas à une séquence audio
def lowpass_filter(audio_sequence, cutoff_frequency, sampling_rate):
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_frequency / nyquist
    b, a = butter(1, normalized_cutoff, btype='low', analog=False)
    filtered_audio = lfilter(b, a, audio_sequence)
    return filtered_audio

# Fonction pour compléter ou tronquer une séquence à une longueur spécifiée
def pad_truncate_sequence(sequence, max_length):
    if len(sequence) < max_length:
        sequence = np.pad(sequence, (0, max_length - len(sequence)))
    else:
        sequence = sequence[:max_length]
    return sequence

# Fonction pour prétraiter les données audio (normaliser le volume, gérer les variations de longueur, filtrer)
def preprocess_audio_data(audio_data, max_length=44100, target_db=-20.0, cutoff_frequency=None):
    preprocessed_data = []
    for class_audio_data in audio_data:
        normalized_class_data = [normalize_volume(audio, target_db) for audio in class_audio_data]
        
        if cutoff_frequency is not None:
            filtered_class_data = [lowpass_filter(audio, cutoff_frequency, sr) for audio, sr in zip(normalized_class_data, [44100]*len(normalized_class_data))]
        else:
            filtered_class_data = normalized_class_data
        
        preprocessed_class_data = [pad_truncate_sequence(audio, max_length) for audio in filtered_class_data]
        preprocessed_data.append(preprocessed_class_data)
    return preprocessed_data

# Fonction pour extraire des caractéristiques
def extract_features(preprocessed_data, sr):
    features = []
    for class_audio_data in preprocessed_data:
        class_features = []
        for audio in class_audio_data:
            # Extraction des MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            # Extraction de la caractéristique Chroma
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            # Extraction de la caractéristique Mel spectrogram
            mel = librosa.feature.melspectrogram(y=audio, sr=sr)
            # Extraction de la caractéristique Spectral Contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            # Extraction de la caractéristique Tonnetz
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)

            # Concaténation de toutes les caractéristiques en un seul vecteur
            combined_features = np.concatenate([mfccs, chroma, mel, contrast, tonnetz])

            class_features.append(combined_features)
        features.append(class_features)
    return features

# Exemple d'utilisation avec un filtre passe-bas à 1000 Hz
data_folder = r'C:\Users\SMZ\Desktop\sounds'
audio_data, labels = load_audio_files(data_folder)

# Affichage des classes existantes
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
print("Classes existantes : ", label_encoder.classes_)

# Exemple de prétraitement avant l'extraction des caractéristiques avec normalisation du volume et filtrage
preprocessed_data = preprocess_audio_data(audio_data, cutoff_frequency=1000)

# Extraction des caractéristiques
features = extract_features(preprocessed_data, sr=44100)






#devision data
from sklearn.model_selection import train_test_split
# Flatten the nested list structure for features
flattened_features = [item for sublist in features for item in sublist]

# Verify the dimensions
print("Dimensions of flattened features:", len(flattened_features), len(flattened_features[0]))

# Now, flattened_features is a list containing all instances across all classes.

# Assuming that encoded_labels is a list with 430 labels corresponding to the instances,
# you can use it directly in your machine learning model.
# Flatten the nested list structure for features
flattened_features = [item for sublist in features for item in sublist]






#model RNN
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Assuming flattened_features is a list of sequences
# Each sequence is represented as a list of feature vectors

# Convert flattened_features to a numpy array
X = np.array(flattened_features)

# Reshape X to have 2 dimensions
# The first dimension is the number of sequences, and the second dimension is the number of features
num_sequences, num_features = X.shape[0], X.shape[1] * X.shape[2]
X = X.reshape((num_sequences, num_features))

# Now, X has two dimensions and can be used with MLPClassifier
X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

# Create the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)

# Train the Random Forest classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate the Random Forest classifier
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf)

# Print the results for the Random Forest classifier
print(f'Random Forest Accuracy: {accuracy_rf}')
print(f'Confusion Matrix:\n{conf_matrix_rf}')
print(f'Classification Report:\n{classification_rep_rf}')



