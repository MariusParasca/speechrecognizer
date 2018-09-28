from Features import Features
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import soundfile as sf
import os

class Preprocesor:
    PROPRIETY_KEYS = ['labels', 'features_dimensions']

    @staticmethod
    def create_structure(path):
        if(os.path.exists(path) == 0):
            os.makedirs(path)

    @staticmethod
    def read_and_save_features(path_to_audio_dataset, path_to_save_features):
        Preprocesor.create_structure(path = path_to_save_features)
        labels = np.sort(os.listdir(path_to_audio_dataset))
        for dir_filename in labels:
            path_to_word_dataset = path_to_audio_dataset + os.sep + dir_filename
            if(os.path.isdir(path_to_word_dataset) == 0):
                continue
            features_pack = []
            for file in tqdm(os.listdir(path_to_word_dataset)):
                input, rate = sf.read(file = path_to_word_dataset + os.sep + file)
                features = Features(input = input, rate = rate)
                features_to_save = features.wav_to_features()
                features_pack.append(features_to_save)
            np.save(file = path_to_save_features + os.sep + dir_filename, arr = features_pack)

    @staticmethod
    def get_labels(path_to_audio_dataset):
        return np.sort(os.listdir(path_to_audio_dataset))

    @staticmethod
    def save_proprieties(path_to_audio_dataset, path_to_features, path_to_save = '.\\proprieties'):
        result = {}
        labels = Preprocesor.get_labels(path_to_audio_dataset)
        result[Preprocesor.PROPRIETY_KEYS[0]] = labels
        features_dimensions = []
        files = os.listdir(path_to_features)
        word_features = np.load(file = path_to_features + os.sep + files[0])
        features_dimensions.append(word_features.shape[1])
        features_dimensions.append(word_features.shape[2])
        result[Preprocesor.PROPRIETY_KEYS[1]] = features_dimensions
        np.save(file = path_to_save, arr = result)
        print('Proprieties saved!')
        
    @staticmethod
    def load_proprieties(path = '.\\proprieties.npy'):
        return np.load(path).item()
    
    @staticmethod
    def get_data_for_training(path_to_features, proprieties, split_ratio = 0.7, random_state = 40):
        features_X = np.load(file = path_to_features + os.sep + 
                             proprieties[Preprocesor.PROPRIETY_KEYS[0]][0] + '.npy')
        labels_y = np.zeros(features_X.shape[0])

        for i, file in enumerate(tqdm(proprieties[Preprocesor.PROPRIETY_KEYS[0]][1:])):
            word_features_x = np.load(path_to_features + os.sep + 
                                      proprieties[Preprocesor.PROPRIETY_KEYS[0]][i + 1] + '.npy')
            features_X = np.vstack((features_X, word_features_x))
            labels_y = np.append(labels_y, np.full(word_features_x.shape[0],
                                                   fill_value = i + 1))
            X_train, X_test, y_train, y_test = train_test_split(features_X, labels_y, test_size = (1 - split_ratio), 
                                                             random_state = random_state, shuffle = True)
        return (X_train, X_test, y_train, y_test)
        
        
        
        
        
        
        
        
        
        
        
        
        