import numpy as np
from scipy.io.wavfile import read
from sklearn import preprocessing
import python_speech_features as psf
import librosa
#silence, index = sr,audio = read('silence.wav')

def extract_features(audio,rate):
    """
    input: audio file with respective sample rate
    
    output: 40 features based on mfccs and their first order delta
    """
    audio, index = librosa.effects.trim(audio, top_db=12, frame_length=12)
    #audio = librosa.effects.remix(audio, intervals=librosa.effects.split(audio))
    #audio = audio - silence
    mfcc_feature = psf.mfcc(audio,rate,nfft = 1200, appendEnergy = True,numcep=20)  
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = psf.delta(mfcc_feature,1)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined
