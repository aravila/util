import os, sys
import stft
import scipy.io.wavfile as wav
from scipy import signal
from subprocess import call
import numpy as np
import pickle
import scipy.io
import srmrpy as srmr
from scipy import io, polyval, polyfit, sqrt, stats
import tables
import librosa
#import librosa.display as disp

class Audiofrontend(object):

    def __init__(self):
        print("Audio frontend utilities")

    def get_feat(self, filepath = '.'):
        """
        Returns the data from a pickle file
        """
        with open(filepath, 'rb') as handle:
            f_open = pickle.load(handle)
        data = f_open['data']
        return data

    def get_filepaths(self, source, dest):
        """
        Create file listing all directories and subdirectories in source
        List of directories will be placed on the filepath defined in dest
        """
        f = open(dest, 'w+')
        for root, directories, files in os.walk(source):
          for filename in directories:
              filepath = os.path.join(root, filename)
              f.write('%s%s' % (filepath, '\n'))
        f.closed

    def load_feat_audio(self, filepath):
        """
        Read a file containing a list of pickle file paths and return audio features in each pickle file
        """
        data = np.empty([0, 257])
        f = open(filepath, 'r')
        line = f.readline().rstrip('\n')
        print('\nReading file %s' % (filepath))
        while (line != ""):
            data = np.concatenate((data, self.get_feat((line.rstrip('\n')))), axis=0)
            line = f.readline()
        f.closed
        return data

    def feat_ext_stft(self, folders=['.'], olap=2, nlength=512, FS=16000):
        """
        Receive a list of folders and look for .wav file to extract STFT
        All audio files are resampled to 16 kHz if FS is not specified
        """
        # Create a empty matrix to store 257-dimensional features
        data = np.empty([0, 257])
        for d in range(0, len(folders)):
            dirs = os.listdir(folders[d].rstrip('\n'))
            for file in dirs:
                if file.endswith('.wav'):
                    data = np.concatenate((data, self.stft_audio2(folders[d].rstrip('\n'), file, olap, nlength, FS)), axis=0)
        return data

    def create_hdf5file(self, hdf5file, dataname, dim):
        """
        Create an empty hdf5 files to store features
        """
        data = np.empty([0, dim])
        f = tables.open_file(hdf5file, mode='w')
        filters = tables.Filters(complevel=5, complib='blosc')
        data_storage = f.create_earray(f.root, dataname,
                                              tables.Atom.from_dtype(data.dtype),
                                              shape=(0, data.shape[-1]),
                                              filters=filters,
                                              expectedrows=len(data))
        for n, (d) in enumerate(zip(data)):
            data_storage.append(data[n][None])
        f.close()

    def extract_feature(self, stftmode = 1, wavpathlist = "../afeatures/data.dat", hdf5file = "../afeatures/srmr_train.hdf5", new = 1, dataname = "train"):
        """
        Extract modulation features from .wav in folders in list.dat and store them in a hdf5 file
        This is meant for large dataset that cannot fit memory
        """
        flist = open(wavpathlist, 'r')
        line = flist.readline().rstrip('\n')
        print('\nReading file %s' % (wavpathlist))
        nline = 0
        while (line != ""):
            nline += 1
            print('Processing files in %s' % (line))
            dirs = os.listdir(line.rstrip('\n'))
            data = self.extract_fea_mode(stftmode, line.rstrip('\n'), dirs)
            if new == 1 and nline == 1:
                self.create_hdf5file(hdf5file, dataname, data.shape[1])

            f = tables.open_file(hdf5file, mode='a')
            if dataname == "train":
                data_storage = f.root.train
            else:
                data_storage = f.root.valid
            for n, (d) in enumerate(zip(data)):
                data_storage.append(data[n][None])
            f.close()
            line = flist.readline()
        flist.close()


    def extract_fea_mode(self, stftmode, folder, dirs, FS=16000):
        """
        Look for .wav file to extract Modulation Features
        All audio files are resampled to 16 kHz if FS is not specified
        """
        data = np.empty(0)
        for file in dirs:
            if file.endswith('.wav'):
                if stftmode == 0:
                    mf = self.srmr_audio(folder, file, FS)
                    mf = np.einsum('ijk->kij', mf)
                    mf = np.reshape(mf, (mf.shape[0], mf.shape[1] * mf.shape[2]))
                    if data.shape[0] == 0:
                        data = mf
                    data = np.concatenate((data, mf), axis=0)
                else:
                    stft = self.stft_audio(folder, file, FS)
                    if data.shape[0] == 0:
                        data = stft
                    data = np.concatenate((data, stft), axis=0)
        return data


    def stft_audio(self, path, file, nlength = 1024, olap = 2, librosamode = 1, FS=16000):
        """
        http://stft.readthedocs.io/en/latest/index.html
        Receive specific folder and file to extract the STFT
        All audio are resample to 16 kHz if FS is not specified
        """
        fs, s = wav.read('%s/%s' % (path, file))
        dim = len(s.shape)
        if (dim>1):
            s = s[:, 0]
        if (fs != FS):
            n_s = round(len(s) * (FS / fs))
            s = signal.resample(s, n_s)
        if librosamode == 0:
            specgram = stft.spectrogram(s, framelength=nlength, overlap=olap)
        else:
            S = librosa.stft(s, n_fft=nlength, hop_length= olap)
            specgram = librosa.logamplitude(np.abs(S))
        return specgram.T

    def srmr_audio(self, path, file, FS=16000):
        """
        http://stft.readthedocs.io/en/latest/index.html
        Receive specific folder and file to extract Modulation Features
        All audio are resample to 16 kHz if FS is not specified
        """
        fs, s = wav.read('%s/%s' % (path, file))
        dim = len(s.shape)
        if (dim>1):
            s = s[:, 0]
        if (fs != FS):
            n_s = round(len(s) * (FS / fs))
            s = signal.resample(s, n_s)
        ratio, energy = srmr.srmr(s, FS, n_cochlear_filters=23, low_freq=125, min_cf=4, max_cf=128, fast=True, norm=False)

        return energy

    def gen_audio_db(self, sourcepath, destpath):
        """
        Receive a folder 'sourcepath' and look for subfolders inside it

        These subfolders should contain .dat files with a list
        of directories containing .wav data

        For each subfolder is created a pickle file with the name of the subfolder
        The pickle file will have the features extracted from .wav files found in
        the directories listed in the .dat file
        """
        datapath = []
        audio_dict = {}
        folders = [x[0] for x in os.walk(sourcepath)]
        for d in range(1, len(folders)):
            dirs = os.listdir(folders[d])
            for file in dirs:
                if file.endswith('.dat'):
                    filename = '%s/%s'%(folders[d],file)
                    basename = os.path.splitext(file)[0]
                    f = open(filename, 'r')
                    line = f.readline().rstrip('\n')
                    print('\nReading file %s'%(basename))
                    while(line != ""):
                        datapath.append(line)
                        line = f.readline()
                    f.closed
                    features = feat_ext_audio(datapath)
                    if len(features) > 1:
                        audio_dict[basename] = features
                    else:
                        print('No .wav file found in %s'%(datapath))
                    datapath = []
            destfile = os.path.basename(os.path.normpath(folders[d]))
            if len(audio_dict) > 0:
                print("Number of folders in pickle file: %d"%(len(audio_dict)))
                with open('%s/%s.pickle'%(destpath,destfile), 'wb') as handle:
                    pickle.dump(audio_dict, handle)

    def resaple_audio(self, folders, FS = 16000):
        """
        This is an alternative to resample .wav files and save them in the filesystem
        """
        for d in range(0, len(folders)):
            rspldir = '%s/%s'%(folders[d],'/rsplwav')
            if not os.path.exists(rspldir):
                os.makedirs(rspldir)
            dirs = os.listdir(folders[d])
            for file in dirs:
                if file.endswith('.wav'):
                    filepath = '%s/%s' %(folders[d],file)
                    destpath = '%s/%s' %(rspldir,file)
                    s = ['sox', '-S', filepath, '-r', str(FS), destpath]
                    call(s)

class Videofrontend(object):

    def __init__(self):
        print("Video frontend utilities")

    def load_feat_video(self, folders = ['./vfeatures']):
        """
        Receive a list of folders and look for .mat file to load extracted video features
        """
        data = np.empty([0, 25344]) # Features dimension for video... needs to be confirmed with Zahid!!!
        for d in range(0, len(folders)):
            dirs = os.listdir(folders[d].rstrip('\n'))
            for file in dirs:
                if file.endswith('.mat'):
                    mat = scipy.io.loadmat('%s/%s'%(folders[d].rstrip('\n'),file))
                    ret = mat[file[:-4]]
                    data = np.concatenate((data, ret), axis=0)
        return data


    def feat_ext_video(self, folders='.', hcrop=144, vcrop=176, overlap=2):
        """
        Receive a list of folders and look for .avi file to extract image frames
        All frames is cropped using a hcrop x vcrop windows size
        """
        data = []
        for d in range(0, len(folders)):
            dirs = os.listdir(folders[d].rstrip('\n'))
            for file in dirs:
                if file.endswith('.avi'):
                    data = data + cropslide_video(folders[d].rstrip('\n'), file, hcrop, vcrop, overlap)
        return data


    def crop_video(self, path, file, hcrop, vcrop):
        """
        Crop each frame using a hcrop x vcrop windows size
        No slide is implemented
        """
        ini = 1
        cropped_frames = []

        cap = cv2.VideoCapture('%s/%s' % (path, file))
        no_frames, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
            cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                for fr in range(1, int(no_frames)):
                    crop_img = frame[ini:(ini + hcrop), ini:(ini + hcrop)]
                    cropped_frames.append(crop_img)

        cap.release()
        cv2.destroyAllWindows()
        return cropped_frames


    def cropslide_video(self, path, file, hcrop, vcrop, overlap=2):
        """
        Crop each frame using a hcrop x vcrop windows size
        Slide is defined by overlap
        """
        v_ini = 1
        h_ini = 1
        cropped_frames = []
        hcrop_step = int(hcrop / overlap)  # Horizontal step - percentage of overlap is given by 1/overlap
        vcrop_step = int(vcrop / overlap)  # Vertical step - percentage of overlap is given by 1/overlap

        cap = cv2.VideoCapture('%s/%s' % (path, file))
        no_frames, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
            cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:                                  # If the video is ok, start processing frames
                for fr in range(1, int(no_frames)):
                    h_step = 0                               # First step on the horizontal axis
                    while (h_ini + hcrop < width):           # If windows goes over the horizontal edge stop
                        v_step = 0                           # First step on the verical axis
                        while (v_ini + vcrop < height):      # If window is over the vertical edge stop
                            v_ini = vcrop_step * v_step + 1  # Incremente initial vertical position by vcrop_step
                            crop_img = frame[h_ini:(h_ini + hcrop), v_ini:(v_ini + vcrop)]  # hcrop x vcrop window size
                            cropped_frames.append(crop_img)  # store window
                            v_step = v_step + 1              # Slide on the vertical axis one step
                        crop_img = frame[h_ini:(h_ini + hcrop), height - vcrop:height]  # Crop vertical axis edge
                        cropped_frames.append(crop_img)      # Store window
                        h_step = h_step + 1                  # Slide on the horizontal and start over vertical axis
                        h_ini = hcrop_step * h_step + 1      # Incremente initial horizontal position by hcrop_step
                    h_ini = width - hcrop + 1                # Prepare to crop the edge of the horizontal axis
                    v_step = 0                               # Start over the vertical axis (last roll)
                    while (v_ini + vcrop < height):
                        v_ini = vcrop_step * v_step + 1      # Incremente initial vertical position by vcrop_step
                        crop_img = frame[h_ini:(h_ini + hcrop), v_ini:(v_ini + vcrop)]  # Crop hcrop x vcrop window size
                        cropped_frames.append(crop_img)      # Store window
                        v_step = v_step + 1                  # Slide on the vertical axis one step

        cap.release()
        cv2.destroyAllWindows()
        return cropped_frames


    def play_video(self, path, file, hcrop, vcrop):
        """
        This method will crop and play an avi video
        """
        ini = 1
        cropped_frames = []
        cap = cv2.VideoCapture('%s/%s' % (path, file))
        no_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                for fr in (1, int(no_frames)):
                    crop_img = frame[ini:(ini + hcrop), ini:(ini + hcrop)]
                    cropped_frames.append(crop_img)
                    cv2.imshow('frame', crop_img)
                    if cv2.waitKey(18) & 0xFF == ord('q'):
                        break
        cap.release()
        cv2.destroyAllWindows()
        return cropped_frames


    def gen_video_db(sourcepath, destpath):
        """
        Receive a folder 'sourcepath' and look for subfolders inside it

        These subfolders should contain .dat files with a list
        of directories containing .avi data

        For each subfolder is created a pickle file with the name of the subfolder
        The pickle file will have the features extracted from .avi files found in
        the directories listed in the .dat file
        """
        datapath = []
        video_dict = {}
        folders = [x[0] for x in os.walk(sourcepath)]
        for d in range(1, len(folders)):
            dirs = os.listdir(folders[d])
            for file in dirs:
                if file.endswith('.dat'):
                    filename = '%s/%s' % (folders[d], file)
                    basename = os.path.splitext(file)[0]
                    f = open(filename, 'r')
                    line = f.readline().rstrip('\n')
                    print('Reading file %s' % (basename))
                    while (line != ""):
                        datapath.append(line)
                        line = f.readline()
                    features = feat_ext_video(datapath)
                    video_dict[basename] = features
                    datapath = []
            destfile = os.path.basename(os.path.normpath(folders[d]))
            with open('%s/%s.pickle' % (destpath, destfile), 'wb') as handle:
                pickle.dump(video_dict, handle)

class MFStats(object):

    def __init__(self, mf_param):
        print("Modulation stats")
        self.mf_param = mf_param

    def get_mf_fea_1(self):
        """
        # Energy distribution of speech along the modulation frequency
        """
        mf_mean = np.mean(self.mf_param, axis=1)
        return mf_mean

    def get_mf_fea_2(self):
        """
        Spectral flatness
        """
        mf_flat = scipy.stats.gmean(self.mf_param, axis=1)/np.mean(self.mf_param, axis=1)
        return mf_flat

    def get_mf_fea_3(self):
        """
        Spectral centroid
        """
        multiplier = np.arange(1, 24)
        mf_num = np.einsum('i,kij->kj', multiplier, self.mf_param)
        mf_denom = np.einsum('kij->kj', self.mf_param)
        mf_cent = mf_num / mf_denom
        return mf_cent

    def get_mf_fea_4(self):
        """
        Modulation spectral centroid
        """
        mf_spect = np.empty(shape=[self.mf_param.shape[0], 0])
        multiplier = np.arange(1, 9)  # 8 Modulation Features
        idx = np.array([[0, 1, 2, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16, 17], [18, 19, 20, 21, 22]])
        for i in range(0, 5):
            aux = self.mf_param[:, idx[i], 0:8]
            mf_num = np.einsum('kij->kj', aux)
            mf_num = np.einsum('j,kj->k', multiplier, mf_num)
            mf_denom = np.einsum('kij->k', aux)
            mf_spect = np.column_stack((mf_spect, mf_num / mf_denom))
        return mf_spect

    def get_mf_fea_5(self):
        """
        Linear regression and square error
        """
        nObs = self.mf_param.shape[0]
        mf_slope = np.empty(shape=[nObs, 0])
        mf_err = np.empty(shape=[nObs, 0])
        xaxis = np.arange(1, 9)  # 8 Modulation Features
        x = np.reshape(xaxis, (1, 8))
        x = np.repeat(x, nObs, axis=0).T
        idx = np.array([[0, 1, 2, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16, 17], [18, 19, 20, 21, 22]])
        for i in range(0, 5):
            aux = self.mf_param[:, idx[i], 0:8]
            mf_vlr = np.einsum('kij->kj', aux).T
            (ar, br) = polyfit(xaxis, mf_vlr, 1)
            xr = polyval([ar, br], x)
            # compute the mean square error
            err = sqrt(sum((xr - mf_vlr) ** 2) / xaxis.shape[0])
            mf_slope = np.column_stack((mf_slope, ar))
            mf_err = np.column_stack((mf_err, err))
        return np.concatenate((mf_slope, mf_err), axis=1)

    def get_stats(self):
        mfstats = self.get_mf_fea_1()
        mfstats = np.concatenate((mfstats, self.get_mf_fea_2()), axis=1)
        mfstats = np.concatenate((mfstats, self.get_mf_fea_3()), axis=1)
        mfstats = np.concatenate((mfstats, self.get_mf_fea_4()), axis=1)
        mfstats = np.concatenate((mfstats, self.get_mf_fea_5()), axis=1)
        return mfstats

    def moving_stats(self,modfea, w_size):
        fea1 = np.empty(shape=[0, modfea.shape[1]])
        fea2 = np.empty(shape=[0, modfea.shape[1]])
        fea3 = np.empty(shape=[0, modfea.shape[1]])
        fea4 = np.empty(shape=[0, modfea.shape[1]])

        extraframe = int(w_size/2)

        modtmp = np.reshape(modfea[0], (1, len(modfea[0])))
        modtmp = np.repeat(modtmp, extraframe, axis=0)
        modfea = np.vstack((modtmp, modfea))

        modtmp = np.reshape(modfea[len(modfea)-1], (1, len(modfea[len(modfea)-1])))
        modtmp = np.repeat(modtmp, extraframe+1, axis=0)
        modfea = np.vstack((modfea, modtmp))

        print("Extracting mean...")
        for i in range(0, len(modfea)):
            mf_mean = np.mean(modfea[0+i:10+i], axis=0)
            fea1 = np.concatenate((fea1, np.reshape(mf_mean, (1, len(mf_mean)))), axis=0)
        print("Extracting std...")
        for i in range(0, len(modfea)):
            mf_std = np.std(modfea[0+i:10+i], axis=0)
            fea2 = np.concatenate((fea2, np.reshape(mf_std, (1, len(mf_std)))), axis=0)
        print("Extracting skewness...")
        for i in range(0, len(modfea)):
            mf_skewness = stats.skew(modfea[0+i:10+i], axis=0)
            fea3 = np.concatenate((fea3, np.reshape(mf_skewness, (1, len(mf_skewness)))), axis=0)
        print("Extracting kurtosis...")
        for i in range(0, len(modfea)):
            mf_kurtosis = stats.kurtosis(modfea[0+i:10+i], axis=0)
            fea4 = np.concatenate((fea4, np.reshape(mf_kurtosis, (1, len(mf_kurtosis)))), axis=0)

        mfstats = np.empty(shape=[fea1.shape[0], 0])
        mfstats = np.concatenate((mfstats, fea1), axis=1)
        mfstats = np.concatenate((mfstats, fea2), axis=1)
        mfstats = np.concatenate((mfstats, fea3), axis=1)
        mfstats = np.concatenate((mfstats, fea4), axis=1)

        return mfstats

