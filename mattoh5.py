from frontend import Videofrontend
from scipy.io import loadmat
import tables
import numpy as np, h5py

class Convmattohdf5(object):

    def __init__(self):
        self.video = Videofrontend()
        print("Convert mat to hdf5")

    def create_hdf5file(self, hdf5train, dataname, dim):
        """
        Create an empty hdf5 files to store features
        """
        data = np.empty([0, dim])
        f = tables.open_file(hdf5train, mode='w')
        filters = tables.Filters(complevel=5, complib='blosc')
        data_storage = f.createEArray(f.root, dataname,
                                              tables.Atom.from_dtype(data.dtype),
                                              shape=(0, data.shape[-1]),
                                              filters=filters,
                                              expectedrows=len(data))

        for n, (d) in enumerate(zip(data)):
            data_storage.append(data[n][None])
        f.close()

    def convert_wtables(self, matCVD, matLIVE1, matLIVE2, hdf5train, n_train, hdf5valid, n_valid, dim):
        """
        Load features from pickle files in list.dat and store them in a hdf5 file
        This is meant for large dataset that cannot fit memory
        """
        self.create_hdf5file(hdf5train, 'train', dim)
        print('\nReading mat files')
        for i in range(1,n_train):
            sample = h5py.File("%s/Sample_%d.mat"%(matCVD, i), 'r')
            datacvd = sample["Sample_%d"%(i)][()].T

            #sample = h5py.File("%s/Sample_%d.mat"%(matLIVE1, i), 'r') - Not working for unknown reason
            sample = loadmat("%s/Sample_%d.mat"%(matLIVE1, i))
            datalive1 = sample["Sample_%d"%(i)]

            #sample = h5py.File("%s/Sample_%d.mat"%(matLIVE2, i), 'r')
            sample = loadmat("%s/Sample_%d.mat"%(matLIVE2, i))
            datalive2 = sample["Sample_%d"%(i)]

            data = np.concatenate((datacvd, datalive1), axis=0)
            data = np.concatenate((data, datalive2), axis=0)

            print("\nAppending Sample_%d.mat content into %s"%(i, hdf5train))
            f = tables.open_file(hdf5train, mode='a')
            data_storage = f.root.train
            for n, (d) in enumerate(zip(data)):
                data_storage.append(data[n][None])
            f.close()


        self.create_hdf5file(hdf5valid, 'valid', dim)
        print('\nReading mat files')
        for i in range(n_train+1,n_train+ n_valid):
            sample = h5py.File("%s/Sample_%d.mat" % (matCVD, i), 'r')
            datacvd = sample["Sample_%d" % (i)][()].T

            # sample = h5py.File("%s/Sample_%d.mat"%(matLIVE1, i), 'r') - Not working for unknown reason
            sample = loadmat("%s/Sample_%d.mat" % (matLIVE1, i))
            datalive1 = sample["Sample_%d" % (i)]

            # sample = h5py.File("%s/Sample_%d.mat"%(matLIVE2, i), 'r')
            sample = loadmat("%s/Sample_%d.mat" % (matLIVE2, i))
            datalive2 = sample["Sample_%d" % (i)]

            data = np.concatenate((datacvd, datalive1), axis=0)
            data = np.concatenate((data, datalive2), axis=0)

            print("\nAppending Sample_%d.mat content into %s" % (i, hdf5train))
            f = tables.open_file(hdf5train, mode='a')
            data_storage = f.root.valid
            for n, (d) in enumerate(zip(data)):
                data_storage.append(data[n][None])
            f.close()

# Mint desktop
#matCVD = "/media/andersonavila/muse02/anderson/Video/CVD"
#matLIVE1 = "/media/andersonavila/muse02/anderson/Video/LIVE1"
#matLIVE2 = "/media/andersonavila/muse02/anderson/Video/LIVE2"
#hdf5file = "/media/andersonavila/muse02/anderson/Video/vqoe.hdf5"
# muse04
matCVD = "/home/zahidakhtar/muse02/anderson/Video/CVD"
matLIVE1 = "/home/zahidakhtar/muse02/anderson/Video/LIVE1"
matLIVE2 = "/home/zahidakhtar/muse02/anderson/Video/LIVE2"
hdf5train = "../vfeatures/vqoe_train.hdf5"
hdf5valid = "../vfeatures/vqoe_valid.hdf5"
video = Convmattohdf5()
video.convert_wtables(matCVD, matLIVE1, matLIVE2, hdf5train, 1750, hdf5valid, 750, 25344)