from frontend import Audiofrontend
import h5py
import tables
import numpy as np

class Convpickletohdf5(object):

    def __init__(self):
        self.audio = Audiofrontend()
        print("Convert pickle to hdf5")

    def convert(self, picklelist = "../afeatures/list.dat", hdf5file = "../afeatures/aqoe.hdf5"):
        """
        Load features from pickle files in list.dat and store them in a hdf5 file
        This is meant for small dataset that can fit memory
        """
        data = self.audio.load_feat_audio(picklelist)
        f = h5py.File(hdf5file, "w")
        f.create_dataset("init", data=abs(data))
        f.close()

    def create_hdf5file(self, hdf5file, dim):
        """
        Create an empty hdf5 files to store features
        """
        data = np.empty([0, dim])
        f = tables.open_file(hdf5file, mode='w')
        filters = tables.Filters(complevel=5, complib='blosc')
        data_storage = f.createEArray(f.root, 'data',
                                              tables.Atom.from_dtype(data.dtype),
                                              shape=(0, data.shape[-1]),
                                              filters=filters,
                                              expectedrows=len(data))
        for n, (d) in enumerate(zip(data)):
            data_storage.append(data[n][None])
        f.close()

    def convert_wtables(self, picklelist = "../afeatures/list.dat", hdf5file = "../afeatures/aqoe.hdf5", dim = 257):
        """
        Load features from pickle files in list.dat and store them in a hdf5 file
        This is meant for large dataset that cannot fit memory
        """
        create_hdf5file(hdf5file, dim)
        flist = open(picklelist, 'r')
        line = flist.readline().rstrip('\n')
        print('\nReading file %s' % (picklelist))
        while (line != ""):
            print('Processing files in %s' % (line))
            data = abs(self.audio.get_feat(line.rstrip('\n').strip()))
            f = tables.open_file(hdf5file, mode='a')
            data_storage = f.root.data
            for n, (d) in enumerate(zip(data)):
                data_storage.append(data[n][None])
            f.close()
            line = flist.readline()
        flist.close()