#
#
#  Please add livsvm directory to the path variable for this code to work
import os, sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Tools/libsvm-3.21/python'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

from svmutil import *

print("test...")
#svm_model.predict = lambda self, x: svm_predict([0], [x], self)[0][0]

#prob = svm_problem([1,-1], [[1,0,1], [-1,0,-1]])

#param = svm_parameter()
#param.kernel_type = LINEAR
#param.C = 10

#m=svm_train(prob, param)

#m.predict([1,1,1])