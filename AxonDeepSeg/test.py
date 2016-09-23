from evaluation.visualization import visualize_results, visualize_learning
from apply_model import myelin, pipeline
from learning.data_construction import build_data
from mrf import learn_mrf
from learn_model import learn_model
import os

def test_prediction():


    #build_data('/Users/viherm/Desktop/CARS','/viherm/Desktop/Train', trainRatio = 0.80)

    model_path = '/Users/viherm/Desktop/AxonSegmentation/AxonDeepSeg/data/models/model_parameters2'
    model_restored_path = '/Users/viherm/Desktop/AxonSegmentation/AxonDeepSeg/data/models/model_parameters1'
    mrf_path = '/Users/viherm/Desktop/AxonSegmentation/AxonDeepSeg/data/models/model_parameters3'

    #visualize_learning(model_path, model_restored_path, start_visu=0)

    image_path = '/Users/viherm/Desktop/CARS/data%s'%6

    pipeline(image_path, model_path, mrf_path)
    visualize_results(image_path)


def test_learning():

    # current_path = os.path.dirname(os.path.abspath(__file__))
    # trainingset_path = current_path+'/data/trainingset'
    # model_path = current_path+'/data/models/model_parametlsers3'
    # model_restored_path = current_path+'/data/models/model_parameters2'
    # learn_model(trainingset_path, model_path=model_path, model_restored_path= model_restored_path)

    image_path = '/Users/viherm/Desktop/CARS/data6'
    model_path = '/Users/viherm/Desktop/AxonSegmentation/AxonDeepSeg/data/models/model_parameters3'
    model_restored_path = '/Users/viherm/Desktop/AxonSegmentation/AxonDeepSeg/data/models/model_parameters1'
    mrf_path = '/Users/viherm/Desktop/AxonSegmentation/AxonDeepSeg/data/models/model_parameters3'

    images_path_mrf = ['/Users/viherm/Desktop/CARS/data2', '/Users/viherm/Desktop/CARS/data5']
    learn_mrf(image_paths = images_path_mrf, model_path = model_path, mrf_path = mrf_path)

#test_learning()

def fun(a,b,c=None,d=None):
    if not c :
        c = 3
        print 'c=',c
    if not d :
        d = 4
        print 'd=',d


fun(1,2,d=4)












