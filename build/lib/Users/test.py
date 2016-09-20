from evaluation.visualization import visualize_results, visualize_learning
from apply_model import myelin, pipeline
from mrf import learn_mrf


def test_prediction():

    #build_data('/AxonSeg/viherm/Desktop/CARS','/AxonSeg/viherm/Desktop/Train', trainRatio = 0.80, folder_number = 1)


    model_path = '/AxonSeg/viherm/Desktop/AxonSegmentation/AxonSeg/data/models/model_parameters2'
    model_restored_path = '/AxonSeg/viherm/Desktop/AxonSegmentation/AxonSeg/data/models/model_parameters1'
    mrf_path = '/AxonSeg/viherm/Desktop/AxonSegmentation/AxonSeg/data/models/mrf_parameters'

    visualize_learning(model_path, model_restored_path, restore = True, start_visu=0)

    image_path = '/AxonSeg/viherm/Desktop/CARS/data%s'%6

    pipeline(image_path, model_path, mrf_path)
    visualize_results(image_path)

def test_learning():
    image_path = '/AxonSeg/viherm/Desktop/CARS/data4'
    model_path = '/AxonSeg/viherm/Desktop/AxonSegmentation/AxonSeg/data/models/model_parameters3'
    mrf_path = '/AxonSeg/viherm/Desktop/AxonSegmentation/AxonSeg/data/models/model_parameters3'
    learn_mrf(image_path,model_path, mrf_path)

test_learning()









