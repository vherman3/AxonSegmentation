from evaluation.visualize_results import visualize_results
from evaluation.visualize_optim import visualize_learning
from apply_model import apply, myelin
from learning.data_construction import build_data
import random

def run_test():

    build_data('/Users/viherm/Desktop/CARS','/Users/viherm/Desktop/Train', trainRatio = 0.80, folder_number = 1)

    model_number = 4

    path = '/Users/viherm/Desktop/CARS/data%s/'%6
    folder = '/Users/viherm/Desktop/AxonSegmentation/Users/data/models'
    visualize_learning(folder, 4, restore = True, restored_model=1, start_visu=0)

    apply(path, folder, model_number, mrf=False)
    myelin(path)

    visualize_results(path)

run_test()


