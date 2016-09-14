from evaluation.visualize_results import visualize_results
from evaluation.visualize_optim import visualize_learning
from apply_model import apply, myelin

def run_test():
    model_number = 4

    visualize_learning(4, restore = True, restored_model=1, start_visu=0)
    path = '/Users/viherm/Desktop/CARS/data%s/'%6

    apply(path, model_number, mrf=False)
    myelin(path)

    visualize_results(path)


