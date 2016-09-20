#----------------------Building training set----------------------------#

data_path = '/Users/viherm/Desktop/CARS'
trainset_path = '/Users/viherm/Desktop/training_set'

from AxonSeg.learning.data_construction import build_data
build_data(data_path, trainset_path, trainRatio=0.80)

# #----------------------Learning process---------------------------------#
# from AxonSeg.learn_model import learn_model
#
# model_path_init = '/Users/viherm/Desktop/AxonSegmentation/AxonSeg/data/models/init_model'
# learn_model(trainset_path = trainset_path, model_path =model_path_init)
#
# model_path_new = '/Users/viherm/Desktop/AxonSegmentation/AxonSeg/data/models/new_model'
# learn_model(trainset_path = trainset_path, model_path = model_path_new, model_restored_path=model_path_init)
#
# #----------------------Visualization of the learning---------------------#
# from AxonSeg.evaluation.visualization import visualize_learning
#
# visualize_learning(model_path_init)
# visualize_learning(model_path_new,model_restored_path=model_path_init, start_visu = 3)
#
# #----------------------Learn MRF---------------------#
# from AxonSeg.mrf import learn_mrf
# data_path = '/Users/viherm/Desktop/CARS/data3'
#
# model_path_new = '/Users/viherm/Desktop/AxonSegmentation/AxonSeg/data/models/new_model'
# mrf_path = model_path_new
#
# learn_mrf(image_path = data_path, mrf_path = mrf_path)
#
# #----------------------Apply model + MRF---------------------#
# from AxonSeg.apply_model import axon_segmentation
#
# image2segment = '/Users/viherm/Desktop/CARS/data5'
# axon_segmentation(image_path=image2segment, model_path = model_path_new, mrf_path = mrf_path)
#
# #----------------------Visualization of the results--------------------#
#
# from AxonSeg.evaluation.visualization import visualize_results
# visualize_results(image2segment)
#
# #----------------------Detect myelin--------------------#
#
# from AxonSeg.apply_model import myelin
# myelin(image2segment, pixel_size=0.3)
# visualize_results(image2segment)





