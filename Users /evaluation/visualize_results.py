import pickle
import matplotlib.pyplot as plt
from scipy.misc import imread
from sklearn.metrics import accuracy_score
from segmentation_scoring import rejectOne_score, dice
from sklearn import preprocessing
import os
from tabulate import tabulate


def visualize_results(path) :

    path_img = path+'image.jpg'
    Mask = False

    if not 'results.pkl' in os.listdir(path):
        print 'results not present'

    file = open(path+'results.pkl','r')
    res = pickle.load(file)

    img_mrf = res['img_mrf']
    prediction = res['prediction']
    image_init = imread(path_img, flatten=False, mode='L')


    plt.figure(1)
    plt.title('With MRF')
    plt.imshow(image_init, cmap=plt.get_cmap('gray'))
    plt.hold(True)
    plt.imshow(img_mrf, alpha=0.7)


    plt.figure(2)
    plt.title('Without MRF')
    plt.imshow(image_init, cmap=plt.get_cmap('gray'))
    plt.hold(True)
    plt.imshow(prediction, alpha=0.7)

    if 'mask.jpg' in os.listdir(path):
        Mask = True
        path_mask = path+'mask.jpg'
        mask = preprocessing.binarize(imread(path_mask, flatten=False, mode='L'), threshold=125)

        acc = accuracy_score(prediction.reshape(-1,1), mask.reshape(-1,1))
        score = rejectOne_score(image_init, mask.reshape(-1, 1), prediction.reshape(-1,1), visualization=False, min_area=1, show_diffusion = True)
        Dice = dice(image_init, mask.reshape(-1, 1), prediction.reshape(-1,1)).mean()
        acc_mrf = accuracy_score(img_mrf.reshape(-1, 1), mask.reshape(-1, 1))
        score_mrf = rejectOne_score(image_init, mask.reshape(-1,1), img_mrf.reshape(-1,1), visualization=False, min_area=1, show_diffusion = True)
        Dice_mrf = dice(image_init, mask.reshape(-1, 1), img_mrf.reshape(-1,1)).mean()

        headers = ["MRF", "accuracy", "sensitivity", "errors", "diffusion", "Dice"]
        table = [["False", acc, score[0], score[1], score[2], Dice],
        ["True", acc_mrf, score_mrf[0], score_mrf[1], score_mrf[2], Dice_mrf]]

        subtitle2 = '\n\n---Scores---\n\n'
        scores = tabulate(table, headers)
        text = subtitle2+scores
        print text

        file = open(path+"Report_results.txt", 'w')
        file.write(text)
        file.close()

    if 'myelin.jpg' in os.listdir(path):
        path_myelin = path + 'myelin.jpg'
        myelin = preprocessing.binarize(imread(path_myelin, flatten=False, mode='L'), threshold=125)


        plt.figure(3)
        plt.title('Myelin')
        plt.imshow(image_init, cmap=plt.get_cmap('gray'))
        plt.hold(True)
        plt.imshow(myelin, alpha=0.7)

        plt.figure(4)
        plt.title('Myelin - Axon')
        plt.imshow(img_mrf, cmap=plt.get_cmap('gray'))
        plt.hold(True)
        plt.imshow(myelin, alpha=0.7)

        if Mask :
            plt.figure(5)
            plt.title('Myelin - GroundTruth')
            plt.imshow(mask, cmap=plt.get_cmap('gray'))
            plt.hold(True)
            plt.imshow(myelin, alpha=0.7)

    plt.show()

#visualize_results(path='/Users/viherm/Desktop/CARS/data6/')