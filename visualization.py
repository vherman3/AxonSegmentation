import matplotlib.pylab as plt
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image


def visualize(y_true,y_pred,img):

    score = accuracy_score(y_true, y_pred)
    print '\n accuracy score : ', score

    h, w = img.shape
    target_pred = y_pred.reshape(h, w)
    target_true = y_true.reshape(h, w)

    fig1 = plt.figure(1)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.hold(True)
    plt.imshow(target_pred, alpha=0.7)
    plt.title('Predicted target - accuracy_score : %s' % round(score, 3))

    fig2 = plt.figure(2)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.hold(True)
    plt.imshow(target_true, alpha=0.7)
    plt.title('True target')

    plt.show()


#results = pickle.load(open("test/test5/classification_res5.pkl", "rb"))
#y_pred = results['y_pred']
#y_true = results['y_test']
#img = results['img_test']
#visualize(y_true, y_pred, img)

#im = Image.open("test/test5/MRF.png").convert('L')
#mrf = np.array(im)
#mrf[mrf==255]=1
#y_pred_smoothed = mrf.reshape(-1, 1)
#score = accuracy_score(y_true, y_pred_smoothed)

#fig1 = plt.figure()
#plt.imshow(img, cmap=plt.get_cmap('gray'))
#plt.hold(True)
#plt.imshow(mrf, alpha=0.7)
#plt.title('MRF smoothing accuracy_score : %s' % round(score, 3))
#plt.show()
