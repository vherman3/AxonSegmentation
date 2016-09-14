import matplotlib.pyplot as plt
import pickle
from os.path import dirname, abspath

######################################################
# Script for the vizualization of the learning process
######################################################

def visualize_learning(folder,model_number, restore = False, restored_model=1, start_visu=0):

    current_path = dirname(abspath(__file__))
    parent_path = dirname(current_path)

    folder_model = folder+'/model_parameters%s'%model_number

    folder_restored_model = folder+"/model_parameters%s/"%restored_model

    file = open(folder_model+'/evolution.pkl','r') # learning variables : loss, accuracy, epoch
    evolution = pickle.load(file)


    if restore :
        file_restored = open(folder_restored_model+'/evolution.pkl','r')
        evolution_restored = pickle.load(file_restored)
        last_epoch = evolution_restored['steps'][-1]

        evolution_merged = {}
        for key in ['steps','accuracy','loss'] :
            evolution_merged[key] = evolution_restored[key]+evolution[key]

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.plot(evolution_merged['steps'][start_visu:], evolution_merged['accuracy'][start_visu:], '-', label = 'accuracy')
        plt.ylabel('Accuracy')
        plt.ylim(ymin = 0.7)
        ax2 = ax.twinx()
        ax2.axvline(last_epoch, color='k', linestyle='--')
        plt.title('Evolution merged (before and after restauration')
        ax2.plot(evolution_merged['steps'][start_visu:], evolution_merged['loss'][start_visu:], '-r', label = 'loss')
        plt.ylabel('Loss')
        plt.ylim(ymax = 100)
        plt.xlabel('Epoch')

    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.plot(evolution['steps'][start_visu:], evolution['accuracy'][start_visu:], '-', label = 'accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(ymin = 0.7)
    ax2 = ax.twinx()
    plt.title('Accuracy and loss evolution')
    ax2.plot(evolution['steps'][start_visu:], evolution['loss'][start_visu:], '-r', label = 'loss')
    plt.ylabel('Loss')
    plt.ylim(ymax = 100)
    plt.xlabel('Epoch')
    plt.show()
