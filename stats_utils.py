import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch import nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import os
from trainTest import call_model

def plot_confusionmat(model, testloader, model_name):
        with open("/home/jschm20/DNN/Miniproject/categories.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
        truelabels = []
        predictions = []
        model.eval()
        print("Getting predictions from test set...")
        for data, target in testloader:
            data = data.cuda()
            target = target.cuda()
            for label in target.data.cpu().numpy():
                truelabels.append(int(label))
            for prediction in call_model(model, data, model_name).data.cpu().numpy().argmax(1):
                predictions.append(int(prediction[0])) 

        # Plot the confusion matrix
        cm = confusion_matrix(truelabels, predictions)
        f1 = f1_score(truelabels, predictions, average='micro')
        percision = precision_score(truelabels, predictions, average='micro')
        recall = recall_score(truelabels, predictions, average='micro')
        # tick_marks = np.arange(len(categories))

        df_cm = pd.DataFrame(cm, index = categories, columns = categories)
        plt.figure(figsize = (4,4))
        sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
        plt.xlabel("Predicted Shape", fontsize = 12)
        plt.ylabel("True Shape", fontsize = 12)
        # plt.show()
        plt.savefig(os.path.join(os.environ["miniproject_output_path"], 'CM_{0}.png'.format(model_name)))
        print("Model F1 Score is: %1.3f" % f1)
        print("Model Percision Score is: %1.3f" % percision)
        print("Model Recall Score is: %1.3f" % recall)