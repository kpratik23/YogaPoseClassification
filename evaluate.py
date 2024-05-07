import numpy as np
from processdata import ProcessData
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from processdata import ProcessData
import joblib
import keras
import argparse
from colorama import Fore,Style

class Evaluate(ProcessData):
    
    def __init__(self):
        super().__init__()
        self.accuracy=None
        self.precision=None
        self.recall=None
        self.auc_roc=None
        self.f1=None
        self.cm=None
    
    def display_results(self):
    
        # Displaying confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.cm, annot=True, cmap='Blues', fmt='d', cbar=False, linewidths=0.5,linecolor="black")
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

        # Displaying classification metrics
        print(f"{Fore.GREEN}Accuracy:", self.accuracy)
        print(f"{Fore.GREEN}Precision (Macro):", self.precision)
        print(f"{Fore.GREEN}Recall (Macro):", self.recall)
        print(f"{Fore.GREEN}F1 Score (Macro):", self.f1)
        print(f"{Fore.GREEN}AUC-ROC:", self.auc_roc)
        Style.RESET_ALL
        
    def evaluate_nn(self,model_path):
        
        
        model=keras.models.load_model(model_path)
        
        X_test, y_test, _ = self.load_csv('test_data.csv')
        processed_X_test = self.preprocess_data(X_test)

        predictions=model.predict(processed_X_test,verbose=False)

        y_pred=[]
        y_true=[]

        for i in range(predictions.shape[0]):
            y_pred.append(predictions[i].argmax())
            y_true.append(y_test[i].argmax())
            
        # confusion matrix
        self.cm = confusion_matrix(y_test, y_pred)

        # Calculate accuracy
        self.accuracy = accuracy_score(y_test, y_pred)

        # Calculate precision, recall, and F1 score for each class and then macro-average
        self.precision = precision_score(y_test, y_pred, average='macro')
        self.recall= recall_score(y_test, y_pred, average='macro')
        self.f1 = f1_score(y_test, y_pred, average='macro')
        self.auc_roc = roc_auc_score(y_test, predictions,multi_class="ovr",average="macro")
        
        self.display_results()
    
    def evaluate_others(self,model_path):
        
        X_test, y_test, _ = self.load_csv('test_data.csv')
        processed_X_test = self.preprocess_data(X_test)
        
        classifier=joblib.load(model_path)

        # Make predictions on the test set
        predictions = classifier.predict_proba(processed_X_test)

        y_pred=classifier.predict(processed_X_test)
        y_true=y_test

        # Calculate confusion matrix
        self.cm = confusion_matrix(y_true, y_pred)

        # Calculate accuracy
        self.accuracy = accuracy_score(y_true, y_pred)

        # Calculate precision, recall, and F1 score for each class and then macro-average
        self.precision = precision_score(y_true, y_pred, average='macro')
        self.recall = recall_score(y_true, y_pred, average='macro')
        self.f1 = f1_score(y_true, y_pred, average='macro')
        self.auc_roc = roc_auc_score(y_test, predictions,multi_class="ovr",average="macro")

        self.display_results()
    
    def evaluate(self,model_path):

        if (model_path.split(".")[-1]=="pkl"):
            self.evaluate_others(model_path)
        elif (model_path.split(".")[-1]=="hdf5"):
            self.evaluate_nn(model_path)
        else:
            print(f"Invalid model path {model_path}")
            print("Exitting ...")
            exit(0)
        


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Inference Options')
    parser.add_argument('--model',required=True, type=str, help='Path to the model weights')
    args=parser.parse_args()
    
    Evaluate().evaluate(args.model)
