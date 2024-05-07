import numpy as np
import tensorflow as tf
import keras
from processdata import ProcessData
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import argparse
from sklearn.model_selection import GridSearchCV
from processdata import ProcessData
        
class GridSearch(ProcessData):
    
    
    def __init__(self,classifier):
        super().__init__()
        self.classifier=classifier
        
    def runsearch(self):
    
        X_train, y_train, class_names = self.load_csv('train_data.csv')
        processed_X_train = self.preprocess_data(X_train)
        X_test, y_test, _ = self.load_csv('test_data.csv')
        processed_X_test = self.preprocess_data(X_test)
        
        if (self.classifier=="knn"):
            
            knn = KNeighborsClassifier()
            
            param_grid = {'n_neighbors': [3, 5, 7, 9], 
                          'weights': ['uniform', 'distance']
            }
            
            grid_search = GridSearchCV(knn, param_grid, cv=5, verbose=1, n_jobs=-1)
            grid_search.fit(np.array(processed_X_train), np.array(y_train))
            best_knn = grid_search.best_estimator_
            best_params = grid_search.best_params_
            accuracy = best_knn.score(processed_X_test,y_test)
            print(f"Best model accuracy: {accuracy}")
            print(best_params)
            
        elif (self.classifier=="gbm"):
            
            gb_clf = GradientBoostingClassifier()
            
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5],
                'max_depth': [3, 5, 7]
            }
            
            grid_search = GridSearchCV(gb_clf, param_grid, cv=5, verbose=1, n_jobs=-1)
            grid_search.fit(np.array(processed_X_train), np.array(y_train))
            best_knn = grid_search.best_estimator_
            best_params = grid_search.best_params_
            accuracy = best_knn.score(processed_X_test,y_test)
            print(f"Best model accuracy: {accuracy}")
            print(best_params)
        
        elif (self.classifier=="rf"):
            
            rf_clf = RandomForestClassifier()
            
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(rf_clf, param_grid, cv=5, verbose=1, n_jobs=-1)
            grid_search.fit(np.array(processed_X_train), np.array(y_train))
            best_knn = grid_search.best_estimator_
            best_params = grid_search.best_params_
            accuracy = best_knn.score(processed_X_test,y_test)
            print(f"Best model accuracy: {accuracy}")
            print(best_params)
        
        else:
            print(f"Algorithm {self.classifier} not supported...")
            print("Exitting...")

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Inference Options')
    parser.add_argument('--algorithm',required=True, type=str, help='Path to the model weights')
    args=parser.parse_args()
    
    gridsearch=GridSearch(args.algorithm)
    gridsearch.runsearch()