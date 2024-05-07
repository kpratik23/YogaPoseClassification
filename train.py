import numpy as np
import tensorflow as tf
import keras
from processdata import ProcessData
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import argparse
import joblib


class Trainer(ProcessData):
    
    def __init__(self):
        super().__init__()
    
    def train_nn(self):
        
        X, y, class_names = self.load_csv('train_data.csv')
        y = keras.utils.to_categorical(y)

        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
        X_test, y_test, _ = self.load_csv('test_data.csv')
        y_test = keras.utils.to_categorical(y_test)
        
        processed_X_train = self.preprocess_data(X)
        # processed_X_val =  self.preprocess_data(X_val)
        processed_X_test = self.preprocess_data(X_test)

        inputs = tf.keras.Input(shape=(34))
        layer = keras.layers.Dense(128, activation=tf.nn.relu6)(inputs)
        layer = keras.layers.Dropout(0.5)(layer)
        layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
        layer = keras.layers.Dropout(0.5)(layer)
        outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

        model = keras.Model(inputs, outputs)


        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        checkpoint_path = "./models/weights.best.hdf5"
        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                    monitor='accuracy',
                                    verbose=1,
                                    save_best_only=True,
                                    mode='max')
        earlystopping = keras.callbacks.EarlyStopping(monitor='accuracy', 
                                                    patience=20)

        # Start training
        print('--------------TRAINING----------------')
        history = model.fit(processed_X_train, y,
                            epochs=300,
                            batch_size=16,
                            callbacks=[checkpoint, earlystopping])

    
    def train_others(self,classifier):

        X_train, y_train, class_names = self.load_csv('train_data.csv')
        processed_X_train = self.preprocess_data(X_train)
        
        if (classifier=="gbm"):
            # {'learning_rate': 0.5, 'max_depth': 7, 'n_estimators': 200}
            gbm_classifier = GradientBoostingClassifier(n_estimators=200, learning_rate=0.5, max_depth=7,verbose=True)
            gbm_classifier.fit(processed_X_train, y_train)
            joblib.dump(gbm_classifier, './models/gbm.pkl')
            print("Training done.\nModel saved at ./models/gbm.pkl\nRun evaluate.py to evaluate the model")
        
        elif (classifier=="knn"):
            # {'n_neighbors': 5, 'weights': 'distance'}
            knn_classifier = KNeighborsClassifier(algorithm="auto",
                                                  n_neighbors=5,
                                                  weights="distance")
            
            knn_classifier.fit(processed_X_train,y_train)
            joblib.dump(knn_classifier, './models/knn.pkl')
            print("Training done.\nModel saved at ./models/knn.pkl\nRun evaluate.py to evaluate the model")

        elif (classifier=="rf"):
            # {'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 50}
            forest = RandomForestClassifier(random_state=42,
                                            max_depth=20,
                                            min_samples_split=5,
                                            n_estimators=50,
                                            min_samples_leaf=2,
                                            verbose=True)
            
            forest.fit(processed_X_train, y_train)
            joblib.dump(forest, './models/rf.pkl')
            print("Training done.\nModel saved at ./models/rf.pkl\nRun evaluate.py to evaluate the model")
        else:
            print(f"{classifier} not supported.")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Inference Options')
    parser.add_argument('--algorithm', choices=['nn', 'rf', 'knn','gbm'],required=True, help='classifier to train')
    args=parser.parse_args()
    
    
    trainer=Trainer()
    
    if (args.algorithm=="nn"):
        trainer.train_nn()
    elif (args.algorithm in ['nn', 'rf', 'knn','gbm']):
        trainer.train_others(args.algorithm)
    else:
        print(f"Invalid algorithm {args.algorithm}")

        
    