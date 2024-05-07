import numpy as np
import pandas as pd
import os



def listfiles(dataset_path):
    
    
    
    print("-------------Training Images-------------")
    
    for classes in os.listdir(dataset_path+"train/"):
        count=0
        for img_path in os.listdir(dataset_path+"train/"+classes):
            count+=1
        print(f"count of class {classes} : {count}")
    
    print("\n-------------Test Images-------------")
    
    
    for classes in os.listdir(dataset_path+"test/"):
        count=0
        for img_path in os.listdir(dataset_path+"test/"+classes):
            count+=1
        print(f"count of class {classes} : {count}")
        

def dataset_info(train_path,test_path):
    dataset=pd.read_csv(f"{train_path}")
    
    print("-------------Training data-------------\n")
    print("tree    :",sum(dataset["class_name"]=="tree"))
    print("warrior :",sum(dataset["class_name"]=="warrior"))
    print("downdog :",sum(dataset["class_name"]=="downdog"))
    print("no_pose :",sum(dataset["class_name"]=="no_pose"))
    
    dataset=pd.read_csv(f"{test_path}")
    

    print("\n-------------Test data-------------\n")
    print("tree    :",sum(dataset["class_name"]=="tree"))
    print("warrior :",sum(dataset["class_name"]=="warrior"))
    print("downdog :",sum(dataset["class_name"]=="downdog"))
    print("no_pose :",sum(dataset["class_name"]=="no_pose"))


            
if __name__=="__main__":
    # listfiles("all_images/")
    dataset_info("train_data.csv","test_data.csv")