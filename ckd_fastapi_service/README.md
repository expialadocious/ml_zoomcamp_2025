# Joshua L. Midterm Project to Deploy FAST API with an XGBoost Classifier Model

I wanted to utilize medical data that addresses the intersection of Chronic Kidney Disease and other comorbidities, and I strived to better understand the relationship between various medical factors in determining if a patient has Chronic Kidney Disease ("CKD")

 * This dataset is from University of California Irvine ML Repository

 * The dataset has 400 observations of anonymous patients who may or may not have Chronic Kidney Disease ("CKD")

 * The data file has 22 features used by the model to predict whether a patient has CKD or not

 * The winning model was saved into a pipeline using pickle 

 * A FAST API was created to host the winning model and can be ran locally using a Docker container
    * Use files in repository and Dockerfile to create docker image and container
    * Launch FAST API inside Docker container and interact with XGBoost Classifier model in the server to predict if patient has "CKD" or not.