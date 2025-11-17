# Joshua L. Midterm Project to Deploy FAST API with an XGBoost Classifier Model

I wanted to utilize medical data that addresses the intersection of Chronic Kidney Disease and other comorbidities, and I strived to better understand the relationship between various medical factors in determining if a patient has Chronic Kidney Disease ("CKD")

 * This dataset is from University of California Irvine ML Repository

 * The dataset has 400 observations of anonymous patients who may or may not have Chronic Kidney Disease ("CKD")

 * The data file has 22 features used by the model to predict whether a patient has CKD or not

 * The winning model was saved into a pipeline using pickle 

 * A FAST API was created to host the winning model and can be ran locally using a Docker container
    * Use files in repository and Dockerfile to create docker image and container

    * Launch FAST API inside Docker container and interact with XGBoost Classifier model in the server to predict if patient has "CKD" or not.

* Feature Information are as followed:
  * age: years           (greater > 0, less < 120)
  * bp: blood pressure   (greater > 0)
  * sg: specific gravity (greater > 0)
  * al: albumin          (greater > 0)
  * su: sugar            (greater > 0)
  * bgr: blood glucose random   (greater > 0)
  * bu: blood urea       (greater > 0)
    sc: serum creatinine  (greater > 0)
    sod: sodium           (greater > 0)
    pot: potassium        (greater > 0)
    hemo: hemoglobin      (greater > 0)
    pcv: packed cell volume         (greater > 0)
    wbcc: white blood cell count    (greater > 0)
    rbcc: red blood cell count      (greater > 0)

    # Categorical features
    pcc: pus cell clumps  ["notpresent", "present"]
    ba: bacterial         ["notpresent", "present"]
    htn: hypertension   ["yes", "no"]
    dm: diabetes        ["yes", "no"]
    cad: coronary artery disease    ["yes", "no"]
    appet: appetite    ["good", "poor"]
    pe: pedal edema    ["yes", "no"]
    ane: anemia     ["yes", "no"]
