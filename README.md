# (Dio-live) AWS Sagemaker
AWS Sagemaker is a service totally managed from AWS to prepare, create, train and implement machine learning models.

**Challenge:** 

Develop a ML model to foresee if a client from a bank will sign-up in a product (CD - Certificate of Deposit). The model will be trained into a marketing dataset that contains information about the clients and external factors.

**Requirements:**

- AWS account

- Machine Learning basics

- Python basics

  #### **Activity**:

1. Access Amazon Sagemaker and select your AWS region (us-west-2 is the standard).

2. Create a notebook instance (Jupyter):

   - Notebook -> notebook instances -> create notebook instance -> yourNotebookInstanceName -> notebook instance type (keep ml.t2.medium) -> Elastic inference (keep none) -> Permissions and Encryption -> IAM role -> create a new role -> select Any S3 bucket -> create role -> create notebook instance.
   - Await the created notebook changes its status from pending to InService.

3.  Prepare the data:

   - Open Jupyter -> new -> choose conda_python3.

   - Execute/Run the code below:

      ```
      # import libraries
     import boto3, re, sys, math, json, os, sagemaker, urllib.request
     from sagemaker import get_execution_role
     import numpy as np
     import pandas as pd
     import matplotlib.pyplot as plt
     from IPython.display import Image
     from IPython.display import display
     from time import gmtime, strftime
     from sagemaker.predictor import csv_serializer
     
     # Define IAM role
     role = get_execution_role()
     prefix = 'sagemaker/DEMO-xgboost-dm'
     my_region = boto3.session.Session().region_name # set the region of the instance
     
     # this line automatically looks for the XGBoost image URI and builds an XGBoost container.
     xgboost_container = sagemaker.image_uris.retrieve("xgboost", my_region, "latest")
     
     print("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + xgboost_container + " container for your SageMaker endpoint.")
      ```

   - Next, run de code below:

     - Enter your S3 bucket name; This name must be unique (only lower case, hifen and numbers allowed).

     ```
     bucket_name = 'your-s3-bucket-name' # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET
     s3 = boto3.resource('s3')
     try:
         if  my_region == 'us-east-1':
           s3.create_bucket(Bucket=bucket_name)
         else: 
           s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={ 'LocationConstraint': my_region })
         print('S3 bucket created successfully')
     except Exception as e:
         print('S3 error: ',e)
     ```

   - Run the code below to download the data (bank_clear.csv) to your SageMaker instance and load the data in a dataframe:

     ```
     try:
       urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
       print('Success: downloaded bank_clean.csv.')
     except Exception as e:
       print('Data load error: ',e)
     
     try:
       model_data = pd.read_csv('./bank_clean.csv',index_col=0)
       print('Success: Data loaded into dataframe.')
     except Exception as e:
         print('Data load error: ',e)
     ```

   - Update/refresh your bucket on S3 and check the file.

   - Mix and split the data in training data and test data. Run the code below. Training data (70%) are used during the loop to train the model and the test data (30%) are used to evaluate the model performance.

      ```
     train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
     print(train_data.shape, test_data.shape)
      ```

4. Train the Machine Learning model

   - The code below formats the header and the 1st column of the training data and, next, load the S3 bucket data. This step is necessary to use the XGBoost pre-built algorithm from Amazon SageMaker:

     ```
     pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
     boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
     s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')
     ```

   - The code below will configure the Amazon SageMaker session, create an instance of the XGBoost model (an estimator) and define the model hiperparameters.

     ```
     sess = sagemaker.Session()
     xgb = sagemaker.estimator.Estimator(xgboost_container,role, instance_count=1, instance_type='ml.m4.xlarge',output_path='s3://{}/{}/output'.format(bucket_name, prefix),sagemaker_session=sess)
     xgb.set_hyperparameters(max_depth=5,eta=0.2,gamma=4,min_child_weight=6,subsample=0.8,silent=0,objective='binary:logistic',num_round=100)
     ```

   - The code below trains the model using gradient optimization on an ml.m4.xlarge instance. After a few minutes, you should see training logs being generated on your Jupyter Notebook.

     ```
     xgb.fit({'train': s3_input_train})
     ```

5. Publish the Machine Learning model

   - Deploy the trained model to an endpoint, format and load the CSV data, then run the model to create predictions.
   - The code below deploys the template to a server and creates a SageMaker endpoint for access. This step may take a few minutes to complete.

   ```
   xgb_predictor = xgb.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')
   ```

   - Running the code below will predict which customers will adhere to the bank's product or not in the test sample.

   ```
   from sagemaker.serializers import CSVSerializer
   
   test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values #load the data into an array
   xgb_predictor.serializer = CSVSerializer() # set the serializer type
   predictions = xgb_predictor.predict(test_data_array).decode('utf-8') # predict!
   predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array
   print(predictions_array.shape)
   ```

6. Evaluate the performance of the trained model

   - The code below compares the current values with those predicted in a table called *Confusion Matrix*. Based on the prediction, it can be concluded that a client will apply for a certificate of deposit (CD) with 89,5% accuracy for the test data clients, an accuracy of 63% for those who will apply and 90% for those who will not sign up.

     ```
     cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
     tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
     print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
     print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
     print("Observed")
     print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
     print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))
     ```

7. Clean/delete the resources created

   - In this step you will clean the environment with resources. Important: Shutting down features that are not being actively used reduces costs and it is a best practice. Failure to close your resources will result in charges to your AWS account.
   - Delete your endpoind, running the code below (In the Jupyter notebook):

    ```
   xgb_predictor.delete_endpoint(delete_endpoint_config=True)
    ```

    - Delete the training artifacts and bucket S3 running the code below (In the Jupyter notebook):

    ```
   bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
   bucket_to_delete.objects.all().delete()
    ```

   - Access Amazon SageMaker to delete your SageMaker notebook: open the SageMaker console -> Notebook -> Notebook instances -> Action (Stop) -> await its status changes to 'stopped' -> Actions (delete) -> Delete.
