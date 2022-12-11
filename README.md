# Dog breed classification  using AWS SageMaker
# By Mostafa Mohamed Mohamed Imam




### this is a project showcasing image classification using AWS services to tune and optimize a pre-trained Deep Learning model using transfer learning and deploying the model to a production grade endpoint with implementation for real-time logging and resource usage statistics 

<br>
<br>

**it is recommended to use "jupyter notebook renderes" extension with vs code while viewing the train_and_deploy notebook to better render the SageMaker Debugger Profiling Report**

<br>
<br>

### Files used

<br>

- hpo.py - This script file contains code that will be used by the hyperparameter tuning jobs to train and test/validate the models with different hyperparameters to find the best hyperparameter

- train_model.py - This script file contains the code that will be used by the training job to train and test/validate the model with the best hyperparameters that were determined got from hyperparameter tuning

- endpoint_inference.py - This script contains code that is used by the deployed endpoint to perform some preprocessing (transformations) , serialization- deserialization and predictions/inferences and post-processing using the saved model from the training job.

- train_and_deploy.ipynb -- This jupyter notebook using thr previous files and orchestrating the project with AWS services.

<br>
<br>

### Project flow

<br>

- We start the project by downloading the dataset, extracting it and then uploading it to an s3 bucket to be able to use it seamlessly with AWS services

<br>

- We then choose to use AdamW optimizer due to it's better generalization over the original Adam due to it's method of decoupling weight decay from learning rate and it's ease of hyperparameter tuning and we choose the following hyperparameters to tune :
    - weight decay => to take better advantage of the decoupling mentioned earlier
    - learning rate => to take better advantage of the decoupling mentioned earlier
    - epochs
    - batch size

<br>

- We use script mode to initiate a batch of hyperparameter optimization jobs using an AWS ml.g4dn.xlarge compute instance which has 4 virtual cpus and 16gbs of memory

<img src="service snapshots\hpo-tuning job  Screenshot 2022-11-16 011859.png" alt="hpo jobs snapshot" title="hpo jobs">

<img src="service snapshots\training jobs for hpo   Screenshot 2022-11-16 012055.png" alt="hpo jobs snapshot" title="hpo jobs">

<br>

- We use the sagemaker.tuner built-in reporting to determine the best hyperparameters

<br>

- We set up profiling and debugging and initiate a training job using the best hyperparameters determined earlier on an AWS ml.g4dn.xlarge instance

<img src="service snapshots\training job  Screenshot 2022-11-16 014516.png" alt="training job snapshot" title="training jobs">


<br>

- We plot tensors 

<img src="service snapshots\plot  Screenshot 2022-11-16 033629.png" alt="tensor plot snapshot" title="tesnor plot">



- The insights gained from the plot are :

    - there is some anomalous behavior since we aren't getting smooth output lines 

    - If we had more credits/resources we could try using a different pre-trained or optimizer and maybe tweak with the number of fully connected layers depending on the results

<br>

- We get insights from SageMaker debugger profiling report (available in project files in the folder "profiler_report")

<br>

- We use the model trained with the best hyperparameters and invoke an endpoint 

<br>

- We invoke the endpoint to make inferences in which the model performs well with the selected test images getting 4 correct inferences out of 5

    - code used to invoke the endpoint: 
    <br>
    ```python
    runtime= boto3.client('runtime.sagemaker')

        test_images = ["./dogImages/test/021.Belgian_sheepdog/Belgian_sheepdog_01488.jpg",
                    "./dogImages/test/020.Belgian_malinois/Belgian_malinois_01452.jpg",
                    "./dogImages/test/091.Japanese_chin/Japanese_chin_06184.jpg",
                    "./dogImages/test/007.American_foxhound/American_foxhound_00519.jpg",
                    "./dogImages/test/110.Norwegian_lundehund/Norwegian_lundehund_07218.jpg"
                    ]

        test_images_expected_output = [21, 20, 91, 7, 110]

        for index in range(len(test_images) ):
            test_image = test_images[index]
            expected_breed_category = test_images_expected_output[index]
            print(f"Test image no: {index+1}")
            test_file_path = os.path.join(test_image)
            print(test_file_path)
            with open(test_file_path , "rb") as f:
                payload = f.read()
                
                print("Below is the test image:")
                Image(filename=test_image)
                
                print(f"Expected dog breed category no : {expected_breed_category}")
                response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                            ContentType='image/jpeg',
                                            Body=payload)
                
                response_body = np.asarray(json.loads( response['Body'].read().decode('utf-8')))
                
                print(f"Response i.e list of probabilities of breed categories: {response_body}")
                
                #finding the highest probability i.e stating the prediction
                dog_breed_prediction = np.argmax(response_body,1) + 1 #adding 1 to align indexes
                print(f"Inference for the above test image is : {dog_breed_prediction}")
    ```
    
- The code basically provides the path for 5 selected images and then reads their data to be sent to the endpoint and get it's class predicted them awaits and displays the inference as well as a probability list containing the probabilities for each class/breed

<br>

**Note 1 : test images are provided and available in the folder "test inference images"**

**Note 2 : there are more service snapshots provided in the folder "service snapshots"**
