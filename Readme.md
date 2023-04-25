# Code run notes

Here is the whole process of running the code:

1. you need to first download the dataset and put it into the "data" folder in the corresponding dataset folder. 

   Take dataset "Assist09" as an example:

   First, you need to first create a folder named "Assist09".

   Then, create "data" folder inside the "Assist09" folder. 

   Finally, rename the downloaded dataset to "Assist09_original.csv" and put it into the "data" folder.

   You can download the dataset here:

   **Assist09**: https://drive.google.com/file/d/1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE/view

   **Assist12**: https://drive.google.com/file/d/1cU6Ft4R3hLqA7G1rIGArVfelSZvc6RxY/view

2. Set non-fixed parameters in *HyperParameter.py*

3. After setting the non-fixed parameters, you need to run the program in the following order, please be sure to run in this order, otherwise errors will be reported: 

   *ProcessData.py*---->*TrainEmbedding.py*---->*JointEmbedding.py*---->*TrainModel.py*

4. The best acc and best auc in the *TrainModel.py* are the final running results, representing best accuracy and best ROC curve area respectively.

# Packages used

   You can use the following command to install the package: 

   ```shell
   pip lnstall tensorflow==1.15.0 pandas numpy scipy sklearn
   ```

   The version of python we are using is 3.7.0

# Contact information

If you have any questions, please feel free to contact us.

our email address is 761744650@qq.com and yjt761744650@gmail.com