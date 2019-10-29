The following steps need to be executed one by one

1. Unzip and copy all files to a working directory. Inside this directory create three sub-directories: "input", "output", and "dict". Put the input files "cybersecurity_training.csv", "cybersecurity_test.csv" and "localized_alerts_data.csv" into the "input" sub-directory. 

2. Run python data_preprocessing_lgb.py. 
This code will process the second data set to generate features, which are then combined with features in the first data set in feature_engineering_lgb.py. Finally, model_lgb.py is executed to train a LightGBM model and generate the prediction file named submission_lgb.txt in the "output" sub-directory. Note that we executed this step in Windows 10 using python 3.7 and lightgbm 2.2.3 and obtained the following local validation scores: 0.9486033716695101, [0.9452770127838978, 0.9505410206718347, 0.9493073026392043, 0.9533315606826068, 0.9429543099275257, 0.9502090233119913] (the ones in the square brackets are scores of each fold and the first one is the average score). If we submit the prediction file to the public leader board, it gets the score of 0.9496.

3. Run python data_preprocessing_xgb.py.
This code will do the same thing as in the case of data_processing_lgb.py. It processed data in the second data set and then respectively execute feature_engineering_xgb.py and model_xgb.py to train an XGBoost model and generate the prediction file named submission_xgb.txt in the "output" sub-directory. Note that we executed this step in Ubuntu 16.04 using python 3.5.2 and xgboost 0.90 and obtained the following scores: 0.9476036789290516, [0.9456142711778773, 0.9494671143123365, 0.9470587017289575, 0.9460414417345777, 0.949836865691509] If we submit the prediction file to the public leader board, it gets the score of 0.9496.

4. Run python ensemble.py
This code generates the ensemble result, which is the simple average of results generated in the previous two steps and stored in the files submission_lgb.txt and submission_xgb.txt. This final prediction result is named submission.txt, also stored in the "output" sub-directory and has the score of 0.9514 in the public leader board and 0.931743 in the final private leader board.
