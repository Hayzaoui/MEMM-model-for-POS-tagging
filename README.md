# MEMM-model-for-POS-tagging

The MEMM model was implemented for serial tagging of parts of speech in a sentence, also known as Part of Speech Tagging. Two models were built - Model 1 (the big one) and Model 2 (the small one) - and language processing tasks were performed on real data. The goal was to analyze the nature of the success and improve the accuracy of the models through careful feature selection and optimization of the training and inference processes.

For Model 1, a set of features were implemented, including f_100-f_107, as well as additional features for capturing numbers and words containing capital letters. The model was trained using the train1.wtag file and inference was performed using the Viterbi algorithm. The model was then tested on the test.wtag file and the accuracy results were reported at the word level. An analysis containing a Confusion Matrix between the 10 tags where the model was most frequently incorrect was also prepared, and suggestions were made for improving the model in order to address confusion between specific tags.

For Model 2, more flexibility was allowed in defining and implementing features, using any combination of the features implemented for Model 1 and additional features as needed. The model was trained using the train2.wtag file, but no dedicated test file was provided. Therefore, a creative approach had to be taken for evaluating the model's performance and addressing the challenge of limited training data. Inference was performed on the appropriate competition file and the labeling results were written into a new file in the wtag format. The results for both models were also submitted in the appropriate competition files, which were judged based on their accuracy.

Throughout the project, careful attention was paid to the characteristics selected for each model and the motivations for any deviations or improvements made to the basic algorithms. The training and inference times for each model were recorded, as well as the hardware used. The features added to each model were explicitly defined and any improvements made to the final models were described, including any feature cutting or optimization techniques. The approach to evaluating Model 2's performance was also described and any challenges encountered in the training and inference processes were addressed.



