# gender/age classification

Pour exécuter le script ```python prediction_age_gender.py``` en modifiant le lien de la photo dans ```image_path```

## The architecture of this project is the following:

* ageWeights: weight for the age model
* build_predicator.py: script to build the age/gender graph (it also call "modelGender.py")
* detected_faces: example of extracted faces (it was mostly done for debugging purpose)
* freeze_graph: the yolo model + the weight (i froze the network)
* genderWeights: gender for the age model (the branch for the gender)
* image: examples of image and their respective output
* log: it 's where I put the tensorboard log
* modelGender.py: script to build the gender graph 
* model.py: helper for building the inception model
* prediction_age_gender.py: the main python script
* utils.py: useful functions
* video: examples of video and its output

## You can run our predicator with the command:

```python prediction_age_gender.py```

* You can modify the value of ```image_path``` in ```prediction_age_gender.py``` to run our model on your own picture/video.

* Note: For the sake of simplicity, I use Keras for the YOLO implementation, but I usetensorflow for the other graphs.

* Note2: Don't forget to download the weight and to put them in th corresponding folder (ageWeights/freeze_graph/genderWeights)
	* This link to each of them is available in the folder

## Know prediction errors

Several prediction errors could happen in the following case:
* The face is too far from the camera
	* The picture used in the training mostly includes people that are relatively closed to the camera
* The picture includes many character that are very closed to each other
	* The picture used in the training mostly includes people are not too closed for each other
* The face is very closed to the border
	* Our system consists in the combinasion of 2 neural networks. The first one extract the faces as a square image, the second one resize them to the correct input size and predicts the corresponding label of each of them. Howeve if the face is closed to the picture's border, the extracted face may not be a square size image, wich force our system to distord the image in order to make it fit in our model. It can lead to pottential errors.
* The lighting condition are very different from our training set
	* The picture used in the test set should have similar distribution as the validation set.
	
## Note:

In deep learning, it important to know that the training set should cover sufficiently the scenarios that you want to score later on. If the classifier sees fully new concepts or contexts it is likely to perform badly. Just a few examples:

* You train only on images from a constraint environment (say, indoor) and try to score images from a different environment (outdoor).
* You train only on images of a certain make and try to score others.
* Your test images have largely different characteristics, e.g. with respect to illumination, background, color, size, position, etc.
* Your test images contain entirely new concepts.

As a result, we invite the reader to fine tune our model in case it makes some prediction errors with their test set (see previous section). Another parameter that ca be adjusted is size of the face that is extracted (go to "utils.py" and search for 'getFacesList' and adjust the size of maxDist)
	
* Each label is written like gender, (age_interval)	
	* The exact age cannot be guessed without large training dataset (which we don't have)
	* Instead we guess the age intervale
		
# The following pictures are example of input/output:



Input            |  Output
:-------------------------:|:-------------------------:
![](/image/Capture.jpg?raw=true)  |  ![](/image/Capture_detected.jpg?raw=true)
![](/image/image_extracted1.jpg?raw=true)  |  ![](/image/image_extracted1_detected.jpg?raw=true)
![](/image/webcam.jpg?raw=true)  |  ![](/image/webcam_detected.jpg?raw=true)
![](/image/olivier.jpg?raw=true)  |  ![](/image/olivier_detected.jpg?raw=true)
![](/image/webcam_test.jpg?raw=true)  |  ![](/image/webcam_test_detected.jpg?raw=true)
![](/image/big_bang_theory4.jpg?raw=true)  |  ![](/image/big_bang_theory4_detected.jpg?raw=true)
![](/image/big_bang_theory2.jpg?raw=true)  |  ![](/image/big_bang_theory2_detected.jpg?raw=true)
![](/image/big_bang_theory5.jpg?raw=true)  |  ![](/image/big_bang_theory5_detected.jpg?raw=true)
![](/image/friends.jpg?raw=true)  |  ![](/image/friends_detected.jpg?raw=true)
![](/image/game-of-thrones.jpg?raw=true)  |  ![](/image/game-of-thrones_detected.jpg?raw=true)
![](/image/how_I_met_your_mother.jpg?raw=true)  |  ![](/image/how_I_met_your_mother_detected.jpg?raw=true)
![](/image/selfi.jpg?raw=true)  |  ![](/image/selfi_detected.jpg?raw=true)
![](/image/selfi2.jpg?raw=true)  |  ![](/image/selfi2_detected.jpg?raw=true)





