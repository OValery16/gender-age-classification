from keras.models import load_model
from keras.models import model_from_json
import tensorflow as tf
from keras import backend as K
from utils import *
import cv2
from build_predicator import *
from utils import *
import matplotlib.pyplot as plt


input_size=416
max_box_per_image   = 10
anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
labels=["face"]
image_path   = "image/alexpmetc.jpg"


# load json and create model
json_file = open('freeze_graph/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json, custom_objects={"tf": tf})
# load weights into new model
loaded_model.load_weights("freeze_graph/model.h5")
print("Loaded model from disk")

g1=Graph()


if image_path[-4:] == '.mp4':
    video_out = image_path[:-4] + '_detected' + image_path[-4:]

    video_reader = cv2.VideoCapture(image_path)

    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    fps = video_reader.get(cv2.CAP_PROP_FPS)

    video_writer = cv2.VideoWriter(video_out,
                           cv2.VideoWriter_fourcc(*'MPEG'), 
                           fps, 
                           (frame_w, frame_h))
    n=0
    skip=5
    savBox=[]
    savPrediction=[]
    for i in tqdm(range(1000)):
        n=n+1
        _, image = video_reader.read()
        if i%skip==0:

            image2 = cv2.resize(image, (input_size, input_size))

            image2 = normalize(image2)

            input_image = image2[:,:,::-1]
            input_image = np.expand_dims(input_image, 0)
            dummy_array = dummy_array = np.zeros((1,1,1,1,max_box_per_image,4))
            input_data=[input_image, dummy_array]
            netout = loaded_model.predict([input_image, dummy_array])[0]
            boxes  = decode_netout2(netout, labels,anchors)
            if len(boxes) != 0:

                listImg=getFacesList(image, boxes)
                listPrediction=g1.classify_age(listImg)
                image = draw_boxes_v2(image, boxes, labels,listPrediction)
                savBox=boxes
                savPrediction=listPrediction
            video_writer.write(image)
        else:
            image = draw_boxes_v2(image, savBox, labels,savPrediction)
            video_writer.write(image)

    video_reader.release()
    video_writer.release()  
else:
    image = cv2.imread(image_path)

    image2 = cv2.resize(image, (input_size, input_size))

    image2 = normalize(image2)

    input_image = image2[:,:,::-1]
    input_image = np.expand_dims(input_image, 0)
    dummy_array = dummy_array = np.zeros((1,1,1,1,max_box_per_image,4))
    input_data=[input_image, dummy_array]
    netout = loaded_model.predict([input_image, dummy_array])[0]
    boxes  = decode_netout2(netout, labels,anchors)
    if len(boxes) != 0:

        listImg=getFacesList(image, boxes)
        listPrediction=g1.classify_age(listImg)
        image = draw_boxes_v2(image, boxes, labels,listPrediction)
        print(len(boxes), 'boxes are found')
        print(listPrediction)
        imgplot = plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

        cv2.imwrite(image_path[:-4] + '_detected2' + image_path[-4:], image)


