import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
import os, cv2
import itertools

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder
from distancing import distancing


main_dir = os.getcwd()
pipeline_config =  os.path.join(main_dir, 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config')
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(main_dir, 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/ckpt-0'))

def load_image_into_numpy_array(image):
  (im_width, im_height, channel) = image.shape
  return image.astype(np.uint8)

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

#map labels for inference decoding
label_map_path = os.path.join(main_dir, 'mscoco_complete_label_map.pbtxt')
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

os.chdir(main_dir)
cap = cv2.VideoCapture('people.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')

from datetime import datetime
print('Inference in Progress ...')
frame = 0
(H, W) = (None, None)
df = pd.DataFrame()
while True:
    count = 0
    now = datetime.now()
    frame += 1
    rects = []
    ret, image_np = cap.read()
    if image_np is None:
      break
    
    #image_np = cv2.resize(image_np,(640,640))
    if W is None or H is None:
      (H, W) = image_np.shape[:2]
      writter = cv2.VideoWriter('output.avi', fourcc, 25.0, (W, H), True)
    
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    
    classes = (detections['detection_classes'].numpy() + label_id_offset).astype(int)
    scores = detections['detection_scores'].numpy()
    boxes = detections['detection_boxes'].numpy()

    for i in range(len(boxes[0])):
      if classes[0][i] == 1 and scores[0][i] > 0.35:
        count += 1
        (ymin, xmin, ymax, xmax) = boxes[0][i]
        box = np.array((ymin, xmin, ymax, xmax)) * np.array([H, W, H, W], dtype=int)
        rects.append(box.astype("int"))

    image_np_with_detections, high_risk = distancing(rects, image_np_with_detections, (225,275))
    details = {'Frame ID' : 'Frame '+str(frame), 'Date and Time' : now.strftime("%d/%m/%Y %H:%M:%S"),
               'Total people' : count, 'People at High Risk' : high_risk}
    df = df.append([details],ignore_index=True)
    df.to_csv('output.csv')
    
    # write video
    writter.write(image_np_with_detections)
    
cap.release()
writter.release()
