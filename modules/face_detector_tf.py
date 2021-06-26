import sys
import numpy as np
import tensorflow as tf
import cv2

# sys.path.append("..")
from modules.utils import label_map_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './models/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './modules/protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def find_face_bounding_box(boxes, scores):
    min_score_thresh = 0.7
    for i in range(0, boxes.shape[0]):
        if scores[i] > min_score_thresh:
            return tuple(boxes[i].tolist())

class FaceDetector:
    def Setup(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.sess = tf.Session(graph=detection_graph, config=gpu_config)

        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
        
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes_tensor = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections_tensor = detection_graph.get_tensor_by_name('num_detections:0')
        
        self.log('init done')

    def PreProcess(self, input):
        self.input = input
        self.image = input['img']

        image_np = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_np_expanded = np.expand_dims(image_np, axis=0)
        self.frame_height, self.frame_width= self.image.shape[:2]

    def Apply(self):
        (self.boxes, self.scores, self.classes, self.num_detections) = self.sess.run(
          [self.boxes_tensor, self.scores_tensor, self.classes_tensor, self.num_detections_tensor],
          feed_dict={self.image_tensor: self.image_np_expanded})

        self.box = find_face_bounding_box(self.boxes[0], self.scores[0])

    def PostProcess(self):
        output = self.input
        output['meta']['obj'] = {
                                    'boxes': self.boxes,
                                    'scores': self.scores,
                                    'classes': self.classes,
                                    'category_index': category_index
                                }
        if self.box is None:
            # Image passthrough
            return output

        ymin, xmin, ymax, xmax = self.box

        (left, right, top, bottom) = (xmin * self.frame_width, xmax * self.frame_width,
                                      ymin * self.frame_height, ymax * self.frame_height)
        
        # print('box found: {} {} {} {}'.format(left, right, top, bottom))

        normalized_box = np.array([left, right, top, bottom])

        output['meta']['obj']['box'] = normalized_box
        return output

    def log(self, s):
        print('[FaceDetector] %s' % s)

