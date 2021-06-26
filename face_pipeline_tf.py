import cv2 
import numpy as np
import time
import copy

from modules.data_reader import DataReader
from modules.face_detector_tf import FaceDetector
from modules.prnet_image_cropper_tf import PRNetImageCropper
from modules.prnet_tf import PRNet
from modules.utils import visualization_utils_color as vis_util
from modules.utils.cv_plot import plot_kpt

# ============ Video Input Modules ============
reader = DataReader()
reader.Setup("./media/test.mp4")

# ============ Face Detection Module ============
face_detector = FaceDetector()
face_detector.Setup()

# ============ PRNet Modules ============
PRNetImageCropper.Setup()

prnet = PRNet()
prnet.Setup()

width = int(reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(reader.cap.get(cv2.CAP_PROP_FPS))
fourcc = int(reader.cap.get(cv2.CAP_PROP_FOURCC))
out_face = cv2.VideoWriter("./media/test_out.avi", 0, fps, (width, height))
out_prnet = cv2.VideoWriter('./media/output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))

try:

    while(True):

        t1 = time.time()
        # Read input
        frame_data = reader.PostProcess()
        if not frame_data:  # end of video 
            break 

        # Obj detection module
        face_detector.PreProcess(frame_data)
        face_detector.Apply()
        face_data = face_detector.PostProcess()

        image = frame_data['img']
        boxes = face_data['meta']['obj']['boxes']
        classes = face_data['meta']['obj']['classes']
        scores = face_data['meta']['obj']['scores']
        category_index = face_data['meta']['obj']['category_index']
        
        image_cp = copy.deepcopy(image)
        vis_util.visualize_boxes_and_labels_on_image_array(
          image_cp,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=4)
        out_face.write(image_cp)

        if "box" in face_data['meta']['obj']:
            prnet_image_cropper = PRNetImageCropper()
            prnet_image_cropper.PreProcess(face_data)
            prnet_image_cropper.Apply()
            cropper_box = prnet_image_cropper.PostProcess()

            prnet.PreProcess(cropper_box)
            prnet.Apply()
            prnet_data = prnet.PostProcess()
            kpt = prnet_data['meta']['obj']["keypoints"]
            vertices = prnet_data['meta']['obj']["vertices"]

            out_prnet.write(plot_kpt(image, kpt))

except KeyboardInterrupt:
    if out_face:
        out_face.release()