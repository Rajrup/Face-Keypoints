import numpy as np
import tensorflow as tf
import time
import os

from modules.predictor_tf import PosPrediction

class PRNet:
    resolution_inp = 256
    resolution_op = 256
    MaxPos = resolution_inp * 1.1
    # DST_PTS = np.array(
    #     [[0, 0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])
    uv_kpt_ind = None
    face_ind = None
    triangles = None

    def Setup(self):
        prefix='./models'

        self.uv_kpt_ind = np.loadtxt(
            prefix + '/PRNet/uv-data/uv_kpt_ind.txt').astype(np.int32)  # 2 x 68 get kpt
        self.face_ind = np.loadtxt(
            prefix + '/PRNet/uv-data/face_ind.txt').astype(np.int32)
        self.triangles = np.loadtxt(
            prefix + '/PRNet/uv-data/triangles.txt').astype(np.int32)

        #---- load PRN
        self.pos_predictor = PosPrediction(
            self.resolution_inp, self.resolution_op)
        prn_path = os.path.join(
            prefix, 'PRNet/net-data/256_256_resfcn256_weight')
        if not os.path.isfile(prn_path + '.data-00000-of-00001'):
            print("please download PRN trained model first.")
            exit()
        self.pos_predictor.restore(prn_path)

        self.log('init done')

    def PreProcess(self, input):
        self.input = input
        self.image = input['img']
        self.tform_params = input['meta']['obj']['tform_params']
        self.cropped_image = input['meta']['obj']['cropped_image']

    def Apply(self):
        image = self.cropped_image

        # new_image = image[np.newaxis, :, :, :]
        # new_image = image.astype(np.float32)
        new_image = image

        self.cropped_pos = self.pos_predictor.predict(new_image)

    def PostProcess(self):
        cropped_vertices = np.reshape(self.cropped_pos, [-1, 3]).T
        z = cropped_vertices[2, :].copy() / self.tform_params[0, 0]
        cropped_vertices[2, :] = 1
        vertices = np.dot(np.linalg.inv(self.tform_params), cropped_vertices)
        vertices = np.vstack((vertices[:2, :], z))
        pos = np.reshape(
            vertices.T, [PRNet.resolution_op, PRNet.resolution_op, 3])

        key_points = pos[self.uv_kpt_ind[1, :], self.uv_kpt_ind[0, :], :]
        all_vertices = np.reshape(pos, [self.resolution_op**2, -1])
        vertices = all_vertices[self.face_ind, :]

        output = self.input
        output['meta']['obj']['keypoints'] = key_points
        output['meta']['obj']['vertices'] = vertices

        return output
    
    def log(self, s):
        print('[PRNet] %s' % s)
