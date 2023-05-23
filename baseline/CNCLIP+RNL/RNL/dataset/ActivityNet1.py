import os
import numpy as np
from torch.utils.data import Dataset

from utils import load_feature, tokenize
import h5py

class ActivityNet(Dataset):
    def __init__(self, feature_path, data, word2vec, is_training=True):
        self.data = data
        self.feature_path = feature_path
        self.is_training = is_training
        self.word2vec = word2vec

    def __getitem__(self, index):
        vid, duration, timestamps, sentence = self.data[index]
        feats = np.load(os.path.join(self.feature_path, '%s.mp4.npy' % vid)).astype(np.float32)
        #feats = load_feature(os.path.join(self.feature_path, '%s.h5' % vid), dataset='ActivityNet')
        fps = feats.shape[0] / duration

        start_frame = int(fps * timestamps[0])
        end_frame = int(fps * timestamps[1])
        if end_frame >= feats.shape[0]:
            end_frame = feats.shape[0] - 1
        if start_frame > end_frame:
            start_frame = end_frame
        assert start_frame <= end_frame
        assert 0 <= start_frame < feats.shape[0]
        assert 0 <= end_frame < feats.shape[0]
        label = np.asarray([start_frame, end_frame]).astype(np.int32)

        words = tokenize(sentence, self.word2vec)
        words_vec = np.asarray([self.word2vec[word] for word in words])
        words_vec = words_vec.astype(np.float32)

        return feats, words_vec, label

    def __len__(self):
        return len(self.data)


class ActivityNetGCN(Dataset):
    def __init__(self, feature_path, text_feature_path, data, word2vec, is_training=True):
        self.data = data
        self.feature_path = feature_path
        self.text_feature_path = text_feature_path
        self.is_training = is_training
        self.word2vec = word2vec
        self.feature_path_face = '/data4/zq/Movies_dataset/code/gen_anno/video_shots/new_window_feat/v_feat_face'
        self.face_profile_h5 =  h5py.File('/data4/zq/Movies_dataset/code/gen_anno/video_shots/data/anno/face_profile_512dim.hdf5', "r")

    def __getitem__(self, index):
        vid, duration, timestamps, sentence, face_list, id2pos, adj_mat, text_file = self.data[index]
        feats = np.load(os.path.join(self.feature_path, '%s.mp4.npy' % vid)).astype(np.float32)
        feats_face = np.load(os.path.join(self.feature_path_face, '%s.mp4.npy' % vid))
        feats_face = feats_face.astype(np.float32)
        clip_face = np.linspace(start=0, stop=feats_face.shape[0] - 1, num=feats.shape[0]).astype(np.int32)
        feats = np.concatenate((feats,feats_face[clip_face]),axis=1)
        # feats = np.load(os.path.join(self.feature_path, '%s.npy' % vid)).astype(np.float32)
        words_vec = np.load(os.path.join(self.text_feature_path, text_file)).astype(np.float32)
        face_feats = []
        for face_profile_id in face_list:
            if face_profile_id in self.face_profile_h5:
                face_feats.append(self.face_profile_h5[face_profile_id]['features'])
        # print(face_feats)
        is_empty = 0
        if len(face_feats) > 0:
            face_feats = np.array(face_feats).astype(np.float32)
        else:
            is_empty = 1
            face_feats = np.zeros((1,512)).astype(np.float32)
        #feats = load_feature(os.path.join(self.feature_path, '%s.h5' % vid), dataset='ActivityNet')
        fps = feats.shape[0] / duration
        adj_mat = np.asarray(adj_mat)
        start_frame = int(fps * timestamps[0])
        end_frame = int(fps * timestamps[1])
        if end_frame >= feats.shape[0]:
            end_frame = feats.shape[0] - 1
        if start_frame > end_frame:
            start_frame = end_frame
        assert start_frame <= end_frame
        assert 0 <= start_frame < feats.shape[0]
        assert 0 <= end_frame < feats.shape[0]
        label = np.asarray([start_frame, end_frame]).astype(np.int32)

        # words_vec = np.asarray([self.word2vec[word] for word in words])
        # words_vec = words_vec.astype(np.float32)

        id2pos = np.asarray(id2pos).astype(np.int64)
        return feats, words_vec, label, id2pos, adj_mat.astype(np.int32), face_feats, is_empty

    def __len__(self):
        return len(self.data)
