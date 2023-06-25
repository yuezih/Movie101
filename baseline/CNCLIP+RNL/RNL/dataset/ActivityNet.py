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
        self.feature_path_face = 'YOUR/PATH/TO/FRAME_FACE'
        self.face_profile_h5 =  h5py.File('YOUR/PATH/TO/PORTRAIT_FACE', "r")
        self.text_feature_path = 'YOUR/PATH/TO/TEXT_FEATURE'

    def __getitem__(self, index):
        # vid, duration, timestamps, sentence, face_list, id2pos, adj_mat, text_file = self.data[index]
        each = self.data[index]
        movie_id = each['movie_id']
        clip_start = np.random.randint(each['clip_head'][0], each['clip_head'][1]+1) if self.is_training else each['clip_head']
        # clip_start = each['clip_head']
        clip_end = clip_start + 200
        gt_s_time = each['start'] - clip_start
        gt_e_time = each['end'] - clip_start
        video_feat = self.get_video_features(movie_id, clip_start, clip_end)
        
        text_feat_id = each['text_feat_id']
        text_feat = self.get_text_features(text_feat_id)

        role_list = each['role_token']
        role_list = [role_token[1] for role_token in role_list]
        role_list = list(set(role_list))

        face_feats = []
        for portrait_id in role_list:
            if portrait_id in self.face_profile_h5:
                face_feats.append(self.face_profile_h5[portrait_id]['features'])

        is_empty = 0
        if len(face_feats) > 0:
            face_feats = np.array(face_feats).astype(np.float32)
        else:
            is_empty = 1
            face_feats = np.zeros((1,512)).astype(np.float32)
        #feats = load_feature(os.path.join(self.feature_path, '%s.h5' % vid), dataset='ActivityNet')
        id2pos = []
        adj_mat = []

        fps = 1
        adj_mat = np.asarray(adj_mat)
        start_frame = int(fps * gt_s_time)
        end_frame = int(fps * gt_e_time)

        label = np.asarray([start_frame, end_frame]).astype(np.int32)

        id2pos = np.asarray(id2pos).astype(np.int64)

        return video_feat, text_feat, label, id2pos, adj_mat.astype(np.int32), face_feats, is_empty

    def __len__(self):
        return len(self.data)

    def get_video_features(self, movie_id, clip_start, clip_end):
        # movie_feats = np.load(os.path.join(self.feature_path, '%s.mp4.npy.npz' % movie_id))['features'].astype(np.float32)
        # clip_movie = np.load(os.path.join(self.feature_path, 'clip', '%s.mp4.npy.npz' % movie_id))['features'].astype(np.float32)
        s3d_movie = np.load(os.path.join(self.feature_path, 's3d', '%s-4.npz' % movie_id))['features'].astype(np.float32)
        face_movie = np.load(os.path.join(self.feature_path_face, '%s.mp4.npy.npz' % movie_id))['features'].astype(np.float32)
        # 取出clip_start到clip_end的特征
        # clip_feature = clip_movie[clip_start:clip_end]
        s3d_feature = s3d_movie[clip_start:clip_end]
        face_feature = face_movie[clip_start:clip_end]
        # 将clip和s3d的特征拼接
        feature = np.concatenate((s3d_feature, face_feature), axis=1)
        return feature
    
    def get_text_features(self, text_feat_id):
        text_feats = np.load(os.path.join(self.text_feature_path, '%s.npy' % text_feat_id)).astype(np.float32)
        return text_feats