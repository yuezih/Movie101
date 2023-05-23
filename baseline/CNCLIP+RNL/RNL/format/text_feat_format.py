import json
import pdb
import os
import sys
import numpy as np
from gensim.models import KeyedVectors
# sys.path.append('../../..')
# from run_on_video.data_utils import ClipFeatureExtractor


def load_json(filename):
    with open(filename, encoding='utf8') as fr:
        return json.load(fr)

def save_json(data, filename):
    with open(filename, "w") as f:
        return json.dump(data, f)

def load_word2vec(filename, binary=False):
    word2vec = KeyedVectors.load_word2vec_format(filename, binary=binary)
    return word2vec


class AnetConvert2DB(object):
    def __init__(self, save_dir, word2vec_path = '../data/glove.840B.300d.txt'):
        # self.text_encoder = ClipFeatureExtractor(
        #     framerate=1/2, size=224, centercrop=True,
        #     model_name_or_path="ViT-B/32", device="cuda"
        # )
        self.save_dir = save_dir
        self.word2vec = load_word2vec(word2vec_path)
    
    def q_feat_convert(self,anno_files):
        count = 0
        pdb.set_trace()
        for anno_file in anno_files:
            data = load_json(anno_file)
            for e_data_dict in data:
                vid, duration, timestamps, sentence, words, id2pos, adj_mat = e_data_dict
                words_vec = np.asarray([self.word2vec[word] for word in words])
                words_vec = words_vec.astype(np.float32)
                save_file = vid + '_' + str(count) +'.npy'
                count = count + 1
                e_data_dict.append(save_file)
                np.save(os.path.join(self.save_dir, save_file), words_vec)
            save_json(data,'new_'+ anno_file)
       

if __name__ == "__main__":
    anno_file = ['val_data_gcn.json','test_data_gcn.json','train_data_gcn.json']
    save_dir = '../data/text/glove/'
    AnetData = AnetConvert2DB(save_dir)
    AnetData.q_feat_convert(anno_file)