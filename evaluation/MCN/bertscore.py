import os
import json

# set cuda
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def build_refs(data_path):
    refs_data = json.load(open(data_path, 'r'))
    with open('test_refs.txt', 'w') as f:
        for each in refs_data:
            f.write(each['content'] + '\n')

def build_hyps(data_path):
    hyps_data = json.load(open(data_path, 'r'))
    with open(f'test_hyps.txt', 'w') as f:
        for nid in hyps_data.keys():
            f.write(hyps_data[nid]['sentences'][0] + '\n')

if __name__ == '__main__':

    refs_path = '../../dataset/Movie101-N/test.json'
    result_path = 'YOUR/PATH/TO/RESULT_FILE'

    build_refs(refs_path)
    build_hyps(result_path)

    os.system(f'bert-score -r test_refs.txt -c test_hyps.txt --lang zh --rescale_with_baseline')