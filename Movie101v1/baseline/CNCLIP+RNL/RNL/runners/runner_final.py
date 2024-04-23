import collections
import logging
import os

import torch

from torch.utils.data import DataLoader

import criteria
from dataloaders.clip_loader import get_dataset
from models_.gcn_final import Model
# from models_.latent_attention import Model
from optimizer.adam_optimizer import AdamOptimizer
from optimizer.lr_scheduler.inverse_square_root_schedule import InverseSquareRootSchedule
from utils import load_word2vec, AverageMeter, TimeMeter, CountMeter
import numpy as np
import pdb
import json
from tqdm import tqdm

def save_json(data, filename):
    with open(filename, "w",encoding='utf-8') as f:
        return json.dump(data, f,ensure_ascii=False)

class Runner:
    def __init__(self, args):
        self.num_updates = 0
        self.args = args
        self.word2vec = None #load_word2vec(args.word2vec_path)
        self._build_loader()
        self._build_model()
        self._build_optimizer()

    def _build_loader(self):
        args = self.args
        train = get_dataset(args.dataset, args.feature_path, args.text_feature_path, args.train_data,
                            self.word2vec, args.max_num_frames, args.max_num_words,
                            args.max_num_nodes, is_training=True)
        val = get_dataset(args.dataset, args.feature_path, args.text_feature_path, args.val_data,
                          self.word2vec, args.max_num_frames, args.max_num_words,
                          args.max_num_nodes, is_training=False)
        test = get_dataset(args.dataset, args.feature_path, args.text_feature_path, args.test_data,
                           self.word2vec, args.max_num_frames, args.max_num_words,
                           args.max_num_nodes, is_training=False)
        self.train_loader = DataLoader(dataset=train, batch_size=self.args.batch_size, num_workers=4, shuffle=True)
        self.val_loader = DataLoader(dataset=val, batch_size=self.args.batch_size, num_workers=4,
                                     shuffle=False) if val else None
        self.test_loader = DataLoader(dataset=test, batch_size=self.args.batch_size, num_workers=4,
                                      shuffle=False) if test else None

    def _build_model(self):
        self.model = Model(self.args)
        print(self.model)
        device_ids = [0]
        # self.inference = self.model.inference
        self.model = self.model.to(torch.device('cuda:%d' % device_ids[0]))
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        if self.args.model_load_path:
            self.model.load_state_dict(torch.load(self.args.model_load_path))

    def _build_optimizer(self):
        l = list(self.model.parameters())
        self.optimizer = AdamOptimizer(self.args, list(self.model.parameters()))
        self.lr_scheduler = InverseSquareRootSchedule(self.args, self.optimizer)

    def train(self):
        if not os.path.exists(self.args.model_saved_path):
            # os.mkdir(self.args.model_saved_path)
            os.makedirs(self.args.model_saved_path)
        for epoch in range(1, self.args.max_num_epochs + 1):
            logging.info('Start Epoch {}'.format(epoch))
            self._train_one_epoch(epoch)
            path = os.path.join(self.args.model_saved_path, 'model-%d' % epoch)
            torch.save(self.model.state_dict(), path)
            logging.info('model saved to %s' % path)
            print ('epoch: {}'.format(epoch))
            self.eval()
        logging.info('Done.')

    def _train_one_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        time_meter = TimeMeter()
        for bid, (video, video_mask, words, word_mask,
                  label, scores, scores_mask, id2pos, node_mask, adj_mat, face_feats, face_mask) in enumerate(self.train_loader, 1):
            self.optimizer.zero_grad()

            model_input = {
                'frames': video.cuda(),
                'frame_mask': video_mask.cuda(), 'words': words.cuda(), 'word_mask': word_mask.cuda(),
                'label': scores.cuda(), 'label_mask': scores_mask.cuda(), 'gt': label.cuda(),
                'node_pos': id2pos.cuda(), 'node_mask': node_mask.cuda(), 'adj_mat': adj_mat.cuda(),
                'face_feats':face_feats, 'face_mask':face_mask
            }
            #import pdb;pdb.set_trace()
            predict_boxes, loss, _, _, _ = self.model(**model_input)
            loss = torch.mean(loss)
            self.optimizer.backward(loss)
      
            self.optimizer.step()
            self.num_updates += 1
            curr_lr = self.lr_scheduler.step_update(self.num_updates)

            loss_meter.update(loss.item())
            time_meter.update()

            if bid % self.args.display_n_batches == 0:
                logging.info('Epoch %d, Batch %d, loss = %.4f, lr = %.5f, %.3f seconds/batch' % (
                    epoch, bid, loss_meter.avg, curr_lr, 1.0 / time_meter.avg
                ))
                loss_meter.reset()

    def eval(self):
        data_loaders = [self.val_loader, self.test_loader]
        meters = collections.defaultdict(lambda: AverageMeter())

        self.model.eval()
        with torch.no_grad():
            for data_loader in data_loaders:
                for bid, (video, video_mask, words, word_mask,
                          label, scores, scores_mask, id2pos, node_mask, adj_mat,face_feats, face_mask) in enumerate(data_loader, 1):
                    self.optimizer.zero_grad()

                    model_input = {
                        'frames': video.cuda(),
                        'frame_mask': video_mask.cuda(), 'words': words.cuda(), 'word_mask': word_mask.cuda(),
                        'label': scores.cuda(), 'label_mask': scores_mask.cuda(), 'gt': label.cuda(),
                        'node_pos': id2pos.cuda(), 'node_mask': node_mask.cuda(), 'adj_mat': adj_mat.cuda(),
                        'face_feats':face_feats, 'face_mask':face_mask
                    }

                    predict_boxes, loss, _, _, _ = self.model(**model_input)
                    loss = torch.mean(loss)

                    meters['loss'].update(loss.item())
                    video_mask = video_mask.cpu().numpy()
                    gt_boxes = model_input['gt'].cpu().numpy()
                    predict_boxes = np.round(predict_boxes.cpu().numpy()).astype(np.int32)
                    gt_starts, gt_ends = gt_boxes[:, 0], gt_boxes[:, 1]
                    predict_starts, predict_ends = predict_boxes[:, 0], predict_boxes[:, 1]
                    predict_starts[predict_starts < 0] = 0
                    seq_len = np.sum(video_mask, -1)
                    predict_ends[predict_ends >= seq_len] = seq_len[predict_ends >= seq_len] - 1
                    IoUs = criteria.calculate_IoU_batch((predict_starts, predict_ends),
                                                        (gt_starts, gt_ends))
                    meters['mIoU'].update(np.mean(IoUs), IoUs.shape[0])
                    for i in range(1, 10, 2):
                        meters['IoU@0.%d' % i].update(np.mean(IoUs >= (i / 10)), IoUs.shape[0])
                print('| ', end='')
                for key, value in meters.items():
                    print('{}, {:.4f}'.format(key, value.avg), end=' | ')
                    logging.info('{}, {:.4f}'.format(key, value.avg)) 
                    meters[key].reset()
                print()

    def eval_new(self):
        data_loaders = [self.test_loader]
        meters = collections.defaultdict(lambda: AverageMeter())
        meters_5 = collections.defaultdict(lambda: CountMeter())

        self.model.eval()
        with torch.no_grad():
            for data_loader in data_loaders:
                rst_dict = []
                # gt_dict = []
                count = 1
                for bid, (video, video_mask, words, word_mask,
                          label, scores, scores_mask, id2pos, node_mask, adj_mat,face_feats, face_mask) in tqdm(enumerate(data_loader, 1)):
                    self.optimizer.zero_grad()

                    model_input = {
                        'frames': video.cuda(),
                        'frame_mask': video_mask.cuda(), 'words': words.cuda(), 'word_mask': word_mask.cuda(),
                        'label': scores.cuda(), 'label_mask': scores_mask.cuda(), 'gt': label.cuda(),
                        'node_pos': id2pos.cuda(), 'node_mask': node_mask.cuda(), 'adj_mat': adj_mat.cuda(),
                         'face_feats':face_feats, 'face_mask':face_mask
                    }

                    predict_boxes, loss, predict_flatten_old, _, _ = self.model(**model_input)
                    loss = torch.mean(loss)

                    # pdb.set_trace()
                    meters['loss'].update(loss.item())
                    video_mask_old = video_mask.cpu().numpy()
                    gt_boxes_old = model_input['gt'].cpu().numpy()
                    predict_boxes_old = np.round(predict_boxes.cpu().numpy()).astype(np.int32).copy()
                    for k in range(predict_boxes.shape[0]):
                        gt_boxes = gt_boxes_old[k]
                        predict_boxes = predict_boxes_old[k]
                        video_mask = video_mask_old[k]
                        predict_flatten = predict_flatten_old[k]
                        gt_starts, gt_ends = gt_boxes[0], gt_boxes[1]
                        predict_starts, predict_ends = predict_boxes[:, 0], predict_boxes[:, 1]
                        predict_starts[predict_starts < 0] = 0
                        seq_len = np.sum(video_mask, -1)
                        # if self.args.dataset == 'ActivityNet':
                        #     seq_len = np.tile(seq_len,(1400,1))
                        #     # seq_len = seq_len[adj_mat.cpu().numpy()>0]
                        #     gt_boxes = np.tile(gt_boxes,(1400,1))
                        #     # gt_boxes = gt_boxes[adj_mat.cpu().numpy()>0]
                        # elif dataset == 'TACOS':
                        #     seq_len = np.tile(seq_len,(800,1)).transpose(1,0)
                        #     seq_len = seq_len[adj_mat.cpu().numpy()>0]
                        #     gt_boxes = np.tile(gt_boxes,(800,1,1)).transpose(1,0,2)
                        #     gt_boxes = gt_boxes[adj_mat.cpu().numpy()>0]
                        predict_ends[predict_ends >= seq_len] = seq_len - 1
                        # IoUs = criteria.calculate_IoU_batch((predict_starts, predict_ends),
                        #                                     (gt_starts, gt_ends))
                        # meters['mIoU'].update(np.mean(IoUs), IoUs.shape[0])
                        predict_flatten = predict_flatten.cpu().numpy()
                        predict_boxes[:, 0], predict_boxes[:, 1] = predict_starts, predict_ends
                        
                        #index = np.argpartition(predict_flatten,-10)[-10:]
                        topn_IoU_matric = criteria.compute_IoU_recall(predict_flatten, predict_boxes, gt_boxes)
                        pred_top5_list,scores = criteria.inference_nms(predict_flatten, predict_boxes,5)
                        meters_5['mIoU'].update(topn_IoU_matric, 1)
                        submission_dict = {"clip_uid":count,"scores":scores, "predicted_times":pred_top5_list,'tiou_5':int(topn_IoU_matric[0][2]),\
                                            'tiou_7':int(topn_IoU_matric[0][3])}
                        rst_dict.append(submission_dict)
                        # gt_dict.append({"clip_uid":count,"gt_times":gt_boxes.tolist()})
                        count += 1
                save_json(rst_dict,'/data5/yzh/MovieUN_v2/IANET/code/ia_net_preds.json')
                # save_json(gt_dict,'/data5/yzh/MovieUN_v2/IANET/code/gt_times.json')
                print('| ', end='')
                print('---------------')
                IoU_threshs = [0.1, 0.3, 0.5, 0.7]
                top_n_list = [1,5]
                topn_IoU_matric, count = meters_5['mIoU'].val, meters_5['mIoU'].count
                res_json = {}
                for i in range(2):
                    for j in range(4):
                        print('{}, {:.4f}'.format('IoU@'+str(top_n_list[i])+'@'+str(IoU_threshs[j]), topn_IoU_matric[i,j]/count), end=' | ')
                        res_json['IoU@'+str(top_n_list[i])+'@'+str(IoU_threshs[j])] = topn_IoU_matric[i,j]/count
                meters_5['mIoU'].reset()
                save_json(res_json, '/data5/yzh/MovieUN_v2/IANET/code/output/grounding/results.json')
                print()
    def eval_save(self):
        data_loaders = [self.test_loader]
        meters = collections.defaultdict(lambda: AverageMeter())
        time_meter = TimeMeter()
        f = open('./our.txt','w')
        self.model.eval()
        with torch.no_grad():
            for data_loader in data_loaders:
                for bid, (video, video_mask, words, word_mask,
                          label, scores, scores_mask, id2pos, node_mask, adj_mat) in enumerate(data_loader, 1):
                    self.optimizer.zero_grad()

                    model_input = {
                        'frames': video.cuda(),
                        'frame_mask': video_mask.cuda(), 'words': words.cuda(), 'word_mask': word_mask.cuda(),
                        'label': scores.cuda(), 'label_mask': scores_mask.cuda(), 'gt': label.cuda(),
                        'node_pos': id2pos.cuda(), 'node_mask': node_mask.cuda(), 'adj_mat': adj_mat.cuda()
                    }

                    predict_boxes, loss, _, a1, a2 = self.model(**model_input)
                    loss = torch.mean(loss)
                    time_meter.update()
                    if bid % self.args.display_n_batches == 0:
                        logging.info('%.3f seconds/batch' % (
                            1.0 / time_meter.avg
                        ))
                    meters['loss'].update(loss.item())
                    a1, a2 = a1.cpu().numpy(), a2.cpu().numpy()
                    np.save('a1.npy',a1)
                    np.save('a2.npy',a2) 
                    video_mask = video_mask.cpu().numpy()
                    gt_boxes = model_input['gt'].cpu().numpy()
                    predict_boxes = np.round(predict_boxes.cpu().numpy()).astype(np.int32)
                    gt_starts, gt_ends = gt_boxes[:, 0], gt_boxes[:, 1]
                    predict_starts, predict_ends = predict_boxes[:, 0], predict_boxes[:, 1]
                    predict_starts[predict_starts < 0] = 0
                    seq_len = np.sum(video_mask, -1)
                    predict_ends[predict_ends >= seq_len] = seq_len[predict_ends >= seq_len] - 1
                    IoUs = criteria.calculate_IoU_batch((predict_starts, predict_ends),
                                                        (gt_starts, gt_ends))
                    for kk in range(predict_starts.shape[0]):
                        f.write('IoU: '+str(IoUs[kk])+' start: '+str(predict_starts[kk])+' ends: '+str(predict_ends[kk])+' gt: '+str(gt_starts[kk])+' '+str(gt_ends[kk])+'\n')
                    meters['mIoU'].update(np.mean(IoUs), IoUs.shape[0])
                    for i in range(1, 10, 2):
                        meters['IoU@0.%d' % i].update(np.mean(IoUs >= (i / 10)), IoUs.shape[0])
                if data_loaders.index(data_loader) == 0:
                    print('--------val')
                else:
                    print('--------test')
                print('| ', end='')
                for key, value in meters.items():
                    print('{}, {:.4f}'.format(key, value.avg), end=' | ')
                    meters[key].reset()
                print()
