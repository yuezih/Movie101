from cap_eval.bleu.bleu import Bleu
from cap_eval.cider.cider import Cider
from cap_eval.meteor.meteor import Meteor

import json
import numpy as np
import pdb

# initialize the caption evaluators
meteor_scorer = Meteor()
cider_scorer = Cider()
bleu_scorer = Bleu(4)

UNK = 0

stoi = json.load(open('/data5/yzh/MovieUN_v2/MovieUN-N/ovp_data/vocab/c2id.json'))

def sent2int(str_sent):
  int_sent = [stoi.get(w, UNK) for w in str_sent]
  return int_sent

def bleu_eval(refs, cands):
  print ("calculating bleu_4 score...")
  bleu, _ = bleu_scorer.compute_score(refs, cands)
  return bleu


def cider_eval(refs, cands):
  print ("calculating cider score...")
  cider, _ = cider_scorer.compute_score(refs, cands)
  return cider


def meteor_eval(refs, cands):
  print ("calculating meteor score...")
  meteor, _ = meteor_scorer.compute_score(refs, cands)
  return meteor


def getNgrams(words_pred, unigrams, bigrams, trigrams, fourgrams):
  # N=1
  for w in words_pred:
    if w not in unigrams:
      unigrams[w] = 0
    unigrams[w] += 1
  # N=2
  for i, w in enumerate(words_pred):
    if i<len(words_pred)-1:
      w_next = words_pred[i+1]
      bigram = '%s_%s' % (w, w_next)
      if bigram not in bigrams:
        bigrams[bigram] = 0
      bigrams[bigram] += 1
  # N=3
  for i, w in enumerate(words_pred):
    if i<len(words_pred)-2:
      w_next = words_pred[i + 1]
      w_next_ = words_pred[i + 2]
      tri = '%s_%s_%s' % (w, w_next, w_next_)
      if tri not in trigrams:
        trigrams[tri] = 0
      trigrams[tri] += 1
  # N=4
  for i, w in enumerate(words_pred):
    if i<len(words_pred)-3:
      w_next = words_pred[i + 1]
      w_next_ = words_pred[i + 2]
      w_next__ = words_pred[i + 3]
      four = '%s_%s_%s_%s' % (w, w_next, w_next_, w_next__)
      if four not in fourgrams:
        fourgrams[four] = 0
      fourgrams[four] += 1
  return unigrams, bigrams, trigrams, fourgrams


def diversity(data_pred):
  div1, div2, re4 = [], [], []
  for i in range(len(data_pred)):
    unigrams, bigrams, trigrams, fourgrams = {}, {}, {}, {}
    if len(data_pred[i]) == 0:
      continue
    else:
      if data_pred[i][-1] == '.':
        para = data_pred[i].split('.')[:-1]
      else:
        para = data_pred[i].split('.')
      for j, pred_sentence in enumerate(para):
        if pred_sentence[-1] == '.':
          pred_sentence = pred_sentence[:-1]
        while len(pred_sentence) > 0 and pred_sentence[-1] == ' ':
          pred_sentence = pred_sentence[:-1]
        while len(pred_sentence) > 0 and pred_sentence[0] == ' ':
          pred_sentence = pred_sentence[1:]
        pred_sentence = pred_sentence.replace(',', ' ')
        while '  ' in pred_sentence:
          pred_sentence = pred_sentence.replace('  ', ' ')

        words_pred = pred_sentence.split(' ')
        unigrams, bigrams, trigrams, fourgrams = getNgrams(words_pred, unigrams, bigrams, trigrams, fourgrams)

    sum_unigrams = sum([unigrams[un] for un in unigrams])
    vid_div1 = float(len(unigrams)) / (float(sum_unigrams) + 1e-28)
    vid_div2 = float(len(bigrams)) / (float(sum_unigrams) + 1e-28)
    vid_re4 = float(sum([max(fourgrams[f]-1,0) for f in fourgrams])) / (float(sum([fourgrams[f] for f in fourgrams])) + 1e-28)

    div1.append(vid_div1)
    div2.append(vid_div2)
    re4.append(vid_re4)
  return np.mean(div1), np.mean(div2), np.mean(re4)


def compute(preds, names, refs):
  refcaps = {}
  candcaps = {}
  for i in range(len(preds)):
    candcaps[i] = [' '.join([c for c in preds[i]])]
    ref_sentence = refs[names[i]]['sentences'][0]
    refcaps[i] = [' '.join([c for c in ref_sentence])]
  # pdb.set_trace()
  bleu = bleu_eval(refcaps, candcaps)
  cider = cider_eval(refcaps, candcaps)
  meteor = meteor_eval(refcaps, candcaps)
  div1, div2, re4 = diversity(preds)
  scores = {'bleu_4':bleu[3], 'bleu_3':bleu[2], 'bleu_2':bleu[1], 'bleu_1':bleu[0],
            'cider':cider, 'meteor':meteor,
            'div1':div1, 'div2':div2, 're4':re4}
  return scores
