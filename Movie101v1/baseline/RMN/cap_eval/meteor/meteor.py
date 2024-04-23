#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help 

import os
import sys
import subprocess
import threading

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'
# print METEOR_JAR

class Meteor:

  def __init__(self):
    self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, \
        '-', '-', '-stdio', '-l', 'en', '-norm']
    self.meteor_p = subprocess.Popen(self.meteor_cmd, \
        cwd=os.path.dirname(os.path.abspath(__file__)), \
        stdin=subprocess.PIPE, \
        stdout=subprocess.PIPE, \
        stderr=subprocess.PIPE)
    # Used to guarantee thread safety
    self.lock = threading.Lock()

  def compute_score(self, gts, res, vid_order=None):
    # assert(gts.keys() == res.keys())
    if vid_order is None:
      vid_order = gts.keys()
    scores = []

    eval_line = 'EVAL'
    self.lock.acquire()
    for i in vid_order:
      assert(len(res[i]) == 1)
      stat = self._stat(res[i][0], gts[i])
      eval_line += ' ||| {}'.format(stat)

    self.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    self.meteor_p.stdin.flush()
    for i in range(0,len(vid_order)):
      scores.append(float(self.meteor_p.stdout.readline().strip()))
    score = float(self.meteor_p.stdout.readline().strip())
    self.lock.release()

    return score, scores

  def method(self):
    return "METEOR"

  def _stat(self, hypothesis_str, reference_list):
    # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
    hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
    score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
    self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
    self.meteor_p.stdin.flush()
    return self.meteor_p.stdout.readline().decode().strip()

  def _score(self, hypothesis_str, reference_list):
    self.lock.acquire()
    # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
    hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
    score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
    self.meteor_p.stdin.write('{}\n'.format(score_line))
    self.meteor_p.stdin.flush()
    stats = self.meteor_p.stdout.readline().strip()
    eval_line = 'EVAL ||| {}'.format(stats)
    # EVAL ||| stats 
    self.meteor_p.stdin.write('{}\n'.format(eval_line))
    self.meteor_p.stdin.flush()
    score = float(self.meteor_p.stdout.readline().strip())
    # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
    # thanks for Andrej for pointing this out
    score = float(self.meteor_p.stdout.readline().strip())
    self.lock.release()
    return score
 
  def __exit__(self):
    self.lock.acquire()
    self.meteor_p.stdin.close()
    self.meteor_p.wait()
    self.lock.release()


import numpy as np
import multiprocessing

def producer_fn(q, scorer, gts, res, vid_order):
  _, ss = scorer.compute_score(gts, res, vid_order=vid_order)
  vid_ss = {}
  for vid, s in zip(vid_order, ss):
    vid_ss[vid] = s
  q.put(vid_ss)

class MeteorMulti(object):
  def __init__(self, num_process=4):
    self.num_process = num_process
    self.scorers = []
    for i in xrange(num_process):
      self.scorers.append(Meteor())

  def compute_score(self, gts, res, vid_order=None):
    if vid_order is None:
      vid_order = gts.keys()
    num_vid = len(vid_order)
    num_split = min(self.num_process, num_vid)
    split_idxs = np.linspace(0, num_vid, num_split+1).astype(np.int32)

    q = Queue(num_split)
    producers = []
    for i in xrange(num_split):
      sub_vid_order = vid_order[split_idxs[i]: split_idxs[i+1]]
      sub_gts = {key: gts[key] for key in sub_vid_order}
      sub_res = {key: res[key] for key in sub_vid_order}

      producers.append(Process(target=producer_fn, 
        args=(q, self.scorers[i], sub_gts, sub_res, sub_vid_order)))
      producers[-1].start()

    vid_score = {}
    for i in xrange(num_split):
      sub_vid_ss = q.get()
      vid_score.update(sub_vid_ss)
    scores = [vid_score[vid] for vid in vid_order]

    return np.mean(scores), scores


