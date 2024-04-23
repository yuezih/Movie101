from __future__ import print_function
from __future__ import division

import os
import json
import datetime
import numpy as np
import glob
import pdb

import framework.configbase


def gen_common_pathcfg(path_cfg_file, is_train=False):
  path_cfg = framework.configbase.PathCfg()
  path_cfg.load(json.load(open(path_cfg_file)))

  output_dir = path_cfg.output_dir

  path_cfg.log_dir = os.path.join(output_dir, 'log')
  path_cfg.model_dir = os.path.join(output_dir, 'model')
  path_cfg.pred_dir = os.path.join(output_dir, 'pred')
  if not os.path.exists(path_cfg.log_dir):
    os.makedirs(path_cfg.log_dir)
  if not os.path.exists(path_cfg.model_dir):
    os.makedirs(path_cfg.model_dir)
  if not os.path.exists(path_cfg.pred_dir):
    os.makedirs(path_cfg.pred_dir)

  if is_train:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    path_cfg.log_file = os.path.join(path_cfg.log_dir, 'log-' + timestamp)
  else:
    path_cfg.log_file = None

  return path_cfg


def find_best_val_models(log_dir, model_dir):
  step_jsons = glob.glob(os.path.join(log_dir, 'val.step.*.json'))
  epoch_jsons = glob.glob(os.path.join(log_dir, 'val.epoch.*.json'))
  val_metrics = {}
  for i, json_name in enumerate(step_jsons + epoch_jsons):
    json_name = os.path.basename(json_name)
    scores = json.load(open(os.path.join(log_dir, json_name)))
    val_metrics[json_name] = scores
  # pdb.set_trace()
  measure_names = list(list(val_metrics.values())[0].keys())
  model_files = {}
  for measure_name in measure_names:
    if 'loss' in measure_name:
      idx = np.argmin([scores[measure_name] for _, scores in val_metrics.items()])
    else:
      idx = np.argmax([scores[measure_name] for _, scores in val_metrics.items()])
    json_name = list(val_metrics.keys())[idx]
    model_file = os.path.join(model_dir, 
      'epoch.%s.th'%(json_name.split('.')[2]) if 'epoch' in json_name \
      else 'step.%s.th'%(json_name.split('.')[2]))
    model_files.setdefault(model_file, [])
    model_files[model_file].append(measure_name)

  name2file = {'-'.join(measure_name): model_file for model_file, measure_name in model_files.items()}

  return name2file
