import argparse
import logging
import os
from optimizer.lr_scheduler import LR_SCHEDULER_REGISTRY

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--dataset', choices=['ActivityNet', 'TACOS', 'Charades'], default='ActivityNet',
                        help='')
    parser.add_argument('--train-data', type=str,
                        default=None,
                        help='')
    parser.add_argument('--val-data', type=str, default=None,
                        help='')
    parser.add_argument('--test-data', type=str, default=None,
                        help='')
    parser.add_argument('--word2vec-path', type=str, default='glove_model.bin',
                        help='')
    parser.add_argument('--feature-path', type=str, default='data/activity-c3d',
                        help='')
    parser.add_argument('--text-feature-path', type=str, default='data/activity-c3d',
                        help='')
    parser.add_argument('--model-saved-path', type=str, default='results/model_%s',
                        help='')
    parser.add_argument('--model-load-path', type=str, default='',
                        help='')
    parser.add_argument('--max-num-words', type=int, default=80,
                        help='')
    parser.add_argument('--max-num-nodes', type=int, default=80,
                        help='')
    parser.add_argument('--max-num-frames', type=int, default=200,
                        help='')
    parser.add_argument('--d-model', type=int, default=512,
                        help='')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='')
    parser.add_argument('--word-dim', type=int, default=768,
                        help='')
    parser.add_argument('--frame-dim', type=int, default=1024,
                        help='')
    parser.add_argument('--num-gcn-layers', type=int, default=2,
                        help='')
    parser.add_argument('--num-attn-layers', type=int, default=2,
                        help='')
    parser.add_argument('--display-n-batches', type=int, default=50,
                        help='')
    parser.add_argument('--max-num-epochs', type=int, default=20,
                        help='')
    parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                       help='weight decay')
    parser.add_argument('--lr-scheduler', default='inverse_sqrt',
                 choices=LR_SCHEDULER_REGISTRY.keys(),
                 help='Learning Rate Scheduler')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    from optimizer.lr_scheduler.inverse_square_root_schedule import InverseSquareRootSchedule
    InverseSquareRootSchedule.add_args(parser)
    from optimizer.adam_optimizer import AdamOptimizer
    AdamOptimizer.add_args(parser)
    return parser.parse_args()


def main(args):
    print(args)
    from runners.runner_final import Runner
    runner = Runner(args)
    if args.train:
        runner.train()
    if args.evaluate:
        runner.eval_new()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.model_saved_path):
        os.makedirs(args.model_saved_path)
    logging.basicConfig(filename= os.path.join(args.model_saved_path,'TSG.txt'), level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main(args)
