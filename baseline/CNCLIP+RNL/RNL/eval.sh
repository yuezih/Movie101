CUDA_VISIBLE_DEVICES=0 \
python main.py \
--dataset ActivityNet \
--feature-path YOUR/PATH/TO/VIDEO_FEATURE \
--train-data data/movie101/train_role.json \
--val-data data/movie101/val_role.json \
--test-data data/movie101/test_role.json \
--max-num-words 80 \
--max-num-nodes 80 \
--max-num-epochs 50 \
--frame-dim 1024 \
--dropout 0.2 \
--warmup-updates 300 \
--warmup-init-lr 1e-06 \
--lr 8e-4 --num-heads 4 \
--num-gcn-layers 2 \
--num-attn-layers 2 \
--weight-decay 1e-7 \
--evaluate  \
--batch-size 256 \
--model-load-path output/grounding/model-14
