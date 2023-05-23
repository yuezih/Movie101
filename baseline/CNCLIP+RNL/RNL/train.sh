CUDA_VISIBLE_DEVICES=7 \
python main.py \
--dataset ActivityNet \
--feature-path /data5/yzh/DATASETS/MovieUN/feature \
--train-data /data5/yzh/MovieUN_v2/MovieUN-G/grounding/train_role.json \
--val-data /data5/yzh/MovieUN_v2/MovieUN-G/grounding/val_role.json \
--test-data /data5/yzh/MovieUN_v2/MovieUN-G/grounding/test_role.json \
--model-saved-path output/grounding_novideoface \
--max-num-words 80 \
--max-num-nodes 80 \
--max-num-epochs 50 \
--frame-dim 1024 \
--d-model 512 \
--dropout 0.2 \
--lr 8e-4 --num-heads 4 \
--warmup-updates 300 \
--warmup-init-lr 1e-06 \
--num-gcn-layers 2 \
--num-attn-layers 2 \
--weight-decay 1e-7 \
--train \
--batch-size 64

# python main.py --dataset ActivityNet --feature-path /data4/zq/Movies_dataset/code/gen_anno/video_shots/new_window_feat/v_feat_nocrop_s3d --train-data /data4/zq/Movies_dataset/code/gen_anno/video_shots/new_window_feat/train_anno_file.json --val-data /data4/zq/Movies_dataset/code/gen_anno/video_shots/new_window_feat/val_anno_file.json --test-data /data4/zq/Movies_dataset/code/gen_anno/video_shots/new_window_feat/tst_anno_file.json --max-num-epochs 20 --dropout 0.2 --warmup-updates 300 --warmup-init-lr 1e-06 --lr 8e-4 --num-heads 4 --num-gcn-layers 2 --num-attn-layers 2 --weight-decay 1e-7 --train  --model-saved-path models_s3d_face --text-feature-path /data4/zq/Movies_dataset/code/gen_anno/video_shots/new_window_feat/t_feat