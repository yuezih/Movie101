CUDA_VISIBLE_DEVICES=1 \
python transformer.py \
../results/model.json \
../results/path.json \
--eval_set tst \
--resume_file ../results/model/epoch.80.th