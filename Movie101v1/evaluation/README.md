# Evaluation

## MCN Evaluation

### EMScore

EMScore is calculated by our fine-tuned CNCLIP model. 

After you prepared the video and text feature for CNCLIP, our scripts helps to calculate the final EMScore:

```bash
python -u MCN/emscore.py \
--image-feats="test_imgs.img_feat.jsonl" \
--text-feats="test_texts.txt_feat.jsonl"
```

### BERTScore

The BERTScore is calculated with [bert_score](https://github.com/Tiiiger/bert_score). Please follow the offical instructions to install the bert_score package. 

Then, we provide a script to calculate the BERTScore of your generated narration. 

```bash
python MCN/bertscore.py
```

### RoleF1

```bash
python MCN/rolef1.py
```

### MNScore

After obtaining the average EMScore, BERTScore and RoleF1, we can calculate the final MNScore:

$MNScore = \frac{1 \cdot EMScore + 4 \cdot BERTScore + 1 \cdot RoleF1}{6}$