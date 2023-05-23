import json

preds = json.load(open('/data5/yzh/MovieUN_v2/IANET/code/ia_net_preds.json', 'r'))
gts = json.load(open(f'/data5/yzh/MovieUN_v2/MovieUN-G/grounding/test_movie.json', 'r'))

# 计算overlap
if __name__ == '__main__':
    assert len(preds) == len(gts)
    overlap_dict = {}
    for vid in range(len(preds)):
        pred = preds[vid]['predicted_times'][0]
        gt = [gts[vid]['start']-gts[vid]['clip_head'], gts[vid]['end']-gts[vid]['clip_head']]
        overlap = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
        overlap_dict[vid] = overlap

    # sort
    sorted_overlap_dict = sorted(overlap_dict.items(), key=lambda x: x[1], reverse=True)
    # print top 6
    for i in range(6):
        vid = sorted_overlap_dict[i][0]
        print(f'========= gt {vid} =========')
        print(f'start: {gts[vid]["start"]}')
        print(f'end: {gts[vid]["end"]}')
        print(f'narration: {gts[vid]["narration"]}')
        print(f'========= pred {vid} =========')
        print(f'anchor: {gts[vid]["clip_head"]+90}, {gts[vid]["clip_head"]+110}')
        print(f'start: {preds[vid]["predicted_times"][0][0]+gts[vid]["clip_head"]}')
        print(f'end: {preds[vid]["predicted_times"][0][1]+gts[vid]["clip_head"]}')
        print()