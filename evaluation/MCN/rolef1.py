import json

role_file = json.load(open('rolef1_data/meta_anno.json', 'r', encoding='utf-8'))
role_gt = json.load(open('rolef1_data/test_pr.json', 'r', encoding='utf-8'))

# Easily misidentified role names
banned = ['司机', '淡定', '大海', '火山', '主持人', '琉璃', '李小龙', '莎莎', '老牛', '小河', '小方', '时分']


preds_path = f'YOUR/PATH/TO/RESULTS'


preds = json.load(open(preds_path, 'r', encoding='utf-8'))
role_list = []
movie_role_dict = {}
for movie_id in role_file.keys():
    movie_roles = role_file[movie_id]
    movie_role_dict[movie_id] = []
    for role in movie_roles.values():
        role_list.append(role['rolename'])
        movie_role_dict[movie_id].append(role['rolename'])

P = []
R = []
F1 = []

for nid in preds.keys():
    sentence = preds[nid]['sentences'][0]
    pred_roles = []
    for role in role_list:
        if role in sentence:
            if role in banned:
                continue
            pred_roles.append(role)
            
    gt_roles = role_gt[nid]['roles']
    if len(gt_roles) == 0:
        P.append(0)
        R.append(0)
        F1.append(0)
        continue
    if len(pred_roles) == 0:
        assert len(gt_roles) != 0
        P.append(0)
        R.append(0)
        F1.append(0)
        continue

    p = len(set(pred_roles) & set(gt_roles)) / len(set(pred_roles))
    r = len(set(pred_roles) & set(gt_roles)) / len(set(gt_roles))
    if p + r == 0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)

    P.append(p)
    R.append(r)
    F1.append(f1)

P = sum(P) / len(P)
R = sum(R) / len(R)
F1 = sum(F1) / len(F1)

print(f'P: {P:.3f}\tR: {R:.3f}\tF1: {F1:.3f}')