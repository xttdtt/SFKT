import os
import numpy as np
from scipy import sparse
from HyperParameter import *

datafolder = os.path.join(dataset, 'Data')
modelfolder = os.path.join(dataset, 'Model')

if dataset == "Assist09":
    with open(os.path.join(datafolder, 'max_skill_len.txt'), 'r') as f:
        max_skill_len = eval(f.read())
else:
    max_skill_len = 1

pro_diff_true = sparse.load_npz(os.path.join(dataset, 'Data/pro_diff_sparse.npz')).toarray()
skill_diff_true = sparse.load_npz(os.path.join(dataset, 'Data/skill_diff_sparse.npz')).toarray()
stu_pro_skill_corr = np.load(os.path.join(datafolder, "stu_pro_skill_corr.npz"), allow_pickle=True)
stu_pro_skill_corr = np.array(stu_pro_skill_corr["stu_pro_skill_corr"])
data_num = len(stu_pro_skill_corr)
stu_id = stu_pro_skill_corr[:, 0]
pro_id = stu_pro_skill_corr[:, 1]
skill_id = stu_pro_skill_corr[:, 2]
true_corr = stu_pro_skill_corr[:, 3]

final_pro_embed = np.load(os.path.join(modelfolder, "final_pro_embed.npz"), allow_pickle=True)["final_pro_embed"]
final_skill_embed = np.load(os.path.join(modelfolder, "final_skill_embed.npz"), allow_pickle=True)["final_skill_embed"]
final_stu_embed = np.load(os.path.join(modelfolder, "final_stu_embed.npz"), allow_pickle=True)["final_stu_embed"]

final_joint_embed = np.zeros((data_num, 1 + max_skill_len))
for line in range(data_num):
    tmp_stu_embed = final_stu_embed[stu_id[line]]
    tmp_stu_embed = tmp_stu_embed.reshape(1, embed_dim)
    tmp_pro_embed = final_pro_embed[pro_id[line]] * pro_diff_true[:, pro_id[line]]
    tmp_pro_embed = tmp_pro_embed.reshape(1, embed_dim)
    tmp_skill_embed = np.zeros((max_skill_len, embed_dim)) if dataset == "Assist09" else final_skill_embed[skill_id[line]]
    if dataset == "Assist09":
        for index in range(len(skill_id[line])):
            tmp_skill_embed[index] = final_skill_embed[skill_id[line][index]] * skill_diff_true[:, skill_id[line][index]]
    else:
        tmp_skill_embed = final_skill_embed[skill_id[line]] * skill_diff_true[:, skill_id[line]]
        tmp_skill_embed = tmp_skill_embed.reshape(1, embed_dim)
    tmp_pro_skill_embed = np.concatenate([tmp_pro_embed, tmp_skill_embed], 0)
    tmp_stu_pro_skill = np.matmul(tmp_stu_embed, np.transpose(tmp_pro_skill_embed))
    final_joint_embed[line] = tmp_stu_pro_skill
print(final_joint_embed.shape)

final_true_corr = np.array(true_corr)
final_stu_id = np.array(stu_id)
np.savez(os.path.join(modelfolder, 'final_true_corr.npz'), final_true_corr=final_true_corr)
np.savez(os.path.join(modelfolder, 'final_stu_id.npz'), final_stu_id=final_stu_id)
np.savez(os.path.join(modelfolder, 'final_joint_embed.npz'), final_joint_embed=final_joint_embed)
