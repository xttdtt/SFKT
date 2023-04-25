import os
import time
import pandas as pd
import numpy as np
from scipy import sparse
from collections import defaultdict
from HyperParameter import *


# delete abnormal answer time
def clean_abnormal(num, mean, std):
    min_normal = mean - 2 * std
    max_normal = mean + 2 * std
    if num > max_normal or num < min_normal:
        return 1
    return 0


# calculate correlation
def correlation(set1, set2):
    union = set(set1).union(set(set2))
    intersection = set(set1).intersection(set(set2))
    return (len(intersection) / len(union))


# save data file
def save_dict(dict_name, file_name):
    with open(file_name, 'w') as f:
        f.write(str(dict_name))


# clean up dataset
def process_data(dataset, datafolder, pre_file, post_file, cols):
    starttime = time.time()
    # read dataset
    df = pd.read_csv(os.path.join(dataset, datafolder, pre_file), encoding='ISO-8859-1', low_memory=False)
    # remove abnormal skills
    df = df.dropna(subset=['skill_id'])
    # remove all scaffolding problems
    df = df[df['original'].isin([1])]
    # delete abnormal answer time
    df = df.dropna(subset=['ms_first_response'])
    df = df[df["ms_first_response"] > 0]
    # student group
    students = df.groupby(['user_id'], as_index=True)
    # delete all answer records of students with less than 5 answers
    delete_students = []
    for student in students:
        if len(student[1]) < 5:
            delete_students.append(student[0])
    df = df[~df['user_id'].isin(delete_students)]
    # extract column information
    df = df[cols]
    # extract all problem
    problems = df['problem_id'].unique()
    delete_lines = []
    for pro_id in range(len(problems)):
        # extract all records of each problem
        tmp_df = df[df['problem_id'] == problems[pro_id]]
        # extract line number
        tmp_lines = tmp_df.index
        # calculate mean and standard deviation
        mean_ms, std_ms = tmp_df["ms_first_response"].mean(), tmp_df["ms_first_response"].std()
        for line in tmp_lines:
            # extract answer time of each line
            tmp_ms = tmp_df[tmp_df.index == line]["ms_first_response"].values
            # judging whether it is an abnormal answering time
            if (clean_abnormal(tmp_ms, mean_ms, std_ms)):
                delete_lines.append(line)
    # delete record where abnormal data is located
    df = df[~df.index.isin(delete_lines)]
    # answer time changed from milliseconds to seconds
    df["ms_first_response"] /= 1000
    df.to_csv(os.path.join(dataset, datafolder, post_file))
    endtime = time.time()
    print("process_data time:", endtime - starttime)


# extract problems and students and their corresponding id
def extract_pro_stu_id(dataset, datafolder, df):
    starttime = time.time()
    # extract all problems and students
    problems, students = df['problem_id'].unique(), df['user_id'].unique()
    # extract total number of problems and students
    num_pro, num_stu = len(problems), len(students)
    # add problems and students corresponding id
    pro_id_dict, stu_id_dict = dict(zip(problems, range(num_pro))), dict(zip(students, range(num_stu)))
    save_dict(list(problems), os.path.join(dataset, datafolder, 'problems.txt'))
    save_dict(list(students), os.path.join(dataset, datafolder, 'students.txt'))
    save_dict(num_pro, os.path.join(dataset, datafolder, 'num_pro.txt'))
    save_dict(num_stu, os.path.join(dataset, datafolder, 'num_stu.txt'))
    save_dict(pro_id_dict, os.path.join(dataset, datafolder, 'pro_id_dict.txt'))
    save_dict(stu_id_dict, os.path.join(dataset, datafolder, 'stu_id_dict.txt'))
    endtime = time.time()
    print("extract_pro_stu_id time:", endtime - starttime)


# extract problem-skill relationships
def extract_pro_skill(dataset, datafolder, df):
    starttime = time.time()
    with open(os.path.join(dataset, datafolder, 'num_pro.txt'), 'r') as f:
        num_pro = eval(f.read())
    with open(os.path.join(dataset, datafolder, 'problems.txt'), 'r') as f:
        problems = eval(f.read())
    pro_skill_adj, skill_id_dict, pro_skill_dict = [], {}, defaultdict(list)
    count_skill, max_skill_len = 0, 0
    for pro_id in range(num_pro):
        # extract all records of each problem
        tmp_df = df[df['problem_id'] == problems[pro_id]]
        # extract skills corresponding to each problem
        tmp_skills = [ele for ele in tmp_df.iloc[0]['skill_id'].split('_')] if dataset == "Assist09" else tmp_df.iloc[0]['skill_id']
        # add skill id
        if dataset == "Assist09":
            max_skill_len = max(len(tmp_skills), max_skill_len)
            for skill in tmp_skills:
                if skill not in skill_id_dict:
                    skill_id_dict[skill] = count_skill
                    count_skill += 1
                # add problem-skill relationship, represented by 0 or 1
                pro_skill_adj.append([pro_id, skill_id_dict[skill], 1])
                pro_skill_dict[pro_id].append(skill_id_dict[skill])
        else:
            if tmp_skills not in skill_id_dict:
                skill_id_dict[tmp_skills] = count_skill
                count_skill += 1
            pro_skill_adj.append([pro_id, skill_id_dict[tmp_skills], 1])
            pro_skill_dict[pro_id] = skill_id_dict[tmp_skills]
    # extract total number of skills
    num_skill = len(skill_id_dict)
    if dataset == "Assist09":
        save_dict(max_skill_len, os.path.join(dataset, datafolder, 'max_skill_len.txt'))
    save_dict(num_skill, os.path.join(dataset, datafolder, 'num_skill.txt'))
    save_dict(skill_id_dict, os.path.join(dataset, datafolder, 'skill_id_dict.txt'))
    save_dict(dict(pro_skill_dict), os.path.join(dataset, datafolder, 'pro_skill_dict.txt'))
    pro_skill_adj = np.array(pro_skill_adj).astype(np.int32)
    pro_skill_sparse = sparse.coo_matrix((pro_skill_adj[:, 2].astype(np.float32), (pro_skill_adj[:, 0], pro_skill_adj[:, 1])), shape=(num_pro, num_skill))
    sparse.save_npz(os.path.join(dataset, datafolder, 'pro_skill_sparse.npz'), pro_skill_sparse)
    endtime = time.time()
    print("extract_pro_skill time:", endtime - starttime)


# extraction of problem difficulty
def extract_pro_diff(dataset, datafolder, df):
    starttime = time.time()
    with open(os.path.join(dataset, datafolder, 'num_pro.txt'), 'r') as f:
        num_pro = eval(f.read())
    with open(os.path.join(dataset, datafolder, 'problems.txt'), 'r') as f:
        problems = eval(f.read())
    pro_diff_adj = []
    for pro_id in range(num_pro):
        # extract all records of each problem
        tmp_df = df[df['problem_id'] == problems[pro_id]]
        # extract all line records where each problem is answered correctly
        tmp_df_corr = tmp_df[tmp_df["correct"] == 1]
        # calculate average correct answer time for each problem,represents answer speed
        tmp_time_pro_corr = 0
        if len(tmp_df_corr):
            tmp_time_pro_corr = tmp_df_corr["ms_first_response"].mean()
        # calculate correct answer rate for each problem,represents answer accuracy
        tmp_acc_pro_corr = len(tmp_df_corr) / len(tmp_df)
        tmp_pro_diff_adj = [0.] * 3
        tmp_pro_diff_adj[0], tmp_pro_diff_adj[1] = tmp_time_pro_corr, tmp_acc_pro_corr
        pro_diff_adj.append(tmp_pro_diff_adj)
    # normalization of answer speed
    pro_diff_adj = np.array(pro_diff_adj).astype(np.float32)
    pro_diff_adj[:, 0] = (pro_diff_adj[:, 0] - np.min(pro_diff_adj[:, 0])) / (np.max(pro_diff_adj[:, 0]) - np.min(pro_diff_adj[:, 0]))
    # calculate problem difficulty = answer accuracy / answer speed
    pro_diff_adj[:, 2] = pro_diff_adj[:, 1] / (pro_diff_adj[:, 0] + 1e-4)
    # normalization of problem difficulty
    pro_diff_adj[:, 2] = (pro_diff_adj[:, 2] - np.min(pro_diff_adj[:, 2])) / (np.max(pro_diff_adj[:, 2]) - np.min(pro_diff_adj[:, 2]))
    pro_diff_list = pro_diff_adj[:, 2]
    pro_diff_sparse = sparse.coo_matrix(pro_diff_list, shape=(1, num_pro))
    sparse.save_npz(os.path.join(dataset, datafolder, 'pro_diff_sparse.npz'), pro_diff_sparse)
    endtime = time.time()
    print("extract_pro_diff time:", endtime - starttime)


# extract student-skill relationship
def extract_stu_skill(dataset, datafolder, df):
    starttime = time.time()
    with open(os.path.join(dataset, datafolder, 'problems.txt'), 'r') as f:
        problems = eval(f.read())
    with open(os.path.join(dataset, datafolder, 'num_pro.txt'), 'r') as f:
        num_pro = eval(f.read())
    with open(os.path.join(dataset, datafolder, 'num_stu.txt'), 'r') as f:
        num_stu = eval(f.read())
    with open(os.path.join(dataset, datafolder, 'num_skill.txt'), 'r') as f:
        num_skill = eval(f.read())
    with open(os.path.join(dataset, datafolder, 'stu_id_dict.txt'), 'r') as f:
        stu_id_dict = eval(f.read())
    with open(os.path.join(dataset, datafolder, 'skill_id_dict.txt'), 'r') as f:
        skill_id_dict = eval(f.read())
    stu_skill_total_adj, stu_skill_corr_adj, stu_skill_adj = np.zeros((num_stu, num_skill)), np.zeros((num_stu, num_skill)), np.zeros((num_stu, num_skill))
    skill_corr_time, skill_corr_count = np.zeros(num_skill), np.zeros(num_skill)
    for pro_id in range(num_pro):
        tmp_df = df[df['problem_id'] == problems[pro_id]]
        tmp_skills = [ele for ele in tmp_df.iloc[0]['skill_id'].split('_')] if dataset == "Assist09" else tmp_df.iloc[0]['skill_id']
        # extract line index
        tmp_lines = tmp_df.index
        for line in tmp_lines:
            # extract student and corresponding id of each line
            tmp_stu = int(tmp_df[tmp_df.index == line]["user_id"].values)
            tmp_stu_id = stu_id_dict[tmp_stu]
            if dataset == "Assist09":
                """
                If student answers problem correctly, 
                it is considered that student correctly answers skills corresponding to this problem
                """
                if tmp_df[tmp_df.index == line]["correct"].values == 1:
                    for skill in tmp_skills:
                        skill_corr_count[skill_id_dict[skill]] += 1
                        skill_corr_time[skill_id_dict[skill]] += tmp_df[tmp_df.index == line]["ms_first_response"].values
                        stu_skill_corr_adj[tmp_stu_id][skill_id_dict[skill]] += 1
                        """
                        Regardless of whether student answer problem correctly or not,
                        number of times the student answer skills corresponding to this problem will be increased by 1
                        """
                        stu_skill_total_adj[tmp_stu_id][skill_id_dict[skill]] += 1
                else:
                    for skill in tmp_skills:
                        stu_skill_total_adj[tmp_stu_id][skill_id_dict[skill]] += 1
            else:
                if tmp_df[tmp_df.index == line]["correct"].values == 1:
                    skill_corr_count[skill_id_dict[skill]] += 1
                    skill_corr_time[skill_id_dict[skill]] += tmp_df[tmp_df.index == line]["ms_first_response"].values
                    stu_skill_corr_adj[tmp_stu_id][skill_id_dict[tmp_skills]] += 1
                    stu_skill_total_adj[tmp_stu_id][skill_id_dict[tmp_skills]] += 1
                else:
                    stu_skill_total_adj[tmp_stu_id][skill_id_dict[tmp_skills]] += 1
    # calculate total number of times skills were answered
    skill_total_list = np.sum(stu_skill_total_adj, 0)
    # calculate total number of times skills were answered correctly
    skill_corr_list = np.sum(stu_skill_corr_adj, 0)
    # calculate skill correct answer rate = total number of correct answers / total number of answers
    skill_diff_acc = np.zeros(num_skill)
    for i in range(num_skill):
        if skill_total_list[i]:
            skill_diff_acc[i] = skill_corr_list[i] / skill_total_list[i]
    # calculate the average correct answer time of skills = total correct answer time / total number of answers
    skill_diff_time = np.zeros(num_skill)
    for i in range(num_skill):
        if skill_corr_count[i]:
            skill_diff_time[i] = skill_corr_time[i] / skill_corr_count[i]
    # normalization of skill answer time
    skill_diff_time = (skill_diff_time - np.min(skill_diff_time)) / (np.max(skill_diff_time) - np.min(skill_diff_time))
    # calculate skill difficulty = correct answer rate / answer time
    skill_diff_list = np.zeros(num_skill)
    for i in range(num_skill):
        if skill_diff_time[i]:
            skill_diff_list[i] = skill_diff_acc[i] / skill_diff_time[i]
    # normalization of skill difficulty
    skill_diff_list = (skill_diff_list - np.min(skill_diff_list)) / (np.max(skill_diff_list) - np.min(skill_diff_list))
    """
    calculate  degree of students' mastery of skills = 
    number of times students answer correctly for skills / number of times students answer skills
    """
    for i in range(num_stu):
        for j in range(num_skill):
            if stu_skill_total_adj[i][j]:
                stu_skill_adj[i][j] = stu_skill_corr_adj[i][j] / stu_skill_total_adj[i][j]
    skill_diff_sparse = sparse.coo_matrix(skill_diff_list, shape=(1, num_skill))
    sparse.save_npz(os.path.join(dataset, datafolder, 'skill_diff_sparse.npz'), skill_diff_sparse)
    stu_skill_sparse = sparse.coo_matrix(stu_skill_adj, shape=(num_stu, num_skill))
    sparse.save_npz(os.path.join(dataset, datafolder, 'stu_skill_sparse.npz'), stu_skill_sparse)
    endtime = time.time()
    print("extract_stu_skill time:", endtime - starttime)


# extract problem-problem, skill-skill relationships
def extract_pro_skill_correlation(dataset, datafolder):
    starttime = time.time()
    pro_skill_coo = sparse.load_npz(os.path.join(dataset, datafolder, "pro_skill_sparse.npz"))
    [num_pro, num_skill] = pro_skill_coo.toarray().shape
    pro_skill_csc = pro_skill_coo.tocsc()
    pro_skill_csr = pro_skill_coo.tocsr()
    """1. extract problem-problem relationships"""
    pro_pro_adj, temp_pro_pro_simi = [], []
    for pro_index in range(num_pro):
        # extract skills corresponding to problem
        tmp_skills = pro_skill_csr.getrow(pro_index).indices
        # extract problems set with same skills as this problem
        similar_pros = pro_skill_csc[:, tmp_skills].indices
        # remove duplicate elements of problems set
        similar_pros = list(set(similar_pros))
        # save problem-problem relationship element position
        zipped = zip([pro_index] * len(similar_pros), similar_pros)
        pro_pro_adj += list(zipped)
        for pro in similar_pros:
            # extract the skills corresponding to each problem in problems set
            tmp_pro_skills = pro_skill_csr.getrow(pro).indices
            # calculate similarity
            temp_pro_pro_simi.append(correlation(tmp_skills, tmp_pro_skills))
    pro_pro_adj = np.array(pro_pro_adj).astype(np.int32)
    pro_pro_sparse = sparse.coo_matrix((temp_pro_pro_simi, (pro_pro_adj[:, 0], pro_pro_adj[:, 1])), shape=(num_pro, num_pro))
    sparse.save_npz(os.path.join(dataset, datafolder, 'pro_pro_sparse.npz'), pro_pro_sparse)
    """2. extract skill-skill relationships"""
    skill_skill_adj, temp_skill_skill_simi = [], []
    for skill_index in range(num_skill):
        tmp_pros = pro_skill_csc.getcol(skill_index).indices
        similar_skills = pro_skill_csr[tmp_pros, :].indices
        similar_skills = list(set(similar_skills))
        zipped = zip([skill_index] * len(similar_skills), similar_skills)
        skill_skill_adj += list(zipped)
        for skill in similar_skills:
            tmp_skill_pros = pro_skill_csc.getcol(skill).indices
            temp_skill_skill_simi.append(correlation(tmp_pros, tmp_skill_pros))
    skill_skill_adj = np.array(skill_skill_adj).astype(np.int32)
    skill_skill_sparse = sparse.coo_matrix((temp_skill_skill_simi, (skill_skill_adj[:, 0], skill_skill_adj[:, 1])), shape=(num_skill, num_skill))
    sparse.save_npz(os.path.join(dataset, datafolder, 'skill_skill_sparse.npz'), skill_skill_sparse)
    endtime = time.time()
    print("extract_pro_skill_correlation time:", endtime - starttime)


"""
extract each line record's students, problems, skills and answers, and convert them into corresponding id for storage  
If dataset is Assist12, this part may take 7 hours to 8 hours to complete
"""
def extract_stu_pro_skill_corr(dataset, datafolder, df):
    starttime = time.time()
    with open(os.path.join(dataset, datafolder, 'pro_id_dict.txt'), 'r') as f:
        pro_id_dict = eval(f.read())
    with open(os.path.join(dataset, datafolder, 'pro_skill_dict.txt'), 'r') as f:
        pro_skill_dict = eval(f.read())
    with open(os.path.join(dataset, datafolder, 'stu_id_dict.txt'), 'r') as f:
        stu_id_dict = eval(f.read())
    df1 = df[["user_id", "problem_id", "skill_id", "correct"]]
    df1["problem_id"] = df1["problem_id"].map(lambda pro: pro_id_dict[pro])
    df1["skill_id"] = df1["problem_id"].map(lambda pro: pro_skill_dict[pro])
    df1["user_id"] = df1["user_id"].map(lambda stu: stu_id_dict[stu])

    stu_pro_skill_corr = []
    for line in df1.index:
        tmp_stu = int(df1[df1.index == line]["user_id"].values)
        tmp_pro = int(df1[df1.index == line]["problem_id"].values)
        tmp_skill = df1[df1.index == line]["skill_id"].values.tolist().pop(0)
        tmp_corr = int(df1[df1.index == line]["correct"].values)
        tmp_stu_pro_skill_corr = [tmp_stu, tmp_pro, tmp_skill, tmp_corr]
        stu_pro_skill_corr.append(tmp_stu_pro_skill_corr)
    np.savez(os.path.join(dataset, datafolder, 'stu_pro_skill_corr.npz'), stu_pro_skill_corr=stu_pro_skill_corr)
    endtime = time.time()
    print("extract_stu_pro_skill_corr time:", endtime - starttime)


if __name__ == '__main__':
    starttime = time.time()
    datafolder = "Data"
    pre_file = dataset + "_original.csv"
    post_file = dataset + ".csv"
    cols = ["user_id", "problem_id", "skill_id", "correct", "ms_first_response"]
    process_data(dataset, datafolder, pre_file, post_file, cols)
    df = pd.read_csv(os.path.join(dataset, datafolder, post_file), encoding="ISO-8859-1", low_memory=True)
    extract_pro_stu_id(dataset, datafolder, df)
    extract_pro_skill(dataset, datafolder, df)
    extract_pro_diff(dataset, datafolder, df)
    extract_stu_skill(dataset, datafolder, df)
    extract_pro_skill_correlation(dataset, datafolder)
    extract_stu_pro_skill_corr(dataset, datafolder, df)
    endtime = time.time()
    print("total time :", endtime - starttime)
