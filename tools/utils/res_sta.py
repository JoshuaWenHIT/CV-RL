SEQ = {"seq1_train_4c": ["uav0000079_00480_v", "uav0000084_00000_v"],
       "seq1_val_4c": ["uav0000086"],
       "seq2_train_9c": ["uav0000150_02310_v", "uav0000222_03150_v", "uav0000316_01288_v", "uav0000357_00920_v",
                         "uav0000361_02323_v"],
       "seq2_val_9c": ["uav0000137", "uav0000182"],
       "seq3_train_6c": ["uav0000326_01035_v"],
       "seq3_val_6c": ["uav0000305"],
       "seq4_train_4c": ["uav0000145_00000_v", "uav0000218_00001_v", "uav0000263_03289_v", "uav0000266_03598_v"],
       "seq4_val_4c": ["uav0000268"],
       }
res = {
    "seq1": {
        "a0": {"n": 0, "r": 0},
        "a1": {"n": 0, "r": 0},
        "a2": {"n": 0, "r": 0},
        "a3": {"n": 0, "r": 0},
        "a4": {"n": 0, "r": 0}
    },
    "seq2": {
        "a0": {"n": 0, "r": 0},
        "a1": {"n": 0, "r": 0},
        "a2": {"n": 0, "r": 0},
        "a3": {"n": 0, "r": 0},
        "a4": {"n": 0, "r": 0}
    },
    "seq3": {
        "a0": {"n": 0, "r": 0},
        "a1": {"n": 0, "r": 0},
        "a2": {"n": 0, "r": 0},
        "a3": {"n": 0, "r": 0},
        "a4": {"n": 0, "r": 0}
    },
    "seq4": {
        "a0": {"n": 0, "r": 0},
        "a1": {"n": 0, "r": 0},
        "a2": {"n": 0, "r": 0},
        "a3": {"n": 0, "r": 0},
        "a4": {"n": 0, "r": 0}
    }
}

log_txt = "/home/joshuawen/WorkSpace/CV-RL/RL/DQN/runs/CV-RL__dqn__1__1678673363/log.txt"
step_s = 477728
step_e = 499947

with open(log_txt, 'r') as f:
    for line in f.readlines():
        line = line.rstrip('\n')
        line_list_pre = [i for i in line.split('-')]
        # print(line_list_pre)
        line_list = [j for j in line_list_pre[-1].split(',')]
        if len(line_list) != 4:
            continue
        print("line_list: ", line_list)
        step = int(line_list[0].split('=')[-1])
        seq = line_list[1].split('=')[-1].split('_')[0]
        a = int(line_list[2].split('=')[-1])
        r = int(line_list[3].split('=')[-1])
        if step_s <= step <= step_e:
            if seq in SEQ["seq1_val_4c"]:
                if a == 0:
                    res["seq1"]["a0"]["n"] += 1
                    res["seq1"]["a0"]["r"] += r
                if a == 1:
                    res["seq1"]["a1"]["n"] += 1
                    res["seq1"]["a1"]["r"] += r
                if a == 2:
                    res["seq1"]["a2"]["n"] += 1
                    res["seq1"]["a2"]["r"] += r
                if a == 3:
                    res["seq1"]["a3"]["n"] += 1
                    res["seq1"]["a3"]["r"] += r
                if a == 4:
                    res["seq1"]["a4"]["n"] += 1
                    res["seq1"]["a4"]["r"] += r
            if seq in SEQ["seq2_val_9c"]:
                if a == 0:
                    res["seq2"]["a0"]["n"] += 1
                    res["seq2"]["a0"]["r"] += r
                if a == 1:
                    res["seq2"]["a1"]["n"] += 1
                    res["seq2"]["a1"]["r"] += r
                if a == 2:
                    res["seq2"]["a2"]["n"] += 1
                    res["seq2"]["a2"]["r"] += r
                if a == 3:
                    res["seq2"]["a3"]["n"] += 1
                    res["seq2"]["a3"]["r"] += r
                if a == 4:
                    res["seq2"]["a4"]["n"] += 1
                    res["seq2"]["a4"]["r"] += r
            if seq in SEQ["seq3_val_6c"]:
                if a == 0:
                    res["seq3"]["a0"]["n"] += 1
                    res["seq3"]["a0"]["r"] += r
                if a == 1:
                    res["seq3"]["a1"]["n"] += 1
                    res["seq3"]["a1"]["r"] += r
                if a == 2:
                    res["seq3"]["a2"]["n"] += 1
                    res["seq3"]["a2"]["r"] += r
                if a == 3:
                    res["seq3"]["a3"]["n"] += 1
                    res["seq3"]["a3"]["r"] += r
                if a == 4:
                    res["seq3"]["a4"]["n"] += 1
                    res["seq3"]["a4"]["r"] += r
            if seq in SEQ["seq4_val_4c"]:
                if a == 0:
                    res["seq4"]["a0"]["n"] += 1
                    res["seq4"]["a0"]["r"] += r
                if a == 1:
                    res["seq4"]["a1"]["n"] += 1
                    res["seq4"]["a1"]["r"] += r
                if a == 2:
                    res["seq4"]["a2"]["n"] += 1
                    res["seq4"]["a2"]["r"] += r
                if a == 3:
                    res["seq4"]["a3"]["n"] += 1
                    res["seq4"]["a3"]["r"] += r
                if a == 4:
                    res["seq4"]["a4"]["n"] += 1
                    res["seq4"]["a4"]["r"] += r
print("res: ", res)
