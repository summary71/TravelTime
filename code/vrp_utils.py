# vrp_utils.py
# ------------------------
# 유틸 함수 모음:
# - geodesic(): 거리 계산
# - get_data(): json에서 data 추출
# - load_data(): json 여러개 load
# - split_data(): train/val/test split
# ------------------------

import json
import numpy as np
from geopy.distance import geodesic as GD
import math


def GetMask(modeltype):
    vrp_data_len = 4 + 7 + 1 + 2 + 6 + 2  # 4 [0,3]: coords, 7 [4,10]: day of week onehot, 1 [11]: hour, 2 [12,13]: [geodesic, manhattan_dist], 6 [14,19]: [fromlat2, fromlng2, tolat2, tolng2, latxy, lngxy], 2 [20,21]  : [departuretime, duration]
    inputmask = [False] * vrp_data_len
    inputmask[4:12] = [True] * 8 # day of week onehot

    if modeltype & 1:  # coord
        inputmask[0:4] = [True] * 4
    if modeltype & 2:  # geodesic dist
        inputmask[12] = True
    if modeltype & 4:  # manhattan dist
        inputmask[13] = True
    if modeltype & 8:  # 2nd order coords
        inputmask[14:20] = [True] * 6

    return inputmask

def GetModelInfo(model_type, model_size):
    model_input_mask = GetMask(model_type)
    small_layer_size = 64 if model_size == 0 else 128 if model_size == 1 else 64
    large_layer_size = 128 if model_size == 0 else 256 if model_size == 1 else 128
    return model_input_mask, small_layer_size, large_layer_size

def geodesic(lat1, lon1, lat2, lon2):
    # 두 지점 간 geodesic 거리(km)
    return GD((lat1, lon1), (lat2, lon2)).km


def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the shortest possible Manhattan-style distance between two geo-coordinates.
    Inputs: latitude and longitude in degrees.
    Output: distance in kilometers.
    """
    # Path 1: north-south first, then east-west at lat2
    d1 = GD((lat1, lon1), (lat2, lon1)).km + GD((lat2, lon1), (lat2, lon2)).km
    
    # Path 2: east-west first, then north-south at lon2
    d2 = GD((lat1, lon1), (lat1, lon2)).km + GD((lat1, lon2), (lat2, lon2)).km

    return min(d1, d2)


def get_data(fname, isInputNorm=False):
    # json 1개 파일에서 data 추출
    data = []
    with open(fname, 'r') as file:
        json_data = json.load(file)
    
    for item in json_data:
        # 위치 정보
        fromlat = item["from"]["lat"]
        fromlng = item["from"]["lng"]
        tolat = item["to"]["lat"]
        tolng = item["to"]["lng"]

        # Filtering
        if fromlat == tolat and fromlng == tolng: continue
        if fromlat < 33.6 or tolat < 33.6: continue
        if fromlng > 130 or tolng > 130: continue
        dist = geodesic(fromlat, fromlng, tolat, tolng)
        if dist < 0.01: continue

        # 시간 정보
        starttime = int(item["departureTime"] / 1000 / 3600) % (24 * 7)
        dow = starttime // 24
        hour = starttime % 24
        hour = hour / 24.0
        duration = int((item["arrivalTime"] - item["departureTime"]) / 1000)

        # if duration is less than 10 seconds, skip
        if duration < 10: continue

        # One-hot encoding for dow
        dow_onehot = [0] * 7
        dow_onehot[dow] = 1


        # # Feature 구성
        # if model_type == 1:
        #     row = [fromlat, fromlng, tolat, tolng] + dow_onehot + [hour_norm] + [item["departureTime"], duration]
        # elif model_type == 2:
        #     row = dow_onehot + [hour_norm] + [dist, item["departureTime"], duration]
        # elif model_type == 3:
        #     row = [fromlat, fromlng, tolat, tolng] + dow_onehot + [hour_norm] + [dist, item["departureTime"], duration]        

        manhattan_dist = manhattan_distance(fromlat, fromlng, tolat, tolng)

        # normalization 수행, 2025.6.25
        if isInputNorm:
            fromlat = (fromlat - 33) / (43 - 33)
            fromlng = (fromlng - 125) / (135 - 125)
            tolat = (tolat - 33) / (43 - 33)
            tolng = (tolng - 125) / (135 - 125)
            dist = dist / 300.0
            manhattan_dist = manhattan_dist / 500.0

        fromlat2 = fromlat * fromlat
        fromlng2 = fromlng * fromlng
        tolat2 = tolat * tolat
        tolng2 = tolng * tolng
        latxy = fromlat* tolat
        lngxy = fromlng * tolng


        row = [fromlat, fromlng, tolat, tolng] + dow_onehot + [hour] + [dist, manhattan_dist, fromlat2, fromlng2, tolat2, tolng2, latxy, lngxy] + [item["departureTime"], duration]        
                
        data.append(row)

    return data

def load_data(json_files, isInputNorm=False):
    # 여러 json 파일에서 전체 data 로드
    all_data = []
    for fcnt, fname in enumerate(json_files):
        if fcnt % max(len(json_files) // 25, 1) == 0:
            print(f"Reading file {fcnt+1}/{len(json_files)}")
        all_data += get_data(fname, isInputNorm)
    print(f"Finished reading {len(json_files)} files.")
    return np.array(sorted(all_data, key=lambda row: row[-2]))

def split_data(data):
    # Train/val/test split
    data = np.array(sorted(data, key=lambda row: row[-2]))
    num_rows = data.shape[0]

    train_end = int(0.7 * num_rows)
    val_end = int(0.85 * num_rows)

    train_data = data[:train_end, :]
    val_data = data[train_end:val_end, :]
    test_data = data[val_end:, :]

    print(f"Data split: Total {num_rows}, Train {len(train_data)}, Validation {len(val_data)}, Test {len(test_data)}")
    return train_data, val_data, test_data


def pair_key(row):
    from_member = (row[0], row[1])
    to_member = (row[2], row[3])
    return tuple(sorted([from_member, to_member]))

def unseen_testdata(train_data, val_data, test_data):
    train_pairs = set(pair_key(row) for row in train_data)
    val_pairs_seen = set(pair_key(row) for row in val_data)
    combined_pairs = train_pairs.union(val_pairs_seen)

    unseen_test_data = []
    for row in test_data:
        key = pair_key(row)
        if key not in combined_pairs:
            unseen_test_data.append(row)

    print(f"unseen test rows: {len(unseen_test_data)}")
    return np.array(unseen_test_data)

def split_into_groups(data, group_boundary):
    groups = [[] for _ in range(len(group_boundary) + 1)]
    for row in data:
        dist = geodesic(row[0], row[1], row[2], row[3])
        inserted = False
        for i, boundary in enumerate(group_boundary):
            if dist < boundary:
                groups[i].append(row)
                inserted = True
                break
        if not inserted:
            groups[-1].append(row)

    for i in range(len(groups)):
        if i == 0:
            print(f"Test Group {i+1} [0,{group_boundary[0]}): {len(groups[i])}")
        elif i < len(group_boundary):
            print(f"Test Group {i+1} [{group_boundary[i-1]},{group_boundary[i]}): {len(groups[i])}")
        else:
            print(f"Test Group {i+1} [{group_boundary[-1]},∞): {len(groups[i])}")

    return tuple(np.array(g) for g in groups)
