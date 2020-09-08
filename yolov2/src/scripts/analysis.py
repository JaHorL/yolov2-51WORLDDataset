import json
import os
import numpy as np
import pandas as pd 
from functools import reduce


file_path = "../result.json"


def parse_result_objs(objs, rts):
    for obj in objs:
        rt = [obj['result_type'], obj['result_prob']] +  obj['result_bbox']
        rts.append(rt)
    return rts

def parse_gt_objs(objs, gts):
    for obj in objs:
        gt = [obj['gt_type'], obj['pixel_rate'], obj['rect_rate']] + obj['gt_bbox']
        gts.append(gt)
    return gts

def parse_tps(tps):
    gts = []
    rts = []
    for frame in tps:
        for objs in frame.values():
            gts = parse_gt_objs(objs, gts)
            rts = parse_result_objs(objs, rts)
    return np.array(gts), np.array(rts)

def parse_fps(fps):
    rts = []
    for frame in fps:
        for objs in frame.values():
            rts = parse_result_objs(objs, rts)
    return np.array(rts)


def parse_fns(fns):
    gts = []
    for frame in fns:
        for objs in frame.values():
            gts = parse_gt_objs(objs, gts)
    return np.array(gts)




if __name__ == "__main__":

    with open(file_path) as fjson:
        results = json.loads(fjson.read())

    tps = parse_tps(results['tps'])
    fps = parse_fps(results['fps'])
    fns = parse_fns(results['fns'])
    print(len(tps[0]), len(fps), len(fns))
    precision = len(tps[0]) / (len(tps[0]) + len(fps))
    recall = len(tps[0]) / (len(tps[0]) + len(fns))
    print(precision, recall)