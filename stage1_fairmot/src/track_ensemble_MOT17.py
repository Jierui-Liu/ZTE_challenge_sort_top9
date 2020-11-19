from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts
from os.path import join,dirname,realpath


def write_results(filename, results, data_type):
    if data_type == 'mot':
        # save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,0\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            # if data_type == 'kitti':
            #     frame_id -= 1
            # for tlwh, track_id in zip(tlwhs, track_ids):
            #     if track_id < 0:
            #         continue
            #     x1, y1, w, h = tlwh
            #     x2, y2 = x1 + w, y1 + h
            #     line = save_format.format(frame=frame_id, id=track_id, \
            #                             x1=int(round(x1)), y1=int(round(y1)), \
            #                             x2=int(round(x2)), y2=int(round(y2)), \
            #                             w=int(round(w)), h=int(round(h)))
            #     f.write(line)
            if frame_id==2:
                for tlwh, track_id in zip(tlwhs, track_ids):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    line = save_format.format(frame=1, id=track_id, \
                                            x1=int(round(x1)), y1=int(round(y1)), \
                                            x2=int(round(x2)), y2=int(round(y2)), \
                                            w=int(round(w)), h=int(round(h)))
                    f.write(line)
                for tlwh, track_id in zip(tlwhs, track_ids):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    line = save_format.format(frame=frame_id, id=track_id, \
                                            x1=int(round(x1)), y1=int(round(y1)), \
                                            x2=int(round(x2)), y2=int(round(y2)), \
                                            w=int(round(w)), h=int(round(h)))
                    f.write(line)
            else:
                if data_type == 'kitti':
                    frame_id -= 1
                for tlwh, track_id in zip(tlwhs, track_ids):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    line = save_format.format(frame=frame_id, id=track_id, \
                                            x1=int(round(x1)), y1=int(round(y1)), \
                                            x2=int(round(x2)), y2=int(round(y2)), \
                                            w=int(round(w)), h=int(round(h)))
                    f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    for path, img, img0 in dataloader:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:06d}.jpg'.format(frame_id)), online_im)
            # cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format(frame_id+1)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    data_type = 'mot'
    result_root = '/home/liujierui/proj/deep_sort_pytorch-master/demo/A-track/ensemble_MOT17_0.75'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        
    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    # jr
    opt.val_mot17=True
    exp_name=''

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, '/home/liujierui/proj/Dataset/MOT17/test')
        #  jr
        # seqs_str = '''Track1
        #               Track4
        #               Track5
        #               Track9
        #               Track10'''
        # data_root = os.path.join(opt.data_dir, '/home/liujierui/proj/Dataset/A-data')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-FRCNN
                      MOT17-04-FRCNN
                      MOT17-05-FRCNN
                      MOT17-09-FRCNN
                      MOT17-10-FRCNN
                      MOT17-11-FRCNN
                      MOT17-13-FRCNN'''
        data_root = os.path.join(opt.data_dir, '/home/liujierui/proj/Dataset/MOT17/train')
    if opt.val_mot15:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]



    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name=exp_name,
         show_image=False,
         save_images=False,
         save_videos=True)
