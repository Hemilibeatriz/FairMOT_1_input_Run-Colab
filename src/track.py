from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import csv
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
from google.colab.patches import cv2_imshow
from datetime import datetime

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts


def write_results(filename, results, data_type):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        if data_type == 'mot':
            writer.writerow(['frame', 'id', 'x1', 'y1', 'w', 'h', 'confidence', 'class', 'visibility', 'truncated', 'num_detections', 'timestamp'])
        elif data_type == 'kitti':
            writer.writerow(['frame', 'id', 'class', 'truncated', 'occluded', 'alpha', 'x1', 'y1', 'x2', 'y2', 'height', 'width','length', 'location', 'rotation_y', 'score', 'num_detections', 'timestamp'])
        else:
            raise ValueError(data_type)

        # Get the current timestamp
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                num_detect = len(tlwhs)
                if data_type == 'mot':
                    writer.writerow([frame_id, track_id, x1, y1, w, h, 1, -1, -1, -1, num_detect, ts])
                elif data_type == 'kitti':
                    writer.writerow([frame_id, track_id, 'pedestrian', 0, 0, -10, x1, y1, x2, y2, -10, -10, -10, -1000, -1000, -1000, -10, num_detect, ts])
    logger.info('save results to {}'.format(filename))


def write_results_incremental(filename, frame_id, tlwhs, track_ids, data_type):
    # Define the base filename and the file extension
    base_filename, file_extension = os.path.splitext(filename)

    # Initialize the part number
    part_number = 0

    # Helper function to get the current part filename
    def get_part_filename(base_filename, part_number):
        return f"{base_filename}_part{part_number}{file_extension}"

    # Get the initial part filename
    current_filename = get_part_filename(base_filename, part_number)

    # Check if the current part file exists and its line count
    while os.path.isfile(current_filename) and sum(1 for _ in open(current_filename)) >= 850:
        part_number += 1
        current_filename = get_part_filename(base_filename, part_number)

    # Check if the current part file needs header
    file_needs_header = not os.path.isfile(current_filename) or sum(1 for _ in open(current_filename)) == 0

    # Open the current part file for appending
    with open(current_filename, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write the header if needed
        if file_needs_header:
            if data_type == 'mot':
                writer.writerow(['frame', 'id', 'x1', 'y1', 'w', 'h', 'confidence', 'class', 'visibility', 'truncated', 'num_detections', 'timestamp'])
            elif data_type == 'kitti':
                writer.writerow(
                    ['frame', 'id', 'class', 'truncated', 'occluded', 'alpha', 'x1', 'y1', 'x2', 'y2', 'height',
                     'width', 'length', 'location', 'rotation_y', 'score', 'num_detections', 'timestamp'])
            else:
                raise ValueError(data_type)

        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Write the data rows
        num_detections = len(tlwhs)
        if data_type == 'kitti':
            frame_id -= 1
        for tlwh, track_id in zip(tlwhs, track_ids):
            if track_id < 0:
                continue
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            if data_type == 'mot':
                writer.writerow([frame_id, track_id, x1, y1, w, h, 1, -1, -1, -1, num_detections, timestamp])
            elif data_type == 'kitti':
                writer.writerow(
                    [frame_id, track_id, 'pedestrian', 0, 0, -10, x1, y1, x2, y2, -10, -10, -10, -1000, -1000, -1000,
                     -10, num_detections, timestamp])

    logger.info('Appended incremental results to {}'.format(current_filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def save_image_with_boxes(save_dir, img, online_targets, frame_idx, time_data, waiting_area, service_area, max_detect_ts, max_detect):
    text_scale = max(1, img.shape[1] / 1600.)


    # Desenhar as áreas de espera e de atendimento
    cv2.rectangle(img, (waiting_area[0], waiting_area[1]), (waiting_area[2], waiting_area[3]), (255, 0, 0), 2)
    cv2.putText(img, 'Waiting Area', (waiting_area[0], waiting_area[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 0), 2)

    cv2.rectangle(img, (service_area[0], service_area[1]), (service_area[2], service_area[3]), (0, 0, 255), 2)
    cv2.putText(img, 'Service Area', (service_area[0], service_area[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)

    # Adicionar o ID do objeto
    cv2.putText(img, f'Quantidade: {max_detect}', (5, int(85 * text_scale)), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), thickness=2)
    cv2.putText(img, f'Maior movimento: {max_detect_ts}', (5, int(55 * text_scale)), cv2.FONT_HERSHEY_SIMPLEX, text_scale,(0, 0, 0), thickness=2)

    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        # Desenhar a bounding box ao redor do objeto rastreado
        cv2.rectangle(img, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])), (0, 255, 0),
                      2)

        # Adicionar o ID do objeto
        cv2.putText(img, f'ID: {tid}', (int(tlwh[0]), int(tlwh[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Adicionar tempos
        total_time = time_data[tid]['total']
        waiting_time = time_data[tid]['waiting']
        service_time = time_data[tid]['service']

        # Adicionar texto com tempo total
        cv2.putText(img, f'Total: {total_time:.2f}s', (int(tlwh[0]), int(tlwh[1] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        # Adicionar texto com tempo de espera
        cv2.putText(img, f'Waiting: {waiting_time:.2f}s', (int(tlwh[0]), int(tlwh[1] - 50)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

        # Adicionar texto com tempo de atendimento
        cv2.putText(img, f'Service: {service_time:.2f}s', (int(tlwh[0]), int(tlwh[1] - 70)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    # Caminho de saída para a imagem processada
    output_path = os.path.join(save_dir, f'{frame_idx:05d}.jpg')
    cv2.imwrite(output_path, img)


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)

    # Configurações do VideoWriter
    video_path = os.path.join(save_dir, 'output_video.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = int(dataloader.img_size[0])
    frame_height = int(dataloader.img_size[1])
    out = cv2.VideoWriter(video_path, fourcc, frame_rate, (frame_width, frame_height))

    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    tracking_data = {}  # Para armazenar os dados de rastreamento

    # Definir áreas de interesse (substituir pelos valores reais)
    waiting_area = (100, 200, 400, 600)
    service_area = (500, 200, 800, 600)

    # Inicializar dicionário para tempos
    time_data = {}

    #incremental_filename = result_filename.replace('.txt', '_incremental.csv')

    # Variáveis para armazenar o número máximo de detecções e o timestamp correspondente
    max_detections = 0
    max_detections_timestamp = ""

    for i, (path, img, img0) in enumerate(dataloader):
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
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

                # Inicializar tempos para o objeto se ainda não existir
                if tid not in time_data:
                    time_data[tid] = {'total': 0, 'waiting': 0, 'service': 0}

                # Atualizar tempo total
                time_data[tid]['total'] += 1 / frame_rate

                # Verificar se o objeto está na área de espera ou de atendimento
                bbox_center = (tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2)
                if is_in_area(bbox_center, waiting_area):
                    time_data[tid]['waiting'] += 1 / frame_rate
                elif is_in_area(bbox_center, service_area):
                    time_data[tid]['service'] += 1 / frame_rate

        timer.toc()
        # save results
        results.append((frame_id, online_tlwhs, online_ids))

        # save results incrementally
        write_results_incremental(result_filename.replace('.txt', '.csv'), frame_id, online_tlwhs, online_ids, data_type)

        # Atualizar o número máximo de detecções e o timestamp correspondente
        num_detections = len(online_tlwhs)
        if num_detections > max_detections:
            max_detections = num_detections
            max_detections_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Save image with bounding boxes and area markings
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
            if save_dir is not None:
                save_image_with_boxes(save_dir, online_im, online_targets, frame_id, time_data, waiting_area,
                                      service_area,max_detections_timestamp, max_detections)

                # Adiciona o frame ao vídeo
                out.write(online_im)

        if show_image:
            cv2_imshow(online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1

    # save results
    write_results(result_filename.replace('.txt', '.csv'), results, data_type)
    out.release()  # Libera o VideoWriter

    # Exibir o timestamp com o maior número de detecções
    print(f"Timestamp com o maior número de detecções ({max_detections} detecções): {max_detections_timestamp}")
    return frame_id, timer.average_time, timer.calls


def is_in_area(center, area):
    x, y = center
    x1, y1, x2, y2 = area
    return x1 <= x <= x2 and y1 <= y <= y2


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

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
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        #seqs_str = '''TUD-Campus'''
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
        #seqs_str = '''MOT16-01 MOT16-07 MOT16-12 MOT16-14'''
        #seqs_str = '''MOT16-06 MOT16-08'''
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
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
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
         exp_name='MOT17_test_public_dla34',
         show_image=True,
         save_images=False,
         save_videos=False)