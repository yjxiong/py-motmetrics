import argparse
import numpy as np
import pandas as pd
import motmetrics as mm


parser = argparse.ArgumentParser()
parser.add_argument('output_file', type=str)
parser.add_argument('gt_file', type=str)

args = parser.parse_args()
columns = ['FrameID', 'TrackID', 'x', 'y', 'w', 'h']
box_columns = ['x', 'y', 'w', 'h']
id_column = 'TrackID'

output_data = pd.DataFrame(np.load(args.output_file)[:, :6], columns=columns)
gt_data = pd.DataFrame(np.load(args.gt_file)[:, :6], columns=columns)

# Create an accumulator that will be updated during each frame
acc = mm.MOTAccumulator(auto_id=True)

max_frame = int(max(max(output_data['FrameID']), max(gt_data['FrameID'])))
print("{} frames in total".format(max_frame))

for i in range(max_frame + 1):
    frame_gt = gt_data[gt_data.FrameID == i]
    frame_output = output_data[output_data.FrameID == i]

    frame_gt_bboxes = frame_gt[box_columns].values
    frame_gt_ids = frame_gt[id_column].values

    frame_output_bboxes = frame_output[box_columns].values
    frame_output_ids = frame_output[id_column].values

    dist_mat = mm.distances.iou_matrix(frame_gt_bboxes, frame_output_bboxes, max_iou=0.5)
    acc.update(frame_gt_ids, frame_output_ids, dist_mat)


mh = mm.metrics.create()

def get_simple_metric(name):

    func = lambda df: df.noraw.Type.isin([name]).sum()
    func.__name__ = name

    return func

mh.register(get_simple_metric('CROSS'), formatter='{:d}'.format)
mh.register(get_simple_metric('BREAK'), formatter='{:d}'.format)
mh.register(get_simple_metric('SWAP'), formatter='{:d}'.format)

name_mapping = mm.io.motchallenge_metric_names

summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics + ['CROSS', 'BREAK', 'SWAP'], name='acc')
strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=name_mapping
)
print(strsummary)
