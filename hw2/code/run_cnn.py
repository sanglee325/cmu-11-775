import argparse
import os.path as osp

import pandas as pd
from pyturbo import Job, Options, System, Task

from stages import CNNFeature, LoadVideo, SaveFeature


class ExtractCNNFeature(System):

    def get_num_pipeline(self, resources, *, args):
        self.args = args
        return len(resources.get('gpu'))

    def get_stages(self, resources):
        resources_cpu = resources.select(cpu=True)
        stages = [
            LoadVideo(resources_cpu.select(cpu=(0, 1)),
                      video_dir=self.args.video_dir),
            CNNFeature(resources, model_name='resnet18',
                       node_name='avgpool', replica_per_gpu=4),
            SaveFeature(resources_cpu.select(cpu=(0, 1)),
                        feature_dir=self.args.cnn_dir),
        ]
        return stages


def parse_args(argv=None):
    parser = argparse.ArgumentParser(__file__)
    parser.add_argument('list_file_path')
    parser.add_argument(
        '--video_dir', default=osp.join(
            osp.dirname(__file__), '../data/videos'))
    parser.add_argument(
        '--cnn_dir', default=osp.join(osp.dirname(__file__), '../data/cnn-val'))
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(argv)
    return args


def build_jobs(args):
    df = pd.read_csv(args.list_file_path)
    video_ids = df['Id']
    jobs = [Job(vid, Task(vid, {'video_id': vid})) for vid in video_ids]
    return jobs


def main(args):
    if args.debug:
        Options.single_sync_pipeline = True
        Options.raise_exception = True
    system = ExtractCNNFeature(args=args)
    system.start()
    jobs = build_jobs(args)
    system.add_jobs(jobs)
    try:
        for job in system.wait_jobs(len(jobs)):
            continue
        system.end()
    except:
        system.terminate()


if __name__ == '__main__':
    main(parse_args())
