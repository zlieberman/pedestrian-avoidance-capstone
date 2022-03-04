import kfp
from kfp import dsl
from kfp.compiler import Compiler
from secrets import username, password
import os
from kubernetes import client as k8s_client


def download_rgb_op(username, password):
    return dsl.ContainerOp(
        name='Download RGB Images',
        image='192.168.152.34:5050/fusion_download_rgb:latest',
        arguments=[
            '--username', username,
            '--password', password
        ],
        file_outputs={
            'video_color_images': '/app/video_color_images',
            'video': '/app/video.mp4'
        }
    )


def train_detector_op(rgb_images):
    op = dsl.ContainerOp(
        name='Train YOLOv3 Detector',
        image='192.168.152.34:5050/fusion_train_detector3:latest',
        arguments=[
            '--images', rgb_images,
        ],
        file_outputs={
            'weights': '/app/yolov3-archive/weights/',
        }
    ).set_gpu_limit(2)
    return op


def fusion_op(rgb_images, video, weights):
    op = dsl.ContainerOp(
        name='Run inference',
        image='192.168.152.34:5050/fusion:latest',
        arguments=[
            "--weights", weights,
            "--config_detection", 'configs/yolov3-drone.yaml',
            # '--thermal_images', rgb_images,
            '--video_path', video,
        ],
        file_outputs={
            # 'thermal_model': '/app/runs/train/exp/weights/'
        }
    ).set_gpu_limit(2)
    op.add_env_variable(k8s_client.V1EnvVar(
        name='WANDB_API_KEY',
        value="b1408e332f2f8060d36b754f59fb9be2a85c8da7"))
    return op


@dsl.pipeline(
    name='Image Pipeline',
    description='An object-detection pipeline'
)
def image_pipeline():
    # _start_op = start_run()
    _download_rgb_op = download_rgb_op(username, password)

    _train_detector_op = train_detector_op(
        dsl.InputArgumentPath(_download_rgb_op.outputs['video_color_images'])
    ).after(_download_rgb_op)

    _fusion_op = fusion_op(
        dsl.InputArgumentPath(_download_rgb_op.outputs['video_color_images']),
        dsl.InputArgumentPath(_download_rgb_op.outputs['video']),
        dsl.InputArgumentPath(_train_detector_op.outputs['weights']),
    ).after(_download_rgb_op)


Compiler().compile(image_pipeline, 'pipeline.zip')
