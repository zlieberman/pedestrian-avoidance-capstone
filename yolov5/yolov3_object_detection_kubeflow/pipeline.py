import kfp
from kfp import dsl
from kfp.compiler import Compiler
from secrets import username, password
import os
from kubernetes import client as k8s_client


def download_rgb_op(username, password):
    return dsl.ContainerOp(
        name='Download RGB Images',
        image='192.168.152.34:5050/video_color_images:latest',
        arguments=[
            '--username', username,
            '--password', password
        ],
        file_outputs={
            'video_color_images': '/app/video_color_images',
        }
    )


def download_thermal_op(username, password):
    return dsl.ContainerOp(
        name='Download Thermal Images',
        image='192.168.152.34:5050/thermal_images:latest',
        arguments=[
            '--username', username,
            '--password', password
        ],
        file_outputs={
            'thermal_images': '/app/labels_05_13_20_thermal',
        }
    )


def download_nir_op(username, password):
    return dsl.ContainerOp(
        name='Download NIR Images',
        image='192.168.152.34:5050/nir_images:latest',
        arguments=[
            '--username', username,
            '--password', password
        ],
        file_outputs={
            'nir_val_images': '/app/labels_06_03_20_NIR',
            'nir_train_images': '/app/labels_05_13_20_NIR',
        }
    )


def download_lidar_op(username, password):
    return dsl.ContainerOp(
        name='Download LIDAR Images',
        image='192.168.152.34:5050/lidar_images:latest',
        arguments=[
            '--username', username,
            '--password', password
        ],
        file_outputs={
            'lidar_images': '/app/lidar_images',
        }
    )


def rgb_train_op(video_color_images):
    op = dsl.ContainerOp(
        name='Train RGB Model',
        image='192.168.152.34:5050/rgb_train:latest',
        arguments=[
            '--video_color_images', video_color_images,
        ],
        file_outputs={
            'rgb_model': '/app/runs/train/exp/weights/'
        }
    ).set_gpu_limit(2)
    op.add_env_variable(k8s_client.V1EnvVar(
        name='WANDB_API_KEY',
        value="b1408e332f2f8060d36b754f59fb9be2a85c8da7"))
    return op


def thermal_train_op(thermal_images):
    op = dsl.ContainerOp(
        name='Train Thermal Model',
        image='192.168.152.34:5050/thermal_train_one:latest',
        arguments=[
            '--thermal_images', thermal_images,
        ],
        file_outputs={
            'thermal_model': '/app/runs/train/exp/weights/'
        }
    ).set_gpu_limit(2)
    op.add_env_variable(k8s_client.V1EnvVar(
        name='WANDB_API_KEY',
        value="b1408e332f2f8060d36b754f59fb9be2a85c8da7"))
    return op


def nir_train_op(nir_train_images, nir_val_images):
    op = dsl.ContainerOp(
        name='Train NIR Model',
        image='192.168.152.34:5050/nir_train:latest',
        arguments=[
            '--nir_train_images', nir_train_images,
            '--nir_val_images', nir_val_images,
        ],
        file_outputs={
            'nir_model': '/app/runs/train/exp/weights/'
        }
    ).set_gpu_limit(2)
    op.add_env_variable(k8s_client.V1EnvVar(
        name='WANDB_API_KEY',
        value="b1408e332f2f8060d36b754f59fb9be2a85c8da7"))
    return op


def lidar_train_op(lidar_images):
    op = dsl.ContainerOp(
        name='Train LIDAR Model',
        image='dbodmer/lidar_train:latest',
        arguments=[
            '--lidar_images', lidar_images,
        ],
        file_outputs={
            'lidar_model': '/app/runs/train/exp/weights/'
        }
    )
    op.add_env_variable(k8s_client.V1EnvVar(
        name='WANDB_API_KEY',
        value="b1408e332f2f8060d36b754f59fb9be2a85c8da7"))
    return op


def fuse_op(video_color_images, rgb_model, nir_model, thermal_model):
    op = dsl.ContainerOp(
        name='Fuse Model',
        image='192.168.152.34:5050/fuse:latest',
        arguments=[
            '--video_color_images', video_color_images,
            '--rgb_model', rgb_model,
            '--nir_model', nir_model,
            '--thermal_model', thermal_model,
            # '--lidar_model', lidar_model,
        ],
        file_outputs={
            # 'fused': '/app/fused.pkl'
        }

    ).set_gpu_limit(2)
    op.add_env_variable(k8s_client.V1EnvVar(
        name='WANDB_API_KEY',
        value="b1408e332f2f8060d36b754f59fb9be2a85c8da7"))
    return op


def inference_op(fused):
    return dsl.ContainerOp(
        name='Run Inference',
        image='dbodmer0612/inference:latest',
        arguments=[
            '--fused', fused,
        ],
        file_outputs={
            'fused': '/app/fused.pkl'
        }
    )


@dsl.pipeline(
    name='Image Pipeline',
    description='An object-detection pipeline'
)
def image_pipeline():
    # _start_op = start_run()
    _download_rgb_op = download_rgb_op(username, password)#.after(_start_op)
    _download_thermal_op = download_thermal_op(username, password)#.after(_start_op)
    _download_nir_op = download_nir_op(username, password)#.after(_start_op)
    # _download_lidar_op = download_lidar_op(username, password)#.after(_start_op)

    # with dsl.Condition('train' == 'train'):
    _rgb_train_op = rgb_train_op(
        dsl.InputArgumentPath(_download_rgb_op.outputs['video_color_images'])
    ).after(_download_rgb_op)

    _thermal_train_op = thermal_train_op(
        dsl.InputArgumentPath(_download_thermal_op.outputs['thermal_images'])
    ).after(_download_thermal_op)

    _nir_train_op = nir_train_op(
        dsl.InputArgumentPath(_download_nir_op.outputs['nir_train_images']),
        dsl.InputArgumentPath(_download_nir_op.outputs['nir_val_images']),
    ).after(_download_nir_op)

    # _lidar_train_op = lidar_train_op(
    #     dsl.InputArgumentPath(_download_lidar_op.outputs['lidar_images'])
    # ).after(_download_lidar_op)

    _fuse_op = fuse_op(
        dsl.InputArgumentPath(_download_rgb_op.outputs['video_color_images']),
        dsl.InputArgumentPath(_rgb_train_op.outputs['rgb_model']),
        dsl.InputArgumentPath(_thermal_train_op.outputs['thermal_model']),
        dsl.InputArgumentPath(_nir_train_op.outputs['nir_model']),
        # dsl.InputArgumentPath(_lidar_train_op.outputs['lidar_model'])
    ).after(_rgb_train_op)

    # with dsl.Condition('inference' == 'inference'):

    # _fuse_op = fuse_op(
    #     dsl.InputArgumentPath(_download_rgb_op.outputs['video_color_images']),
    #     dsl.InputArgumentPath(_download_thermal_op.outputs['thermal_images']),
    #     # dsl.InputArgumentPath(_download_nir_op.outputs['nir_images']),
    #     # dsl.InputArgumentPath(_download_lidar_op.outputs['lidar_images'])
    # ).after(_download_nir_op)

    # _rgb_inference_op = inference_op(
    #     dsl.InputArgumentPath(_fuse_op.outputs['fused'])
    # ).after(_fuse_op)

Compiler().compile(image_pipeline, 'pipeline.zip')
