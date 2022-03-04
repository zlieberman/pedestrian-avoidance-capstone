
from kfp import dsl
from kfp.compiler import Compiler
from secrets import *


def mount_smb_op():
    return dsl.ContainerOp(
        name="Mount SMB Share",
        image="dbodmer0612/mount_smb:latest",
        arguments=['--username', username,
                   '--password', password],
        file_outputs={
            'json_file': '/app/json_data.json',
        }
    )


def preprocess_op(json_file):
    return dsl.ContainerOp(
        name='Preprocess Data',
        image='dbodmer0612/preprocess:latest',
        arguments=['--json_file', json_file],
        file_outputs={
            'data_split': '/app/data_split.npy',
        }
    )


def train_op(data_split):
    return dsl.ContainerOp(
        name='Train Model',
        image='dbodmer0612/train:latest',
        arguments=[
            '--data_split', data_split
        ],
        file_outputs={
            'model': '/app/model.h5',
            'loss_plot': '/app/plot.png'
        }
    )


def test_op(json_file, model):
    return dsl.ContainerOp(
        name='Test Model',
        image='dbodmer0612/train:latest',
        arguments=[
            '--json_file', json_file,
            '--model', model,
        ],
    )


@dsl.pipeline(
   name='data transfer pipeline',
   description='An example pipeline that loads an image from the SMB share and overlays a box'
)
def boston_pipeline():
    _mount_smb_op = mount_smb_op()
    _preprocess_op = preprocess_op(
        dsl.InputArgumentPath(_mount_smb_op.outputs['json_file'])
    ).after(_mount_smb_op)
    _train_op = train_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['data_split'])
    ).after(_preprocess_op)
    _test_op = test_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['data_split'])
        , dsl.InputArgumentPath(_train_op.outputs['model'])).after(_train_op)


Compiler().compile(boston_pipeline, 'pipeline.zip')
