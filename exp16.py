import os
import json
from omegaconf import OmegaConf, open_dict
import nemo.collections.asr as nemo_asr

from clearml import Task
task = Task.init(project_name='ASR UZRU multitoken', task_name='new RU and UZ data added LR=0.00005')
data_dir = "datasets"

TRAIN_MANIFEST_UZ = os.path.join(data_dir,'UZ_train_manifest_no_RU.json')
TEST_MANIFEST_UZ = os.path.join(data_dir, 'UZ_test_manifest_no_RU.json')

TRAIN_MANIFEST_RU = os.path.join(data_dir,'ru_train_manifest_result1.json')
TEST_MANIFEST_RU = os.path.join(data_dir, 'ru_test_manifest_result.json')

TRAIN_MANIFEST_RU_IIN_NUMBERS = os.path.join(data_dir,'numbers_ru_train_manifest.json')
TEST_MANIFEST_RU_IIN_NUMBERS = os.path.join(data_dir,'iin_numbers_ru_test_manifest.json')

TRAIN_MANIFEST_UZ_NUMBERS = os.path.join(data_dir,'train_manifest_uz_numbers_10.json')
TEST_MANIFEST_UZ_NUMBERS = os.path.join(data_dir,'test_manifest_uz_numbers_10.json')

TRAIN_NEW_MANIFEST_UZ = os.path.join(data_dir,'uz_new_train_manifest.json')
TEST_NEW_MANIFEST_UZ = os.path.join(data_dir, 'uz_new_test_manifest.json')

TRAIN_NEW_MANIFEST_RU = os.path.join(data_dir,'ru_new_train_manifest.json')
TEST_NEW_MANIFEST_RU = os.path.join(data_dir, 'ru_new_test_manifest.json')

TEST_MANIFEST_ALL =os.path.join(data_dir, 'test_manifest_all.json')

task.upload_artifact(name='exp16 UZ train Cleaned RU', artifact_object=TRAIN_MANIFEST_UZ)
task.upload_artifact(name='exp16 UZ test cleaned RU', artifact_object=TEST_MANIFEST_UZ)

task.upload_artifact(name='exp16 RU train', artifact_object=TRAIN_MANIFEST_RU)
task.upload_artifact(name='exp16 RU test', artifact_object=TEST_MANIFEST_RU)

task.upload_artifact(name='exp16 RU IIN NUMBERS train ', artifact_object=TRAIN_MANIFEST_RU_IIN_NUMBERS)
task.upload_artifact(name='exp16 RU IIN NUMBERS test', artifact_object=TEST_MANIFEST_RU_IIN_NUMBERS)

task.upload_artifact(name='exp16 UZ NUMBERS train', artifact_object=TRAIN_MANIFEST_UZ_NUMBERS)
task.upload_artifact(name='exp16 UZ NUMBERS test', artifact_object=TEST_MANIFEST_UZ_NUMBERS)


task.upload_artifact(name='exp16 UZ new train', artifact_object=TRAIN_NEW_MANIFEST_UZ)
task.upload_artifact(name='exp16 UZ new test', artifact_object=TEST_NEW_MANIFEST_UZ)
task.upload_artifact(name='exp16 RU new train',  artifact_object=TRAIN_NEW_MANIFEST_RU)
task.upload_artifact(name='exp16 RU new test',  artifact_object=TEST_NEW_MANIFEST_RU)

task.upload_artifact(name='exp16 test all',  artifact_object=TEST_MANIFEST_ALL)

asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path="/home/user/TRANSDUCER_r1.11.0/EXP16-MULTI_UZRU/experiments/Transducer-Model-MULTI-UZRU-NEW/2024-02-10_07-09-35/checkpoints/Transducer-Model-MULTI-UZRU-NEW-3.nemo")

import torch
from pytorch_lightning import Trainer
import pytorch_lightning as ptl
from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter()

if torch.cuda.is_available():
  accelerator = 'gpu'
else:
  accelerator = 'cpu'

EPOCHS = 20


# Initialize a Trainer for the Transducer model
trainer = Trainer(devices=1, accelerator=accelerator, max_epochs=EPOCHS,
                  enable_checkpointing=False, logger=False,
                  log_every_n_steps=1000, check_val_every_n_epoch=1,accumulate_grad_batches=1,val_check_interval=0.5)

train_ds = {}
train_ds['manifest_filepath'] = [TRAIN_MANIFEST_RU,TRAIN_MANIFEST_UZ,TRAIN_MANIFEST_RU_IIN_NUMBERS, TRAIN_MANIFEST_UZ_NUMBERS, TRAIN_NEW_MANIFEST_UZ,TRAIN_NEW_MANIFEST_RU ]
train_ds['sample_rate'] = 16000
train_ds['batch_size'] = 16
train_ds['fused_batch_size'] = 16
train_ds['shuffle'] = True
train_ds['max_duration'] = 16.7
train_ds['pin_memory'] = True
train_ds['is_tarred'] = False
train_ds['num_workers'] = 8

asr_model.set_trainer(trainer)
asr_model.setup_training_data(train_data_config=train_ds) 

validation_ds = {}
validation_ds['sample_rate'] = 16000
validation_ds['manifest_filepath'] = [TEST_MANIFEST_ALL, TEST_MANIFEST_UZ, TEST_MANIFEST_RU, TEST_MANIFEST_RU_IIN_NUMBERS, TEST_MANIFEST_UZ_NUMBERS, TEST_NEW_MANIFEST_UZ, TEST_NEW_MANIFEST_RU]
validation_ds['batch_size'] = 16
validation_ds['shuffle'] = False
validation_ds['num_workers'] = 8


asr_model.setup_multiple_validation_data(val_data_config=validation_ds) 

optimizer_conf = {}

optimizer_conf['name'] = 'novograd'
optimizer_conf['lr'] =  0.00005
optimizer_conf['betas'] =  [0.9, 0.0]
optimizer_conf['weight_decay'] = 0.001

sched = {}
sched['name'] = 'CosineAnnealing'
sched['warmup_steps'] = None
sched['warmup_ratio'] = None
sched['min_lr'] = 1e-6
optimizer_conf['sched'] = sched
# cfg/joint/jointnet/dropout

asr_model.setup_optimization(optimizer_conf)

asr_model.wer.log_predictions = True
asr_model.compute_eval_loss = True

asr_model.summarize()
# print(asr_model)
# Prepare NeMo's Experiment manager to handle checkpoint saving and logging for us
from nemo.utils import exp_manager

# Environment variable generally used for multi-node multi-gpu training.
# In notebook environments, this flag is unnecessary and can cause logs of multiple training runs to overwrite each other.
# os.environ.pop('NEMO_EXPM_VERSION', None)

exp_config = exp_manager.ExpManagerConfig(
    exp_dir=f'experiments/',
    name=f"Transducer-Model-MULTI-UZRU-NEW-LR-5",
    checkpoint_callback_params=exp_manager.CallbackParams(
        monitor="val_wer",
        mode="min",
        always_save_nemo=True,
        save_best_model=True,
        save_top_k = 20,
    ),
)

exp_config = OmegaConf.structured(exp_config)

logdir = exp_manager.exp_manager(trainer, exp_config)

# Release resources prior to training
import gc
gc.collect()

if accelerator == 'gpu':
  torch.cuda.empty_cache()

params_dictionary = exp_config
task.connect(params_dictionary)

# Train the model
trainer.fit(asr_model)

#  writer.add_scalar(logdir)
#  writer.close()
     
