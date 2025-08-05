import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import argparse
from shutil import copyfile

import torch
import torch.multiprocessing as mp

from core.trainer import Trainer
from core.dist import (
    get_world_size,
    get_local_rank,
    get_global_rank,
    get_master_ip,
)
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

parser = argparse.ArgumentParser(description='RetouchFormer Full Dataset Training')
parser.add_argument('-c',
                    '--config',
                    default='./configs/RetouchFormer_full_trainer.json',
                    type=str)
parser.add_argument('-p', '--port', default='2345', type=str)
args = parser.parse_args()


def main_worker(rank, config):
    if 'local_rank' not in config:
        config['local_rank'] = config['global_rank'] = rank
    if config['distributed']:
        torch.cuda.set_device(int(config['local_rank']))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=config['init_method'],
                                             world_size=config['world_size'],
                                             rank=config['global_rank'],
                                             group_name='mtorch')
        print(f'🚀 Using GPU {int(config["global_rank"])}-{int(config["local_rank"])} for training')
    
    config['save_dir'] = os.path.join(
        config['save_dir'],
        '{}_{}'.format(config['model']['net'],
                       os.path.basename(args.config).split('.')[0]))

    config['save_metric_dir'] = os.path.join(
        './scores_full',
        '{}_{}'.format(config['model']['net'],
                       os.path.basename(args.config).split('.')[0]))

    if torch.cuda.is_available():
        config['device'] = torch.device("cuda:{}".format(config['local_rank']))
    else:
        config['device'] = 'cpu'

    if (not config['distributed']) or config['global_rank'] == 0:
        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs(config['save_metric_dir'], exist_ok=True)
        config_path = os.path.join(config['save_dir'],
                                   args.config.split('/')[-1])
        if not os.path.isfile(config_path):
            copyfile(args.config, config_path)
        print(f'📁 Created output directories:')
        print(f'  Checkpoints: {config["save_dir"]}')
        print(f'  Metrics: {config["save_metric_dir"]}')

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    print("🎯 RetouchFormer Full Dataset Training Script")
    print("="*60)
    print("⚠️  Using ORIGINAL methodology (100% authors' implementation)")

    torch.backends.cudnn.benchmark = True
    mp.set_sharing_strategy('file_system')

    # loading configs
    config = json.load(open(args.config))
    print(f"📋 Loaded config from: {args.config}")
    
    # setting distributed configurations
    config['world_size'] = get_world_size()
    config['init_method'] = f"tcp://{get_master_ip()}:{args.port}"
    config['distributed'] = True if config['world_size'] > 1 else False
    
    print(f"🔧 Training Configuration:")
    print(f"  World size: {config['world_size']}")
    print(f"  Distributed: {config['distributed']}")
    print(f"  Dataset: {config['train_data_loader']['dataroot']}")
    print(f"  Iterations: {config['trainer']['iterations']}")
    print(f"  Batch size: {config['trainer']['batch_size']}")
    print(f"  Learning rate: {config['trainer']['lr']}")
    print(f"  Scheduler periods: {config['trainer']['scheduler']['periods']}")
    print(f"  Restart weights: {config['trainer']['scheduler']['restart_weights']}")
    
    # Debug mode information
    debug_config = config['trainer'].get('debug_mode', {})
    debug_enabled = debug_config.get('enabled', False)
    if debug_enabled:
        debug_epochs = debug_config.get('debug_epochs', 5)
        batches_per_epoch = debug_config.get('batches_per_debug_epoch', 10)
        print(f"\n🐛 DEBUG MODE ENABLED:")
        print(f"  First {debug_epochs} epochs will use only {batches_per_epoch} batches each")
        print(f"  This is a sanity check to ensure training starts properly")
        print(f"  After epoch {debug_epochs}, full training will begin")
    
    # setup distributed parallel training environments
    if get_master_ip() == "127.0.0.1":
        # manually launch distributed processes
        mp.spawn(main_worker, nprocs=config['world_size'], args=(config, ))
    else:
        # multiple processes have been launched by openmpi
        config['local_rank'] = get_local_rank()
        config['global_rank'] = get_global_rank()
        main_worker(-1, config) 