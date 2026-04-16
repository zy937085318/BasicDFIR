import datetime
import logging
import math
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
import time
import torch
torch.set_num_threads(2)
from os import path as osp
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options


def init_tb_loggers(opt, result_dir):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        # tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
        tb_logger = init_tb_logger(log_dir=result_dir)  # Save TensorBoard logs in the same result directory for better organization
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # Create result directory for training comparison images with training info
    # Format: result_<model_name>_<dataset_name>_<scale>_<timestamp>_<microsecond>
    # Each training run gets a unique folder with timestamp and microsecond
    model_name = opt.get('name', 'unknown')
    dataset_name = opt['datasets']['train'].get('name', 'unknown')
    scale = opt.get('scale', 4)
    timestamp = get_time_str()
    # Add microseconds to ensure uniqueness
    microsecond = int((time.time() % 1) * 1000000)
    result_dir_name = f"result_{model_name}_{dataset_name}_x{scale}_{timestamp}_{microsecond:06d}"
    result_dir = osp.join(opt['path']['experiments_root'], result_dir_name)
    # print('##################')
    # print(opt['path']['experiments_root'])
    if opt['rank'] == 0:
        # Ensure directory doesn't exist, if it does, add more random suffix
        counter = 0
        original_result_dir = result_dir
        while osp.exists(result_dir):
            counter += 1
            result_dir = f"{original_result_dir}_{counter}"
        os.makedirs(result_dir, exist_ok=True)

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    if opt['rank'] == 0:
        logger.info(f'Training result images will be saved to: {result_dir}')
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt, result_dir)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # create model
    model = build_model(opt)
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # Save training result comparison images every 100 iterations
            # Disabled: Commented out to avoid generating validation images every 100 steps
            # if current_iter % 100 == 0 and opt['rank'] == 0:
            #     # Set network to eval mode
            #     if hasattr(model, 'net_g'):
            #         model.net_g.eval()
            #     with torch.no_grad():
            #         # Get a sample from current batch (use a copy to avoid affecting training)
            #         sample_data = {}
            #         sample_data['lq'] = train_data['lq'][:1].clone()  # Take first image
            #         if 'gt' in train_data:
            #             sample_data['gt'] = train_data['gt'][:1].clone()
            #         model.feed_data(sample_data)
            #         # Use test() method which handles sampling correctly
            #         # This will use GT shape if available and start from upsampled LQ
            #         model.test()
            #         if hasattr(model, 'save_training_results'):
            #             model.save_training_results(current_iter, result_dir)
            #     # Set network back to train mode
            #     if hasattr(model, 'net_g'):
            #         model.net_g.train()

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                logger.info(opt.get('description', ''))
                if len(val_loaders) > 1:
                    # Check if model supports multiple validation datasets
                    # SRModel and its subclasses (like FlowModel) support multiple validation datasets
                    from basicsr.models.sr_model import SRModel
                    if not isinstance(model, SRModel):
                        logger.warning('Multiple validation datasets are *only* supported by SRModel and its subclasses.')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        # end of iter

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()

    # After finishing training, keep only the saved files (models, visualizations,
    # training states) corresponding to the best validation iteration and remove
    # other intermediate saved files to save disk space.
    # In distributed training only the main process (rank 0) should perform pruning
    path = opt['path']['experiments_root']
    if opt.get('rank', 0) == 0:
        try:
            # Only try pruning when validation metrics exist and were run during training
            if opt.get('val') is not None and opt['val'].get('metrics') is not None and hasattr(model,
                                                                                                'best_metric_results'):
                import json

                # For each metric (possibly across datasets) record its best iteration.
                # Collect all best iterations (there may be duplicates) and keep files
                # corresponding to any of these iterations.
                best_iters = set()
                metric_best_map = {}  # metric -> list of (dataset, val, iter)
                for dataset_name, record in model.best_metric_results.items():
                    for metric, info in record.items():
                        it = info.get('iter', -1)
                        val = info.get('val', None)
                        if it is None or it == -1:
                            # skip uninitialized
                            continue
                        best_iters.add(int(it))
                        metric_best_map.setdefault(metric, []).append(
                            {'dataset': dataset_name, 'iter': int(it), 'val': val, 'better': info.get('better')})

                if len(best_iters) == 0:
                    logger.warning('No valid best iterations found in best_metric_results. Skip pruning.')
                else:
                    logger.info(f'Best iterations collected from metrics: {sorted(best_iters)}')

                # save a small summary file with best iterations and metric info
                best_info = dict(best_iters=sorted(best_iters), metric_best_map=metric_best_map,
                                 all_best_metric_results=model.best_metric_results)
                try:
                    with open(osp.join(path, 'best_metric_results.json'), 'w') as f:
                        json.dump(best_info, f, indent=2)
                except Exception:
                    logger.warning('Failed to write best metric summary file.')

                # If no valid best iterations, skip pruning
                if len(best_iters) == 0:
                    logger.warning('No valid best iteration found. Skip pruning saved files.')
                else:
                    # Helper to remove files not matching keep condition
                    def prune_files(root_dir, keep_check_fn):
                        if not osp.exists(root_dir):
                            return
                        for root, dirs, files in os.walk(root_dir):
                            for fn in files:
                                fp = osp.join(root, fn)
                                try:
                                    if not keep_check_fn(fn):
                                        os.remove(fp)
                                except Exception as e:
                                    logger.warning(f'Failed to remove file: {fp}, error: {e}')

                    # Build a keeper check using all best_iters
                    def keep_if_best_iters_in_name(fn):
                        for it in best_iters:
                            # match patterns like _{iter}. (before extension) or _{iter}.pth
                            if f'_{it}.' in fn or fn.endswith(f'_{it}.pth') or fn == f'{it}.state':
                                return True
                        return False

                    # Prune models: keep files that include any best_iter pattern
                    models_dir = opt['path'].get('models')
                    if models_dir:
                        prune_files(models_dir, lambda fn: any(
                            f'_{it}.' in fn or fn.endswith(f'_{it}.pth') for it in best_iters))

                    # Prune training states: keep {best_iter}.state for any best_iter
                    states_dir = opt['path'].get('training_states')
                    if states_dir:
                        prune_files(states_dir, lambda fn: any(fn == f'{it}.state' for it in best_iters))

                    # Prune visualizations: keep files that include _{best_iter} before the extension
                    vis_dir = opt['path'].get('visualization')
                    if vis_dir:
                        prune_files(vis_dir, lambda fn: any(f'_{it}.' in fn for it in best_iters))
            else:
                logger.info('No validation metrics or best_metric_results found; skip pruning saved files.')
        except Exception:
            logger.exception('Unexpected error when pruning saved files for best iteration.')
        # Finally, rename the experiments root to indicate finishing
        try:
            os.rename(path, path + '_finished')
        except Exception:
            # If rename fails, just log a warning but don't crash
            logger.warning(f'Failed to rename {path} to {path}_finished')
    else:
        logger.info('Non-master process: skip pruning saved files and rename.')

    # path = opt['path']['experiments_root']
    # os.rename(path, path + '_finished')



if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
