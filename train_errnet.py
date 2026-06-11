from os.path import exists, join
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
from util.train_eval import (
    PeriodicEvaluator,
    build_eval_dataloaders,
    parse_eval_datasets,
    parse_eval_epoch_schedule,
)
import data

opt = TrainOptions().parse()

cudnn.benchmark = True

opt.display_freq = 10

if opt.debug:
    opt.display_id = 1
    opt.display_freq = 20
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 100
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True

# processed datasets prepared by datasets/prepare_train_data.py and datasets/prepare_test_data.py
datadir = './datasets/processed_data'

datadir_syn = join(datadir, 'VOCdevkit/VOC2012/PNGImages')
datadir_real = join(datadir, 'real_train')

reflection_synthesis_kwargs = {
    'kernel_sizes': opt.syn_kernel_sizes,
    'enhanced_synthesis': not opt.no_reflection_aug,
    'reflection_alpha': (opt.reflection_alpha_low, opt.reflection_alpha_high),
    'transmission_alpha': (opt.transmission_alpha_low, opt.transmission_alpha_high),
    'reflection_color_jitter': opt.reflection_color_jitter,
    'reflection_shift': opt.reflection_shift,
    'reflection_noise_std': opt.reflection_noise_std,
    'reflection_jpeg_prob': opt.reflection_jpeg_prob,
    'reflection_jpeg_quality': (opt.reflection_jpeg_quality_low, opt.reflection_jpeg_quality_high),
}


def parse_float_list(value):
    return [float(v.strip()) for v in str(value).split(',') if v.strip()]


def _append_optional_source(datasets_list, ratios, dataset, ratio, label):
    ratio = max(0.0, min(1.0, ratio))
    if ratio <= 0:
        return datasets_list, ratios
    print('[i] using optional {} training data: {} imgs from {}'.format(
        label, len(dataset), getattr(dataset, 'datadir', 'unknown')))
    return datasets_list + [dataset], ratios + [ratio]


def append_optional_sources(datasets_list, base_ratios):
    optional_sources = []

    if opt.identity_train_ratio > 0 and not opt.identity_train_dir:
        raise ValueError('identity_train_ratio is set but identity_train_dir is missing.')

    if opt.openrr_train_dir:
        if not exists(opt.openrr_train_dir):
            raise FileNotFoundError('OpenRR training directory does not exist: {}'.format(opt.openrr_train_dir))
        openrr_ratio = max(0.0, min(1.0, opt.openrr_train_ratio))
        if openrr_ratio > 0:
            openrr_dataset = datasets.CEILTestDataset(
                opt.openrr_train_dir,
                size=opt.max_dataset_size,
                enable_transforms=True)
            optional_sources.append(('OpenRR', openrr_dataset, openrr_ratio))

    if opt.identity_train_dir:
        if not exists(opt.identity_train_dir):
            raise FileNotFoundError('Identity training directory does not exist: {}'.format(opt.identity_train_dir))
        identity_ratio = max(0.0, min(1.0, opt.identity_train_ratio))
        if identity_ratio > 0:
            identity_fns = read_fns(opt.identity_train_filelist) if opt.identity_train_filelist else None
            identity_size = opt.identity_train_size
            if opt.max_dataset_size is not None:
                identity_size = opt.max_dataset_size if identity_size is None else min(identity_size, opt.max_dataset_size)
            identity_dataset = datasets.IdentityDataset(
                opt.identity_train_dir,
                fns=identity_fns,
                size=identity_size,
                enable_transforms=True)
            optional_sources.append(('identity', identity_dataset, identity_ratio))

    optional_ratio_sum = sum(ratio for _, _, ratio in optional_sources)
    if optional_ratio_sum >= 1.0:
        raise ValueError('Optional dataset ratios must sum to less than 1.0, got {}'.format(optional_ratio_sum))

    base_sum = sum(base_ratios)
    if base_sum <= 0:
        raise ValueError('Base train fusion ratios must sum to a positive number.')
    scaled_base = [(ratio / base_sum) * (1.0 - optional_ratio_sum) for ratio in base_ratios]

    for label, dataset, ratio in optional_sources:
        if ratio > 0:
            print('[i] using optional {} training data: {} imgs from {}'.format(
                label, len(dataset), getattr(dataset, 'datadir', 'unknown')))

    return datasets_list + [dataset for _, dataset, _ in optional_sources], scaled_base + [ratio for _, _, ratio in optional_sources]

train_dataset = datasets.CEILDataset(
    datadir_syn, read_fns('VOC2012_224_train_png.txt'), size=opt.max_dataset_size, enable_transforms=True, 
    low_sigma=opt.low_sigma, high_sigma=opt.high_sigma,
    low_gamma=opt.low_gamma, high_gamma=opt.high_gamma,
    **reflection_synthesis_kwargs)

train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)

train_datasets = [train_dataset, train_dataset_real]
train_ratios = parse_float_list(opt.train_fusion_ratios)
if len(train_ratios) != len(train_datasets):
    raise ValueError('--train_fusion_ratios must provide {} values, got {}'.format(
        len(train_datasets), len(train_ratios)))
train_datasets, train_ratios = append_optional_sources(train_datasets, train_ratios)

train_dataset_fusion = datasets.FusionDataset(train_datasets, train_ratios)

train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches, 
    num_workers=opt.nThreads, pin_memory=True)

"""Main Loop"""
engine = Engine(opt)
eval_names = parse_eval_datasets(opt.train_eval_datasets)
eval_loaders = build_eval_dataloaders(opt, datadir, eval_names)
eval_interval_iters = max(1, int(len(train_dataloader_fusion) * opt.train_eval_interval_epochs))
eval_epoch_schedule = parse_eval_epoch_schedule(opt.train_eval_epoch_schedule)
periodic_evaluator = PeriodicEvaluator(
    engine,
    eval_loaders,
    eval_interval_iters,
    enabled=not opt.no_train_eval,
    save_best=not opt.no_save_best_eval,
    best_label='best_eval',
    epoch_schedule=eval_epoch_schedule)

def set_learning_rate(lr):
    if hasattr(engine.model, 'set_learning_rate'):
        engine.model.set_learning_rate(lr)
        return
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)

if opt.resume:
    periodic_evaluator.maybe_eval(force=True, tag='resume_start')

# define training strategy 
engine.model.opt.lambda_gan = 0
if engine.epoch >= 20:
    engine.model.opt.lambda_gan = opt.lambda_gan
# engine.model.opt.lambda_gan = 0.01
set_learning_rate(opt.lr)
while engine.epoch < opt.nEpochs:
    if engine.epoch == 20:
        engine.model.opt.lambda_gan = opt.lambda_gan # gan loss is added after epoch 20
    if engine.epoch == 30:
        set_learning_rate(opt.lr * 0.5)
    if engine.epoch == 40:
        set_learning_rate(opt.lr * 0.1)
    if engine.epoch == 45:
        optional_ratios = train_dataset_fusion.fusion_ratios[2:]
        remaining_ratio = max(0.0, 1.0 - sum(optional_ratios))
        ratio = [0.5 * remaining_ratio, 0.5 * remaining_ratio] + optional_ratios
        print('[i] adjust fusion ratio to {}'.format(ratio))
        train_dataset_fusion.fusion_ratios = ratio
        set_learning_rate(opt.lr * 0.5)
    if engine.epoch == 50:
        set_learning_rate(opt.lr * 0.1)

    engine.train(train_dataloader_fusion, on_iter_end=periodic_evaluator.on_iter_end)
    if eval_epoch_schedule is not None:
        periodic_evaluator.maybe_eval_epoch(engine.epoch)
    else:
        periodic_evaluator.maybe_eval(force=False, tag='epoch_{}_end'.format(engine.epoch))
