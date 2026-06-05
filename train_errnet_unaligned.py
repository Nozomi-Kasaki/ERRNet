from os.path import join
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

# processed datasets prepared by datasets/prepare_train_data.py and datasets/prepare_test_data.py
datadir = './datasets/processed_data'
raw_datadir = './datasets/raw_data'

datadir_syn = join(datadir, 'VOCdevkit/VOC2012/PNGImages')
datadir_real = join(datadir, 'real_train')
datadir_unaligned = join(raw_datadir, 'Dataset/DSLR/unaligned_train250')

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

train_dataset = datasets.CEILDataset(
    datadir_syn,
    read_fns('VOC2012_224_train_png.txt'),
    size=opt.max_dataset_size,
    low_sigma=opt.low_sigma,
    high_sigma=opt.high_sigma,
    low_gamma=opt.low_gamma,
    high_gamma=opt.high_gamma,
    **reflection_synthesis_kwargs)
train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)

train_dataset_unaligned = datasets.CEILTestDataset(datadir_unaligned, enable_transforms=True, flag={'unaligned':True}, size=None)

train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_unaligned, train_dataset_real], [0.25,0.5,0.25])


train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches, 
    num_workers=opt.nThreads, pin_memory=True)


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

"""Main Loop"""
def set_learning_rate(lr):
    if hasattr(engine.model, 'set_learning_rate'):
        engine.model.set_learning_rate(lr)
        return
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)


set_learning_rate(opt.lr)
target_epoch = opt.nEpochs
if opt.resume and engine.epoch >= target_epoch:
    target_epoch = engine.epoch + 20
    print('[i] resume epoch is not below --nEpochs; finetuning for 20 extra epochs until {}'.format(target_epoch))

if opt.resume:
    periodic_evaluator.maybe_eval(force=True, tag='resume_start')

while engine.epoch < target_epoch:
    if engine.epoch == 65:
        set_learning_rate(opt.lr * 0.5)
    if engine.epoch == 70:
        set_learning_rate(opt.lr * 0.1)
        
    engine.train(train_dataloader_fusion, on_iter_end=periodic_evaluator.on_iter_end)
    if eval_epoch_schedule is not None:
        periodic_evaluator.maybe_eval_epoch(engine.epoch)
    else:
        periodic_evaluator.maybe_eval(force=False, tag='epoch_{}_end'.format(engine.epoch))
