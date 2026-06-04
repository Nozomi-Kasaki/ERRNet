from os.path import join

import data.reflect_dataset as datasets


BASELINE_METRICS = {
    'ceilnet_table2': {'PSNR': 27.88, 'SSIM': 0.9407, 'NCC': 0.9808, 'LMSE': 0.0048},
    'real20': {'PSNR': 23.55, 'SSIM': 0.8285, 'NCC': 0.8877, 'LMSE': 0.0201},
    'objects': {'PSNR': 24.85, 'SSIM': 0.8980, 'NCC': 0.9817, 'LMSE': 0.0029},
    'postcard': {'PSNR': 22.07, 'SSIM': 0.8773, 'NCC': 0.9463, 'LMSE': 0.0044},
    'wild': {'PSNR': 25.18, 'SSIM': 0.8860, 'NCC': 0.9359, 'LMSE': 0.0083},
}


EVAL_DATASETS = {
    'ceilnet_table2': {
        'dataset_name': 'testdata_table2',
        'path': 'testdata_CEILNET_table2',
    },
    'real20': {
        'dataset_name': 'testdata_real',
        'path': 'real20',
        'max_long_edge': 512,
    },
    'objects': {
        'dataset_name': 'testdata_objects',
        'path': 'objects',
    },
    'postcard': {
        'dataset_name': 'testdata_postcard',
        'path': 'postcard',
    },
    'wild': {
        'dataset_name': 'testdata_wild',
        'path': 'wild',
    },
    'sir2_withgt': {
        'dataset_name': 'testdata_sir2',
        'path': 'sir2_withgt',
    },
}


def parse_eval_datasets(value):
    names = [name.strip() for name in value.split(',') if name.strip()]
    unknown = [name for name in names if name not in EVAL_DATASETS]
    if unknown:
        raise ValueError('Unknown eval datasets: {}'.format(', '.join(unknown)))
    return names


def build_eval_dataloaders(opt, data_root, dataset_names):
    eval_loaders = []
    for name in dataset_names:
        spec = EVAL_DATASETS[name]
        dataset = datasets.CEILTestDataset(
            join(data_root, spec['path']),
            max_long_edge=spec.get('max_long_edge'),
        )
        dataloader = datasets.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=opt.nThreads,
            pin_memory=True,
        )
        eval_loaders.append((name, spec['dataset_name'], dataloader))
    return eval_loaders


def meters_to_dict(meters):
    return {key: meters[key] for key in meters.keys()}


def dataset_selection_score(metrics, baseline):
    parts = []
    for key in ('PSNR', 'SSIM'):
        if key in metrics and key in baseline:
            parts.append((metrics[key] - baseline[key]) / max(abs(baseline[key]), 1e-8))
    if not parts:
        return 0.0
    return sum(parts) / len(parts)


def selection_score(results):
    scores = []
    for name, metrics in results.items():
        baseline = BASELINE_METRICS.get(name)
        if baseline is not None:
            scores.append(dataset_selection_score(metrics, baseline))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def average_metric_deltas(results):
    deltas = {}
    counts = {}
    for name, metrics in results.items():
        baseline = BASELINE_METRICS.get(name)
        if baseline is None:
            continue
        for key in ('PSNR', 'SSIM', 'NCC', 'LMSE'):
            if key in metrics and key in baseline:
                deltas[key] = deltas.get(key, 0.0) + metrics[key] - baseline[key]
                counts[key] = counts.get(key, 0) + 1
    return {key: deltas[key] / counts[key] for key in deltas}


def format_metrics(metrics):
    fields = []
    for key in ('PSNR', 'SSIM', 'NCC', 'LMSE'):
        if key in metrics:
            fields.append('{}={:.4f}'.format(key, metrics[key]))
    return ', '.join(fields)


def format_delta(metrics, baseline):
    fields = []
    for key in ('PSNR', 'SSIM', 'NCC'):
        if key in metrics and key in baseline:
            fields.append('d{}={:+.4f}'.format(key, metrics[key] - baseline[key]))
    if 'LMSE' in metrics and 'LMSE' in baseline:
        fields.append('dLMSE={:+.4f}'.format(metrics['LMSE'] - baseline['LMSE']))
    return ', '.join(fields)


class PeriodicEvaluator(object):
    def __init__(
            self, engine, eval_loaders, interval_iters, enabled=True,
            save_best=True, best_label='best_eval'):
        self.engine = engine
        self.eval_loaders = eval_loaders
        self.interval_iters = max(1, int(interval_iters))
        self.enabled = enabled and len(eval_loaders) > 0
        self.save_best = save_best
        self.best_label = best_label
        self.best_score = None
        self.last_eval_iter = None

    def maybe_eval(self, force=False, tag=None):
        if not self.enabled:
            return None
        current_iter = self.engine.iterations
        if not force:
            if self.last_eval_iter is not None and current_iter - self.last_eval_iter < self.interval_iters:
                return None
        self.last_eval_iter = current_iter

        tag = tag or 'iter_{}'.format(current_iter)
        print('\n[i] full evaluation at {} (epoch {}, iter {})'.format(
            tag, self.engine.epoch, current_iter))

        results = {}
        for short_name, dataset_name, dataloader in self.eval_loaders:
            meters = self.engine.eval(dataloader, dataset_name=dataset_name)
            metrics = meters_to_dict(meters)
            results[short_name] = metrics
            baseline = BASELINE_METRICS.get(short_name)
            if baseline is None:
                print('[eval] {}: {}'.format(short_name, format_metrics(metrics)))
            else:
                print('[eval] {}: {} | vs baseline: {}'.format(
                    short_name, format_metrics(metrics), format_delta(metrics, baseline)))

        score = selection_score(results)
        avg_deltas = average_metric_deltas(results)
        if avg_deltas:
            print('[eval] mean metric deltas: {}'.format(', '.join(
                '{}={:+.4f}'.format(key, avg_deltas[key])
                for key in ('PSNR', 'SSIM', 'NCC', 'LMSE')
                if key in avg_deltas)))
        print('[eval] selection score (mean relative PSNR/SSIM): {:+.6f}'.format(score))

        if self.save_best and (self.best_score is None or score > self.best_score):
            self.best_score = score
            self.engine.model.save(label=self.best_label)
            print('[eval] saved new best checkpoint: {} (score {:+.6f})'.format(
                self.best_label, score))

        return results

    def on_iter_end(self, engine, batch_index, total_batches):
        self.maybe_eval(force=False)
