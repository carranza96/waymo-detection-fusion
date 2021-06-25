import os.path as osp

import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(BaseEvalHook):

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
<<<<<<< HEAD
            self.save_best_checkpoint(runner, key_score)

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self.save_best_checkpoint(runner, key_score)

    def save_best_checkpoint(self, runner, key_score):
        best_score = runner.meta['hook_msgs'].get(
            'best_score', self.init_value_map[self.rule])
        if self.compare_func(key_score, best_score):
            best_score = key_score
            runner.meta['hook_msgs']['best_score'] = best_score
            last_ckpt = runner.meta['hook_msgs']['last_ckpt']
            runner.meta['hook_msgs']['best_ckpt'] = last_ckpt
            mmcv.symlink(
                last_ckpt,
                osp.join(runner.work_dir, f'best_{self.key_indicator}.pth'))
            time_stamp = runner.epoch + 1 if self.by_epoch else runner.iter + 1
            self.logger.info(f'Now best checkpoint is epoch_{time_stamp}.pth.'
                             f'Best {self.key_indicator} is {best_score:0.4f}')

    def evaluate(self, runner, results):
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        if self.save_best is not None:
            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]
        else:
            return None

=======
            self._save_ckpt(runner, key_score)
>>>>>>> c4f9706e3028799b83c12b54b2a8e1fd74aef724


class DistEvalHook(BaseDistEvalHook):

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmdet.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
