import math
import torch
import torch.nn as nn
from functools import partial

from .kl_div import KLDivergence
from .dist_kd import DIST


class KDLoss():
    '''
    kd loss wrapper.
    '''

    def __init__(self, student, teacher, ori_loss, kd_method='kdt4', student_module='', teacher_module='', ori_loss_weight=1.0, kd_loss_weight=1.0):
        self.student = student
        self.teacher = teacher
        self.ori_loss = ori_loss
        self.ori_loss_weight = ori_loss_weight
        self.kd_method = kd_method
        self.kd_loss_weight = kd_loss_weight

        self._teacher_out = None
        self._student_out = None

        # init kd loss
        if kd_method == 'kd':
            self.kd_loss = KLDivergence(tau=4)
        elif kd_method == 'dist':
            self.kd_loss = DIST(beta=1, gamma=1)
        elif kd_method.startswith('kdt'):
            tau = float(kd_method[3:])
            self.kd_loss = KLDivergence(tau)
        else:
            raise RuntimeError(f'KD method {kd_method} not found.')

        # register forward hook
        self._register_forward_hook(student, student_module, teacher=False)
        self._register_forward_hook(teacher, teacher_module, teacher=True)

        teacher.eval()

    def __call__(self, x, targets):
        with torch.no_grad():
            self.teacher(x)

        # compute ori loss of student
        logits = self.student(x)
        ori_loss = self.ori_loss(logits, targets)

        # compute kd loss
        kd_loss = self.kd_loss(self._student_out, self._teacher_out)
        del self._student_out, self._teacher_out
        return ori_loss * self.ori_loss_weight + kd_loss * self.kd_loss_weight

    def _register_forward_hook(self, model, name, teacher=False):
        if name == '':
            # use the output of model
            model.register_forward_hook(partial(self._forward_hook, teacher=teacher))
        else:
            module = None
            for k, m in model.named_modules():
                if k == name:
                    module = m
                    break
            module.register_forward_hook(partial(self._forward_hook, teacher=teacher))

    def _forward_hook(self, module, input, output, teacher=False):
        if teacher:
            self._teacher_out = output[0] if len(output) == 1 else output
        else:
            self._student_out = output[0] if len(output) == 1 else output


