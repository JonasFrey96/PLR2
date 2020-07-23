#!/usr/bin/env python3
"""
Originally from https://github.com/ethnhe/PVN3D

MIT License

Copyright (c) 2020 Yisheng He

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import torch
import cv2
import numpy as np
import pickle as pkl
import time

def gaussian_kernel(distance, bandwidth):
    return (1 / (bandwidth * torch.sqrt(2 * torch.tensor(np.pi)))) \
        * torch.exp(-0.5 * ((distance / bandwidth)) ** 2)


class MeanShiftTorch():
    def __init__(self, bandwidth=0.05, max_iter=300):
        self.bandwidth = bandwidth
        self.stop_thresh = bandwidth * 1e-3
        self.max_iter = max_iter

    def fit(self, A):
        """
        params: A: [N, 3]
        """
        N, c = A.size()
        it = 0
        C = A.clone()
        while True:
            it += 1
            Ar = A.view(1, N, c).repeat(N, 1, 1)
            Cr = C.view(N, 1, c).repeat(1, N, 1)
            dis = torch.norm(Cr - Ar, dim=2)
            w = gaussian_kernel(dis, self.bandwidth).view(N, N, 1)
            new_C = torch.sum(w * Ar, dim=1) / torch.sum(w, dim=1)
            # new_C = C + shift_offset
            Adis = torch.norm(new_C - C, dim=1)
            # print(C, new_C)
            C = new_C
            if torch.max(Adis) < self.stop_thresh or it > self.max_iter:
                # print("torch meanshift total iter:", it)
                break
        # find biggest cluster
        Cr = A.view(N, 1, c).repeat(1, N, 1)
        dis = torch.norm(Ar - Cr, dim=2)
        num_in = torch.sum(dis < self.bandwidth, dim=1)
        max_num, max_idx = torch.max(num_in, 0)
        labels = dis[max_idx] < self.bandwidth
        return C[max_idx, :], labels

    def fit_batch_npts(self, A):
        """
        params: A: [bs, n_kps, pts, 3]
        """
        bs, n_kps, N, cn = A.size()
        it = 0
        C = A.clone()
        while True:
            it += 1
            Ar = A.view(bs, n_kps, 1, N, cn).repeat(1, 1, N, 1, 1)
            Cr = C.view(bs, n_kps, N, 1, cn).repeat(1, 1, 1, N, 1)
            dis = torch.norm(Cr - Ar, dim=4)
            w = gaussian_kernel(dis, self.bandwidth).view(bs, n_kps, N, N, 1)
            new_C = torch.sum(w * Ar, dim=3) / torch.sum(w, dim=3)
            # new_C = C + shift_offset
            Adis = torch.norm(new_C - C, dim=3)
            # print(C, new_C)
            C = new_C
            if torch.max(Adis) < self.stop_thresh or it > self.max_iter:
                # print("torch meanshift total iter:", it)
                break
        # find biggest cluster
        Cr = A.view(N, 1, cn).repeat(1, N, 1)
        dis = torch.norm(Ar - Cr, dim=4)
        num_in = torch.sum(dis < self.bandwidth, dim=3)
        # print(num_in.size())
        max_num, max_idx = torch.max(num_in, 2)
        dis = torch.gather(dis, 2, max_idx.reshape(bs, n_kps, 1))
        labels = dis < self.bandwidth
        ctrs = torch.gather(
            C, 2, max_idx.reshape(bs, n_kps, 1, 1).repeat(1, 1, 1, cn)
        )
        return ctrs, labels

