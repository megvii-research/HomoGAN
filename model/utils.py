import torch
import torch.nn as nn
from PIL import Image
#from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import os
import torch.nn.functional as F


def RescaleH(H, rescale_size, patch_size):
    batch_size = H.size()[0]
    M = torch.Tensor([[patch_size[0] / rescale_size[0], 0, 0], [0, patch_size[1] / rescale_size[1], 0], [0, 0, 1]]) \
        .view(1, 3, 3).repeat(batch_size, 1, 1).to(H.device)
    M_inv = torch.inverse(M)
    H = torch.matmul(torch.matmul(M_inv, H), M)
    return H


def WarpMat(offset, ori_size, patch_size):
    batch_size = offset.size()[0]
    offset = offset.contiguous().view(batch_size, -1, 2)

    offset[:, :, 0] = offset[:, :, 0] * (ori_size[0] / patch_size[0])
    offset[:, :, 1] = offset[:, :, 1] * (ori_size[1] / patch_size[1])

    src_pt = torch.Tensor(
        [[0, 0], [ori_size[0] - 1, 0], [0, ori_size[1] - 1], [ori_size[0] - 1, ori_size[1] - 1]]).float().to(
        offset.device).view(1, 4, 2).repeat(batch_size, 1, 1)
    dst_pt = src_pt + offset

    solution = DLT(batch_size, nums_pt=4)

    H = solution(src_pt=src_pt, dst_pt=dst_pt)

    # M = torch.Tensor([[patch_size[0] / ori_size[0], 0, 0], [0, patch_size[1] / ori_size[1], 0], [0, 0, 1]]) \
    #     .view(1, 3, 3).repeat(batch_size, 1, 1).to(offset.device)
    # M_inv = torch.inverse(M)
    # H = torch.matmul(torch.matmul(M_inv, H), M)

    offset[:, :, 0] = offset[:, :, 0] * (patch_size[0] / ori_size[0])
    offset[:, :, 1] = offset[:, :, 1] * (patch_size[1] / ori_size[1])
    return H


def Transform(H, input_map, start, patch_size, start_zero=False):
    start = start
    if start_zero:
        start = torch.zeros_like(start).to(start.device)

    warp_img, flow = WarpImages(input_map, H, start, patch_size)
    return warp_img, flow


class DLT(nn.Module):
    def __init__(self, batch_size, nums_pt=4):
        super(DLT, self).__init__()
        self.batch_size = batch_size
        self.nums_pt = nums_pt

    def forward(self, src_pt, dst_pt, method='Axb'):
        assert method in ["Ax0", "Axb"]
        self.batch_size, self.nums_pt = src_pt.shape[0], src_pt.shape[1]
        """

        :param src_pt:
        :param dst_pt:
        :param method: Axb(Full Rank Decomposition, inv_SVD) or Ax0 (SVD)
        :return:
        """
        xy1 = torch.cat((src_pt, src_pt.new_ones(self.batch_size, self.nums_pt, 1)), dim=-1)
        xyu = torch.cat((xy1, xy1.new_zeros((self.batch_size, self.nums_pt, 3))), dim=-1)
        xyd = torch.cat((xy1.new_zeros((self.batch_size, self.nums_pt, 3)), xy1), dim=-1)
        M1 = torch.cat((xyu, xyd), dim=-1).view(self.batch_size, -1, 6)
        M2 = torch.matmul(dst_pt.view(-1, 2, 1), src_pt.view(-1, 1, 2)).view(self.batch_size, -1, 2)
        M3 = dst_pt.view(self.batch_size, -1, 1)

        if method == 'Ax0':
            A = torch.cat((M1, -M2, -M3), dim=-1)
            U, S, V = torch.svd(A)
            V = V.transpose(-2, -1).conj()
            H = V[:, -1].view(self.batch_size, 3, 3)
            H *= (1 / H[:, -1, -1].view(self.batch_size, 1, 1))
            return H

        elif method == 'Axb':
            A = torch.cat((M1, -M2), dim=-1)
            B = M3
            A_inv = torch.inverse(A)
            H = torch.cat((torch.matmul(A_inv, B).view(-1, 8), src_pt.new_ones((self.batch_size, 1))), 1).view(
                self.batch_size, 3, 3)
            return H


def WarpImages(input_map, H, start, patch_size):
    def bilinear_interpote(batch_img, batch_x, batch_y, patch_size, bs, imgh, imgw, imgc):
        batch_x = torch.clamp(batch_x, min=0, max=imgw - 1).view(-1, )
        batch_y = torch.clamp(batch_y, min=0, max=imgh - 1).view(-1, )

        # select four points around the interpolated point
        batch_x0 = torch.floor(batch_x).long()
        batch_x1 = batch_x0 + 1
        batch_y0 = torch.floor(batch_y).long()
        batch_y1 = batch_y0 + 1

        #
        batch_x0 = torch.clamp(batch_x0, min=0, max=imgw - 1)
        batch_x1 = torch.clamp(batch_x1, min=0, max=imgw - 1)
        batch_y0 = torch.clamp(batch_y0, min=0, max=imgh - 1)
        batch_y1 = torch.clamp(batch_y1, min=0, max=imgh - 1)

        dim1 = imgw * imgh
        dim2 = imgw

        base = torch.arange(0, bs, dtype=torch.int).to(batch_img.device)
        base = base * dim1
        base = base.repeat_interleave(patch_size[1] * patch_size[0], axis=0)
        base_y0 = base + batch_y0 * dim2
        base_y1 = base + batch_y1 * dim2

        Ia = base_y0 + batch_x0
        Ib = base_y1 + batch_x0
        Ic = base_y0 + batch_x1
        Id = base_y1 + batch_x1

        batch_img = batch_img.reshape(-1, imgc)

        batch_pa = torch.gather(batch_img, dim=0, index=Ia.unsqueeze(-1).repeat(1, imgc))
        batch_pb = torch.gather(batch_img, dim=0, index=Ib.unsqueeze(-1).repeat(1, imgc))
        batch_pc = torch.gather(batch_img, dim=0, index=Ic.unsqueeze(-1).repeat(1, imgc))
        batch_pd = torch.gather(batch_img, dim=0, index=Id.unsqueeze(-1).repeat(1, imgc))

        # computing the weight of the four points
        wa = ((batch_x1 - batch_x) * (batch_y1 - batch_y)).view(-1, 1)
        wb = ((batch_x1 - batch_x) * (batch_y - batch_y0)).view(-1, 1)
        wc = ((batch_x - batch_x0) * (batch_y1 - batch_y)).view(-1, 1)
        wd = ((batch_x - batch_x0) * (batch_y - batch_y0)).view(-1, 1)

        warp_img = (wa * batch_pa + wb * batch_pb + wc * batch_pc + wd * batch_pd)  # * kw_mask
        return warp_img.view(bs, patch_size[1], patch_size[0], imgc)

    bs, imgc, imgh, imgw = input_map.size()
    device = input_map.device

    # transformation xy coordinates

    new_batch_xy = torch.zeros((bs, patch_size[0] * patch_size[1], 2)).to(device)
    batch_x = torch.arange(patch_size[0]).repeat(patch_size[1]).view(1, -1, 1).repeat(bs, 1, 1).to(device)
    batch_y = torch.repeat_interleave(torch.arange(patch_size[1]), repeats=patch_size[0]).view(1, -1, 1). \
        repeat(bs, 1, 1).to(device)

    goal = torch.cat([batch_x, batch_y, torch.ones_like(batch_x, dtype=torch.float).to(device)], -1).permute(0, 2, 1)

    img_pt = torch.bmm(H, goal).permute(0, 2, 1)

    small = 1e-7
    smallers = 1e-6 * (1.0 - torch.ge(torch.abs(img_pt[:, :, 2]), small).float())
    new_batch_xy[:, :, 0] = img_pt[:, :, 0] / (img_pt[:, :, 2] + smallers)
    new_batch_xy[:, :, 1] = img_pt[:, :, 1] / (img_pt[:, :, 2] + smallers)

    flow = new_batch_xy - goal.permute(0, 2, 1)[:, :, :2]
    offset_batch_xy = goal.permute(0, 2, 1)[:, :, :2] + start.view(bs, 1, 2) + flow
    new_batch_img = bilinear_interpote(input_map.permute(0, 2, 3, 1).view(bs, -1, imgc),
                                       offset_batch_xy[:, :, 0],
                                       offset_batch_xy[:, :, 1], patch_size, bs, imgh, imgw, imgc)
    new_batch_img = new_batch_img.permute(0, 3, 1, 2)
    return new_batch_img, flow.view(bs, patch_size[1], patch_size[0], 2)


def CropPatchFromFull(patch_size, full_img, start, rescale=False):
    def bilinear_interpote(batch_img, batch_x, batch_y, patch_size, bs, imgh, imgw, imgc):
        batch_x = torch.clamp(batch_x, min=0, max=imgw - 1).view(-1, )
        batch_y = torch.clamp(batch_y, min=0, max=imgh - 1).view(-1, )

        # select four points around the interpolated point
        batch_x0 = torch.floor(batch_x).long()
        batch_x1 = batch_x0 + 1
        batch_y0 = torch.floor(batch_y).long()
        batch_y1 = batch_y0 + 1

        #
        batch_x0 = torch.clamp(batch_x0, min=0, max=imgw - 1)
        batch_x1 = torch.clamp(batch_x1, min=0, max=imgw - 1)
        batch_y0 = torch.clamp(batch_y0, min=0, max=imgh - 1)
        batch_y1 = torch.clamp(batch_y1, min=0, max=imgh - 1)

        dim1 = imgw * imgh
        dim2 = imgw

        base = torch.arange(0, bs, dtype=torch.int).to(batch_img.device)
        base = base * dim1
        base = base.repeat_interleave(patch_size[1] * patch_size[0], axis=0)
        base_y0 = base + batch_y0 * dim2
        base_y1 = base + batch_y1 * dim2

        Ia = base_y0 + batch_x0
        Ib = base_y1 + batch_x0
        Ic = base_y0 + batch_x1
        Id = base_y1 + batch_x1

        batch_img = batch_img.reshape(-1, imgc)

        batch_pa = torch.gather(batch_img, dim=0, index=Ia.unsqueeze(-1).repeat(1, imgc))
        batch_pb = torch.gather(batch_img, dim=0, index=Ib.unsqueeze(-1).repeat(1, imgc))
        batch_pc = torch.gather(batch_img, dim=0, index=Ic.unsqueeze(-1).repeat(1, imgc))
        batch_pd = torch.gather(batch_img, dim=0, index=Id.unsqueeze(-1).repeat(1, imgc))

        # computing the weight of the four points
        wa = ((batch_x1 - batch_x) * (batch_y1 - batch_y)).view(-1, 1)
        wb = ((batch_x1 - batch_x) * (batch_y - batch_y0)).view(-1, 1)
        wc = ((batch_x - batch_x0) * (batch_y1 - batch_y)).view(-1, 1)
        wd = ((batch_x - batch_x0) * (batch_y - batch_y0)).view(-1, 1)

        warp_img = (wa * batch_pa + wb * batch_pb + wc * batch_pc + wd * batch_pd)  # * kw_mask
        return warp_img.view(bs, patch_size[1], patch_size[0], imgc)

    bs, c, h, w = full_img.size()
    device = full_img.device
    pw, ph = patch_size

    batch_x = torch.arange(pw).repeat(ph).view(1, -1, 1).repeat(bs, 1, 1).to(device) + \
              start[:, :, :1].repeat(1, pw * ph, 1)
    batch_y = torch.repeat_interleave(torch.arange(ph), repeats=pw).view(1, -1, 1).repeat(bs, 1, 1).to(device) + \
              start[:, :, 1:].repeat(1, pw * ph, 1)
    if rescale:
        new_batch_img = bilinear_interpote(full_img.permute(0, 2, 3, 1).view(bs, -1, c),
                                           batch_x,
                                           batch_y, patch_size, bs, h, w, c)
    else:
        batch_x = batch_x.view(-1, 1)
        batch_y = batch_y.view(-1, 1)

        full_img_flatten = full_img.permute(0, 2, 3, 1).reshape(-1, c)
        dim1 = h * w
        dim2 = w

        base = torch.arange(0, bs, dtype=torch.int).to(device).view(-1, 1)
        base = base * dim1
        base = base.repeat_interleave(pw * ph, axis=0)
        base_y0 = base + batch_y * dim2
        Ip = base_y0 + batch_x
        new_batch_img = torch.gather(full_img_flatten, dim=0, index=Ip.long().repeat(1, c))

    return new_batch_img.reshape(bs, patch_size[1], patch_size[0], c).permute(0, 3, 1, 2)


def get_grid(batch_size, H, W, start=0):
    if torch.cuda.is_available():
        xx = torch.arange(0, W).cuda()
        yy = torch.arange(0, H).cuda()
    else:
        xx = torch.arange(0, W)
        yy = torch.arange(0, H)
    xx = xx.view(1, -1).repeat(H, 1)
    yy = yy.view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    ones = torch.ones_like(xx).cuda() if torch.cuda.is_available() else torch.ones_like(xx)
    grid = torch.cat((xx, yy, ones), 1).float()

    grid[:, :2, :, :] = grid[:, :2, :, :] + start
    return grid


def get_src_p(batch_size, patch_size_h, patch_size_w, divides, axis_t=False):
    small_gap_sz = [patch_size_h // divides, patch_size_w // divides]
    mesh_num = divides + 1
    if torch.cuda.is_available():
        xx = torch.arange(0, mesh_num).cuda()
        yy = torch.arange(0, mesh_num).cuda()
    else:
        xx = torch.arange(0, mesh_num)
        yy = torch.arange(0, mesh_num)

    xx = xx.view(1, -1).repeat(mesh_num, 1)
    yy = yy.view(-1, 1).repeat(1, mesh_num)
    xx = xx.view(1, 1, mesh_num, mesh_num) * small_gap_sz[1]
    yy = yy.view(1, 1, mesh_num, mesh_num) * small_gap_sz[0]
    xx[:, :, :, -1] = xx[:, :, :, -1] - 1
    yy[:, :, -1, :] = yy[:, :, -1, :] - 1
    if axis_t:
        ones = torch.ones_like(xx).cuda() if torch.cuda.is_available() else torch.ones_like(xx)
        src_p = torch.cat((xx, yy, ones), 1).repeat(batch_size, 1, 1, 1).float()
    else:
        src_p = torch.cat((xx, yy), 1).repeat(batch_size, 1, 1, 1).float()

    return src_p


def chunk_2D(img, h_num, w_num, h_dim=2, w_dim=3):
    bs, c, h, w = img.shape
    img = img.chunk(h_num, h_dim)
    img = torch.cat(img, dim=w_dim)
    img = img.chunk(h_num * w_num, w_dim)
    return torch.cat(img, dim=1).reshape(bs, c, h_num, w_num, h // h_num, w // w_num)


def get_point_pairs(src_p, divide):  # src_p: shape=(bs, 2, h, w)
    bs = src_p.shape[0]
    src_p = src_p.repeat_interleave(2, axis=2).repeat_interleave(2, axis=3)
    src_p = src_p[:, :, 1:-1, 1:-1]
    src_p = chunk_2D(src_p, divide, divide).reshape(bs, -1, 2, 2, 2)
    src_p = src_p.permute(0, 1, 3, 4, 2).reshape(bs, divide * divide, 4, 2)
    return src_p


def DLT_solve(src_p, off_set):
    bs, _, divide = src_p.shape[:3]
    divide = divide - 1

    src_ps = get_point_pairs(src_p, divide)
    off_sets = get_point_pairs(off_set, divide)

    bs, n, h, w = src_ps.shape
    N = bs * n

    src_ps = src_ps.reshape(N, h, w)
    off_sets = off_sets.reshape(N, h, w)

    dst_p = src_ps + off_sets

    ones = torch.ones(N, 4, 1).cuda() if torch.cuda.is_available() else torch.ones(N, 4, 1)
    xy1 = torch.cat((src_ps, ones), axis=2)
    zeros = torch.zeros_like(xy1).cuda() if torch.cuda.is_available() else torch.zeros_like(xy1)
    xyu, xyd = torch.cat((xy1, zeros), axis=2), torch.cat((zeros, xy1), axis=2)

    M1 = torch.cat((xyu, xyd), axis=2).reshape(N, -1, 6)
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1),
        src_ps.reshape(-1, 1, 2),
    ).reshape(N, -1, 2)

    A = torch.cat((M1, -M2), axis=2)
    b = dst_p.reshape(N, -1, 1)

    Ainv = torch.inverse(A)

    h8 = torch.matmul(Ainv, b).reshape(N, 8)

    H = torch.cat((h8, ones[:, 0, :]), axis=1).reshape(N, 3, 3)
    H = H.reshape(bs, n, 3, 3)
    return H


def get_flow(H_mat_mul, patch_indices, patch_size_h, patch_size_w, divide, point_use=False):
    batch_size = H_mat_mul.shape[0]
    small_gap_sz = [patch_size_h // divide, patch_size_w // divide]

    small = 1e-7

    H_mat_pool = H_mat_mul.reshape(batch_size, divide, divide, 3, 3)  # .transpose(2,1)
    H_mat_pool = H_mat_pool.repeat_interleave(small_gap_sz[0], axis=1).repeat_interleave(small_gap_sz[1], axis=2)

    if point_use and H_mat_pool.shape[2] != patch_indices.shape[2]:
        H_mat_pool = H_mat_pool.permute(0, 3, 4, 1, 2).contiguous()
        H_mat_pool = F.pad(H_mat_pool, pad=(0, 1, 0, 1, 0, 0), mode="replicate")
        H_mat_pool = H_mat_pool.permute(0, 3, 4, 1, 2).contiguous()

    pred_I2_index_warp = patch_indices.permute(0, 2, 3, 1).unsqueeze(4).contiguous()

    pred_I2_index_warp = torch.matmul(H_mat_pool, pred_I2_index_warp).squeeze(-1).permute(0, 3, 1, 2).contiguous()
    T_t = pred_I2_index_warp[:, 2:3, ...]
    smallers = 1e-6 * (1.0 - torch.ge(torch.abs(T_t), small).float())
    T_t = T_t + smallers  #
    v1 = pred_I2_index_warp[:, 0:1, ...]
    v2 = pred_I2_index_warp[:, 1:2, ...]
    v1 = v1 / T_t
    v2 = v2 / T_t
    pred_I2_index_warp = torch.cat((v1, v2), 1)
    vgrid = patch_indices[:, :2, ...]

    flow = pred_I2_index_warp - vgrid
    return flow, vgrid


def transformer(I, vgrid, train=True):
    # I: Img, shape: batch_size, 1, full_h, full_w
    # vgrid: vgrid, target->source, shape: batch_size, 2, patch_h, patch_w
    # outsize: (patch_h, patch_w)

    def _interpolate(im, x, y, out_size, scale_h):
        # x: x_grid_flat
        # y: y_grid_flat
        # out_size: same as im.size
        # scale_h: True if normalized
        # constants
        num_batch, num_channels, height, width = im.size()

        out_height, out_width = out_size[0], out_size[1]
        # zero = torch.zeros_like([],dtype='int32')
        zero = 0
        max_y = height - 1
        max_x = width - 1

        # do sampling
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)  # same as np.clip
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)

        dim1 = width * height
        dim2 = width

        if torch.cuda.is_available():
            base = torch.arange(0, num_batch).int().cuda()
        else:
            base = torch.arange(0, num_batch).int()

        base = base * dim1
        base = base.repeat_interleave(out_height * out_width, axis=0)  
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im = im.permute(0, 2, 3, 1).contiguous()
        im_flat = im.reshape([-1, num_channels]).float()

        idx_a = idx_a.unsqueeze(-1).long()
        idx_a = idx_a.expand(out_height * out_width * num_batch, num_channels)
        Ia = torch.gather(im_flat, 0, idx_a)

        idx_b = idx_b.unsqueeze(-1).long()
        idx_b = idx_b.expand(out_height * out_width * num_batch, num_channels)
        Ib = torch.gather(im_flat, 0, idx_b)

        idx_c = idx_c.unsqueeze(-1).long()
        idx_c = idx_c.expand(out_height * out_width * num_batch, num_channels)
        Ic = torch.gather(im_flat, 0, idx_c)

        idx_d = idx_d.unsqueeze(-1).long()
        idx_d = idx_d.expand(out_height * out_width * num_batch, num_channels)
        Id = torch.gather(im_flat, 0, idx_d)

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return output

    def _transform(I, vgrid, scale_h):

        C_img = I.shape[1]
        B, C, H, W = vgrid.size()

        x_s_flat = vgrid[:, 0, ...].reshape([-1])
        y_s_flat = vgrid[:, 1, ...].reshape([-1])
        out_size = vgrid.shape[2:]
        input_transformed = _interpolate(I, x_s_flat, y_s_flat, out_size, scale_h)

        output = input_transformed.reshape([B, H, W, C_img])
        return output

    # scale_h = True
    output = _transform(I, vgrid, scale_h=False)
    if train:
        output = output.permute(0, 3, 1, 2).contiguous()
    return output


def get_warp_flow(img, flow, start=0):
    batch_size, _, patch_size_h, patch_size_w = flow.shape
    grid_warp = get_grid(batch_size, patch_size_h, patch_size_w, start)[:, :2, :, :] + flow
    img_warp = transformer(img, grid_warp)
    return img_warp


def upsample2d_flow_as(inputs, target_as, mode="bilinear", if_rate=False):
    _, _, h, w = target_as.size()
    if if_rate:
        _, _, h_, w_ = inputs.size()
        inputs[:, 0, :, :] *= (w / w_)
        inputs[:, 1, :, :] *= (h / h_)
    res = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    return res

