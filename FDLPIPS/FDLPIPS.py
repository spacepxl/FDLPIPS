import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import VGG


class FDLPIPS_2D(torch.nn.Module):
    def __init__(
        self,
        patch_size = [5, 5],
        stride = 1,
        num_proj = 64,
        phase_weight = 0,
        fdl_weights = [1.0, 1.0, 1.0, 1.0, 1.0],
        lpips_weights = [1.0, 1.0, 1.0, 1.0, 1.0],
        image_range = [0.0, 1.0],
    ):
        """
        patch_size, stride, num_proj: SWD slice parameters
        phase_weight: weight for phase branch
        fdl_weights: weight per layer of FDL loss
        lpips_weights: weight per layer of LPIPS loss
        image_range: minimum and maximum expected color values of input images
        """
        super().__init__()
        self.stride = stride
        self.phase_weight = phase_weight
        self.fdl_weights = fdl_weights
        self.lpips_weights = lpips_weights
        
        # torchvision models expect [0, 1] to then apply the standard imagenet norm
        self.shift = -image_range[0]
        self.scale = 1 / (image_range[1] - image_range[0])
        
        self.model = VGG()
        
        for idx in range(len(self.model.chns)):
            rand = torch.randn(num_proj, self.model.chns[idx], *patch_size)
            rand = rand / rand.view(rand.shape[0], -1).norm(dim=1)[:, None, None, None]
            self.register_buffer(f"rand_{idx}", rand)

    def resample_projections(self):
        for idx in range(len(self.model.chns)):
            rand = getattr(self, f"rand_{idx}")
            rand_new = torch.randn_like(rand)
            rand_new = rand_new / rand_new.view(rand_new.shape[0], -1).norm(dim=1)[:, None, None, None]
            rand.copy_(rand_new)

    def forward_once(self, x, y, idx):
        """
        x, y: input image tensors with the shape of (N, C, H, W)
        """
        rand = getattr(self, f"rand_{idx}")
        projx = F.conv2d(x, rand, stride=self.stride).flatten(2)
        projy = F.conv2d(y, rand, stride=self.stride).flatten(2)

        # sort the convolved input
        projx = torch.sort(projx, dim=-1)[0]
        projy = torch.sort(projy, dim=-1)[0]

        # compute the mean of the sorted convolved input
        s = torch.abs(projx - projy).mean([1, 2])
        return s

    def forward(self, x, y, reduce_batch=True, resample=False):
        if resample:
            self.resample_projections()
        
        x = self.model((x + self.shift) * self.scale)
        y = self.model((y + self.shift) * self.scale)
        
        fdl_score = []
        lpips_score = []
        for idx in range(len(x)):
            # Transform to Fourier Space
            fft_x = torch.fft.fftn(x[idx], dim=(-2, -1))
            fft_y = torch.fft.fftn(y[idx], dim=(-2, -1))

            # get the magnitude and phase of the extracted features
            x_mag = torch.abs(fft_x)
            y_mag = torch.abs(fft_y)
            s_amplitude = self.forward_once(x_mag, y_mag, idx)

            if self.phase_weight > 0:
                x_phase = torch.angle(fft_x)
                y_phase = torch.angle(fft_y)
                s_phase = self.forward_once(x_phase, y_phase, idx)
                s_amplitude += s_phase * self.phase_weight

            fdl_score.append(s_amplitude * self.fdl_weights[idx])

            # Calculate LPIPS, reusing model features
            x_norm = F.normalize(x[idx], p=2, dim=1, eps=1e-10).flatten(1)
            y_norm = F.normalize(y[idx], p=2, dim=1, eps=1e-10).flatten(1)
            lpips_score.append(torch.abs(x_norm - y_norm).mean(1) * self.lpips_weights[idx])

        fdl_score = sum(fdl_score) * 0.001 # arbitrary scale to similar range as lpips
        lpips_score = sum(lpips_score)
        
        if reduce_batch:
            fdl_score = fdl_score.mean()
            lpips_score = lpips_score.mean()
        
        return fdl_score, lpips_score


class FDLPIPS_3D(torch.nn.Module):
    def __init__(
        self,
        patch_size = [5, 5, 5],
        stride = 1,
        num_proj = 64,
        phase_weight = 0,
        fdl_weights = [1.0, 1.0, 1.0, 1.0, 1.0],
        lpips_weights = [1.0, 1.0, 1.0, 1.0, 1.0],
        image_range = [0.0, 1.0],
    ):
        """
        patch_size, stride, num_proj: SWD slice parameters
        phase_weight: weight for phase branch
        fdl_weights: weight per layer of FDL loss
        lpips_weights: weight per layer of LPIPS loss
        image_range: minimum and maximum expected color values of input images
        """
        super().__init__()
        self.stride = stride
        self.phase_weight = phase_weight
        self.fdl_weights = fdl_weights
        self.lpips_weights = lpips_weights
        
        # torchvision models expect [0, 1] to then apply the standard imagenet norm
        self.shift = -image_range[0]
        self.scale = 1 / (image_range[1] - image_range[0])
        
        self.model = VGG()
        
        for idx in range(len(self.model.chns)):
            rand = torch.randn(num_proj, self.model.chns[idx], *patch_size)
            rand = rand / rand.view(rand.shape[0], -1).norm(dim=1)[:, None, None, None, None]
            self.register_buffer(f"rand_{idx}", rand)

    def resample_projections(self):
        for idx in range(len(self.model.chns)):
            rand = getattr(self, f"rand_{idx}")
            rand_new = torch.randn_like(rand)
            rand_new = rand_new / rand_new.view(rand_new.shape[0], -1).norm(dim=1)[:, None, None, None, None]
            rand.copy_(rand_new)

    def forward_once(self, x, y, idx):
        """
        x, y: input image tensors with the shape of (N, C, F, H, W)
        """
        rand = getattr(self, f"rand_{idx}")
        projx = F.conv3d(x, rand, stride=self.stride).flatten(2)
        projy = F.conv3d(y, rand, stride=self.stride).flatten(2)

        # sort the convolved input
        projx = torch.sort(projx, dim=-1)[0]
        projy = torch.sort(projy, dim=-1)[0]

        # compute the mean of the sorted convolved input
        s = torch.abs(projx - projy).mean([1, 2])
        return s

    def forward(self, x, y, reduce_batch=True, resample=False):
        if resample:
            self.resample_projections()
        
        frames_x = []
        frames_y = []
        for frame in range(x.shape[-3]):
            frames_x.append(self.model((x[:, :, frame] + self.shift) * self.scale))
            frames_y.append(self.model((y[:, :, frame] + self.shift) * self.scale))
        
        x = []
        y = []
        for idx in range(len(self.model.chns)):
            x.append(torch.stack([frame[idx] for frame in frames_x], dim=2))
            y.append(torch.stack([frame[idx] for frame in frames_y], dim=2))
        
        fdl_score = []
        lpips_score = []
        for idx in range(len(x)):
            # Transform to Fourier Space
            fft_x = torch.fft.fftn(x[idx], dim=(-3, -2, -1))
            fft_y = torch.fft.fftn(y[idx], dim=(-3, -2, -1))

            # get the magnitude and phase of the extracted features
            x_mag = torch.abs(fft_x)
            y_mag = torch.abs(fft_y)
            s_amplitude = self.forward_once(x_mag, y_mag, idx)

            if self.phase_weight > 0:
                x_phase = torch.angle(fft_x)
                y_phase = torch.angle(fft_y)
                s_phase = self.forward_once(x_phase, y_phase, idx)
                s_amplitude += s_phase * self.phase_weight

            fdl_score.append(s_amplitude * self.fdl_weights[idx])

            # Calculate LPIPS, reusing model features
            x_norm = F.normalize(x[idx], p=2, dim=1, eps=1e-10).flatten(1)
            y_norm = F.normalize(y[idx], p=2, dim=1, eps=1e-10).flatten(1)
            lpips_score.append(torch.abs(x_norm - y_norm).mean(1) * self.lpips_weights[idx])

        fdl_score = sum(fdl_score) * 0.001 # arbitrary scale to similar range as lpips
        lpips_score = sum(lpips_score)
        
        if reduce_batch:
            fdl_score = fdl_score.mean()
            lpips_score = lpips_score.mean()
        
        return fdl_score, lpips_score


# if __name__ == '__main__':
    # print("FDLPIPS_2D")
    # X = torch.rand((4, 3, 128, 128)).cuda()
    # Y = torch.rand((4, 3, 128, 128)).cuda()
    # loss = FDLPIPS_2D().cuda()
    # c = loss(X,Y)
    # print('loss:', c)
    # c = loss(X,Y, reduce_batch=False)
    # print('loss:', c)
    
    
    # print("FDLPIPS_3D")
    # X = torch.rand((1, 3, 13, 128, 128)).cuda()
    # Y = torch.rand((1, 3, 13, 128, 128)).cuda()
    # loss = FDLPIPS_3D().cuda()
    # c = loss(X,Y)
    # print('loss:', c)