import numpy as np
from scipy.ndimage import filters, measurements, interpolation
from math import pi
import math

def imresize(img, scale=None, output_shape=None, kernel=None, antialiasing=True, kernel_shift_flag=False, ds_method='direct'):
    # First standardize values and fill missing arguments (if needed) by deriving scale from output shape or vice versa
    scale, output_shape = fix_scale_and_size(img.shape, output_shape, scale)

    # For a given numeric kernel case, just do convolution and sub-sampling (downscaling only)
    if type(kernel) == np.ndarray and scale <= 1:
        if ds_method=='direct':
            return numeric_kernel_dir(img, kernel, scale, output_shape, kernel_shift_flag)
        elif ds_method=='bicubic':
            return numeric_kernel_bic(img, kernel, scale, output_shape, kernel_shift_flag)
        else:
            raise ValueError('Downscaling method should be \'direct\' or \'bicubic\'.')

    method, kernel_width = {
        "cubic": (cubic, 4.0),
        "lanczos2": (lanczos2, 4.0),
        "lanczos3": (lanczos3, 6.0),
        "linear": (linear, 2.0),
        None: (cubic, 4.0)  # set default interpolation method as cubic
    }.get(kernel)

    img = img.transpose(2, 0, 1)
    in_C, in_H, in_W = img.shape
    out_C, out_H, out_W =in_C, output_shape[0], output_shape[1]
    # out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, method, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, method, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = np.zeros([in_C, in_H + sym_len_Hs + sym_len_He, in_W])
    img_aug[:, sym_len_Hs:sym_len_Hs+in_H,:]=img

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = np.arange(sym_patch.shape[1] - 1, -1, -1).astype(np.int64)
    sym_patch_inv = np.take(sym_patch,inv_idx, axis=1)
    img_aug[:,0:0+sym_len_Hs,:]=sym_patch_inv

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = np.arange(sym_patch.shape[1] - 1, -1, -1).astype(np.int64)
    sym_patch_inv = np.take(sym_patch, inv_idx, axis=1)
    img_aug[:,sym_len_Hs + in_H:sym_len_Hs + in_H+sym_len_He,:]=sym_patch_inv

    out_1 = np.zeros([in_C, out_H, in_W])
    kernel_width = weights_H.shape[1]
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[0, i, :] = np.matmul(img_aug[0, idx:idx + kernel_width, :].transpose(1, 0),(weights_H[i]))
        out_1[1, i, :] = np.matmul(img_aug[1, idx:idx + kernel_width, :].transpose(1, 0),(weights_H[i]))
        out_1[2, i, :] = np.matmul(img_aug[2, idx:idx + kernel_width, :].transpose(1, 0),(weights_H[i]))

    # process W dimension
    # symmetric copying
    out_1_aug = np.zeros([in_C, out_H, in_W + sym_len_Ws + sym_len_We])
    out_1_aug[:,:,sym_len_Ws:sym_len_Ws+in_W]=out_1

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = np.arange(sym_patch.shape[2] - 1, -1, -1).astype(np.int64)
    sym_patch_inv = np.take(sym_patch, inv_idx, axis=2)
    out_1_aug[:,:,0: 0+sym_len_Ws]=sym_patch_inv

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = np.arange(sym_patch.shape[2] - 1, -1, -1).astype(np.int64)
    sym_patch_inv = np.take(sym_patch, inv_idx,axis=2)
    out_1_aug[:,:,sym_len_Ws + in_W:sym_len_Ws + in_W+ sym_len_We]=sym_patch_inv

    out_2 = np.zeros([in_C, out_H, out_W])
    kernel_width = weights_W.shape[1]
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[0, :, i] = np.matmul(out_1_aug[0, :, idx:idx + kernel_width],(weights_W[i]))
        out_2[1, :, i] = np.matmul(out_1_aug[1, :, idx:idx + kernel_width],(weights_W[i]))
        out_2[2, :, i] = np.matmul(out_1_aug[2, :, idx:idx + kernel_width],(weights_W[i]))

    out_2=out_2.transpose(1,2,0)
    return out_2

# For predefined interpolation methods.
def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):

    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = np.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = np.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = np.broadcast_to(left.reshape(out_length, 1),[out_length, P]) + np.broadcast_to(np.linspace(0, P - 1, P).reshape(
        1, P),[out_length, P])

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = np.broadcast_to(u.reshape(out_length, 1),[out_length, P])- indices

    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * kernel(distance_to_center * scale)
    else:
        weights = kernel(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = np.sum(weights, 1).reshape(out_length, 1)
    weights = np.broadcast_to(weights / weights_sum, [out_length, P])

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = np.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices[:, 1:P - 2+1]
        weights = weights[:, 1:P - 2+1]
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices[:,0:P - 2+1]
        weights = weights[:,0: P - 2+1]

    sym_len_s = -np.min(indices) + 1
    sym_len_e = np.max(indices) - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)

# To get scale and output shape
def fix_scale_and_size(input_shape, output_shape, scale_factor):
    # Fixing output-shape (if given): extending it to the size of the input-shape, by assigning the original input-size
    # to all the unspecified dimensions
    if scale_factor is not None:
        # By default, if scale-factor is a scalar we assume 2d resizing and duplicate it.
        if np.isscalar(scale_factor):
            scale_factor = [scale_factor, scale_factor]

        # We extend the size of scale-factor list to the size of the input by assigning 1 to all the unspecified scales
        scale_factor = list(scale_factor)
        scale_factor.extend([1] * (len(input_shape) - len(scale_factor)))

    if output_shape is not None:
        output_shape = list(np.uint(np.array(output_shape))) + list(input_shape[len(output_shape):])

    # Dealing with the case of non-give scale-factor, calculating according to output-shape. note that this is
    # sub-optimal, because there can be different scales to the same output-shape.
    if scale_factor is None:
        scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)

    # Dealing with missing output-shape. calculating according to scale-factor
    if output_shape is None:
        output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))

    scale_factor=scale_factor[0]
    return scale_factor, output_shape

# For arbitrary kernel with .mat file
def numeric_kernel_dir(im, kernel, scale_factor, output_shape, kernel_shift_flag):
    # See kernel_shift function to understand what this is
    if kernel_shift_flag:
        kernel = kernel_shift(kernel, scale_factor)

    # First run a correlation (convolution with flipped kernel)
    out_im = np.zeros_like(im)
    for channel in range(np.ndim(im)):
        out_im[:, :, channel] = filters.correlate(im[:, :, channel], kernel)

    # Then subsample and return
    return out_im[np.round(np.linspace(0, im.shape[0] - 1 / scale_factor, output_shape[0])).astype(int)[:, None],
                  np.round(np.linspace(0, im.shape[1] - 1 / scale_factor, output_shape[1])).astype(int), :]

def numeric_kernel_bic(im, kernel, scale_factor, output_shape, kernel_shift_flag):
    # See kernel_shift function to understand what this is
    if kernel_shift_flag:
        kernel = kernel_shift(kernel, scale_factor)

    # First run a correlation (convolution with flipped kernel)
    out_im = np.zeros_like(im)
    for channel in range(np.ndim(im)):
        out_im[:, :, channel] = filters.correlate(im[:, :, channel], kernel)

    # Then subsample and return
    return imresize(out_im, scale_factor, output_shape, kernel='cubic')

def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel:
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between od and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (sf - (kernel.shape[0] % 2))

    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass

    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    kernel = np.pad(kernel, np.int(np.ceil(np.max(shift_vec))) + 1, 'constant')

    # Finally shift the kernel and return
    return interpolation.shift(kernel, shift_vec)

# These next functions are all interpolation methods. x is the distance from the left pixel center
def cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5*absx3 - 2.5*absx2 + 1) * (absx <= 1) +
            (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * ((1 < absx) & (absx <= 2)))

def lanczos2(x):
    return (((np.sin(pi*x) * np.sin(pi*x/2) + np.finfo(np.float32).eps) /
             ((pi**2 * x**2 / 2) + np.finfo(np.float32).eps))
            * (abs(x) < 2))

def lanczos3(x):
    return (((np.sin(pi*x) * np.sin(pi*x/3) + np.finfo(np.float32).eps) /
            ((pi**2 * x**2 / 3) + np.finfo(np.float32).eps))
            * (abs(x) < 3))

def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))
