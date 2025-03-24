# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn

import numpy as np
import math
from scipy.signal import convolve2d

pad = 2

def demosaicImage(image, method):
    ''' Demosaics image.

    Args:
        img: np.array of size NxM.
        method: demosaicing method (baseline or nn).

    Returns:
        Color image of size NxMx3 computed using method.
    '''

    if method.lower() == "baseline":
        return demosaicBaseline(image.copy())
    elif method.lower() == 'nn':
        return demosaicNN(image.copy()) # Implement this
    elif method.lower() == 'linear':
        return demosaicLinear(image.copy()) # Implement this
    elif method.lower() == 'adagrad':
        return demosaicAdagrad(image.copy()) # Implement this
    elif method.lower() == 'transf_linear':
        return demosaicTransformLinear(image.copy())
    elif method.lower() == 'transf_log':
        return demosaicTransformLog(image.copy())
    else:
        raise ValueError("method {} unkown.".format(method))


def demosaicBaseline(img):
    '''Baseline demosaicing.
    
    Replaces missing values with the mean of each color channel.
    
    Args:
        img: np.array of size NxM.

    Returns:
        Color image of sieze NxMx3 demosaiced using the baseline 
        algorithm.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape

    red_values = img[1:image_height:2, 1:image_width:2]
    mean_value = red_values.mean()
    mos_img[:, :, 0] = mean_value
    mos_img[1:image_height:2, 1:image_width:2, 0] = img[1:image_height:2, 1:image_width:2]

    blue_values = img[0:image_height:2, 0:image_width:2]
    mean_value = blue_values.mean()
    mos_img[:, :, 2] = mean_value
    mos_img[0:image_height:2, 0:image_width:2, 2] = img[0:image_height:2, 0:image_width:2]

    mask = np.ones((image_height, image_width))
    mask[0:image_height:2, 0:image_width:2] = -1
    mask[1:image_height:2, 1:image_width:2] = -1
    green_values = mos_img[mask > 0]
    mean_value = green_values.mean()

    green_channel = img
    green_channel[mask < 0] = mean_value
    mos_img[:, :, 1] = green_channel

    return mos_img

def getRGBChannels(img, mode = None, returnBayer=False):

    if mode : img = np.pad(img, pad_width=pad, mode=mode)
    else:  img = np.pad(img, pad_width=pad)

    imageHeight, imageWidth = img.shape

    bayer_red = np.tile([[0, 0], [0, 1]], [math.ceil(imageHeight / 2), math.ceil(imageWidth / 2)])
    bayer_blue = np.tile([[1, 0], [0, 0]], [math.ceil(imageHeight / 2), math.ceil(imageWidth / 2)])
    bayer_green = np.tile([[0, 1], [1, 0]], [math.ceil(imageHeight / 2), math.ceil(imageWidth / 2)])

    if imageHeight < math.ceil(imageHeight / 2) * 2:
        bayer_red = bayer_red[:-1, :]
        bayer_blue = bayer_blue[:-1, :]
        bayer_green = bayer_green[:-1, :]

    if imageWidth < math.ceil(imageWidth / 2) * 2:
        bayer_red = bayer_red[:, :-1]
        bayer_blue = bayer_blue[:, :-1]
        bayer_green = bayer_green[:, :-1]

    red_image = img * bayer_red
    blue_image = img * bayer_blue
    green_image = img * bayer_green

    if not returnBayer:
        return red_image, blue_image, green_image
    else:
        return red_image, blue_image, green_image, bayer_red, bayer_blue

def combine_image(red, green, blue):
    imageHeight, imageWidth = blue.shape
    image = np.zeros((imageHeight, imageWidth, 3))
    image[:, :, 0] = red
    image[:, :, 1] = green
    image[:, :, 2] = blue
    image[image > 1] = 1

    image = image[pad:imageHeight - pad, pad:imageWidth - pad, :]
    return image


def demosaicNN(img):
    '''Nearest neighbor demosaicing.
    
    Args:
        img: np.array of size NxM.
    '''

    red_image, blue_image, green_image = getRGBChannels(img, 'reflect')

    green = green_image + convolve2d(green_image, [[0, 1]], mode='same')

    channelImages = [red_image, blue_image]
    for channelImage in channelImages:
        image_1 = convolve2d(channelImage, [[0, 1], [1, 0]], mode='same')
        image_2 = convolve2d(channelImage, [[0, 0], [0, 1]], mode='same')
        channelImage += image_1 + image_2

    return combine_image(channelImages[0], green, channelImages[1])

def convolveChannelImage(channelImage):
    image_1 = convolve2d(channelImage, np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]) / 4, mode='same')
    image_2 = convolve2d(channelImage, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 2, mode='same')
    return image_1 + image_2

def convolveGreenImage(greenImage):
    return convolve2d(greenImage, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4, mode='same')

def demosaicLinear(img):
    '''Linear demosaicing.
    
    Args:
        img: np.array of size NxM.
    '''

    red_image, blue_image, green_image = getRGBChannels(img, 'linear_ramp')

    green = green_image + convolveGreenImage(green_image)

    channelImages = [red_image, blue_image]
    for channelImage in channelImages:
        channelImage += convolveChannelImage(channelImage)

    return combine_image(channelImages[0], green, channelImages[1])

def runGradient(channelImage, green, img_padded, start, imageHeight, imageWidth):

    for x in range(start, imageHeight - pad, 2):
        for y in range(start, imageWidth - pad, 2):

            h_grad = abs((channelImage[x, y - pad] + channelImage[x, y + pad]) / 2 - channelImage[x, y])

            v_grad = abs((channelImage[x - pad, y] + channelImage[x + pad, y]) / 2 - channelImage[x, y])

            if h_grad < v_grad:
                img_padded[x, y, 1] = (green[x, y - 1] + green[x, y + 1]) / 2

            elif h_grad > v_grad:
                img_padded[x, y, 1] = (green[x - 1, y] + green[x + 1, y]) / 2

            else:
                img_padded[x, y, 1] = (green[x - 1, y] + green[x + 1, y] + green[x, y - 1] + green[x, y + 1]) / 4

    return channelImage + convolveChannelImage(channelImage)

def demosaicAdagrad(img):
    '''Adaptive Gradient demosaicing.
    
    Args:
        img: np.array of size NxM.
    '''

    imageHeight, imageWidth = img.shape

    img_padded = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    img_padded = np.pad(img_padded, pad_width=pad, mode='reflect')

    red_image, blue_image, green_image = getRGBChannels(img, 'reflect')

    red = runGradient(red_image, green_image, img_padded, 0, imageHeight, imageWidth)

    blue = runGradient(blue_image, green_image, img_padded, 1, imageHeight, imageWidth)

    return combine_image(red, img_padded[:, :, 1], blue)


### Extra Credit
def fillMinValue(image):
    min_image = np.min(image[image != 0])
    image[image == 0] = min_image
    return image

### Extra Credit
def demosaicTransformLinear(img):
    '''Transformed linear demosaicing.

       Args:
           img: np.array of size NxM.
       '''

    red_image, blue_image, green_image, bayer_red, bayer_blue = getRGBChannels(img, 'reflect', returnBayer=True)

    green = green_image + convolveGreenImage(green_image)

    green = fillMinValue(green)

    red_image /= green
    red_image *= bayer_red
    red = red_image + convolveChannelImage(red_image)

    blue_image /= green
    blue_image *= bayer_blue
    blue = blue_image + convolveChannelImage(blue_image)

    return combine_image(red * green, green, blue * green)

### Extra Credit
def demosaicTransformLog(img):
    '''Transformed Logarithmic demosaicing.

       Args:
           img: np.array of size NxM.
       '''
    red_image, blue_image, green_image, bayer_red, bayer_blue = getRGBChannels(img, 'reflect', returnBayer=True)

    green = green_image + convolveGreenImage(green_image)

    green = fillMinValue(green)

    blue_image = fillMinValue(blue_image)
    blue_image = np.log(blue_image / green)
    blue_image *= bayer_blue
    blue = blue_image + convolveChannelImage(blue_image)

    red_image = fillMinValue(red_image)
    red_image = np.log(red_image / green)
    red_image *= bayer_red
    red = red_image + convolveChannelImage(red_image)

    return combine_image(np.exp(red) * green, green, np.exp(blue) * green)