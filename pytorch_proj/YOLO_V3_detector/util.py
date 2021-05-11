import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    """

    :param prediction: output of yolo
    :param inp_dim: input image dimensions
    :param anchors: ...
    :param num_classes: ...
    :param CUDA: CUDA flag
    :return:
    """
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    # print("stride={0} inp_dim={1} prediction.size2={2}".format(stride,inp_dim, prediction.size(2)))

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    # sigmoid x, y, obj confidence
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)
    prediction[:,:,:2] += x_y_offset

    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    prediction[:,:,5:5+num_classes] = torch.sigmoid((prediction[:,:,5:5+num_classes]))
    prediction[:,:,:4] *= stride

    return prediction


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def write_results(prediction, confidence, num_classes, nms_conf=.4):
    """

    :param prediction: |
    :param confidence: obj score threshold
    :param num_classes: |
    :param nms_conf: NMS IoU threshold
    :return:
    """
    conf_mask = (prediction[:,:,4]>confidence).float().unsequeeze(2)
    prediction = prediction*conf_mask

    # transform ceter-x, etc. to top-left, etc.
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    # the gt varies from img to img, so we here loop over dim-0(batch size)
    batch_size= prediction.size[0]
    write = False
    for ind in range(batch_size):
        # 1 img at a time
        image_pred = prediction[ind]
        # max_conf -- max number; max_conf_score -- max index
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = [image_pred[:,:,:5], max_conf, max_conf_score]
        img_pred = torch.cat(seq, 1)

        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        # try-catch block for pytorch version bugs(since .4 and later will be OK)
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue
        if image_pred_.shape == 0:
            continue

        # index(-1) stores the class index
        # unique function gives an tensor of 1-dim of the classes detected
        img_classes = unique(image_pred_[:, -1])

        for cls in img_classes:
            # perform NMS classwise