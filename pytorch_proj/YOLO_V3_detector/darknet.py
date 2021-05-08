import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import *

def get_test_inpuy():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))
    img_ = img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = img_.cuda()
    return img_


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.bolcks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.bolcks)

    def forward(self, x, cuda):
        print(x.shape)
        modules = self.bolcks[1:]
        outputs = {}
        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])
            if (module_type == "convolutional" or module_type == "upsample"):
                x = self.module_list[i](x)
            elif(module_type == "route"):
                layers = module["layers"]
                layers = [int(a) for a in layers]
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
            elif module_type == "shortcut":
                from_ = int (module["from"])
                x = outputs[i-1] + outputs[i+from_]
            elif(module_type == "yolo"):
                anchors = self.module_list[i][0].anchors
                # print(anchors)
                # print("*"*10)
                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, cuda)
                if not write:
                    detections = x
                    write = 1
                    # print(detections.shape)
                else:
                    detections = torch.cat((detections, x), 1)
                    # print(detections.shape)
            outputs[i] = x
        return detections


def parse_cfg(cfg_file):
    """
    :param cfg_file: the path of cfg file
    :return: the blocks of neural network in the type of list
    """
    file = open(cfg_file, 'r')
    lines = file.read().split('\n')                     # 分行
    lines = [x for x in lines if len(x)>0]              # 去掉空白行
    lines = [x for x in lines if x[0] != '#']           # 去掉注释
    lines = [x.rstrip().lstrip() for x in lines]        # 去掉左右空白

    block = {}
    blocks = []
    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks


def create_modules(blocks):
    """
    :param blocks: the output of parse_cfg
    :return: nn.ModuleList
    """
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        #get the info of conv layer
        if(x['type']=="convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size -1) // 2
            else:
                pad = 0

            # add the conv layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride,
                             pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)
            # add the bn layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
            # add the activation layer
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

        # get&add the upsampling layer
        elif(x["type"]=="upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="bilinear")
            module.add_module("upsample %d"%index, upsample)

        # get&add the route layer, just stores the numbers, the concatenate operation will be
        # manually called in forward function
        elif(x["type"] == "route"):
            x["layers"] = x["layers"].split(",")
            #start of a route
            start = int(x["layers"][0])
            #end of a route(if exists)
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            if start >0:
                start -= index
            if end >0:
                end -=index
            route  = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end<0:
                filters = output_filters[index+start] + output_filters[index+end]
            else:
                filters = output_filters[index+start]

        # shortcut (skip connection)
        elif(x["type"] == "shortcut"):
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        #YOLO layer
        elif[x["type"] == "yolo"]:
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        # restore the module and the filter number
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


model = Darknet("cfg/yolo_v3.cfg")
model = model.cuda()
inp = get_test_inpuy()
pred = model(inp, torch.cuda.is_available())
print(pred.shape)