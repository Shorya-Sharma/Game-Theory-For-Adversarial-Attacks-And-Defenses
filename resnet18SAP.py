import random
from torch.distributions.multinomial import Multinomial

class SAP(nn.Module):
    """SimpleModel represents a nn.Module of Stochastic Activation Pruning.
    The original paper is https://arxiv.org/abs/1803.01442.
    Attributes
    ----------
    self.ratio float : ratio of pruning which can be larger than 1.0.
    self.is_valid bool : if this flag is True, inject SAP.
    """
    def __init__(self, ratio=1, is_valid=False):
        """
        Parameters
        ----------
        ratio float : ratio of pruning which can be larger than 1.0.
        is_valid bool : if this flag is True, inject SAP.
        """
        super(SAP, self).__init__()
        self.ratio = ratio
        self.is_valid = is_valid

    def forward(self, inputs):
        """
        If self.training or not self.is_valid, just return inputs.
        If self.is_valid apply SAP to inputs and return the result tensor.
        Parameters
        ----------
        inputs torch.Tensor : input tensor whose shape is [b, c, h, w].
        Returns
        -------
        outputs torch.Tensor : just return inputs or stochastically pruned inputs.
        """
        # print("SAP: ", self.is_valid)
        # if self.training or not self.is_valid:
        if not self.is_valid:
            return inputs
        else:
            b, c, h, w = inputs.shape
            inputs_1d = inputs.reshape([b, c * h * w])  # [b, c * h * w]
            # print(inputs_1d)
            outputs = torch.zeros_like(inputs_1d)  # outputs with 0 initilization
           
            inputs_1d_sum = torch.sum(torch.abs(inputs_1d), dim=-1, keepdim=True)
            inputs_1d_prob = torch.abs(inputs_1d) / inputs_1d_sum
            
            # r: num_nodes
            num_sample = int(c * h * w * self.ratio)  

            # multinomial(total_count:int, probs:tensor, logits:tensor)
            idx = Multinomial(num_sample, inputs_1d_prob).sample()

            # if nonzero, keep; else, drop, let be zeroes
            outputs[idx.nonzero(as_tuple=True)] = inputs_1d[idx.nonzero(as_tuple=True)]

            # pdb.set_trace()
            # scale up
            outputs = outputs / (1 - (1-inputs_1d_prob)**num_sample + 1e-12)
            outputs = outputs.reshape([b, c, h, w])  # [b, c, h, w]
            # print("OUT: ", outputs)
        return outputs

    
class BasicBlockSAP(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_valid=False):
        super(BasicBlockSAP, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.sap1 = SAP(is_valid=is_valid)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.sap1(out)
        return out

    
class ResNetSAP(nn.Module):
    """Model represents a model mainly used in experiments.
    Attributes
    ----------
    self.num_classes int : number of classes of dataset.
    self.layers nn.ModuleDict : ModuleDict of models.
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetSAP, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.sap = SAP(is_valid)
        # self.sap1 = nn.Dropout(0.5)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, is_valid=True)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, is_valid=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, is_valid))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.sap(out, batch_idx)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)

def ResNet18SAP():
    return ResNetSAP(BasicBlockSAP, [2, 2, 2, 2])    
