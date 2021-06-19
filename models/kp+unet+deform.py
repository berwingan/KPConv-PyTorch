#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network architectures
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#

from models.blocks import *
import numpy as np


def p2p_fitting_regularizer(net):

    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, KPConv) and m.deformable:

            ##############
            # Fitting loss
            ##############

            # Get the distance to closest input point and normalize to be independant from layers
            KP_min_d2 = m.min_d2 / (m.KP_extent ** 2)

            # Loss will be the square distance to closest input point. We use L1 because dist is already squared
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations
            KP_locs = m.deformed_KP / m.KP_extent

            # Point should not be close to each other
            for i in range(net.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)
                repulsive_loss += net.l1(rep_loss, torch.zeros_like(rep_loss)) / net.K

    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)


class KPCNN(nn.Module):
    """
    Class defining KPCNN
    """

    def __init__(self, config):
        super(KPCNN, self).__init__()

        #####################
        # Network opperations
        #####################

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points

        # Save all block operations in a list of modules
        self.block_ops = nn.ModuleList()

        # Loop over consecutive blocks
        block_in_layer = 0
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.block_ops.append(block_decider(block,
                                                r,
                                                in_dim,
                                                out_dim,
                                                layer,
                                                config))


            # Index of block in this layer
            block_in_layer += 1

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim


            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
                block_in_layer = 0

        self.head_mlp = UnaryBlock(out_dim, 1024, False, 0)
        self.head_softmax = UnaryBlock(1024, config.num_classes, False, 0)

        ################
        # Network Losses
        ################

        self.criterion = torch.nn.CrossEntropyLoss()
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):

        # Save all block operations in a list of modules
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        for block_op in self.block_ops:
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, labels)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    @staticmethod
    def accuracy(outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        predicted = torch.argmax(outputs.data, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        return correct / total


class KPFCNN(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config, lbl_values, ign_lbls):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)

        
        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []
        ###################################
        self.path_blocks = nn.ModuleList() #BERWIN CHANGE HERE 
        self.path_deform_blocks =nn.ModuleList() #right after resnet
        self.upsample_path_blocks = nn.ModuleList()
        #check how wide the skip paths should be BERWIN CHANGE HERE #################
        path_size =-1
        for block_i, block in enumerate(config.architecture):
            if('upsample' in block):
                path_size +=1 #this should be 3 as of right now 
        layer_count = path_size
        upsample_count = path_size
        upsample_var = 0 #only starts after the first strided layer 
        # Loop over consecutive blocks
        
        for block_i, block in enumerate(config.architecture):
            #create upsample for in path
            if upsample_var>0 and 'strided' in block:
                self.upsample_layer_blocks = nn.ModuleList()
                for num_block in range(upsample_count):
                    self.upsample_layer_blocks.append(block_decider('nearest_upsample',
                                                                    r,
                                                                    in_dim,
                                                                    out_dim,
                                                                    layer,
                                                                    config))
                upsample_count-=1
                self.upsample_path_blocks.append(self.upsample_layer_blocks)

            #create skip path blocks here 
            if layer_count>0 and 'strided' in block:
                upsample_var+=1
                self.layer_path = nn.ModuleList()
                self.layer_deform = nn.ModuleList()
                mul_var = 1
                for num_block in range(layer_count):
                    self.layer_path.append(block_decider('resnetb',
                                                        r,
                                                        (in_dim*2)+(out_dim*mul_var),
                                                        out_dim,
                                                        layer,
                                                        config))
                    self.layer_deform.append(block_decider('resnetb_deformable',
                                                        r,
                                                        out_dim,
                                                        out_dim,
                                                        layer,
                                                        config))
                    mul_var+=1 #for the first layer it should create 3 blocks
                layer_count-=1
                self.path_deform_blocks.append(self.layer_deform)#same struc as resnet
                self.path_blocks.append(self.layer_path) #[0][2] -top layer, last path block
            
            ##############CHANGE ENDS HERE FOR PATH BLOCKS
            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        unary_in_dim_var = 0
        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat -unary block after the upsample
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)
                self.decoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim+(out_dim*(unary_in_dim_var)),
                                                    out_dim,
                                                    layer,
                                                    config))
                unary_in_dim_var+=1#to change the unary indim

            # Apply the good block function defining tf ops
            else:#upsample and other blocks goes here 
                self.decoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0)

        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):

        # Get input features
        x = batch.features.clone().detach()
        seg_outputs = []#use this layer for supervising all 4 top layer 

        x = self.encoder_blocks[0](x,batch)
        x0_0 = self.encoder_blocks[1](x,batch)
        x = self.encoder_blocks[2](x0_0,batch)#check all right after 
        x = self.encoder_blocks[3](x,batch)
        x1_0 = self.encoder_blocks[4](x,batch)
        x0_1 = self.path_blocks[0][0](torch.cat([x0_0, self.upsample_path_blocks[0][0](x1_0,batch)],dim=1),batch)
        x0_1 = self.path_deform_blocks[0][0](x0_1,batch)#deform
        #segout append here

        x = self.encoder_blocks[5](x1_0,batch)#check right after encoder
        x = self.encoder_blocks[6](x,batch)
        x2_0 = self.encoder_blocks[7](x,batch)
        x1_1 = self.path_blocks[1][0](torch.cat([x1_0,self.upsample_path_blocks[1][0](x2_0,batch)],dim=1),batch)
        x1_1 = self.path_deform_blocks[1][0](x1_1,batch)#deform
        x0_2 = self.path_blocks[0][1](torch.cat([x0_0,x0_1,self.upsample_path_blocks[0][1](x1_1,batch)],dim=1),batch)
        x0_2 = self.path_deform_blocks[0][1](x0_2,batch)#deform 
        #segout here

        x = self.encoder_blocks[8](x2_0,batch)#right after
        x = self.encoder_blocks[9](x,batch)
        x3_0 = self.encoder_blocks[10](x,batch)
        x2_1 = self.path_blocks[2][0](torch.cat([x2_0, self.upsample_path_blocks[2][0](x3_0,batch)],dim=1),batch)
        x2_1 = self.path_deform_blocks[2][0](x2_1,batch)#deform
        x1_2 = self.path_blocks[1][1](torch.cat([x1_0,x1_1,self.upsample_path_blocks[1][1](x2_1,batch)],dim=1),batch)
        x1_2 = self.path_deform_blocks[1][1](x1_2,batch)#deform
        x0_3 = self.path_blocks[0][2](torch.cat([x0_0,x0_1,x0_2, self.upsample_path_blocks[0][2](x1_2,batch)],dim=1),batch)
        x0_3 = self.path_deform_blocks[0][2](x0_3,batch)#deform
        #segout here

        x = self.encoder_blocks[11](x3_0,batch)
        x = self.encoder_blocks[12](x,batch)
        x4_0 = self.encoder_blocks[13](x,batch)#last block of encoder

        #decoding starts
        x = self.decoder_blocks[0](x4_0,batch)#upsample
        x3_1 = self.decoder_blocks[1](torch.cat([x3_0,x],dim=1),batch)
        x = self.decoder_blocks[2](x3_1,batch)#upsample
        x2_2 = self.decoder_blocks[3](torch.cat([x2_0,x2_1,x],dim=1),batch)
        x = self.decoder_blocks[4](x2_2,batch)#upsample
        x1_3 = self.decoder_blocks[5](torch.cat([x1_0,x1_1,x1_2,x],dim=1),batch)
        x = self.decoder_blocks[6](x1_3,batch)
        x0_4 = self.decoder_blocks[7](torch.cat([x0_0,x0_1,x0_2,x0_3,x],dim=1),batch)
        #segout here

        # Loop over consecutive blocks
        # skip_x = []
        # for block_i, block_op in enumerate(self.encoder_blocks):
        #     if block_i in self.encoder_skips: #skip features is here, 
        #         skip_x.append(x)
        #     x = block_op(x, batch)

        # for block_i, block_op in enumerate(self.decoder_blocks):
        #     if block_i in self.decoder_concats:
        #         x = torch.cat([x, skip_x.pop()], dim=1)
        #     x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x0_4, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model

        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total





















