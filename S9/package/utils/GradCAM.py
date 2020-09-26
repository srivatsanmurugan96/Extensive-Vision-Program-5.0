import numpy as np
import sys
import cv2
import torch
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt


class GradCAM(object):
    def __init__(self, model_arch, targetLayer, classes):
        self.model = model_arch
        self.classes = classes

        # Hook to particular layer to fetch Activation and Gradient values
        self.activation = {}
        self.gradient = {}

        def backward_hook(module, grad_input, grad_output):
            self.gradient['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activation['value'] = output
            return None

        targetLayer.register_forward_hook(forward_hook)  # To fetch values of activation to the layer(feed forward)
        targetLayer.register_backward_hook(backward_hook)  # To fetch values of gradients to the layer(back propagation)
        # Gradients are the derivative of layer weight pixels w.r.t
        # Prediction value of respcetive class element.

    # To display all parameters to get Layer related Info
    def displayParams(self):
        for name, param in self.model.named_parameters():
            print(name)

    # Visualise GradCAM
    def visualiseGradcam(self, heatmap, input, normFlag=True):
        if (normFlag):
            ### Retriving the preprocessed actual image
            image = input.numpy()[0, :]
            image[0, :] = (image[0, :] * 0.2023) + 0.4914  # de-normalise = (normalised * std) + mean
            image[1, :] = (image[1, :] * 0.1994) + 0.4822
            image[2, :] = (image[2, :] * 0.2010) + 0.4465
            image = np.transpose(image, (1, 2, 0))  # convertng from [ channel, row, column] to [ row, column, channel]
            image = np.minimum(image, 255)

        cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        imageRGB = 255 * image / np.max(image)
        cam_image = (np.float32(cam)) + np.float32(imageRGB)
        cam_image = 255 * cam_image / np.max(cam_image)

        ### Actual
        actImage = cv2.resize(imageRGB, (150, 150))  # Resize image to 200,200
        # cv2_imshow(actImage.squeeze())

        cam = (np.float32(cam)) + np.float32(image)
        cam = 255 * cam / np.max(cam)
        ### GradCAM
        gradCam = cv2.resize(cam, (150, 150))
        # cv2_imshow(gradCam.squeeze())

        ### GradCAM + Image
        CamImg = cv2.resize(cam_image, (150, 150))
        # cv2_imshow(CamImg.squeeze())

        ### GradCAM overlat with image
        img_overlay = cv2.addWeighted(actImage, 0.7, gradCam, 0.3, 0)
        # cv2_imshow(img_overlay.squeeze())

        final_frame = cv2.hconcat((actImage, gradCam, CamImg))
        cv2_imshow(final_frame.squeeze())
        print("     ACTUAL          GRADCAM           Both Gradcam+Image")

    # Main GradCAM implementation
    def forward(self, input, class_idx=None, retain_graph=False):

        self.model.eval()
        logit = self.model(input)

        if class_idx is None:
            predVal = logit[:, logit.max(1)[-1]].squeeze()
        else:
            # print("Actual value : ", self.classes[class_idx])
            predVal = logit[:, class_idx].squeeze()

        pred = logit.argmax(dim=1, keepdim=True)
        # print("predicted value : ", self.classes[pred.item()])

        self.model.zero_grad()  # Making all gradiants stored in layers to zero before the operation
        predVal.backward(retain_graph=retain_graph)  # sending the prediction value of class backward to layer

        gradients = self.gradient[
            'value']  # Fetching values of activation to the layer(In feed forward)
        activations = self.activation['value']  # Fetching values of gradients to the layer(In back propagation)

        # Converting gradiants and activation value from tensor to numpy
        # All logics are implemented in numpy
        activations = activations.detach().numpy()
        gradients = gradients.detach().numpy()

        ### convertng from [ channel, row, column] to [ row, column, channel]
        output, grads_val = np.transpose(activations[0, :], (1, 2, 0)), np.transpose(gradients[0, :, :, :], (1, 2, 0))
        print("activation Size : ", np.shape(output))

        ### GAP to the gradiants
        weights = np.mean(grads_val, axis=(0, 1))

        cam = np.ones(output.shape[0: 2], dtype=np.float32)  # Intialising array to hold weighted activation output
        ### Multiplting GAP gradiants with activation inputs to layer
        ### And coverting it from W x W x n to W x W x 1 (eg: 8x8x256 => 8x8x1)
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]
        ### Applying Relu
        cam = np.maximum(cam, 0)

        ### Resizing frame to image size (eg: 8x8x1 => 32x32x1)
        cam = cv2.resize(cam, (32, 32))

        ### Normalising image to make heatmap
        cam = cam - np.min(cam)
        heatmap = cam / np.max(cam)

        return cam, heatmap, pred.item()