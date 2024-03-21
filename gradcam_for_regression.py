import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns
import os


class gradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activation = None
        self.hook_handle = None
        self.hook_layers()

    def __del__(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()

    def hook_layers(self):
        def backward_hook_function(module, grad_input, grad_output):
            self.gradients = grad_input[0]

        def forward_hook_function(module, input, output):
            self.activation = output

        self.target_layer.register_forward_hook(forward_hook_function)
        # self.target_layer.register_backward_hook(backward_hook_function)
        self.hook_handle = self.target_layer.register_full_backward_hook(backward_hook_function)

    def get_target_layer_output(self, input_image):
        self.model.eval()
        output = self.model(input_image)
        self.model.zero_grad()

        if str(next(self.model.parameters()).device) == "cpu":
            gradients = self.gradients.data.numpy()[0]
            activation = self.activation.data.numpy()[0]
        else:
            gradients = self.gradients.data.cpu().numpy()[0]
            activation = self.activation.data.cpu().numpy()[0]

        return activation, gradients

    def generate_cam(self, input_image, target_class=None, adjusted_resolution=False):
        # self.model.eval()
        output = self.model(input_image)
        self.model.zero_grad()
        if target_class is not None:
            one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_().to(input_image.device)
            one_hot_output[0][target_class] = 1
            output.backward(gradient=one_hot_output, retain_graph=True)
        else:
            output.backward(torch.ones_like(output), retain_graph=True)

        if str(next(self.model.parameters()).device) == "cpu":
            gradients = self.gradients.data.numpy()[0]
            activation = self.activation.data.numpy()[0]
        else:
            gradients = self.gradients.data.cpu().numpy()[0]
            activation = self.activation.data.cpu().numpy()[0]

        # 3채널 이미지 입니다.
        if len(input_image.shape) == 4:
            weights = np.mean(gradients, axis=0)
            cam = np.zeros(dtype=np.float32, shape=activation.shape[1:])

            for k, w in enumerate(weights):
                cam += w * activation[k, :, :]
            # for idx, v in enumerate(activation):
            #     cam += np.dot(np.mean(gradients, axis=0), activation[idx, :, :])
            if target_class is not None:
                cam = np.maximum(cam, 0)
            if adjusted_resolution:
                cam = cv2.resize(cam, (input_image.size(2), input_image.size(3)))
        # 2채널 이미지 입니다. (여기서는 1차원데이터의 12리드)
        elif len(input_image.shape) == 3:
            # 테스트 - 코드
            weights = np.mean(gradients, axis=(0))
            cam = np.zeros(activation.shape[1:], dtype=np.float32)
            for k, w in enumerate(weights):
                cam += w * activation[k, :]

            # 활성화 맵을 생성합니다.
            cam = np.mean(activation, axis=0).squeeze()
            cam = np.maximum(cam, 0)
            cam = cam / np.max(cam)

        return cam, activation, gradients

    @staticmethod
    def show_cam_on_image(input_data, cam, selected_idx=0, ECG_min=None, ECG_ptp=None, signature="", save=None, visualize=True):
        cam = np.array(cam)[selected_idx]
        plot_figure_size = (8, 5)

        fig, ax = plt.subplots(nrows=12, figsize=plot_figure_size)
        canvas = FigureCanvas(fig)
        t = np.arange(0, 4096)
        img_MAX_height = 0
        init_ECG_leads = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

        assert selected_idx < len(input_data), f"selected_idx must be less than the array length"

        for lead_idx, lead_name in enumerate(init_ECG_leads):
            feed_ECG = input_data[selected_idx, :, lead_idx]
            if ECG_ptp is not None:
                feed_ECG = feed_ECG * ECG_ptp[lead_idx] + ECG_min[lead_idx]

            ax[lead_idx].plot(t, np.squeeze(feed_ECG), color="black", linewidth=1)
            ptp = np.ptp(np.squeeze(feed_ECG))
            if img_MAX_height < ptp:
                img_MAX_height = ptp

            ax[lead_idx].text(0.05, 0.95, lead_name + "-lead", transform=ax[lead_idx].transAxes,
                              verticalalignment='top',
                              fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax[lead_idx].set_xlim([t[0], t[-1]])

        plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        print(img_MAX_height)
        canvas.draw()

        graph_img = np.array(fig.canvas.get_renderer()._renderer)
        plt.close()

        fig, ax = plt.subplots(figsize=plot_figure_size)
        canvas = FigureCanvas(fig)
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_BGRA2BGR)
        print(graph_img.shape)

        print(f"grad_cams.shape: {cam.shape}")
        grad_cams = np.expand_dims(np.average(cam, axis=0), axis=0)
        top10percent = np.percentile(grad_cams, 90, interpolation="linear")
        print(f"min:{np.min(grad_cams)}, max:{np.max(grad_cams)}, percentile_95:{top10percent}")
        print(f"grad_cams.shape top10percent: {top10percent.shape}")

        grad_cams = np.where(grad_cams < top10percent, 0, grad_cams)
        print(f"grad_cams.shape expand_dims: {grad_cams.shape}")
        grad_cams = cv2.resize(grad_cams, (graph_img.shape[1], graph_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        print(f"grad_cams.shape resize: {grad_cams.shape}")


        sns.heatmap(grad_cams, cmap='bwr', cbar=False, yticklabels=False, xticklabels=False,
                    vmin=np.min(grad_cams), vmax=np.max(grad_cams))
        plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        canvas.draw()
        saliency_img = np.array(fig.canvas.get_renderer()._renderer)
        saliency_img = cv2.cvtColor(saliency_img, cv2.COLOR_RGB2BGR)
        print(f"grad_cams.shape : {grad_cams.shape}")
        blended_img = cv2.addWeighted(graph_img, 0.7, saliency_img, 0.3, 0)

        ### 저장
        if save is not None:
            ext = os.path.splitext(save)[1]
            result, n = cv2.imencode(ext, blended_img)
            if result:
                with open(save, mode='w+b') as f:
                    n.tofile(f)

        while (visualize):
            cv2.imshow(f"Heap map-{signature}", blended_img)
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()
                break
            else:
                continue

        plt.close()

    @staticmethod
    def draw_grouped_cam(input_data, cam, group_id=0, ECG_min=None, ECG_ptp=None, signature="", save=None, visualize=True):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        import seaborn as sns
        import os

        # input_data = input_data.transpose(0, 2, 1)
        # cam = cam.transpose(1, 0)
        plot_figure_size = (8, 5)

        fig, ax = plt.subplots(nrows=12, figsize=plot_figure_size)
        canvas = FigureCanvas(fig)
        t = np.arange(0, 4096)

        init_ECG_leads = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

        for lead_idx, lead_name in enumerate(init_ECG_leads):
            # feed_ECG = np.median(input_data[:, :, lead_idx], axis=0)
            img_MAX_height = 0
            img_MIN_height = 0
            for lead_data in input_data[:, :, lead_idx]:
                feed_ECG = lead_data
                if ECG_ptp is not None:
                    feed_ECG = feed_ECG * ECG_ptp[lead_idx] + ECG_min[lead_idx]

                ax[lead_idx].plot(t, np.squeeze(feed_ECG), color="black", linewidth=1)
                # ptp = np.ptp(np.squeeze(feed_ECG))
                # if img_MAX_height < ptp:
                #     img_MAX_height = ptp
                if img_MAX_height < np.max(feed_ECG):
                    img_MAX_height = np.max(feed_ECG)
                if img_MIN_height > np.min(feed_ECG):
                    img_MIN_height = np.min(feed_ECG)

                ax[lead_idx].text(0.05, 0.95, lead_name + "-lead", transform=ax[lead_idx].transAxes,
                                  verticalalignment='top',
                                  fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax[lead_idx].set_xlim([t[0], t[-1]])
                ax[lead_idx].set_ylim([img_MIN_height, img_MAX_height])

        plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        print(img_MAX_height)
        # plt.show()
        canvas.draw()

        graph_img = np.array(fig.canvas.get_renderer()._renderer)
        plt.close()

        fig, ax = plt.subplots(figsize=plot_figure_size)
        canvas = FigureCanvas(fig)
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_BGRA2BGR)
        print(graph_img.shape)

        print(f"grad_cams.shape: {cam.shape}")
        grad_cams = np.expand_dims(np.average(cam, axis=0), axis=0)
        print(f"grad_cams.shape expand_dims: {grad_cams.shape}")
        grad_cams = cv2.resize(grad_cams, (graph_img.shape[1], graph_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        print(f"grad_cams.shape resize: {grad_cams.shape}")
        top10percent = np.percentile(grad_cams, 95, interpolation="linear")
        print(f"min:{np.min(grad_cams)}, max:{np.max(grad_cams)}, percentile_95:{top10percent}")
        print(f"grad_cams.shape top10percent: {top10percent.shape}")

        sns.heatmap(grad_cams, cmap='bwr', cbar=False, yticklabels=False, xticklabels=False,
                    vmin=0, vmax=1)
                    # vmin=np.min(grad_cams), vmax=np.max(grad_cams))
        # sns_heatmap = sns_heatmap.get_figure()
        plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        # plt.show()
        canvas.draw()
        saliency_img = np.array(fig.canvas.get_renderer()._renderer)
        saliency_img = cv2.cvtColor(saliency_img, cv2.COLOR_RGB2BGR)
        print(f"grad_cams.shape : {grad_cams.shape}")
        blended_img = cv2.addWeighted(graph_img, 0.7, saliency_img, 0.3, 0)

        ### 저장
        if save is not None:
            ext = os.path.splitext(save)[1]
            result, n = cv2.imencode(ext, blended_img)
            if result:
                with open(save, mode='w+b') as f:
                    n.tofile(f)

        while (visualize):
            cv2.imshow(f"Heap map-{signature}", blended_img)

            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()
                break
            else:
                continue

        plt.close()

if __name__ == "__main__":
    def preprocess_image(img_path):
        raw_img = cv2.imread(img_path)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(raw_img, (224, 224))
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = Variable(img.unsqueeze(0))
        return img, raw_img

    img_path = "C:/Users/oskwon/Downloads/CIFAR-10-images-master/test/dog/0054.jpg"
    input_image, raw_img = preprocess_image(img_path)

    model = models.resnet50(pretrained=True)
    target_layer = model.layer4[2].conv3

    grad_cam = gradCAM(model, target_layer)
    cam = grad_cam.generate_cam(input_image, target_class=0, adjusted_resolution=True, is_individual=True)  # Change target class as needed

    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    img = cv2.resize(raw_img, (224, 224))
    cam_image = show_cam_on_image(img / 255.0, cam)
    plt.imshow(cam_image)
    plt.show()
