import os
import torch
from torchvision import transforms as tf
from PIL import Image
from DRDR.unet import UNetModel
from DRDR.space_diffusion import SpacedDiffusion
from DRDR.resizer import Resizer
from data import saveImage
try:
    from torchvision.transforms import InterpolationMode
    IB = InterpolationMode.BICUBIC
except ImportError:
    IB = Image.BICUBIC



def img_to_tensor(img_path, size=512):
    IB = InterpolationMode.BICUBIC
    img = Image.open(img_path).convert("RGB")

    tf_method = tf.Compose([
        tf.Resize((size, size), IB),
        tf.ToTensor(),
        tf.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    return tf_method(img).unsqueeze(0)

def lqp_to_tensor(x, size=512):
    IB = InterpolationMode.BICUBIC
    tf_method = tf.Resize((size, size), IB)

    return tf_method(x)


class DR2E():
    def __init__(self, sr="SPAR", cuda=0):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
        self.device = torch.device("cuda:0")

        # The first stage: DR2
        model = UNetModel(in_channels=3, out_channels=6)
        self.dr2 = SpacedDiffusion(denoise_fn=model, section=[100])
        self.dr2.model.eval().to(self.device)
        self.dr2.device = self.device
        self.dr2.model.load_state_dict(torch.load("./weights/DR2_FFHQ.pkl"))

        # The second Stage: Enhancement
        self.sr_name = sr
        if sr == "SPAR":
            from SPAR.models import createModel
            
            self.sr = createModel("SPARNetHD2D").eval().to(self.device)
            self.sr.load_state_dict(torch.load("./weights/SPARNetHD_FFHQ_DR2aug.pth"))
        else:
            raise NotImplementedError("Unkonwn enhancement module name: %s" % (sr))
        print("Model loaded successfully.")

    def inference(self, img_pth, save_dir=None, N=8, tau=0):
        filename = img_pth.split('/')[-1].split('.')[0]
        suffix = "(N%02dT%02d)" % (N, tau)

        if not save_dir:
            save_dir = "./test_images/output/"

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if not os.path.exists(os.path.join(save_dir, "dr2")):
            os.mkdir(os.path.join(save_dir, "dr2"))
        if not os.path.exists(os.path.join(save_dir, "enhancement")):
            os.mkdir(os.path.join(save_dir, "enhancement"))

        if N > 1:
            down_N = N
            shape = (1, 3, 256, 256)
            shape_d = (1, 3, 256 // down_N, 256 // down_N)
            down = Resizer(shape, 1 / down_N).to(self.device)
            up = Resizer(shape_d, down_N).to(self.device)
            resizers = (down, up)
        else:
            resizers = None
        
        # Fixed omega
        # omega = min(tau + 25, 100)

        # Variant omega
        if N == 1:
            omega = min(tau + 5, 100)
        elif 1 < N <= 2:
            omega = min(tau + 10, 100)
        elif 2 < N <= 4:
            omega = min(tau + 20, 100)
        elif 4 < N <= 8:
            omega = min(tau + 25, 100)
        else:
            omega = min(tau + 30, 100)


        with torch.no_grad():
            print("[%s]..." % (img_pth), end="")

            # ==================== DR2 Pre-processing =====================
            x = img_to_tensor(img_pth, 256).to(self.device)
            lqp = self.dr2.dr2_sample_loop(
                ref_img=x,
                resizers=resizers,
                start_step=omega,
                output_step=tau,
            )
            saveImage(
                lqp.cpu(),
                os.path.join(save_dir, "dr2"),
                "%s_%s" % (filename, suffix),
            )

            print("Degradation Removed;", end=" ")

            # ==================== Enhance Module =====================
            lqp = lqp_to_tensor(torch.clamp(lqp, -1, 1))
            sr = self.sr(lqp)
            saveImage(
                sr.cpu(),
                os.path.join(save_dir, "enhancement"),
                "%s_%s_%s" % (filename, suffix, self.sr_name),
            )

            print("Enhancement Finished.")
