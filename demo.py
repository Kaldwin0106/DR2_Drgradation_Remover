import os
from dr2e import DR2E


if __name__ == "__main__":
    # Use SPARNet as enhancement module.
    restorer = DR2E(sr="SPAR")

    dir_para = [    
        # [data_dir, (N, \tau)]
        ["./test_images/input/01/", (4, 22)],
        ["./test_images/input/02/", (8, 35)],
        ["./test_images/input/03/", (16, 35)],
    ] 
    # Change controlling parameters (N, \tau) for better trade-off between
    # fidelitty and degrdation removal effect.

    for item in dir_para:
        img_names = os.listdir(item[0])
        N, tau = item[1]

        for name in img_names:
            img_pth = os.path.join(item[0], name)
            restorer.inference(img_pth=img_pth, N=N, tau=tau)

