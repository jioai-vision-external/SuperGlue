#! /usr/bin/env python3

from pathlib import Path
import cv2
import matplotlib.cm as cm
import torch

from models.matching import Matching
from models.utils import (make_matching_plot_fast, read_image)
torch.set_grad_enabled(False)


class SuperGlueInfer():
    """ Class to find keypoints between two images using superglue.
    Inputs
        keypoint_threshold: Confidence threshold for finding the keypoints
                            by superglue on single image.
        model_type: Model Type can be ['indoor', 'outdoor']
        nms_radius: SuperPoint Non Maximum Suppression (NMS) radius
        sinkhorn_iterations: Number of Sinkhorn iterations performed by SuperGlue
        max_keypoints: Maximum number of keypoints detected by Superpoint (-1 keeps all keypoints)
    """
    def __init__(self, model_type='indoor', keypoint_threshold=0.005, nms_radius=4,
                sinkhorn_iterations=20, max_keypoints=-1):
        self.keypoint_threshold = keypoint_threshold
        self.model_type = model_type
        self.nms_radius = nms_radius
        self.sinkhorn_iterations = sinkhorn_iterations
        self.max_keypoints = max_keypoints

    def register_image(self, kps1, kps2, image_1, image_2):
        """ Function to register image using keypoints.
        Inputs
            kps1: Matching keypoints in image1.
            kp2: Matching keypoints in image2.
            image_1: Input image1.
            image_2: Input image2.
        Returns
            aligned_img: Aligned image on the original image
            registerd_img: For verification, addition of registered image and original image.
        """
        # use the homography matrix to align the images
        (hm_matrix, mask) = cv2.findHomography(kps1, kps2, method=cv2.RANSAC)
        (img1_h, img1_w) = image_1.shape[:2]
        aligned_img = cv2.warpPerspective(image_1, hm_matrix, (img1_w, img1_h))
        registerd_img = cv2.addWeighted(image_2, 0.5, aligned_img, 0.5, 0.0)
        return aligned_img, registerd_img

    def predict_kps(self, image1_path, image2_path, output_dir=None, bool_register_image=True, resize_param=[640,480],
            device='cpu',match_threshold=0.2,):
        """ Function to predict matching keypoints between two images.
        Inputs
            image1_path: Path to first input image.
            image2_path: Path to second input image.
            output_dir:  Output directory path to store the keypoint matching visualizations.
            resize_param: Input Resolution, both images will be resized to this size.
            device: 'cuda' or 'cpu'
            match_threshold: After keypoints are generated, match threshold is used by superglue as confidence threshold
            
        Returns
            out_kps: dictionary which returns kps0, kps1 -> keypoints on two images
                     mkpts0, mkpts1 -> matching keypoints
                     mkconf -> confidence values of each matching keypoints
                     num_kps -> number of keypoints
        """
        config = {
            'superpoint': {
                'nms_radius': self.nms_radius,
                'keypoint_threshold': self.keypoint_threshold,
                'max_keypoints': self.max_keypoints
            },
            'superglue': {
                'weights': self.model_type,
                'sinkhorn_iterations': self.sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }
        matching = Matching(config).eval().to(device)
        keys = ['keypoints', 'scores', 'descriptors']

        image1, ref_tensor, scale, image1_bgr = read_image(path=image1_path, device=device, resize=resize_param, rotation=0, resize_float=False)
        image2, in_tensor, scale, image2_bgr = read_image(path=image2_path, device=device, resize=resize_param, rotation=0, resize_float=False)

        last_data = matching.superpoint({'image': ref_tensor})
        last_data = {k+'0': last_data[k] for k in keys}
        last_data['image0'] = ref_tensor
        last_frame = image1
        last_image_id = 0
        stem0, stem1 = last_image_id, 1

        pred = matching({**last_data, 'image1': in_tensor})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mkconf = confidence[valid]
        out_kps = {"kpts0": kpts0, "kpts1": kpts1, "mkpts0": mkpts0,
                    "mkpts1": mkpts1, "mkconf": mkconf, 'num_kps': len(mkpts1)}
        
        if bool_register_image:
            print("Registering Image")
            aligned, reg = self.register_image(out_kps['mkpts0'], out_kps['mkpts1'], image1_bgr, image2_bgr)
        else:
            aligned, reg = None, None
        if output_dir is not None:
            print('==> Will write outputs to {}'.format(output_dir))
            Path(output_dir).mkdir(exist_ok=True)

            color = cm.jet(confidence[valid])
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0))
            ]
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {:06}:{:06}'.format(stem0, stem1),
            ]
            out = make_matching_plot_fast(
                last_frame, image2, kpts0, kpts1, mkpts0, mkpts1, color, text,
                path=None, show_keypoints=True, small_text=small_text)
            stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            out_file = str(Path(output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, out)

            if aligned is not None:
                out_file_aligned = str(Path(output_dir, 'aligned.png'))
                out_file_registered = str(Path(output_dir, 'registed.png'))
                print('\nWriting Aligned to {}'.format(out_file_aligned))
                print('\nWriting Registered to {}'.format(out_file_registered))
                cv2.imwrite(out_file_aligned, aligned)
                cv2.imwrite(out_file_registered, reg)
        return out_kps