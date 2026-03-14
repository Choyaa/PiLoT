import logging
import pickle
from typing import Optional, Dict, Tuple, Union
from omegaconf import DictConfig, OmegaConf as oc
import torch

from .feature_extractor import FeatureExtractor
from .refiners import BaseRefiner

from ..pixlib.utils.experiments import load_checkpoint, load_experiment
from ..pixlib.models import get_model
from ..pixlib.geometry import Camera

logger = logging.getLogger(__name__)
# TODO: despite torch.no_grad in BaseModel, requires_grad flips in ref interp
torch.set_grad_enabled(False)


class Localizer:
    def __init__(self, conf: Union[DictConfig, Dict],
                 device: Optional[torch.device] = None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')
        # Loading feature extractor and optimizer from experiment or scratch
        conf = oc.create(conf)
        conf_features = conf.features.get('conf', {})
        conf_optim = conf.get('optimizer', {})
        if conf.get('experiment'):
            pipeline = load_experiment(
                    conf.experiment,
                    {'extractor': conf_features, 'optimizer': conf_optim})
            pipeline = pipeline.to(device)
            logger.debug(
                'Use full pipeline from experiment %s with config:\n%s',
                conf.experiment, oc.to_yaml(pipeline.conf))
            extractor = pipeline.extractor
            optimizer = pipeline.optimizer
            if isinstance(optimizer, torch.nn.ModuleList):
                optimizer = list(optimizer)
        elif conf.get('checkpoint'):
            pipeline = load_checkpoint(
                    conf.checkpoint,
                    conf)
            pipeline = pipeline.to(device)
            extractor = pipeline.extractor
            optimizer = pipeline.optimizer
            if isinstance(optimizer, torch.nn.ModuleList):
                optimizer = list(optimizer)
        else:
            assert 'name' in conf.features
            
            extractor = get_model(conf.features.name)(conf_features)
            optimizer = get_model(conf.optimizer.name)(conf_optim)

        self.conf = conf
        self.device = device
        self.optimizer = optimizer
        self.extractor = FeatureExtractor(
            extractor, device, conf.features.get('preprocessing', {}))

    def run_query(self, name: str, camera: Camera):
        raise NotImplementedError




class RenderLocalizer(Localizer):
    def __init__(self, conf: Union[DictConfig, Dict],
                 device: Optional[torch.device] = None):
        super().__init__(conf, device)
        self.refiner = BaseRefiner(
            self.device, self.optimizer, self.extractor, 
            self.conf.refinement)
    def run_query(self, name: str, camera: Camera, ref_camera: Camera, render_frame, query_T = None, render_T = None, Points_3D_ECEF = None, dd = None, gt_pose_dict = None, last_frame_info = {}, query_resize_ratio = 1,image_query=None):
        ret = self.refiner.refine_query_pose(name, camera, ref_camera, render_frame, query_T, render_T, Points_3D_ECEF, dd = dd, last_frame_info = last_frame_info, query_resize_ratio = query_resize_ratio,image_query= image_query)
        return ret
