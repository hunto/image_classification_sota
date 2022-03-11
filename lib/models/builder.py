import yaml
import torchvision

from .nas_model import gen_nas_model
from .darts_model import gen_darts_model
from .mobilenet_v1 import MobileNetV1
from . import resnet


def build_model(args):
    if args.model.lower() == 'nas_model':
        # model with architectures specific in yaml file
        model = gen_nas_model(yaml.safe_load(open(args.model_config, 'r')), drop_rate=args.drop, 
                              drop_path_rate=args.drop_path_rate, auxiliary_head=args.auxiliary)

    elif args.model.lower() == 'darts_model':
        # DARTS evaluation models
        model = gen_darts_model(yaml.safe_load(open(args.model_config, 'r')), args.dataset, drop_rate=args.drop, 
                                drop_path_rate=args.drop_path_rate, auxiliary_head=args.auxiliary)

    elif args.model.lower() == 'nas_pruning_model':
        # model with architectures specific in yaml file
        # the model is searched by pruning algorithms
        from edgenn.models import EdgeNNModel
        model_config = yaml.safe_load(open(args.model_config, 'r'))
        channel_settings = model_config.pop('channel_settings')
        model = gen_nas_model(model_config, drop_rate=args.drop, drop_path_rate=args.drop_path_rate, auxiliary_head=args.auxiliary)
        edgenn_model = EdgeNNModel(model, loss_fn=None, pruning=True, input_shape=args.input_shape)
        logger.info(edgenn_model.graph)
        edgenn_model.fold_dynamic_nn(channel_settings['choices'], channel_settings['bins'], channel_settings['min_bins'])
        logger.info(model)

    elif args.model.lower().startswith('resnet'):
        # resnet variants (the same as torchvision)
        model = getattr(resnet, args.model.lower())(num_classes=args.num_classes)

    elif args.model.lower() == 'mobilenet_v1':
        # mobilenet v1
        model = MobileNetV1(num_classes=args.num_classes)

    elif args.model.startswith('tv_'):
        # build model using torchvision
        import torchvision
        model = getattr(torchvision.models, args.model[3:])(pretrained=False)

    elif args.model.startswith('timm_'):
        # build model using timm
        import timm
        model = timm.create_model(args.model[5:], pretrained=False)

    else:
        raise RuntimeError(f'Model {args.model} not found.')

    return model


def build_edgenn_model(args, edgenn_cfgs=None):
    import edgenn
    if args.model.lower() in ['nas_model', 'nas_pruning_model']:
        # gen model with yaml config first
        model = gen_nas_model(yaml.load(open(args.model_config, 'r'), Loader=yaml.FullLoader), drop_rate=args.drop, drop_path_rate=args.drop_path_rate)
        # wrap the model with EdgeNNModel
        model = edgenn.models.EdgeNNModel(model, loss_fn, pruning=(args.model=='nas_pruning_model'))

    elif args.model == 'edgenn':
        # build model from edgenn
        model = edgenn.build_model(edgenn_cfgs.model)

    else:
        raise RuntimeError(f'Model {args.model} not found.')

    return model
