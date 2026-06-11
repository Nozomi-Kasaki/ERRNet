from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)        
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')        
        self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        self.parser.add_argument('--train_eval_datasets', type=str, default='ceilnet_table2,real20,objects,postcard,wild', help='comma-separated benchmark datasets for periodic training-time evaluation')
        self.parser.add_argument('--train_eval_interval_epochs', type=float, default=0.5, help='periodic full evaluation interval measured in training epochs')
        self.parser.add_argument('--train_eval_epoch_schedule', type=str, default=None, help='optional epoch schedule such as 5:50,1:70; evaluate every 5 epochs until 50, then every epoch until 70')
        self.parser.add_argument('--no_train_eval', action='store_true', help='disable periodic full evaluation during training')
        self.parser.add_argument('--no_save_best_eval', action='store_true', help='do not save the best checkpoint selected by periodic evaluation')
        self.parser.add_argument('--reset_epoch_on_load', action='store_true', help='treat --icnn_path as initialization and restart epoch/iteration counters from zero')
        self.parser.add_argument('--openrr_train_dir', type=str, default=None, help='optional OpenRR-style paired training directory with blended/transmission_layer subfolders')
        self.parser.add_argument('--openrr_train_ratio', type=float, default=0.25, help='fraction of training samples drawn from the optional OpenRR dataset')
        self.parser.add_argument('--train_fusion_ratios', type=str, default='0.7,0.3', help='comma-separated base fusion ratios for the default synthetic and real training sets')

        # for training (Note: in train_errnet.py, we mannually tune the training protocol, but you can also use following setting by modifying the code in errnet_model.py)
        self.parser.add_argument('--nEpochs', '-n', type=int, default=60, help='# of epochs to run')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        self.parser.add_argument('--wd', type=float, default=0, help='weight decay for adam')

        self.parser.add_argument('--low_sigma', type=float, default=2, help='min sigma in synthetic dataset')
        self.parser.add_argument('--high_sigma', type=float, default=5, help='max sigma in synthetic dataset')
        self.parser.add_argument('--low_gamma', type=float, default=1.3, help='max gamma in synthetic dataset')
        self.parser.add_argument('--high_gamma', type=float, default=1.3, help='max gamma in synthetic dataset')
        self.parser.add_argument('--syn_kernel_sizes', type=str, default='7,9,11,15,21', help='comma-separated Gaussian kernel sizes for synthetic reflection blur')
        self.parser.add_argument('--no_reflection_aug', action='store_true', help='disable enhanced reflection synthesis augmentation')
        self.parser.add_argument('--reflection_alpha_low', type=float, default=0.35, help='min synthetic reflection intensity multiplier')
        self.parser.add_argument('--reflection_alpha_high', type=float, default=1.0, help='max synthetic reflection intensity multiplier')
        self.parser.add_argument('--transmission_alpha_low', type=float, default=0.92, help='min transmission attenuation in synthetic mixtures')
        self.parser.add_argument('--transmission_alpha_high', type=float, default=1.0, help='max transmission attenuation in synthetic mixtures')
        self.parser.add_argument('--reflection_color_jitter', type=float, default=0.08, help='per-channel reflection color jitter range')
        self.parser.add_argument('--reflection_shift', type=int, default=6, help='max pixel shift for reflection layer before synthesis')
        self.parser.add_argument('--reflection_noise_std', type=float, default=0.01, help='Gaussian noise std added to synthetic mixtures')
        self.parser.add_argument('--reflection_jpeg_prob', type=float, default=0.20, help='probability of JPEG compression on synthetic mixtures')
        self.parser.add_argument('--reflection_jpeg_quality_low', type=int, default=55, help='min JPEG quality for synthetic mixtures')
        self.parser.add_argument('--reflection_jpeg_quality_high', type=int, default=95, help='max JPEG quality for synthetic mixtures')
        
        # data augmentation
        self.parser.add_argument('--batchSize', '-b', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=str, default='224,336,448', help='scale images to multiple size')
        self.parser.add_argument('--fineSize', type=str, default='224,224', help='then crop to this size')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')

        # for discriminator
        self.parser.add_argument('--which_model_D', type=str, default='disc_vgg', choices=['disc_vgg', 'disc_patch'])
        self.parser.add_argument('--gan_type', type=str, default='rasgan', help='gan/sgan : Vanilla GAN; rasgan : relativistic gan')
        
        # loss weight
        self.parser.add_argument('--unaligned_loss', type=str, default='vgg', help='learning rate policy: vgg|mse|ctx|ctx_vgg')
        self.parser.add_argument('--vgg_layer', type=int, default=31, help='vgg layer of unaligned loss')
        self.parser.add_argument('--loss_profile', type=str, default='structure', choices=['legacy', 'structure'], help='aligned-data reconstruction loss profile')
        self.parser.add_argument('--lambda_mse', type=float, default=0.2, help='weight for MSE reconstruction loss')
        self.parser.add_argument('--lambda_charbonnier', type=float, default=1.0, help='weight for robust Charbonnier reconstruction loss')
        self.parser.add_argument('--lambda_gradient', type=float, default=0.4, help='weight for first-order gradient loss')
        self.parser.add_argument('--lambda_laplacian', type=float, default=0.1, help='weight for Laplacian edge loss')
        self.parser.add_argument('--lambda_ssim', type=float, default=0.2, help='weight for differentiable SSIM loss')
        self.parser.add_argument('--lambda_adapter_consistency', type=float, default=0.0, help='weight for keeping adapter output close to the ERRNet backbone output')
        self.parser.add_argument('--lambda_adapter_sparsity', type=float, default=0.0, help='weight for sparse gated correction magnitude in adapter models')
        self.parser.add_argument('--adapter_freeze_backbone_epochs', type=int, default=0, help='number of initial epochs that train only the adapter head')
        self.parser.add_argument('--adapter_backbone_lr_scale', type=float, default=1.0, help='learning-rate multiplier for adapter backbone parameter group')
        self.parser.add_argument('--adapter_lr_scale', type=float, default=1.0, help='learning-rate multiplier for adapter head parameter group')
        
        self.parser.add_argument('--lambda_gan', type=float, default=0.01, help='weight for gan loss')
        self.parser.add_argument('--lambda_vgg', type=float, default=0.1, help='weight for vgg loss')
        
        self.isTrain = True
