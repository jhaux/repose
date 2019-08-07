from tqdm import trange
import numpy as np
import os
from PIL import Image

from repose.keypointer import KeyPointer
from repose.keypoint_metrics import multi_thresh_pck, l2_metric

from repose.util import average_keypoints
from repose.util import make_latex_table, SIstring, LatexString
from repose.util import compile_table

from edflow.data.util import adjust_support
from edflow.custom_logging import get_logger
from edflow.util import set_default, set_value, retrieve


LOGGER = get_logger(__file__)


STANDART_KPS = np.array(
     [[140.26817181,  46.54231771],
      [133.239308,    62.68013491],
      [114.07508489,  62.66841043],
      [82.84061812,   71.74110339],
      [49.57522829,   75.76762688],
      [153.37176487,  62.67037789],
      [180.58877111,  70.74156227],
      [208.82030103,  71.75278198],
      [135.25632113, 127.17084251],
      [123.14381424, 127.18696081],
      [114.07040428, 175.55412537],
      [100.96522807, 216.88775698],
      [148.34837326, 126.17318282],
      [160.44491098, 172.54367428],
      [170.52955213, 213.86912628],
      [135.25650468,  42.51193385],
      [143.30006424,  42.52114882],
      [126.19122843,  42.51358583],
      [146.33472146,  43.5294833],
      [174.54557865, 224.94029919],
      [180.58716501, 221.92157673],
      [168.51536114, 217.89877019],
      [106.02684471, 229.98212647],
      [98.96623761,  228.96643264],
      [97.95422632,  219.9003304]]
)


METRICS = {
        'l2': l2_metric,
        'pck': multi_thresh_pck
        }


BACKENDS = {
        'openpose': KeyPointer
        }


PCK_THRESH = [0.01, 0.025, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5]


class RePoseEval(object):
    '''Iterates over a target and generated dataset and dectes keypoints on
    the generated images, then compares them to the target keypoints. Expects
    the target keypoints to be in absolute pixel values realtive to the frame
    of the target image (e.g. :math:`[0, 255]^2`. If the environment variable
    ``DEBUG_MODE=True`` is set keypoints are oslo detected on the target data.
    For this the target dataset must contain target rgb images at the key
    ``target``.
    '''

    def __init__(self,
                 data_in_im_key='target',
                 data_in_kp_key='keypoints',
                 data_out_im_key='frame_gen',
                 data_out_kp_key='keypoints',
                 metrics=['l2', 'pck'],
                 metrics_kwargs={'l2': {}, 'pck': {'threshold': PCK_THRESH}},
                 backend='openpose',
                 force_recalculation=False,
                 strategy='calc_all',
                 backend_kwargs={},
                 num_pose_render=500):
        '''
        Parameters:
        -----------
        data_in_im_key : str
            Key in labels of the ``data_in`` dataset, at which the
            ground truth image can be found.
        data_in_kp_key : str
            Key in labels of the ``data_in`` dataset, at which the ground truth
            keypoints can be found.
        data_out_im_key : str
            Key in labels of the ``data_out`` dataset, at which the
            generated image can be found.
        data_out_kp_key : str
            Key in labels of the ``data_out`` dataset, at which the keypoints
            estimated from the generated image can be found. If this key is
            found in the labels, no re-estimation of keypoints on the generated
            images is done. If it is not found, the keypoints are estimated.
        metrics : list(str)
            Defines the way the keypoints are compared. Must be one of
                - ``l2``
                - ``pck``
        metrics_kwargs : dict(str, dict)
            Keyword Arguments passed to the metric functions each time they are
            called. If metrics is ``['l2']`` metrics_kwargs must be ``{'l2': {...}}``.
        backend : str
            Defines the keypoint estimator. Must be one of 
                - ``openpose``
                - ``alphapose`` (not yet implemented)
        force_recalculation : bool
            If set to True, will re-estimate the keypoints on the generated
            images.
        strategy : str
            What to do if the keypoint models of the backend and the ground
            truth keypoints do not match, i.e. one is openpose BODY_25 and the
            other is COCO_17.
                - ``calc_all``: Will also estimate the keypoints of the ground
                    truth images if model mismatch is detected. This will add
                    a key to ``data_out.labels`` of the form
                    ``keypoints.model``, which is checked the next time this
                    callback is run on the data. If ``force_recalculation`` is
                    ``False`` at that point, these keypoints are loaded and
                    used, so that no recalculation is needed.
                - ``raise``: Will raise an error if model mismatch is detected.
        backendkwargs : dict
            Keyword arguments passed to the backend at construction time.
        num_pose_render : int
            The number of images for which the pose
            detecionts are rendered on top of the frames. Will be turned
            into a video afterwards using ffmpeg.
        '''

        self.kiim = data_in_im_key
        self.kikp = data_in_kp_key
        self.koim = data_out_im_key
        self.kokp = data_out_kp_key

        self.metrics = {n: METRICS[n] for n in metrics}
        self.metrics_kwargs = metrics_kwargs

        self.strategy = strategy

        self.n_render = num_pose_render

        self.detect_kps = None  # lazy construction in estimate_keypoints
        self.backend = backend
        self.backend_kwargs = backend_kwargs

        self.force = force_recalculation

    def __call__(self, root, data_in, data_out, config=None):
        ''' Expected keys:
            ``data_in.labels``:
                ``'keypoints'``: numpy array containing all keypoints for each
                    frame. Expected to contain only one person per frame.
            ``data_out``:
                ``'frame_gen'``: RGB(A) image of a person generated by the
                    model.

            ``data_in``:
                ``'target'``: Only if environment variable ``DEBUG_MODE=True``
                    is set. RGB(A) image of the target person.
        '''

        self.root = root
        self.repose_path = os.path.join(root, 'repose')
        os.makedirs(self.repose_path, exist_ok=True)
        self.render_path = os.path.join(self.repose_path, 'render')
        os.makedirs(self.render_path, exist_ok=True)

        key = 'keypoints_{}'.format(self.backend)
        print(key)
        print(sorted(data_out.labels.keys()))
        if key in data_out.labels and not self.force:
            all_kps_gen = data_out.labels[key]
        else:
            all_kps_gen = self.estimate_keypoints(root,
                                                  data_out,
                                                  self.koim,
                                                  key,
                                                  'gen')

        kps_target = retrieve(data_in.labels, self.kikp)[:len(data_out)]

        gen_shape = np.shape(all_kps_gen)
        trg_shape = np.shape(kps_target)

        model_mismatch = not np.all(np.equal(gen_shape, trg_shape))
        target_out_key = 'keypoints_target_{}'.format(self.backend)
    
        if model_mismatch:
            if self.strategy == 'raise':
                raise ValueError('Keypoint Models of backend and supplied '
                                 + 'target keypoints do not match:\n'
                                 + 'target: {}\nbackend: {}'.format(gen_shape,
                                                                    trg_shape))
            elif target_out_key in data_out.labels and not self.force:
                kps_target = np.array(data_out.labels[target_out_key])
            else:
                kps_target = self.estimate_keypoints(root,
                                                     data_in,
                                                     self.kiim,
                                                     target_out_key,
                                                     'targ')

        confs_gen = all_kps_gen[..., 2]
        confs_target = kps_target[..., 2]
        kps_gen = all_kps_gen[..., :2]
        kps_target = kps_target[..., :2]

        metric_results = {}
        for name, metric in self.metrics.items():
            args = [kps_target, kps_gen, confs_target, confs_gen]
            kwargs = self.metrics_kwargs[name]
            metric_results[name] = metric(*args, **kwargs)

        self.store_results(metric_results)

        self.render_pose_video('gen_%04d.png', 'gen_plus_kps.mp4')
        self.render_pose_video('targ_%04d.png', 'targ_plus_kps.mp4')

        in1 = os.path.join(self.repose_path, 'targ_plus_kps.mp4')
        in2 = os.path.join(self.repose_path, 'gen_plus_kps.mp4')
        self.render_vids_side_by_side(in1, in2)

    def estimate_keypoints(self, root, dset, im_key, kp_key, n):
        if self.detect_kps is None:
            self.detect_kps = BACKENDS[self.backend](**self.backend_kwargs)

        all_kps_est = []

        for idx in trange(len(dset), desc='KP ' + n):
            image = dset[idx][im_key]

            openpose_image = rgb2openpose(image)

            do_render = idx < self.n_render
            kps_est = self.detect_kps(openpose_image, do_render)
            if do_render:
                kps_est, render_est = kps_est

                render = Image.fromarray(rgb2openpose(render_est))

                savename = '{}_{:0>4}.png'.format(n, idx)
                savepath = os.path.join(self.render_path, savename)

                render.save(savepath)
            kps_est = average_keypoints(kps_est).squeeze()

            all_kps_est += [kps_est]

        all_kps_est = np.array(all_kps_est)

        # Store keypoints, s.t. they will be loaded by EvalDataFolder
        shape = 'x'.join([str(i) for i in list(np.shape(all_kps_est))])
        dtype = all_kps_est.dtype

        kpe_name = '-*-'.join([str(kp_key), str(shape), str(dtype)])

        kpg_path = os.path.join(root, 'model_outputs', kpe_name)
        np.save(kpg_path, all_kps_est)

        return all_kps_est

    def store_results(self, metric_results, debug=''):

        repose_path = self.repose_path
        for name, results in metric_results.items():
            if name == 'pck':
                print(results)
                pck_name = os.path.join(repose_path,
                                        'pck_kps{}.npz'.format(debug))
                pck_tab_name = os.path.join(repose_path,
                                            'pck_kps{}.tex'.format(debug))
                pck_pk_tab_name = os.path.join(repose_path,
                                               'pck_per_kps{}.tex'
                                               .format(debug))

                pck_threshs = sorted(results.keys())
                mean_pcks = [results[k]['mean'] for k in pck_threshs]
                per_kp_pcks = [results[k]['per_kp'] for k in pck_threshs]

                np.savez(pck_name,
                         pcks=mean_pcks,
                         pcks_per_kp=per_kp_pcks,
                         thresholds=pck_threshs)

                make_pck_table(pck_tab_name,
                               [LatexString(self.root)],
                               mean_pcks,
                               pck_threshs)

                make_pck_per_kp_table(pck_pk_tab_name,
                                      [LatexString(self.root)],
                                      per_kp_pcks,
                                      pck_threshs)

                mean_pose_precision_plot(self.repose_path,
                                         'pose_precision',
                                         per_kp_pcks,
                                         pck_threshs)
            else:
                result_path = os.path.join(repose_path, '{}.npz'.format(name))
                np.savez(result_path, **results)

    def render_pose_video(self,
                          pattern='gen_%04d.png',
                          name='gen_plus_kps.mp4'):
        pattern = os.path.join(self.render_path, pattern)
        name = os.path.join(self.repose_path, name)
        command = 'ffmpeg -y -r 25 -i {} '.format(pattern)
        command += '-c:v mpeg4 -vf fps=25 -pix_fmt yuv420p '
        command += name

        print(command)
        os.system(command)

    def render_vids_side_by_side(self, input1, input2):
        command = 'ffmpeg -y -i {} -i {} '.format(input1, input2)
        command += '-filter_complex '
        command += '\'[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]\' '
        command += '-map [vid] -c:v mpeg4 '
        command += os.path.join(self.repose_path, 'target_vs_gen.mp4')

        print(command)
        os.system(command)


def rgb2openpose(image):
    '''Prepares an image to be interpreted by openpose:
    RGB->BGR, supp.: 0->1
    '''
    image = np.stack([image[..., 2], image[..., 1], image[..., 0]], axis=-1)
    image = adjust_support(image, '0->255')

    return image


def make_pck_table(name, rows, pck_values, thresholds):
    content = {}
    row_names = ['Model'] + rows
    for pck_at_t, thresh in zip(pck_values, thresholds):
        col_name = LatexString('PCK@{} [%]'.format(thresh))
        content[col_name] = [SIstring(pck_at_t)]

    tab_str = make_latex_table(content, rows=row_names)

    with open(name, 'w+') as lf:
        lf.write(tab_str)


OPENPOSE_KPS = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",

    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",

    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",

    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",

    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
]


def make_pck_per_kp_table(name, rows, pck_values, thresholds):
    content = {}

    col_names = ['PCK'] + OPENPOSE_KPS

    print(np.shape(pck_values))
    print(np.shape(rows))

    for i, pck_at_t in enumerate(pck_values):
        t = thresholds[i]
        row_name = LatexString('PCK@{} [%]'.format(t))
        content[row_name] = [SIstring(p) for p in pck_at_t]

    print(np.shape(content.keys()))

    tab_str = make_latex_table(content, cols=col_names)

    with open(name, 'w+') as lf:
        lf.write(tab_str)

    # This will probably not work out of the box on most systems!
    compile_table(tab_str, name)


def make_l2_table(name, rows, l2_values):
    content = {}
    content['L2 kp distance [px]'] = [SIstring(m, s) for m, s in l2_values]
    row_names = ['Model'] + rows

    tab_str = make_latex_table(content, rows=row_names)

    with open(name, 'w+') as lf:
        lf.write(tab_str)
    # This will probably not work out of the box on most systems!
    compile_table(tab_str, name)


def mean_pose_precision_plot(root,
                             name,
                             pcks_per_kp,
                             thresholds,
                             ending='.png'):
    '''Shows the precision of detected keypoints on a map of a pretty standart
    pose as colored circles.

    Args:
        root (str) Where to save the figures.
        name (str): basename used for the image. Don't add a png at the end
            use the :attr:`ending` argument if you must.
        pcks_per_kp (np.ndarray): Keypoint pck values. Shape: ``[T, K]``, where
            ``T`` is the number of applied thresholds and ``K`` the number of
            keypoints. Should be same as ``len(STANDART_KPS)``.
        thresholds (np.ndarray): The ``T`` corresponding thresholds applied
            when calculating the pck metric.
        ending (str): File ending to append to name string before saving the
            figures. Don't forget the full stop!
    '''

    import matplotlib
    import matplotlib.pyplot as plt

    cmap = matplotlib.cm.get_cmap('RdYlGn')
    default_size = 5

    f_all, AX_all = plt.subplots(1, len(thresholds))
    for pcks, thresh, ax_all in zip(pcks_per_kp, thresholds, AX_all):
        f_single, ax_single = plt.subplots(1, 1)

        colors = [cmap(p) for p in pcks]
        sizes = [default_size ** (1 + p) for p in pcks]

        for ax in [ax_single, ax_all]:
            ax.axis('off')
            ax.scatter(STANDART_KPS[..., 0],
                       STANDART_KPS[..., 1],
                       c=colors,
                       s=sizes,
                       cmap=cmap)

            ax.set_title('PCK@{}'.format(thresh))
            ax.set_aspect(1)
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[1], ylims[0])

        cax, _ = matplotlib.colorbar.make_axes(ax_single)
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap)
        cbar.set_label('PCK@{} [%]'.format(thresh))

        f_single.savefig(os.path.join(root, name + '_' + str(thresh) + ending))

    cax, _ = matplotlib.colorbar.make_axes(AX_all[-1])
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap)
    cbar.set_label('PCK@{} [%]'.format(thresh))
    f_all.savefig(os.path.join(root, name + '_all' + ending))


def default_repose_eval(root, data_in, data_out, config):

    # Set data_out to be data_in
    debug_mode = os.environ.get('DEBUG_MODE', 'False') == 'True'
    print('DEBUG', debug_mode)

    LOGGER.info("Setting up repose eval...")

    repose_config = config.get('repose_kwargs', {})
    if debug_mode:
        data_out = data_in
        koim = set_default(repose_config, 'data_out_im_key', 'target')
    else:
        koim = set_default(repose_config, 'data_out_im_key', 'frame_gen')
    set_value(repose_config, 'data_in_kp_key', 'target_keypoints_rel')

    # Only use pck for now
    set_value(repose_config, 'metrics', ['pck'])
    threshs = set_default(repose_config,
                          'metrics_kwargs/pck/thresholds',
                          PCK_THRESH)

    # For scaling the keypoints from relative to absolute
    gen_size = data_out[0][koim]
    if isinstance(gen_size, list):
        gen_size = gen_size[0]
    gen_size = np.array(gen_size.shape[:2])

    rp_eval = RePoseEval(**repose_config)

    LOGGER.info("Running repose eval...")

    print(len(data_in))
    print(len(data_out))
    print(repose_config)

    rp_eval(root, data_in, data_out, config)

    LOGGER.info("repose eval finished!")
