'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:00:34
Description: 
'''
import ray 
from pathlib import Path
from util.videoswap import *
from test_video_swapmulti import *

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def get_app(opt):
    if opt.crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'    
    app = Face_detect_crop(name='antelope', root='checkpoints/insightface_models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode = mode)
    return app

def get_swap_model(opt):
    model = create_model(opt)
    model.eval()
    return model
    
@ray.remote
class Processor:
    def __init__(self, opt, id_vetor) -> None:

        self.temp_results_dir = opt.temp_path
        self.crop_size = opt.crop_size
        self.no_simswaplogo = opt.no_simswaplogo
        self.use_mask = opt.use_mask
        

        app = get_app(opt)

        spNorm =SpecificNorm()
        if self.use_mask:
            n_classes = 19
            net = BiSeNet(n_classes=n_classes)
            net.cuda()
            save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
            net.load_state_dict(torch.load(save_pth))
            net.eval()
        else:
            net =None
            
        swap_model = get_swap_model(opt)
            
        self.id_vetor = id_vetor
        self.swap_model = swap_model
        self.detect_model = app
        self.net = net
        self.spNorm = spNorm
        
        
    # @torch.no_grad
    async def process_frame(self, frame_index, frame):
        id_vetor = self.id_vetor
        swap_model = self.swap_model
        detect_model = self.detect_model
        temp_results_dir = self.temp_results_dir
        crop_size = self.crop_size
        no_simswaplogo = self.no_simswaplogo
        use_mask = self.use_mask
        net = self.net
        spNorm = self.spNorm
        
        detect_results = detect_model.get(frame, crop_size)
        save_jpg_path = os.path.join(temp_results_dir, f'frame_{frame_index:0>7d}.jpg')
        
        if detect_results is not None:
            frame_align_crop_list = detect_results[0]
            frame_mat_list = detect_results[1]
            swap_result_list = []
            frame_align_crop_tenor_list = []
            
            for frame_align_crop in frame_align_crop_list:
                frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop, cv2.COLOR_BGR2RGB))[None,...].cuda()
                swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
                swap_result_list.append(swap_result)
                frame_align_crop_tenor_list.append(frame_align_crop_tenor)
            
            reverse2wholeimage(frame_align_crop_tenor_list, swap_result_list, frame_mat_list, crop_size, frame, logoclass,
                            save_jpg_path, no_simswaplogo,
                            pasring_model=net, use_mask=use_mask, norm=spNorm)
        else:
            frame = frame.astype(np.uint8)
            if not no_simswaplogo:
                frame = logoclass.apply_frames(frame)
            cv2.imwrite(save_jpg_path, frame)
        
        return frame_index


    
if __name__ == '__main__':

    visible_gpus = "1,2,3,4"

    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus

    # used for ray only
    gpus = list(map(int, visible_gpus.split(",")))
    num_gpus = len(gpus)
    print(f"[INFO] num_gpus: {num_gpus}, gpus: {gpus}")


    # Initialize Ray
    ray.init()

    logoclass = watermark_image('./simswaplogo/simswaplogo.png')

    
    # ... (keep the existing setup code)
    opt = TestOptions().parse()

    start_epoch, epoch_iter = 1, 0
    crop_size = opt.crop_size

    torch.nn.Module.dump_patches = True

    temp_results_dir=opt.temp_path
    if  os.path.exists(temp_results_dir):
            shutil.rmtree(temp_results_dir)
    os.makedirs(opt.temp_path, exist_ok=True)
        
        
    video_path = opt.video_path
    video_forcheck = VideoFileClip(video_path)
    if video_forcheck.audio is None:
        no_audio = True
    else:
        no_audio = False

    del video_forcheck

    if not no_audio:
        video_audio_clip = AudioFileClip(video_path)
        
        
    app = get_app(opt)
    model = get_swap_model(opt)

    video = cv2.VideoCapture(video_path)
    ret = True
    frame_index = 0
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
            
    # get source image id
    with torch.no_grad():
        pic_a = opt.pic_a_path
        # img_a = Image.open(pic_a).convert('RGB')
        img_a_whole = cv2.imread(pic_a)
        img_a_align_crop, _ = app.get(img_a_whole,crop_size)
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # convert numpy to tensor
        img_id = img_id.cuda()
        # img_att = img_att.cuda()

        #create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)



    num_gpu_per_processor = 0.2
    num_actors = (int(1.0 / num_gpu_per_processor)* num_gpus)

    print(f"Number of actors doing processing: {num_actors}")
    print(f"Number of files to process: {frame_count}")
        
    # # Create the Processor actors
    # actors = [Processor.options(num_gpus=num_gpu_per_processor).remote(opt, id_vetor=latend_id) for _ in range(num_actors)]
    # files_submitted = 0
    # futures = []
    # # for video_path in mp4_files: 
    # for frame_index in tqdm(range(frame_count)):
    #     ret, frame = video.read()
    #     if ret:
    #         actor = actors[files_submitted % len(actors)]
    #         future = actor.process_frame.remote(frame_index, frame)
    #         futures.append(future)
    #     else:
    #         break            

    # # Wait for all tasks to complete
    # results = ray.get(futures)
    # video.release()

    # Create the Processor actors
    actors = [Processor.options(num_gpus=num_gpu_per_processor).remote(opt, id_vetor=latend_id) for _ in range(num_actors)]

    all_results = []
    chunk_size = 300
    for chunk_start in range(0, frame_count, chunk_size):
        chunk_end = min(chunk_start + chunk_size, frame_count)
        print(f"Processing chunk {chunk_start} to {chunk_end}")

        futures = []
        for frame_index in tqdm(range(chunk_start, chunk_end)):
            save_jpg_path = os.path.join(temp_results_dir, f'frame_{frame_index:0>7d}.jpg')
            if Path(save_jpg_path).exists():
                continue
            
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = video.read()
            if ret:
                actor = actors[frame_index % len(actors)]
                future = actor.process_frame.remote(frame_index=frame_index, frame=frame)
                futures.append(future)
                # break
            else:
                break

        # Wait for all tasks in this chunk to complete
        chunk_results = ray.get(futures)
        all_results.extend(chunk_results)

        # # Clear GPU memory
        # for actor in actors:
        #     ray.get(actor.clear_gpu_memory.remote())

    video.release()



    # merge video
    path = os.path.join(temp_results_dir,'*.jpg')
    image_filenames = sorted(glob.glob(path))

    clips = ImageSequenceClip(image_filenames,fps = fps)

    if not no_audio:
        clips = clips.set_audio(video_audio_clip)

    save_path = opt.output_path
    clips.write_videofile(save_path,audio_codec='aac')
        

    # Shutdown Ray
    ray.shutdown()