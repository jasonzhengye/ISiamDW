# ISiamDW
Enhanced Visual Tracking with Inverted Residual Mobile Block: A SiamRPN-based Approach


Download testing datasets
Download datasets and put them into testing_dataset directory. Jsons of commonly used datasets can be downloaded from [Google Drive or BaiduYun](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F). If you want to test tracker on new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to setting testing_dataset




Test tracker
cd experiments/siamrpn_r50_l234_dwxcorr
python -u ../../tools/test.py 	\
	--snapshot model.pth 	\ # model path
	--dataset VOT2018 	\ # dataset name
	--config config.yaml	  # config file


Eval tracker
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset VOT2018        \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix 'model'   # tracker_name




 Training
 Prepare training dataset, detailed preparations are listed in training_dataset directory.
 Download pretrained backbones from [Google Drive](https://drive.google.com/drive/folders/1DuXVWVYIeynAcvt9uxtkuleV6bs6e3T9) and put them in pretrained_models directory

 VID http://image-net.org/challenges/LSVRC/2017/
YOUTUBEBB https://research.google.com/youtube-bb/
DET http://image-net.org/challenges/LSVRC/2017/
COCO  http://cocodataset.org/

 After training, you can test snapshots on VOT dataset. For AlexNet,, you need to test snapshots from 35 to 50 epoch.
