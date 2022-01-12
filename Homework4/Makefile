.PHONY: install download_data make_trianing_lr

install: 
	@echo "Installing Dependencies..."
	pip install -r requirements.txt
	cd KAIR_Repo ; pip install -r requiremet.txt

download_data:
	@echo "Downloading Data..."
	rm -rf data
	mkdir -p data
	gdown https://drive.google.com/uc?id=1GL_Rh1N-WjrvF_-YOKOyvq0zrV6TF4hb -O "data/dataset.zip"

make_trianing_lr: 
	@echo "Making Training LR Images..."
	rm -rf data/training_lr
	mkdir -p data/training_lr
	python make_training_lr.py

train: 
	cd KAIR; python main_train_psnr.py --opt options/train_msrresnet_psnr.json

create_result_zip:
	cd KAIR/results/swinir_classical_sr_x3; zip swinir_sr_20000.zip ./*.png 
	mv KAIR/results/swinir_classical_sr_x3/swinir_sr_20000.zip .

reproduce: 
	rm -rf KAIR_Repo/superresolution/models
	mkdir -p KAIR_Repo/superresolution/models
	@gdown https://drive.google.com/uc?id=1sVyh0qiXb16j5Ave_8r2_pbApunes09q/ -O "KAIR_Repo/superresolution/models/95000_G.pth"
	@gdown https://drive.google.com/uc?id=1PwKxXt9sSAHRJN6A3NbewtbcOYbye8Qd/ -O "KAIR_Repo/superresolution/models/95000_E.pth"
	cd KARI_Repo; python main_test_psnr.py --opt options/test_msrresnet_psnr.json 
	@echo "The resulting 3x HR images are stored in KAIR_Repo/superresolution/images/evaluation{checkpoints iterations}/"
