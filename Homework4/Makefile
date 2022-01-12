.PHONY: install data_preparation trian inference

install: 
	@echo "Installing Dependencies..."
	pip install -r requirements.txt
	cd KAIR && pip install -r requirement.txt

data_preparation:
	@echo "Downloading Data..."
	rm -rf data
	mkdir -p data
	gdown https://drive.google.com/uc?id=1GL_Rh1N-WjrvF_-YOKOyvq0zrV6TF4hb -O "data/datasets.zip"
	unzip -q data/datasets.zip -d data/ && rm data/datasets.zip
	cp data/testing_lr_images/testing_lr_images/* data/testing_lr_images/
	cp data/training_hr_images/training_hr_images/* data/training_hr_images/
	rm -rf data/training_hr_images/training_hr_images/ data/testing_lr_images/testing_lr_images/
	python image_preprocess.py

train:
	cp train.py KAIR/
	@echo "Training SwinIR model..."
	cd KAIR; python train.py --opt ../options/train_swinir_sr_customized.json

inference:
	cp test.py KAIR/
	@echo "Start inference..."
	cd KAIR && python test.py \
							--task classical_sr \
							--scale 3 \
							--training_patch_size 48 \
							--model_path ../models/best.pth \
							--folder_lq ../data/testing_lr_images/

result:
	@echo "Generate result.zip"
	cd KAIR/results/swinir_classical_sr_x3; zip result.zip ./*.png
	mv KAIR/results/swinir_classical_sr_x3/result.zip .

reproduce: 
	rm -rf KAIR_Repo/superresolution/models
	mkdir -p KAIR_Repo/superresolution/models
	@gdown https://drive.google.com/uc?id=1sVyh0qiXb16j5Ave_8r2_pbApunes09q/ -O "KAIR_Repo/superresolution/models/95000_G.pth"
	@gdown https://drive.google.com/uc?id=1PwKxXt9sSAHRJN6A3NbewtbcOYbye8Qd/ -O "KAIR_Repo/superresolution/models/95000_E.pth"
	cd KARI_Repo; python main_test_psnr.py --opt options/test_msrresnet_psnr.json 
	@echo "The resulting 3x HR images are stored in KAIR_Repo/superresolution/images/evaluation{checkpoints iterations}/"
