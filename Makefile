
run_api:
	-@uvicorn api.fast:app --reload  # load web server with code autoreload

redeploy_cloudrun:
	-@gcloud run deploy --image eu.gcr.io/optimal-jigsaw-296709/project_spotify --platform managed --region europe-west1



# ----------------------------------
#      GOOGLE CLOUD PLATFORM STUFF
# ----------------------------------
# project id
PROJECT_ID=optimal-jigsaw-296709# Replace with your Project's ID

# bucket name
BUCKET_NAME=project_spotify_pellissier# Use your Project's name as it should be unique

REGION=europe-west1 # Choose your region https://cloud.google.com/storage/docs/locations#available_locations

PYTHON_VERSION=3.7
FRAMEWORK=scikit-learn
RUNTIME_VERSION=1.15

# path of the file to upload to gcp (the path of the file should be absolute or should match the directory where the make command is run)
LOCAL_PATH=/home/achille/code/a-pellissier/Project_Spotify_502/raw_data/generated_spectrograms_small # Replace with your local path to the `train_1k.csv` and make sure to put it between quotes

# bucket directory in which to store the uploaded file (we choose to name this data as a convention)
BUCKET_FOLDER=data
BUCKET_TRAINING_FOLDER=generated_spectrograms_small

# name for the uploaded file inside the bucket folder (here we choose to keep the name of the uploaded file)
# BUCKET_FILE_NAME=another_file_name_if_I_so_desire.csv
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

##### Package params  - - - - - - - - - - - - - - - - - - -

PACKAGE_NAME=Project_Spotify_502
FILENAME=model_dl

##### Job - - - - - - - - - - - - - - - - - - - - - - - - -

JOB_NAME=project_spotify_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')

##### Machine Type - - - - - - - - - - - - - - - - - - - - - - - - -

MACHINE_TYPE=n1-standard-16


set_project:
	-@gcloud config set project ${PROJECT_ID}

create_bucket:
	-@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}


upload_data:
  # -@gsutil cp train_1k.csv gs://wagon-ml-my-bucket-name/data/train_1k.csv
	-@gsutil -m cp -r ${LOCAL_PATH} gs://${BUCKET_NAME}




run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs \
		--scale-tier CUSTOM \
		--master-machine-type ${MACHINE_TYPE}

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ __pycache__
	@rm -fr build dist *.dist-info *.egg-info
	@rm -fr */*.pyc


# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* Project_Spotify_502/*.py

black:
	@black scripts/* Project_Spotify_502/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit=$(VIRTUAL_ENV)/lib/python*

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr Project_Spotify_502-*.dist-info
	@rm -fr Project_Spotify_502.egg-info

install:
	@pip install . -U

all: clean install test black check_code


uninstal:
	@python setup.py install --record files.txt
	@cat files.txt | xargs rm -rf
	@rm -f files.txt

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u lologibus2

pypi:
	@twine upload dist/* -u lologibus2

