# Particle Filter SLAM

### Directory Structure

```
data
|______dataRGBD
        |___Disparity*
        |___RGB*
|______Encoders*.npz
|______Hokuyo*.npz
|______Imu*.npz
_______Kinect*.npz
src
|______lidar.py
|______load_data.py
|______map.py
|______plots.py
|______pr2_utils.py.py
|______texture_map.py
|______trajectory.py

results (generated)
|______*dataset_number
        |____ *plots and images

environment.yml
requirements.txt
run.py
README.md
```

### Usage

The code is arranged in the above directory structure. All the image RGB, depth and sensor data should reside in `data` folder under the respective subfolder.


To run the code, a conda virtual environment needs to be created as shown
```
conda env create -f environment.yml
conda activate proj2
```

Now, once the environment is ready, the code can be run using the following command
```
python run.py --dataset DATASET
```
`DATASET` is any integer for which we have data i.e. from `20` to `21`

eg.
```bash
python run.py --dataset 20
```
This will run the code on dataset `20`.

If you want to run the code for all the datasets, it can be done by typing the following command on a unix system. This takes about 50 minutes to run on my system.
```bash
bash run.sh
```

### Results
Once the code run is complete for a dataset, a folder is generated as `results/{dataset_num}/` which contains all the plots.
