# pythonfun

tensorflow installation so I never have to figure this out again. quite annoying but still less of a headache than rocm.
```
https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-723/install-guide/index.html#installing-pip
https://www.tensorflow.org/install/pip
https://rpmfusion.org/Howto/CUDA
https://rpmfusion.org/Configuration
```
tf only supports ubuntu officially but getting it on fedora isn't that much harder, on a fresh fedora install need to get the rpm free/nonfree packages to get CUDA:
https://rpmfusion.org/Configuration
```
sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
```
then to get cuda
https://rpmfusion.org/Howto/CUDA
```
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora35/x86_64/cuda-fedora35.repo
sudo dnf clean all
sudo dnf module disable nvidia-driver
sudo dnf -y install cuda
```

I think you need to restart terminal/machine for the drivers to go into effect, but hopefully nvidia-smi works if you were to run it after the fact. then follow gpu instructions on tf page:
https://www.tensorflow.org/install/pip
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```
this stuff is all on the tf install page, but there are things with nvidia-tensorrt and bashrc that need to be done

tensorrt
https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-723/install-guide/index.html#installing-pip

I feel like I installed some sort of nvidia machine learning repo as well from rpm
```
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/python3.9/site-packages/tensorrt/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
the link I just realized has a work around if your cuda version is later than the tensorflow that you have (downloads libnvinfer8 but expects 7).
having done this before on a cluster it isn't just libnvinfer that causes problems
you can downgrade (step 4 of 4.3.1. Using The NVIDIA Machine Learning Network Repo For RPM Installation) or create a symbolic link between 7 and 8 which will work just fine

```
ln -s /home/christian/miniconda3/envs/tf/lib/python3.9/site-packages/tensorrt/libnvinfer_plugin.so.8 /home/christian/miniconda3/envs/tf/lib/python3.9/site-packages/tensorrt/libnvinfer_plugin.so.7
```
I also had to add this to my .bashrc
https://stackoverflow.com/questions/74956134/could-not-load-dynamic-library-libnvinfer-so-7
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/christian/miniconda3/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/christian/miniconda3/lib/python3.9/site-packages/tensorrt/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/christian/miniconda3/envs/tf/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/christian/miniconda3/envs/tf/lib/python3.9/site-packages/tensorrt/
```
can check that it has worked with:
```
python3 -c "import tensorflow as tf; print(tf.test.is_gpu_available(cuda_only=True))"
```
ADDENDUM: **it's free if you don't value your time**


I ran into a somewhat famous problem with looping login on fedora that I'm not really what the cause is.
It could be gnome, it could be xorg, it could be nvidia, tbh not sure but switching desktop environments to kde plasma was what worked for me in the end.
KDE/Wayland got me past the login screen as a GUI. pressing shift after turning on allowed me to boot up in recovery mode and get onto terminal emulator with f3.

nvidia kmod removal, I did this but not sure it actually did anything.
```
https://ask.fedoraproject.org/t/trouble-installing-proprietary-nvidia-driver/24167/6
```

swapping gnome for kde safely:
```
https://unix.stackexchange.com/questions/483692/safely-uninstalling-gnome-desktop-environment-on-fedora-29
```

gnome shell stuff, I don't think .ICEauthority file was the culprit but it's nice to know.
```
https://unix.stackexchange.com/questions/549466/cant-login-through-gdm-loop-in-login-on-particular-user
```

when attempting the following:
```
https://www.reddit.com/r/LocalLLaMA/comments/11o6o3f/how_to_install_llama_8bit_and_4bit/
```

get homebrew to be recognized 
```
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
```

nice little way of getting homebrew gcc to work with setup_cuda.py
```
https://www.reddit.com/r/Fedora/comments/usnu0x/comment/ig0t9y7/?utm_source=share&utm_medium=web2x&context=3
```


