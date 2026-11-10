"""Use full-window offline time resampling without changing original inputs."""
import sys
import data_resampled
sys.modules['data']=data_resampled
from train import main

if __name__ == '__main__':
    main()
