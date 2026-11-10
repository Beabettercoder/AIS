"""CARLA using the full-window offline interpolation protocol."""
import sys
import data_resampled
sys.modules['data'] = data_resampled
from train_carla import main

if __name__ == '__main__':
    main()
