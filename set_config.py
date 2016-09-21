# USAGE
# python set_config.py

import argparse
import ConfigParser as cp

config = cp.RawConfigParser()

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path_axonseg", required=True, help="absolute path of the AxonSeg toolbox (matlab) - Used for myelin detection")
args = vars(ap.parse_args())
path_axonseg = args["path_axonseg"]

config.add_section('paths')
config.set('paths', 'path_axonseg',path_axonseg )

config.write(open('./data/config.cfg','w'))



