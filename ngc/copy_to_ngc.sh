#!/bin/bash

cd ../..
rsync -ravh --progress --exclude="*.pyc" --exclude=".git" --exclude="__pycache__" --exclude="*.egg-info" avdata/ ngc-avdata/
