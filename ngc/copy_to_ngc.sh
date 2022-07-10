#!/bin/bash

cd ../..
rsync -ravh --progress --exclude="*.pyc" --exclude=".git" --exclude="__pycache__" --exclude="*.egg-info" --exclude=".pytest_cache" trajdata/ ngc-trajdata/
