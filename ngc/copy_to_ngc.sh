#!/bin/bash

cd ../..
rsync -ravh --progress --exclude="*.pyc" --exclude=".git" --exclude="__pycache__" avdata/ ngc-avdata/
