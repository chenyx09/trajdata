rsync -ravh --progress --exclude ".gitlab*" --exclude "public/" --exclude "opendrive" --exclude "copy_to_public.sh" --exclude "*.pyc" --exclude "ngc/" --exclude "*.egg-info/" --exclude "__pycache__/" ./* public/