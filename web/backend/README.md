cd backend

# START UDOP:
source /opt/conda/bin/activate pytorch
python udop.py

# START SBERT:
docker run -p 5000:5000 naye971012/my-image-name
