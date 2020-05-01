

#pip install numpy==1.18.2  pillow==6.2.1  && pip install  https://github.com/arita37/mlmodels/archive/dev.zip  && wget https://raw.githubusercontent.com/arita37/mlmodels/dev/requirements_fake.txt  && pip install -r requirements_fake.txt  



pip install numpy==1.18.2  pillow==6.2.1  
mkdir z   
cd z && git clone https://github.com/arita37/mlmodels.git  && cd mlmodels && pip install -e . -r requirements.txt    &&  pip install -r requirements_fake.txt  



