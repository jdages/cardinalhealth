 FROM tensorflow/tensorflow
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 80
CMD python ./App.py