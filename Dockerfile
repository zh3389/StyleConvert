FROM hdgigante/python-opencv:4.7.0-debian
WORKDIR /app
COPY . /app

RUN pip3 install -r requirements.txt --break-system-packages && pip3 cache purge

EXPOSE 9002
CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9002", "--reload" ]