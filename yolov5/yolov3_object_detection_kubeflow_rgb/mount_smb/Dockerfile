FROM python:3.7-slim

WORKDIR /app

COPY mount_smb.py ./mount_smb.py

RUN pip install -U smbprotocol

ENTRYPOINT ["python", "mount_smb.py"]