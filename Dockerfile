FROM python:3.10.8-bullseye

RUN pip install numpy==1.23.4
RUN pip install cvxopt==1.2.7
RUN pip install matplotlib==3.6.2