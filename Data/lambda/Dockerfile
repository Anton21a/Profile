FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pipeline.py lambda_function.py ./

CMD ["lambda_function.lambda_handler"]