FROM public.ecr.aws/lambda/python:3.11

COPY requirements.lambda.txt .
RUN pip install -r requirements.lambda.txt --no-cache-dir -t ${LAMBDA_TASK_ROOT}

COPY src ${LAMBDA_TASK_ROOT}/src
COPY data ${LAMBDA_TASK_ROOT}/data

CMD ["src.api.app.handler"]
