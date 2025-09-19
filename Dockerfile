FROM public.ecr.aws/lambda/python:3.11

# Réduire le bruit TF et stabiliser les résultats numériques
ENV TF_ENABLE_ONEDNN_OPTS=0 \
    KERAS_BACKEND=tensorflow

# Dépendances Python backend
COPY requirements-backend.txt ./
RUN pip install --no-cache-dir -r requirements-backend.txt

# Code applicatif (tout le projet)
COPY . ./

# Handler Lambda (module:function)
CMD ["backend_lambda.handler"]
