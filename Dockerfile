# AWS Lambda container: the vision pipeline, the agent and the site builder.
# The build context is a standalone checkout (see export.sh), which has sitebuilder/.
FROM public.ecr.aws/lambda/python:3.13
COPY requirements.txt ./
RUN grep -v '^pytest' requirements.txt > runtime.txt && pip install --no-cache-dir -r runtime.txt
COPY models ./models
COPY menuvision ./menuvision
COPY demo/menu.jpg demo/site.yaml ./demo/
COPY web ./web
COPY webapp.py ./
COPY sitebuilder ./sitebuilder
# Checkout files can be owner-only (umask 077); Lambda runs the code as a non-root user.
RUN chmod -R a+rX /var/task
CMD ["webapp.handler"]
