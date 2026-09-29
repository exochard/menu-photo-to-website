#!/usr/bin/env bash
# Build the Lambda image, push it to ECR and serve it from a public Function URL.
#   ./deploy.sh          create or update everything in eu-central-1
#   ./deploy.sh --down   delete the function, its URL, the role and the image repository
# Needs AWS CLI v2 signed in (`aws login`) and Docker. Safe to rerun.
set -euo pipefail
region=${AWS_REGION:-eu-central-1}
name=menu-agent
role=menu-agent-lambda
here=$(cd "$(dirname "$0")" && pwd)
account=$(aws sts get-caller-identity --query Account --output text)
registry=$account.dkr.ecr.$region.amazonaws.com
image=$registry/$name:latest
export AWS_REGION=$region AWS_PAGER=""

if [ "${1:-}" = "--down" ]; then
  aws lambda delete-function-url-config --function-name $name 2>/dev/null || true
  aws lambda delete-function --function-name $name 2>/dev/null || true
  aws iam detach-role-policy --role-name $role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true
  aws iam delete-role --role-name $role 2>/dev/null || true
  aws ecr delete-repository --repository-name $name --force >/dev/null 2>&1 || true
  echo "removed $name from $region"
  exit 0
fi

aws ecr describe-repositories --repository-names $name >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name $name >/dev/null
aws ecr get-login-password | docker login --username AWS --password-stdin "$registry" >/dev/null
# Build from a standalone layout: this folder if it has sitebuilder/, else a fresh export.
context=$here
if [ ! -d "$here/sitebuilder" ]; then
  context=$(mktemp -d)/menu-agent
  "$here/export.sh" "$context" >/dev/null
  cp -r "$here/models" "$context/models"
fi
docker build --platform linux/amd64 --provenance=false -t "$image" "$context"
docker push "$image" >/dev/null

if ! aws iam get-role --role-name $role >/dev/null 2>&1; then
  aws iam create-role --role-name $role --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  # Logs only: the function reads nothing else in the account.
  aws iam attach-role-policy --role-name $role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  sleep 10  # a new role takes a few seconds to become assumable
fi

if aws lambda get-function --function-name $name >/dev/null 2>&1; then
  aws lambda update-function-code --function-name $name --image-uri "$image" >/dev/null
else
  aws lambda create-function --function-name $name --package-type Image --code ImageUri="$image" \
    --role "arn:aws:iam::$account:role/$role" --memory-size 2048 --timeout 60 --architectures x86_64 >/dev/null
fi
aws lambda wait function-updated-v2 --function-name $name
# A public demo: at most two runs at once, so a burst of traffic cannot run up usage.
aws lambda put-function-concurrency --function-name $name --reserved-concurrent-executions 2 >/dev/null \
  || echo "note: reserved concurrency not set (new accounts have a low concurrency quota)"

if ! aws lambda get-function-url-config --function-name $name >/dev/null 2>&1; then
  aws lambda create-function-url-config --function-name $name --auth-type NONE >/dev/null
  aws lambda add-permission --function-name $name --statement-id public-url \
    --action lambda:InvokeFunctionUrl --principal '*' --function-url-auth-type NONE >/dev/null
  aws lambda add-permission --function-name $name --statement-id public-invoke \
    --action lambda:InvokeFunction --principal '*' --invoked-via-function-url >/dev/null 2>&1 || true
fi
url=$(aws lambda get-function-url-config --function-name $name --query FunctionUrl --output text)
echo "live: $url"
echo "logs: aws logs tail /aws/lambda/$name --follow"
