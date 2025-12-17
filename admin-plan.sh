set -euo pipefail

export AWS_REGION=us-east-1
export STG_FN=intellpulse-api-staging
export PROD_FN=intellpulse-api
export STG_URL='https://hrimhqter7wxdiwzcwiwmo5ubu0xplqb.lambda-url.us-east-1.on.aws'
export API_KEY='REPLACE_ME_API_KEY'
export KEY_HASH='7154cbb60b70'
export ADMIN_KEY='REPLACE_ME_ADMIN_KEY'

aws lambda get-function-configuration --function-name "$STG_FN" --region "$AWS_REGION" --query 'Environment.Variables' --output json > stg-vars.json
python - <<'PY'
import json, os
d=json.load(open("stg-vars.json"))
d["ADMIN_KEY"]=os.environ["ADMIN_KEY"]
open("stg-env.json","w").write(json.dumps({"Variables": d}))
print("wrote stg-env.json")
PY
aws lambda update-function-configuration --function-name "$STG_FN" --region "$AWS_REGION" --environment file://stg-env.json
aws lambda wait function-updated --function-name "$STG_FN" --region "$AWS_REGION"

aws lambda get-function-configuration --function-name "$PROD_FN" --region "$AWS_REGION" --query 'Environment.Variables' --output json > prod-vars.json
python - <<'PY'
import json, os
d=json.load(open("prod-vars.json"))
d["ADMIN_KEY"]=os.environ["ADMIN_KEY"]
open("prod-env.json","w").write(json.dumps({"Variables": d}))
print("wrote prod-env.json")
PY
aws lambda update-function-configuration --function-name "$PROD_FN" --region "$AWS_REGION" --environment file://prod-env.json
aws lambda wait function-updated --function-name "$PROD_FN" --region "$AWS_REGION"

curl -s -i -H "x-admin-key: $ADMIN_KEY" "$STG_URL/admin/plan?key_hash=$KEY_HASH"

python - <<'PY'
import json, os
body={"key_hash":os.environ["KEY_HASH"],"plan":"free","daily_limit":2000,"note":"staging set"}
open("plan.json","w").write(json.dumps(body))
print("wrote plan.json")
PY

curl -s -i -X POST -H "Content-Type: application/json" -H "x-admin-key: $ADMIN_KEY" --data-binary @plan.json "$STG_URL/admin/plan"

curl -s -i -H "x-api-key: $API_KEY" "$STG_URL/usage"
curl -s -i -H "x-api-key: $API_KEY" "$STG_URL/signal?asset=BTC-USD&mode=combined"
