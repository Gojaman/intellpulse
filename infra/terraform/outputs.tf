output "api_base_url" {
  value = aws_apigatewayv2_api.http_api.api_endpoint
}

output "signals_bucket" {
  value = aws_s3_bucket.signals.bucket
}
