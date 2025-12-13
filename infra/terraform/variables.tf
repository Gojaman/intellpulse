variable "aws_region" {
  description = "AWS region to deploy to"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Prefix for naming AWS resources"
  type        = string
  default     = "intellpulse"
}

variable "image_uri" {
  description = "ECR image URI including tag (e.g. 123.dkr.ecr.us-east-1.amazonaws.com/intellpulse-api:latest)"
  type        = string
}
