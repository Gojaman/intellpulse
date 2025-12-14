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
  description = "ECR image URI including tag"
  type        = string
}

# Optional safety: force you to pass TF_VAR_image_uri or -var
# (no default on purpose)

