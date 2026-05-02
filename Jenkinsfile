pipeline {
    agent {
        kubernetes {
            label 'jenkins-agent'
            defaultContainer 'gcloud'
            yaml """
apiVersion: v1
kind: Pod
metadata:
  labels:
    role: jenkins
spec:
  serviceAccountName: jenkins-ksa
  nodeSelector:
    role: jenkins
  containers:
  - name: gcloud
    image: google/cloud-sdk:latest
    command: ['sleep', 'infinity']
    resources:
      requests:
        cpu: 200m
        memory: 256Mi
  - name: docker
    image: docker:24-dind
    securityContext:
      privileged: true
    env:
    - name: DOCKER_TLS_CERTDIR
      value: ""
    resources:
      requests:
        cpu: 200m
        memory: 256Mi
"""
        }
    }

    environment {
        PROJECT_ID   = 'ticket-support-01'
        REGION       = 'us-central1'
        REPO         = 'it-support-rag'
        SERVICE      = 'it-support-rag'
        IMAGE_BASE   = "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    sh 'git config --global --add safe.directory "*"'
                    env.SHORT_SHA = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
                    echo "Building commit: ${env.SHORT_SHA}"
                }
            }
        }

        stage('Build Image') {
            steps {
                container('docker') {
                    sh """
                        docker build \
                          -t ${IMAGE_BASE}:${SHORT_SHA} \
                          -t ${IMAGE_BASE}:latest \
                          -f issue_support/Dockerfile \
                          issue_support
                    """
                }
            }
        }

        stage('Push to Artifact Registry') {
            steps {
                container('gcloud') {
                    sh "gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet"
                }
                container('docker') {
                    sh "docker push --all-tags ${IMAGE_BASE}"
                }
            }
        }

        stage('Deploy to Cloud Run') {
            steps {
                container('gcloud') {
                    sh """
                        gcloud run deploy ${SERVICE} \
                          --image=${IMAGE_BASE}:${SHORT_SHA} \
                          --region=${REGION} \
                          --project=${PROJECT_ID} \
                          --platform=managed \
                          --allow-unauthenticated \
                          --port=8000 \
                          --memory=2Gi \
                          --cpu=2 \
                          --timeout=300 \
                          --cpu-boost \
                          --vpc-connector=milvus-connector \
                          --vpc-egress=private-ranges-only \
                          --set-env-vars=APP_MILVUS_URI=http://10.128.0.7:19530,MILVUS_DB_NAME=demo,GCS_BUCKET=ticket-support-01-dvc,UPLOAD_DIR=/app/data/uploads,LLM_PROMPT_DIR=/app/prompts/llm_prompt.txt,RAG_PROMPT_DIR=/app/prompts/rag_prompt.txt,ROUTER_PROMPT_DIR=/app/prompts/router_prompt.txt,EVALUATOR_LLM_PROMPT_DIR=/app/prompts/evaluator_llm_prompt.txt,GUARDRAILS_PROMPT_DIR=/app/prompts/guardrails_llm_prompt.txt,RAG_SCORE_THRESHOLD=0.35,RAG_DEBUG=false \
                          --set-secrets=GROQ_API_KEY=groq-api-key:latest,DEMO_API_KEY=demo-api-key:latest
                    """
                }
            }
        }
    }

    post {
        success {
            echo "Deployment successful — ${SERVICE}:${SHORT_SHA} is live on Cloud Run."
        }
        failure {
            echo "Pipeline failed at stage: ${currentBuild.currentResult}"
        }
    }
}
