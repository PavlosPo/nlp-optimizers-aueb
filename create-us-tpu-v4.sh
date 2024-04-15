export ZONE="us-central2-b" # Choose this according to your quota
export TPU_TYPE="v4-32" # Choose this according to your quota
export PROJECT_ID="lmtc-zero" # Your project ID

# I recommend to use this alias and put it in your .zshrc/.bashrc
# alias gtpu="gcloud alpha compute tpus tpu-vm" 


while true
do
  export TPU_NAME="aueb-nlp-optimisers"
  gcloud alpha compute tpus tpu-vm create ${TPU_NAME} \
    --zone=${ZONE} \
    --accelerator-type=${TPU_TYPE} \
    --version=tpu-vm-pt-2.0 \
    --project=${PROJECT_ID}
  
  # Check the exit code of the gcloud command
  if [ $? -eq 0 ]; then
    echo "TPU creation successful. Exiting loop."
    break
  else
    echo "TPU creation failed. Retrying..."
    sleep 60 # Add a delay before retrying
  fi
done
