for dir in */pileup; do
    # Extract the parent directory name to use in the S3 path
    parent_dir=$(basename $(dirname "$dir"))

    # Construct the source path for the local files you want to sync
    src_path="${dir}/pileup_mod/"

    # Construct the target path on S3
    dest_path="s3://tf-binding-sites/pretraining/data/cell_lines/${parent_dir}/pileup"

    # Sync .gz and .tbi files, delete others from the destination (S3) that do not exist in the source or don't match the criteria
    aws s3 sync "$src_path" "$dest_path" --exclude "*" --include "*.gz" --include "*.tbi" --delete
done
